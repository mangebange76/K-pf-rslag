# stockapp/manual_collect.py
from __future__ import annotations

import math
import typing as t

import pandas as pd
import streamlit as st

# ── Fetchers (tålbar import) ───────────────────────────────────────────────
try:
    from .fetchers.yahoo import get_all as yahoo_get_all
except Exception:
    yahoo_get_all = None  # type: ignore

try:
    # Vi använder både kompakt och verbose för debug
    from .fetchers.fmp import get_all as fmp_get_all, get_all_verbose as fmp_get_all_verbose
except Exception:
    fmp_get_all = None  # type: ignore
    fmp_get_all_verbose = None  # type: ignore

try:
    from .fetchers.sec import get_all as sec_get_all  # returns dict
except Exception:
    sec_get_all = None  # type: ignore


# ── (Valfri) sheets-integration ────────────────────────────────────────────
# Appen brukar redan ha ett flöde som sparar hela df:et.
# Men om stockapp.sheets finns, försöker vi spara direkt på knapptryck.
_sheets_ok = False
_sheets_save_df = None  # type: ignore
try:
    from .sheets import save_dataframe  # föredragen
    _sheets_ok = True
    _sheets_save_df = save_dataframe
except Exception:
    # fallback: ibland heter funktionen annorlunda
    try:
        from .sheets import write_dataframe as save_dataframe  # type: ignore
        _sheets_ok = True
        _sheets_save_df = save_dataframe
    except Exception:
        _sheets_ok = False
        _sheets_save_df = None  # type: ignore


# ── Konfiguration: fält-prioritet per källa ────────────────────────────────
# OBS: Namnen här ska matcha dina kolumnrubriker i Google Sheet.
FIELD_PRIORITY: dict[str, list[str]] = {
    "Kurs": ["yahoo", "fmp", "sec"],
    "P/S TTM": ["fmp", "yahoo", "sec"],
    "Market Cap (M)": ["fmp", "yahoo", "sec"],
    "Utestående aktier (milj.)": ["sec", "fmp", "yahoo"],
    "Omsättning (M)": ["fmp", "sec", "yahoo"],
    "Kassa (M)": ["sec", "fmp", "yahoo"],
    "Valuta": ["yahoo", "fmp", "sec"],
    "Bolagsnamn": ["yahoo", "fmp", "sec"],
    "Börs": ["fmp", "yahoo", "sec"],
    "Sektor": ["fmp", "yahoo", "sec"],
    "Bransch": ["fmp", "yahoo", "sec"],
    "Industri": ["fmp", "yahoo", "sec"],
    # Lägg fler vid behov…
}

# Om dina rubriker har synonymer — mappa dem här till en kanonisk nyckel.
ALIASES: dict[str, str] = {
    # exempel: "P/S-TTM": "P/S TTM",
}


# ── Hjälpfunktioner ────────────────────────────────────────────────────────
def _canon(field: str) -> str:
    """Normalisera fältnamn utifrån ALIASES."""
    return ALIASES.get(field, field)

def _is_nan(x: t.Any) -> bool:
    return isinstance(x, float) and math.isnan(x)

def _safe(v: t.Any) -> bool:
    return v is not None and v != "" and not _is_nan(v)

def _count_nonempty(d: dict | None) -> int:
    if not isinstance(d, dict):
        return 0
    return sum(1 for _, v in d.items() if _safe(v))

def _pick_value(field: str, yv: dict, fv: dict, sv: dict) -> tuple[t.Any, str | None]:
    """
    Returnerar (värde, källa) enligt prioritet per fält.
    """
    f = _canon(field)
    order = FIELD_PRIORITY.get(f, ["yahoo", "fmp", "sec"])
    for src in order:
        if src == "yahoo" and _safe(yv.get(f)):
            return yv[f], "Yahoo"
        if src == "fmp" and _safe(fv.get(f)):
            return fv[f], "FMP"
        if src == "sec" and _safe(sv.get(f)):
            return sv[f], "SEC"
    return None, None

def _merge_preview(cur_row: dict, yv: dict, fv: dict, sv: dict) -> pd.DataFrame:
    """
    Skapar en tabell med 'Fält', 'Före', 'Efter', 'Källa' för de fält vi kan uppdatera.
    Visar endast rader där värdet skulle ändras eller där före-värdet är tomt och efter ej tomt.
    """
    # Kandidatfält = union av (prioritetstabell + keys som faktiskt kommer från källorna)
    fields: set[str] = set(FIELD_PRIORITY.keys()) | set(yv.keys()) | set(fv.keys()) | set(sv.keys())
    rows: list[dict[str, t.Any]] = []

    for field in sorted(fields):
        field_c = _canon(field)
        before = cur_row.get(field_c)
        after, src = _pick_value(field_c, yv, fv, sv)

        # visa bara meningsfulla diffar
        if _safe(after):
            if not _safe(before) or before != after:
                rows.append({
                    "Fält": field_c,
                    "Före": before,
                    "Efter": after,
                    "Källa": src,
                })

    if not rows:
        return pd.DataFrame(columns=["Fält", "Före", "Efter", "Källa"])
    dfp = pd.DataFrame(rows)
    return dfp[["Fält", "Före", "Efter", "Källa"]]

def _apply_merge_to_df(df: pd.DataFrame, row_idx: int, merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tar diff-tabellen (Fält/Före/Efter/Källa) och skriver 'Efter' till df på angiven radindex.
    """
    if merged_df.empty:
        return df
    df2 = df.copy()
    for _, r in merged_df.iterrows():
        col = str(r["Fält"])
        val = r["Efter"]
        if col in df2.columns:
            df2.iat[row_idx, df2.columns.get_loc(col)] = val
        else:
            # Om kolumn saknas, skapa den (för att inte tappa data)
            df2[col] = None
            df2.iat[row_idx, df2.columns.get_loc(col)] = val
    return df2


# ── Huvudvy ────────────────────────────────────────────────────────────────
def manual_collect_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    UI för enskild uppdatering (Yahoo/FMP/SEC) och sparning.
    Returnerar ev. uppdaterad DataFrame (df) som appen kan fortsätta använda.
    """
    if df is None or df.empty:
        st.warning("Ingen data att visa.")
        return df

    # Init session state för draft-källor
    st.session_state.setdefault("draft_yahoo", {})
    st.session_state.setdefault("draft_fmp", {})
    st.session_state.setdefault("draft_sec", {})

    # Välj ticker
    tickers = []
    colname_ticker = None
    for cand in ["Ticker", "ticker", "Symbol", "symbol"]:
        if cand in df.columns:
            colname_ticker = cand
            tickers = list(pd.unique(df[cand].dropna().astype(str)))
            break
    if not tickers:
        st.error("Kunde inte hitta kolumnen 'Ticker' (eller 'Symbol') i Data-fliken.")
        return df

    st.caption("Välj ticker")
    selected_ticker = st.selectbox("", sorted(tickers), key="manual_collect_ticker")  # noqa: B008
    if not selected_ticker:
        return df

    # Plocka aktuell rad
    mask = df[colname_ticker] == selected_ticker
    if not mask.any():
        st.error(f"Hittade ingen rad med {colname_ticker}='{selected_ticker}'.")
        return df
    row_idx = df.index[mask][0]
    cur_row = df.loc[row_idx].to_dict()

    # Knappar
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Hämta från Yahoo", use_container_width=True):
            if yahoo_get_all is None:
                st.error("Yahoo-fetchern saknas.")
            else:
                try:
                    st.session_state["draft_yahoo"] = yahoo_get_all(selected_ticker) or {}
                except Exception as e:
                    st.error(f"Fel vid Yahoo-hämtning: {e}")

    with col2:
        if st.button("Hämta från FMP", use_container_width=True):
            if fmp_get_all is None:
                st.error("FMP-fetchern saknas.")
            else:
                try:
                    st.session_state["draft_fmp"] = fmp_get_all(selected_ticker) or {}
                except Exception as e:
                    st.error(f"Fel vid FMP-hämtning: {e}")

    with col3:
        if st.button("Hämta från SEC", use_container_width=True):
            if sec_get_all is None:
                st.error("SEC-fetchern saknas.")
            else:
                try:
                    st.session_state["draft_sec"] = sec_get_all(selected_ticker) or {}
                except Exception as e:
                    st.error(f"Fel vid SEC-hämtning: {e}")

    # Summering
    cnt_y = _count_nonempty(st.session_state.get("draft_yahoo"))
    cnt_f = _count_nonempty(st.session_state.get("draft_fmp"))
    cnt_s = _count_nonempty(st.session_state.get("draft_sec"))
    st.markdown(f"**Summering:** Yahoo={cnt_y}, FMP={cnt_f}, SEC={cnt_s}")

    # FMP Debug-expander
    try:
        if fmp_get_all_verbose is not None:
            with st.expander("FMP debug (mappade fält + varningar)"):
                try:
                    mapped, fields, warns = fmp_get_all_verbose(selected_ticker)
                    st.write("Fält som mappas till appen:", fields)
                    if warns:
                        st.write("Varningar:", " | ".join(warns))
                    st.json(mapped)
                except Exception as e:
                    st.info(f"FMP debug kunde inte visas: {e}")
    except Exception:
        pass

    # Förhandsgranska
    show_preview = st.button("🔍 Förhandsgranska skillnader")
    preview_df = pd.DataFrame()
    if show_preview:
        preview_df = _merge_preview(
            cur_row,
            st.session_state.get("draft_yahoo", {}) or {},
            st.session_state.get("draft_fmp", {}) or {},
            st.session_state.get("draft_sec", {}) or {},
        )
        if preview_df.empty:
            st.info("Inga förändringar att spara.")
        else:
            st.dataframe(preview_df, use_container_width=True)

    # Spara
    if st.button("💾 Spara till Google Sheets", use_container_width=True):
        # Om ingen förhandsvisning renderats, skapa en on-the-fly för att spara rätt
        if preview_df.empty:
            preview_df = _merge_preview(
                cur_row,
                st.session_state.get("draft_yahoo", {}) or {},
                st.session_state.get("draft_fmp", {}) or {},
                st.session_state.get("draft_sec", {}) or {},
            )
        if preview_df.empty:
            st.warning("Inget att spara.")
            return df

        df2 = _apply_merge_to_df(df, row_idx, preview_df)

        # Försök skriva tillbaka med sheets-modulen om den finns,
        # annars returnerar vi df2 så app.py kan ta vid.
        if _sheets_ok and callable(_sheets_save_df):
            try:
                _sheets_save_df(df2)  # skriv hela df till Google Sheets
                st.success("Sparat till Google Sheets.")
            except Exception as e:
                st.warning(f"Kunde inte spara via sheets-modulen: {e}\nReturnerar uppdaterat df till appen.")
                st.session_state["draft_yahoo"] = {}
                st.session_state["draft_fmp"] = {}
                st.session_state["draft_sec"] = {}
                return df2
        else:
            st.info("Ingen sheets-funktion hittad – returnerar uppdaterat df till appen.")

        # Nollställ drafts efter spar
        st.session_state["draft_yahoo"] = {}
        st.session_state["draft_fmp"] = {}
        st.session_state["draft_sec"] = {}
        return df2

    # Kort visning av aktuell rad
    with st.expander("Visa aktuell rad (kort info)"):
        show_cols = [c for c in ["Ticker", "Bolagsnamn", "Kurs", "Valuta", "P/S TTM",
                                 "Utestående aktier (milj.)", "Omsättning (M)", "Kassa (M)"]
                     if c in df.columns]
        st.dataframe(df.loc[[row_idx], show_cols] if show_cols else df.loc[[row_idx]], use_container_width=True)

    return df
