# stockapp/manual_collect.py
from __future__ import annotations

import math
import typing as t

import pandas as pd
import streamlit as st

# ── Fetchers ───────────────────────────────────────────────────────────────
try:
    from .fetchers.yahoo import get_all as yahoo_get_all
except Exception:
    yahoo_get_all = None  # type: ignore

try:
    from .fetchers.fmp import (
        get_all as fmp_get_all,
        get_all_verbose as fmp_get_all_verbose,
        format_fetch_summary as fmp_format_summary,
    )
except Exception:
    fmp_get_all = None  # type: ignore
    fmp_get_all_verbose = None  # type: ignore
    fmp_format_summary = lambda s, f, w: "FMP: (ingen formatterare)"

try:
    from .fetchers.sec import get_all as sec_get_all
except Exception:
    sec_get_all = None  # type: ignore


# ── (Valfri) Sheets-integration ────────────────────────────────────────────
_sheets_ok = False
_sheets_save_df = None  # type: ignore
try:
    from .sheets import save_dataframe  # föredragen
    _sheets_ok = True
    _sheets_save_df = save_dataframe
except Exception:
    try:
        from .sheets import write_dataframe as save_dataframe  # type: ignore
        _sheets_ok = True
        _sheets_save_df = save_dataframe
    except Exception:
        _sheets_ok = False
        _sheets_save_df = None  # type: ignore


# ── Fält-prioritet (matchar dina rubriker) ────────────────────────────────
FIELD_PRIORITY: dict[str, list[str]] = {
    "Kurs": ["yahoo", "fmp", "sec"],
    "P/S": ["fmp", "yahoo", "sec"],
    "Market Cap": ["fmp", "yahoo", "sec"],
    "Market Cap (M)": ["fmp", "yahoo", "sec"],
    "Utestående aktier (milj.)": ["sec", "fmp", "yahoo"],
    "Omsättning i år (M)": ["fmp", "sec", "yahoo"],
    "Kassa (M)": ["sec", "fmp", "yahoo"],
    "Valuta": ["yahoo", "fmp", "sec"],
    "Bolagsnamn": ["yahoo", "fmp", "sec"],
    "Börs": ["fmp", "yahoo", "sec"],
    "Sektor": ["fmp", "yahoo", "sec"],
    "Industri": ["fmp", "yahoo", "sec"],
    "Bransch": ["fmp", "yahoo", "sec"],
    # lägg fler vid behov
}

ALIASES: dict[str, str] = {
    "P/S TTM": "P/S",
    "P/S (TTM, modell)": "P/S",
}

# ── Hjälpare ───────────────────────────────────────────────────────────────
def _canon(field: str) -> str:
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
    fields: set[str] = set(FIELD_PRIORITY.keys()) | set(yv.keys()) | set(fv.keys()) | set(sv.keys())
    rows: list[dict[str, t.Any]] = []
    for field in sorted(fields):
        f = _canon(field)
        before = cur_row.get(f)
        after, src = _pick_value(f, yv, fv, sv)
        if _safe(after):
            if not _safe(before) or before != after:
                rows.append({"Fält": f, "Före": before, "Efter": after, "Källa": src})
    if not rows:
        return pd.DataFrame(columns=["Fält", "Före", "Efter", "Källa"])
    return pd.DataFrame(rows)[["Fält", "Före", "Efter", "Källa"]]

def _apply_merge_to_df(df: pd.DataFrame, row_idx: int, merged_df: pd.DataFrame) -> pd.DataFrame:
    if merged_df.empty:
        return df
    df2 = df.copy()
    for _, r in merged_df.iterrows():
        col = str(r["Fält"])
        val = r["Efter"]
        if col not in df2.columns:
            df2[col] = None
        df2.iat[row_idx, df2.columns.get_loc(col)] = val
    return df2


# ── Huvudvy ────────────────────────────────────────────────────────────────
def manual_collect_view(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        st.warning("Ingen data att visa.")
        return df

    # init draft + diagnostik
    st.session_state.setdefault("draft_yahoo", {})
    st.session_state.setdefault("draft_fmp", {})
    st.session_state.setdefault("draft_sec", {})
    st.session_state.setdefault("fmp_diag", {"fields": [], "warnings": [], "summary": ""})

    # hitta ticker-kolumn
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

    # aktuell rad
    mask = df[colname_ticker] == selected_ticker
    if not mask.any():
        st.error(f"Hittade ingen rad med {colname_ticker}='{selected_ticker}'.")
        return df
    row_idx = df.index[mask][0]
    cur_row = df.loc[row_idx].to_dict()

    # knappar
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Hämta från Yahoo", use_container_width=True):
            if yahoo_get_all is None:
                st.error("Yahoo-fetchern saknas.")
            else:
                try:
                    st.session_state["draft_yahoo"] = yahoo_get_all(selected_ticker) or {}
                    st.success(f"Yahoo hämtade {len(st.session_state['draft_yahoo'])} fält.")
                except Exception as e:
                    st.error(f"Fel vid Yahoo-hämtning: {e}")

    with col2:
        if st.button("Hämta från FMP", use_container_width=True):
            if fmp_get_all_verbose is None:
                st.error("FMP-fetchern saknas.")
            else:
                try:
                    mapped, fields, warns = fmp_get_all_verbose(selected_ticker)
                    st.session_state["draft_fmp"] = mapped or {}
                    st.session_state["fmp_diag"] = {
                        "fields": fields or [],
                        "warnings": warns or [],
                        "summary": fmp_format_summary("FMP", fields or [], warns or []),
                    }
                    if fields:
                        st.success(f"FMP hämtade {len(fields)} fält: {', '.join(fields)}")
                    else:
                        if warns:
                            st.warning("FMP hämtade 0 fält. " + " | ".join(warns))
                        else:
                            st.warning("FMP hämtade 0 fält.")
                except Exception as e:
                    st.error(f"Fel vid FMP-hämtning: {e}")

    with col3:
        if st.button("Hämta från SEC", use_container_width=True):
            if sec_get_all is None:
                st.error("SEC-fetchern saknas.")
            else:
                try:
                    st.session_state["draft_sec"] = sec_get_all(selected_ticker) or {}
                    st.success(f"SEC hämtade {len(st.session_state['draft_sec'])} fält.")
                except Exception as e:
                    st.error(f"Fel vid SEC-hämtning: {e}")

    # summering
    cnt_y = _count_nonempty(st.session_state.get("draft_yahoo"))
    cnt_f = _count_nonempty(st.session_state.get("draft_fmp"))
    cnt_s = _count_nonempty(st.session_state.get("draft_sec"))
    st.markdown(f"**Summering:** Yahoo={cnt_y}, FMP={cnt_f}, SEC={cnt_s}")

    # FMP debug – visar senaste diagnostik; gör INTE nya API-anrop
    with st.expander("FMP debug (mappade fält + varningar)"):
        diag = st.session_state.get("fmp_diag", {}) or {}
        fields = diag.get("fields", [])
        warns = diag.get("warnings", [])
        summary = diag.get("summary", "")
        if summary:
            st.write(summary)
        if fields:
            st.write("Fält:", ", ".join(fields))
        if warns:
            st.write("Varningar:", " | ".join(warns))
        st.json(st.session_state.get("draft_fmp", {}))

    # förhandsgranskning
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

    # spara
    if st.button("💾 Spara till Google Sheets", use_container_width=True):
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

        if _sheets_ok and callable(_sheets_save_df):
            try:
                _sheets_save_df(df2)
                st.success("Sparat till Google Sheets.")
            except Exception as e:
                st.warning(f"Kunde inte spara via sheets-modulen: {e}\nReturnerar uppdaterat df till appen.")
                # nollställ drafts
                st.session_state["draft_yahoo"] = {}
                st.session_state["draft_fmp"] = {}
                st.session_state["draft_sec"] = {}
                return df2
        else:
            st.info("Ingen sheets-funktion hittad – returnerar uppdaterat df till appen.")

        # nollställ drafts efter spar
        st.session_state["draft_yahoo"] = {}
        st.session_state["draft_fmp"] = {}
        st.session_state["draft_sec"] = {}
        return df2

    # kort vy av aktuell rad
    with st.expander("Visa aktuell rad (kort info)"):
        show_cols = [c for c in [
            "Ticker", "Bolagsnamn", "Kurs", "Valuta", "P/S",
            "Utestående aktier (milj.)", "Omsättning i år (M)", "Kassa (M)"
        ] if c in df.columns]
        st.dataframe(df.loc[[row_idx], show_cols] if show_cols else df.loc[[row_idx]], use_container_width=True)

    return df
