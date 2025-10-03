# -*- coding: utf-8 -*-
"""
stockapp.manual_collect
-----------------------
Manuell insamlings-vy för en vald ticker med 4 knappar:
- Hämta från Yahoo
- Hämta från FMP
- Hämta från SEC
- Spara till Google Sheets

Visar:
- antal fält per källa
- förhandsgranskning av skillnader (före/efter + källa)
- skyddar manuella prognosfält ("Omsättning i år (M)", "Omsättning nästa år (M)")

Kräver att fetchers finns: stockapp.fetchers.yahoo/fmp/sec (sec är valfri).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import pandas as pd
import streamlit as st

# fetchers
from .fetchers.yahoo import get_all as yf_get_all
from .fetchers.fmp import get_all as fmp_get_all
try:
    from .fetchers.sec import get_all as sec_get_all  # valfri
except Exception:
    sec_get_all = None  # type: ignore

# app helpers
from .config import FINAL_COLS
from .utils import ensure_schema, now_stamp, to_float, format_large_number, risk_label_from_mcap, with_backoff
from .storage import spara_data


# Manuella fält som ALDRIG skrivs över automatiskt
MANUAL_FIELDS = {"Omsättning i år (M)", "Omsättning nästa år (M)"}

# Käll-prioritet (vilken vinner om flera har samma fält)
FIELD_PRIORITY = {
    "Bolagsnamn": ["yahoo", "fmp", "sec"],
    "Valuta": ["yahoo", "fmp", "sec"],
    "Kurs": ["yahoo", "fmp", "sec"],
    "Market Cap": ["yahoo", "fmp", "sec"],
    "Utestående aktier (milj.)": ["fmp", "yahoo", "sec"],

    "EV/EBITDA (ttm)": ["fmp", "yahoo", "sec"],
    "P/B": ["fmp", "yahoo", "sec"],
    "P/S": ["fmp", "yahoo", "sec"],
    "Gross margin (%)": ["fmp", "yahoo", "sec"],
    "Operating margin (%)": ["fmp", "yahoo", "sec"],
    "Net margin (%)": ["fmp", "yahoo", "sec"],
    "ROE (%)": ["fmp", "yahoo", "sec"],
    "Debt/Equity": ["fmp", "yahoo", "sec"],
    "Net debt / EBITDA": ["fmp", "sec", "yahoo"],
    "FCF Yield (%)": ["fmp", "yahoo", "sec"],
    "Dividend yield (%)": ["fmp", "yahoo", "sec"],
    "Dividend payout (FCF) (%)": ["fmp", "sec", "yahoo"],
    "Kassa (M)": ["fmp", "sec", "yahoo"],

    "P/S Q1": ["fmp", "yahoo", "sec"],
    "P/S Q2": ["fmp", "yahoo", "sec"],
    "P/S Q3": ["fmp", "yahoo", "sec"],
    "P/S Q4": ["fmp", "yahoo", "sec"],

    "Sektor": ["fmp", "yahoo", "sec"],
    "Industri": ["fmp", "yahoo", "sec"],
}

def _safe(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float):
        return not math.isnan(v)
    if isinstance(v, str):
        return v.strip() != ""
    return True

def _pick_value(field: str, yv: Dict[str, Any], fv: Dict[str, Any], sv: Dict[str, Any]) -> Tuple[Any, str]:
    """Välj värde + källa enligt FIELD_PRIORITY (fallback yahoo→fmp→sec)."""
    prio = FIELD_PRIORITY.get(field, ["yahoo", "fmp", "sec"])
    for src in prio:
        if src == "yahoo" and _safe(yv.get(field)):
            return yv[field], "Yahoo"
        if src == "fmp" and _safe(fv.get(field)):
            return fv[field], "FMP"
        if src == "sec" and _safe(sv.get(field)):
            return sv[field], "SEC"
    return None, ""

def _ps_avg_from_quarters(d: Dict[str, Any]) -> Optional[float]:
    qs: List[float] = []
    for q in ("P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"):
        v = to_float(d.get(q, None))
        if v is not None and v > 0:
            qs.append(v)
    if not qs:
        return None
    return round(sum(qs) / float(len(qs)), 4)

def _merge_preview(current: pd.Series, yv: Dict[str, Any], fv: Dict[str, Any], sv: Dict[str, Any]) -> pd.DataFrame:
    """
    Bygger en tabell: [Fält, Före, Efter, Källa] för alla kända fält (FINAL_COLS).
    Skyddar MANUAL_FIELDS (Efter blir None för dem).
    """
    rows = []
    fields = [c for c in FINAL_COLS if c not in ("TS Kurs", "TS Full", "TS Omsättning i år", "TS Omsättning nästa år")]
    for f in fields:
        before = current.get(f, None)
        after = None
        src = ""

        if f in MANUAL_FIELDS:
            after = None
            src = "Manuell (skyddad)"
        else:
            after, src = _pick_value(f, yv, fv, sv)

        # Sätt Risklabel baserat på Market Cap om fältet är Risklabel
        if f == "Risklabel":
            mc = to_float(current.get("Market Cap", None))
            if mc is None or mc <= 0:
                # kolla om nya MC kommer i merge
                mc_after, _src2 = _pick_value("Market Cap", yv, fv, sv)
                mc = to_float(mc_after)
            if mc is not None and mc > 0:
                after = risk_label_from_mcap(mc)
                src = src or "Beräknad"

        rows.append([f, before, after, src])

    prev = pd.DataFrame(rows, columns=["Fält", "Före", "Efter", "Källa"])

    # Lägg P/S-snitt (Q1..Q4) i förhandsgranskningen
    ps_after = _ps_avg_from_quarters({r[0]: r[2] for r in rows})
    if ps_after is not None:
        i = prev.index[prev["Fält"] == "P/S-snitt (Q1..Q4)"]
        if len(i) > 0:
            prev.at[i[0], "Efter"] = ps_after
            if not prev.at[i[0], "Källa"]:
                prev.at[i[0], "Källa"] = "Beräknad"

    # Filtrera ut rader där Efter är None ELLER lika med Före (visa bara ändringar)
    def _eq(a, b) -> bool:
        if a is None and b is None:
            return True
        return str(a) == str(b)

    preview = prev[(prev["Efter"].notna()) & (~prev.apply(lambda r: _eq(r["Före"], r["Efter"]), axis=1))]
    return preview.reset_index(drop=True)

def _apply_and_save(df: pd.DataFrame, ticker: str, preview: pd.DataFrame) -> Tuple[pd.DataFrame, int, bool]:
    """
    Skriver ändringar från preview tillbaka till df för vald ticker
    (respekterar MANUAL_FIELDS), stämplar TS-kolumner, sparar till Sheets.
    Returnerar (df_out, antal_fält, pris_ändrades).
    """
    df = ensure_schema(df.copy(), FINAL_COLS)
    tkr = str(ticker).upper().strip()
    idxs = df.index[df["Ticker"].astype(str).str.upper() == tkr].tolist()
    if not idxs:
        # skapa tom rad
        base = {c: None for c in FINAL_COLS}
        base["Ticker"] = tkr
        df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
        ridx = df.index[-1]
    else:
        ridx = idxs[0]

    old_price = to_float(df.at[ridx, "Kurs"]) if "Kurs" in df.columns else None

    changed = 0
    for _, row in preview.iterrows():
        f = str(row["Fält"])
        if f in MANUAL_FIELDS:
            continue
        if f not in df.columns:
            continue
        df.at[ridx, f] = row["Efter"]
        changed += 1

    # Risklabel sanity (om Market Cap finns)
    mc = to_float(df.at[ridx, "Market Cap"]) if "Market Cap" in df.columns else None
    if mc is not None and mc > 0 and "Risklabel" in df.columns:
        df.at[ridx, "Risklabel"] = risk_label_from_mcap(mc)

    # Tidsstämplar
    now = now_stamp()
    if "TS Full" in df.columns:
        df.at[ridx, "TS Full"] = now

    new_price = to_float(df.at[ridx, "Kurs"]) if "Kurs" in df.columns else None
    price_changed = False
    if new_price is not None and (old_price is None or float(new_price) != float(old_price)):
        if "TS Kurs" in df.columns:
            df.at[ridx, "TS Kurs"] = now
        price_changed = True

    # Spara
    with_backoff(spara_data, df)

    return df, changed, price_changed


def manual_collect_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit-vy: välj ticker → hämta (Yahoo/FMP/SEC) → förhandsgranska → spara.
    Returnerar uppdaterad df (om sparad), annars original.
    """
    st.header("🧰 Manuell insamling (4 knappar)")

    df = ensure_schema(df.copy(), FINAL_COLS)
    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return df

    # välj ticker
    tickers = df["Ticker"].astype(str).tolist()
    tkr = st.selectbox("Välj ticker", options=tickers, index=0)
    cur_row = df[df["Ticker"].astype(str) == str(tkr)].iloc[0] if not df.empty else pd.Series(dtype=object)

    # status-lagring
    st.session_state.setdefault("draft_yahoo", {})
    st.session_state.setdefault("draft_fmp", {})
    st.session_state.setdefault("draft_sec", {})

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    # 1) Yahoo
    if col1.button("Hämta från Yahoo"):
        try:
            y = yf_get_all(tkr) or {}
            st.session_state["draft_yahoo"] = y
            st.success(f"Yahoo: hämtade {len(y)} fält.")
        except Exception as e:
            st.error(f"Yahoo-fel: {e}")

    # 2) FMP
    if col2.button("Hämta från FMP"):
        try:
            f = fmp_get_all(tkr) or {}
            st.session_state["draft_fmp"] = f
            st.success(f"FMP: hämtade {len(f)} fält.")
        except Exception as e:
            st.error(f"FMP-fel: {e}")

    # 3) SEC (valfri)
    if sec_get_all is None:
        col3.button("Hämta från SEC", disabled=True)
        st.caption("SEC-modul saknas eller är inaktiverad.")
    else:
        if col3.button("Hämta från SEC"):
            try:
                s = sec_get_all(tkr) or {}
                st.session_state["draft_sec"] = s
                st.success(f"SEC: hämtade {len(s)} fält.")
            except Exception as e:
                st.error(f"SEC-fel: {e}")

    # kort summering
    ycnt = len(st.session_state["draft_yahoo"])
    fcnt = len(st.session_state["draft_fmp"])
    scnt = len(st.session_state["draft_sec"]) if sec_get_all is not None else 0
    st.write(f"**Summering:** Yahoo={ycnt}, FMP={fcnt}, SEC={scnt}")

    # Förhandsgranskning (visa bara ändringar)
    if st.button("🔍 Förhandsgranska skillnader"):
        prev = _merge_preview(cur_row, st.session_state["draft_yahoo"], st.session_state["draft_fmp"], st.session_state["draft_sec"] if sec_get_all else {})
        if prev.empty:
            st.info("Inga skillnader att spara.")
        else:
            # Formatera några kända fält
            mask_mc = prev["Fält"] == "Market Cap"
            if mask_mc.any():
                for i in prev.index[mask_mc]:
                    before = prev.at[i, "Före"]
                    after = prev.at[i, "Efter"]
                    prev.at[i, "Före"] = format_large_number(to_float(before), "USD") if before is not None else before
                    prev.at[i, "Efter"] = format_large_number(to_float(after), "USD") if after is not None else after

            st.session_state["preview_table"] = prev
            st.dataframe(prev, use_container_width=True, hide_index=True)

    # 4) Spara
    if col4.button("💾 Spara till Google Sheets"):
        prev = st.session_state.get("preview_table", pd.DataFrame(columns=["Fält", "Före", "Efter", "Källa"]))
        if prev.empty:
            # Om ingen preview gjorts: generera on-the-fly så vi ändå kan spara
            prev = _merge_preview(cur_row, st.session_state["draft_yahoo"], st.session_state["draft_fmp"], st.session_state["draft_sec"] if sec_get_all else {})
        if prev.empty:
            st.warning("Inget att spara (inga ändringar).")
            return df

        df2, changed, price_changed = _apply_and_save(df, tkr, prev)
        st.success(f"Sparade {changed} fält för {tkr}.{' Kurs ändrades.' if price_changed else ''}")
        # Nollställ preview och drafts för att undvika förvirring
        st.session_state["preview_table"] = pd.DataFrame(columns=["Fält", "Före", "Efter", "Källa"])
        st.session_state["draft_yahoo"] = {}
        st.session_state["draft_fmp"] = {}
        st.session_state["draft_sec"] = {}
        return df2

    # Visa aktuell rad (kort)
    with st.expander("Visa aktuell rad (kort info)"):
        show_cols = [c for c in ["Bolagsnamn", "Ticker", "Valuta", "Kurs", "Market Cap", "Risklabel", "Sektor", "Industri"] if c in df.columns]
        st.write(cur_row[show_cols])

    return df
