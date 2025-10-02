# -*- coding: utf-8 -*-
"""
Orkestrerar full uppdatering för en ticker:
- Yahoo → FMP → SEC (i den ordningen)
- Merge till dina svensk-namngivna kolumner
- Stämplar TS Full och (om pris) TS Kurs
- Lämnar manuella prognoser orörda
- Returnerar (df_out, log_meddelande)

Publik:
    run_update_full(df: pd.DataFrame, ticker: str, user_rates: dict) -> (pd.DataFrame, str)
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import math

import pandas as pd
import numpy as np
import streamlit as st

# Konfiguration & utils
from stockapp.config import (
    FINAL_COLS,
    USE_YAHOO,
    USE_FMP,
    USE_SEC,
)
from stockapp.utils import (
    ensure_schema,
    to_float,
    now_stamp,
    stamp_fields_ts,
    risk_label_from_mcap,
)

# Försök importera fetchers – alla är valfria
_yahoo = None
_fmp = None
_sec = None
try:
    from stockapp.fetchers import yahoo as _yahoo  # type: ignore
except Exception:
    pass
try:
    from stockapp.fetchers import fmp as _fmp  # type: ignore
except Exception:
    pass
try:
    from stockapp.fetchers import sec as _sec  # type: ignore
except Exception:
    pass


# ------------------------------------------------------------
# Hjälp: safe fetch med flera möjliga funktionsnamn
# ------------------------------------------------------------
def _call_fetch_all(mod, ticker: str) -> Dict[str, object]:
    """
    Försöker anropa en av: get_all_fields / fetch_all / fetch
    Returnerar dict med nyckeltal (kan vara tom).
    """
    if mod is None:
        return {}
    for name in ("get_all_fields", "fetch_all", "fetch"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                out = fn(ticker)
                return out if isinstance(out, dict) else {}
            except Exception:
                return {}
    return {}

def _call_price_only(mod, ticker: str) -> Optional[float]:
    """
    Försöker anropa get_live_price / get_price
    """
    if mod is None:
        return None
    for name in ("get_live_price", "get_price"):
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                p = fn(ticker)
                if p is not None:
                    f = float(p)
                    return f if f > 0 else None
            except Exception:
                return None
    return None


# ------------------------------------------------------------
# Hjälp: normalisera källnycklar → våra kolumner
# ------------------------------------------------------------
def _normalize_source_dict(src: Dict[str, object]) -> Dict[str, object]:
    """
    Tar ett käll-dict (från Yahoo/FMP/SEC) och mappar tänkbara nycklar till
    dina standardkolumner. Okända nycklar ignoreras tyst.
    """
    if not isinstance(src, dict):
        return {}

    # Mappningar: "möjlig källnyckel" → "vår kolumn"
    MAP = {
        # Identitet/klassning
        "shortName": "Bolagsnamn",
        "name": "Bolagsnamn",
        "longName": "Bolagsnamn",
        "sector": "Sektor",
        "industry": "Industri",
        "currency": "Valuta",

        # Pris / MCap / shares
        "price": "Kurs",
        "regularMarketPrice": "Kurs",
        "marketCap": "Market Cap",
        "sharesOutstanding": "Utestående aktier (milj.)",  # om i styck; vi konverterar nedan till milj.

        # P/S (nivå + 4 kvartal)
        "ps": "P/S",
        "psQ1": "P/S Q1",
        "psQ2": "P/S Q2",
        "psQ3": "P/S Q3",
        "psQ4": "P/S Q4",

        # Lönsamhet / värdering
        "grossMargin": "Gross margin (%)",
        "operatingMargin": "Operating margin (%)",
        "netMargin": "Net margin (%)",
        "roe": "ROE (%)",
        "evToEbitda": "EV/EBITDA (ttm)",
        "pb": "P/B",
        "fcfYield": "FCF Yield (%)",
        "dividendYield": "Dividend yield (%)",
        "dividendPayoutFCF": "Dividend payout (FCF) (%)",
        "netDebtToEbitda": "Net debt / EBITDA",
        "debtToEquity": "Debt/Equity",

        # Kassa
        "cash": "Kassa (M)",
        "cashM": "Kassa (M)",
    }

    out: Dict[str, object] = {}
    for k, v in src.items():
        key = MAP.get(k, None)
        if not key:
            continue
        out[key] = v

    # Post-process: sharesOutstanding i styck → “milj.”
    if "Utestående aktier (milj.)" in out:
        try:
            shares = float(out["Utestående aktier (milj.)"])
            # om det ser ut som styck (större än 1e3) → konvertera till miljoner
            if shares > 1e3:
                out["Utestående aktier (milj.)"] = shares / 1e6
        except Exception:
            pass

    # Procentfält – normalisera om de kan komma som 0..1
    for pct_col in ("Gross margin (%)", "Operating margin (%)", "Net margin (%)", "FCF Yield (%)", "Dividend yield (%)"):
        if pct_col in out:
            try:
                val = float(out[pct_col])
                # Om uppenbart 0..1 → skala till %
                if 0.0 <= val <= 1.0:
                    out[pct_col] = val * 100.0
            except Exception:
                pass

    return out


def _merge_sources(yahoo: Dict[str, object], fmp: Dict[str, object], sec: Dict[str, object]) -> Dict[str, object]:
    """
    Merge-strategi (prioritetsordning per fält):
      Pris/Kurs: Yahoo → FMP → SEC
      MCap, Sektor/Industri, Valuta: Yahoo → FMP
      Marginaler/EV/EBITDA/ROE/PB/FCF yield/Utdelning: FMP → Yahoo
      Utest. aktier: SEC → FMP → Yahoo
      P/S & P/S Q1..Q4: Yahoo → FMP
      Kassa: FMP → SEC
    """
    merged: Dict[str, object] = {}

    def pick(col: str, *sources: Dict[str, object]):
        for s in sources:
            if s and col in s and s[col] is not None and str(s[col]) != "":
                merged[col] = s[col]
                return

    # Kärnidentitet
    pick("Bolagsnamn", yahoo, fmp)
    pick("Sektor", yahoo, fmp)
    pick("Industri", yahoo, fmp)
    pick("Valuta", yahoo, fmp)

    # Pris / Market Cap
    pick("Kurs", yahoo, fmp, sec)
    pick("Market Cap", yahoo, fmp)
    # Shares
    pick("Utestående aktier (milj.)", sec, fmp, yahoo)

    # P/S och kvartal
    pick("P/S", yahoo, fmp)
    for q in ("P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"):
        pick(q, yahoo, fmp)

    # Marginaler & värderingar
    for col in (
        "Gross margin (%)",
        "Operating margin (%)",
        "Net margin (%)",
        "ROE (%)",
        "EV/EBITDA (ttm)",
        "P/B",
        "FCF Yield (%)",
        "Dividend yield (%)",
        "Dividend payout (FCF) (%)",
        "Net debt / EBITDA",
        "Debt/Equity",
    ):
        pick(col, fmp, yahoo)

    # Kassa
    pick("Kassa (M)", fmp, sec)

    return merged


# ------------------------------------------------------------
# Publik: kör full uppdatering för en ticker
# ------------------------------------------------------------
def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """
    Kör Yahoo→FMP→SEC, mergea, stämpla, lämna manuella prognoser orörda.
    Skriver bara kolumner som finns i FINAL_COLS.
    """
    if df is None or df.empty:
        return df, "Tom databas"

    # Hitta rad
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(ticker).upper()]
    if len(ridx) == 0:
        return df, f"{ticker}: Ticker saknas i tabellen"
    idx = ridx[0]

    # 1) Hämta källor (defensivt)
    y_raw, f_raw, s_raw = {}, {}, {}
    y_norm, f_norm, s_norm = {}, {}, {}
    sources_used: List[str] = []

    # Yahoo
    if USE_YAHOO and _yahoo is not None:
        y_raw = _call_fetch_all(_yahoo, ticker)
        y_norm = _normalize_source_dict(y_raw)
        if y_norm:
            sources_used.append("Yahoo")

        # säkerställ pris via separat snabbfunktion om saknas
        if "Kurs" not in y_norm:
            p = _call_price_only(_yahoo, ticker)
            if p:
                y_norm["Kurs"] = p

    # FMP
    if USE_FMP and _fmp is not None:
        f_raw = _call_fetch_all(_fmp, ticker)
        f_norm = _normalize_source_dict(f_raw)
        if f_norm:
            sources_used.append("FMP")

    # SEC
    if USE_SEC and _sec is not None:
        s_raw = _call_fetch_all(_sec, ticker)
        s_norm = _normalize_source_dict(s_raw)
        if s_norm:
            sources_used.append("SEC")

    # 2) Merge
    merged = _merge_sources(y_norm, f_norm, s_norm)

    # 3) Post-calc här:
    #    - P/S-snitt (Q1..Q4)
    ps_q = []
    for q in ("P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"):
        if q in merged:
            try:
                ps_q.append(float(merged[q]))
            except Exception:
                pass
    if ps_q:
        merged["P/S-snitt (Q1..Q4)"] = float(np.nanmean(ps_q))

    #    - Risklabel från Market Cap
    mcap = to_float(merged.get("Market Cap"))
    if mcap and mcap > 0:
        merged["Risklabel"] = risk_label_from_mcap(mcap)

    # 4) Filtrera till kolumner vi *får* skriva
    allowed = set(FINAL_COLS) | {
        "Senast uppdaterad källa",
        "TS Full",
        "TS Kurs",
        # Vi tar höjd för att några fält kan saknas i FINAL_COLS men ändå är bra att skriva:
        "P/S-snitt (Q1..Q4)",
        "Risklabel",
    }
    to_write = {k: v for k, v in merged.items() if k in allowed}

    # 5) Lämna *manuella prognoser* orörda (de skrivs aldrig av orkestratorn)
    MANUAL_KEYS = {"Omsättning i år (M)", "Omsättning nästa år (M)"}
    for k in list(to_write.keys()):
        if k in MANUAL_KEYS:
            to_write.pop(k, None)

    # 6) Skriv in i df
    df = ensure_schema(df, FINAL_COLS)
    for k, v in to_write.items():
        df.at[idx, k] = v

    # 7) Stämplar
    # Pris stämplas om pris fanns i merged
    if "Kurs" in to_write:
        df = stamp_fields_ts(df, ["Kurs"], ts_suffix=" TS")
        # eller om du använder explicit TS-kolumner:
        df.at[idx, "TS Kurs"] = now_stamp()

    # Alltid TS Full
    df.at[idx, "TS Full"] = now_stamp()

    # Källa
    if sources_used:
        df.at[idx, "Senast uppdaterad källa"] = "+".join(sources_used)
    else:
        df.at[idx, "Senast uppdaterad källa"] = "Inga"

    # 8) Summera logg
    written = sorted(list(to_write.keys()))
    return df, f"{ticker}: uppdaterade {len(written)} fält via {df.at[idx, 'Senast uppdaterad källa']}"
