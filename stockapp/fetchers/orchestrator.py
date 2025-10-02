# -*- coding: utf-8 -*-
"""
stockapp.fetchers.orchestrator
------------------------------
Kör full uppdatering för en ticker genom flera källor i följd:
1) Yahoo (pris/kapitalisering/kvartals-PS m.m.)
2) FMP (EV/EBITDA, FCF-yield, Debt/Equity, P/B, sektor/industri/valuta m.m.)
3) SEC (kompletterande siffror, kassaposter etc.)

Strategi:
- Vi samlar upp dict från varje källa: {nyckel: värde}
- Mappar till DF-kolumner via _map_to_df_fields(...)
- Mergeregler: första källa (Yahoo) har prioritet; senare källor fyller luckor.
- Beräknar P/S-snitt (Q1..Q4) om kvartal finns.
- Sätter "TS Full" och "Senast uppdaterad källa".
- Returnerar (df_out, logtext)
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st

# Försök importera fetchers; om någon saknas hoppar vi bara över den
try:
    from .yahoo import get_all_fields as _yahoo_all
except Exception:  # pragma: no cover
    _yahoo_all = None  # type: ignore

try:
    from .fmp import get_all_fields as _fmp_all
except Exception:  # pragma: no cover
    _fmp_all = None  # type: ignore

try:
    from .sec import get_all_fields as _sec_all
except Exception:  # pragma: no cover
    _sec_all = None  # type: ignore

# verktyg/konfig
from ..utils import now_stamp, to_float  # konsekventa helpers
from ..config import FINAL_COLS

# ------------------------------------------------------------
# Mapping från käll-nycklar till DF-kolumner
# ------------------------------------------------------------
# Vi stödjer både engelska & svenska kolumnnamn där appen använt båda.
def _map_to_df_fields(src: Dict[str, Any]) -> Dict[str, Any]:
    """Mappar generiska källnycklar -> dina DF-kolumner."""
    out: Dict[str, Any] = {}

    # Bas – pris/kapitalisering/antal aktier
    if "price" in src:
        out["Kurs"] = to_float(src.get("price"))
    if "marketCap" in src:
        out["Market Cap"] = to_float(src.get("marketCap"))
    if "sharesOutstanding" in src:
        so = to_float(src.get("sharesOutstanding"))
        if so is not None and so > 0:
            # Vi sätter båda nycklarna (med & utan " (milj.)") för kompatibilitet
            out["Utestående aktier"] = so / 1e6
            out["Utestående aktier (milj.)"] = so / 1e6

    # Valuta
    if "currency" in src and src.get("currency"):
        out["Valuta"] = str(src.get("currency")).upper()

    # Sektor/industri – både ENG & SWE
    if "sector" in src and src.get("sector"):
        out["Sektor"] = str(src.get("sector"))
        out["Sector"] = str(src.get("sector"))
    if "industry" in src and src.get("industry"):
        out["Industri"] = str(src.get("industry"))
        out["Industry"] = str(src.get("industry"))

    # Multiplar / marginaler
    if "psTTM" in src:
        out["P/S"] = to_float(src.get("psTTM"))
    if "evEbitdaTTM" in src:
        out["EV/EBITDA (ttm)"] = to_float(src.get("evEbitdaTTM"))
    if "debtToEquity" in src:
        out["Debt/Equity"] = to_float(src.get("debtToEquity"))
    if "pb" in src:
        out["P/B"] = to_float(src.get("pb"))
    if "dividendYield" in src:
        out["Dividend yield (%)"] = to_float(src.get("dividendYield"))
    if "payoutFCF" in src:
        out["Dividend payout (FCF) (%)"] = to_float(src.get("payoutFCF"))
    if "fcfYield" in src:
        out["FCF Yield (%)"] = to_float(src.get("fcfYield"))

    # Marginaler – skriv både ENG & SWE där det finns
    if "grossMargin" in src:
        gm = to_float(src.get("grossMargin"))
        out["Gross margin (%)"] = gm
        out["Bruttomarginal (%)"] = gm
    if "operatingMargin" in src:
        om = to_float(src.get("operatingMargin"))
        out["Operating margin (%)"] = om
        out["Rörelsemarginal (%)"] = om
    if "netMargin" in src:
        nm = to_float(src.get("netMargin"))
        out["Net margin (%)"] = nm
        out["Nettomarginal (%)"] = nm
    if "roe" in src:
        out["ROE (%)"] = to_float(src.get("roe"))

    # Kassaposition – om vi får full valuta, visa även i miljoner
    if "cashAndEquivalents" in src:
        cash = to_float(src.get("cashAndEquivalents"))
        if cash is not None:
            out["Kassa (M)"] = cash / 1e6

    # Kvartals-P/S om källa ger psQ1..psQ4
    for i in (1, 2, 3, 4):
        key = f"psQ{i}"
        if key in src:
            out[f"P/S Q{i}"] = to_float(src.get(key))

    return {k: v for k, v in out.items() if v is not None}


def _merge_keep_first(dst: Dict[str, Any], add: Dict[str, Any]) -> None:
    """Fyll luckor i dst med add, men skriv inte över befintliga (prioritet: tidigare källa)."""
    for k, v in add.items():
        if k not in dst or dst[k] is None or (isinstance(dst[k], float) and math.isnan(dst[k])):
            dst[k] = v


def _compute_ps_avg(payload: Dict[str, Any]) -> None:
    """Beräkna P/S-snitt (Q1..Q4) om någon eller flera kvartal finns."""
    qs = [payload.get("P/S Q1"), payload.get("P/S Q2"), payload.get("P/S Q3"), payload.get("P/S Q4")]
    vals = [float(x) for x in qs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if vals:
        payload["P/S-snitt (Q1..Q4)"] = float(np.mean(vals))


def _apply_payload_to_df(df: pd.DataFrame, ticker: str, payload: Dict[str, Any]) -> pd.DataFrame:
    """Skriv in payload till rätt rad i df; skapa ev. rad om ticker saknas."""
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    if not mask.any():
        # skapa ny rad med FINAL_COLS som stomme
        base = {c: np.nan for c in FINAL_COLS}
        base["Ticker"] = ticker
        df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
        mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()

    idx = df.index[mask][0]
    for k, v in payload.items():
        if k not in df.columns:
            # lägg till kolumn on-the-fly
            df[k] = np.nan
        df.at[idx, k] = v

    # TS Full & källa
    df.at[idx, "TS Full"] = now_stamp()
    return df


# ------------------------------------------------------------
# Publikt API
# ------------------------------------------------------------
def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, str]:
    """
    Kör full uppdatering för 'ticker' och returnerar (df_out, logstr).
    - Anropar Yahoo → FMP → SEC.
    - Mergar data och skriver in i df.
    """
    tkr = (ticker or "").upper().strip()
    if not tkr:
        return df, "Ingen ticker"

    payload: Dict[str, Any] = {}
    used_sources: List[str] = []
    logs: List[str] = []

    # 1) Yahoo
    if _yahoo_all is not None:
        try:
            y = _yahoo_all(tkr) or {}
            if y:
                used_sources.append("Yahoo")
                mapped = _map_to_df_fields(y)
                _merge_keep_first(payload, mapped)
                logs.append(f"Yahoo: {len(mapped)} fält")
        except Exception as e:
            logs.append(f"Yahoo: fel {e}")

    # 2) FMP
    if _fmp_all is not None:
        try:
            f = _fmp_all(tkr) or {}
            if f:
                used_sources.append("FMP")
                mapped = _map_to_df_fields(f)
                _merge_keep_first(payload, mapped)
                logs.append(f"FMP: {len(mapped)} fält")
        except Exception as e:
            logs.append(f"FMP: fel {e}")

    # 3) SEC
    if _sec_all is not None:
        try:
            s = _sec_all(tkr) or {}
            if s:
                used_sources.append("SEC")
                mapped = _map_to_df_fields(s)
                _merge_keep_first(payload, mapped)
                logs.append(f"SEC: {len(mapped)} fält")
        except Exception as e:
            logs.append(f"SEC: fel {e}")

    # Om inget kom in alls – logga och returnera original
    if not payload:
        logs.append("Inga fält från någon källa.")
        return df, " | ".join(logs)

    # Beräkna P/S-snitt (Q1..Q4) om möjligt
    _compute_ps_avg(payload)

    # Sätt "Senast uppdaterad källa"
    if used_sources:
        payload["Senast uppdaterad källa"] = ",".join(used_sources)

    # Skriv in i DF
    df2 = _apply_payload_to_df(df.copy(), tkr, payload)
    return df2, " | ".join(logs)
