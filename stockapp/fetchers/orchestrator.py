# stockapp/fetchers/orchestrator.py
# -*- coding: utf-8 -*-
"""
Orchestrator för full uppdatering av en enskild ticker.

Publikt API:
    run_update_full(df, ticker, user_rates) -> (df_out, log_str)

Källordning & strategi:
    1) Yahoo: pris + snabbprofil
    2) FMP:   fundamenta (marginaler, EV/EBITDA, FCF, utdelning, m.m.)
    3) SEC:   P/S-kvartal (robust sorterad på datum för att få med Dec/Jan)
Luckor fylls successivt. TS stämplas för varje uppdaterat fält.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import math
import numpy as np
import pandas as pd

from ..utils import (
    safe_float,
    stamp_fields_ts,
    risk_label_from_mcap,
    now_stamp,
)
from ..rates import hamta_valutakurs

# -----------------------------
# Käll-importer (tolerant)
# -----------------------------
try:
    from .yahoo import get_live_price as _yahoo_price
except Exception:
    _yahoo_price = None  # type: ignore

try:
    # bör returnera dict med nycklar när det går: se _apply_update() nedan
    from .yahoo import fetch_ticker as _yahoo_fetch
except Exception:
    _yahoo_fetch = None  # type: ignore

try:
    from .fmp import fetch_ticker as _fmp_fetch
except Exception:
    _fmp_fetch = None  # type: ignore

try:
    # bör returnera lista av dicts: [{"date": "YYYY-MM-DD", "ps": float}, ...]
    from .sec import fetch_ps_quarters as _sec_ps
except Exception:
    _sec_ps = None  # type: ignore


# -----------------------------
# Hjälpare
# -----------------------------
PRICE_COLS = ["Kurs", "Aktuell kurs"]
MCAP_COLS  = ["Market Cap (valuta)", "Market Cap", "Market Cap (SEK)"]
PSQ_COLS   = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]

# map inkommande nycklar → (kolumnnamn i df, ev. transform)
_FIELD_MAP: Dict[str, str] = {
    "currency": "Valuta",
    "price": "Kurs",                      # alternativa skrivningar hanteras i _apply_update
    "shares_outstanding": "Utestående aktier",
    "market_cap": "Market Cap (valuta)",
    "ps_ttm": "P/S",
    "ps_q1": "P/S Q1",
    "ps_q2": "P/S Q2",
    "ps_q3": "P/S Q3",
    "ps_q4": "P/S Q4",
    "sector": "Sektor",
    "industry": "Industri",
    "gross_margin": "Bruttomarginal (%)",
    "net_margin": "Nettomarginal (%)",
    "debt_to_equity": "Debt/Equity",
    "ev_ebitda": "EV/EBITDA",
    "fcf_m": "FCF (M)",
    "cash_m": "Kassa (M)",
    "runway_quarters": "Runway (kvartal)",
    "dividend_yield_pct": "Dividend Yield (%)",
    "payout_ratio_cf_pct": "Payout Ratio CF (%)",
}

def _apply_update(df: pd.DataFrame, ridx, data: Dict[str, object], updated: List[str]) -> None:
    """
    Applicera inkommande 'data' mot df[ridx] och registrera uppdaterade fältnamn i 'updated'.
    Hanterar dubbla prisfält (Kurs/Aktuell kurs) och olika MCAP-kolumner.
    """
    if not isinstance(data, dict):
        return

    for k, v in data.items():
        col = _FIELD_MAP.get(k)
        if not col:
            continue

        # värde → float om numeriskt
        val = v
        if isinstance(v, (int, float, np.integer, np.floating, str)):
            try:
                val = float(str(v).replace(",", "."))
            except Exception:
                val = v

        if col in df.columns:
            df.loc[ridx, col] = val
            updated.append(col)
        else:
            # pris special: skriv även "Aktuell kurs" om den kolumnen finns
            if col == "Kurs":
                if "Kurs" in df.columns:
                    df.loc[ridx, "Kurs"] = val
                    updated.append("Kurs")
                if "Aktuell kurs" in df.columns:
                    df.loc[ridx, "Aktuell kurs"] = val
                    updated.append("Aktuell kurs")
            # Market Cap special: försök flera
            elif col.startswith("Market Cap"):
                for mc in MCAP_COLS:
                    if mc in df.columns:
                        df.loc[ridx, mc] = val
                        updated.append(mc)
                        break
            # PS Q1..Q4 special är redan mappad via _FIELD_MAP
            else:
                # Om kolumn saknas totalt ignorerar vi (schema-skydd sker i ensure_schema på annat ställe)
                pass


def _calc_ps_avg(df: pd.DataFrame, ridx) -> Optional[float]:
    vals = []
    for c in PSQ_COLS:
        if c in df.columns:
            vals.append(safe_float(df.at[ridx, c], np.nan))
        else:
            vals.append(np.nan)
    arr = [x for x in vals if not math.isnan(x)]
    if not arr:
        return None
    avg = float(np.mean(arr))
    if "P/S-snitt" in df.columns:
        df.loc[ridx, "P/S-snitt"] = avg
    else:
        # skapa kolumn on-the-fly om saknas
        df["P/S-snitt"] = df.get("P/S-snitt", np.nan)
        df.loc[ridx, "P/S-snitt"] = avg
    return avg


def _ensure_mcap_sek(df: pd.DataFrame, ridx, user_rates: Dict[str, float]) -> None:
    cur = str(df.at[ridx, "Valuta"]) if "Valuta" in df.columns else "USD"
    rate = hamta_valutakurs(cur, user_rates)
    # välj basmcap
    mcap = None
    for c in MCAP_COLS:
        if c in df.columns:
            mcap = safe_float(df.at[ridx, c], np.nan)
            if not math.isnan(mcap):
                break
    if mcap is None or math.isnan(mcap):
        return
    if "Market Cap (SEK)" in df.columns:
        df.loc[ridx, "Market Cap (SEK)"] = float(mcap) * float(rate)


def _apply_ps_from_quarters(df: pd.DataFrame, ridx, ps_rows: List[Dict[str, object]]) -> List[str]:
    """
    ps_rows: list of {"date": "YYYY-MM-DD", "ps": float}
    Tar de 4 senaste *kronologiskt* och mappar till P/S Q1..Q4 (Q1=nyast).
    """
    upd: List[str] = []
    if not ps_rows:
        return upd
    # sortera på datum
    def _parse(d):
        try:
            return pd.to_datetime(str(d))
        except Exception:
            return pd.NaT

    rows = sorted(ps_rows, key=lambda r: _parse(r.get("date")), reverse=True)
    latest4 = rows[:4]
    # lägg in Q1..Q4 (Q1 = nyast)
    for i, rec in enumerate(latest4, start=1):
        ps = safe_float(rec.get("ps"), np.nan)
        if math.isnan(ps):
            continue
        col = f"P/S Q{i}"
        if col in df.columns:
            df.loc[ridx, col] = float(ps)
            upd.append(col)
    return upd


# -----------------------------
# Publik funktion
# -----------------------------
def run_update_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """
    Uppdaterar en enskild ticker. Returnerar (df_out, log_text).
    """
    if df is None or df.empty:
        return df, "Tomt dataframe"

    tkr = str(ticker).upper().strip()
    idxs = df.index[df["Ticker"].astype(str).str.upper() == tkr]
    if len(idxs) == 0:
        return df, f"{tkr}: Ticker hittades inte i tabellen."
    ridx = idxs[0]

    updated_fields: List[str] = []
    sources: List[str] = []

    # 1) Yahoo: live price + profil/stats
    yahoo_data: Dict[str, object] = {}
    try:
        if _yahoo_price:
            p = _yahoo_price(tkr)
            if p and p > 0:
                yahoo_data["price"] = p
        if _yahoo_fetch:
            extra = _yahoo_fetch(tkr) or {}
            if isinstance(extra, dict):
                yahoo_data.update(extra)
        if yahoo_data:
            _apply_update(df, ridx, yahoo_data, updated_fields)
            sources.append("Yahoo")
    except Exception as e:
        sources.append(f"Yahoo(! {e})")

    # 2) FMP: fundamenta
    fmp_data: Dict[str, object] = {}
    try:
        if _fmp_fetch:
            f = _fmp_fetch(tkr) or {}
            if isinstance(f, dict):
                fmp_data.update(f)
        if fmp_data:
            _apply_update(df, ridx, fmp_data, updated_fields)
            sources.append("FMP")
    except Exception as e:
        sources.append(f"FMP(! {e})")

    # 3) SEC: P/S kvartal – ta de 4 senaste kronologiskt (hanterar Dec/Jan)
    try:
        if _sec_ps:
            ps_rows = _sec_ps(tkr) or []
            ps_upd = _apply_ps_from_quarters(df, ridx, ps_rows)
            updated_fields.extend(ps_upd)
            if ps_rows:
                sources.append("SEC")
    except Exception as e:
        sources.append(f"SEC(! {e})")

    # Efter-sammanställning
    # – P/S-snitt
    psavg = _calc_ps_avg(df, ridx)
    if psavg is not None:
        updated_fields.append("P/S-snitt")

    # – Market Cap (SEK) från (valuta) × kurs
    _ensure_mcap_sek(df, ridx, user_rates)
    if "Market Cap (SEK)" in updated_fields or "Market Cap (valuta)" in updated_fields or "Market Cap" in updated_fields:
        updated_fields.append("Market Cap (SEK)")

    # – Risklabel
    #   välj mcap (valuta) i första hand, annars SEK
    mcap_val = np.nan
    for c in ["Market Cap (valuta)", "Market Cap", "Market Cap (SEK)"]:
        if c in df.columns:
            mcap_val = safe_float(df.at[ridx, c], np.nan)
            if not math.isnan(mcap_val):
                break
    if not math.isnan(mcap_val):
        if "Risklabel" in df.columns:
            df.loc[ridx, "Risklabel"] = risk_label_from_mcap(mcap_val)
            updated_fields.append("Risklabel")

    # – TS-stämpla alla uppdaterade fält
    if updated_fields:
        # unika & filtrera bort ev. icke-existerande
        unique_fields = [c for c in sorted(set(updated_fields)) if c in df.columns]
        df = stamp_fields_ts(df, unique_fields, ts_suffix=" TS")

    # – meta
    if "Senast auto-uppdaterad" in df.columns:
        df.loc[ridx, "Senast auto-uppdaterad"] = now_stamp()
    if "Senast uppdaterad källa" in df.columns:
        df.loc[ridx, "Senast uppdaterad källa"] = ", ".join([s for s in sources if s])

    # Loggtext
    if not sources and not updated_fields:
        return df, f"{tkr}: Inga ändringar hittades."
    return df, f"{tkr}: Uppdaterad ({', '.join(sorted(set(updated_fields)))}) via {', '.join(sources)}"
