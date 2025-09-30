# -*- coding: utf-8 -*-
"""
stockapp/orchestrator.py

Orkestrerar hämtning från Yahoo + SEC + FMP, mergar och sätter tidsstämplar.
- run_update_full(ticker, df_row=None, user_rates=None)  -> (vals, debug, meta)
- run_update_price_only(ticker, df_row=None, user_rates=None) -> (vals, debug, meta)

Returnerar alltid en 3-tuple (vals, debug, meta) för att matcha batch-runnern.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Callable
from datetime import datetime

import streamlit as st

# Våra fetchers
from .fetchers import yahoo as fyahoo
from .fetchers import sec as fsec
from .fetchers import fmp as ffmp

# Hjälp & konfiguration
from .config import TS_FIELDS  # dict: fältnamn -> tidsstämpelkolumn (t.ex. "P/S" -> "P/S TS")
from .utils import now_stamp

# ------------------------------------------------------------
# Policy: vilka fält är MANUELLA och ska aldrig auto-överskrivas?
# ------------------------------------------------------------
MANUAL_FIELDS = {
    "Omsättning i år (prognos, manuell)",
    "Omsättning nästa år (prognos, manuell)",
    "GAV (SEK)",
}

# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _coalesce(*vals, default=None):
    for v in vals:
        if v is not None:
            return v
    return default

def _apply_timestamps(vals: Dict[str, Any], ts_fields: Dict[str, str], ts: Optional[str] = None) -> None:
    """
    Sätter tidsstämpel för alla fält som finns i 'vals' och har en TS-kolumn definierad.
    Krav från användaren: datum ska uppdateras även om värdet inte ändrats.
    """
    stamp = ts or now_stamp()
    for key, val in list(vals.items()):
        if key in ts_fields:
            vals[ts_fields[key]] = stamp

def _safe_call(fn: Callable, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        out, dbg = fn(*args, **kwargs)
        if not isinstance(out, dict):
            out = {}
        if not isinstance(dbg, dict):
            dbg = {}
        return out, dbg
    except Exception as e:
        return {}, {"error": str(e)}

def _hist_price_lookup_from_yahoo(ticker: str) -> Optional[Callable]:
    """
    Försök få en historisk pris-funktion från Yahoo-fetchern.
    Om fetchern inte exponerar fabriksfunktion, returnera None (SEC P/S Qx hoppat).
    """
    try:
        if hasattr(fyahoo, "make_hist_price_lookup"):
            return fyahoo.make_hist_price_lookup(ticker)
    except Exception:
        pass
    return None

# ------------------------------------------------------------
# Merge-policy
# ------------------------------------------------------------
def _merge_vals(yv: Dict[str, Any], sv: Dict[str, Any], fv: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mergar fält från Yahoo (yv), SEC (sv), FMP (fv) enligt enkel policy:
    - Bas: Yahoo (pris, valuta, mcap, sektor/industri)
    - SEC vinner för Utestående aktier, P/S & P/S Q1..Q4, TTM-baserade PS (när finns)
    - FMP fyller på kvalitetsmått (EV/EBITDA, marginaler, skuld, kassor, CF, FCF osv)
    - Manuella fält i MANUAL_FIELDS lämnas orörda (de sätts inte här)
    """
    out: Dict[str, Any] = {}

    # 1) Börja med Yahoo-bas
    for k, v in (yv or {}).items():
        if k in MANUAL_FIELDS:
            continue
        out[k] = v

    # 2) Lägg SEC för sina domäner
    sec_prefer = {
        "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        # om du i framtiden adderar TTM-namn här: "Revenue TTM", etc.
    }
    for k, v in (sv or {}).items():
        if k in MANUAL_FIELDS:
            continue
        if k in sec_prefer or k not in out:
            out[k] = v

    # 3) Lägg FMP för kvalitativa mått / komplettering
    for k, v in (fv or {}).items():
        if k in MANUAL_FIELDS:
            continue
        if (k not in out) or (out.get(k) in (None, "", 0, 0.0)):
            out[k] = v

    # Sätt alltid Ticker (om Yahoo inte lyckades sätta)
    if "Ticker" not in out and yv.get("Ticker"):
        out["Ticker"] = yv.get("Ticker")

    return out

# ------------------------------------------------------------
# Publika funktioner
# ------------------------------------------------------------
def run_update_price_only(*args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Uppdatera endast kurs/market cap/valuta/namn/sektor/industri (Yahoo).
    Kan anropas som:
        run_update_price_only(ticker)
        run_update_price_only(ticker, df_row, user_rates)
        run_update_price_only(ticker=t, user_rates=...)
    Returnerar (vals, debug, meta)
    """
    # Flexibel args-parsning
    ticker = None
    if args:
        ticker = args[0]
    ticker = kwargs.get("ticker", ticker)
    if not ticker:
        return {}, {"error": "missing ticker"}, {"runner": "price_only"}

    yv, ydbg = _safe_call(fyahoo.fetch_yahoo_combo, ticker)
    vals = {}
    # Plocka basfält
    for key in ("Ticker", "Namn", "Valuta", "Kurs", "Market Cap", "Sektor", "Industri", "Land"):
        if key in yv:
            vals[key] = yv[key]

    # Tidsstämpla
    _apply_timestamps(vals, TS_FIELDS)
    meta = {"runner": "price_only", "sources": ["yahoo"]}
    debug = {"yahoo": ydbg}
    return vals, debug, meta


def run_update_full(*args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Full auto-uppdatering:
    - Yahoo bas (kurs, valuta, mcap, namn, sektor/industri, ev. hist-pris-fn)
    - SEC (aktier, kvartalsintäkter->TTM, P/S, P/S Q1..Q4 via hist-pris)
    - FMP (EV/EBITDA, marginaler, skuld/kassa, CF/FCF m.m.)
    Sätter tidsstämplar för allt som skrivs.

    Anrop:
        run_update_full(ticker)
        run_update_full(ticker, df_row, user_rates)
        run_update_full(ticker=t, user_rates=...)

    Return: (vals, debug, meta)
    """
    # Flexibel args-parsning för batch-runner-kompatibilitet
    # args: (ticker, df_row, user_rates)
    # kwargs: ticker=..., df_row=..., user_rates=...
    ticker = None
    df_row = None
    user_rates = None
    if len(args) >= 1:
        ticker = args[0]
    if len(args) >= 2:
        df_row = args[1]
    if len(args) >= 3:
        user_rates = args[2]
    ticker = kwargs.get("ticker", ticker)
    df_row = kwargs.get("df_row", df_row)
    user_rates = kwargs.get("user_rates", user_rates)

    if not ticker:
        return {}, {"error": "missing ticker"}, {"runner": "full"}

    debug: Dict[str, Any] = {}
    sources_used = []

    # 1) Yahoo
    yv, ydbg = _safe_call(fyahoo.fetch_yahoo_combo, ticker)
    debug["yahoo"] = ydbg
    sources_used.append("yahoo")

    # Hämta market cap & prisvaluta för SEC/FMP-kontext
    mcap = _coalesce(yv.get("Market Cap"), None)
    price_ccy = _coalesce(yv.get("Valuta"), "USD")

    # Historisk prisfunktion för SEC P/S Qx (om möjlig)
    hist_price_fn = _hist_price_lookup_from_yahoo(ticker)

    # 2) SEC (snåla inte med parametrar om vi har dem)
    sv, sdbg = _safe_call(
        fsec.fetch_sec_combo,
        ticker,
        market_cap=mcap,
        price_ccy=price_ccy,
        hist_price_lookup=hist_price_fn,
        override_shares=None,
    )
    debug["sec"] = sdbg
    sources_used.append("sec")

    # 3) FMP
    fv, fdbg = _safe_call(ffmp.fetch_fmp_combo, ticker)
    debug["fmp"] = fdbg
    sources_used.append("fmp")

    # 4) Merge
    merged = _merge_vals(yv, sv, fv)

    # 5) Sätt tidsstämplar för allt vi faktiskt skriver
    _apply_timestamps(merged, TS_FIELDS)

    meta = {"runner": "full", "sources": sources_used}
    return merged, debug, meta
