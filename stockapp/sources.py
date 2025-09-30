# -*- coding: utf-8 -*-
"""
Kombinerar Yahoo + FMP + SEC.
Exponerar:
- run_update_full(ticker, user_rates=None, prefer_sec=False) -> (data, facts, log)
- run_update_price_only(ticker)                              -> (data, facts, log)
Alla tre fetchers returnerar (data, facts, log).
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import time

from .fetchers.yahoo import fetch_yahoo
from .fetchers.fmp   import fetch_fmp
from .fetchers.sec   import fetch_sec

from .config import USE_FMP, USE_SEC

def _merge(base: Dict[str, Any], bfacts: Dict[str, str],
           newd: Dict[str, Any], nfacts: Dict[str, str],
           prefer_existing_ts: bool = False) -> None:
    """
    Enkelt “bäst tillgänglig”-merge:
    - tar alltid icke-None
    - om båda finns: behåll befintligt om prefer_existing_ts (annars skriv över)
    """
    for k, v in (newd or {}).items():
        if v is None:
            continue
        if k not in base:
            base[k] = v; bfacts[k] = nfacts.get(k, "")
        else:
            # skriv över, förutom om vi explicit vill behålla tidigare
            if not prefer_existing_ts:
                base[k]  = v
                bfacts[k] = nfacts.get(k, bfacts.get(k, ""))

def run_update_price_only(ticker: str) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """
    Minimal uppdatering (kurs/MCAP/aktier + namn/sektor om möjligt).
    """
    data, facts, log = {}, {}, []
    y_d, y_f, y_l = fetch_yahoo(ticker)
    _merge(data, facts, y_d, y_f)
    log += [f"Yahoo: {m}" for m in y_l]
    return data, facts, log

def run_update_full(ticker: str, user_rates: Optional[Dict[str, float]] = None,
                    prefer_sec: bool = False) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """
    Full uppdatering:
    1) Yahoo (baseline)
    2) FMP   (fundamentals)   – om USE_FMP
    3) SEC   (US-komplettering) – om USE_SEC
    """
    t0 = time.time()
    data: Dict[str, Any]  = {}
    facts: Dict[str, str] = {}
    log:  List[str]       = []

    # 1) Yahoo
    y_d, y_f, y_l = fetch_yahoo(ticker)
    _merge(data, facts, y_d, y_f)
    log += [f"Yahoo: {m}" for m in y_l]

    # 2) FMP
    if USE_FMP:
        f_d, f_f, f_l = fetch_fmp(ticker)
        _merge(data, facts, f_d, f_f)  # FMP får gärna skriva över Yahoo om mer exakt
        log += [f"FMP: {m}" for m in f_l]
    else:
        log.append("FMP: USE_FMP=False – hoppar över.")

    # 3) SEC
    if USE_SEC:
        s_d, s_f, s_l = fetch_sec(ticker)
        # Om prefer_sec=True låter vi SEC skriva över ev. Yahoo/FMP för konfliktfält
        _merge(data, facts, s_d, s_f, prefer_existing_ts=not prefer_sec)
        log += [f"SEC: {m}" for m in s_l]
    else:
        log.append("SEC: USE_SEC=False – hoppar över.")

    log.append(f"Sammanfogning klar på {time.time()-t0:.2f}s")
    return data, facts, log
