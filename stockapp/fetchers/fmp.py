# -*- coding: utf-8 -*-
"""
stockapp/fetchers/fmp.py

Hämtar data från FinancialModelingPrep (FMP):
- Profil (namn, valuta, sektor, bransch), pris, market cap, utestående aktier
- P/S (nu) och P/S Q1–Q4 (kvartalsvis)
- Extra nyckeltal (om tillgängligt): Debt/Equity, bruttomarginal, nettomarginal,
  utdelningsyield, payout, OCF/FCF TTM, kassa/kortfristiga placeringar m.m.

OBS:
- Sätter *inte* Omsättning idag/nästa år (de gör du manuellt).
- Returnerar (vals, debug) där `vals` kan skrivas in i din DataFrame.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, List
import time
import requests
import streamlit as st

# Bas-konfig (hämtas ur secrets om finns)
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 1.5))
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 15))

# ----------------------------- Hjälpare --------------------------------------


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _sleep_soft():
    if FMP_CALL_DELAY > 0:
        time.sleep(FMP_CALL_DELAY)


def _fmp_get(path: str, params: Optional[dict] = None) -> Tuple[Optional[Any], int]:
    """
    Throttlad GET mot FMP med enkel circuit breaker på 429.
    Returnerar (json, statuscode).
    """
    # 429-circuit breaker
    block_until = st.session_state.get("fmp_block_until")
    if block_until and time.time() < block_until:
        return None, 429

    params = (params or {}).copy()
    if FMP_KEY:
        params["apikey"] = FMP_KEY

    url = f"{FMP_BASE}/{path}"
    last_sc = 0
    last_json = None

    for attempt in range(3):
        try:
            _sleep_soft()
            r = requests.get(url, params=params, timeout=20)
            last_sc = r.status_code
            try:
                last_json = r.json()
            except Exception:
                last_json = None

            if 200 <= last_sc < 300:
                return last_json, last_sc

            # Hantera 429 (rate limit)
            if last_sc == 429:
                # sätt spärr
                st.session_state["fmp_block_until"] = time.time() + (60 * FMP_BLOCK_MINUTES)
                time.sleep(1.0 + attempt)
                continue

            # Transienta fel
            if last_sc in (502, 503, 504):
                time.sleep(1.0 + attempt)
                continue

            return last_json, last_sc
        except Exception:
            time.sleep(1.0 + attempt)
            continue

    return last_json, last_sc


def _pick_symbol(yahoo_ticker: str) -> str:
    """
    Validera symbol via quote-short, annars search. Faller tillbaka till uppercase.
    """
    sym = str(yahoo_ticker).strip().upper()
    js, sc = _fmp_get(f"api/v3/quote-short/{sym}")
    if isinstance(js, list) and js:
        return sym
    js2, sc2 = _fmp_get("api/v3/search", {"query": sym, "limit": 1})
    if isinstance(js2, list) and js2:
        alt = js2[0].get("symbol")
        if alt:
            return str(alt).upper()
    return sym


# ----------------------------- Kärnlogik -------------------------------------


def _pull_profile(sym: str, out: Dict, dbg: Dict):
    prof, sc = _fmp_get(f"api/v3/profile/{sym}")
    dbg["profile_sc"] = sc
    if isinstance(prof, list) and prof:
        p0 = prof[0]
        name = p0.get("companyName") or p0.get("company_name")
        if name:
            out["Bolagsnamn"] = name
        cur = p0.get("currency")
        if cur:
            out["Valuta"] = str(cur).upper()
        sec = p0.get("sector")
        if sec:
            out["Sektor"] = sec
        ind = p0.get("industry")
        if ind:
            out["Bransch"] = ind
        # Dividend yield (TTM) kan ibland finnas i profile
        dy = _safe_float(p0.get("lastDiv"), 0.0)  # inte yield, men senaste utdelning/aktie
        if dy > 0 and "Årlig utdelning" not in out:
            out["Årlig utdelning"] = dy


def _pull_quote(sym: str, out: Dict, dbg: Dict):
    q, sc = _fmp_get(f"api/v3/quote/{sym}")
    dbg["quote_sc"] = sc
    if isinstance(q, list) and q:
        q0 = q[0]
        price = _safe_float(q0.get("price"))
        if price > 0:
            out["Aktuell kurs"] = price
        mcap = _safe_float(q0.get("marketCap"))
        if mcap > 0:
            out["Market Cap"] = mcap
        sh = _safe_float(q0.get("sharesOutstanding"))
        if sh > 0:
            # lagra i miljoner (matcha din databas)
            out["Utestående aktier"] = sh / 1e6


def _pull_ratios_ttm(sym: str, out: Dict, dbg: Dict):
    rttm, sc = _fmp_get(f"api/v3/ratios-ttm/{sym}")
    dbg["ratios_ttm_sc"] = sc
    if isinstance(rttm, list) and rttm:
        r0 = rttm[0]
        ps = _safe_float(r0.get("priceToSalesTTM")) or _safe_float(r0.get("priceToSalesRatioTTM"))
        if ps > 0:
            out["P/S"] = ps
        # Debt/Equity
        de = _safe_float(r0.get("debtEquityTTM") or r0.get("debtToEquity"))
        if de > 0:
            out["Debt/Equity"] = de
        # Marginaler
        gm = _safe_float(r0.get("grossProfitMarginTTM") or r0.get("grossMarginTTM")) * 100.0
        if gm > 0:
            out["Bruttomarginal (%)"] = gm
        nm = _safe_float(r0.get("netProfitMarginTTM") or r0.get("netMarginTTM")) * 100.0
        if nm > 0:
            out["Nettomarginal (%)"] = nm
        # Dividend yield/payout (om tillgängligt)
        dy = _safe_float(r0.get("dividendYielTTM") or r0.get("dividendYieldTTM")) * 100.0
        if dy > 0:
            out["Dividend Yield (%)"] = dy
        payout = _safe_float(r0.get("payoutRatioTTM")) * 100.0
        if payout > 0:
            out["Payout Ratio (%)"] = payout


def _pull_key_metrics_ttm(sym: str, out: Dict, dbg: Dict):
    km, sc = _fmp_get(f"api/v3/key-metrics-ttm/{sym}")
    dbg["key_metrics_ttm_sc"] = sc
    if isinstance(km, list) and km:
        k0 = km[0]
        # P/S om saknas
        if "P/S" not in out:
            ps = _safe_float(k0.get("priceToSalesRatioTTM") or k0.get("priceToSalesTTM"))
            if ps > 0:
                out["P/S"] = ps
        # EV/EBITDA (kan vara intressant för scoring)
        ev_ebitda = _safe_float(k0.get("enterpriseValueOverEBITDATTM"))
        if ev_ebitda > 0:
            out["EV/EBITDA"] = ev_ebitda


def _pull_ratios_quarterly(sym: str, out: Dict, dbg: Dict):
    rq, sc = _fmp_get(f"api/v3/ratios/{sym}", {"period": "quarter", "limit": 8})
    dbg["ratios_q_sc"] = sc
    if not (isinstance(rq, list) and rq):
        return
    # Ta de 4 senaste med giltigt priceToSalesRatio
    ps_vals: List[float] = []
    for row in rq:
        v = _safe_float(row.get("priceToSalesRatio"))
        if v > 0:
            ps_vals.append(v)
        if len(ps_vals) >= 4:
            break
    # Mappa till Q1..Q4
    for i, v in enumerate(ps_vals[:4], start=1):
        out[f"P/S Q{i}"] = float(v)


def _pull_cashflow_ttm(sym: str, out: Dict, dbg: Dict):
    # OCF/FCF TTM
    cfttm, sc_cf = _fmp_get(f"api/v3/cash-flow-statement-ttm/{sym}")
    dbg["cf_ttm_sc"] = sc_cf
    if isinstance(cfttm, list) and cfttm:
        c0 = cfttm[0]
        ocf = _safe_float(c0.get("operatingCashFlowTTM"))
        if ocf != 0:
            out["Operativt kassafl. TTM"] = ocf
        fcf = _safe_float(c0.get("freeCashFlowTTM"))
        if fcf != 0:
            out["Fritt kassaflöde TTM"] = fcf

    # Balans (kassa/kortfristiga placeringar)
    bs_ttm, sc_bs = _fmp_get(f"api/v3/balance-sheet-statement-ttm/{sym}")
    dbg["bs_ttm_sc"] = sc_bs
    if isinstance(bs_ttm, list) and bs_ttm:
        b0 = bs_ttm[0]
        cash1 = _safe_float(b0.get("cashAndShortTermInvestmentsTTM"))
        cash2 = _safe_float(b0.get("cashAndCashEquivalentsTTM"))
        cash = cash1 if cash1 > 0 else cash2
        if cash > 0:
            out["Kassa & ST-invest TTM"] = cash


# ----------------------------- Publikt API -----------------------------------


def fetch_fmp_full(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Full FMP-hämtning för en ticker.
    Returnerar (vals, debug).
    - Sätter: Bolagsnamn, Valuta, Sektor, Bransch, Aktuell kurs, Market Cap, Utestående aktier
              P/S, P/S Q1..Q4
              (Extra) Debt/Equity, Bruttomarginal (%), Nettomarginal (%),
                      Dividend Yield (%), Payout Ratio (%),
                      EV/EBITDA, Operativt kassafl. TTM, Fritt kassaflöde TTM,
                      Kassa & ST-invest TTM
    - Sätter *inte* Omsättning idag/nästa år (manuellt fält).
    """
    vals: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"ticker": ticker}

    if not FMP_KEY:
        dbg["error"] = "FMP_API_KEY saknas"
        return vals, dbg

    sym = _pick_symbol(ticker)
    dbg["symbol"] = sym

    try:
        _pull_profile(sym, vals, dbg)
    except Exception as e:
        dbg["profile_err"] = str(e)

    try:
        _pull_quote(sym, vals, dbg)
    except Exception as e:
        dbg["quote_err"] = str(e)

    try:
        _pull_ratios_ttm(sym, vals, dbg)
    except Exception as e:
        dbg["ratios_ttm_err"] = str(e)

    # Om P/S saknas – key metrics TTM som fallback
    try:
        if "P/S" not in vals:
            _pull_key_metrics_ttm(sym, vals, dbg)
    except Exception as e:
        dbg["key_metrics_ttm_err"] = str(e)

    # P/S historik (kvartal)
    try:
        _pull_ratios_quarterly(sym, vals, dbg)
    except Exception as e:
        dbg["ratios_q_err"] = str(e)

    # Kassaflöden / Kassa
    try:
        _pull_cashflow_ttm(sym, vals, dbg)
    except Exception as e:
        dbg["cf_bs_err"] = str(e)

    # Sätt källa för spårning i appen (kan visas i UI)
    dbg["source"] = "FMP"
    return vals, dbg
