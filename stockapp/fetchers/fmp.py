# stockapp/fetchers/fmp.py
# -*- coding: utf-8 -*-
"""
FMP-hämtare (Financial Modeling Prep).

Kräver st.secrets:
  FMP_API_KEY      – din API-nyckel
  (valfritt) FMP_BASE – bas-URL, default "https://financialmodelingprep.com"

Publikt API:
    get_live_price(ticker) -> Optional[float]
    fetch_ticker(ticker)   -> Dict[str, Any]

Returnerade nycklar (om tillgängliga):
    currency                -> "USD" / "SEK" / ...
    price                   -> float
    shares_outstanding      -> float (antal aktier)
    market_cap              -> float (i bolagets valuta)
    ps_ttm                  -> float (TTM P/S)
    sector                  -> str
    industry                -> str
    gross_margin            -> float (%)  [0..100]
    net_margin              -> float (%)  [0..100]
    debt_to_equity          -> float (kvot)
    ev_ebitda               -> float
    fcf_m                   -> float (miljoner, valuta = company currency)
    cash_m                  -> float (miljoner)
    runway_quarters         -> float (≈ hur många kvartal kassan räcker vid negativ FCF)
    dividend_yield_pct      -> float (%)
    payout_ratio_cf_pct     -> float (%), approx via (utdelningar / FCF) om data finns
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import math
import requests
import streamlit as st


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(str(x).replace(",", "."))
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _base_url() -> str:
    return st.secrets.get("FMP_BASE", "https://financialmodelingprep.com").rstrip("/")


def _apikey() -> str:
    return st.secrets.get("FMP_API_KEY", "")


def _get(path: str, params: Dict[str, Any] | None = None) -> Tuple[Optional[Any], int]:
    """
    GET mot FMP. Returnerar (json, status_code) eller (None, sc).
    """
    if params is None:
        params = {}
    key = _apikey()
    if not key:
        return None, 0
    params = {**params, "apikey": key}
    url = f"{_base_url()}{path}"
    try:
        r = requests.get(url, params=params, timeout=20)
        sc = r.status_code
        if sc != 200:
            return None, sc
        js = r.json()
        return js, sc
    except Exception:
        return None, 0


# ------------------------------------------------------------
# Publika funktioner
# ------------------------------------------------------------
def get_live_price(ticker: str) -> Optional[float]:
    """
    Livekurs via /api/v3/quote/{ticker}
    """
    js, sc = _get(f"/api/v3/quote/{ticker}")
    if not js or not isinstance(js, list) or not js:
        return None
    price = _to_float(js[0].get("price"))
    if price is not None and price > 0:
        return price
    # fallback: try regularMarketPrice
    price = _to_float(js[0].get("previousClose"))
    return price


def fetch_ticker(ticker: str) -> Dict[str, Any]:
    """
    Hämtar normaliserat paket nyckeltal för en ticker via FMP.
    Returnerar endast nycklar som har värde (onödiga fält utelämnas).
    """
    out: Dict[str, Any] = {}
    api_key = _apikey()
    if not api_key:
        return out  # ingen nyckel => ingen data

    # --------------------------------------
    # 1) QUOTE – pris & market cap
    # --------------------------------------
    q, _ = _get(f"/api/v3/quote/{ticker}")
    if isinstance(q, list) and q:
        q0 = q[0]
        price = _to_float(q0.get("price")) or _to_float(q0.get("previousClose"))
        if price is not None and price > 0:
            out["price"] = float(price)
        mcap = _to_float(q0.get("marketCap"))
        if mcap is not None and mcap > 0:
            out["market_cap"] = float(mcap)

    # --------------------------------------
    # 2) PROFILE – valuta, sektor, industri, aktier
    # --------------------------------------
    prof, _ = _get(f"/api/v3/profile/{ticker}")
    if isinstance(prof, list) and prof:
        p0 = prof[0]
        curr = p0.get("currency")
        if curr:
            out["currency"] = str(curr).upper()
        sector = p0.get("sector")
        industry = p0.get("industry")
        if sector:
            out["sector"] = str(sector)
        if industry:
            out["industry"] = str(industry)
        # FMP har ofta sharesOutstanding i profile
        shares = _to_float(p0.get("sharesOutstanding"))
        if shares and shares > 0:
            out["shares_outstanding"] = float(shares)

    # --------------------------------------
    # 3) RATIOS TTM – P/S TTM, marginaler, D/E, EV/EBITDA, dividend yield
    # --------------------------------------
    ratios, _ = _get(f"/api/v3/ratios-ttm/{ticker}")
    if isinstance(ratios, list) and ratios:
        r0 = ratios[0]
        ps_ttm = _to_float(r0.get("priceToSalesTTM"))
        if ps_ttm and ps_ttm > 0:
            out["ps_ttm"] = float(ps_ttm)

        gm = _to_float(r0.get("grossProfitMarginTTM"))
        if gm is not None:
            out["gross_margin"] = float(gm * 100.0)  # konvertera kvot -> %
        nm = _to_float(r0.get("netProfitMarginTTM"))
        if nm is not None:
            out["net_margin"] = float(nm * 100.0)

        de = _to_float(r0.get("debtEquityRatioTTM"))
        if de is not None:
            out["debt_to_equity"] = float(de)

        ev_ebitda = _to_float(r0.get("enterpriseValueOverEBITDATTM"))
        if ev_ebitda is not None:
            out["ev_ebitda"] = float(ev_ebitda)

        dy = _to_float(r0.get("dividendYieldTTM"))
        if dy is not None:
            out["dividend_yield_pct"] = float(dy * 100.0)  # kvot -> %

    # --------------------------------------
    # 4) CASH FLOW (annual) – free cash flow + utdelningar
    # --------------------------------------
    cf, _ = _get(f"/api/v3/cash-flow-statement/{ticker}", params={"period": "annual", "limit": 1})
    fcflow_val = None
    dividends_paid = None
    if isinstance(cf, list) and cf:
        c0 = cf[0]
        # freeCashFlow varierar i namngivning, men FMP använder "freeCashFlow"
        fcflow_val = _to_float(c0.get("freeCashFlow"))
        if fcflow_val is not None:
            out["fcf_m"] = float(fcflow_val) / 1e6  # miljoner

        dividends_paid = _to_float(c0.get("dividendsPaid"))  # oftast negativt tal

    # --------------------------------------
    # 5) BALANCE SHEET (annual) – Cash & Cash Equivalents
    # --------------------------------------
    bs, _ = _get(f"/api/v3/balance-sheet-statement/{ticker}", params={"period": "annual", "limit": 1})
    total_cash_val = None
    if isinstance(bs, list) and bs:
        b0 = bs[0]
        # vanliga fält: cashAndCashEquivalents, cashAndShortTermInvestments
        total_cash_val = _to_float(b0.get("cashAndCashEquivalents")) or _to_float(b0.get("cashAndShortTermInvestments"))
        if total_cash_val is not None:
            out["cash_m"] = float(total_cash_val) / 1e6  # miljoner

    # --------------------------------------
    # 6) Runway (kvartal) – grov approx baserat på negativt FCF
    # --------------------------------------
    if total_cash_val is not None and fcflow_val is not None and fcflow_val < 0:
        cash_m = float(total_cash_val) / 1e6
        burn_m_per_year = abs(float(fcflow_val)) / 1e6
        if burn_m_per_year > 0:
            out["runway_quarters"] = float(4.0 * (cash_m / burn_m_per_year))

    # --------------------------------------
    # 7) Payout ratio (CF-baserad) approx: utdelningar / FCF (om FCF > 0)
    # --------------------------------------
    if fcflow_val is not None and fcflow_val > 0 and dividends_paid is not None and dividends_paid != 0:
        # dividendsPaid är oftast negativt → ta absoluta beloppet
        payout = abs(dividends_paid) / float(fcflow_val)
        out["payout_ratio_cf_pct"] = float(payout * 100.0)

    # --------------------------------------
    # 8) Valuta sista utväg
    # --------------------------------------
    if "currency" not in out:
        # FMP returnerar ofta USD om inget annat – men vi sätter inte default
        # hellre lämna tomt än att riskera fel.
        pass

    return out
