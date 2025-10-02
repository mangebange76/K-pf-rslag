# -*- coding: utf-8 -*-
"""
stockapp.fetchers.fmp
---------------------
Hämtar nyckeltal från FinancialModelingPrep (FMP) och returnerar ett dict
med KÄLLNYCKLAR som vår orkestrator normaliserar vidare:

- Identitet/klassning: sector, industry, currency
- Pris & värdering: price, marketCap, sharesOutstanding, ps (TTM)
- Marginaler: grossMargin, operatingMargin, netMargin
- Avkastning: roe
- Multiplar: evToEbitda, pb
- Kassaflöde: fcfYield, dividendYield, dividendPayoutFCF
- Skuld: netDebtToEbitda, debtToEquity
- Likviditet: cash
- P/S per kvartal (approximativt): psQ1, psQ2, psQ3, psQ4
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple

import math
import time
import requests
import streamlit as st


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _base_url() -> str:
    return st.secrets.get("FMP_BASE", "https://financialmodelingprep.com").rstrip("/")

def _apikey() -> str:
    key = st.secrets.get("FMP_API_KEY", "")
    if not key:
        raise RuntimeError("FMP_API_KEY saknas i st.secrets")
    return str(key)

def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Optional[Any]:
    """GET med API-nyckel och enkel felhantering."""
    if params is None:
        params = {}
    params["apikey"] = _apikey()
    url = f"{_base_url()}{path}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        return j
    except Exception:
        return None

def _to_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return default
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default

def _pct01_to_pct(x: Optional[float]) -> Optional[float]:
    """Om värde ser ut som 0..1, skala till %."""
    if x is None:
        return None
    try:
        fx = float(x)
    except Exception:
        return None
    if 0.0 <= fx <= 1.0:
        return fx * 100.0
    return fx

def _safe_div(n: Optional[float], d: Optional[float]) -> Optional[float]:
    if n is None or d is None or d == 0:
        return None
    return n / d


# ------------------------------------------------------------
# Publika hjälp-funktioner (valfria)
# ------------------------------------------------------------
def get_live_price(ticker: str) -> Optional[float]:
    """Pris från quote endpoint (fallback om Yahoo saknas)."""
    q = _get(f"/api/v3/quote/{ticker.upper()}")
    try:
        row = (q or [])[0]
    except Exception:
        return None
    price = _to_float(row.get("price")) or _to_float(row.get("previousClose"))
    if price and price > 0:
        return price
    return None


# ------------------------------------------------------------
# Huvud: hämta alla fält vi kan
# ------------------------------------------------------------
def get_all_fields(ticker: str) -> Dict[str, Any]:
    """
    Returnerar ett dict med källnycklar:
      price, marketCap, sharesOutstanding, sector, industry, currency,
      ps, psQ1..psQ4,
      grossMargin, operatingMargin, netMargin, roe,
      evToEbitda, pb,
      fcfYield, dividendYield, dividendPayoutFCF,
      netDebtToEbitda, debtToEquity,
      cash
    Saknas data – lämna nyckeln borta (orkestratorn tar hand om merge).
    """
    tkr = str(ticker).upper().strip()
    out: Dict[str, Any] = {}

    # ---- 1) Quote: price, marketCap, sharesOutstanding, p/s (TTM?), currency
    q = _get(f"/api/v3/quote/{tkr}")
    if isinstance(q, list) and q:
        q0 = q[0] or {}
        price = _to_float(q0.get("price")) or _to_float(q0.get("previousClose"))
        if price and price > 0:
            out["price"] = price

        mcap = _to_float(q0.get("marketCap"))
        if mcap and mcap > 0:
            out["marketCap"] = mcap

        sh = _to_float(q0.get("sharesOutstanding"))
        if sh and sh > 0:
            out["sharesOutstanding"] = sh  # OBS: i styck (orkestratorn konverterar vid behov)

        # P/S TTM (om tillgängligt)
        ps_ttm = _to_float(q0.get("priceToSalesTrailing12Months"))
        if ps_ttm and ps_ttm > 0:
            out["ps"] = ps_ttm

        cur = q0.get("currency") or q0.get("financialCurrency")
        if isinstance(cur, str) and cur:
            out["currency"] = cur.upper()

    # ---- 2) Profile: sector, industry, currency (fallback)
    prof = _get(f"/api/v3/profile/{tkr}")
    if isinstance(prof, list) and prof:
        p0 = prof[0] or {}
        sec = p0.get("sector")
        ind = p0.get("industry")
        cur = p0.get("currency")

        if isinstance(sec, str) and sec:
            out["sector"] = sec
        if isinstance(ind, str) and ind:
            out["industry"] = ind
        if isinstance(cur, str) and cur and "currency" not in out:
            out["currency"] = cur.upper()

    # ---- 3) Ratios TTM: marginaler, multiplar, utdelning, skuldsättning
    ratios = _get(f"/api/v3/ratios-ttm/{tkr}")
    if isinstance(ratios, list) and ratios:
        r0 = ratios[0] or {}
        # Margins
        gm = _pct01_to_pct(_to_float(r0.get("grossProfitMarginTTM")))
        om = _pct01_to_pct(_to_float(r0.get("operatingProfitMarginTTM")))
        nm = _pct01_to_pct(_to_float(r0.get("netProfitMarginTTM")))
        if gm is not None:
            out["grossMargin"] = gm
        if om is not None:
            out["operatingMargin"] = om
        if nm is not None:
            out["netMargin"] = nm

        # ROE
        roe = _pct01_to_pct(_to_float(r0.get("returnOnEquityTTM") or r0.get("roeTTM")))
        if roe is not None:
            out["roe"] = roe

        # EV/EBITDA (enterpriseValueMultipleTTM ≈ EV/EBITDA)
        ev_eb = _to_float(r0.get("enterpriseValueMultipleTTM"))
        if ev_eb is not None:
            out["evToEbitda"] = ev_eb

        # P/B
        pb = _to_float(r0.get("priceToBookRatioTTM"))
        if pb is not None:
            out["pb"] = pb

        # Dividend yield (TTM)
        dy = _pct01_to_pct(_to_float(r0.get("dividendYieldTTM")))
        if dy is not None:
            out["dividendYield"] = dy

        # Debt/Equity
        de = _to_float(r0.get("debtEquityRatioTTM") or r0.get("debtToEquityTTM"))
        if de is not None:
            out["debtToEquity"] = de

    # ---- 4) Key-metrics TTM: extra nycklar (ibland överlappar ratios)
    km = _get(f"/api/v3/key-metrics-ttm/{tkr}")
    if isinstance(km, list) and km:
        k0 = km[0] or {}
        # Om EV/EBITDA saknas från ratios, försök här:
        if "evToEbitda" not in out:
            ev_eb2 = _to_float(k0.get("enterpriseValueToEbitdaTTM") or k0.get("enterpriseValueToEbitda"))
            if ev_eb2 is not None:
                out["evToEbitda"] = ev_eb2

        # P/B fallback
        if "pb" not in out:
            pb2 = _to_float(k0.get("priceToBookRatioTTM"))
            if pb2 is not None:
                out["pb"] = pb2

    # ---- 5) Balance sheet (quarter, senaste) för Cash & NetDebt
    bs = _get(f"/api/v3/balance-sheet-statement/{tkr}", params={"period": "quarter", "limit": 1})
    cash_val = None
    net_debt = None
    if isinstance(bs, list) and bs:
        b0 = bs[0] or {}
        cash_val = _to_float(b0.get("cashAndCashEquivalents"))
        if cash_val is None:
            cash_val = _to_float(b0.get("cashAndShortTermInvestments"))
        if cash_val is not None:
            # returnera i "valutaenheter", orkestratorn skalar till M vid behov
            out["cash"] = cash_val

        # Net debt om fält finns, annars beräkna (TotalDebt - Cash)
        net_debt = _to_float(b0.get("netDebt"))
        if net_debt is None:
            total_debt = _to_float(b0.get("totalDebt"))
            if total_debt is not None and cash_val is not None:
                net_debt = total_debt - cash_val

    # ---- 6) Cash-flow (quarter, 4 st) för FCF & utdelningar
    cf = _get(f"/api/v3/cash-flow-statement/{tkr}", params={"period": "quarter", "limit": 4})
    fcf_ttm = None
    div_paid_ttm_abs = None  # positivt belopp
    if isinstance(cf, list) and cf:
        fcf_sum = 0.0
        div_sum_abs = 0.0
        any_fcf = False
        any_div = False
        for row in cf:
            fcf_q = _to_float(row.get("freeCashFlow"))
            if fcf_q is not None:
                fcf_sum += fcf_q
                any_fcf = True
            div_q = _to_float(row.get("dividendsPaid"))
            if div_q is not None:
                # dividendsPaid är oftast negativt i FMP → använd absolutbelopp
                div_sum_abs += abs(div_q)
                any_div = True
        if any_fcf:
            fcf_ttm = fcf_sum
        if any_div:
            div_paid_ttm_abs = div_sum_abs

    # fcfYield = FCF_TTM / MarketCap * 100
    mcap = out.get("marketCap")
    if fcf_ttm is not None and isinstance(mcap, (int, float)) and mcap and mcap > 0:
        fcf_y = (fcf_ttm / float(mcap)) * 100.0
        out["fcfYield"] = fcf_y

    # dividendPayoutFCF = (utdelningar / FCF) * 100 (om FCF > 0)
    if fcf_ttm is not None and fcf_ttm > 0 and div_paid_ttm_abs is not None:
        out["dividendPayoutFCF"] = (div_paid_ttm_abs / fcf_ttm) * 100.0

    # ---- 7) Income statement (quarter, 4 st) för EBITDA & Revenue (till psQ1..Q4)
    inc = _get(f"/api/v3/income-statement/{tkr}", params={"period": "quarter", "limit": 4})
    ebitda_ttm = None
    rev_q_list: List[float] = []
    if isinstance(inc, list) and inc:
        e_sum = 0.0
        any_e = False
        for row in inc:
            e = _to_float(row.get("ebitda"))
            if e is not None:
                e_sum += e
                any_e = True
            r = _to_float(row.get("revenue"))
            if r is not None and r > 0:
                rev_q_list.append(r)
        if any_e:
            ebitda_ttm = e_sum

    # netDebt / EBITDA
    if net_debt is not None and ebitda_ttm and ebitda_ttm > 0:
        out["netDebtToEbitda"] = net_debt / ebitda_ttm

    # psQ1..psQ4 (approx): marketCap / (revenue_q * 4)
    # Endast om vi både har mcap och åtminstone 1 kvartal revenue
    if isinstance(mcap, (int, float)) and mcap and rev_q_list:
        # Vi vill ha senaste 4 i fallande kronologi: inc returnerar normalt senaste först
        # Säkra ordning: använd exakt listan som kom (limit=4 => redan de senaste)
        ps_quarters: List[Optional[float]] = []
        for rq in rev_q_list:
            ps_q = _safe_div(float(mcap), (rq * 4.0))
            ps_quarters.append(ps_q)
        # Fyll till 4 element med None om färre
        while len(ps_quarters) < 4:
            ps_quarters.append(None)
        # Map till nycklar
        for i, key in enumerate(("psQ1", "psQ2", "psQ3", "psQ4")):
            if ps_quarters[i] is not None:
                out[key] = ps_quarters[i]

    # ---- 8) Sista finputs: om ps saknas, beräkna TTM ps = mcap / rev_ttm
    if "ps" not in out and rev_q_list:
        rev_ttm = sum(rev_q_list) if len(rev_q_list) >= 1 else None
        if rev_ttm and mcap and rev_ttm > 0:
            out["ps"] = float(mcap) / float(rev_ttm)

    # ---- 9) Slut – rensa upp lite urspårade värden
    # Konvertera ev. procent > 1 som råkvoter (nästan alltid redan redan fixat)
    for k in ("grossMargin", "operatingMargin", "netMargin", "roe", "fcfYield", "dividendYield", "dividendPayoutFCF"):
        if k in out:
            out[k] = _to_float(out[k])

    # EV/EBITDA/pb/debtToEquity/netDebtToEbitda/ps/psQi – floatify
    for k in ("evToEbitda", "pb", "debtToEquity", "netDebtToEbitda", "ps", "psQ1", "psQ2", "psQ3", "psQ4"):
        if k in out:
            out[k] = _to_float(out[k])

    # cash, marketCap, sharesOutstanding, price – floatify
    for k in ("cash", "marketCap", "sharesOutstanding", "price"):
        if k in out:
            out[k] = _to_float(out[k])

    # currency/sector/industry – strippade strings
    for k in ("currency", "sector", "industry"):
        if k in out and isinstance(out[k], str):
            out[k] = out[k].strip()

    return out
