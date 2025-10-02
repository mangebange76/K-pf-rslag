# -*- coding: utf-8 -*-
"""
stockapp.fetchers.sec
---------------------
Hämtar kvartals-/årsdata från SEC (data.sec.gov) och härleder nyckeltal.

Returnerar ett dict med KÄLLNYCKLAR som vår orkestrator förstår:
- sharesOutstanding            (CommonStockSharesOutstanding, instant)
- cash                         (CashAndCashEquivalents*, instant)
- grossMargin (%)             = GrossProfitTTM / RevenueTTM * 100
- operatingMargin (%)         = OperatingIncomeTTM / RevenueTTM * 100
- netMargin (%)               = NetIncomeTTM / RevenueTTM * 100
- roe (%)                     = NetIncomeTTM / Avg(Equity Instants) * 100  (fallback: senaste equity)
- debtToEquity                = (ShortTermDebt + CurrentPortionLTD + LongTermDebt) / StockholdersEquity
- netDebtToEbitda             = (TotalDebt - Cash) / EBITDA_TTM

Dessutom returneras underlag (om tillgängligt):
- revenueTTM, grossProfitTTM, operatingIncomeTTM, netIncomeTTM
- ebitdaTTM, equity (instant), totalDebt (instant)

OBS:
- SEC tillhandahåller inte pris/marketCap → lämnas tomt här (fylls av Yahoo/FMP).
- Valuta antas USD (SEC-filings). 'currency' sätts till "USD" om något returneras.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import math
import time
import json

import requests
import streamlit as st


# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
_SEC_UA = st.secrets.get("SEC_USER_AGENT", "contact@example.com")
_SEC_BASE = "https://data.sec.gov"
_TICKERS_INDEX = "https://www.sec.gov/files/company_tickers.json"

_HEADERS = {
    "User-Agent": _SEC_UA,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
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

def _pick_unit(units: Dict[str, list], prefer: str = "USD") -> Optional[List[dict]]:
    """
    Välj en lista av datapunkter för given valuta ('USD' eller liknande).
    Faller tillbaka på första nyckeln om USD saknas.
    """
    if not isinstance(units, dict):
        return None
    if prefer in units:
        return units.get(prefer) or None
    # fallback: ta första nyckeln
    for k, v in units.items():
        if isinstance(v, list):
            return v
    return None

def _sort_by_end_desc(points: List[dict]) -> List[dict]:
    """
    Sortera datapunkter på 'end' (YYYY-MM-DD) fallande, med fallback på 'fy'/'fp'.
    """
    def _key(p: dict):
        end = p.get("end") or ""
        # sorts by end; None sist
        return (0 if end else 1, end)
    return sorted(points, key=_key, reverse=True)

def _latest_instant(points: List[dict]) -> Optional[dict]:
    """
    Plocka senaste "instant" (qtrs == 0) – helst från 10-Q/10-K.
    """
    if not points:
        return None
    # sortera
    pts = _sort_by_end_desc(points)
    for p in pts:
        if p.get("qtrs", 0) == 0:
            return p
    # fallback: ta första
    return pts[0]

def _latest_quarters(points: List[dict], n: int = 4) -> List[dict]:
    """
    Hämta senaste N kvartalsvärden (qtrs == 1). Om färre finns – returnera färre.
    """
    out: List[dict] = []
    if not points:
        return out
    pts = _sort_by_end_desc(points)
    for p in pts:
        if int(p.get("qtrs", 0)) == 1:
            out.append(p)
            if len(out) >= n:
                break
    return out

@st.cache_data(ttl=86400, show_spinner=False)
def _load_ticker_index() -> Dict[str, str]:
    """
    Hämtar SEC:s ticker-index → {TICKER: CIK-10-digits-str}.
    Cache i 24h.
    """
    try:
        r = requests.get(_TICKERS_INDEX, headers={"User-Agent": _SEC_UA}, timeout=20)
        if r.status_code != 200:
            return {}
        j = r.json()
        # Strukturen: { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
        out: Dict[str, str] = {}
        for _, row in j.items():
            t = str(row.get("ticker") or "").upper()
            cik_num = int(row.get("cik_str") or 0)
            if t and cik_num:
                out[t] = f"{cik_num:010d}"
        return out
    except Exception:
        return {}

def _cik_for_ticker(ticker: str) -> Optional[str]:
    t = str(ticker or "").upper().strip()
    if not t:
        return None
    idx = _load_ticker_index()
    return idx.get(t)

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_company_facts(cik10: str) -> Optional[dict]:
    """
    Läser companyfacts JSON från SEC.
    Cache i 1h för att inte slå i rate limits.
    """
    try:
        url = f"{_SEC_BASE}/api/xbrl/companyfacts/CIK{cik10}.json"
        r = requests.get(url, headers=_HEADERS, timeout=25)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def _find_fact_units(cf: dict, taxonomy: str) -> Optional[Dict[str, list]]:
    """Navigerar cf['facts']['us-gaap'][taxonomy]['units'] säkert."""
    try:
        return cf.get("facts", {}).get("us-gaap", {}).get(taxonomy, {}).get("units", None)
    except Exception:
        return None

def _sum_last4(points: List[dict]) -> Optional[float]:
    if not points:
        return None
    vals = [_to_float(p.get("val")) for p in points if p.get("val") is not None]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals))

def _avg_last2_inst(points: List[dict]) -> Optional[float]:
    if not points:
        return None
    # ta 2 senaste instanter
    inst = [p for p in _sort_by_end_desc(points) if int(p.get("qtrs", 0)) == 0]
    vals = [_to_float(p.get("val")) for p in inst[:2]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))

# ------------------------------------------------------------
# Huvud: hämta & härled nyckeltal
# ------------------------------------------------------------
def get_all_fields(ticker: str) -> Dict[str, Any]:
    """
    Returnerar ett dict med SEC-härledda fält:
      sharesOutstanding, cash, grossMargin, operatingMargin, netMargin,
      roe, debtToEquity, netDebtToEbitda
    + underlag: revenueTTM, grossProfitTTM, operatingIncomeTTM, netIncomeTTM, ebitdaTTM,
                equity, totalDebt
    """
    out: Dict[str, Any] = {}
    tkr = str(ticker or "").upper().strip()
    if not tkr:
        return out

    cik = _cik_for_ticker(tkr)
    if not cik:
        return out

    cf = _fetch_company_facts(cik)
    if not cf:
        return out

    # 0) Grundantagande: USD
    out["currency"] = "USD"

    # 1) Shares outstanding (instant)
    units = _find_fact_units(cf, "CommonStockSharesOutstanding")
    if units:
        pts = _pick_unit(units, "USD") or _pick_unit(units)  # värdet är "shares" men indexeras ibland ändå
        p = _latest_instant(pts or [])
        v = _to_float(p.get("val")) if p else None
        if v and v > 0:
            out["sharesOutstanding"] = v  # styck

    # 2) Cash (instant) – CashAndCashEquivalents* eller inkl. restricted
    cash_units = _find_fact_units(cf, "CashAndCashEquivalentsAtCarryingValue")
    if not cash_units:
        cash_units = _find_fact_units(cf, "CashAndCashEquivalentsIncludingRestrictedCash")
    if cash_units:
        pts = _pick_unit(cash_units, "USD") or _pick_unit(cash_units)
        p = _latest_instant(pts or [])
        cv = _to_float(p.get("val")) if p else None
        if cv is not None:
            out["cash"] = cv

    # 3) Total debt (instant) = LTD + ShortTermBorrowings + CurrentPortionLTD (så gott det går)
    def _inst_val(tax: str) -> Optional[float]:
        u = _find_fact_units(cf, tax)
        if not u:
            return None
        pts = _pick_unit(u, "USD") or _pick_unit(u)
        p = _latest_instant(pts or [])
        return _to_float(p.get("val")) if p else None

    ltd = _inst_val("LongTermDebtNoncurrent") or _inst_val("LongTermDebt")
    std = _inst_val("ShortTermBorrowings")
    cur_ltd = _inst_val("LongTermDebtCurrent")
    total_debt = 0.0
    has_any_debt = False
    for part in (ltd, std, cur_ltd):
        if part is not None:
            total_debt += float(part)
            has_any_debt = True
    if has_any_debt:
        out["totalDebt"] = total_debt

    # 4) Equity (instant)
    eq = _inst_val("StockholdersEquity") or _inst_val("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    if eq is not None:
        out["equity"] = eq

    # 5) Kvartalsserier (4 st): revenue, grossProfit, operatingIncome, netIncome, depreciationAmortization
    def _q4_sum(tax: str) -> Optional[float]:
        u = _find_fact_units(cf, tax)
        if not u:
            return None
        pts = _pick_unit(u, "USD") or _pick_unit(u)
        q = _latest_quarters(pts or [], n=4)
        return _sum_last4(q)

    # Revenue (försök flera kandidat-taggar)
    rev = _q4_sum("Revenues") or _q4_sum("SalesRevenueNet") or _q4_sum("RevenueFromContractWithCustomerExcludingAssessedTax")
    if rev is not None:
        out["revenueTTM"] = rev

    gp = _q4_sum("GrossProfit")
    if gp is not None:
        out["grossProfitTTM"] = gp

    op = _q4_sum("OperatingIncomeLoss")
    if op is not None:
        out["operatingIncomeTTM"] = op

    ni = _q4_sum("NetIncomeLoss")
    if ni is not None:
        out["netIncomeTTM"] = ni

    da = _q4_sum("DepreciationAndAmortization")
    # EBITDA_TTM ≈ OperatingIncomeTTM + DepreciationAndAmortization_TTM
    ebitda_ttm = None
    if op is not None and da is not None:
        ebitda_ttm = op + da
        out["ebitdaTTM"] = ebitda_ttm

    # 6) Margins (TTM)
    if rev and rev > 0:
        if gp is not None:
            out["grossMargin"] = float(gp / rev * 100.0)
        if op is not None:
            out["operatingMargin"] = float(op / rev * 100.0)
        if ni is not None:
            out["netMargin"] = float(ni / rev * 100.0)

    # 7) ROE (TTM NetIncome / avg equity instants)
    # försök: ta equity-instant-serie och medelvärde av 2 senaste
    eq_units = _find_fact_units(cf, "StockholdersEquity") or _find_fact_units(cf, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    if ni is not None and eq_units:
        pts = _pick_unit(eq_units, "USD") or _pick_unit(eq_units)
        avg2 = _avg_last2_inst(pts or [])
        base_eq = avg2 if avg2 and avg2 > 0 else _to_float((_latest_instant(pts or []) or {}).get("val"))
        if base_eq and base_eq > 0:
            out["roe"] = float(ni / base_eq * 100.0)

    # 8) Debt/Equity
    if "equity" in out and out["equity"] and out["equity"] > 0 and has_any_debt:
        out["debtToEquity"] = float(total_debt / out["equity"])

    # 9) NetDebt/EBITDA
    if has_any_debt and "cash" in out and out["cash"] is not None and ebitda_ttm and ebitda_ttm > 0:
        net_debt = float(total_debt - float(out["cash"] or 0.0))
        out["netDebtToEbitda"] = float(net_debt / ebitda_ttm)

    # Rensa orimliga procent (ibland enorma värden vid små nämnare)
    for pk in ("grossMargin", "operatingMargin", "netMargin", "roe"):
        if pk in out:
            v = _to_float(out[pk])
            if v is None:
                del out[pk]
            else:
                # klipp extrema värden till rimligt spann (-200..200)
                out[pk] = max(-200.0, min(200.0, float(v)))

    # Returnera endast fält med värde
    clean: Dict[str, Any] = {}
    for k, v in out.items():
        if v is None:
            continue
        clean[k] = v

    return clean
