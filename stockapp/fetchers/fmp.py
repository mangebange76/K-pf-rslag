# -*- coding: utf-8 -*-
"""
stockapp.fetchers.fmp
---------------------
FinancialModelingPrep (FMP) fetcher.
Kräver FMP_API_KEY i st.secrets. (Ingen extern modul – bara requests.)

Publikt API:
- get_all(ticker) -> Dict[str, Any]
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import requests
import streamlit as st

def _fmp_base() -> str:
    # Låt användaren överskriva bas-URL via secrets, annars officiell
    return st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")

def _fmp_key() -> str:
    return st.secrets.get("FMP_API_KEY", "")

def _get_json(path: str, params: Dict[str, Any] | None = None, timeout: float = 12.0) -> Any:
    key = _fmp_key()
    if not key:
        return {"__error__": "Missing FMP_API_KEY"}
    base = _fmp_base().rstrip("/")
    url = f"{base}{path}"
    p = dict(params or {})
    p["apikey"] = key
    try:
        r = requests.get(url, params=p, timeout=timeout)
        if r.status_code != 200:
            return {"__error__": f"HTTP {r.status_code}"}
        return r.json()
    except Exception as e:
        return {"__error__": str(e)}

def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _as_pct(val) -> Optional[float]:
    v = _safe_float(val, None)
    if v is None:
        return None
    return round(v * 100.0, 2)

def _fmt_millions(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x) / 1e6
    except Exception:
        return None

def get_all(ticker: str) -> Dict[str, Any]:
    """
    Hämtar ett rimligt paket nyckeltal från FMP.
    Returnerar ett dict med svenska kolumnnamn när möjligt.
    """
    out: Dict[str, Any] = {}
    logs: List[str] = []

    # 1) Profile – sektor, industri, valuta, mcap, price, shares
    prof = _get_json(f"/api/v3/profile/{ticker}")
    if isinstance(prof, list) and prof:
        p0 = prof[0] or {}
        sector = p0.get("sector")
        if sector:
            out["Sektor"] = str(sector)
        industry = p0.get("industry")
        if industry:
            out["Industri"] = str(industry)

        mcap = _safe_float(p0.get("mktCap"), None)
        if mcap is not None:
            out["Market Cap"] = mcap

        price = _safe_float(p0.get("price"), None)
        if price is not None:
            out["Kurs"] = price

        ccy = p0.get("currency")
        if ccy:
            out["Valuta"] = str(ccy)

        shares = _safe_float(p0.get("sharesOutstanding"), None)
        if shares:
            out["Utestående aktier (milj.)"] = _fmt_millions(shares)

        dy = _safe_float(p0.get("lastDiv"), None)
        if dy and price and price > 0:
            # Om lastDiv är "annual dividend per share", kan vi grovt beräkna yield
            out["Dividend yield (%)"] = round((dy / price) * 100.0, 2)
    elif isinstance(prof, dict) and prof.get("__error__"):
        logs.append(f"profile: {prof['__error__']}")

    # 2) Ratios TTM – marginaler, ROE, P/B, DE, EV/EBITDA, PS, DY TTM
    ratios = _get_json(f"/api/v3/ratios-ttm/{ticker}")
    if isinstance(ratios, list) and ratios:
        r0 = ratios[0] or {}
        # Marginaler
        if r0.get("grossProfitMarginTTM") is not None:
            out["Gross margin (%)"] = round(float(r0["grossProfitMarginTTM"]) * 100.0, 2)
        if r0.get("operatingProfitMarginTTM") is not None:
            out["Operating margin (%)"] = round(float(r0["operatingProfitMarginTTM"]) * 100.0, 2)
        if r0.get("netProfitMarginTTM") is not None:
            out["Net margin (%)"] = round(float(r0["netProfitMarginTTM"]) * 100.0, 2)

        # Lönsamhet & multiplar
        if r0.get("returnOnEquityTTM") is not None:
            out["ROE (%)"] = round(float(r0["returnOnEquityTTM"]) * 100.0, 2)
        if r0.get("priceToBookRatioTTM") is not None:
            out["P/B"] = _safe_float(r0["priceToBookRatioTTM"], None)
        if r0.get("debtEquityRatioTTM") is not None:
            out["Debt/Equity"] = _safe_float(r0["debtEquityRatioTTM"], None)

        # EV/EBITDA
        ev_eb = r0.get("enterpriseValueOverEBITDATTM")
        if ev_eb is None:
            ev_eb = r0.get("evToEbitdaTTM")
        if ev_eb is not None:
            out["EV/EBITDA (ttm)"] = _safe_float(ev_eb, None)

        # P/S TTM
        ps = r0.get("priceToSalesRatioTTM")
        if ps is not None:
            out["P/S"] = _safe_float(ps, None)

        # Dividend yield TTM
        dy_ttm = r0.get("dividendYielTTM") or r0.get("dividendYieldTTM")
        if dy_ttm is not None:
            out["Dividend yield (%)"] = round(float(dy_ttm) * 100.0, 2)
    elif isinstance(ratios, dict) and ratios.get("__error__"):
        logs.append(f"ratios-ttm: {ratios['__error__']}")

    # 3) Cash flow (senaste kvartal) – FCF och utdelningspayout mot FCF
    cfs = _get_json(f"/api/v3/cash-flow-statement/{ticker}", params={"period": "quarter", "limit": 1})
    if isinstance(cfs, list) and cfs:
        c0 = cfs[0] or {}
        fcf = _safe_float(c0.get("freeCashFlow"), None)
        if fcf is not None:
            out["FCF (M)"] = _fmt_millions(fcf)
        div_paid = _safe_float(c0.get("dividendsPaid"), None)  # ofta negativt
        if div_paid is not None and fcf and fcf != 0:
            # Payout som andel av FCF (= utdelningar / FCF)
            payout = abs(div_paid) / abs(fcf) * 100.0
            out["Dividend payout (FCF) (%)"] = round(payout, 1)
    elif isinstance(cfs, dict) and cfs.get("__error__"):
        logs.append(f"cash-flow: {cfs['__error__']}")

    # 4) Balansräkning – kassa (senaste kvartal)
    bs = _get_json(f"/api/v3/balance-sheet-statement/{ticker}", params={"period": "quarter", "limit": 1})
    if isinstance(bs, list) and bs:
        b0 = bs[0] or {}
        cash = _safe_float(b0.get("cashAndCashEquivalents"), None)
        if cash is not None:
            out["Kassa (M)"] = _fmt_millions(cash)
    elif isinstance(bs, dict) and bs.get("__error__"):
        logs.append(f"balance-sheet: {bs['__error__']}")

    # 5) Income statement – ta 4 senaste quarter revenue för att approximera P/S Q1..Q4
    inc = _get_json(f"/api/v3/income-statement/{ticker}", params={"period": "quarter", "limit": 4})
    revs: List[float] = []
    if isinstance(inc, list) and inc:
        for it in inc:
            r = _safe_float(it.get("revenue"), None)
            if r is not None:
                revs.append(float(r))
        if revs:
            # Fyll P/S Q1..Q4 via mcap/revenue (approx), Q1 = senaste kvartalet
            m = out.get("Market Cap", None)
            if m:
                if len(revs) > 0 and revs[0] > 0:
                    out["P/S Q1"] = float(m) / float(revs[0])
                if len(revs) > 1 and revs[1] > 0:
                    out["P/S Q2"] = float(m) / float(revs[1])
                if len(revs) > 2 and revs[2] > 0:
                    out["P/S Q3"] = float(m) / float(revs[2])
                if len(revs) > 3 and revs[3] > 0:
                    out["P/S Q4"] = float(m) / float(revs[3])
    elif isinstance(inc, dict) and inc.get("__error__"):
        logs.append(f"income-statement: {inc['__error__']}")

    # 6) (valfritt) Market cap history – kan användas för risklabel etc. (hoppar nu)
    # mc_hist = _get_json(f"/api/v3/market-capitalization/{ticker}", params={"limit": 5})

    out["__fmp_fields__"] = len([k for k in out.keys() if not k.startswith("__")])
    if logs:
        out["__fmp_log__"] = "; ".join(logs)
    return out
