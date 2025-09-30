# -*- coding: utf-8 -*-
"""
stockapp/fetchers/fmp.py

FMP-hämtare:
- Namn, valuta, sektor, bransch, pris, market cap, shares via /profile och /quote
- P/S (TTM) via /ratios-ttm -> /key-metrics-ttm -> marketCap/revenueTTM
- P/S Q1..Q4 via /ratios?period=quarter&limit=4
- EV/EBITDA, Debt/Equity, bruttomarginal, nettomarginal om tillgängligt
- Returnerar (vals, debug); skriver inte manuella prognosfält

Kräver: st.secrets["FMP_API_KEY"], valfritt st.secrets["FMP_BASE"]
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import time
import requests
import streamlit as st

# ----------------------------- Hjälpare --------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default

def _fmp_base() -> str:
    return st.secrets.get("FMP_BASE", "https://financialmodelingprep.com").rstrip("/")

def _fmp_key() -> str:
    return st.secrets.get("FMP_API_KEY", "")

def _get(path: str, params: Optional[dict] = None, sleep: float = 0.0):
    """
    Enkel GET med API-nyckel och lite snäll throttling.
    Returnerar (json, status_code)
    """
    base = _fmp_base()
    url = f"{base}/{path.lstrip('/')}"
    q = dict(params or {})
    k = _fmp_key()
    if k:
        q["apikey"] = k
    if sleep > 0:
        time.sleep(sleep)
    try:
        r = requests.get(url, params=q, timeout=20)
        try:
            j = r.json()
        except Exception:
            j = None
        return j, r.status_code
    except Exception:
        return None, 0


# ----------------------------- Publikt API -----------------------------------

def fetch_fmp_combo(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Hämtar data för 'ticker' från FMP och bygger ut P/S (TTM) + P/S Q1..Q4.
    Returnerar (vals, debug). Sätter INTE manuella prognosfält.
    """
    sym = str(ticker).strip().upper()
    vals: Dict[str, Any] = {}
    dbg: Dict[str, Any] = {"ticker": sym, "source": "FMP"}

    # ---- Profile ----
    prof, sc_prof = _get(f"api/v3/profile/{sym}", sleep=0.5)
    dbg["profile_sc"] = sc_prof
    if isinstance(prof, list) and prof:
        p0 = prof[0] or {}
        if p0.get("companyName"): vals["Bolagsnamn"] = p0["companyName"]
        if p0.get("currency"):    vals["Valuta"]     = str(p0["currency"]).upper()
        if p0.get("sector"):      vals["Sektor"]     = p0["sector"]
        if p0.get("industry"):    vals["Bransch"]    = p0["industry"]
        if p0.get("price") is not None:
            vals["Aktuell kurs"] = _safe_float(p0.get("price"))
        if p0.get("mktCap") is not None:  # ibland heter det mktCap i profile
            vals["Market Cap"] = _safe_float(p0.get("mktCap"))
        if p0.get("marketCap") is not None:
            vals["Market Cap"] = _safe_float(p0.get("marketCap"))
        if p0.get("sharesOutstanding") is not None:
            try:
                vals["Utestående aktier"] = float(p0["sharesOutstanding"]) / 1e6
                dbg["_shares_source"] = "FMP profile sharesOutstanding"
            except Exception:
                pass

    # ---- Quote (pris + marketCap) ----
    quote, sc_quote = _get(f"api/v3/quote/{sym}", sleep=0.4)
    dbg["quote_sc"] = sc_quote
    if isinstance(quote, list) and quote:
        q0 = quote[0] or {}
        if "price" in q0:
            vals["Aktuell kurs"] = _safe_float(q0.get("price"), _safe_float(vals.get("Aktuell kurs"), 0.0))
        if q0.get("marketCap") is not None:
            vals["Market Cap"] = _safe_float(q0.get("marketCap"))

    # ---- Shares fallback: /v4/shares_float ----
    if "Utestående aktier" not in vals:
        flo, sc_flo = _get(f"api/v4/shares_float/{sym}", sleep=0.4)
        dbg["shares_float_sc"] = sc_flo
        if isinstance(flo, list):
            for it in flo:
                n = it.get("outstandingShares") or it.get("sharesOutstanding")
                if n:
                    try:
                        vals["Utestående aktier"] = float(n) / 1e6
                        dbg["_shares_source"] = "FMP v4 shares_float"
                        break
                    except Exception:
                        pass

    # ---- Ratios TTM (P/S, EV/EBITDA, marginaler, D/E) ----
    rttm, sc_rttm = _get(f"api/v3/ratios-ttm/{sym}", sleep=0.5)
    dbg["ratios_ttm_sc"] = sc_rttm
    if isinstance(rttm, list) and rttm:
        r0 = rttm[0] or {}
        ps_ttm = r0.get("priceToSalesRatioTTM") or r0.get("priceToSalesTTM")
        if ps_ttm:
            try:
                vals["P/S"] = float(ps_ttm)
                dbg["_ps_source"] = "ratios-ttm"
            except Exception:
                pass
        # EV/EBITDA
        ev_ebitda = r0.get("enterpriseValueOverEBITDA") or r0.get("evToEbitdaTTM")
        if ev_ebitda is not None:
            vals["EV/EBITDA"] = _safe_float(ev_ebitda, 0.0)
        # Marginaler (i proportioner) -> %
        gm = r0.get("grossProfitMarginTTM")
        if gm is not None:
            vals["Bruttomarginal (%)"] = _safe_float(gm, 0.0) * 100.0
        pm = r0.get("netProfitMarginTTM")
        if pm is not None:
            vals["Nettomarginal (%)"] = _safe_float(pm, 0.0) * 100.0
        # Debt/Equity
        de = r0.get("debtEquityRatioTTM") or r0.get("debtEquityRatio")
        if de is not None:
            vals["Debt/Equity"] = _safe_float(de, 0.0)

    # ---- Key-metrics TTM (fallback P/S/EV/EBITDA) ----
    if "P/S" not in vals:
        kttm, sc_kttm = _get(f"api/v3/key-metrics-ttm/{sym}", sleep=0.5)
        dbg["key_metrics_ttm_sc"] = sc_kttm
        if isinstance(kttm, list) and kttm:
            k0 = kttm[0] or {}
            ps = k0.get("priceToSalesRatioTTM") or k0.get("priceToSalesTTM")
            if ps:
                try:
                    vals["P/S"] = float(ps)
                    dbg["_ps_source"] = "key-metrics-ttm"
                except Exception:
                    pass
            if "EV/EBITDA" not in vals:
                ev_eb = k0.get("evToEbitdaTTM")
                if ev_eb is not None:
                    vals["EV/EBITDA"] = _safe_float(ev_eb, 0.0)

    # ---- P/S via marketCap/revenueTTM (fallback) ----
    mcap = _safe_float(vals.get("Market Cap"), 0.0)
    if "P/S" not in vals and mcap > 0:
        isttm, sc_isttm = _get(f"api/v3/income-statement-ttm/{sym}", sleep=0.4)
        dbg["income_stmt_ttm_sc"] = sc_isttm
        rev_ttm = 0.0
        if isinstance(isttm, list) and isttm:
            cand = isttm[0] or {}
            for k in ("revenueTTM", "revenue"):
                if cand.get(k) is not None:
                    rev_ttm = _safe_float(cand.get(k), 0.0)
                    if rev_ttm > 0:
                        break
        if rev_ttm > 0:
            vals["P/S"] = mcap / rev_ttm
            dbg["_ps_source"] = "calc(marketCap/revenueTTM)"

    # ---- P/S Q1..Q4 från ratios quarterly ----
    rq, sc_rq = _get(f"api/v3/ratios/{sym}", params={"period": "quarter", "limit": 4}, sleep=0.4)
    dbg["ratios_quarter_sc"] = sc_rq
    if isinstance(rq, list) and rq:
        for i, row in enumerate(rq[:4], start=1):
            psq = row.get("priceToSalesRatio")
            if psq is not None:
                try:
                    vals[f"P/S Q{i}"] = float(psq)
                except Exception:
                    pass

    # ---- Sista shares-säkring via implied (om saknas) ----
    if "Utestående aktier" not in vals:
        px = _safe_float(vals.get("Aktuell kurs"), 0.0)
        if mcap > 0 and px > 0:
            vals["Utestående aktier"] = (mcap / max(px, 1e-9)) / 1e6
            dbg["_shares_source"] = "implied(marketCap/price)"

    return vals, dbg
