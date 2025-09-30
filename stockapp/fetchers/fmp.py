# -*- coding: utf-8 -*-
"""
FinancialModelingPrep (FMP) fetcher
Returnerar (data, facts, log) för en ticker.
- data  : dict med nyckeltal (svenska fältnamn som i appen)
- facts : dict med {fält: "FMP/<endpoint>"} för spårbarhet
- log   : list[str] med händelser
Kräver st.secrets["FMP_API_KEY"].
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import time
import requests
import streamlit as st

FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")

def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20):
    if not params:
        params = {}
    params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    js = r.json()
    return js if js is not None else {}

def _safe_float(x) -> Optional[float]:
    try:
        if x in (None, "", "NaN"):
            return None
        return float(x)
    except Exception:
        return None

def fetch_fmp(ticker: str) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    """
    Huvudfunktion. Hämtar profil/quote/ttm-mått/balans/kassaflöde m.m.
    """
    data: Dict[str, Any]  = {}
    facts: Dict[str, str] = {}
    log:  List[str]       = []

    if not FMP_KEY:
        log.append("FMP: Ingen API-nyckel – hoppar över.")
        return data, facts, log

    t0 = time.time()

    # --- Profile (bolagsnamn, sektor/industri, utdelningsdetaljer mm)
    try:
        prof = _get(f"/api/v3/profile/{ticker}")
        if isinstance(prof, list) and prof:
            p = prof[0]
            if p.get("companyName"):
                data["Namn"] = p["companyName"]; facts["Namn"] = "FMP/profile"
            if p.get("sector"):
                data["Sektor"] = p["sector"];     facts["Sektor"] = "FMP/profile"
            if p.get("industry"):
                data["Bransch"] = p["industry"];   facts["Bransch"] = "FMP/profile"
            # Trailing dividend yield (om tillgänglig)
            dy = _safe_float(p.get("lastDiv"))
            px = _safe_float(p.get("price"))
            if dy is not None and px and px > 0:
                # grov annualiserad yield om lastDiv är kvartalsutdelning
                data["Dividend yield (%)"] = float(dy * 4.0 / px * 100.0)
                facts["Dividend yield (%)"] = "FMP/profile(lastDiv)"
    except Exception as e:
        log.append(f"FMP profile: {e}")

    # --- Quote (pris, market cap, antal aktier)
    try:
        q = _get(f"/api/v3/quote/{ticker}")
        if isinstance(q, list) and q:
            q0 = q[0]
            mcap  = _safe_float(q0.get("marketCap"))
            price = _safe_float(q0.get("price"))
            shares = _safe_float(q0.get("sharesOutstanding"))

            if mcap is not None:
                data["Market Cap"] = float(mcap)
                facts["Market Cap"] = "FMP/quote"
            if shares is not None:
                data["Utestående aktier (milj.)"] = float(shares) / 1e6
                facts["Utestående aktier (milj.)"] = "FMP/quote"
            if price is not None:
                data["Senaste kurs"] = float(price)
                facts["Senaste kurs"] = "FMP/quote"
    except Exception as e:
        log.append(f"FMP quote: {e}")

    # --- Key metrics TTM (P/B, FCF yield, ROE, mm)
    try:
        km = _get(f"/api/v3/key-metrics-ttm/{ticker}")
        if isinstance(km, list) and km:
            k0 = km[0]
            pb = _safe_float(k0.get("pbRatioTTM")) or _safe_float(k0.get("priceToBookRatioTTM"))
            if pb is not None:
                data["P/B"] = float(pb); facts["P/B"] = "FMP/key-metrics-ttm"

            roe = _safe_float(k0.get("roeTTM")) or _safe_float(k0.get("returnOnEquityTTM"))
            if roe is not None:
                data["ROE (%)"] = float(roe) * 100.0; facts["ROE (%)"] = "FMP/key-metrics-ttm"

            fcf_ttm = _safe_float(k0.get("freeCashFlowTTM"))
            if fcf_ttm is not None:
                data["FCF (TTM)"] = float(fcf_ttm); facts["FCF (TTM)"] = "FMP/key-metrics-ttm"
                if data.get("Market Cap") and data["Market Cap"] > 0:
                    data["FCF Yield (%)"] = float(fcf_ttm) / float(data["Market Cap"]) * 100.0
                    facts["FCF Yield (%)"] = "FMP/key-metrics-ttm"
    except Exception as e:
        log.append(f"FMP key-metrics-ttm: {e}")

    # --- Ratios TTM (marginaler mm)
    try:
        rr = _get(f"/api/v3/ratios-ttm/{ticker}")
        if isinstance(rr, list) and rr:
            r0 = rr[0]
            gm = _safe_float(r0.get("grossProfitMarginTTM"))
            om = _safe_float(r0.get("operatingProfitMarginTTM"))
            nm = _safe_float(r0.get("netProfitMarginTTM"))
            if gm is not None:
                data["Bruttomarginal (%)"] = float(gm) * 100.0; facts["Bruttomarginal (%)"] = "FMP/ratios-ttm"
            if om is not None:
                data["Operating margin (%)"] = float(om) * 100.0; facts["Operating margin (%)"] = "FMP/ratios-ttm"
            if nm is not None:
                data["Net margin (%)"] = float(nm) * 100.0; facts["Net margin (%)"] = "FMP/ratios-ttm"
    except Exception as e:
        log.append(f"FMP ratios-ttm: {e}")

    # --- Income statement TTM (EBITDA, för nettoskuld/EBITDA)
    ebitda_ttm: Optional[float] = None
    try:
        ist = _get(f"/api/v3/income-statement-ttm/{ticker}")
        if isinstance(ist, list) and ist:
            ebitda_ttm = _safe_float(ist[0].get("ebitdaTTM"))
    except Exception as e:
        log.append(f"FMP income-statement-ttm: {e}")

    # --- Balance sheet (skuld, kassa, equity) och Debt/Equity
    total_debt = None
    cash_eq = None
    equity = None
    try:
        bs = _get(f"/api/v3/balance-sheet-statement/{ticker}", params={"limit": 1, "period": "quarter"})
        if isinstance(bs, list) and bs:
            b0 = bs[0]
            total_debt = _safe_float(b0.get("totalDebt")) or _safe_float(b0.get("shortLongTermDebtTotal"))
            cash_eq    = _safe_float(b0.get("cashAndCashEquivalents"))
            equity     = _safe_float(b0.get("totalStockholdersEquity")) or _safe_float(b0.get("totalEquity"))
            if cash_eq is not None:
                data["Kassa"] = float(cash_eq); facts["Kassa"] = "FMP/balance-sheet"
            if equity and total_debt is not None and equity != 0:
                data["Debt/Equity"] = float(total_debt) / float(equity)
                facts["Debt/Equity"] = "FMP/balance-sheet"
    except Exception as e:
        log.append(f"FMP balance-sheet: {e}")

    # --- Net debt/EBITDA
    try:
        if total_debt is not None and ebitda_ttm and ebitda_ttm != 0:
            net_debt = float(total_debt) - float(cash_eq or 0.0)
            data["Net debt / EBITDA"] = float(net_debt) / float(ebitda_ttm)
            facts["Net debt / EBITDA"] = "FMP(balance+income-ttm)"
    except Exception as e:
        log.append(f"FMP ND/EBITDA: {e}")

    # --- Dividend payout vs FCF (grovt): summan utdelningar TTM / FCF TTM
    try:
        divs = _get(f"/api/v3/historical-price-full/stock_dividend/{ticker}", params={"serietype": "line"})
        ttm_div = 0.0
        if isinstance(divs, dict):
            hist = divs.get("historical", []) or []
            # summera ~senaste 4 st (om kvartalsutdelare)
            for d in hist[:4]:
                v = _safe_float(d.get("dividend"))
                if v:
                    ttm_div += float(v)
        # multiplicera per aktie med antal aktier ≈ market cap / price
        price = data.get("Senaste kurs")
        mcap  = data.get("Market Cap")
        fcf   = data.get("FCF (TTM)")
        if price and mcap and fcf and fcf != 0 and ttm_div > 0:
            approx_shares = float(mcap) / float(price)
            total_div_cash = ttm_div * approx_shares
            data["Dividend payout (FCF) (%)"] = float(total_div_cash) / float(fcf) * 100.0
            facts["Dividend payout (FCF) (%)"] = "FMP/stock_dividend + quote + FCF"
    except Exception as e:
        log.append(f"FMP payout(FCF) beräkning: {e}")

    log.append(f"FMP klart på {time.time() - t0:.2f}s")
    return data, facts, log
