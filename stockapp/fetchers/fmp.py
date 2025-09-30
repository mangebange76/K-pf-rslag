# -*- coding: utf-8 -*-
"""
FinancialModelingPrep (FMP) fetcher

Publik:
    fetch_fmp(ticker: str) -> (data: dict, status_code: int, source: str)

Hämtar:
  - Namn, Sektor, Bransch
  - Senaste kurs, Market Cap, Utestående aktier (milj.)
  - P/S (beräknat) samt P/S Q1..Q4 via rullande TTM (4 kvartal)
  - Marginaler: Bruttomarginal, Operating margin, Net margin
  - Debt/Equity, P/B, Dividend yield (om tillgängligt)

Kräver:
  st.secrets["FMP_API_KEY"]
Valfritt:
  st.secrets["FMP_BASE"] (default: https://financialmodelingprep.com)
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import requests
import streamlit as st

TIMEOUT = 15


# ---------------------------- helpers ----------------------------
def _base() -> str:
    return st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")


def _apikey() -> str:
    return st.secrets.get("FMP_API_KEY", "")


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], int]:
    if params is None:
        params = {}
    params = dict(params)
    k = _apikey()
    if k:
        params["apikey"] = k
    url = f"{_base()}{path}"
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, r.status_code
        j = r.json()
        return j, r.status_code
    except Exception:
        return None, 599


def _take_first(lst: Any) -> Dict[str, Any]:
    if isinstance(lst, list) and lst:
        if isinstance(lst[0], dict):
            return lst[0]
    return {}


def _sum4(qvals: List[float], start: int) -> Optional[float]:
    seg = qvals[start:start+4]
    if len(seg) == 4 and all(x is not None for x in seg):
        return float(sum(seg))
    return None


# ---------------------------- core fetch ----------------------------
def fetch_fmp(ticker: str) -> Tuple[Dict[str, Any], int, str]:
    """
    Returnerar (data, status_code, "FMP").
    data är anpassad till vårt schema.
    """
    if not _apikey():
        # Svara tydligt att FMP ej aktivt – låt orchestrator falla vidare.
        return {}, 460, "FMP (saknar API-nyckel)"

    # 1) Profil + quote
    prof, sc1 = _get(f"/api/v3/profile/{ticker}")
    quote, sc2 = _get(f"/api/v3/quote/{ticker}")

    # 2) Kvartals-IS för TTM-bygge (8 st räcker)
    inc_q, sc3 = _get(f"/api/v3/income-statement/{ticker}", {"period": "quarter", "limit": 8})

    # 3) Ratios TTM (marginaler, D/E, P/B, Dividend yield)
    ratios_ttm, sc4 = _get(f"/api/v3/ratios-ttm/{ticker}", {"limit": 1})

    # Om ALLT failar -> ge upp
    if all(sc not in (200,) for sc in (sc1, sc2, sc3, sc4)):
        return {}, (sc1 or sc2 or sc3 or sc4 or 599), "FMP"

    info: Dict[str, Any] = {}

    # -------- profil/quote --------
    p = _take_first(prof)
    q = _take_first(quote)

    name = p.get("companyName") or p.get("companyNameShort") or p.get("symbol")
    sector = p.get("sector")
    industry = p.get("industry")

    price = q.get("price", p.get("price"))
    mcap = q.get("marketCap", p.get("mktCap"))
    shares = q.get("sharesOutstanding", p.get("sharesOutstanding"))

    info["Namn"] = name
    info["Sektor"] = sector
    info["Bransch"] = industry
    info["Senaste kurs"] = float(price) if isinstance(price, (int, float)) else None
    info["Market Cap"] = float(mcap) if isinstance(mcap, (int, float)) else None
    info["Utestående aktier (milj.)"] = (float(shares) / 1e6) if isinstance(shares, (int, float)) else None

    # -------- ratios (TTM) --------
    rt = _take_first(ratios_ttm)
    # marginaler i procent
    gm = rt.get("grossProfitMarginTTM")
    opm = rt.get("operatingProfitMarginTTM")
    npm = rt.get("netProfitMarginTTM")
    de = rt.get("debtEquityRatioTTM")
    pb = rt.get("priceToBookRatioTTM")
    dy = rt.get("dividendYieldTTM")  # ibland None för tillväxtbolag

    info["Bruttomarginal (%)"] = float(gm) * 100.0 if isinstance(gm, (int, float)) else None
    info["Operating margin (%)"] = float(opm) * 100.0 if isinstance(opm, (int, float)) else None
    info["Net margin (%)"] = float(npm) * 100.0 if isinstance(npm, (int, float)) else None
    info["Debt/Equity"] = float(de) if isinstance(de, (int, float)) else None
    info["P/B"] = float(pb) if isinstance(pb, (int, float)) else None
    info["Dividend yield (%)"] = float(dy) * 100.0 if isinstance(dy, (int, float)) else None

    # -------- P/S (TTM) + historik --------
    q_rows = inc_q if isinstance(inc_q, list) else []
    # FMP kvartalsfält är "revenue"
    q_pairs: List[Tuple[str, Optional[float]]] = []
    for row in q_rows:
        d = row.get("date")  # '2025-01-26' etc
        rev = row.get("revenue")
        if d:
            try:
                # validera datum
                datetime.fromisoformat(d)
                q_pairs.append((d, float(rev) if isinstance(rev, (int, float)) else None))
            except Exception:
                pass
    # nyast först
    q_pairs.sort(key=lambda x: x[0], reverse=True)

    q_vals = [v for _, v in q_pairs]
    q_dates = [d for d, _ in q_pairs]

    # TTM revenue (nyaste 4)
    ttm = _sum4(q_vals, 0)
    if ttm and info.get("Market Cap"):
        info["P/S"] = float(info["Market Cap"]) / float(ttm)
    else:
        info["P/S"] = None  # ingen tvingad fallback om vi inte har båda delarna
    # Yahoo-nyckeln finns inte här – lämna None
    info["P/S (Yahoo)"] = None

    # P/S Q1..Q4 via rullande fönster, baserat på NUVARANDE mcap (approx)
    for idx, label in enumerate(["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]):
        ps_val = None
        if info.get("Market Cap"):
            ttm_i = _sum4(q_vals, idx)
            if ttm_i and ttm_i > 0:
                ps_val = float(info["Market Cap"]) / float(ttm_i)
        info[label] = ps_val

    return info, 200, "FMP"
