# -*- coding: utf-8 -*-
"""
Yahoo Finance-fetcher

Publik:
    fetch_yahoo(ticker: str) -> (data: dict, status_code: int, source: str)

Returnerar ett data-dict med nycklar som matchar vårt datablad:
    - "Namn", "Sektor", "Bransch"
    - "Senaste kurs", "Market Cap", "Utestående aktier (milj.)"
    - "P/S (Yahoo)", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"

P/S-fallbacks:
    1) Pris/sales från Yahoo (defaultKeyStatistics.priceToSalesTrailing12Months
       eller summaryDetail.priceToSales)
    2) Annars räknas P/S = MarketCap / TTM-omsättning (senaste 4 kvartal)
       där TTM fås via incomeStatementHistoryQuarterly.

Notera:
    - Vi nyttjar Yahoo quoteSummary-endpoints direkt via requests
      (ingen yfinance-beroende -> mindre problem i hostade miljöer).
    - Vi försöker flera moduler i samma anrop för att minska latency.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import requests
from datetime import datetime

YA_BASE = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"

# Försök få flera moduler i ett skott – minskar roundtrips.
YA_MODULES = ",".join([
    "price",
    "summaryProfile",
    "assetProfile",            # vissa tickers använder assetProfile istället för summaryProfile
    "defaultKeyStatistics",
    "financialData",
    "summaryDetail",
    "incomeStatementHistoryQuarterly",
])

TIMEOUT = 15


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def _get(url: str, params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], int]:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        sc = r.status_code
        if sc != 200:
            return None, sc
        j = r.json()
        return j, sc
    except Exception:
        return None, 599


def _first_result(j: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return (j or {}).get("quoteSummary", {}).get("result", [{}])[0] or {}
    except Exception:
        return {}


def _get_profile(block: Dict[str, Any]) -> Dict[str, Any]:
    """
    Yahoo använder olika nycklar beroende på ticker:
      - 'summaryProfile' och/eller 'assetProfile'
    """
    prof = block.get("summaryProfile") or block.get("assetProfile") or {}
    # Fält: 'sector', 'industry'
    return {
        "Sektor": prof.get("sector"),
        "Bransch": prof.get("industry"),
    }


def _get_price(block: Dict[str, Any]) -> Dict[str, Any]:
    price = block.get("price") or {}
    # Nycklar: regularMarketPrice, currency, longName/shortName, marketCap
    p = (price.get("regularMarketPrice") or {}).get("raw")
    mcap = (price.get("marketCap") or {}).get("raw")
    name = price.get("longName") or price.get("shortName")
    curr = price.get("currency")
    return {
        "Senaste kurs": float(p) if p is not None else None,
        "Market Cap": float(mcap) if mcap is not None else None,
        "Valuta": curr,
        "Namn": name,
    }


def _get_key_stats(block: Dict[str, Any]) -> Dict[str, Any]:
    ks = block.get("defaultKeyStatistics") or {}
    shares = (ks.get("sharesOutstanding") or {}).get("raw")
    ps_ttm = (ks.get("priceToSalesTrailing12Months") or {}).get("raw")
    pb = (ks.get("priceToBook") or {}).get("raw")
    return {
        "Utestående aktier (milj.)": (float(shares) / 1e6) if shares else None,
        "P/S (Yahoo)": float(ps_ttm) if ps_ttm is not None else None,
        "P/B": float(pb) if pb is not None else None,
    }


def _get_summary_detail(block: Dict[str, Any]) -> Dict[str, Any]:
    sd = block.get("summaryDetail") or {}
    # ibland finns P/S även här
    ps = (sd.get("priceToSales") or {}).get("raw")
    dy = (sd.get("dividendYield") or {}).get("raw")
    return {
        "P/S (Yahoo) (alt)": float(ps) if ps is not None else None,
        "Dividend yield (%)": float(dy) * 100.0 if dy is not None else None,
    }


def _get_financial(block: Dict[str, Any]) -> Dict[str, Any]:
    fd = block.get("financialData") or {}
    # 'totalRevenue' är ofta TTM, ibland 'revenue'/'trailingAnnualRevenue'
    ttm_rev = (fd.get("totalRevenue") or {}).get("raw")
    if ttm_rev is None:
        ttm_rev = (fd.get("revenue") or {}).get("raw")
    if ttm_rev is None:
        ttm_rev = (fd.get("trailingAnnualRevenue") or {}).get("raw")

    gross = (fd.get("grossMargins") or {}).get("raw")
    opm = (fd.get("operatingMargins") or {}).get("raw")
    npm = (fd.get("profitMargins") or {}).get("raw")
    de = (fd.get("debtToEquity") or {}).get("raw")
    pb = (fd.get("priceToBook") or {}).get("raw")

    # safe -> procent
    return {
        "_TTM_revenue_guess": float(ttm_rev) if ttm_rev is not None else None,
        "Bruttomarginal (%)": float(gross) * 100.0 if gross is not None else None,
        "Operating margin (%)": float(opm) * 100.0 if opm is not None else None,
        "Net margin (%)": float(npm) * 100.0 if npm is not None else None,
        "Debt/Equity": float(de) if de is not None else None,
        "P/B": float(pb) if pb is not None else None,
    }


def _get_quarter_revenues(block: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Hämtar kvartalsvisa totalRevenue från Yahoo (om finns).
    Returnerar lista [(date_iso, revenue_float), ...] sorterad nyast->äldst
    """
    out: List[Tuple[str, float]] = []
    hist = (block.get("incomeStatementHistoryQuarterly") or {}).get("incomeStatementHistory") or []
    for q in hist:
        end = (q.get("endDate") or {}).get("fmt")  # '2025-01-26' etc
        rev = (q.get("totalRevenue") or {}).get("raw")
        if end and rev is not None:
            try:
                # verifiera datumformat
                datetime.fromisoformat(end)
                out.append((end, float(rev)))
            except Exception:
                pass
    # Yahoo returnerar oftast nyast först; vi säkerställer sorteringen.
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _rolling_ttm_ps(mcap: Optional[float], q_revs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Beräknar P/S för de senaste 4 TTM-fönstren baserat på kvartalsintäkter
    och ETABLERADE Market Cap (nuvarande) som approximation.
    Returnerar [(period_end_iso, ps_value), ...] nyast->äldst, upp till 4 värden.
    """
    out: List[Tuple[str, float]] = []
    if not mcap or mcap <= 0 or not q_revs:
        return out

    # Bygg rullande 4-kvartals-summor
    # q_revs är nyast->äldst: [ (Qn, val), (Qn-1, val), ... ]
    vals = [v for _, v in q_revs]
    dates = [d for d, _ in q_revs]

    for i in range(0, min(4, len(vals))):
        # fönster från i .. i+3
        if i + 4 <= len(vals):
            ttm = sum(vals[i:i+4])
            if ttm and ttm > 0:
                ps = float(mcap) / float(ttm)
                out.append((dates[i], ps))
    return out  # högst 4 poster


# ------------------------------------------------------------
# Publik fetch
# ------------------------------------------------------------
def fetch_yahoo(ticker: str) -> Tuple[Dict[str, Any], int, str]:
    """
    Huvudfunktion. Försöker hämta all nödvändig data från Yahoo.
    Returnerar (data_dict, status_code, "Yahoo").

    data_dict innehåller BARA nycklar som vår databas förstår.
    """
    url = f"{YA_BASE}/{ticker}"
    j, sc = _get(url, {"modules": YA_MODULES})
    if not j or sc != 200:
        return {}, sc, "Yahoo"

    block = _first_result(j)

    info: Dict[str, Any] = {}
    info.update(_get_profile(block))        # Sektor, Bransch
    info.update(_get_price(block))          # Namn, Senaste kurs, Market Cap, Valuta
    info.update(_get_key_stats(block))      # Utestående aktier, P/S (Yahoo), P/B
    info.update(_get_summary_detail(block)) # ev. alternativ P/S + Dividend yield
    info.update(_get_financial(block))      # TTM revenue (guess) + marginaler + Debt/Equity

    # Konsolidera P/S (Yahoo)
    ps_yahoo = info.get("P/S (Yahoo)")
    if ps_yahoo is None:
        alt_ps = info.get("P/S (Yahoo) (alt)")
        if alt_ps is not None:
            info["P/S (Yahoo)"] = float(alt_ps)

    # Fallback P/S (TTM) = MCAP / TTM-omsättning
    if info.get("P/S (Yahoo)") is None:
        mcap = info.get("Market Cap")
        ttm_rev = info.get("_TTM_revenue_guess")
        if mcap and ttm_rev and ttm_rev > 0:
            info["P/S (Yahoo)"] = float(mcap) / float(ttm_rev)

    # P/S (nu, beräknat) – vi sätter samma som Yahoo om finns, annars fallback
    info["P/S"] = float(info.get("P/S (Yahoo)")) if info.get("P/S (Yahoo)") is not None else None

    # P/S historik Q1..Q4 via kvartalsintäkter (rullande TTM-fönster)
    q_revs = _get_quarter_revenues(block)
    ps_hist = _rolling_ttm_ps(info.get("Market Cap"), q_revs)

    # ps_hist är [(date, ps), ...] nyast->; vi mappas till Q1..Q4
    # där Q1 = nyaste
    for idx, label in enumerate(["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]):
        try:
            info[label] = float(ps_hist[idx][1])
        except Exception:
            info[label] = None

    # Rensa bort interna nycklar
    info.pop("_TTM_revenue_guess", None)
    info.pop("P/S (Yahoo) (alt)", None)
    info.pop("Valuta", None)  # valutan används inte som kolumn i vårt schema

    return info, 200, "Yahoo"
