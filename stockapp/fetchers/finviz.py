# stockapp/fetchers/finviz.py
from __future__ import annotations

import re
from typing import Dict, Any

import streamlit as st

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

FINVIZ_URL = "https://finviz.com/quote.ashx?t={ticker}"

UA = st.secrets.get("FINVIZ_USER_AGENT") or (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _parse_shorthand_num(s: str) -> float:
    """
    Parsar finviz-format: '123.4M', '1.2B', '789K' → absolut tal.
    Returnerar 0.0 om det inte går.
    """
    if not s:
        return 0.0
    s = s.strip().replace(",", "")
    m = re.match(r"^(-?\d+(?:\.\d+)?)([KMBT]?)$", s, flags=re.I)
    if not m:
        # ibland '—' eller 'N/A'
        return 0.0
    val = float(m.group(1))
    suf = m.group(2).upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(suf, 1.0)
    return val * mult

def _parse_percent(s: str) -> float:
    if not s:
        return 0.0
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except Exception:
        return 0.0

def _text(node) -> str:
    return (node.get_text(separator=" ", strip=True) if node is not None else "").strip()

@st.cache_data(ttl=600, show_spinner=False)
def get_overview(ticker: str) -> Dict[str, Any]:
    """
    Hämtar nyckeltal från finviz snapshot (om möjligt).
    Returnerar alltid en dict med nycklar appen förväntar sig; saknas värde → 0/"".
    Keys:
      price, ps_ttm, pb, market_cap, shares_outstanding,
      dividend_yield_pct, payout_ratio_pct,
      gross_margins_pct, operating_margins_pct, profit_margins_pct
    """
    out: Dict[str, Any] = {
        "price": 0.0,
        "ps_ttm": 0.0,
        "pb": 0.0,
        "market_cap": 0.0,
        "shares_outstanding": 0.0,
        "dividend_yield_pct": 0.0,
        "payout_ratio_pct": 0.0,
        "gross_margins_pct": 0.0,
        "operating_margins_pct": 0.0,
        "profit_margins_pct": 0.0,
    }

    if requests is None or BeautifulSoup is None:
        return out

    tkr = ticker.upper().strip()
    url = FINVIZ_URL.format(ticker=tkr)

    try:
        r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, timeout=30)
        r.raise_for_status()
    except Exception:
        return out

    try:
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception:
        return out

    # 1) Pris (header)
    try:
        # finviz har pris i en span med id "quote-price" ibland, annars i huvudtabellens "Price"
        price_span = soup.select_one("td.fullview-title b")
        # fallback: i snapshot-tabellen
        if not price_span:
            price_label = soup.find("td", string=re.compile(r"^Price$", flags=re.I))
            if price_label and price_label.find_next_sibling("td"):
                price_txt = _text(price_label.find_next_sibling("td"))
                out["price"] = _to_float(price_txt)
        else:
            price_txt = _text(price_span)
            out["price"] = _to_float(price_txt)
    except Exception:
        pass

    # 2) Snapshot-tabell: nyckel → värde (tabellen är två kolumner per rad: label | value)
    snapshot: Dict[str, str] = {}
    try:
        # snapshot ligger i tabellen med class "snapshot-table2"
        table = soup.select_one("table.snapshot-table2")
        if table:
            tds = table.find_all("td")
            # Gå i steg om 2: (label, value)
            for i in range(0, len(tds) - 1, 2):
                label = _text(tds[i])
                val = _text(tds[i + 1])
                if label:
                    snapshot[label] = val
    except Exception:
        pass

    # 3) Mappa relevanta fält
    # Market Cap
    out["market_cap"] = _parse_shorthand_num(snapshot.get("Market Cap", ""))

    # Shares Outstanding
    # OBS: Finviz "Shs Outstand" är ofta i M (miljoner), men ibland > 1000 → tolka via suffix.
    sh_out_raw = snapshot.get("Shs Outstand", "") or snapshot.get("Shs Outstand ", "")
    sh_out = _parse_shorthand_num(sh_out_raw)  # absolut tal
    out["shares_outstanding"] = sh_out

    # Multiplar
    out["ps_ttm"] = _to_float(snapshot.get("P/S", "").replace(",", ""))
    out["pb"]     = _to_float(snapshot.get("P/B", "").replace(",", ""))

    # Utdelning
    # Finviz anger Dividends → 'Dividend %' och 'Payout'
    out["dividend_yield_pct"] = _parse_percent(snapshot.get("Dividend %", "")) or _parse_percent(snapshot.get("Dividend % ", ""))
    # 'Payout' är i %, ibland '—'
    out["payout_ratio_pct"] = _parse_percent(snapshot.get("Payout", ""))

    # Marginaler
    out["gross_margins_pct"]     = _parse_percent(snapshot.get("Gross Margin", ""))
    out["operating_margins_pct"] = _parse_percent(snapshot.get("Operating Margin", ""))
    out["profit_margins_pct"]    = _parse_percent(snapshot.get("Profit Margin", ""))

    return out
