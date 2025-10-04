# stockapp/fetchers/finviz.py
from __future__ import annotations

import re
from typing import Dict, Any, Tuple

import streamlit as st
import requests

try:
    from bs4 import BeautifulSoup  # rekommenderas
except Exception:
    BeautifulSoup = None  # type: ignore


FINVIZ_URL = "https://finviz.com/quote.ashx"
UA = st.secrets.get("FINVIZ_USER_AGENT") or (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def _num(s: str) -> float:
    """Konvertera Finviz-värden till float. Hanterar M/B/T, %, och '-'."""
    if s is None:
        return 0.0
    t = str(s).strip().replace(",", "")
    if not t or t == "-":
        return 0.0
    # procent
    if t.endswith("%"):
        try:
            return float(t[:-1])
        except Exception:
            return 0.0
    # stora tal med suffix
    m = re.match(r"^(-?\d+(\.\d+)?)([KMBT])$", t, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        suf = m.group(3).upper()
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[suf]
        return val * mult
    # vanliga floats
    try:
        return float(t)
    except Exception:
        return 0.0


def _pairs_from_html(html: str) -> Dict[str, str]:
    """
    Plocka ut nyckel:värde från 'snapshot-table2'.
    Vi försöker först med BeautifulSoup (stabilt), annars regex fallback.
    """
    out: Dict[str, str] = {}
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("table.snapshot-table2")
        if table:
            tds = table.find_all("td")
            # cellerna går i par: [KEY, VALUE, KEY, VALUE, ...]
            for i in range(0, len(tds) - 1, 2):
                key = tds[i].get_text(strip=True)
                val = tds[i + 1].get_text(strip=True)
                if key:
                    out[key] = val
    if not out:
        # enkel regex fallback (mindre robust men funkar ofta)
        # matcha td>KEY<td>VALUE i följd
        pattern = re.compile(
            r"<td[^>]*class=['\"]?snapshot-td2[^>]*>(?P<key>[^<]+)</td>\s*"
            r"<td[^>]*class=['\"]?snapshot-td2[^>]*>(?P<val>[^<]+)</td>",
            re.IGNORECASE,
        )
        for m in pattern.finditer(html):
            key = re.sub(r"\s+", " ", m.group("key").strip())
            val = re.sub(r"\s+", " ", m.group("val").strip())
            if key:
                out[key] = val
    return out


@st.cache_data(ttl=600, show_spinner=False)
def get_overview(ticker: str) -> Dict[str, Any]:
    """
    Hämtar och mappar centrala Finviz-metrics för ett ticker.
    Returnerar dict med nycklar som matchar appens schema där det är rimligt:
      price, ps_ttm, pb, dividend_yield_pct, payout_ratio_pct,
      market_cap, shares_outstanding, gross_margins_pct, operating_margins_pct, profit_margins_pct
    Obs: Finviz visar i USD.
    """
    tkr = ticker.upper().strip()
    params = {"t": tkr}
    headers = {"User-Agent": UA, "Referer": "https://finviz.com/"}
    r = requests.get(FINVIZ_URL, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text

    kv = _pairs_from_html(html)

    # Nycklar vi brukar hitta på finviz (snapshot):
    # Price, P/S, P/B, EPS (ttm), Dividends, Dividend %, Payout, Market Cap, Shs Outstand,
    # Gross Margin, Oper. Margin, Profit Margin
    price = _num(kv.get("Price", "0"))
    ps_ttm = _num(kv.get("P/S", "0"))
    pb = _num(kv.get("P/B", "0"))

    # Dividend yield (%)
    div_pct = _num(kv.get("Dividend %", "0"))

    # Payout ratio (%)
    payout = _num(kv.get("Payout", "0"))

    # Market cap och utestående aktier
    mcap = _num(kv.get("Market Cap", "0"))
    shs_out = _num(kv.get("Shs Outstand", "0"))

    # Marginaler (i %)
    gm = _num(kv.get("Gross Margin", "0"))
    om = _num(kv.get("Oper. Margin", "0"))
    pm = _num(kv.get("Profit Margin", "0"))

    return {
        "price": price,
        "ps_ttm": ps_ttm,
        "pb": pb,
        "dividend_yield_pct": div_pct,
        "payout_ratio_pct": payout,
        "market_cap": mcap,
        "shares_outstanding": shs_out,
        "gross_margins_pct": gm,
        "operating_margins_pct": om,
        "profit_margins_pct": pm,
        "source": "finviz",
    }
