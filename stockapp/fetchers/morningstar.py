# stockapp/fetchers/morningstar.py
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import requests
import streamlit as st

# ------------------------- Konfiguration -------------------------

UA = st.secrets.get("MORNINGSTAR_USER_AGENT") or (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

# Vanliga US-exchange-koder på Morningstar
DEFAULT_EXCHANGES = ("xnas", "xnys", "xase", "xngs")

MS_HOST = "https://www.morningstar.com"
QUOTE_PATH = "/stocks/{exch}/{ticker}/quote"

# Regex för __NEXT_DATA__-scriptet
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(?P<json>{.*?})</script>',
    re.DOTALL | re.IGNORECASE,
)


# ------------------------- Hjälpfunktioner ------------------------

def _safe_float(x: Any) -> float:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0


def _percentize(v: float) -> float:
    """Om v ser ut att vara 0–1, konvertera till %; annars lämna."""
    if v <= 0:
        return 0.0
    return v * 100.0 if v <= 1.0 else v


def _extract_next_data(html: str) -> Optional[Dict[str, Any]]:
    m = NEXT_DATA_RE.search(html or "")
    if not m:
        return None
    raw = m.group("json")
    try:
        return json.loads(raw)
    except Exception:
        return None


def _walk(d: Any) -> Iterable[Tuple[str, Any]]:
    """
    Platt nyckelgång genom ett djupt JSON-träd:
    ger (key, value) för alla löv och mellanliggande nycklar.
    """
    stack = [d]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                yield (str(k), v)
                if isinstance(v, (dict, list, tuple)):
                    stack.append(v)
        elif isinstance(cur, (list, tuple)):
            for v in cur:
                if isinstance(v, (dict, list, tuple)):
                    stack.append(v)


def _find_first_number(tree: Any, keys_like: Iterable[str]) -> float:
    """
    Letar igenom JSON-trädet och returnerar första siffervärdet vars
    närliggande 'key' matchar någon av substrings i keys_like.
    """
    keys_lc = [k.lower() for k in keys_like]
    for k, v in _walk(tree):
        kv = k.lower()
        if any(key in kv for key in keys_lc):
            num = _safe_float(v)
            if num != 0.0:
                return num
    return 0.0


def _first_of(tree: Any, key_candidates: Iterable[str]) -> float:
    return _find_first_number(tree, key_candidates)


def _fetch_html_for_exchanges(ticker: str, exchanges: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Prova flera exchange-koder tills vi får 200 + innehåll.
    Returnerar (html, exch) eller (None, None).
    """
    headers = {
        "User-Agent": UA,
        "Accept-Language": "en-US,en;q=0.8",
        "Referer": MS_HOST + "/",
    }
    sess = requests.Session()
    sess.headers.update(headers)

    for exch in exchanges:
        url = MS_HOST + QUOTE_PATH.format(exch=exch, ticker=ticker.lower())
        try:
            r = sess.get(url, timeout=20)
            if r.status_code == 200 and r.text and "__NEXT_DATA__" in r.text:
                return (r.text, exch)
        except Exception:
            pass
    return (None, None)


# ------------------------- Publik funktion -------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_overview(ticker: str, exchanges: Iterable[str] = DEFAULT_EXCHANGES) -> Dict[str, Any]:
    """
    Hämtar Morningstar quote-sida och plockar fram vanliga nyckeltal.
    Returnerar en dict som harmoniserar med vår app där möjligt:

      - price
      - market_cap
      - shares_outstanding
      - ps_ttm
      - pb
      - dividend_yield_pct
      - payout_ratio_pct
      - book_value_per_share
      - gross_margins_pct
      - operating_margins_pct
      - profit_margins_pct

    Obs:
      * Morningstar kan kräva inlogg/region ibland → då returneras partial/empty.
      * Alla fält är best effort (strukturen kan ändras).
    """
    tkr = ticker.upper().strip()

    html, exch = _fetch_html_for_exchanges(tkr, exchanges)
    if not html:
        return {"source": "morningstar", "exchange": exch or "", "error": "no_html"}

    data = _extract_next_data(html)
    if not data:
        return {"source": "morningstar", "exchange": exch or "", "error": "no_next_data"}

    # ----- Försök hitta värden i JSON-trädet -----
    # Pris (last/regular)
    price = 0.0
    for key_group in [
        ("lastPrice", "last", "regularMarketPrice", "price"),
        ("closePrice", "previousClose", "close"),
    ]:
        price = _first_of(data, key_group)
        if price > 0:
            break

    # Market cap och shares
    market_cap = _first_of(data, ("marketCap", "marketCapitalization"))
    shares_out = _first_of(data, ("sharesOutstanding", "shsOut", "shs_outstanding"))

    # P/S & P/B (TTM)
    ps_ttm = _first_of(data, ("priceToSalesRatioTTM", "psRatioTTM", "priceToSalesTTM", "psTTM"))
    pb = _first_of(data, ("priceToBookRatio", "pbRatio", "priceBook"))

    # Dividend yield (%) och payout (%)
    div_yield = _first_of(data, ("dividendYield", "trailingDividendYield"))
    payout = _first_of(data, ("payoutRatio", "dividendPayoutRatio"))

    # Book value / share
    bvps = _first_of(data, ("bookValuePerShare", "bookValuePerShareTTM"))

    # Marginaler
    gm = _first_of(data, ("grossMarginTTM", "grossMargin"))
    om = _first_of(data, ("operatingMarginTTM", "operatingMargin"))
    pm = _first_of(data, ("netMarginTTM", "netMargin", "profitMargin"))

    # Normalisera procent (Morningstar ger ofta i 0–1)
    div_yield_pct = _percentize(div_yield)
    payout_pct = _percentize(payout)
    gm_pct = _percentize(gm)
    om_pct = _percentize(om)
    pm_pct = _percentize(pm)

    return {
        "source": "morningstar",
        "exchange": exch or "",
        "price": price,
        "market_cap": market_cap,
        "shares_outstanding": shares_out,
        "ps_ttm": ps_ttm,
        "pb": pb,
        "dividend_yield_pct": div_yield_pct,
        "payout_ratio_pct": payout_pct,
        "book_value_per_share": bvps,
        "gross_margins_pct": gm_pct,
        "operating_margins_pct": om_pct,
        "profit_margins_pct": pm_pct,
    }
