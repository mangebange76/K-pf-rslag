# stockapp/fetchers/morningstar.py
from __future__ import annotations

import re
from typing import Dict, Any, Optional, Tuple

import streamlit as st

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore


# Vi provar Morningstars publika US-sidor (NASDAQ/NYSE/AMEX).
# Observera: Morningstar byter ibland DOM-struktur. Den här fetchern är
# "best-effort" och har robusta fallbacks (alla värden 0 om parsning misslyckas).
BASES = [
    "https://www.morningstar.com/stocks/xnas/{t}/quote",  # NASDAQ
    "https://www.morningstar.com/stocks/xnys/{t}/quote",  # NYSE
    "https://www.morningstar.com/stocks/xase/{t}/quote",  # AMEX
]

UA = st.secrets.get("MORNINGSTAR_USER_AGENT") or (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
TIMEOUT = 30


def _to_float(s) -> float:
    try:
        if s is None:
            return 0.0
        if isinstance(s, (int, float)):
            return float(s)
        txt = str(s).strip().replace(",", "")
        # ta bort procent och parenteser
        txt = txt.replace("%", "")
        # “—” eller “N/A”
        if not txt or txt.lower() in {"—", "na", "n/a", "null"}:
            return 0.0
        return float(txt)
    except Exception:
        return 0.0


def _clean_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _pick(el: Optional[object]) -> str:
    try:
        return el.get_text(" ", strip=True) if el is not None else ""
    except Exception:
        return ""


def _extract_from_kv_table(soup: "BeautifulSoup") -> Dict[str, str]:
    """
    Försöker hitta key-value tabeller på Morningstar-sidan (key-statistics).
    Vi plockar ut textpar som (label, value).
    """
    out: Dict[str, str] = {}
    if soup is None:
        return out

    # Morningstar har flera sektioner – vi scannar generellt alla tabeller/listor
    # och försöker hitta välkända label-strängar.
    candidates = soup.find_all(["table", "section", "div"])
    for node in candidates:
        # plocka rader
        rows = []
        rows += getattr(node, "find_all", lambda *_: [])("tr")
        rows += getattr(node, "find_all", lambda *_: [])("li")

        for r in rows:
            tds = getattr(r, "find_all", lambda *_: [])(["td", "span", "div"])
            if not tds or len(tds) < 2:
                continue
            label, value = "", ""
            # hitta första icke-tomma som label, och nästa som value
            for td in tds:
                text = _pick(td)
                if not text:
                    continue
                if not label:
                    label = text
                elif not value:
                    value = text
                    break
            if not label or not value:
                continue
            key = _clean_label(label)
            out[key] = value

    return out


def _extract_price(soup: "BeautifulSoup") -> float:
    """Försök hitta aktuellt pris (visas på flera ställen i header)."""
    if soup is None:
        return 0.0
    # Vanliga ställen: element med data-test eller tydlig "Price"
    # 1) Sök efter siffror nära ordet "Price"
    try:
        price_label = soup.find(string=re.compile(r"\bPrice\b", flags=re.I))
        if price_label:
            # titta på närliggande noder
            parent = price_label.parent if hasattr(price_label, "parent") else None
            if parent:
                # leta efter första text som ser ut som ett tal
                cand = parent.find_next(string=re.compile(r"\d"))
                if cand:
                    # rensa ev. $-tecken
                    m = re.search(r"([$\s]*)(-?\d+(?:\.\d+)?)", str(cand))
                    if m:
                        return _to_float(m.group(2))
    except Exception:
        pass

    # 2) Hitta stora siffer-element i headern (fallback)
    try:
        big_nums = soup.select("h2, h1, div, span")
        for el in big_nums[:400]:
            txt = _pick(el)
            if txt and re.match(r"^\$?\d+(?:\.\d+)?$", txt.strip()):
                return _to_float(txt.strip().lstrip("$"))
    except Exception:
        pass

    return 0.0


@st.cache_data(ttl=600, show_spinner=False)
def get_overview(ticker: str) -> Dict[str, Any]:
    """
    Försöker hämta: price, ps_ttm, pb, market_cap, shares_outstanding,
    dividend_yield_pct, payout_ratio_pct, book_value_per_share,
    gross_margins_pct, operating_margins_pct, profit_margins_pct.

    Returnerar alltid en dict; saknas värden -> 0/"".
    """
    out: Dict[str, Any] = {
        "price": 0.0,
        "ps_ttm": 0.0,
        "pb": 0.0,
        "market_cap": 0.0,
        "shares_outstanding": 0.0,
        "dividend_yield_pct": 0.0,
        "payout_ratio_pct": 0.0,
        "book_value_per_share": 0.0,
        "gross_margins_pct": 0.0,
        "operating_margins_pct": 0.0,
        "profit_margins_pct": 0.0,
    }

    if requests is None or BeautifulSoup is None:
        return out

    tkr = (ticker or "").strip().lower()
    if not tkr:
        return out

    # Prova flera bas-URL:er
    html = None
    for base in BASES:
        url = base.format(t=tkr)
        try:
            r = requests.get(
                url,
                headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
                timeout=TIMEOUT,
            )
            if r.status_code == 200 and r.text and ("Price" in r.text or "Valuation" in r.text):
                html = r.text
                break
        except Exception:
            continue

    if not html:
        return out

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return out

    # Pris
    price = _extract_price(soup)
    if price > 0:
        out["price"] = price

    # Plocka ut key-value-statistik
    kv = _extract_from_kv_table(soup)

    # Mappningar (vi söker generiska labels i lägretsad text)
    def _get_like(keys) -> str:
        for k, v in kv.items():
            for want in keys:
                if want in k:
                    return v
        return ""

    # PS / PB
    out["ps_ttm"] = _to_float(_get_like(["price/sales", "price to sales"]))
    out["pb"]     = _to_float(_get_like(["price/book", "price to book"]))

    # Market Cap & Shares (om de finns uttryckligt)
    mc = _get_like(["market cap", "market capitalization"])
    out["market_cap"] = _to_float(mc.replace("$", "")) if mc else 0.0

    sh = _get_like(["shares outstanding"])
    # Morningstar visar ibland i M/B – försök tolka suffix
    if sh:
        m = re.match(r"^(-?\d+(?:\.\d+)?)([KMBT])?$", sh.replace(",", ""), flags=re.I)
        if m:
            val = float(m.group(1))
            suf = (m.group(2) or "").upper()
            mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(suf, 1.0)
            out["shares_outstanding"] = val * mult
        else:
            out["shares_outstanding"] = _to_float(sh)

    # Utdelning
    out["dividend_yield_pct"] = _to_float(_get_like(["dividend yield", "yield %"]))
    out["payout_ratio_pct"]   = _to_float(_get_like(["payout ratio"]))

    # Book value / share
    out["book_value_per_share"] = _to_float(_get_like(["book value per share", "book value/ share", "bvps"]))

    # Marginaler
    out["gross_margins_pct"]     = _to_float(_get_like(["gross margin"]))
    out["operating_margins_pct"] = _to_float(_get_like(["operating margin"]))
    out["profit_margins_pct"]    = _to_float(_get_like(["net margin", "profit margin"]))

    return out
