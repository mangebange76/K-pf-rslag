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

FINVIZ_URL = "https://finviz.com/quote.ashx?t={t}"
UA = st.secrets.get("FINVIZ_USER_AGENT") or "Mozilla/5.0"

def _to_float(x) -> float:
    try: return float(str(x).replace(",","").strip())
    except: return 0.0

def _pct(s: str) -> float:
    return _to_float(str(s).replace("%",""))

def _shortnum(s: str) -> float:
    s = str(s).replace(",","").strip()
    m = re.match(r"^(-?\d+(\.\d+)?)([KMBT])?$", s, re.I)
    if not m: return 0.0
    v = float(m.group(1)); suf = (m.group(3) or "").upper()
    mult = {"K":1e3,"M":1e6,"B":1e9,"T":1e12}.get(suf,1.0)
    return v*mult

@st.cache_data(ttl=600, show_spinner=False)
def get_overview(ticker: str) -> Dict[str, Any]:
    out = {"price":0.0,"ps_ttm":0.0,"pb":0.0,"shares_outstanding":0.0,
           "dividend_yield_pct":0.0,"payout_ratio_pct":0.0}
    if requests is None or BeautifulSoup is None or not ticker: return out
    try:
        r = requests.get(FINVIZ_URL.format(t=ticker.upper()), headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        snap = {}
        table = soup.select_one("table.snapshot-table2")
        if table:
            tds = table.find_all("td")
            for i in range(0, len(tds)-1, 2):
                snap[tds[i].get_text(strip=True)] = tds[i+1].get_text(strip=True)
        out["ps_ttm"] = _to_float(snap.get("P/S",""))
        out["pb"]     = _to_float(snap.get("P/B",""))
        out["dividend_yield_pct"] = _pct(snap.get("Dividend %",""))
        out["payout_ratio_pct"]   = _pct(snap.get("Payout",""))
        out["shares_outstanding"] = _shortnum(snap.get("Shs Outstand",""))
        # pris
        pnode = soup.find("td", string=re.compile(r"^Price$", re.I))
        if pnode and pnode.find_next_sibling("td"):
            out["price"] = _to_float(pnode.find_next_sibling("td").get_text(strip=True))
        return out
    except Exception:
        return out
