from __future__ import annotations
import re
from typing import Dict, Any, Optional
import streamlit as st

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

BASES = [
    "https://www.morningstar.com/stocks/xnas/{t}/quote",
    "https://www.morningstar.com/stocks/xnys/{t}/quote",
    "https://www.morningstar.com/stocks/xase/{t}/quote",
]
UA = st.secrets.get("MORNINGSTAR_USER_AGENT") or "Mozilla/5.0"

def _f(x) -> float:
    try: return float(str(x).replace(",","").replace("%","").strip())
    except: return 0.0

def _get_like(d: Dict[str,str], keys) -> str:
    for k,v in d.items():
        kl = k.lower()
        for want in keys:
            if want in kl: return v
    return ""

def _kv(soup: "BeautifulSoup") -> Dict[str,str]:
    out={}
    if not soup: return out
    for node in soup.find_all(["table","section","div","ul","ol"]):
        rows = []
        rows += node.find_all("tr")
        rows += node.find_all("li")
        for r in rows:
            cells = r.find_all(["td","span","div"])
            vals=[c.get_text(" ",strip=True) for c in cells if c.get_text(strip=True)]
            if len(vals)>=2:
                out[vals[0]]=vals[1]
    return out

@st.cache_data(ttl=600, show_spinner=False)
def get_overview(ticker: str) -> Dict[str, Any]:
    out={"price":0.0,"ps_ttm":0.0,"pb":0.0,"dividend_yield_pct":0.0,"payout_ratio_pct":0.0}
    if requests is None or BeautifulSoup is None or not ticker: return out
    html=None
    for b in BASES:
        try:
            r=requests.get(b.format(t=ticker.lower()), headers={"User-Agent": UA}, timeout=30)
            if r.status_code==200 and ("Price" in r.text or "Valuation" in r.text):
                html=r.text; break
        except Exception:
            continue
    if not html: return out
    soup=BeautifulSoup(html,"html.parser")
    stats=_kv(soup)
    out["ps_ttm"]=_f(_get_like(stats,["price/sales","price to sales"]))
    out["pb"]    =_f(_get_like(stats,["price/book","price to book"]))
    out["dividend_yield_pct"]=_f(_get_like(stats,["dividend yield","yield %"]))
    out["payout_ratio_pct"]  =_f(_get_like(stats,["payout ratio"]))
    # pris – försök hitta en siffra nära "Price"
    try:
        pr_label = soup.find(string=re.compile(r"\bPrice\b", re.I))
        if pr_label:
            cand = pr_label.parent.find_next(string=re.compile(r"\d"))
            m=re.search(r"(-?\d+(\.\d+)?)", str(cand))
            if m: out["price"]=_f(m.group(1))
    except Exception:
        pass
    return out
