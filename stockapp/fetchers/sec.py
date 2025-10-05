from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd
import streamlit as st

try:
    import requests
except Exception:
    requests = None  # type: ignore

try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

SEC_UA = st.secrets.get("SEC_USER_AGENT") or "youremail@example.com"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

def _get(url: str) -> Any:
    if requests is None: return None
    r=requests.get(url, headers={"User-Agent": SEC_UA, "Accept":"application/json"}, timeout=30)
    r.raise_for_status(); return r.json()

@st.cache_data(ttl=86400, show_spinner=False)
def _map_ticker_cik() -> Dict[str,str]:
    try:
        d = _get(SEC_TICKERS)
        out={}
        for _,v in (d or {}).items():
            t=str(v.get("ticker","")).upper()
            c=str(v.get("cik_str",""))
            if t and c: out[t]=f"{int(c):010d}"
        return out
    except Exception:
        return {}

def _series(facts: Dict[str,Any], tag: str, unit: str) -> List[Dict[str,Any]]:
    node=(facts.get("facts") or {}).get("us-gaap",{}).get(tag,{})
    arr=(node.get("units") or {}).get(unit,[])
    out=[]
    for it in arr:
        end=str(it.get("end") or "")
        val=it.get("val")
        form=str(it.get("form") or "")
        if not end or val is None: continue
        if form and form not in {"10-Q","10-K","20-F","40-F"}: continue
        out.append({"end":end,"val":float(val)})
    out=sorted(out, key=lambda x:x["end"])
    return out[-12:]

@st.cache_data(ttl=3600, show_spinner=False)
def get_pb_quarters(ticker: str) -> Dict[str,Any]:
    out={"pb_quarters": [], "details": []}
    if not ticker or requests is None or yf is None: return out
    tik=ticker.upper().strip()
    cik=_map_ticker_cik().get(tik)
    if not cik: return out
    try:
        facts=_get(FACTS.format(cik=cik))
    except Exception:
        return out
    eq=_series(facts, "StockholdersEquity", "USD")
    if not eq:
        eq=_series(facts, "CommonStockholdersEquity", "USD")
    sh=_series(facts, "CommonStockSharesOutstanding", "shares")
    if not eq or not sh: return out
    eq_by={e["end"]: e["val"] for e in eq}
    sh_by={s["end"]: s["val"] for s in sh}
    dates=sorted(set(eq_by.keys()) & set(sh_by.keys()))
    det=[]; pairs=[]
    for d in dates[::-1]:
        equity=eq_by[d]; shares=sh_by[d]
        if equity<=0 or shares<=0: continue
        bvps = equity / shares
        try:
            t=yf.Ticker(tik)
            dt=pd.to_datetime(d)
            h=t.history(start=(dt-pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
                        end  =(dt+pd.Timedelta(days=3)).strftime("%Y-%m-%d"))
            price=float(h["Close"].iloc[(h.index-dt).abs().argmin()]) if not h.empty else 0.0
        except Exception:
            price=0.0
        pb = (price/bvps) if (price>0 and bvps>0) else 0.0
        det.append({"date":d,"equity":round(equity,2),"shares":round(shares,2),
                    "bvps":round(bvps,4),"price":round(price,4),"pb":round(pb,4)})
        if pb>0: pairs.append((d, round(pb,2)))
        if len(det)>=8: break
    out["details"]=sorted(det, key=lambda x:x["date"], reverse=True)
    out["pb_quarters"]=pairs[:4]
    return out
