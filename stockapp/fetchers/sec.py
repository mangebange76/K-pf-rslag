# stockapp/fetchers/sec.py
from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Tuple

import requests
import streamlit as st

# SEC kräver User-Agent
_UA = st.secrets.get("SEC_USER_AGENT", "yourname@example.com")

HEADERS = {
    "User-Agent": _UA,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TTL_SECS = 60 * 60  # 1h cache


def _pad_cik(cik: int | str) -> str:
    s = str(cik).strip()
    return s.zfill(10)


@st.cache_data(ttl=TTL_SECS, show_spinner=False)
def _ticker_to_cik(ticker: str) -> str:
    """
    Mappar ticker -> CIK via officiella listan.
    Hämtas en gång och cacheas.
    """
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers={"User-Agent": _UA}, timeout=30)
        r.raise_for_status()
        data = r.json()
        # data är en dict med index som nycklar
        t = ticker.upper()
        for _, row in data.items():
            if str(row.get("ticker", "")).upper() == t:
                return _pad_cik(row.get("cik_str"))
    except Exception:
        pass
    return ""


def _closest_price_on_date(ticker: str, date_str: str) -> float:
    """
    Hämtar närmaste stängningskurs (±3 dagar) runt 'date_str' via Yahoo query2.
    Vi använder Yahoo's snabba endpoint för pris-historik.
    """
    try:
        d = dt.datetime.fromisoformat(date_str)
    except Exception:
        return 0.0

    start = int((d - dt.timedelta(days=3)).timestamp())
    end = int((d + dt.timedelta(days=3)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"period1": start, "period2": end, "interval": "1d"}
    r = requests.get(url, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    js = r.json()
    closes = (js.get("chart", {}).get("result", [{}])[0]
              .get("indicators", {}).get("quote", [{}])[0].get("close", []))
    # ta sista icke-None
    for v in reversed(closes or []):
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return 0.0


def _pick_facts(facts: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
    """Plocka ut alla perioder för första nyckel som finns."""
    for k in keys:
        node = facts.get("us-gaap", {}).get(k)
        if not node:
            continue
        # Föredra kvartalsvisa ("Q") från 'units'
        for unit, arr in (node.get("units") or {}).items():
            # Equity är i USD, shares i shares
            if not isinstance(arr, list):
                continue
            return arr  # returnera första träffens alla observationer
    return []


@st.cache_data(ttl=TTL_SECS, show_spinner=False)
def get_pb_quarters(ticker: str) -> Dict[str, Any]:
    """
    Beräknar P/B för de senaste upp till 4 kvartalen:
      P/B = (stängningskurs vid periodens slut) / (Equity / Antal aktier)
    Returnerar:
      {"pb_quarters": [(ISO-datum, pb-float), ...], "warnings": [str,...]}
    """
    out: Dict[str, Any] = {"pb_quarters": [], "warnings": []}
    try:
        cik = _ticker_to_cik(ticker)
        if not cik:
            out["warnings"].append("Hittade ingen CIK för tickern.")
            return out

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Hämta equity och aktier
        # Equity: StockholdersEquity eller inkl NCI
        eq_arr = _pick_facts(data.get("facts", {}), [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ])
        sh_arr = _pick_facts(data.get("facts", {}), [
            "CommonStockSharesOutstanding",
            "EntityCommonStockSharesOutstanding",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
        ])

        if not eq_arr or not sh_arr:
            out["warnings"].append("Saknar equity/aktier i SEC-facts.")
            return out

        # Gör om till dict med datum -> värde (ta senaste observation per datum)
        def to_map(arr):
            m = {}
            for item in arr:
                d = item.get("end")
                v = item.get("val")
                if d and v is not None:
                    m[d] = float(v)
            return m

        eq_map = to_map(eq_arr)
        sh_map = to_map(sh_arr)

        # Intersektion av datum
        common_dates = sorted(set(eq_map.keys()) & set(sh_map.keys()))
        if not common_dates:
            out["warnings"].append("Hittade inga gemensamma datum mellan equity och aktier.")
            return out

        pairs: List[Tuple[str, float]] = []
        # Vi tar de senaste kvartalen (max 6, sen klipper vi till 4 med pris)
        for d in reversed(common_dates[-8:]):
            equity = float(eq_map[d])
            shares = float(sh_map[d])
            if equity <= 0 or shares <= 0:
                continue
            bvps = equity / shares
            price = _closest_price_on_date(ticker, d)
            if price <= 0 or bvps <= 0:
                continue
            pb = price / bvps
            pairs.append((d, round(pb, 2)))
            if len(pairs) >= 4:
                break

        out["pb_quarters"] = pairs
        if not pairs:
            out["warnings"].append("Kunde inte beräkna P/B för någon kvartalsperiod.")
        return out

    except requests.HTTPError as he:
        out["warnings"].append(f"HTTP-fel från SEC: {he}")
        return out
    except Exception as e:
        out["warnings"].append(f"Fel vid SEC-hämtning: {e}")
        return out
