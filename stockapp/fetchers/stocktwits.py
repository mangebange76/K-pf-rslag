# stockapp/fetchers/stocktwits.py
from __future__ import annotations

import math
import time
from typing import Dict, Any, List, Optional

import requests
import streamlit as st
from datetime import datetime, timezone, timedelta

try:
    from dateutil import parser as dtparser  # robust parsedatum
except Exception:
    dtparser = None  # type: ignore

# ------------------------- Konfiguration -------------------------

UA = st.secrets.get("STOCKTWITS_USER_AGENT") or (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
SHOW_URL   = "https://api.stocktwits.com/api/2/symbols/show/{ticker}.json"


# -------------------------- Hjälpfunktioner -----------------------

def _safe_int(x) -> int:
    try:
        if x is None:
            return 0
        i = int(x)
        return i if i >= 0 else 0
    except Exception:
        return 0

def _safe_float(x) -> float:
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0

def _parse_created_at(s: str) -> Optional[datetime]:
    """
    Stocktwits created_at t.ex: 'Mon, 05 Oct 2015 18:45:00 -0400'
    Försök med dateutil, annars med strptime.
    Returnerar UTC-aware datetime.
    """
    if not s:
        return None
    try:
        if dtparser:
            d = dtparser.parse(s)
        else:
            # Fallback: Mon, 05 Oct 2015 18:45:00 -0400
            d = datetime.strptime(s, "%a, %d %b %Y %H:%M:%S %z")
        return d.astimezone(timezone.utc)
    except Exception:
        return None

def _within_last_hours(d: datetime, hours: int = 24) -> bool:
    try:
        now = datetime.now(timezone.utc)
        return (now - d) <= timedelta(hours=hours)
    except Exception:
        return False


# --------------------------- API-funktioner -----------------------

def _fetch_stream_page(ticker: str, max_id: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if max_id is not None:
        params["max"] = int(max_id)
    headers = {"User-Agent": UA}
    url = STREAM_URL.format(ticker=ticker.upper().strip())
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def _fetch_symbol_show(ticker: str) -> Dict[str, Any]:
    headers = {"User-Agent": UA}
    url = SHOW_URL.format(ticker=ticker.upper().strip())
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def get_symbol_summary(ticker: str, pages: int = 3, sleep_between: float = 0.3) -> Dict[str, Any]:
    """
    Sammanfattning för en symbol:
      - stw_messages_24h
      - stw_bull_24h
      - stw_bear_24h
      - stw_bull_ratio  (bull/(bull+bear))
      - stw_watchlist_count
      - stw_avg_msgs_per_hour_24h
      - last_message_at (ISO)
    Läser upp till 'pages' sidor från streams/symbol (paginering via 'max').
    """
    tkr = ticker.upper().strip()
    total_msgs_24h = 0
    bull_24h = 0
    bear_24h = 0
    last_message_at: Optional[datetime] = None

    # --------- Hämta stream-sidor ----------
    next_max: Optional[int] = None
    for _ in range(max(1, pages)):
        try:
            js = _fetch_stream_page(tkr, max_id=next_max)
        except Exception:
            break

        msgs: List[Dict[str, Any]] = js.get("messages", []) or []
        if not msgs:
            break

        # uppdatera next_max för äldre meddelanden
        oldest_id = None
        for m in msgs:
            mid = _safe_int(m.get("id"))
            if mid and (oldest_id is None or mid < oldest_id):
                oldest_id = mid
        if oldest_id:
            next_max = oldest_id - 1

        # summera 24h
        found_any_recent = False
        for m in msgs:
            ts = _parse_created_at(m.get("created_at", ""))
            if ts:
                if (last_message_at is None) or (ts > last_message_at):
                    last_message_at = ts
                if _within_last_hours(ts, 24):
                    found_any_recent = True
                    total_msgs_24h += 1
                    sentiment = (m.get("entities") or {}).get("sentiment") or m.get("sentiment") or {}
                    # vissa poster har 'basic' {'bullish': True/False}, andra har 'sentiment': {'basic': 'Bullish'/'Bearish'}
                    if isinstance(sentiment, dict):
                        basic = sentiment.get("basic")
                        if isinstance(basic, str):
                            s = basic.lower()
                            if "bull" in s:
                                bull_24h += 1
                            elif "bear" in s:
                                bear_24h += 1
                        else:
                            # ibland boolean flags
                            if sentiment.get("bullish") is True:
                                bull_24h += 1
                            if sentiment.get("bearish") is True:
                                bear_24h += 1

        # Om inga nya inom 24h i denna sida, avbryt tidigt
        if not found_any_recent:
            break

        time.sleep(sleep_between)

    # --------- Hämta show (watchlist_count m.m.) ----------
    watchlist_count = 0
    try:
        js2 = _fetch_symbol_show(tkr)
        # struktur: {"symbol": {"id":..., "symbol":"AMD", "watchlist_count": 123456, ...}}
        sym = js2.get("symbol") or {}
        watchlist_count = _safe_int(sym.get("watchlist_count"))
        # fallback: visa i vissa svar ligger i sym.get("watchlist") eller liknande
        if not watchlist_count:
            watchlist_count = _safe_int(sym.get("watchlist"))
    except Exception:
        pass

    # bull-ratio
    denom = bull_24h + bear_24h
    bull_ratio = (bull_24h / denom * 100.0) if denom > 0 else 0.0

    avg_msgs_per_hour = (total_msgs_24h / 24.0) if total_msgs_24h > 0 else 0.0

    return {
        "stw_messages_24h": int(total_msgs_24h),
        "stw_bull_24h": int(bull_24h),
        "stw_bear_24h": int(bear_24h),
        "stw_bull_ratio": round(bull_ratio, 2),
        "stw_watchlist_count": int(watchlist_count),
        "stw_avg_msgs_per_hour_24h": round(avg_msgs_per_hour, 3),
        "last_message_at": last_message_at.isoformat() if last_message_at else "",
        "source": "stocktwits",
    }
