# stockapp/fetchers/stocktwits.py
from __future__ import annotations

import time
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

try:
    import requests
except Exception:
    requests = None  # type: ignore


BASE = "https://api.stocktwits.com/api/2"
UA = st.secrets.get("STW_USER_AGENT") or (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)
TIMEOUT = 20
PAGE_LIMIT = 5          # max antal API-sidor vi hämtar (för att inte bli tunga)
SLEEP_BETWEEN = 0.6     # artigt delay mellan sid-hämtningar


def _get(url: str, params: Optional[dict] = None) -> Optional[dict]:
    if requests is None:
        return None
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_symbol_summary(ticker: str) -> Dict[str, Any]:
    """
    Sammanfattning för en Stocktwits-symbol (senaste 24h):
      - stw_messages_24h
      - stw_bull_24h
      - stw_bear_24h
      - stw_bull_ratio
      - stw_watchlist_count
      - stw_avg_msgs_per_hour_24h
      - last_message_at (ISO8601)
    """
    out: Dict[str, Any] = {
        "stw_messages_24h": 0,
        "stw_bull_24h": 0,
        "stw_bear_24h": 0,
        "stw_bull_ratio": 0.0,
        "stw_watchlist_count": 0,
        "stw_avg_msgs_per_hour_24h": 0.0,
        "last_message_at": "",
    }

    symbol = (ticker or "").strip().upper()
    if not symbol or requests is None:
        return out

    now = pd.Timestamp.utcnow()
    cutoff = now - pd.Timedelta(hours=24)

    # --- 1) Watchlist count ---
    sym_json = _get(f"{BASE}/symbols/{symbol}.json")
    if isinstance(sym_json, dict):
        try:
            sym = sym_json.get("symbol") or {}
            out["stw_watchlist_count"] = int(sym.get("watchlist_count", 0) or 0)
        except Exception:
            pass

    # --- 2) Meddelanden (paginera bakåt tills äldre än 24h eller PAGE_LIMIT nåtts) ---
    total_msgs = 0
    bull = 0
    bear = 0
    last_dt = None

    max_id = None
    for _ in range(PAGE_LIMIT):
        params = {}
        if max_id:
            params["max"] = max_id

        data = _get(f"{BASE}/streams/symbol/{symbol}.json", params=params)
        if not isinstance(data, dict):
            break

        msgs = data.get("messages") or []
        if not msgs:
            break

        for m in msgs:
            # created_at: "2025-01-18T15:31:02Z"
            try:
                c_at = pd.to_datetime(m.get("created_at"), utc=True)
            except Exception:
                continue

            if last_dt is None or c_at > last_dt:
                last_dt = c_at

            if c_at < cutoff:
                # vi är utanför 24h-fönstret; avsluta paginering helt
                msgs = []  # empty to break outer
                break

            total_msgs += 1
            # sentiment
            sent = (((m.get("entities") or {}).get("sentiment") or {}).get("basic") or "").lower()
            if sent == "bullish":
                bull += 1
            elif sent == "bearish":
                bear += 1

        # sätt nästa max (paginering bakåt)
        try:
            last_id = int(msgs[-1]["id"]) if msgs else None
        except Exception:
            last_id = None

        if last_id is None or (msgs == []):
            break

        # nästa sida
        max_id = last_id - 1
        time.sleep(SLEEP_BETWEEN)

    out["stw_messages_24h"] = int(total_msgs)
    out["stw_bull_24h"] = int(bull)
    out["stw_bear_24h"] = int(bear)
    out["stw_bull_ratio"] = round((bull / total_msgs * 100.0), 2) if total_msgs > 0 else 0.0
    out["stw_avg_msgs_per_hour_24h"] = round(total_msgs / 24.0, 2) if total_msgs > 0 else 0.0
    out["last_message_at"] = (last_dt.isoformat() if last_dt is not None else "")

    return out
