# stockapp/sources.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from .config import TS_FIELDS
from .utils import now_stamp

# ---------------------------------------------------------------------
# Konstanter (källnamn)
# ---------------------------------------------------------------------
AUTO_SOURCE_PRICE    = "Auto (yfinance: price/basic)"
AUTO_SOURCE_FULL_YF  = "Auto (yfinance: price+financials)"
AUTO_SOURCE_SEC_FMP  = "Auto (SEC→FMP→yfinance)"

# ---------------------------------------------------------------------
# Hjälpmetoder för uppdatering & stämpling
# ---------------------------------------------------------------------

def _stamp_ts(df: pd.DataFrame, row_idx: int, field: str, date_str: Optional[str] = None):
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    try:
        df.at[row_idx, ts_col] = date_str or now_stamp()
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _safe_set(df: pd.DataFrame, row_idx: int, field: str, value, changed: List[str], force_stamp: bool = True):
    """Sätt fältet, lägg till i changed om värdet skiftar. TS stämplas alltid för spårade fält."""
    if field not in df.columns:
        return
    old = df.at[row_idx, field]
    same = False
    try:
        same = (pd.isna(old) and pd.isna(value)) or (str(old) == str(value))
    except Exception:
        same = False

    df.at[row_idx, field] = value
    if not same:
        changed.append(field)

    if force_stamp and field in TS_FIELDS:
        _stamp_ts(df, row_idx, field)

def _coerce_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _mean_pos(values: List[float]) -> float:
    clean = [float(v) for v in values if _coerce_float(v, 0.0) > 0]
    return round(float(np.mean(clean)), 2) if clean else 0.0

# ---------------------------------------------------------------------
# yfinance-hjälpare
# ---------------------------------------------------------------------

def _yfi_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}

def _yfi_history_close_on_or_before(tkr: str, dt: pd.Timestamp) -> Optional[float]:
    try:
        start = (pd.to_datetime(dt) - pd.Timedelta(days=30)).date()
        end   = (pd.to_datetime(dt) + pd.Timedelta(days=2)).date()
        hist = yf.Ticker(tkr).history(start=start, end=end, interval="1d")
        if hist is None or hist.empty:
            return None
        hist = hist.sort_index()
        idx_dates = list(hist.index.date)
        closes = list(hist["Close"].values)
        for j in range(len(idx_dates)-1, -1, -1):
            if idx_dates[j] <= pd.to_datetime(dt).date():
                try:
                    return float(closes[j])
                except Exception:
                    return None
        return None
    except Exception:
        return None

def _yfi_quarterly_revenue_rows(ticker: str) -> List[Tuple[pd.Timestamp, float]]:
    """[(period_end, revenue), ...] nyast→äldst."""
    t = yf.Ticker(ticker)
    # quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue","TotalRevenue","Revenues","Revenue","Sales",
                "Total revenue","Revenues from contracts with customers"
            ]
            for key in cand_rows:
                if key in idx:
                    row = qf.loc[key].dropna()
                    out = []
                    for c, v in row.items():
                        try:
                            d = (c.to_pydatetime() if hasattr(c, "to_pydatetime") else pd.to_datetime(c)).normalize()
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # fallback: income_stmt quarterly
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = (c.to_pydatetime() if hasattr(c, "to_pydatetime") else pd.to_datetime(c)).normalize()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def _ttm_windows(values: List[Tuple[pd.Timestamp, float]], need: int = 4) -> List[Tuple[pd.Timestamp, float]]:
    out: List[Tuple[pd.Timestamp, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

# ---------------------------------------------------------------------
# SEC-hjälpare
# ---------------------------------------------------------------------

def _sec_headers() -> dict:
    # Bästa praxis kräver UA. Om du har streamlit.secrets, använd den.
    ua = "StockApp/1.0 (contact: example@example.com)"
    try:
        import streamlit as st  # type: ignore
        ua = st.secrets.get("SEC_USER_AGENT", ua)
    except Exception:
        pass
    return {"User-Agent": ua}

@lru_cache(maxsize=1)
def _sec_ticker_map() -> Dict[str, str]:
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_sec_headers(), timeout=30)
        if r.status_code != 200:
            return {}
        data = r.json()
        out = {}
        for _, v in (data or {}).items():
            try:
                out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
            except Exception:
                pass
        return out
    except Exception:
        return {}

def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _sec_companyfacts(cik10: str):
    try:
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json", headers=_sec_headers(), timeout=30)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

def _parse_iso(d: str):
    try:
        return pd.to_datetime(d).date()
    except Exception:
        return None

def _is_instant_entry(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True
    d1 = _parse_iso(str(start)); d2 = _parse_iso(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2
        except Exception:
            return False
    return False

def _collect_share_entries(facts: dict) -> list:
    entries = []
    facts_all = (facts.get("facts") or {})
    sources = [
        ("dei", ["EntityCommonStockSharesOutstanding", "EntityCommonSharesOutstanding"]),
        ("us-gaap", ["CommonStockSharesOutstanding", "ShareIssued"]),
        ("ifrs-full", ["NumberOfSharesIssued", "IssuedCapitalNumberOfShares", "OrdinarySharesNumber", "NumberOfOrdinaryShares"]),
    ]
    unit_keys = ("shares", "USD_shares", "Shares", "SHARES")
    for taxo, keys in sources:
        sect = facts_all.get(taxo, {})
        for key in keys:
            fact = sect.get(key)
            if not fact:
                continue
            units = fact.get("units") or {}
            for uk in unit_keys:
                arr = units.get(uk)
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    if not _is_instant_entry(it):
                        continue
                    end = _parse_iso(str(it.get("end", "")))
                    val = it.get("val", None)
                    if end and val is not None:
                        try:
                            v = float(val)
                            entries.append({"end": end, "val": v})
                        except Exception:
                            pass
    return entries

def _sec_latest_shares_robust(facts: dict) -> float:
    rows = _collect_share_entries(facts)
    if not rows:
        return 0.0
    newest = max(r["end"] for r in rows)
    todays = [r for r in rows if r["end"] == newest]
    total = 0.0
    for r in todays:
        try:
            total += float(r["val"])
        except Exception:
            pass
    return total if total > 0 else 0.0

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A")}),
    ]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    prefer_units = ("USD","CAD","EUR","GBP")

    for taxo, cfg in taxos:
        gaap = (facts.get("facts") or {}).get(taxo, {})
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list):
                    continue
                tmp = []
                for it in arr:
                    form = (it.get("form") or "").upper()
                    if not any(f in form for f in cfg["forms"]):
                        continue
                    end = _parse_iso(str(it.get("end", "")))
                    start = _parse_iso(str(it.get("start", "")))
                    val = it.get("val", None)
                    if not (end and start and val is not None):
                        continue
                    try:
                        dur = (end - start).days
                    except Exception:
                        dur = None
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((pd.to_datetime(end), v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

# ---------------------------------------------------------------------
# FX (Frankfurter → exchangerate.host)
# ---------------------------------------------------------------------

@lru_cache(maxsize=128)
def _fx_rate(base: str, quote: str) -> float:
    base = (base or "").upper()
    quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v:
                return float(v)
    except Exception:
        pass
    return 1.0

# ---------------------------------------------------------------------
# FMP-hjälpare (vänlig fallback för P/S om du har nyckel)
# ---------------------------------------------------------------------

def _fmp_key_base():
    base = "https://financialmodelingprep.com"
    key = ""
    try:
        import streamlit as st  # type: ignore
        base = st.secrets.get("FMP_BASE", base)
        key  = st.secrets.get("FMP_API_KEY", key)
    except Exception:
        pass
    return base, key

def _fmp_get(path: str, params=None):
    base, key = _fmp_key_base()
    p = (params or {}).copy()
    if key:
        p["apikey"] = key
    url = f"{base}/{path}"
    try:
        r = requests.get(url, params=p, timeout=20)
        if 200 <= r.status_code < 300:
            return r.json(), r.status_code
        return None, r.status_code
    except Exception:
        return None, 0

def _fmp_ps_ttm(symbol: str) -> Optional[float]:
    data, sc = _fmp_get(f"api/v3/ratios-ttm/{symbol}", None)
    if isinstance(data, list) and data:
        v = data[0].get("priceToSalesRatioTTM") or data[0].get("priceToSalesTTM")
        try:
            if v and float(v) > 0:
                return float(v)
        except Exception:
            return None
    # fallback key-metrics-ttm
    data, sc = _fmp_get(f"api/v3/key-metrics-ttm/{symbol}", None)
    if isinstance(data, list) and data:
        v = data[0].get("priceToSalesRatioTTM") or data[0].get("priceToSalesTTM")
        try:
            if v and float(v) > 0:
                return float(v)
        except Exception:
            return None
    # calc from marketCap / revenueTTM
    q, sc = _fmp_get(f"api/v3/quote/{symbol}", None)
    if isinstance(q, list) and q:
        mc = q[0].get("marketCap")
        try:
            mc = float(mc or 0.0)
        except Exception:
            mc = 0.0
        isttm, sc2 = _fmp_get(f"api/v3/income-statement-ttm/{symbol}", None)
        rev = 0.0
        if isinstance(isttm, list) and isttm:
            cand = isttm[0]
            for k in ("revenueTTM", "revenue"):
                if cand.get(k) is not None:
                    try:
                        rev = float(cand[k]); break
                    except Exception:
                        pass
        if mc > 0 and rev > 0:
            return mc / rev
    return None

# ---------------------------------------------------------------------
# Härledda beräkningar (per rad)
# ---------------------------------------------------------------------

def _recompute_row_derivatives(df: pd.DataFrame, ridx: int):
    ps_q = [df.at[ridx, c] if c in df.columns else 0.0 for c in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]]
    ps_avg = _mean_pos([_coerce_float(x, 0.0) for x in ps_q])
    df.at[ridx, "P/S-snitt"] = ps_avg

    shares_m = _coerce_float(df.at[ridx, "Utestående aktier"], 0.0)
    if shares_m <= 0 or ps_avg <= 0:
        for c in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
            if c in df.columns:
                df.at[ridx, c] = 0.0
        return

    rev_today_m   = _coerce_float(df.at[ridx, "Omsättning idag"], 0.0)
    rev_next_m    = _coerce_float(df.at[ridx, "Omsättning nästa år"], 0.0)
    rev_2y_m      = _coerce_float(df.at[ridx, "Omsättning om 2 år"], 0.0)
    rev_3y_m      = _coerce_float(df.at[ridx, "Omsättning om 3 år"], 0.0)

    if rev_next_m > 0 and (rev_2y_m <= 0 or rev_3y_m <= 0):
        cagr = _coerce_float(df.at[ridx, "CAGR 5 år (%)"], 0.0)
        if cagr > 100.0:
            cagr = 50.0
        if cagr < 0.0:
            cagr = 2.0
        g = cagr / 100.0
        if rev_2y_m <= 0:
            rev_2y_m = round(rev_next_m * (1.0 + g), 2)
            df.at[ridx, "Omsättning om 2 år"] = rev_2y_m
        if rev_3y_m <= 0:
            rev_3y_m = round(rev_next_m * ((1.0 + g) ** 2), 2)
            df.at[ridx, "Omsättning om 3 år"] = rev_3y_m

    def _px(rev_m):
        if rev_m > 0:
            return round((rev_m * ps_avg) / shares_m, 2)
        return 0.0

    if "Riktkurs idag" in df.columns:    df.at[ridx, "Riktkurs idag"]    = _px(rev_today_m)
    if "Riktkurs om 1 år" in df.columns: df.at[ridx, "Riktkurs om 1 år"] = _px(rev_next_m)
    if "Riktkurs om 2 år" in df.columns: df.at[ridx, "Riktkurs om 2 år"] = _px(rev_2y_m)
    if "Riktkurs om 3 år" in df.columns: df.at[ridx, "Riktkurs om 3 år"] = _px(rev_3y_m)

# ---------------------------------------------------------------------
# Runners: yfinance basic/full
# ---------------------------------------------------------------------

def run_update_price_only(df: pd.DataFrame, user_rates: Dict[str, float], ticker: str, **kwargs) -> Tuple[pd.DataFrame, List[str], str]:
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df, [], "Ingen ticker angiven."
    if "Ticker" not in df.columns or tkr not in set(df["Ticker"].astype(str).str.upper()):
        return df, [], f"{tkr} hittades inte i tabellen."

    ridx = df.index[df["Ticker"].astype(str).str.upper() == tkr][0]
    changed: List[str] = []

    info = _yfi_info(tkr)

    # price
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = yf.Ticker(tkr).history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None and price > 0:
        _safe_set(df, ridx, "Aktuell kurs", float(price), changed)

    # name
    name = info.get("shortName") or info.get("longName")
    if name:
        _safe_set(df, ridx, "Bolagsnamn", str(name), changed, force_stamp=False)

    # currency
    ccy = info.get("currency")
    if ccy:
        _safe_set(df, ridx, "Valuta", str(ccy).upper(), changed, force_stamp=False)

    # dividend
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try:
            _safe_set(df, ridx, "Årlig utdelning", float(div_rate), changed, force_stamp=False)
        except Exception:
            pass

    # shares → M
    shares = info.get("sharesOutstanding")
    if shares is not None and float(shares) > 0:
        _safe_set(df, ridx, "Utestående aktier", float(shares)/1e6, changed)

    # market cap (om kolumn finns)
    if "Market cap (nu)" in df.columns:
        mc = info.get("marketCap")
        if mc is not None and float(mc) > 0:
            _safe_set(df, ridx, "Market cap (nu)", float(mc), changed, force_stamp=False)

    _note_auto_update(df, ridx, AUTO_SOURCE_PRICE)
    _recompute_row_derivatives(df, ridx)

    msg = f"{tkr}: uppdaterade {', '.join(changed) if changed else 'inga fält (oförändrat)'}."
    return df, changed, msg


def run_update_full(df: pd.DataFrame, user_rates: Dict[str, float], ticker: str, force_stamp: bool = True, **kwargs) -> Tuple[pd.DataFrame, List[str], str]:
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df, [], "Ingen ticker angiven."
    if "Ticker" not in df.columns or tkr not in set(df["Ticker"].astype(str).str.upper()):
        return df, [], f"{tkr} hittades inte i tabellen."

    ridx = df.index[df["Ticker"].astype(str).str.upper() == tkr][0]
    changed: List[str] = []

    info = _yfi_info(tkr)

    # price
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = yf.Ticker(tkr).history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None and price > 0:
        _safe_set(df, ridx, "Aktuell kurs", float(price), changed, force_stamp=force_stamp)

    # name/currency/dividend
    name = info.get("shortName") or info.get("longName")
    if name:
        _safe_set(df, ridx, "Bolagsnamn", str(name), changed, force_stamp=False)
    ccy = info.get("currency")
    if ccy:
        _safe_set(df, ridx, "Valuta", str(ccy).upper(), changed, force_stamp=False)
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try:
            _safe_set(df, ridx, "Årlig utdelning", float(div_rate), changed, force_stamp=False)
        except Exception:
            pass

    # shares → M
    shares = info.get("sharesOutstanding")
    if shares is not None and float(shares) > 0:
        _safe_set(df, ridx, "Utestående aktier", float(shares)/1e6, changed, force_stamp=force_stamp)

    # market cap (om kolumn finns)
    if "Market cap (nu)" in df.columns:
        mc = info.get("marketCap")
        if mc is not None and float(mc) > 0:
            _safe_set(df, ridx, "Market cap (nu)", float(mc), changed, force_stamp=False)

    # P/S TTM + P/S Q1..Q4 via kvartal
    q_rows = _yfi_quarterly_revenue_rows(tkr)
    ttm_list = _ttm_windows(q_rows, need=4)

    if ttm_list:
        ttm_end0, ttm0 = ttm_list[0]
        mc_now = info.get("marketCap")
        try:
            mc_now = float(mc_now or 0.0)
        except Exception:
            mc_now = 0.0
        if mc_now <= 0:
            px_now = float(price or 0.0)
            so     = float(shares or 0.0)
            if px_now > 0 and so > 0:
                mc_now = px_now * so
        if mc_now > 0 and ttm0 and ttm0 > 0:
            _safe_set(df, ridx, "P/S", float(mc_now / ttm0), changed, force_stamp=force_stamp)

        implied_shares = None
        try:
            px_now = float(price or 0.0)
            if (mc_now or 0.0) > 0 and px_now > 0:
                implied_shares = mc_now / px_now
        except Exception:
            implied_shares = None
        if implied_shares is None or implied_shares <= 0:
            implied_shares = float(shares or 0.0)

        if implied_shares and implied_shares > 0:
            for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
                if ttm_rev and ttm_rev > 0:
                    px_hist = _yfi_history_close_on_or_before(tkr, d_end)
                    if px_hist and px_hist > 0:
                        mcap_hist = implied_shares * float(px_hist)
                        _safe_set(df, ridx, f"P/S Q{idx}", float(mcap_hist / ttm_rev), changed, force_stamp=force_stamp)

    _note_auto_update(df, ridx, AUTO_SOURCE_FULL_YF)
    _recompute_row_derivatives(df, ridx)

    msg = f"{tkr}: uppdaterade {', '.join(changed) if changed else 'inga fält (oförändrat)'}."
    return df, changed, msg

# ---------------------------------------------------------------------
# Runner: SEC→FMP→yfinance
# ---------------------------------------------------------------------

def run_update_full_sec_combo(df: pd.DataFrame, user_rates: Dict[str, float], ticker: str, force_stamp: bool = True, **kwargs) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Full uppdatering:
      1) SEC (kvartalsintäkter + robust shares, TTM, FX-konvertering) → P/S och P/S Q1..Q4
      2) FMP för P/S TTM (fallback) om SEC inte ger resultat
      3) yfinance fallback (pris/info och kvartar) om fortfarande luckor
    OBS: Uppdaterar inte 'Omsättning idag'/'Omsättning nästa år' (manuella fält).
    """
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df, [], "Ingen ticker angiven."
    if "Ticker" not in df.columns or tkr not in set(df["Ticker"].astype(str).str.upper()):
        return df, [], f"{tkr} hittades inte i tabellen."

    ridx = df.index[df["Ticker"].astype(str).str.upper() == tkr][0]
    changed: List[str] = []
    sources_used: List[str] = []

    # 0) yfinance basics (pris/namn/valuta/divid, implied shares fallback)
    info = _yfi_info(tkr)
    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = yf.Ticker(tkr).history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if price is not None and price > 0:
        _safe_set(df, ridx, "Aktuell kurs", float(price), changed, force_stamp=force_stamp)
    name = info.get("shortName") or info.get("longName")
    if name:
        _safe_set(df, ridx, "Bolagsnamn", str(name), changed, force_stamp=False)
    ccy = info.get("currency")
    if ccy:
        _safe_set(df, ridx, "Valuta", str(ccy).upper(), changed, force_stamp=False)
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        try:
            _safe_set(df, ridx, "Årlig utdelning", float(div_rate), changed, force_stamp=False)
        except Exception:
            pass
    yfi_shares = info.get("sharesOutstanding")
    px_ccy = (str(ccy).upper() if ccy else "USD")

    # 1) SEC
    cik = _sec_cik_for(tkr)
    used_sec = False
    if cik:
        facts, sc = _sec_companyfacts(cik)
        if sc == 200 and isinstance(facts, dict):
            # shares via SEC
            sec_shares = _sec_latest_shares_robust(facts)  # styck
            shares_used = 0.0
            if (yfi_shares or 0) and float(yfi_shares) > 0:
                shares_used = float(yfi_shares)
                shares_source = "yfinance implied/sharesOutstanding"
            elif sec_shares and sec_shares > 0:
                shares_used = float(sec_shares)
                shares_source = "SEC instant"
            else:
                shares_used = 0.0
                shares_source = "unknown"

            if shares_used > 0:
                _safe_set(df, ridx, "Utestående aktier", shares_used/1e6, changed, force_stamp=force_stamp)

            # SEC kvartalsintäkter + unit
            q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
            if q_rows and rev_unit:
                fx = 1.0
                if rev_unit.upper() != px_ccy:
                    fx = _fx_rate(rev_unit.upper(), px_ccy) or 1.0
                # räkna TTM i prisvalutan
                ttm = _ttm_windows(q_rows, need=4)
                ttm_px = [(d, v * fx) for (d, v) in ttm]

                # market cap nu
                mc_now = info.get("marketCap")
                try:
                    mc_now = float(mc_now or 0.0)
                except Exception:
                    mc_now = 0.0
                if mc_now <= 0 and (shares_used > 0) and (price or 0) > 0:
                    mc_now = shares_used * float(price or 0)

                # P/S nu
                if mc_now > 0 and ttm_px:
                    end0, ttm0 = ttm_px[0]
                    if ttm0 > 0:
                        _safe_set(df, ridx, "P/S", float(mc_now / ttm0), changed, force_stamp=force_stamp)

                # P/S Q1..Q4 historik (implied shares konstant + px på/strax före TTM-slut)
                implied_shares = None
                try:
                    px_now = float(price or 0.0)
                    if (mc_now or 0.0) > 0 and px_now > 0:
                        implied_shares = mc_now / px_now
                except Exception:
                    implied_shares = None
                if (implied_shares is None or implied_shares <= 0) and shares_used > 0:
                    implied_shares = shares_used

                if implied_shares and implied_shares > 0:
                    for idx, (d_end, ttm_rev_px) in enumerate(ttm_px[:4], start=1):
                        if ttm_rev_px and ttm_rev_px > 0:
                            px_hist = _yfi_history_close_on_or_before(tkr, d_end)
                            if px_hist and px_hist > 0:
                                mcap_hist = implied_shares * float(px_hist)
                                _safe_set(df, ridx, f"P/S Q{idx}", float(mcap_hist / ttm_rev_px), changed, force_stamp=force_stamp)
                used_sec = True
                sources_used.append("SEC")
        # om SEC inte gav något går vi vidare
    # 2) FMP P/S fallback om P/S saknas eller <=0
    if _coerce_float(df.at[ridx, "P/S"] if "P/S" in df.columns else 0.0, 0.0) <= 0.0:
        ps_fmp = _fmp_ps_ttm(tkr)
        if ps_fmp and ps_fmp > 0:
            _safe_set(df, ridx, "P/S", float(ps_fmp), changed, force_stamp=force_stamp)
            sources_used.append("FMP P/S")

    # 3) yfinance-fallback för kvartar om P/S Q1..Q4 saknas
    need_hist = any(_coerce_float(df.at[ridx, f"P/S Q{i}"] if f"P/S Q{i}" in df.columns else 0.0, 0.0) <= 0.0 for i in range(1,5))
    if need_hist:
        q_rows_yf = _yfi_quarterly_revenue_rows(tkr)
        ttm_list = _ttm_windows(q_rows_yf, need=4)
        info = _yfi_info(tkr) if not info else info
        mc_now = info.get("marketCap")
        try:
            mc_now = float(mc_now or 0.0)
        except Exception:
            mc_now = 0.0
        if mc_now <= 0:
            px_now = float(price or 0.0)
            so     = float(yfi_shares or 0.0)
            if px_now > 0 and so > 0:
                mc_now = px_now * so

        implied_shares = None
        try:
            px_now = float(price or 0.0)
            if (mc_now or 0.0) > 0 and px_now > 0:
                implied_shares = mc_now / px_now
        except Exception:
            implied_shares = None
        if (implied_shares is None or implied_shares <= 0) and (yfi_shares or 0) > 0:
            implied_shares = float(yfi_shares)

        if implied_shares and implied_shares > 0 and ttm_list:
            for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
                cur_val = _coerce_float(df.at[ridx, f"P/S Q{idx}"] if f"P/S Q{idx}" in df.columns else 0.0, 0.0)
                if cur_val <= 0.0 and ttm_rev and ttm_rev > 0:
                    px_hist = _yfi_history_close_on_or_before(tkr, d_end)
                    if px_hist and px_hist > 0:
                        mcap_hist = implied_shares * float(px_hist)
                        _safe_set(df, ridx, f"P/S Q{idx}", float(mcap_hist / ttm_rev), changed, force_stamp=force_stamp)
            sources_used.append("yfinance(hist)")

    # Notera källa + härledda
    _note_auto_update(df, ridx, AUTO_SOURCE_SEC_FMP)
    _recompute_row_derivatives(df, ridx)

    src_txt = " + ".join(sources_used) if sources_used else "yfinance"
    msg = f"{tkr}: uppdaterade {', '.join(changed) if changed else 'inga fält (oförändrat)'} • Källor: {src_txt}."
    return df, changed, msg
