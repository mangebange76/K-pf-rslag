# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yfinance as yf

from .config import TS_FIELDS

# ====== Små helpers ==========================================================

def _now_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def _find_row_index(df: pd.DataFrame, ticker: str):
    mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
    idxs = df.index[mask].tolist()
    return idxs[0] if idxs else None

def _stamp_ts(df: pd.DataFrame, ridx: int, field: str, when: str | None = None):
    ts_col = TS_FIELDS.get(field)
    if ts_col and ts_col in df.columns:
        df.at[ridx, ts_col] = when or _now_date_str()

def _note_auto(df: pd.DataFrame, ridx: int, source: str):
    if "Senast auto-uppdaterad" in df.columns:
        df.at[ridx, "Senast auto-uppdaterad"] = _now_date_str()
    if "Senast uppdaterad källa" in df.columns:
        df.at[ridx, "Senast uppdaterad källa"] = source

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _fmt_changes(changed: list[str]) -> str:
    return ", ".join(changed) if changed else "inga fält"

# ====== Yahoo helpers ========================================================

def _yfi_info(tkr: yf.Ticker) -> dict:
    try:
        return tkr.info or {}
    except Exception:
        return {}

def _yfi_price_currency_name_div(ticker: str) -> dict:
    out = {"Aktuell kurs": 0.0, "Valuta": "USD", "Bolagsnamn": "", "Årlig utdelning": 0.0}
    t = yf.Ticker(ticker)

    info = _yfi_info(t)
    # Price
    px = info.get("regularMarketPrice")
    if px is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h.columns:
                px = float(h["Close"].iloc[-1])
        except Exception:
            px = None
    if px is not None:
        out["Aktuell kurs"] = _safe_float(px, 0.0)

    # Currency & name
    ccy = info.get("currency") or "USD"
    out["Valuta"] = str(ccy).upper()
    out["Bolagsnamn"] = info.get("shortName") or info.get("longName") or ""

    # Dividend (annualized)
    div_rate = info.get("dividendRate")
    if div_rate is not None:
        out["Årlig utdelning"] = _safe_float(div_rate, 0.0)

    return out

def _yfi_marketcap_and_shares(ticker: str, price_hint: float | None = None) -> tuple[float, float]:
    """Return (market_cap, shares) in absolute units (shares in *pieces*)."""
    t = yf.Ticker(ticker)
    info = _yfi_info(t)

    mcap = _safe_float(info.get("marketCap"), 0.0)
    px = price_hint if price_hint is not None else _safe_float(info.get("regularMarketPrice"), 0.0)

    shares = 0.0
    so = _safe_float(info.get("sharesOutstanding"), 0.0)
    if so > 0:
        shares = so
    elif mcap > 0 and px > 0:
        shares = mcap / px

    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px

    return mcap, shares

def _quarterly_revenue_rows(ticker: str) -> list[tuple[date, float]]:
    """[(period_end_date, revenue)], newest→oldest"""
    t = yf.Ticker(ticker)

    # Preferred: quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            for key in [
                "Total Revenue", "TotalRevenue", "Revenues", "Revenue",
                "Sales", "Revenues from contracts with customers"
            ]:
                if key in qf.index:
                    row = qf.loc[key].dropna()
                    out = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            out.append((d, _safe_float(v, 0.0)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    if out:
                        return out
    except Exception:
        pass

    # Fallback: income_stmt
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    out.append((d, _safe_float(v, 0.0)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            if out:
                return out
    except Exception:
        pass

    return []

def _ttm_windows(values: list[tuple[date, float]], need: int = 4) -> list[tuple[date, float]]:
    """values is [(end_date, quarterly_val)] newest→oldest; build up to `need` TTM sums."""
    out = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _prices_on_or_before(ticker: str, dates: list[date]) -> dict[date, float]:
    """Get Close on or before each date."""
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        idx_dates = list(hist.index.date)
        closes = list(hist["Close"].values)
        out = {}
        for d in dates:
            px = None
            for j in range(len(idx_dates) - 1, -1, -1):
                if idx_dates[j] <= d:
                    try:
                        px = float(closes[j])
                    except Exception:
                        px = None
                    break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}

# ====== Public runners =======================================================

def run_update_price_only(df: pd.DataFrame, user_rates: dict, ticker: str, **kwargs):
    """
    Minimal runner: uppdaterar endast Aktuell kurs (+ namn/valuta om saknas),
    stämplar 'Senast auto-uppdaterad' och källa.

    Returnerar: (df, changed_fields_list, message)
    """
    ridx = _find_row_index(df, ticker)
    if ridx is None:
        # tom changed-lista, felmeddelande
        return df, [], f"{ticker}: hittades inte i tabellen."

    base = _yfi_price_currency_name_div(ticker)
    changed: list[str] = []

    for k in ["Aktuell kurs", "Bolagsnamn", "Valuta", "Årlig utdelning"]:
        if k in df.columns and base.get(k) is not None:
            before = df.at[ridx, k] if k in df.columns else None
            df.at[ridx, k] = base[k]
            if str(before) != str(base[k]):
                changed.append(k)

    _note_auto(df, ridx, source="Pris (Yahoo)")
    msg = f"{ticker}: uppdaterade { _fmt_changes(changed) }."
    return df, changed, msg

def run_update_full(df: pd.DataFrame, user_rates: dict, ticker: str, force_stamp: bool = True, **kwargs):
    """
    Full runner (Yahoo-baserad, gratis): uppdaterar pris, namn, valuta, utdelning,
    utestående aktier (implied), P/S (TTM) nu och P/S Q1–Q4, P/S-snitt.
    Lämnar 'Omsättning idag/nästa år' orörda (manuell policy).

    Returnerar: (df, changed_fields_list, message)
    """
    ridx = _find_row_index(df, ticker)
    if ridx is None:
        return df, [], f"{ticker}: hittades inte i tabellen."

    base = _yfi_price_currency_name_div(ticker)
    px = float(base.get("Aktuell kurs", 0.0))
    mcap_now, shares_abs = _yfi_marketcap_and_shares(ticker, price_hint=px)

    changed: list[str] = []

    # Basfält
    for k in ["Aktuell kurs", "Bolagsnamn", "Valuta", "Årlig utdelning"]:
        if k in df.columns and base.get(k) is not None:
            before = df.at[ridx, k] if k in df.columns else None
            df.at[ridx, k] = base[k]
            if str(before) != str(base[k]):
                changed.append(k)

    # Utestående aktier (miljoner)
    if shares_abs > 0 and "Utestående aktier" in df.columns:
        before = df.at[ridx, "Utestående aktier"]
        df.at[ridx, "Utestående aktier"] = round(shares_abs / 1e6, 6)
        if force_stamp:
            _stamp_ts(df, ridx, "Utestående aktier")
        elif _safe_float(before) != _safe_float(df.at[ridx, "Utestående aktier"]):
            _stamp_ts(df, ridx, "Utestående aktier")
        if str(before) != str(df.at[ridx, "Utestående aktier"]):
            changed.append("Utestående aktier")

    # P/S nu + historik
    q_rows = _quarterly_revenue_rows(ticker)
    if q_rows:
        ttm_list = _ttm_windows(q_rows, need=4)  # [(end_date, ttm)]
        q_dates = [d for (d, _) in ttm_list]
        px_map = _prices_on_or_before(ticker, q_dates)

        # P/S (nu)
        if mcap_now > 0 and ttm_list and "P/S" in df.columns:
            ttm0 = float(ttm_list[0][1])
            if ttm0 > 0:
                before = df.at[ridx, "P/S"]
                df.at[ridx, "P/S"] = float(mcap_now) / float(ttm0)
                if force_stamp:
                    _stamp_ts(df, ridx, "P/S")
                elif _safe_float(before) != _safe_float(df.at[ridx, "P/S"]):
                    _stamp_ts(df, ridx, "P/S")
                if str(before) != str(df.at[ridx, "P/S"]):
                    changed.append("P/S")

        # P/S Q1..Q4 (historik) med samma shares_abs
        if shares_abs > 0:
            for idx_q, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
                p = _safe_float(px_map.get(d_end), 0.0)
                if p > 0 and ttm_rev > 0:
                    mcap_hist = float(shares_abs) * float(p)
                    field = f"P/S Q{idx_q}"
                    if field in df.columns:
                        before = df.at[ridx, field]
                        df.at[ridx, field] = float(mcap_hist) / float(ttm_rev)
                        if force_stamp:
                            _stamp_ts(df, ridx, field)
                        elif _safe_float(before) != _safe_float(df.at[ridx, field]):
                            _stamp_ts(df, ridx, field)
                        if str(before) != str(df.at[ridx, field]):
                            changed.append(field)

    # P/S-snitt
    if "P/S-snitt" in df.columns:
        vals = []
        for q in [1, 2, 3, 4]:
            col = f"P/S Q{q}"
            v = _safe_float(df.at[ridx, col] if col in df.columns else 0.0, 0.0)
            if v > 0:
                vals.append(v)
        df.at[ridx, "P/S-snitt"] = round(float(np.mean(vals)) if vals else 0.0, 2)

    _note_auto(df, ridx, source="Auto (Yahoo)")
    msg = f"{ticker}: uppdaterade { _fmt_changes(changed) }." if changed else f"{ticker}: inga fält ändrades (stämplade ändå tid/källa)."
    return df, changed, msg
