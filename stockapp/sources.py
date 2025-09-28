# stockapp/sources.py
# -*- coding: utf-8 -*-
"""
Hämtare av kurser & nyckeltal via yfinance (lagliga gratis-källor).

Publika "runner"-funktioner:
- run_update_price_only(ticker, *_, **__) -> (vals: dict, debug: dict)
- run_update_full(ticker, *_, **__)       -> (vals: dict, debug: dict)

OBS: Sätter INTE 'Omsättning idag' / 'Omsättning nästa år' (manuella fält).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# -------------------------------------------------------------
# Små hjälpare
# -------------------------------------------------------------
def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _yfi_info(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}


def _to_date(obj) -> Optional[date]:
    try:
        if hasattr(obj, "date"):
            return obj.date()
        return pd.to_datetime(obj).date()
    except Exception:
        return None


# -------------------------------------------------------------
# Yahoo-basics
# -------------------------------------------------------------
def _fetch_yahoo_basics(ticker: str) -> Tuple[dict, dict]:
    """Hämtar pris, namn, valuta, market cap, shares, sektor/industri, utdelning."""
    dbg = {}
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "Dividend Yield (%)": 0.0,
        "MarketCap (nu)": 0.0,
        "Sektor": "",
        "Industri": "",
        # "Utestående aktier" (miljoner) sätts strax
    }
    t = yf.Ticker(ticker)
    info = _yfi_info(t)
    dbg["info_keys"] = list(info.keys())[:40]

    # Namn/valuta
    out["Bolagsnamn"] = str(info.get("shortName") or info.get("longName") or "")[:256]
    out["Valuta"] = str(info.get("currency") or "USD").upper()

    # Pris
    # primärt regularMarketPrice -> fallback dagens "Close"
    px = info.get("regularMarketPrice", None)
    if px is None:
        try:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        except Exception:
            px = None
    if px is not None:
        out["Aktuell kurs"] = float(px)

    # Market cap & implied shares
    mcap = info.get("marketCap", None)
    if mcap is None:
        # fallback via sharesOutstanding*price
        so = info.get("sharesOutstanding", None)
        if so and out["Aktuell kurs"] > 0:
            mcap = float(so) * float(out["Aktuell kurs"])
    out["MarketCap (nu)"] = float(mcap or 0.0)

    shares = info.get("sharesOutstanding", None)
    if not shares and out["MarketCap (nu)"] > 0 and out["Aktuell kurs"] > 0:
        # implied
        shares = float(out["MarketCap (nu)"]) / float(out["Aktuell kurs"])
        dbg["shares_source"] = "implied(mcap/price)"
    else:
        dbg["shares_source"] = "info.sharesOutstanding" if shares else "unknown"

    if shares and float(shares) > 0:
        out["Utestående aktier"] = float(shares) / 1e6  # vi lagrar i miljoner

    # Sektor/industri
    out["Sektor"] = str(info.get("sector", "") or "")
    out["Industri"] = str(info.get("industry", "") or "")

    # Utdelning/Dividend yield
    # dividendRate = årlig summa per aktie (ca), dividendYield = direktavkastning (ratio, ex 0.024 = 2.4%)
    div_rate = info.get("dividendRate", None)
    if div_rate is not None:
        out["Årlig utdelning"] = _safe_float(div_rate, 0.0)

    dy = info.get("dividendYield", None)
    if dy is not None:
        out["Dividend Yield (%)"] = float(dy) * 100.0

    return out, dbg


# -------------------------------------------------------------
# Kvartalsintäkter, TTM & historiska priser
# -------------------------------------------------------------
def _quarterly_revenues(t: yf.Ticker) -> List[Tuple[date, float]]:
    """
    Försöker läsa kvartalsintäkter: [(period_end_date, revenue)], nyast→äldst.
    Hämtar från t.quarterly_financials. Fixar ibland 'årsskiftet' (dec/jan).
    """
    try:
        qf = t.quarterly_financials
        if not isinstance(qf, pd.DataFrame) or qf.empty:
            return []
        # hitta rad för intäkter
        candidates = [
            "Total Revenue", "TotalRevenue", "Revenues", "Revenue", "Sales",
            "Revenues from contracts with customers", "Revenues From Contracts With Customers"
        ]
        row = None
        for rname in candidates:
            if rname in qf.index:
                row = qf.loc[rname].dropna()
                break
        if row is None or row.empty:
            return []
        out: List[Tuple[date, float]] = []
        for c, v in row.items():
            d = _to_date(c)
            if d and v is not None:
                out.append((d, float(v)))
        out.sort(key=lambda x: x[0], reverse=True)

        # Deduplicera “nära” dubletter (ex jan/feb på samma Q). Behåll nyare.
        ded: List[Tuple[date, float]] = []
        prev: Optional[Tuple[date, float]] = None
        for d, val in out:
            if prev is None:
                ded.append((d, val)); prev = (d, val); continue
            # om tidigare är inom 40 dagar => hoppa äldre
            if (prev[0] - d).days <= 40:
                # ignore this older
                continue
            ded.append((d, val))
            prev = (d, val)
        return ded
    except Exception:
        return []


def _ttm_windows(values: List[Tuple[date, float]], need: int = 4) -> List[Tuple[date, float]]:
    """
    Bygger TTM-summor över glidande fönster av 4 kvartal:
    [(end_date0, ttm0), (end_date1, ttm1), ...], nyast→äldst.
    """
    out: List[Tuple[date, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out


def _history_close_map(ticker: str, dates: List[date]) -> Dict[date, float]:
    """
    Returnerar stängningskurs på eller närmast FÖRE varje datum i 'dates'.
    """
    if not dates:
        return {}
    dmin = min(dates) - timedelta(days=14)
    dmax = max(dates) + timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if not isinstance(hist, pd.DataFrame) or hist.empty or "Close" not in hist:
            return {}
        hist = hist.sort_index()
        idx = list(pd.to_datetime(hist.index).date)
        closes = list(hist["Close"].astype(float).values)
        out: Dict[date, float] = {}
        for want in dates:
            px = None
            for j in range(len(idx) - 1, -1, -1):
                if idx[j] <= want:
                    px = float(closes[j]); break
            if px is not None:
                out[want] = px
        return out
    except Exception:
        return {}


def _ps_series_from_ttm_and_prices(
    shares_abs: float,
    ttm_list: List[Tuple[date, float]],
    px_map: Dict[date, float],
) -> Dict[str, float]:
    """
    Bygger P/S Q1..Q4 från TTM-lista och historiska priser.
    """
    out: Dict[str, float] = {}
    if shares_abs <= 0:
        return out
    for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
        px = _safe_float(px_map.get(d_end))
        if px > 0 and ttm_rev > 0:
            mcap_hist = shares_abs * px
            out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev)
    return out


def _mcap_series_from_prices(
    shares_abs: float,
    px_map: Dict[date, float],
    dates: List[date],
) -> Dict[str, float]:
    """
    Bygger MCAP Q1..Q4 (absoluta nivåer) vid kvartalsslut.
    """
    out: Dict[str, float] = {}
    if shares_abs <= 0:
        return out
    for idx, d in enumerate(dates[:4], start=1):
        px = _safe_float(px_map.get(d))
        if px > 0:
            out[f"MCAP Q{idx}"] = float(shares_abs * px)
    return out


# -------------------------------------------------------------
# Lönsamhet/kapitalstruktur via yfinance statements
# -------------------------------------------------------------
def _derive_ratios_financials(t: yf.Ticker) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Hämtar grova nyckeltal: gross/net margin, D/E, cash, FCF, EV/EBITDA om möjligt.
    """
    out: Dict[str, float] = {}
    dbg: Dict[str, str] = {}

    info = _yfi_info(t)

    # --- Margins (senaste årsdata om finns) ---
    try:
        is_annual = t.financials  # yearly income statement
        if isinstance(is_annual, pd.DataFrame) and not is_annual.empty:
            # plocka senaste kolumn (senast rapporterade år)
            col = is_annual.columns[0]
            rev = _safe_float(is_annual.loc.get("Total Revenue", [np.nan])[0] if "Total Revenue" in is_annual.index else np.nan)
            gp  = _safe_float(is_annual.loc.get("Gross Profit",  [np.nan])[0] if "Gross Profit"  in is_annual.index else np.nan)
            ni  = _safe_float(is_annual.loc.get("Net Income",    [np.nan])[0] if "Net Income"    in is_annual.index else np.nan)
            if rev > 0:
                if gp > 0:
                    out["Bruttomarginal (%)"] = gp / rev * 100.0
                if ni != 0:
                    out["Nettomarginal (%)"] = ni / rev * 100.0
            dbg["is_year_col"] = str(col)
    except Exception:
        pass

    # --- Balance sheet: Debt/Equity, Cash ---
    try:
        bs = t.balance_sheet
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            col = bs.columns[0]
            debt = 0.0
            for key in ["Total Debt", "Long Term Debt", "Short Long Term Debt", "Short Term Debt", "ShortLongTermDebtTotal", "Short Term Borrowings"]:
                if key in bs.index and not pd.isna(bs.loc[key, col]):
                    debt += _safe_float(bs.loc[key, col])
            eq = _safe_float(bs.loc.get("Total Stockholder Equity", [np.nan])[0] if "Total Stockholder Equity" in bs.index else np.nan)
            if eq != 0:
                out["Debt/Equity"] = debt / eq
            cash = 0.0
            for key in ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash And Short Term Investments"]:
                if key in bs.index and not pd.isna(bs.loc[key, col]):
                    cash = _safe_float(bs.loc[key, col]); break
            if cash > 0:
                out["Kassa (senast årsdata)"] = cash
            dbg["bs_year_col"] = str(col)
    except Exception:
        pass

    # --- Cashflow: FCF ---
    try:
        cf = t.cashflow
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            col = cf.columns[0]
            ocf = _safe_float(cf.loc.get("Total Cash From Operating Activities", [np.nan])[0] if "Total Cash From Operating Activities" in cf.index else np.nan)
            capex = _safe_float(cf.loc.get("Capital Expenditures", [0.0])[0] if "Capital Expenditures" in cf.index else 0.0)
            fcf = ocf - abs(capex)
            if fcf != 0:
                out["Free Cash Flow"] = fcf
            dbg["cf_year_col"] = str(col)
    except Exception:
        pass

    # --- EV/EBITDA (försök via info) ---
    try:
        ev = _safe_float(info.get("enterpriseValue", 0.0))
        ebitda = _safe_float(info.get("ebitda", 0.0))
        if ev > 0 and ebitda > 0:
            out["EV/EBITDA"] = ev / ebitda
        else:
            # ev/ebitda direkt?
            ev_ebitda = _safe_float(info.get("enterpriseToEbitda", 0.0))
            if ev_ebitda > 0:
                out["EV/EBITDA"] = ev_ebitda
    except Exception:
        pass

    return out, dbg


# -------------------------------------------------------------
# Runners (anropas av batch/knappar)
# -------------------------------------------------------------
def run_update_price_only(ticker: str, *_, **__) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Hämtar endast bas: pris/valuta/namn/market cap. Skriver inte manuella fält.
    """
    vals: Dict[str, object] = {}
    debug: Dict[str, object] = {"runner": "price_only", "ticker": ticker}

    basics, dbg = _fetch_yahoo_basics(ticker)
    vals.update({k: v for k, v in basics.items() if k in ["Bolagsnamn", "Valuta", "Aktuell kurs", "MarketCap (nu)"]})
    # Om shares fanns i basics, ta med (kan hjälpa senare beräkningar)
    if "Utestående aktier" in basics and basics["Utestående aktier"]:
        vals["Utestående aktier"] = basics["Utestående aktier"]

    debug["basics"] = dbg
    return vals, debug


def run_update_full(ticker: str, *_, **__) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Full uppdatering via Yahoo:
    - pris/valuta/namn/market cap/sector/industry/utdelning
    - Utestående aktier (implied/info)
    - kvartalsintäkter (senaste), TTM-intäkter
    - P/S (nu) + P/S Q1..Q4, MCAP Q1..Q4
    - nyckeltal: bruttomarginal, nettomarginal, D/E, Kassa, FCF, EV/EBITDA
    OBS: Uppdaterar INTE 'Omsättning idag' / 'Omsättning nästa år'.
    """
    vals: Dict[str, object] = {}
    debug: Dict[str, object] = {"runner": "full", "ticker": ticker}

    # 1) Basics
    basics, dbg = _fetch_yahoo_basics(ticker)
    vals.update(basics)
    debug["basics"] = dbg

    # 2) Kvartalsintäkter -> TTM
    t = yf.Ticker(ticker)
    qrows = _quarterly_revenues(t)  # [(date, rev)], nyast→äldst
    debug["quarters_found"] = [(str(d), float(v)) for (d, v) in qrows[:6]]
    ttm_list = _ttm_windows(qrows, need=6)  # bygg fler än 4 så vi kan hantera “dec/jan”
    debug["ttm_list"] = [(str(d), float(v)) for (d, v) in ttm_list[:6]]

    # 3) P/S TTM (nu)
    mcap_now = _safe_float(vals.get("MarketCap (nu)", 0.0))
    if mcap_now > 0 and ttm_list:
        ltm_now = _safe_float(ttm_list[0][1])
        if ltm_now > 0:
            vals["P/S"] = float(mcap_now / ltm_now)

    # 4) P/S Q1..Q4 från historik (shares * pris vid kvartalsdatumen / TTM den dagen)
    shares_abs = 0.0
    if "Utestående aktier" in vals and _safe_float(vals["Utestående aktier"]) > 0:
        shares_abs = float(vals["Utestående aktier"]) * 1e6
    elif mcap_now > 0 and _safe_float(vals.get("Aktuell kurs", 0.0)) > 0:
        shares_abs = float(mcap_now / float(vals["Aktuell kurs"]))
        debug["shares_source_fallback"] = "implied(mcap/price)"

    if ttm_list:
        dates = [d for (d, _) in ttm_list]
        px_map = _history_close_map(ticker, dates)
        ps_hist = _ps_series_from_ttm_and_prices(shares_abs, ttm_list, px_map)
        vals.update(ps_hist)

        # MCAP Q1..Q4
        mcap_hist = _mcap_series_from_prices(shares_abs, px_map, dates)
        vals.update(mcap_hist)

    # 5) Marginaler/skuldsättning/FCF/EV-EBITDA
    ratios, dbg_ratios = _derive_ratios_financials(t)
    vals.update(ratios)
    debug["ratios_src"] = dbg_ratios

    # 6) P/S-snitt (positiva)
    ps_q = [vals.get(f"P/S Q{i}", 0.0) for i in range(1, 5)]
    ps_clean = [float(x) for x in ps_q if _safe_float(x) > 0]
    vals["P/S-snitt"] = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0

    # 7) Return
    return vals, debug
