# -*- coding: utf-8 -*-
"""
stockapp/sources.py

Runner-funktioner för batch och on-demand uppdateringar.

Publika:
    runner_price_only_yf(ticker, df, user_rates) -> (df, changed_fields, source_label)
    runner_full_combo(ticker, df, user_rates)    -> (df, changed_fields, source_label)

Not:
- Rör inte "Omsättning idag" / "Omsättning nästa år" (manuellt enligt krav).
- Sätter "Senast auto-uppdaterad" + "Senast uppdaterad källa".
- TS-stämplar för: Utestående aktier, P/S, P/S Q1..Q4.
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Any
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import yfinance as yf

# ------------------------------ TS-kolumner ----------------------------------

TS_FIELDS: Dict[str, str] = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",          # lämnas orörd av runners
    "Omsättning nästa år": "TS_Omsättning nästa år",  # lämnas orörd av runners
}

# ------------------------------ Hjälpare -------------------------------------

def _now_date_str() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Europe/Stockholm")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _safe_float(v, default=np.nan) -> float:
    try:
        if v is None:
            return default
        fv = float(v)
        # vissa yfinance-fält är "nan" som float
        if isinstance(fv, float) and np.isnan(fv):
            return default
        return fv
    except Exception:
        return default

def _set_if_meaningful(df: pd.DataFrame, ridx: int, field: str, value, changed: List[str]):
    """
    Sätter df[ridx, field] om value är "meningsfullt".
    Regler:
      - Sträng: måste vara icke-tom
      - Tal: > 0 för P/S, P/S Q*, Utestående aktier; >= 0 för övriga numeriska
    TS-stämplar om fältet är spårat.
    """
    if field not in df.columns:
        return
    ok = False
    if isinstance(value, str):
        if value.strip():
            ok = True
    else:
        fv = _safe_float(value, np.nan)
        if field in ("P/S","Utestående aktier","P/S Q1","P/S Q2","P/S Q3","P/S Q4"):
            ok = (not np.isnan(fv)) and (fv > 0)
        else:
            ok = (not np.isnan(fv)) and (fv >= 0)

    if not ok:
        return

    old = df.at[ridx, field] if field in df.columns else None
    # Skriv alltid (även om samma) för att vi vill kunna stämpla "Senast auto-uppdaterad"
    df.at[ridx, field] = value
    changed.append(field)

    # TS-stämpel per fält
    ts_col = TS_FIELDS.get(field)
    if ts_col and ts_col in df.columns:
        df.at[ridx, ts_col] = _now_date_str()

def _note_auto_header(df: pd.DataFrame, ridx: int, source_label: str):
    if "Senast auto-uppdaterad" in df.columns:
        df.at[ridx, "Senast auto-uppdaterad"] = _now_date_str()
    if "Senast uppdaterad källa" in df.columns:
        df.at[ridx, "Senast uppdaterad källa"] = source_label

def _implied_shares(mcap: float, price: float) -> float:
    if _safe_float(mcap, np.nan) > 0 and _safe_float(price, np.nan) > 0:
        return float(mcap) / float(price)
    return np.nan

# ---------- Yahoo pipeline: kurs, namn, valuta + nyckeltal + P/S historik ----

def _yahoo_prices_for_dates(ticker: str, dates: List[pd.Timestamp]) -> Dict[pd.Timestamp, float]:
    """
    Dagliga stängningar runt efterfrågade datum; returnerar priset på/närmast FÖRE datum.
    """
    out: Dict[pd.Timestamp, float] = {}
    if not dates:
        return out
    start = min(dates) - pd.Timedelta(days=14)
    end   = max(dates) + pd.Timedelta(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start.to_pydatetime(), end=end.to_pydatetime(), interval="1d", auto_adjust=False)
        if hist is None or hist.empty:
            return out
        hist = hist.sort_index()
        idx_dates = [pd.Timestamp(d).tz_localize(None) for d in hist.index]  # pandas Timestamp
        closes = list(hist["Close"].values)
        for d in dates:
            d_naive = pd.Timestamp(d).tz_localize(None)
            px = None
            for j in range(len(idx_dates)-1, -1, -1):
                if idx_dates[j] <= d_naive:
                    try:
                        px = float(closes[j])
                    except Exception:
                        px = None
                    break
            if px is not None:
                out[pd.Timestamp(d).normalize()] = px
    except Exception:
        return out
    return out

def _yfi_quarterly_revenues(t: yf.Ticker) -> List[Tuple[pd.Timestamp, float]]:
    """
    Hämtar kvartalsintäkter från yfinance.quarterly_financials om möjligt.
    Returnerar [(period_end, value), ...] nyast→äldst.
    """
    # 1) quarterly_financials – vanligast
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand = [
                "Total Revenue","TotalRevenue","Revenues","Revenue","Sales",
                "Total revenue","Revenues from contracts with customers"
            ]
            for key in cand:
                if key in idx:
                    row = qf.loc[key].dropna()
                    out = []
                    for c, v in row.items():
                        try:
                            d = pd.Timestamp(c).normalize()
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) fallback income_stmt (kvartal) om det råkar finnas i denna yfinance-version
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = pd.Timestamp(c).normalize()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

def _ttm_windows(values: List[Tuple[pd.Timestamp, float]], need: int = 4) -> List[Tuple[pd.Timestamp, float]]:
    """
    Gör TTM-summor av kvartalsrader (nyast→äldst).
    """
    out: List[Tuple[pd.Timestamp, float]] = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values)-3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _fetch_fields_via_yahoo(ticker: str) -> Dict[str, Any]:
    """
    Hämtar fält via Yahoo Finance.
    Rör inte 'Omsättning idag' / 'Omsättning nästa år'.

    Returnerar en dict med potentiella fält (skrivs sen in med _set_if_meaningful).
    """
    out: Dict[str, Any] = {}
    t = yf.Ticker(ticker)

    # --- info-basics
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # Namn / valuta / kurs
    name = info.get("shortName") or info.get("longName")
    if name: out["Bolagsnamn"] = str(name)
    ccy  = info.get("currency")
    if ccy: out["Valuta"] = str(ccy).upper()

    price = info.get("regularMarketPrice")
    if price is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            price = None
    if _safe_float(price, np.nan) > 0:
        out["Aktuell kurs"] = float(price)

    # Sektor/industri
    if info.get("sector"):
        out["Sektor"] = str(info.get("sector"))
    if info.get("industry"):
        out["Industri"] = str(info.get("industry"))

    # Utdelning / yield
    div_rate = _safe_float(info.get("dividendRate"), np.nan)  # per år, i bolagets valuta
    if not np.isnan(div_rate):
        out["Årlig utdelning"] = float(div_rate)
    dy = _safe_float(info.get("dividendYield"), np.nan)
    if not np.isnan(dy):
        out["Utdelningsyield"] = float(dy * 100.0)

    # Marginaler / skuldsättning / kassa / multiplar (om finns i info)
    gm = _safe_float(info.get("grossMargins"), np.nan)
    if not np.isnan(gm): out["Bruttomarginal"] = float(gm * 100.0)
    pm = _safe_float(info.get("profitMargins"), np.nan)
    if not np.isnan(pm): out["Nettomarginal"] = float(pm * 100.0)

    de = _safe_float(info.get("debtToEquity"), np.nan)
    if not np.isnan(de): out["Debt/Equity"] = float(de)

    total_cash = _safe_float(info.get("totalCash"), np.nan)
    if not np.isnan(total_cash): out["Kassa"] = float(total_cash)

    ev_ebitda = _safe_float(info.get("enterpriseToEbitda"), np.nan)
    if not np.isnan(ev_ebitda): out["EV/EBITDA"] = float(ev_ebitda)

    # Marketcap / shares
    mcap_now = _safe_float(info.get("marketCap"), np.nan)
    if _safe_float(price, np.nan) > 0:
        implied = _implied_shares(mcap_now, price)
    else:
        implied = np.nan

    if implied > 0:
        out["Utestående aktier"] = float(implied) / 1e6  # i miljoner

    # P/S (nu) och historik via TTM-fönster
    q_rows = _yfi_quarterly_revenues(t)
    if len(q_rows) >= 4 and _safe_float(price, np.nan) > 0:
        # TTM listor nyast→
        ttm_list = _ttm_windows(q_rows, need=4)
        # Historiska priser på TTM-slut
        q_dates = [d for (d, _) in ttm_list]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)

        # P/S (nu): om marketcap och ttm[0]
        if mcap_now > 0 and ttm_list:
            ltm_now = ttm_list[0][1]
            if ltm_now > 0:
                out["P/S"] = float(mcap_now / ltm_now)

        # P/S Q1..Q4
        if implied > 0:
            for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
                px = px_map.get(pd.Timestamp(d_end).normalize())
                if px and ttm_rev and ttm_rev > 0:
                    mcap_hist = float(implied) * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev)
                # Spara också slutdatum om du vill presentera det i vyer
                out[f"Periodslut Q{idx}"] = pd.Timestamp(d_end).strftime("%Y-%m-%d")

    return out

# ------------------------------ RUNNERS --------------------------------------

def runner_price_only_yf(ticker: str,
                         df: pd.DataFrame,
                         user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Uppdaterar endast kurs (och gärna valuta/namn om saknas) via yfinance.
    Returnerar (df, changed_fields, source_label).
    """
    ticker = str(ticker).upper().strip()
    if "Ticker" not in df.columns or ticker not in df["Ticker"].astype(str).str.upper().values:
        # ingen rad att skriva till
        return df, [], "Kurs (Yahoo Finance)"

    ridx = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]
    changed: List[str] = []

    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        price = info.get("regularMarketPrice")
        if price is None:
            try:
                h = t.history(period="1d")
                if not h.empty and "Close" in h:
                    price = float(h["Close"].iloc[-1])
            except Exception:
                price = None
        _set_if_meaningful(df, ridx, "Aktuell kurs", price, changed)

        # Valuta & namn (endast om saknas)
        if _safe_float(price, np.nan) > 0:
            if str(df.at[ridx, "Valuta"]) in ("", "nan", "None", "NaN") if "Valuta" in df.columns else True:
                ccy = info.get("currency")
                if ccy:
                    _set_if_meaningful(df, ridx, "Valuta", str(ccy).upper(), changed)
            if str(df.at[ridx, "Bolagsnamn"]) in ("", "nan", "None", "NaN") if "Bolagsnamn" in df.columns else True:
                name = info.get("shortName") or info.get("longName")
                if name:
                    _set_if_meaningful(df, ridx, "Bolagsnamn", str(name), changed)

        _note_auto_header(df, ridx, "Kurs (Yahoo Finance)")
    except Exception:
        # lämna changed tom om det inte gick
        pass

    return df, changed, "Kurs (Yahoo Finance)"


def runner_full_combo(ticker: str,
                      df: pd.DataFrame,
                      user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Full uppdatering (utan att röra manuella prognos-fält).
    Primärt Yahoo-baserad; hook: om du lägger en provider i st.session_state["provider_full_fetch"]
    (callable: ticker -> dict), så används den istället.

    Returnerar (df, changed_fields, source_label).
    """
    ticker = str(ticker).upper().strip()
    if "Ticker" not in df.columns or ticker not in df["Ticker"].astype(str).str.upper().values:
        return df, [], "Auto (Yahoo)"

    ridx = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]
    changed: List[str] = []

    # 1) Provider-hook (om du senare vill köra SEC/FMP/Finhub)
    provider = st.session_state.get("provider_full_fetch")
    if callable(provider):
        try:
            fetched = provider(ticker)  # förväntas vara dict med samma fältnamn som df
        except Exception:
            fetched = {}
    else:
        fetched = _fetch_fields_via_yahoo(ticker)

    # 2) Skriv in fält (OBS: rör ej de manuella prognosfälten!)
    ignore_fields = {"Omsättning idag", "Omsättning nästa år"}
    for k, v in (fetched or {}).items():
        if k in ignore_fields:
            continue
        _set_if_meaningful(df, ridx, k, v, changed)

    # 3) Header-källa
    _note_auto_header(df, ridx, "Auto (Yahoo)")

    # 4) Return
    return df, changed, "Auto (Yahoo)"
