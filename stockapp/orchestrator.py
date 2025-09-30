# stockapp/orchestrator.py
# -*- coding: utf-8 -*-
"""
Samordnar datainsamling från flera källor (Yahoo, SEC, FMP) och uppdaterar DataFrame.
Exponerar:
  - run_update_price(df, ticker, save=False)
  - run_update_full(df, ticker, save=False)
  - run_batch_update(df, tickers, mode="full", progress_cb=None, save=False)
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# Lokala moduler
from .config import TS_FIELDS, FINAL_COLS
from .storage import spara_data
from .utils import now_stamp, ensure_schema, safe_float

# Försök importera fetchers — alla är frivilliga och används när de finns.
# Yahoo är "must-have" – men vi skyddar ändå med fallback mot yfinance direkt.
_yahoo_ok = _sec_ok = _fmp_ok = True
try:
    from .fetchers.yahoo import (
        fetch_basic,                 # (ticker) -> {"Bolagsnamn","Valuta","Aktuell kurs","Sector","Industry"} (subset kan saknas)
        fetch_quarterlies_and_ps,    # (ticker) -> {"P/S":float,"P/S Q1":...,"P/S Q4":...,"Market Cap":float,"Utestående aktier":float_milj}
        fetch_financials_extra,      # (ticker) -> ev {"Debt/Equity","Gross Margin","Net Margin","Cash & Equivalents","Operating CF","Free CF", ...}
    )
except Exception:
    _yahoo_ok = False

try:
    from .fetchers.sec import (
        sec_latest_shares_robust,        # (ticker)-> shares (antal, inte miljoner)
        sec_quarterly_revenues_dated,    # (ticker)-> List[(date, revenue_value, unit)]
        build_ps_from_sec_and_prices,    # (ticker, shares, revenues)-> {"P/S","P/S Q1"..,"P/S Q4"}
    )
except Exception:
    _sec_ok = False

try:
    from .fetchers.fmp import (
        fmp_price_ps_shares,             # (ticker)-> {"Aktuell kurs","P/S","Utestående aktier"(miljoner), "Market Cap"}
        fmp_analyst_revenue_estimates,   # (ticker)-> {"Omsättning idag","Omsättning nästa år"}  (miljoner i prisvaluta)
    )
except Exception:
    _fmp_ok = False

# Absolut sista fallback om Yahoo-modulen saknas:
try:
    import yfinance as yf
except Exception:
    yf = None


# ----------------------------------------------------------------------
# Hjälpare: slå in ändringar + stämpla TS och källa
# ----------------------------------------------------------------------
def _apply_changes(
    df: pd.DataFrame,
    ridx: int,
    new_vals: Dict[str, object],
    source: str,
    stamp_even_if_same: bool = True
) -> List[str]:
    """
    Skriver new_vals till df.iloc[ridx], stämplar TS_* för fält i TS_FIELDS,
    och sätter 'Senast auto-uppdaterad' + 'Senast uppdaterad källa'.
    Returnerar lista över fält som anses ändrade (för logg).
    """
    changed: List[str] = []
    for f, v in new_vals.items():
        if f not in df.columns:
            continue
        old = df.at[ridx, f]
        same = (str(old) == str(v))
        # Vi vill tidsstämpla även om värdet var samma, om stamp_even_if_same=True.
        if (not same) or stamp_even_if_same:
            df.at[ridx, f] = v
            if f in TS_FIELDS:
                df.at[ridx, TS_FIELDS[f]] = now_stamp()
            if not same:
                changed.append(f)

    # Alltid sätt meta
    df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = source
    return changed


def _find_row_index(df: pd.DataFrame, ticker: str) -> Optional[int]:
    m = df.index[df["Ticker"].astype(str).str.upper() == str(ticker).upper()]
    if len(m) == 0:
        return None
    return int(m[0])


# ----------------------------------------------------------------------
# Fallback: bara Yahoo pris/valuta/namn om moduler saknas
# ----------------------------------------------------------------------
def _fallback_yf_basic(ticker: str) -> Dict[str, object]:
    out = {}
    if yf is None:
        return out
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        px = info.get("regularMarketPrice")
        if px is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        if px is not None:
            out["Aktuell kurs"] = float(px)

        cur = info.get("currency")
        if cur:
            out["Valuta"] = str(cur).upper()

        name = info.get("shortName") or info.get("longName")
        if name:
            out["Bolagsnamn"] = str(name)

        sector = info.get("sector")
        industry = info.get("industry")
        if sector: out["Sector"] = str(sector)
        if industry: out["Industry"] = str(industry)
        return out
    except Exception:
        return out


# ----------------------------------------------------------------------
# Publika API
# ----------------------------------------------------------------------
def run_update_price(
    df: pd.DataFrame,
    ticker: str,
    save: bool = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]], str]:
    """
    Uppdaterar enbart pris/valuta/namn (snabb).
    Returnerar (df, changed_map, status_text).
    """
    df = ensure_schema(df)
    ridx = _find_row_index(df, ticker)
    if ridx is None:
        return df, {}, f"Ingen förändring: {ticker} hittades inte i tabellen."

    vals: Dict[str, object] = {}
    source_bits = []

    # Yahoo basic
    try:
        if _yahoo_ok:
            y = fetch_basic(ticker)
            if isinstance(y, dict):
                vals.update({k: v for k, v in y.items() if v not in (None, "", 0, 0.0)})
                source_bits.append("Yahoo")
        else:
            fb = _fallback_yf_basic(ticker)
            if fb:
                vals.update(fb)
                source_bits.append("yfinance")
    except Exception:
        pass

    if not vals:
        return df, {}, f"Inga fält att uppdatera för {ticker}."

    changed = _apply_changes(df, ridx, vals, source="Pris-auto (" + " + ".join(source_bits) + ")")
    if save:
        spara_data(df, do_snapshot=False)

    changed_map = {ticker: changed} if changed else {}
    if not changed:
        return df, changed_map, f"Inga ändringar (pris) för {ticker}."
    return df, changed_map, f"Pris uppdaterat för {ticker}: {', '.join(changed)}."


def run_update_full(
    df: pd.DataFrame,
    ticker: str,
    save: bool = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, object]]:
    """
    Full uppdatering för en ticker:
      - Bas: namn, valuta, aktuell kurs, sector/industry
      - P/S (TTM), P/S Q1..Q4
      - Utestående aktier (miljoner)
      - Market Cap (nu)
      - (om tillgängligt) finansiella nyckeltal, kassaflöden, cash m.m.
      - (om tillgängligt) Omsättning idag/nästa år från FMP
    Returnerar (df, changed_map, debug_info).
    """
    df = ensure_schema(df)
    ridx = _find_row_index(df, ticker)
    if ridx is None:
        return df, {}, {"status": f"{ticker} hittades inte i tabellen."}

    vals: Dict[str, object] = {}
    debug: Dict[str, object] = {"ticker": ticker, "sources": []}

    # 1) Yahoo basic (namn, valuta, pris, sector/industry)
    try:
        if _yahoo_ok:
            yb = fetch_basic(ticker)
            if isinstance(yb, dict):
                vals.update({k: v for k, v in yb.items() if v not in (None, "", 0, 0.0)})
                debug["yahoo_basic"] = yb
                debug["sources"].append("Yahoo basic")
        else:
            fb = _fallback_yf_basic(ticker)
            if fb:
                vals.update(fb)
                debug["yfinance_basic"] = fb
                debug["sources"].append("yfinance basic")
    except Exception as e:
        debug["yahoo_basic_err"] = str(e)

    # 2) SEC (aktier + P/S via SEC-revenues + priser)
    sec_vals: Dict[str, object] = {}
    if _sec_ok:
        try:
            sh = sec_latest_shares_robust(ticker)  # antal aktier (stycken)
            if sh and sh > 0:
                sec_vals["Utestående aktier"] = float(sh) / 1e6  # spara i miljoner
            revs = sec_quarterly_revenues_dated(ticker, want=8)  # senaste 8 för robust fönster
            if revs:
                ps_pack = build_ps_from_sec_and_prices(ticker, shares=sh, revenues=revs)
                for k in ("P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Market Cap"):
                    v = ps_pack.get(k)
                    if v is not None:
                        sec_vals[k] = float(v)
            if sec_vals:
                vals.update({k: v for k, v in sec_vals.items() if v not in (None, "", 0, 0.0)})
                debug["sec"] = sec_vals
                debug["sources"].append("SEC")
        except Exception as e:
            debug["sec_err"] = str(e)

    # 3) FMP (snabb P/S/shares/mcap + ev. analyst-estimates)
    if _fmp_ok:
        try:
            fmpv = fmp_price_ps_shares(ticker)
            if isinstance(fmpv, dict):
                # Skriv endast in om de inte finns från SEC/Yahoo eller är mer rimliga (>0)
                for k in ("Aktuell kurs", "P/S", "Utestående aktier", "Market Cap"):
                    v = fmpv.get(k)
                    if v is not None and float(v) > 0 and (k not in vals or float(vals.get(k, 0)) <= 0):
                        vals[k] = float(v)
                debug["fmp_quote"] = fmpv
                debug["sources"].append("FMP-quote")
        except Exception as e:
            debug["fmp_quote_err"] = str(e)

        try:
            est = fmp_analyst_revenue_estimates(ticker)
            if isinstance(est, dict):
                for k in ("Omsättning idag", "Omsättning nästa år"):
                    v = est.get(k)
                    if v is not None and float(v) > 0:
                        vals[k] = float(v)
                debug["fmp_estimates"] = est
                debug["sources"].append("FMP-estimates")
        except Exception as e:
            debug["fmp_estimates_err"] = str(e)

    # 4) Yahoo kvartal & extra nyckeltal (om kvar saknas)
    try:
        if _yahoo_ok:
            qps = fetch_quarterlies_and_ps(ticker)
            if isinstance(qps, dict):
                # komplettera luckor
                for k in ("P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Market Cap", "Utestående aktier"):
                    v = qps.get(k)
                    if v is not None and float(v) > 0 and (k not in vals or float(vals.get(k, 0)) <= 0):
                        vals[k] = float(v)
                debug["yahoo_qps"] = qps
                debug["sources"].append("Yahoo-PS")
    except Exception as e:
        debug["yahoo_qps_err"] = str(e)

    try:
        if _yahoo_ok and "Sector" not in vals:
            extra = fetch_financials_extra(ticker)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if v is not None and v != "":
                        vals[k] = v
                debug["yahoo_fin_extra"] = extra
                debug["sources"].append("Yahoo-extra")
    except Exception as e:
        debug["yahoo_extra_err"] = str(e)

    # Om vi inte ens fått pris – sista nödfallback
    if "Aktuell kurs" not in vals and yf is not None:
        fb = _fallback_yf_basic(ticker)
        if fb:
            vals.update(fb)
            debug["fallback_yf_added"] = True
            debug["sources"].append("yfinance-fallback")

    if not vals:
        return df, {}, {"status": f"Inga fält uppdaterade för {ticker}.", **debug}

    changed = _apply_changes(df, ridx, vals, source="Auto (SEC/FMP/Yahoo)")
    if save:
        spara_data(df, do_snapshot=False)

    changed_map = {ticker: changed} if changed else {}
    debug["status"] = "ok" if changed else "no-change"
    debug["changed"] = changed
    return df, changed_map, debug


def run_batch_update(
    df: pd.DataFrame,
    tickers: List[str],
    mode: str = "full",                 # "full" eller "price"
    progress_cb: Optional[Callable[[int, int, str, str], None]] = None,
    save: bool = False
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Kör batch-uppdatering för givna tickers.
    progress_cb(i, n, ticker, status) – om satt, anropas för UI (t.ex. "3/20 NVDA – ok")
    Returnerar (df, changed_map, errors_map)
    """
    df = ensure_schema(df)
    total = len(tickers)
    changed_all: Dict[str, List[str]] = {}
    errors: Dict[str, List[str]] = {}

    for i, tkr in enumerate(tickers, start=1):
        status_txt = ""
        try:
            if mode == "price":
                df, chg, status_txt = run_update_price(df, tkr, save=False)
            else:
                df, chg, dbg = run_update_full(df, tkr, save=False)
                status_txt = dbg.get("status", "ok")
            if chg:
                changed_all.update(chg)
        except Exception as e:
            errors.setdefault(tkr, []).append(str(e))
            status_txt = f"fel: {e}"

        if progress_cb:
            try:
                progress_cb(i, total, tkr, status_txt)
            except Exception:
                pass

    if save:
        try:
            spara_data(df, do_snapshot=False)
        except Exception as e:
            # Lägg på errors om det sket sig i save
            errors.setdefault("_save", []).append(str(e))

    return df, changed_all, errors


__all__ = [
    "run_update_price",
    "run_update_full",
    "run_batch_update",
]
