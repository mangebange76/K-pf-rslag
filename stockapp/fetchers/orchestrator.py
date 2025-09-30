# -*- coding: utf-8 -*-
"""
Orkestrerar hämtningar från Yahoo / FMP / SEC och normaliserar resultatet.
Returnerar rad-dict i vårt schema + loggtext.

Publika funktioner:
- run_update_price_only(ticker, user_rates) -> (row_dict, log)
- run_update_full(ticker, user_rates)      -> (row_dict, log)
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Iterable, Optional

import math
import importlib
import traceback

from ..config import FACT_COLS
from ..utils import now_stamp

# ------------------------------------------------------------
# Hjälp
# ------------------------------------------------------------
def _import_optional(modpath: str):
    try:
        return importlib.import_module(modpath)
    except Exception:
        return None

def _pick_first(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def _merge_pref(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Fyll dst med värden från src där dst saknar innehåll."""
    for k, v in (src or {}).items():
        if k not in dst or dst[k] in (None, "", float("nan")):
            dst[k] = v

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace(" ", "").replace(",", ".")
        fx = float(x)
        if math.isnan(fx):
            return None
        return fx
    except Exception:
        return None

def _avg_safe(vals: Iterable[Any]) -> Optional[float]:
    xs = [_to_float(v) for v in vals]
    xs = [v for v in xs if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)

def _risk_label(mcap_usd: Optional[float]) -> Optional[str]:
    if mcap_usd is None:
        return None
    # USD-trösklar (ex. Micro/Small/Mid/Large/Mega)
    if mcap_usd < 300e6:
        return "Micro"
    if mcap_usd < 2e9:
        return "Small"
    if mcap_usd < 10e9:
        return "Mid"
    if mcap_usd < 200e9:
        return "Large"
    return "Mega"

# ------------------------------------------------------------
# Ladda fetchers (valfritt – moduler kan saknas)
# ------------------------------------------------------------
_yahoo = _import_optional("stockapp.fetchers.yahoo")
_fmp   = _import_optional("stockapp.fetchers.fmp")
_sec   = _import_optional("stockapp.fetchers.sec")

# För att undvika hårda funktionsnamn använder vi "duck typing": vi testar flera namn
def _call_fetcher(mod, candidates, *args, **kwargs):
    if mod is None:
        return None
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                return fn(*args, **kwargs)
            except Exception:
                # swallow – vi har andra källor
                pass
    return None


# ------------------------------------------------------------
# Publika API
# ------------------------------------------------------------
def run_update_price_only(ticker: str, user_rates: Dict[str, float]) -> Tuple[Dict, str]:
    """
    Hämtar endast senast betald kurs (+ ev. valuta) och sätter TS Kurs.
    """
    log_lines = [f"[{ticker}] PRICE-ONLY"]
    out: Dict[str, Any] = {"Ticker": ticker}

    y = _call_fetcher(
        _yahoo,
        ["fetch_price", "get_price", "yahoo_price"],
        ticker,
    )
    if isinstance(y, dict):
        out["Kurs"] = _to_float(y.get("Kurs"))
        out["Valuta"] = y.get("Valuta") or y.get("Currency")
        log_lines.append("Yahoo: pris OK")
    elif isinstance(y, (int, float)):
        out["Kurs"] = _to_float(y)
        log_lines.append("Yahoo: pris OK (num)")
    else:
        log_lines.append("Yahoo: pris MISS")

    # fallback FMP
    if out.get("Kurs") is None:
        f = _call_fetcher(_fmp, ["fetch_price", "get_price"], ticker)
        if isinstance(f, dict):
            out["Kurs"] = _to_float(f.get("Kurs"))
            out["Valuta"] = out.get("Valuta") or f.get("Valuta")
            log_lines.append("FMP: pris OK")
        elif isinstance(f, (int, float)):
            out["Kurs"] = _to_float(f)
            log_lines.append("FMP: pris OK (num)")
        else:
            log_lines.append("FMP: pris MISS")

    out["TS Kurs"] = now_stamp()
    return out, "\n".join(log_lines)


def run_update_full(ticker: str, user_rates: Dict[str, float]) -> Tuple[Dict, str]:
    """
    Full uppdatering: försök hämta alla nyckeltal vi använder.
    Källprioritet: Yahoo -> FMP -> SEC. Slår samman till en rad.
    """
    log = [f"[{ticker}] FULL-UPDATE"]
    row: Dict[str, Any] = {"Ticker": ticker}

    # --- Yahoo
    y_facts = _call_fetcher(
        _yahoo,
        ["fetch_facts", "get_core_facts", "yahoo_facts"],
        ticker,
    )
    if isinstance(y_facts, dict):
        _merge_pref(row, y_facts)
        log.append("Yahoo: facts OK")
    else:
        log.append("Yahoo: facts MISS")

    y_ps = _call_fetcher(
        _yahoo,
        ["fetch_quarterly_ps", "get_quarter_ps", "yahoo_quarter_ps"],
        ticker,
    )
    if isinstance(y_ps, dict):
        _merge_pref(row, y_ps)
        log.append("Yahoo: P/S Q1..Q4 OK")
    else:
        log.append("Yahoo: P/S Q1..Q4 MISS")

    # --- FMP
    f_facts = _call_fetcher(
        _fmp,
        ["fetch_facts", "get_core_facts", "fmp_facts"],
        ticker,
    )
    if isinstance(f_facts, dict):
        _merge_pref(row, f_facts)
        log.append("FMP: facts OK")
    else:
        log.append("FMP: facts MISS")

    f_ps = _call_fetcher(
        _fmp,
        ["fetch_quarterly_ps", "get_quarter_ps", "fmp_quarter_ps"],
        ticker,
    )
    if isinstance(f_ps, dict):
        _merge_pref(row, f_ps)
        log.append("FMP: P/S Q1..Q4 OK")
    else:
        log.append("FMP: P/S Q1..Q4 MISS")

    # --- SEC
    s_facts = _call_fetcher(
        _sec,
        ["fetch_facts", "get_core_facts", "sec_facts"],
        ticker,
    )
    if isinstance(s_facts, dict):
        _merge_pref(row, s_facts)
        log.append("SEC: facts OK")
    else:
        log.append("SEC: facts MISS")

    s_ps = _call_fetcher(
        _sec,
        ["fetch_quarterly_ps", "get_quarter_ps", "sec_quarter_ps"],
        ticker,
    )
    if isinstance(s_ps, dict):
        _merge_pref(row, s_ps)
        log.append("SEC: P/S Q1..Q4 OK")
    else:
        log.append("SEC: P/S Q1..Q4 MISS")

    # Derivera/laga
    # P/S-snitt
    row["P/S-snitt (Q1..Q4)"] = _avg_safe([row.get("P/S Q1"), row.get("P/S Q2"), row.get("P/S Q3"), row.get("P/S Q4")])

    # Risklabel från mcap (förutsätter USD, men “Market Cap” vi lagrar är normaliserad hos fetchers)
    risk = _risk_label(_to_float(row.get("Market Cap")))
    if risk:
        row["Risklabel"] = risk

    # timestamps
    row["TS Full"] = now_stamp()

    return row, "\n".join(log)
