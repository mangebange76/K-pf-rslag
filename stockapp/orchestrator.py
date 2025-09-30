# stockapp/orchestrator.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd

# Streamlit är valfritt (vi använder det bara om det finns)
try:
    import streamlit as st
except Exception:
    st = None  # typ: ignore

# Konfig & utils
from .config import TS_FIELDS
from .utils import (
    ensure_schema,
    now_stamp,
    stamp_fields_ts,
)

# -------------------------------------------------------------------
# Försök importera fetchers (hantera om någon saknas)
# -------------------------------------------------------------------
_YAHOO = {}
_SEC = {}
_FMP = {}

try:
    from .fetchers.yahoo import fetch_basics as yahoo_fetch_basics
    from .fetchers.yahoo import fetch_quarters_and_ttm as yahoo_fetch_quarters
    from .fetchers.yahoo import fetch_quality_metrics as yahoo_fetch_quality
    _YAHOO["ok"] = True
except Exception as e:
    _YAHOO["ok"] = False
    _YAHOO["err"] = str(e)

try:
    from .fetchers.sec import fetch_sec_shares_and_quarters
    _SEC["ok"] = True
except Exception as e:
    _SEC["ok"] = False
    _SEC["err"] = str(e)

try:
    from .fetchers.fmp import fetch_fmp_ps_light
    _FMP["ok"] = True
except Exception as e:
    _FMP["ok"] = False
    _FMP["err"] = str(e)


# -------------------------------------------------------------------
# Hjälpare: merge/normalize
# -------------------------------------------------------------------
_NUMERIC_FIELDS_POSITIVE_ONLY = {
    # fält där 0.0 inte är meningsfullt – skriv inte över med 0 om vi redan hade >0
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Utestående aktier", "Omsättning TTM", "Market Cap",
}

def _merge_into(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """
    Lägg in src i dst med enkla regler:
    - Skriv om strängar om de inte är tomma.
    - Skriv om numeriska om de är > 0, för fält i _NUMERIC_FIELDS_POSITIVE_ONLY.
      För övriga numeriska fält: skriv även 0.0 (t.ex. skuldsättning kan vara 0).
    """
    if not isinstance(src, dict):
        return
    for k, v in src.items():
        if v is None:
            continue
        if isinstance(v, str):
            if v.strip():
                dst[k] = v.strip()
        elif isinstance(v, (int, float)):
            if k in _NUMERIC_FIELDS_POSITIVE_ONLY:
                try:
                    if float(v) > 0:
                        dst[k] = float(v)
                except Exception:
                    pass
            else:
                # tillåt 0.0 för många kvalitetsmått (t.ex. skuldsättning)
                try:
                    dst[k] = float(v)
                except Exception:
                    pass
        else:
            # för listor/dict: skriv rakt av
            dst[k] = v

def _compose_source_label(parts: List[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return "Auto"
    return "Auto (" + " → ".join(parts) + ")"

# vilka fält ska TS-stämplas om de finns i uppdateringen?
_TS_TRACKED_FIELDS = set(TS_FIELDS.keys())

# -------------------------------------------------------------------
# Skriva tillbaka till DataFrame
# -------------------------------------------------------------------
def update_df_with_vals(df: pd.DataFrame, ticker: str, new_vals: Dict[str, Any], source_label: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Skriv in värden för 'ticker' i df. Skapar INTE nya rader – om ticker saknas returneras oförändrad df.
    Stämplar TS-kolumner för alla fält som finns i new_vals och ingår i TS_FIELDS, även om värdet är oförändrat.
    Sätter också 'Senast auto-uppdaterad' + 'Senast uppdaterad källa'.
    Returnerar (df, changed_fields).
    """
    df = ensure_schema(df)
    mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
    if mask.sum() == 0:
        # Hittas ej -> appen kan visa info/varning
        return df, []

    idx = df.index[mask][0]
    changed: List[str] = []

    # Skriv in fält
    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            # skapa ny kolumn (gärna numerisk 0.0 utom typiska strängfält)
            if f in ("Ticker", "Bolagsnamn", "Valuta"):
                df[f] = ""
            else:
                df[f] = 0.0

        old = df.at[idx, f]
        # skriv alltid, men markera "changed" bara om det faktiskt ändrades text/numeriskt
        should_mark_changed = False
        try:
            if isinstance(v, (int, float)) and isinstance(old, (int, float, float)):
                should_mark_changed = (pd.isna(old) and not pd.isna(v)) or (float(old) != float(v))
            else:
                should_mark_changed = (str(old) != str(v))
        except Exception:
            should_mark_changed = True

        df.at[idx, f] = v
        if should_mark_changed:
            changed.append(f)

    # TS-stämpla ALLA spårade fält som är med i uppdateringen
    fields_to_stamp = [f for f in (new_vals or {}).keys() if f in _TS_TRACKED_FIELDS]
    if fields_to_stamp:
        df = stamp_fields_ts(df, idx, fields_to_stamp)

    # uppdatera auto-meta
    df.at[idx, "Senast auto-uppdaterad"] = now_stamp()
    df.at[idx, "Senast uppdaterad källa"] = source_label

    return df, changed


# -------------------------------------------------------------------
# Runners
# -------------------------------------------------------------------
def run_update_price(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Endast snabb kurs-uppdatering (Yahoo).
    Sätter (om tillgängligt): Aktuell kurs, Valuta, Bolagsnamn, Market Cap.
    Returnerar (df, debug).
    """
    debug: Dict[str, Any] = {"ticker": ticker, "runner": "price"}
    df = ensure_schema(df)
    mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
    if mask.sum() == 0:
        debug["error"] = "Ticker not found"
        return df, debug

    vals: Dict[str, Any] = {}
    source_parts: List[str] = []

    # Yahoo basics
    if _YAHOO.get("ok"):
        try:
            yb = yahoo_fetch_basics(ticker)
            _merge_into(vals, {k: yb.get(k) for k in ("Aktuell kurs", "Valuta", "Bolagsnamn", "Market Cap")})
            source_parts.append("Yahoo")
            debug["yahoo_basics"] = yb
        except Exception as e:
            debug["yahoo_err"] = str(e)
    else:
        debug["yahoo_unavailable"] = _YAHOO.get("err")

    src = _compose_source_label(source_parts)
    df, changed = update_df_with_vals(df, ticker, vals, src)
    debug["changed"] = changed
    return df, debug


def run_update_full(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full uppdatering för ett bolag:
      1) SEC (shares + kvartal/TTM) – om tillgängligt
      2) Yahoo quarters/TTM + kvalitet + basics
      3) FMP (lätt) – endast P/S som sista fallback
    Skriver EJ 'Omsättning i år (förv.)' eller 'Omsättning nästa år (förv.)' (de är manuella).
    Returnerar (df, debug).
    """
    debug: Dict[str, Any] = {"ticker": ticker, "runner": "full"}
    df = ensure_schema(df)
    mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
    if mask.sum() == 0:
        debug["error"] = "Ticker not found"
        return df, debug

    vals: Dict[str, Any] = {}
    source_parts: List[str] = []

    # 1) SEC
    if _SEC.get("ok"):
        try:
            secd = fetch_sec_shares_and_quarters(ticker)
            # typiskt: Utestående aktier, Omsättning TTM, P/S, P/S Q1..Q4 (i bolagets prisvaluta om vi lyckats konvertera i fetchern)
            _merge_into(vals, secd)
            source_parts.append("SEC")
            debug["sec"] = secd
        except Exception as e:
            debug["sec_err"] = str(e)
    else:
        debug["sec_unavailable"] = _SEC.get("err")

    # 2) Yahoo quarters/quality/basics
    if _YAHOO.get("ok"):
        try:
            yq = yahoo_fetch_quarters(ticker)
            _merge_into(vals, yq)  # Omsättning TTM, P/S, P/S Q1..Q4, Utestående (implied) m.m.
            debug["yahoo_quarters"] = yq
        except Exception as e:
            debug["yahoo_quarters_err"] = str(e)

        try:
            yqual = yahoo_fetch_quality(ticker)
            _merge_into(vals, yqual)  # Sector, Industry, EV/EBITDA, ROIC/ROE, Cash, OCF, FCF, Margins, D/E, Div, Payout FCF m.m.
            debug["yahoo_quality"] = yqual
        except Exception as e:
            debug["yahoo_quality_err"] = str(e)

        try:
            yb = yahoo_fetch_basics(ticker)
            _merge_into(vals, {k: yb.get(k) for k in ("Aktuell kurs", "Valuta", "Bolagsnamn", "Market Cap")})
            debug["yahoo_basics"] = yb
            source_parts.append("Yahoo")
        except Exception as e:
            debug["yahoo_basics_err"] = str(e)
    else:
        debug["yahoo_unavailable"] = _YAHOO.get("err")

    # 3) FMP (endast P/S som sista chans om saknas)
    if ("P/S" not in vals or not vals.get("P/S")) and _FMP.get("ok"):
        try:
            fmpd = fetch_fmp_ps_light(ticker)
            if isinstance(fmpd, dict) and fmpd.get("P/S"):
                vals["P/S"] = float(fmpd["P/S"])
                source_parts.append("FMP")
            debug["fmp"] = fmpd
        except Exception as e:
            debug["fmp_err"] = str(e)
    elif not _FMP.get("ok"):
        debug["fmp_unavailable"] = _FMP.get("err")

    # Klar – skriv tillbaka
    src = _compose_source_label(source_parts)
    df, changed = update_df_with_vals(df, ticker, vals, src)
    debug["changed"] = changed
    return df, debug


# -------------------------------------------------------------------
# Batch-runner (enkel – UI/urval sköts i batch-modulen)
# -------------------------------------------------------------------
def run_batch_update(
    df: pd.DataFrame,
    tickers: List[str],
    mode: str = "full",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Kör uppdatering på en lista tickers. mode ∈ {"full","price"}
    Returnerar (df, log) där log innehåller 'ok', 'fail', och 'details'.
    """
    df = ensure_schema(df)
    log = {"ok": [], "fail": [], "details": []}

    n = len(tickers)
    for i, t in enumerate(tickers, start=1):
        # progress UI (om streamlit finns)
        if st is not None:
            st.sidebar.write(f"Uppdaterar {i}/{n}: {t}")

        try:
            if mode == "price":
                df, dbg = run_update_price(df, t)
            else:
                df, dbg = run_update_full(df, t)

            if dbg.get("error"):
                log["fail"].append(t)
            else:
                log["ok"].append(t)
            log["details"].append({t: dbg})
        except Exception as e:
            log["fail"].append(t)
            log["details"].append({t: {"runner": mode, "error": str(e)}})

    return df, log
