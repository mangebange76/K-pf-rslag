# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Robust imports (modulerna du redan har)
# ------------------------------------------------------------
# config – vi utgår från dina konstanter men appen tål om ngt saknas
try:
    from stockapp.config import (
        APP_TITLE,
        FINAL_COLS,
        TS_FIELDS,
        PROPOSALS_PAGE_SIZE,
        BATCH_DEFAULT_SIZE,
        STANDARD_VALUTAKURSER,   # ev. inte strikt nödvändig här
    )
except Exception:
    APP_TITLE = "K-pf-rslag"
    PROPOSALS_PAGE_SIZE = 5
    BATCH_DEFAULT_SIZE = 10
    STANDARD_VALUTAKURSER = {"USD": 10.0, "EUR": 11.0, "CAD": 7.5, "NOK": 1.0, "SEK": 1.0}
    # Bas-säkerhet så appen ändå kan starta:
    FINAL_COLS = [
        "Ticker","Bolagsnamn","Valuta","Kurs","Antal aktier","Market Cap",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning i år (M)","Omsättning nästa år (M)",
        "Sektor","Risklabel"
    ]
    TS_FIELDS = ["Kurs TS","Full TS","Omsättning i år (M) TS","Omsättning nästa år (M) TS"]

from stockapp.storage import hamta_data, spara_data
from stockapp.utils import (
    ensure_schema,
    dedupe_tickers,
    safe_float,
    parse_date,
    format_large_number,
    add_oldest_ts_col,
    stamp_fields_ts,
    risk_label_from_mcap,
)
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
)

# Valfria fetchers – används om de finns
try:
    from stockapp.fetchers.orchestrator import run_update_full
except Exception:
    run_update_full = None

try:
    from stockapp.fetchers.yahoo import get_live_price as _yahoo_price
except Exception:
    _yahoo_price = None

# ------------------------------------------------------------
# Hjälpfunktioner – NORMALISERA KOLUMNER FRÅN DITT SHEET
# ------------------------------------------------------------
# Mappningar för att dina rubriker ska funka med resten av koden
NORMALIZE_MAP = {
    # pris/marknad
    "Aktuell kurs": "Kurs",
    "Market Cap (valuta)": "Market Cap",
    "Market Cap (SEK)": "Market Cap (SEK)",
    "Utestående aktier": "Utestående aktier",  # behåll – vi tar inte (milj.)-antagande här
    # P/S
    "P/S-snitt": "P/S-snitt (Q1..Q4)",
    # marginaler / nyckeltal
    "Bruttomarginal (%)": "Gross margin (%)",
    "Nettomarginal (%)": "Net margin (%)",
    "EV/EBITDA": "EV/EBITDA (ttm)",
    "Dividend Yield (%)": "Dividend yield (%)",
    "Payout Ratio CF (%)": "Dividend payout (FCF) (%)",
    # prognoser
    "Omsättning idag": "Omsättning i år (M)",
    "Omsättning nästa år": "Omsättning nästa år (M)",
    # övrigt
    "Industri": "Industri",
    "Risklabel": "Risklabel",
}

# TS-prefix → suffix-konvertering, ex. "TS_P/S" → "P/S TS"
def _normalize_ts_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_names = {}
    for c in out.columns:
        if isinstance(c, str) and c.startswith("TS_"):
            base = c[3:].strip()
            # specialfall: "Omsättning idag"/"Omsättning nästa år"
            if base == "Omsättning idag":
                base = "Omsättning i år (M)"
            elif base == "Omsättning nästa år":
                base = "Omsättning nästa år (M)"
            new_names[c] = f"{base} TS"
    if new_names:
        out = out.rename(columns=new_names)
    return out

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mappar dina rubriker till interna – i MINNET, påverkar inte Google Sheet."""
    out = df.copy()
    # 1) först fixa TS-kolumner (TS_ → ... TS)
    out = _normalize_ts_columns(out)
    # 2) mappa “mänskliga namn” → interna
    rename_map = {c: NORMALIZE_MAP[c] for c in out.columns if c in NORMALIZE_MAP}
    if rename_map:
        out = out.rename(columns=rename_map)
    # 3) säkerställ några kolumner
    for need in ["Kurs","Market Cap","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)","Valuta","Antal aktier","Sektor","Risklabel","Bolagsnamn"]:
        if need not in out.columns:
            out[need] = np.nan
    # om P/S-snitt saknas men Q1..Q4 finns – räkna
    if "P/S-snitt (Q1..Q4)" in out.columns:
        qs = [q for q in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"] if q in out.columns]
        if qs:
            out["P/S-snitt (Q1..Q4)"] = pd.to_numeric(out[qs].mean(axis=1), errors="coerce")
    # risklabel om saknas men Market Cap finns
    if "Risklabel" in out.columns and "Market Cap" in out.columns:
        mask = out["Risklabel"].isna() | (out["Risklabel"]=="") | (out["Risklabel"].astype(str)=="nan")
        if mask.any():
            out.loc[mask, "Risklabel"] = out.loc[mask, "Market Cap"].apply(risk_label_from_mcap)
    return out

# ------------------------------------------------------------
# State-init & IO
# ------------------------------------------------------------
def _init_state():
    if "_df_ref" not in st.session_state:
        st.session_state["_df_ref"] = pd.DataFrame()

    if "_rates_init" not in st.session_state:
        saved = las_sparade_valutakurser()
        for k in ["USD","EUR","CAD","NOK","SEK"]:
            st.session_state[f"rate_{k}"] = float(saved.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))
        st.session_state["_rates_init"] = True

    # batch (round-robin light)
    st.session_state.setdefault("batch_order_mode", "Äldst först")
    st.session_state.setdefault("batch_ts_basis", "Alla TS (äldst av alla)")
    st.session_state.setdefault("batch_size", int(BATCH_DEFAULT_SIZE))
    st.session_state.setdefault("batch_order_list", [])
    st.session_state.setdefault("batch_cursor", 0)
    st.session_state.setdefault("batch_processed_cycle", [])
    st.session_state.setdefault("batch_queue", [])
    st.session_state.setdefault("batch_log", [])

    # vy
    st.session_state.setdefault("view", "Investeringsförslag")
    st.session_state.setdefault("page", 1)
    st.session_state.setdefault("page_size", int(PROPOSALS_PAGE_SIZE))

    # edit
    st.session_state.setdefault("edit_index", 0)

def _load_df() -> pd.DataFrame:
    try:
        raw = hamta_data()
    except Exception as e:
        st.error(f"🚫 Kunde inte läsa data från Google Sheet: {e}")
        return pd.DataFrame()

    # normalisera kolumner i minnet
    df = _normalize_columns(raw)
    # se till att schema finns (lägger till saknade kolumner)
    df = ensure_schema(df, FINAL_COLS)
    # dedupe (i minnet)
    df2, dups = dedupe_tickers(df)
    if dups:
        st.info(f"ℹ️ Dubbletter ignoreras i minnet: {', '.join(dups)}")
    return df2

def _save_df(df: pd.DataFrame):
    try:
        spara_data(df)
        st.success("✅ Sparat till Google Sheet.")
    except Exception as e:
        st.error(f"🚫 Kunde inte spara: {e}")

# ------------------------------------------------------------
# Valutor – sidopanel
# ------------------------------------------------------------
def _sidebar_rates() -> Dict[str, float]:
    with st.sidebar.expander("💱 Valutakurser (→ SEK)", expanded=True):
        if st.button("Hämta automatiskt"):
            fetched, misses, provider = hamta_valutakurser_auto()
            for k, v in fetched.items():
                st.session_state[f"rate_{k}"] = float(v)
            spara_valutakurser(fetched)
            if misses:
                st.warning("Kunde inte hämta: " + ", ".join(misses))
            st.toast(f"Valutor uppdaterade via {provider}.")
            st.rerun()

        usd = st.number_input("USD", value=float(st.session_state["rate_USD"]), key="rate_USD", step=0.01)
        eur = st.number_input("EUR", value=float(st.session_state["rate_EUR"]), key="rate_EUR", step=0.01)
        cad = st.number_input("CAD", value=float(st.session_state["rate_CAD"]), key="rate_CAD", step=0.01)
        nok = st.number_input("NOK", value=float(st.session_state["rate_NOK"]), key="rate_NOK", step=0.01)
        sek = st.number_input("SEK", value=float(st.session_state["rate_SEK"]), key="rate_SEK", step=0.01)

        rates = {"USD": usd, "EUR": eur, "CAD": cad, "NOK": nok, "SEK": sek}
        if st.button("Spara kurser"):
            spara_valutakurser(rates)
            st.toast("Sparade valutakurser.")
    return rates

# ------------------------------------------------------------
# Batch – round-robin
# ------------------------------------------------------------
def _detect_ts_cols_for_basis(df: pd.DataFrame, basis: str) -> List[str]:
    b = (basis or "").lower()
    cols = [c for c in df.columns if isinstance(c, str)]
    # “Kurs TS”
    if "kurs" in b:
        wanted = [k for k in ["Kurs TS","TS Kurs","Pris TS","TS Pris"] if k in cols]
        return wanted
    # “Full TS”
    if "full" in b:
        wanted = [k for k in ["Full TS","TS Full"] if k in cols]
        return wanted
    # “Alla TS” – tom lista => utils tar min(alla TS)
    return []

def _build_order_list(df: pd.DataFrame, mode: str, ts_basis: str) -> List[str]:
    work = df.copy()
    work["Ticker"] = work["Ticker"].astype(str)
    if mode == "Äldst först":
        ts_cols = _detect_ts_cols_for_basis(work, ts_basis)
        if ts_cols:
            work = add_oldest_ts_col(work, ts_cols=ts_cols, dest_col="__oldest_ts__")
        else:
            work = add_oldest_ts_col(work, dest_col="__oldest_ts__")
        work = work.sort_values(by="__oldest_ts__", ascending=True, na_position="first")
    elif mode == "A–Ö":
        work = work.sort_values(by="Ticker", ascending=True)
    else:
        work = work.sort_values(by="Ticker", ascending=False)
    return work["Ticker"].tolist()

def _ensure_order(df: pd.DataFrame):
    if not st.session_state.get("batch_order_list"):
        st.session_state["batch_order_list"] = _build_order_list(
            df, st.session_state["batch_order_mode"], st.session_state["batch_ts_basis"]
        )
        st.session_state["batch_cursor"] = 0
        st.session_state["batch_processed_cycle"] = []

def _create_next_queue(df: pd.DataFrame, size: int) -> List[str]:
    _ensure_order(df)
    order = st.session_state["batch_order_list"]
    if not order:
        return []
    n = len(order)
    done = set(st.session_state.get("batch_processed_cycle", []))
    if len(done) >= n:  # ny cykel
        done = set()
    q: List[str] = []
    idx = int(st.session_state.get("batch_cursor", 0))
    i = 0
    while len(q) < min(size, n) and i < n*2:
        t = order[idx]
        if t not in done and t not in q:
            q.append(t)
        idx = (idx + 1) % n
        i += 1
    st.session_state["batch_cursor"] = idx
    return q

def _runner_price(df: pd.DataFrame, tkr: str) -> Tuple[pd.DataFrame, str]:
    if _yahoo_price is None:
        return df, "Yahoo-källa saknas"
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns inte"
    try:
        px = _yahoo_price(str(tkr))
        if px and px > 0:
            df.loc[ridx, "Kurs"] = float(px)
            df = stamp_fields_ts(df, ["Kurs"], ts_suffix=" TS")
            return df, "OK (kurs)"
        return df, "Pris saknas"
    except Exception as e:
        return df, f"Fel: {e}"

def _runner_full(df: pd.DataFrame, tkr: str) -> Tuple[pd.DataFrame, str]:
    if run_update_full is None:
        return _runner_price(df, tkr)
    try:
        out = run_update_full(df, tkr, {"USD": st.session_state["rate_USD"]})  # user_rates om din orchestrator vill ha
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], str(out[1])
        if isinstance(out, pd.DataFrame):
            return out, "OK (full)"
        return df, "Orchestrator: oväntat svar"
    except Exception as e:
        return df, f"Fel: {e}"

def _run_batch(df: pd.DataFrame, queue: List[str], mode: str) -> pd.DataFrame:
    if not queue:
        st.info("Ingen batchkö.")
        return df

    total = len(queue)
    bar = st.sidebar.progress(0, text=f"0/{total}")
    done = 0
    work = df.copy()
    done_set = set(st.session_state.get("batch_processed_cycle", []))
    local_log: List[str] = []

    for tkr in queue:
        if mode == "price":
            work, msg = _runner_price(work, tkr)
        else:
            work, msg = _runner
