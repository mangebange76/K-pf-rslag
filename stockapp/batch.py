# -*- coding: utf-8 -*-
"""
stockapp/batch.py

Batch-körning, köhantering och sidopanel-kontroller.
- Bygg/återställ batch-kö utifrån A–Ö eller äldst TS först
- Kör batch med 1/X text + progressbar
- Fortsätt-knapp
- Uppdatera endast kurser (alla eller en ticker) om runner_price_only ges

Runner-API:
    runner_full(ticker: str, df: pd.DataFrame, user_rates: dict) -> (pd.DataFrame, list[str], str)
    runner_price_only(ticker: str, df: pd.DataFrame, user_rates: dict) -> (pd.DataFrame, list[str], str)

save_cb(df) kallas efter lyckad batch (om angett)
recompute_cb(df) kallas för att räkna om derivatfält efter uppdateringar (om angett)
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------- Hjälpare (TS) ----------------------------------

TS_FIELDS = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

def _oldest_any_ts(row: pd.Series) -> pd.Timestamp:
    dates = []
    for ts_col in TS_FIELDS.values():
        if ts_col in row and str(row.get(ts_col, "")).strip():
            try:
                d = pd.to_datetime(str(row[ts_col]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    if not dates:
        return pd.NaT
    return pd.Series(dates).min()

def _add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
    filler = pd.Timestamp("2099-12-31")
    work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(filler)
    return work

# --------------------------- Kö-hantering ------------------------------------

_SS_Q      = "ss_batch_queue"     # List[str] tickers
_SS_POS    = "ss_batch_pos"       # int position i kön (0-baserad)
_SS_LOG    = "ss_batch_log"       # dict med changed/misses
_SS_LASTSZ = "ss_batch_last_size" # int senaste byggstorlek
_SS_SORT   = "ss_batch_sort"      # senast valda sorteringen

def _ensure_state_defaults():
    st.session_state.setdefault(_SS_Q, [])
    st.session_state.setdefault(_SS_POS, 0)
    st.session_state.setdefault(_SS_LOG, {"changed": {}, "misses": {}})
    st.session_state.setdefault(_SS_LASTSZ, 0)
    st.session_state.setdefault(_SS_SORT, "A–Ö (bolagsnamn)")

def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """
    Returnerar ticker-lista i vald sortering.
    """
    if sort_mode.startswith("Äldst"):
        work = _add_oldest_ts_col(df)
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
        tickers = [str(t).upper() for t in work["Ticker"].tolist() if str(t).strip()]
        return tickers
    else:
        work = df.copy()
        work = work.sort_values(by=["Bolagsnamn","Ticker"])
        return [str(t).upper() for t in work["Ticker"].tolist() if str(t).strip()]

def _build_queue(df: pd.DataFrame, sort_mode: str, batch_size: int) -> None:
    tickers = _pick_order(df, sort_mode)
    if batch_size > 0:
        tickers = tickers[:batch_size]
    st.session_state[_SS_Q] = tickers
    st.session_state[_SS_POS] = 0
    st.session_state[_SS_LASTSZ] = len(tickers)
    st.session_state[_SS_SORT] = sort_mode
    st.session_state[_SS_LOG] = {"changed": {}, "misses": {}}

def _append_log_changed(ticker: str, fields: List[str]):
    log = st.session_state.get(_SS_LOG, {"changed": {}, "misses": {}})
    if fields:
        log["changed"].setdefault(ticker, [])
        log["changed"][ticker].extend(fields)
    st.session_state[_SS_LOG] = log

def _append_log_miss(ticker: str, reason: str = "(inga nya fält)"):
    log = st.session_state.get(_SS_LOG, {"changed": {}, "misses": {}})
    log["misses"].setdefault(ticker, [])
    log["misses"][ticker].append(reason)
    st.session_state[_SS_LOG] = log

def _update_global_last_log():
    """
    Kopiera senaste loggen till en global 'last_auto_log'
    för visning i kontroll-vyn.
    """
    st.session_state["last_auto_log"] = st.session_state.get(_SS_LOG, {"changed": {}, "misses": {}})

# --------------------------- Körning -----------------------------------------

RunnerFn = Callable[[str, pd.DataFrame, Dict[str, float]], Tuple[pd.DataFrame, List[str], str]]

def _run_queue(df: pd.DataFrame,
               user_rates: Dict[str, float],
               runner: RunnerFn,
               save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
               recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Kör nuvarande kö från pos..end. Visar progressbar + 1/X.
    Returnerar ev. uppdaterad df.
    """
    q = st.session_state.get(_SS_Q, [])
    pos = int(st.session_state.get(_SS_POS, 0))
    total = len(q)
    if total == 0 or pos >= total:
        st.info("Ingen kö att köra.")
        return df

    progress = st.progress(0.0, text=f"Kör batch {pos}/{total}")
    status = st.empty()

    df_cur = df
    for i in range(pos, total):
        tkr = str(q[i]).upper()
        status.write(f"Uppdaterar {i+1}/{total}: **{tkr}**")
        try:
            df_cur, changed_fields, source_label = runner(tkr, df_cur, user_rates)
            if changed_fields:
                _append_log_changed(tkr, changed_fields)
            else:
                _append_log_miss(tkr, "(inga nya fält)")
        except Exception as e:
            _append_log_miss(tkr, f"error: {e}")

        st.session_state[_SS_POS] = i + 1
        progress.progress((i+1)/max(total,1), text=f"Kör batch {i+1}/{total}")

    progress.empty()
    status.empty()

    # Recompute & save
    if callable(recompute_cb):
        try:
            df_cur = recompute_cb(df_cur)
        except Exception:
            pass
    if callable(save_cb):
        try:
            save_cb(df_cur)
        except Exception as e:
            st.warning(f"Kunde inte spara efter batch: {e}")

    _update_global_last_log()
    st.success("Batch körning klar.")
    return df_cur

# --------------------------- Sidopanel ---------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    runner_full: Optional[RunnerFn] = None,
    runner_price_only: Optional[RunnerFn] = None,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Renderar sidopanelens batch-kontroller och hanterar användarens åtgärder.
    Returnerar ev. uppdaterad df.
    """
    _ensure_state_defaults()
    df_out = df

    st.sidebar.subheader("🛠️ Batch-körning")
    sort_mode = st.sidebar.selectbox("Sortering för batch", ["A–Ö (bolagsnamn)", "Äldst uppdaterade först (alla spårade fält)"], index=0)
    batch_size = st.sidebar.number_input("Batch-storlek", min_value=1, max_value=500, value=max(1, st.session_state.get(_SS_LASTSZ, 20) or 20), step=1)

    col_b1, col_b2 = st.sidebar.columns(2)
    with col_b1:
        if st.button("📋 Bygg/Återställ kö"):
            _build_queue(df, sort_mode, int(batch_size))
            st.sidebar.success(f"Kö byggd: {len(st.session_state[_SS_Q])} tickers")
    with col_b2:
        if st.button("🧹 Töm kö"):
            st.session_state[_SS_Q] = []
            st.session_state[_SS_POS] = 0
            st.sidebar.info("Kö rensad.")

    # Visa kö-status
    q = st.session_state.get(_SS_Q, [])
    pos = int(st.session_state.get(_SS_POS, 0))
    if q:
        st.sidebar.caption(f"Kö: {pos}/{len(q)} körda")
        with st.sidebar.expander("Visa kö", expanded=False):
            if len(q) <= 200:
                st.write(", ".join(q))
            else:
                st.write(", ".join(q[:200]) + f" ... (+{len(q)-200} till)")

    # Kör batch (full updatering)
    if st.sidebar.button("▶️ Kör batch (full uppdatering)", disabled=not callable(runner_full) or not q):
        if not callable(runner_full):
            st.sidebar.error("Ingen runner_full är registrerad.")
        else:
            df_out = _run_queue(df_out, user_rates, runner_full, save_cb=save_cb, recompute_cb=recompute_cb)

    # Fortsätt om kö kvar
    if st.sidebar.button("⏩ Fortsätt batch", disabled=not callable(runner_full) or not q or pos >= len(q)):
        if callable(runner_full):
            df_out = _run_queue(df_out, user_rates, runner_full, save_cb=save_cb, recompute_cb=recompute_cb)

    st.sidebar.markdown("---")

    # Uppdatera endast kurser (alla i tabellen)
    if st.sidebar.button("🔁 Uppdatera endast kurser (alla)", disabled=not callable(runner_price_only)):
        if not callable(runner_price_only):
            st.sidebar.error("Ingen runner_price_only är registrerad.")
        else:
            # bygg en temporär kö med alla tickers
            tickers = [str(t).upper() for t in df["Ticker"].tolist() if str(t).strip()]
            progress = st.sidebar.progress(0.0, text=f"Kurser 0/{len(tickers)}")
            for i, tkr in enumerate(tickers, start=1):
                try:
                    df_out, changed_fields, src = runner_price_only(tkr, df_out, user_rates)
                    if changed_fields:
                        _append_log_changed(tkr, changed_fields)
                except Exception as e:
                    _append_log_miss(tkr, f"price-only error: {e}")
                progress.progress(i/max(len(tickers),1), text=f"Kurser {i}/{len(tickers)}")
            progress.empty()
            if callable(recompute_cb):
                try:
                    df_out = recompute_cb(df_out)
                except Exception:
                    pass
            if callable(save_cb):
                try:
                    save_cb(df_out)
                except Exception as e:
                    st.sidebar.warning(f"Kunde inte spara efter kursuppdatering: {e}")
            _update_global_last_log()
            st.sidebar.success("Kursuppdatering (alla) klar.")

    # Uppdatera endast kurs för vald ticker
    tkr_sel = st.sidebar.text_input("Ticker (för kurs-uppdatering)", value="")
    if st.sidebar.button("🔁 Uppdatera kurs (vald ticker)", disabled=not callable(runner_price_only) or not tkr_sel.strip()):
        if callable(runner_price_only):
            try:
                df_out, changed_fields, src = runner_price_only(tkr_sel.strip().upper(), df_out, user_rates)
                if changed_fields:
                    _append_log_changed(tkr_sel.strip().upper(), changed_fields)
                if callable(recompute_cb):
                    df_out = recompute_cb(df_out)
                if callable(save_cb):
                    save_cb(df_out)
                _update_global_last_log()
                st.sidebar.success(f"Kurs uppdaterad: {tkr_sel.strip().upper()}")
            except Exception as e:
                st.sidebar.error(f"Kunde inte uppdatera kurs för {tkr_sel}: {e}")

    return df_out
