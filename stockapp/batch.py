# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Callable, Optional
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
import streamlit as st

from .config import TS_FIELDS
from .sources import run_update_price_only as _fallback_price_runner
from .sources import run_update_full as _fallback_full_runner

# ---------------------------------------------------------
# Hjälpare: beräkna äldsta TS-datum per rad
# ---------------------------------------------------------

def _row_oldest_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for k in TS_FIELDS.values():
        if k in row and str(row[k]).strip():
            d = pd.to_datetime(str(row[k]).strip(), errors="coerce")
            if pd.notna(d):
                dates.append(d)
    return min(dates) if dates else pd.NaT

def _add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    w["_oldest_any_ts"] = w.apply(_row_oldest_ts, axis=1)
    w["_oldest_any_ts_fill"] = pd.to_datetime(w["_oldest_any_ts"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    return w

def _compute_order(df: pd.DataFrame, mode: str) -> List[str]:
    """
    mode: 'Äldst först' eller 'A–Ö'
    Returnerar en lista av tickers i den ordning de bör köras.
    """
    base = df.copy()
    base["Ticker"] = base["Ticker"].astype(str)
    if mode.startswith("Äldst"):
        w = _add_oldest_ts_col(base)
        w = w.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
        order = w["Ticker"].tolist()
    else:
        w = base.sort_values(by=["Bolagsnamn","Ticker"])
        order = w["Ticker"].tolist()
    # Deduplicera
    seen = set()
    uniq = []
    for t in order:
        tu = t.upper().strip()
        if tu and tu not in seen:
            seen.add(tu)
            uniq.append(t)
    return uniq

# ---------------------------------------------------------
# Batchkörning
# ---------------------------------------------------------

RunnerFn = Callable[[pd.DataFrame, str, Dict[str, float]], Tuple[pd.DataFrame, Dict[str, Tuple[float,float]], str]]

def run_batch_update(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    tickers: List[str],
    runner: Optional[RunnerFn] = None,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    progress_label: str = "Uppdaterar",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Kör uppdatering för en lista av tickers.
    - runner: funktion (df, ticker, user_rates) -> (df2, changed_fields_dict, status_msg)
    - save_cb: om angiven, kallas efter batchen (en gång).
    Returnerar (df2, log) där log innehåller 'changed', 'misses', 'count_changed', 'count_nochange', 'duration_s'.
    """
    if runner is None:
        # Fallback – försök hämta från session_state (price_runner/update_runner)
        runner = st.session_state.get("active_runner", None) or st.session_state.get("price_runner", None)
        if runner is None:
            runner = _fallback_price_runner

    total = len(tickers)
    if total == 0:
        return df, {"changed": {}, "misses": {}, "count_changed": 0, "count_nochange": 0, "duration_s": 0.0}

    prog = st.progress(0)
    counter_txt = st.empty()

    start = time.time()
    changed_all: Dict[str, Dict] = {}
    misses: Dict[str, str] = {}
    count_changed = 0
    count_nochange = 0

    df2 = df.copy()
    for i, tkr in enumerate(tickers, start=1):
        counter_txt.write(f"{progress_label}: {i}/{total}  —  {tkr}")
        try:
            df2, changed_map, msg = runner(df2, tkr, user_rates)  # full eller price-only
            if changed_map and len(changed_map) > 0:
                changed_all[tkr] = changed_map
                count_changed += 1
            else:
                count_nochange += 1
        except Exception as e:
            misses[tkr] = str(e)

        prog.progress(int(i/total*100))

    duration = time.time() - start
    log = {
        "changed": changed_all,
        "misses": misses,
        "count_changed": count_changed,
        "count_nochange": count_nochange,
        "duration_s": round(duration, 2),
        "ran_n": total,
        "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if callable(save_cb):
        try:
            save_cb(df2)
        except Exception as e:
            st.error(f"Misslyckades spara efter batch: {e}")

    # Spara logg i sessionen
    st.session_state["batch_log"] = log

    # Städa progress UI
    prog.empty()
    counter_txt.write(f"Klart: {total}/{total}")

    return df2, log

# ---------------------------------------------------------
# Sidopanel
# ---------------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame, Dict[str,float]], pd.DataFrame]] = None,
    runner: Optional[RunnerFn] = None,
    key_prefix: str = "batch"
) -> pd.DataFrame:
    """
    Visar batch-kontroller i sidopanelen.
    Returnerar df (uppdaterad om körning gjorts).
    """
    with st.sidebar:
        st.subheader("⚙️ Batch-uppdatering")

        sort_mode = st.selectbox("Sortering", ["Äldst först", "A–Ö (bolagsnamn)"], key=f"{key_prefix}_sort")
        order = _compute_order(df, sort_mode)

        # Pekare per sorteringsläge så vi inte alltid kör samma
        pointer_key = f"{key_prefix}_ptr_{'oldest' if sort_mode.startswith('Äldst') else 'alpha'}"
        if pointer_key not in st.session_state:
            st.session_state[pointer_key] = 0

        batch_size = st.number_input("Batch-storlek", min_value=1, max_value=100, value=20, step=1, key=f"{key_prefix}_size")

        # Visa nästa N tickers från pekaren
        ptr = int(st.session_state[pointer_key])
        nxt = order[ptr:ptr+batch_size]
        if not nxt and len(order) > 0:
            # wrap
            ptr = 0
            st.session_state[pointer_key] = 0
            nxt = order[:batch_size]

        st.caption(f"Nästa körning: {len(nxt)} st (position {ptr+1}–{min(ptr+len(nxt),len(order))} av {len(order)})")
        if nxt:
            st.code(", ".join(nxt), language="text")
        else:
            st.info("Inga tickers att köra.")

        mode = st.radio("Uppdateringsläge", ["Endast kurs", "Full uppdatering"], horizontal=False, key=f"{key_prefix}_mode")

        colA, colB = st.columns(2)
        run_clicked = False
        with colA:
            if st.button("▶️ Kör denna batch", key=f"{key_prefix}_run"):
                run_clicked = True
        with colB:
            if st.button("⏭️ Hoppa fram (utan att köra)", key=f"{key_prefix}_skip"):
                st.session_state[pointer_key] = min(len(order), ptr + batch_size)
                st.info(f"Flyttade pekaren till {st.session_state[pointer_key] + 1}.")
                return df

        if run_clicked and nxt:
            # Välj runner
            active_runner = runner
            if active_runner is None:
                if mode.startswith("Endast"):
                    active_runner = st.session_state.get("price_runner") or _fallback_price_runner
                else:
                    active_runner = st.session_state.get("update_runner") or _fallback_full_runner

            # Kör
            st.write("Startar batch…")
            df2, log = run_batch_update(
                df, user_rates, nxt, runner=active_runner, save_cb=save_cb,
                progress_label=("Uppdaterar pris" if mode.startswith("Endast") else "Full uppdatering")
            )

            # Recompute härledda fält om callback finns
            if callable(recompute_cb):
                try:
                    df2 = recompute_cb(df2, user_rates)
                    if callable(save_cb):
                        save_cb(df2)
                except Exception as e:
                    st.error(f"Kunde inte beräkna härledda fält efter batch: {e}")

            # Stega pekaren framåt
            st.session_state[pointer_key] = min(len(order), ptr + len(nxt))

            # Visa summering
            st.success(
                f"Klar. Ändrade: {log['count_changed']}, oförändrade: {log['count_nochange']}, missar: {len(log['misses'])}, tid: {log['duration_s']}s."
            )

            return df2

    return df
