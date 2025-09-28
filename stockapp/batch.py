# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Callable, List, Optional

from .utils import add_oldest_ts_col
from .config import TS_FIELDS  # kan vara nyttigt framöver, lämnas kvar

# -------------------------------------------------------------

def _normalize_runner_result(res, df_fallback: pd.DataFrame):
    """
    Normalisera runner-result till (df, changed_list, msg).
    Tillåt formaten:
      - (df, changed, msg)
      - (df, msg)
      - (df,)
      - df
      - annat → msg=str(res)
    """
    df_out = df_fallback
    changed = []
    msg = ""
    try:
        if isinstance(res, tuple):
            if len(res) == 3:
                df_out, changed, msg = res
            elif len(res) == 2:
                df_out, msg = res
            elif len(res) == 1:
                df_out = res[0]
        elif isinstance(res, pd.DataFrame):
            df_out = res
        else:
            msg = str(res)
    except Exception as e:
        msg = f"Runner-resultat kunde inte normaliseras: {e}"
    return df_out, changed, msg

def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """Returnera tickerordning enligt val: Äldst-TS först eller A–Ö."""
    if sort_mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn", "Ticker"], ascending=[True, True, True])
        order = list(work["Ticker"].astype(str))
    else:
        work = df.sort_values(by=["Bolagsnamn", "Ticker"])
        order = list(work["Ticker"].astype(str))
    return [str(t).strip().upper() for t in order if str(t).strip()]

# -------------------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: dict,
    save_cb: Callable[[pd.DataFrame, bool], None],
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    runner: Optional[Callable] = None,
) -> Optional[pd.DataFrame]:
    """
    Renderar batch-panel i sidopanelen, hanterar kö och körning.
    Returnerar uppdaterad df om en körning genomfördes, annars None.
    """
    ss = st.session_state
    if "batch_queue" not in ss:
        ss.batch_queue = []
    if "batch_log" not in ss:
        ss.batch_log = {}

    # Val för ordning och batchstorlek
    st.sidebar.selectbox("Sortera", ["Äldst uppdaterade först (alla fält)", "A–Ö (bolagsnamn)"],
                         key="batch_sort_mode", index=0)
    st.sidebar.number_input("Batchstorlek när du lägger till", min_value=1, max_value=200, value=20, step=1, key="batch_size")

    # Lägg till i kö / töm kö
    col_add1, col_add2 = st.sidebar.columns([1, 1])
    with col_add1:
        if st.button("➕ Lägg till batch"):
            order = _pick_order(df, st.session_state.batch_sort_mode)
            to_add = []
            seen = set(ss.batch_queue)
            for t in order:
                if t not in seen:
                    to_add.append(t)
                    seen.add(t)
                if len(to_add) >= int(st.session_state.batch_size):
                    break
            ss.batch_queue.extend(to_add)
            st.sidebar.success(f"La till {len(to_add)} tickers i batchkön.")
    with col_add2:
        if st.button("🗑️ Töm kö"):
            ss.batch_queue = []
            st.sidebar.info("Batchkön tömd.")

    if not ss.batch_queue:
        st.sidebar.caption("Ingen kö ännu. Lägg till via “➕ Lägg till batch”.")
        return None

    st.sidebar.write(f"**Kö ({len(ss.batch_queue)}):** {', '.join(ss.batch_queue[:30])}" +
                     (" …" if len(ss.batch_queue) > 30 else ""))

    # Körningsknappar
    col_run1, col_run2, col_run3 = st.sidebar.columns([1, 1, 1])
    with col_run1:
        run_one = st.button("▶️ Kör 1")
    with col_run2:
        run_n = st.button("⏭️ Kör 5")
    with col_run3:
        run_all = st.button("⏯️ Kör alla")

    if not (run_one or run_n or run_all):
        return None

    if runner is None:
        st.sidebar.error("Ingen runner vald. Välj 'Full auto' eller 'Endast kurs' ovan.")
        return None

    # Antal att köra
    if run_one:
        n_to_run = 1
    elif run_n:
        n_to_run = min(5, len(ss.batch_queue))
    else:
        n_to_run = len(ss.batch_queue)

    # Kör batch
    df_work = df.copy()
    tickers_now = ss.batch_queue[:n_to_run]
    progress = st.sidebar.progress(0.0, text="Startar batch …")
    status_txt = st.sidebar.empty()

    any_changed = False
    for i, tkr in enumerate(tickers_now, start=1):
        try:
            res = runner(df_work, user_rates, tkr)
            df_work, changed, msg = _normalize_runner_result(res, df_work)
            any_changed = any_changed or bool(changed)
            ss.batch_log[tkr] = {"changed": changed, "msg": msg}
            status_txt.write(f"{i}/{n_to_run}: {msg or (tkr + ' klart.')}")
        except Exception as e:
            err_msg = f"{tkr}: Fel: {e}"
            ss.batch_log[tkr] = {"changed": [], "msg": err_msg}
            status_txt.write(f"{i}/{n_to_run}: {err_msg}")
        progress.progress(i / max(1, n_to_run), text=f"Kör {i}/{n_to_run}")

    # Ta bort körda tickers ur kön
    ss.batch_queue = ss.batch_queue[n_to_run:]

    # Recompute efteråt (om du vill räkna P/S-snitt centralt etc.)
    if recompute_cb is not None:
        try:
            df_work = recompute_cb(df_work)
        except Exception:
            pass

    # Spara
    try:
        save_cb(df_work, False)
    except Exception as e:
        st.sidebar.error(f"Misslyckades spara: {e}")

    st.sidebar.success(f"Batch klar. {n_to_run} tickers körda. {len(ss.batch_queue)} kvar i kö.")
    with st.sidebar.expander("Senaste batchlogg"):
        st.json({k: ss.batch_log[k] for k in tickers_now})

    return df_work
