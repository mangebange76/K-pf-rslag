# stockapp/batch.py
# -*- coding: utf-8 -*-
"""
Batch-kontroller i sidopanelen + hjälpfunktioner.

Publik:
- sidebar_batch_controls(df, user_rates, save_cb=None, recompute_cb=None, runner=None) -> pd.DataFrame
- run_batch_update(df, user_rates, tickers, runner, make_snapshot=False, save_cb=None, recompute_cb=None)
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Callable, Optional
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import streamlit as st

from .config import TS_FIELDS
from .sources import run_update_full, run_update_price_only


# ------------------------------------------------------------
# Interna hjälpare
# ------------------------------------------------------------

def _safe_to_datetime(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        d = pd.to_datetime(str(s), errors="coerce")
        if pd.isna(d):
            return None
        return d.to_pydatetime()
    except Exception:
        return None

def _oldest_ts_for_row(row: pd.Series) -> Optional[datetime]:
    """Äldsta tidsstämpeln bland TS_-kolumner som finns i DF."""
    dates: List[datetime] = []
    for tracked_field, ts_col in TS_FIELDS.items():
        if ts_col in row and str(row[ts_col]).strip():
            d = _safe_to_datetime(str(row[ts_col]).strip())
            if d is not None:
                dates.append(d)
    if not dates:
        return None
    return min(dates)

def _pick_order(df: pd.DataFrame, mode: str) -> List[str]:
    """
    Returnerar lista av tickers i vald ordning.
    mode: "Äldst TS först" | "A–Ö (Ticker)"
    """
    work = df.copy()
    if "Ticker" not in work.columns:
        return []

    if mode.startswith("Äldst"):
        work["_oldest_any_ts"] = work.apply(_oldest_ts_for_row, axis=1)
        # sortera: None → längst bak, därefter stigande datum (äldst först)
        work["_oldest_sort"] = work["_oldest_any_ts"].apply(lambda d: d if d is not None else datetime(2999, 1, 1))
        work = work.sort_values(by=["_oldest_sort", "Ticker"], ascending=[True, True])
        order = [str(t).upper() for t in work["Ticker"].tolist()]
        return order
    else:
        order = [str(t).upper() for t in work["Ticker"].tolist()]
        order = sorted(order)
        return order

def _unique_preserve(items: List[str]) -> List[str]:
    """Unika (case-insensitive), bibehåll ordning."""
    seen = set()
    out = []
    for x in items:
        k = str(x).upper()
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


# ------------------------------------------------------------
# Körning
# ------------------------------------------------------------

RunnerFn = Callable[[pd.DataFrame, str, Dict[str, float]], Tuple[pd.DataFrame, List[str], str]]

def run_batch_update(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    tickers: List[str],
    runner: RunnerFn,
    make_snapshot: bool = False,
    save_cb: Optional[Callable[[pd.DataFrame, bool], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    """
    Kör runner för en lista tickers. Returnerar (df2, changed_map, msg_map).
    changed_map: {ticker: [fält,...]}
    msg_map: {ticker: "status-text"}
    """
    total = len(tickers)
    changed_map: Dict[str, List[str]] = {}
    msg_map: Dict[str, str] = {}

    prog = st.sidebar.progress(0.0)
    prog_txt = st.sidebar.empty()

    df2 = df.copy()
    for i, tkr in enumerate(tickers, start=1):
        try:
            df2, changed, msg = runner(df2, tkr, user_rates)
            if changed:
                changed_map[tkr] = changed
            msg_map[tkr] = msg or ""
        except Exception as e:
            msg_map[tkr] = f"Fel: {e}"

        frac = i / max(1, total)
        prog.progress(frac)
        prog_txt.write(f"**Körning:** {i}/{total}  —  {tkr}")

    # Efter batch: recompute + spara
    if recompute_cb is not None:
        try:
            df2 = recompute_cb(df2)
        except Exception as e:
            st.sidebar.warning(f"Beräkningar misslyckades: {e}")

    if save_cb is not None:
        try:
            save_cb(df2, make_snapshot)
        except Exception as e:
            st.sidebar.error(f"Misslyckades spara till Google Sheets: {e}")

    st.sidebar.success("Batchkörning klar.")
    return df2, changed_map, msg_map


# ------------------------------------------------------------
# Sidopanel: kontroller
# ------------------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame, bool], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    runner: Optional[RunnerFn] = None,
) -> pd.DataFrame:
    """
    Bygger hela batchpanelen i sidopanelen.
    Använder st.session_state["batch_queue"], ["batch_pointer"], ["batch_last_results"].
    Returnerar ev. uppdaterad df.
    """
    st.sidebar.subheader("🛠️ Batch-körning")

    # Runner-val
    runner_map: Dict[str, RunnerFn] = {
        "Full (Yahoo)": run_update_full,
        "Pris endast": run_update_price_only,
    }
    default_runner_name = "Full (Yahoo)" if runner is None else None
    runner_name = st.sidebar.selectbox(
        "Runner",
        options=list(runner_map.keys()) if runner is None else ["(extern runner)"],
        index=0,
        help="Välj vilken uppdateringsstrategi som ska användas per ticker."
    )
    use_runner = runner if runner is not None else runner_map[runner_name]

    # Sortering
    sort_mode = st.sidebar.radio(
        "Ordning för batch",
        ["Äldst TS först", "A–Ö (Ticker)"],
        horizontal=False,
    )

    # Storlek att lägga i kö
    default_batch_size = st.sidebar.number_input(
        "Batch-storlek (hur många tickers ska läggas i kön vid 'Skapa batch')",
        min_value=1, max_value=500, value=20, step=1
    )

    # Skapa batch-kö
    if st.sidebar.button("📦 Skapa batch från tabellen"):
        order = _pick_order(df, sort_mode)
        order = _unique_preserve(order)
        st.session_state["batch_queue"] = order[:default_batch_size] if default_batch_size > 0 else order
        st.session_state["batch_pointer"] = 0
        st.session_state["batch_last_results"] = {}
        st.sidebar.success(f"Skapade batch-kö med {len(st.session_state['batch_queue'])} tickers.")

    # Visa status för kön
    queue: List[str] = st.session_state.get("batch_queue", [])
    ptr: int = int(st.session_state.get("batch_pointer", 0))
    remaining = max(0, len(queue) - ptr)

    if queue:
        st.sidebar.info(f"Kö: {len(queue)} tickers  —  Nästa index: {ptr}  —  Kvar: {remaining}")
        st.sidebar.code(", ".join(queue[max(0, ptr - 3):min(len(queue), ptr + 7)]) or "-", language="text")
    else:
        st.sidebar.write("Ingen batch-kö ännu.")

    # Kör “nästa N”
    step = st.sidebar.number_input("Kör nästa N", min_value=1, max_value=max(1, remaining or 1), value=min(10, remaining or 1), step=1)
    run_next = st.sidebar.button("▶️ Kör nästa N")

    # Kör “hela kön”
    run_all = st.sidebar.button("⏩ Kör hela kön")

    # Rensa
    if st.sidebar.button("🗑️ Rensa batch-kö"):
        st.session_state["batch_queue"] = []
        st.session_state["batch_pointer"] = 0
        st.session_state["batch_last_results"] = {}
        st.sidebar.success("Batch-kö rensad.")

    df2 = df

    # Körning
    if run_next and queue and ptr < len(queue):
        to_run = queue[ptr : ptr + int(step)]
        df2, changed_map, msg_map = run_batch_update(
            df2, user_rates, to_run, use_runner,
            make_snapshot=False,
            save_cb=save_cb,
            recompute_cb=recompute_cb
        )
        # uppdatera pointer och logg
        st.session_state["batch_pointer"] = ptr + len(to_run)
        last = st.session_state.get("batch_last_results", {})
        last.update({k: {"changed": changed_map.get(k, []), "msg": msg_map.get(k, "")} for k in to_run})
        st.session_state["batch_last_results"] = last

    if run_all and queue and ptr < len(queue):
        to_run = queue[ptr:]
        df2, changed_map, msg_map = run_batch_update(
            df2, user_rates, to_run, use_runner,
            make_snapshot=False,
            save_cb=save_cb,
            recompute_cb=recompute_cb
        )
        st.session_state["batch_pointer"] = len(queue)
        last = st.session_state.get("batch_last_results", {})
        last.update({k: {"changed": changed_map.get(k, []), "msg": msg_map.get(k, "")} for k in to_run})
        st.session_state["batch_last_results"] = last

    # Senaste körlogg
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📒 Senaste körlogg**")
    last = st.session_state.get("batch_last_results", {})
    if not last:
        st.sidebar.write("–")
    else:
        # Visa senaste ~20
        items = list(last.items())[-20:]
        for tkr, info in items:
            ch = info.get("changed", [])
            msg = info.get("msg", "")
            st.sidebar.write(f"**{tkr}** — {msg}")
            if ch:
                st.sidebar.caption("Ändrade fält: " + ", ".join(ch))

    return df2
