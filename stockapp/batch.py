# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
from typing import Callable, List, Dict, Any

from .utils import add_oldest_ts_col, now_stamp
from .config import TS_FIELDS

RunnerFn = Callable[[pd.DataFrame, int, dict], tuple[pd.DataFrame, bool, str]]

def _pick_order(df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if sort_mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        return work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).reset_index(drop=True)
    else:
        return df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)

def _ensure_queue():
    if "_batch_queue" not in st.session_state:
        st.session_state["_batch_queue"] = []   # list of tickers
    if "_batch_pos" not in st.session_state:
        st.session_state["_batch_pos"] = 0
    if "_batch_log" not in st.session_state:
        st.session_state["_batch_log"] = []     # list of dict results

def _format_log_entry(tkr: str, ok: bool, msg: str, changed_fields: list[str] | None = None) -> Dict[str, Any]:
    return {
        "ticker": tkr,
        "ok": bool(ok),
        "msg": str(msg or ""),
        "changed": changed_fields or [],
        "ts": now_stamp(),
    }

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: dict,
    save_cb: Callable[[pd.DataFrame], None],
    recompute_cb: Callable[[pd.DataFrame], pd.DataFrame],
    runner: RunnerFn,
) -> pd.DataFrame:
    """
    Batch-panel i sidopanelen.
    - Sortering: A–Ö eller Äldst först
    - Batchstorlek
    - Skapa kö → sparas i session_state
    - Kör N nu → processar från aktuell position
    - Visar progress (1/X) och en kort logg
    """
    _ensure_queue()

    st.sidebar.markdown("### 📦 Batchuppdatering")
    sort_mode = st.sidebar.radio("Sortera", ["Äldst först (alla TS)", "A–Ö (bolagsnamn)"], horizontal=False)
    batch_size = st.sidebar.number_input("Batchstorlek", min_value=1, max_value=200, value=20, step=1)

    colq1, colq2, colq3 = st.sidebar.columns([1,1,1])
    with colq1:
        if st.button("🧰 Skapa kö"):
            ordered = _pick_order(df, sort_mode)
            st.session_state["_batch_queue"] = list(ordered["Ticker"].astype(str))
            st.session_state["_batch_pos"] = 0
            st.session_state["_batch_log"] = []
            st.sidebar.success(f"Kö skapad med {len(st.session_state['_batch_queue'])} tickers.")

    with colq2:
        if st.button("⏭️ Skippa 1"):
            if st.session_state["_batch_pos"] < len(st.session_state["_batch_queue"]):
                st.session_state["_batch_pos"] += 1

    with colq3:
        if st.button("♻️ Återställ kö"):
            st.session_state["_batch_queue"] = []
            st.session_state["_batch_pos"] = 0
            st.session_state["_batch_log"] = []
            st.sidebar.info("Batch-kön återställd.")

    # Visa status
    tq = len(st.session_state["_batch_queue"])
    pos = st.session_state["_batch_pos"]
    if tq > 0:
        st.sidebar.write(f"Köstatus: {pos}/{tq} (nästa: {st.session_state['_batch_queue'][pos] if pos < tq else '—'})")

    if st.sidebar.button(f"▶️ Kör {int(batch_size)} nu"):
        # Kör upp till batch_size, eller till kön tar slut
        run_count = 0
        total = min(batch_size, max(0, tq - pos))
        if total == 0:
            st.sidebar.warning("Ingen kö att köra. Skapa en kö först.")
        else:
            prog = st.sidebar.progress(0.0, text=f"0/{total}")
            for i in range(total):
                cur_idx = st.session_state["_batch_pos"]
                if cur_idx >= len(st.session_state["_batch_queue"]):
                    break
                tkr = st.session_state["_batch_queue"][cur_idx]
                # Hitta index i df
                try:
                    ridx = df.index[df["Ticker"].astype(str) == str(tkr)][0]
                except Exception:
                    st.session_state["_batch_log"].append(_format_log_entry(tkr, False, "Ticker hittades inte i tabellen"))
                    st.session_state["_batch_pos"] += 1
                    prog.progress((i+1)/total, text=f"{i+1}/{total}")
                    continue

                # Kör vald runner
                try:
                    df, changed, msg = runner(df, ridx, user_rates)
                    st.session_state["_batch_log"].append(_format_log_entry(tkr, True, msg, []))
                except Exception as e:
                    changed = False
                    st.session_state["_batch_log"].append(_format_log_entry(tkr, False, f"Fel: {e}"))

                st.session_state["_batch_pos"] += 1
                run_count += 1
                prog.progress((i+1)/total, text=f"{i+1}/{total}")

            # Recompute & spara efter en batch
            try:
                df = recompute_cb(df)
            except Exception:
                pass
            try:
                save_cb(df)
            except Exception:
                pass
            st.sidebar.success(f"Klar: körde {run_count} av {total} i batchen.")

    # Visa senaste 10 loggrader
    if st.session_state["_batch_log"]:
        st.sidebar.markdown("**Senaste batchlogg (10):**")
        for row in st.session_state["_batch_log"][-10:]:
            emoji = "✅" if row["ok"] else "⚠️"
            st.sidebar.write(f"{emoji} {row['ticker']}: {row['msg']}")

    return df
