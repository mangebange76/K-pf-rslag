# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# typer
RunnerFn = Callable[[str], Tuple[Dict, Dict]]  # -> (vals, debug)
SaveFn   = Callable[[pd.DataFrame], None]
RecompFn = Callable[[pd.DataFrame], pd.DataFrame]

# -----------------------------
# Hjälp
# -----------------------------

def _now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _oldest_any_ts(row: pd.Series) -> pd.Timestamp | None:
    dates = []
    for c in row.index:
        if str(c).startswith("TS_"):
            s = str(row.get(c, "")).strip()
            if s:
                try:
                    d = pd.to_datetime(s, errors="coerce")
                    if pd.notna(d):
                        dates.append(d)
                except Exception:
                    pass
    return min(dates) if dates else None

def _add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
    work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return work

def _stamp_ts_for_field(df: pd.DataFrame, ridx: int, field: str):
    ts_col = f"TS_{field}"
    if ts_col in df.columns:
        df.at[ridx, ts_col] = _now_stamp()

def _apply_vals_to_row(df: pd.DataFrame, ridx: int, vals: Dict, source_label: str):
    """
    Skriver in vals till df-rad (oavsett om samma värde fanns),
    stämplar TS_<fält> om sådan kolumn finns, samt sätter auto-uppdaterad/källa.
    """
    for k, v in vals.items():
        if k not in df.columns:
            # skapa numerisk eller textkolumn on-the-fly
            if isinstance(v, (int, float, np.floating)):
                df[k] = 0.0
            else:
                df[k] = ""
        df.at[ridx, k] = v
        _apply_ts_if_exists(df, ridx, k)

    # meta
    if "Senast auto-uppdaterad" in df.columns:
        df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    else:
        df["Senast auto-uppdaterad"] = ""
        df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()

    if "Senast uppdaterad källa" in df.columns:
        df.at[ridx, "Senast uppdaterad källa"] = source_label
    else:
        df["Senast uppdaterad källa"] = ""
        df.at[ridx, "Senast uppdaterad källa"] = source_label

def _apply_ts_if_exists(df: pd.DataFrame, ridx: int, field: str):
    ts_name = f"TS_{field}"
    if ts_name in df.columns:
        df.at[ridx, ts_name] = _now_stamp()

# -----------------------------
# UI & Körning
# -----------------------------

def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    if sort_mode.startswith("Äldst"):
        work = _add_oldest_ts_col(df)
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"], ascending=[True, True])
    else:
        work = df.sort_values(by=["Bolagsnamn","Ticker"], ascending=[True, True])
    order = [str(r.get("Ticker","")).upper().strip() for _, r in work.iterrows() if str(r.get("Ticker","")).strip()]
    return order

def _ensure_state():
    st.session_state.setdefault("batch_queue", [])
    st.session_state.setdefault("batch_last_log", [])
    st.session_state.setdefault("batch_mode", "full")  # "full" eller "price"

def _append_log(msg: str):
    st.session_state["batch_last_log"].append(f"[{_now_stamp()}] {msg}")
    # cap
    if len(st.session_state["batch_last_log"]) > 300:
        st.session_state["batch_last_log"] = st.session_state["batch_last_log"][-300:]

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: SaveFn | None,
    recompute_cb: RecompFn | None,
    runner_full: RunnerFn,
    runner_price: RunnerFn,
) -> pd.DataFrame:
    """
    Sidebar-panelen för batch.
    Returnerar ev. uppdaterad df.
    """
    _ensure_state()
    queue: List[str] = st.session_state["batch_queue"]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧮 Batch-körning")

    # Val av runner & sort
    mode = st.sidebar.selectbox("Vad ska uppdateras?", ["Full auto","Endast kurs"], index=(0 if st.session_state["batch_mode"]=="full" else 1))
    st.session_state["batch_mode"] = "full" if mode.startswith("Full") else "price"

    sort_mode = st.sidebar.selectbox("Sortera", ["Äldst uppdaterade först (alla TS)","A–Ö (bolagsnamn)"], index=0)
    batch_size = int(st.sidebar.number_input("Storlek (skapa kö av topp N)", min_value=1, max_value=500, value=20, step=1))
    avoid_dupes = st.sidebar.checkbox("Ta bara de som inte redan ligger i kön", value=True)
    do_save = st.sidebar.checkbox("💾 Spara efter körning", value=True)

    col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
    with col_btn1:
        if st.button("Skapa kö"):
            order = _pick_order(df, sort_mode)
            pick = order[:batch_size]
            if avoid_dupes:
                pick = [t for t in pick if t not in queue]
            st.session_state["batch_queue"] = pick + queue  # lägg i front
            _append_log(f"Skapade kö med {len(pick)} tickers. Mode={st.session_state['batch_mode']}")
    with col_btn2:
        if st.button("Rensa kö"):
            st.session_state["batch_queue"] = []
            _append_log("Kö rensad.")
    with col_btn3:
        if st.button("Ta bort första"):
            if st.session_state["batch_queue"]:
                t = st.session_state["batch_queue"].pop(0)
                _append_log(f"Tog bort {t} från kö-toppen.")

    # Visa kö
    queue = st.session_state["batch_queue"]
    if queue:
        st.sidebar.write(f"**Kö ({len(queue)}):** " + ", ".join(queue[:20]) + ("..." if len(queue)>20 else ""))
    else:
        st.sidebar.info("Kön är tom.")

    # Körningsknappar
    n_run = int(st.sidebar.number_input("Antal att köra (vid 'Kör N')", min_value=1, max_value=200, value=10, step=1))
    col_run1, col_run2, col_run3 = st.sidebar.columns(3)
    do1 = doN = doAll = False
    with col_run1:
        do1 = st.button("▶️ Kör 1")
    with col_run2:
        doN = st.button(f"⏩ Kör {n_run}")
    with col_run3:
        doAll = st.button("🏁 Kör alla")

    runner = runner_full if st.session_state["batch_mode"] == "full" else runner_price

    # Körning
    if do1 or doN or doAll:
        to_do = 1 if do1 else (n_run if doN else len(queue))
        to_do = min(to_do, len(queue))
        if to_do <= 0:
            st.sidebar.info("Inget att köra.")
            return df

        bar = st.sidebar.progress(0.0)
        stat = st.sidebar.empty()
        done = 0

        run_list = list(queue[:to_do])
        for tk in run_list:
            # pop från queue-toppen
            if not st.session_state["batch_queue"]:
                break
            head = st.session_state["batch_queue"].pop(0)
            if head != tk:
                # desync; säkerställ att vi kör rätt ändå
                tk = head

            # kör runner
            try:
                vals, dbg = runner(tk)
            except Exception as e:
                _append_log(f"{tk}: runner error: {type(e).__name__}: {e}")
                vals, dbg = {}, {"error": f"{type(e).__name__}: {e}"}

            # applicera i df
            ridxs = df.index[df["Ticker"].astype(str).str.upper().str.strip() == tk]
            if len(ridxs) == 0:
                _append_log(f"{tk}: hittades inte i tabellen.")
            else:
                ridx = int(ridxs[0])
                source_name = dbg.get("runner", "batch")
                _apply_vals_to_row(df, ridx, vals, source_label=f"Batch ({source_name})")

            done += 1
            bar.progress(done / float(to_do))
            stat.write(f"Kört {done}/{to_do} – kvar i kö: {len(st.session_state['batch_queue'])}")

        # efter körning: räkna om + spara
        if recompute_cb:
            try:
                df = recompute_cb(df)
            except Exception as e:
                _append_log(f"Recompute error: {type(e).__name__}: {e}")
        if do_save and save_cb:
            try:
                save_cb(df)
                _append_log(f"Sparade efter körning ({done} st).")
            except Exception as e:
                _append_log(f"Save error: {type(e).__name__}: {e}")

    # Logg
    st.sidebar.markdown("**Senaste batchlogg**")
    logs = st.session_state.get("batch_last_log", [])
    if logs:
        st.sidebar.code("\n".join(logs[-20:]))

    return df
