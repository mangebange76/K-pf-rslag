# stockapp/batch.py
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Callable, Optional
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from .utils import (
    TS_FIELDS,
    ensure_schema,
    konvertera_typer,
    uppdatera_berakningar,
    add_oldest_ts_col,
)
from .storage import spara_data
from .sources import run_update_full, run_update_price_only

RunnerFunc = Callable[[str, pd.DataFrame, Dict[str, float]], Tuple[Dict, Dict, str]]

# =========================
# Hjälpare
# =========================

def _today_str() -> str:
    try:
        return datetime.now().strftime("%Y-%m-%d")
    except Exception:
        return str(datetime.now().date())

def _apply_vals_to_row(df: pd.DataFrame, row_idx: int, vals: Dict, source: str) -> Dict[str, List[str]]:
    """
    Skriver in fält från 'vals' till df-raden 'row_idx'. TS-fält stämplas ALLTID
    om nyckeln finns i vals (även om värdet blev samma), enligt din önskan.
    Sätter också Senast auto-uppdaterad & källa.
    Returnerar dict med 'changed' och 'same' listor för loggning.
    """
    changed, same = [], []
    for f, v in vals.items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f]
        old_s = "" if pd.isna(old) else str(old)
        new_s = "" if v is None else str(v)
        if old_s != new_s:
            df.at[row_idx, f] = v
            changed.append(f)
        else:
            same.append(f)
        # TS-stämpel om fältet spåras
        if f in TS_FIELDS:
            ts_col = TS_FIELDS[f]
            if ts_col in df.columns:
                df.at[row_idx, ts_col] = _today_str()

    # Auto metadata
    if "Senast auto-uppdaterad" in df.columns:
        df.at[row_idx, "Senast auto-uppdaterad"] = _today_str()
    if "Senast uppdaterad källa" in df.columns:
        df.at[row_idx, "Senast uppdaterad källa"] = source

    return {"changed": changed, "same": same}

def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """
    Bygger ordningslista av tickers baserat på valt sorteringsläge.
    """
    if sort_mode == "Äldst först (TS)":
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn"], ascending=[True, True])
        return [str(x).upper() for x in work["Ticker"].astype(str).tolist()]
    # default: A–Ö på bolagsnamn
    work = df.copy().sort_values(by=["Bolagsnamn", "Ticker"])
    return [str(x).upper() for x in work["Ticker"].astype(str).tolist()]

def _default_runner(ticker: str, df: pd.DataFrame, user_rates: Dict[str, float]) -> Tuple[Dict, Dict, str]:
    """Fallback-runner om ingen runner skickas in: Yahoo full."""
    vals, debug, source = run_update_full(ticker, df=df, user_rates=user_rates)
    return vals, debug, source

def _default_price_runner(ticker: str, df: pd.DataFrame, user_rates: Dict[str, float]) -> Tuple[Dict, Dict, str]:
    """Pris-only runner wrapper så signaturen matchar."""
    vals, debug = run_update_price_only(ticker)
    return vals, debug, "Yahoo (price-only)"


# =========================
# Batch-kärna
# =========================

def run_batch_update(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    tickers: List[str],
    runner: RunnerFunc,
    make_snapshot: bool = False,
    save_cb: Optional[Callable[[pd.DataFrame, bool], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame, Dict[str, float]], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Kör uppdatering för given ticker-lista med vald runner.
    Visar progress och 1/X-text i sidopanelen.
    Returnerar (df_updated, log)
    """
    total = len(tickers)
    pb = st.sidebar.progress(0.0)
    txt = st.sidebar.empty()

    # Logg
    log = {"changed": {}, "misses": {}, "queue_info": {"total": total}}

    # För säkerhets skull – schema/typer
    df = ensure_schema(df)
    df = konvertera_typer(df)

    for i, tkr in enumerate(tickers, start=1):
        tkr_u = str(tkr).strip().upper()
        txt.write(f"Uppdaterar {i}/{total}: {tkr_u}")
        try:
            vals, debug, source = runner(tkr_u, df, user_rates)
            # hitta rad
            mask = (df["Ticker"].astype(str).str.upper() == tkr_u)
            if not mask.any():
                log["misses"][tkr_u] = ["Ticker finns ej i tabellen"]
            else:
                ridx = df.index[mask][0]
                result = _apply_vals_to_row(df, ridx, vals or {}, source=source)
                # logg
                if result["changed"]:
                    log["changed"].setdefault(tkr_u, []).extend(result["changed"])
                else:
                    # även om inga changed – notera att vi stämplat TS/senast-auto
                    log["misses"].setdefault(tkr_u, []).append("(inga faktiska värdeändringar)")
        except Exception as e:
            log["misses"][tkr_u] = [f"error: {e}"]

        pb.progress(i / max(1, total))

    # Räkna om
    if recompute_cb:
        df = recompute_cb(df, user_rates)
    else:
        df = uppdatera_berakningar(df, user_rates)

    # Spara
    if save_cb:
        save_cb(df, make_snapshot)
    else:
        spara_data(df, do_snapshot=make_snapshot)

    st.sidebar.success("Batch klar.")
    return df, log


# =========================
# Batch-panel i sidopanelen
# =========================

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame, bool], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame, Dict[str, float]], pd.DataFrame]] = None,
    runner: Optional[RunnerFunc] = None,
    price_runner: Optional[RunnerFunc] = None,
) -> Optional[pd.DataFrame]:
    """
    UI-kontroller i sidopanelen för att bygga kö och köra batchar.
    Returnerar ev. uppdaterad df om körning gjordes, annars None.
    """
    if runner is None:
        runner = _default_runner
    if price_runner is None:
        price_runner = _default_price_runner

    st.sidebar.subheader("🛠️ Batch-körning")

    sort_mode = st.sidebar.selectbox("Ordning", ["A–Ö (bolagsnamn)", "Äldst först (TS)"])
    batch_size = st.sidebar.number_input("Batchstorlek", min_value=1, max_value=200, value=10, step=1)

    col_a, col_b, col_c = st.sidebar.columns(3)
    with col_a:
        if st.button("Bygg kö"):
            order = _pick_order(df, sort_mode)
            st.session_state["batch_queue"] = order
            st.session_state["batch_pos"] = 0
            st.sidebar.success(f"Kö byggd ({len(order)} tickers).")
    with col_b:
        if st.button("Töm kö"):
            st.session_state["batch_queue"] = []
            st.session_state["batch_pos"] = 0
            st.sidebar.info("Kö tömd.")
    with col_c:
        make_snapshot = st.checkbox("Snapshot före skrivning", value=False, key="batch_snap")

    # Status
    queue = st.session_state.get("batch_queue", [])
    pos = int(st.session_state.get("batch_pos", 0))
    left = max(0, len(queue) - pos)
    if queue:
        st.sidebar.caption(f"Köstatus: {pos}/{len(queue)} (kvar: {left})")
    else:
        st.sidebar.caption("Ingen aktiv kö.")

    # Kör pris-only nästa N
    col1, col2 = st.sidebar.columns(2)
    updated_df: Optional[pd.DataFrame] = None

    with col1:
        if st.button("📈 Kör pris (N)"):
            if not queue or pos >= len(queue):
                st.sidebar.warning("Kön är tom.")
            else:
                to_run = queue[pos: pos + int(batch_size)]
                df2, log = run_batch_update(
                    df, user_rates, to_run, price_runner,
                    make_snapshot=False,  # pris-only: ingen snapshot
                    save_cb=save_cb, recompute_cb=recompute_cb
                )
                st.session_state["batch_pos"] = pos + len(to_run)
                st.session_state["last_batch_log"] = log
                updated_df = df2

    with col2:
        if st.button("🔄 Kör full (N)"):
            if not queue or pos >= len(queue):
                st.sidebar.warning("Kön är tom.")
            else:
                to_run = queue[pos: pos + int(batch_size)]
                df2, log = run_batch_update(
                    df, user_rates, to_run, runner,
                    make_snapshot=bool(st.session_state.get("batch_snap", False)),
                    save_cb=save_cb, recompute_cb=recompute_cb
                )
                st.session_state["batch_pos"] = pos + len(to_run)
                st.session_state["last_batch_log"] = log
                updated_df = df2

    # Kör hela kön
    if st.sidebar.button("🏁 Kör hela kön (full)"):
        if not queue or pos >= len(queue):
            st.sidebar.warning("Kön är tom.")
        else:
            to_run = queue[pos:]
            df2, log = run_batch_update(
                df, user_rates, to_run, runner,
                make_snapshot=bool(st.session_state.get("batch_snap", False)),
                save_cb=save_cb, recompute_cb=recompute_cb
            )
            st.session_state["batch_pos"] = len(queue)
            st.session_state["last_batch_log"] = log
            updated_df = df2

    # Visa 1/X text separat (om vi nyss körde något visar run_batch_update redan live-status)
    if queue:
        st.sidebar.write(f"**1/X-status:** {pos}/{len(queue)} (kvar {left})")

    return updated_df
