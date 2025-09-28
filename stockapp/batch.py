# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd

from .utils import add_oldest_ts_col, ensure_ticker_col, recompute_derived


RunnerFn = Callable[[pd.DataFrame, str, Dict[str, float]], Tuple[pd.DataFrame, List[str], str]]
SaveFn = Callable[[pd.DataFrame], None]


# ------------------------------------------------------------
# Interna hjälpmetoder
# ------------------------------------------------------------
def _ensure_batch_state():
    if "_batch_order" not in st.session_state:
        st.session_state["_batch_order"] = []       # lista med tickers i köordning
    if "_batch_cursor" not in st.session_state:
        st.session_state["_batch_cursor"] = 0       # index i _batch_order
    if "_batch_log" not in st.session_state:
        st.session_state["_batch_log"] = {          # körlogg
            "changed": {},      # ticker -> [kolumn1, kolumn2, ...]
            "misses": {},       # ticker -> [orsak]
            "errors": {},       # ticker -> "error text"
            "runs": 0           # antal körda tickers i denna session
        }
    if "_runner" not in st.session_state:
        st.session_state["_runner"] = None          # kan bindas utifrån


def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    df = ensure_ticker_col(df)
    if sort_mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn", "Ticker"])
        return [t for t in work["Ticker"].tolist() if str(t).strip()]
    # default A–Ö
    work = df.sort_values(by=[c for c in ["Bolagsnamn","Ticker"] if c in df.columns])
    return [t for t in work["Ticker"].tolist() if str(t).strip()]


def _diff_row(before: pd.Series, after: pd.Series) -> List[str]:
    changed = []
    for c in after.index:
        b = str(before.get(c, ""))
        a = str(after.get(c, ""))
        if b != a:
            changed.append(c)
    return changed


def _default_runner(df: pd.DataFrame, ticker: str, user_rates: Dict[str,float]) -> Tuple[pd.DataFrame, List[str], str]:
    """
    No-op fallback om ingen runner är satt. Returnerar oförändrat df.
    """
    return df, [], "no-runner"


def _get_runner(explicit_runner: Optional[RunnerFn]) -> RunnerFn:
    if explicit_runner is not None:
        return explicit_runner
    if st.session_state.get("_runner") is not None:
        return st.session_state["_runner"]
    return _default_runner


def _position_text() -> str:
    order = st.session_state["_batch_order"]
    cur = st.session_state["_batch_cursor"]
    total = len(order)
    # 1/X text (cursor är 0-baserad, men visa 1-baserat)
    pos = min(cur + 1, max(total, 1))
    return f"{pos}/{max(total,1)}"


# ------------------------------------------------------------
# Publika metoder
# ------------------------------------------------------------
def run_batch_update(df: pd.DataFrame,
                     user_rates: Dict[str,float],
                     tickers: List[str],
                     runner: Optional[RunnerFn] = None,
                     save_cb: Optional[SaveFn] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Kör given runner på en lista av tickers. Visar progress (1/X).
    Returnerar (df, logdict).
    """
    _ensure_batch_state()
    run = _get_runner(runner)

    if not tickers:
        st.info("Inget att köra.")
        return df, st.session_state["_batch_log"]

    progress = st.sidebar.progress(0.0, text=f"Kör batch … {0}/{len(tickers)}")
    status = st.sidebar.empty()

    log = st.session_state["_batch_log"]
    changed_any = False

    for i, tkr in enumerate(tickers, start=1):
        tkr = str(tkr).strip().upper()
        status.write(f"Kör {i}/{len(tickers)}: **{tkr}**  – batchpos { _position_text() }")
        try:
            # Snapshot "before"
            if "Ticker" in df.columns and (df["Ticker"] == tkr).any():
                ridx = df.index[df["Ticker"] == tkr][0]
                before = df.loc[ridx].copy()
            else:
                before = pd.Series({})

            df, changed_cols, status_str = run(df, tkr, user_rates)
            changed_cols = changed_cols or []

            # Snapshot "after" + diff om runner ej lämnat lista
            if "Ticker" in df.columns and (df["Ticker"] == tkr).any():
                ridx2 = df.index[df["Ticker"] == tkr][0]
                after = df.loc[ridx2]
                if not changed_cols and not before.empty:
                    changed_cols = _diff_row(before, after)

            if changed_cols:
                log["changed"].setdefault(tkr, [])
                for c in changed_cols:
                    if c not in log["changed"][tkr]:
                        log["changed"][tkr].append(c)
                changed_any = True
            else:
                log["misses"][tkr] = [status_str or "(ingen ändring)"]

            log["runs"] += 1
        except Exception as e:
            log["errors"][tkr] = str(e)

        progress.progress(i/len(tickers), text=f"Kör batch … {i}/{len(tickers)}")

    # Recompute derivat + spara en gång
    df = recompute_derived(df)
    if save_cb is not None and changed_any:
        try:
            save_cb(df)
            st.sidebar.success("Ändringar sparade.")
        except Exception as e:
            st.sidebar.warning(f"Kunde inte spara: {e}")
    elif not changed_any:
        st.sidebar.info("Ingen faktisk ändring – ingen sparning.")

    return df, log


def sidebar_batch_controls(df: pd.DataFrame,
                           user_rates: Dict[str,float],
                           save_cb: Optional[SaveFn] = None,
                           default_sort: str = "Äldst uppdaterade först (alla fält)",
                           runner: Optional[RunnerFn] = None) -> pd.DataFrame:
    """
    Batchpanelen i sidopanelen. Skapar en kö och låter dig köra stegvis,
    N-steg eller hela kön. Visar 1/X och en enkel körlogg.
    """
    _ensure_batch_state()

    st.sidebar.subheader("🚚 Batch-körning")

    sort_mode = st.sidebar.selectbox(
        "Sortera",
        ["Äldst uppdaterade först (alla fält)", "A–Ö (bolagsnamn)"],
        index=0 if default_sort.startswith("Äldst") else 1,
        key="batch_sort_mode"
    )
    batch_size = st.sidebar.number_input("Batchstorlek (antal i kön)", min_value=1, max_value=500, value=20, step=1, key="batch_size")

    colA, colB = st.sidebar.columns([1,1])
    with colA:
        if st.button("🧮 Skapa kö", key="btn_make_queue"):
            order_all = _pick_order(df, sort_mode)
            # Ta bort redan körda tickers från loggens "changed/misses/errors" för att inte låsa oss
            already_done = set()
            for d in ("changed","misses","errors"):
                already_done |= set(st.session_state["_batch_log"].get(d, {}).keys())
            # Filtrera bort redan körda så vi får "nästa" batch
            remaining = [t for t in order_all if t not in already_done]
            st.session_state["_batch_order"] = remaining[:int(batch_size)]
            st.session_state["_batch_cursor"] = 0
            st.sidebar.success(f"Kö skapad ({len(st.session_state['_batch_order'])} tickers).")
    with colB:
        if st.button("🧹 Återställ kö", key="btn_reset_queue"):
            st.session_state["_batch_order"] = []
            st.session_state["_batch_cursor"] = 0
            st.sidebar.info("Kö återställd.")

    order = st.session_state["_batch_order"]
    cur = st.session_state["_batch_cursor"]
    total = len(order)

    if total > 0:
        st.sidebar.caption(f"🔢 Köläge: { _position_text() }  —  kvar: {max(total - cur, 0)}")

        # Kör-enheter
        c1, c2, c3 = st.sidebar.columns([1,1,1])
        with c1:
            if st.button("▶️ Kör nästa", key="btn_run_next"):
                to_run = order[cur:cur+1]
                if to_run:
                    df2, log = run_batch_update(df, user_rates, to_run, runner=runner, save_cb=save_cb)
                    df[:] = df2  # uppdatera referensen
                    st.session_state["_batch_cursor"] = min(cur + 1, total)
        with c2:
            if st.button("⏭️ Kör 5", key="btn_run_5"):
                to_run = order[cur:cur+5]
                if to_run:
                    df2, log = run_batch_update(df, user_rates, to_run, runner=runner, save_cb=save_cb)
                    df[:] = df2
                    st.session_state["_batch_cursor"] = min(cur + len(to_run), total)
        with c3:
            if st.button("⏩ Kör hela kön", key="btn_run_all"):
                to_run = order[cur:]
                if to_run:
                    df2, log = run_batch_update(df, user_rates, to_run, runner=runner, save_cb=save_cb)
                    df[:] = df2
                    st.session_state["_batch_cursor"] = total

        # Lista nästa 10 i kön
        with st.sidebar.expander("📋 Köinnehåll (nästa 10)", expanded=False):
            nxt = order[cur:cur+10]
            if nxt:
                st.write(", ".join(nxt))
            else:
                st.caption("— tomt —")

    # Körlogg
    with st.sidebar.expander("📒 Körlogg", expanded=False):
        log = st.session_state["_batch_log"]
        st.caption(f"Antal körningar i sessionen: {int(log.get('runs',0))}")
        if log.get("changed"):
            st.markdown("**Ändringar**")
            st.json(log["changed"])
        if log.get("misses"):
            st.markdown("**Missar**")
            st.json(log["misses"])
        if log.get("errors"):
            st.markdown("**Fel**")
            st.json(log["errors"])

    return df
