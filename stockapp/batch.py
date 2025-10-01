# -*- coding: utf-8 -*-
"""
stockapp.batch
--------------
Sidopanel för batchuppdateringar.

Publik funktion:
- sidebar_batch_controls(
      df, user_rates,
      save_cb=None, recompute_cb=None, runner=None, key_prefix="batch",
      default_batch_size=20,
      default_sort_mode="Äldst först (TS)",
      default_upd_mode="Endast kurs",
  )
    * Visar batch-UI i sidopanelen
    * Returnerar ev. uppdaterad DataFrame (df)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from .config import FINAL_COLS, TS_FIELDS
from .utils import add_oldest_ts_col, ensure_schema


# ----------------------------------------------------
# Hjälp: välj ordning
# ----------------------------------------------------
def _pick_order(df: pd.DataFrame, mode: str) -> List[str]:
    """
    Returnerar en lista med tickers i vald ordning.
    mode: "A-Ö (Ticker)" eller "Äldst först (TS)"
    """
    work = ensure_schema(df, FINAL_COLS)
    if "Ticker" not in work.columns:
        return []

    if mode == "Äldst först (TS)":
        work = add_oldest_ts_col(work, TS_FIELDS, out_col="_oldest_ts")
        # sortera None först (saknar TS), sedan stigande datum
        work = work.sort_values(by="_oldest_ts", ascending=True, na_position="first")
        order = list(work["Ticker"].astype(str))
    else:
        # standard: A-Ö på ticker
        order = list(work["Ticker"].astype(str).sort_values())
    return order


# ----------------------------------------------------
# Hjälp: bygg default-runner om ingen skickas in
# ----------------------------------------------------
def _default_runner() -> Callable[[pd.DataFrame, str, Dict[str, float], str], Tuple[pd.DataFrame, bool, str]]:
    """
    Skapar en runner som använder orchestrator om den finns.
    Signatur: (df, ticker, user_rates, mode) -> (df2, changed:bool, msg:str)
    mode: "price" eller "full"
    """
    try:
        from .orchestrator import run_update_combo, run_update_full, run_update_prices
    except Exception:
        # Minimal no-op runner
        def _noop(df: pd.DataFrame, t: str, user_rates: Dict[str, float], mode: str):
            return df, False, f"Ingen orchestrator hittades – hoppade över {t}."
        return _noop

    def _runner(df: pd.DataFrame, t: str, user_rates: Dict[str, float], mode: str):
        if mode == "price":
            try:
                out = run_update_prices(df, t, user_rates)
                if isinstance(out, tuple) and len(out) == 3:
                    return out
                return out, True, f"Pris uppdaterat: {t}"
            except Exception as e:
                return df, False, f"{t}: Fel (pris): {e}"

        # full uppdatering
        try:
            out = run_update_combo(df, t, user_rates)
            if isinstance(out, tuple) and len(out) == 3:
                return out
            return out, True, f"Full uppdatering klar: {t}"
        except Exception:
            try:
                out = run_update_full(df, t, user_rates)
                if isinstance(out, tuple) and len(out) == 3:
                    return out
                return out, True, f"Full uppdatering klar: {t}"
            except Exception as e:
                return df, False, f"{t}: Fel (full): {e}"

    return _runner


# ----------------------------------------------------
# Publik: sidopanel för batch
# ----------------------------------------------------
def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    runner: Optional[Callable[[pd.DataFrame, str, Dict[str, float], str], Tuple[pd.DataFrame, bool, str]]] = None,
    key_prefix: str = "batch",
    default_batch_size: int = 20,
    default_sort_mode: str = "Äldst först (TS)",
    default_upd_mode: str = "Endast kurs",
) -> pd.DataFrame:
    """
    Visar kontroller i sidopanelen för att:
      1) välja ordning (A–Ö eller Äldst först (TS))
      2) välja batchstorlek (default kan styras)
      3) skapa en batchkö (tickers) med rullande cursor
      4) köra batchen (endast kurs eller full uppdatering)
    - Progressbar med text "i/X"
    - Logg sparas i st.session_state["_batch_log"]
    - Ingen st.experimental_rerun används

    Returnerar ev. uppdaterad df.
    """
    if "_batch_log" not in st.session_state:
        st.session_state["_batch_log"] = []
    if "_batch_queue" not in st.session_state:
        st.session_state["_batch_queue"] = []
    if runner is None:
        runner = st.session_state.get("_runner") or _default_runner()

    # --- UI ---
    st.sidebar.markdown("### Batch-uppdatering")

    sort_options = ["A-Ö (Ticker)", "Äldst först (TS)"]
    sort_idx = sort_options.index(default_sort_mode) if default_sort_mode in sort_options else 1
    sort_mode = st.sidebar.radio(
        "Sortera",
        options=sort_options,
        index=sort_idx,
        key=f"{key_prefix}_sortmode",
        help="Välj ordning för hur tickers plockas till batchen.",
    )

    batch_size = st.sidebar.number_input(
        "Batchstorlek",
        min_value=1,
        max_value=200,
        value=int(default_batch_size),
        step=1,
        key=f"{key_prefix}_size",
    )

    upd_options = ["Endast kurs", "Full uppdatering"]
    upd_idx = upd_options.index(default_upd_mode) if default_upd_mode in upd_options else 0
    upd_mode = st.sidebar.radio(
        "Uppdateringstyp",
        options=upd_options,
        index=upd_idx,
        key=f"{key_prefix}_updmode",
    )
    mode_flag = "price" if upd_mode == "Endast kurs" else "full"

    # Rullande cursor per sorteringsläge
    cursor_key = f"{key_prefix}_cursor_{sort_mode}"
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = 0

    # Bygg ordning & skapa batch (rullande fönster)
    if st.sidebar.button("Skapa batch", key=f"{key_prefix}_build"):
        order = _pick_order(df, sort_mode)
        if not order:
            st.sidebar.warning("Hittade inga tickers.")
        else:
            cur = int(st.session_state[cursor_key] or 0)
            n = int(batch_size)
            # Ta nästa fönster
            window = order[cur : cur + n]
            if not window:
                # wrap och börja om
                cur = 0
                window = order[cur : cur + n]
            # Uppdatera cursor (wrap om vi passerar slutet)
            cur = cur + len(window)
            if cur >= len(order):
                cur = 0
            st.session_state[cursor_key] = cur

            st.session_state["_batch_queue"] = window[:]
            st.sidebar.success(f"Skapade batch med {len(window)} tickers.")

    # Visa aktuell kö
    if st.session_state["_batch_queue"]:
        st.sidebar.write("**Kö:**", ", ".join(st.session_state["_batch_queue"]))
    else:
        st.sidebar.info("Kön är tom. Klicka **Skapa batch**.")

    # Kör batch
    if st.sidebar.button("Kör batch", key=f"{key_prefix}_run"):
        queue = list(st.session_state["_batch_queue"])
        n = len(queue)
        if n == 0:
            st.sidebar.warning("Ingen kö att köra.")
        else:
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            log_local: List[str] = []

            df2 = df.copy()
            for i, t in enumerate(queue, start=1):
                status.text(f"{i}/{n}: {t}")
                try:
                    df2, changed, msg = runner(df2, t, user_rates, mode_flag)
                    log_local.append(msg or f"{t}: klart.")
                except Exception as e:  # noqa: BLE001
                    log_local.append(f"{t}: Fel: {e}")
                prog.progress(i / n)

            # spara logg
            st.session_state["_batch_log"].extend(log_local)
            st.sidebar.success("Batch klar.")
            st.sidebar.write("\n".join(log_local[-10:]))

            # ev. recompute
            if recompute_cb is not None:
                try:
                    df2 = recompute_cb(df2)
                except Exception as e:  # noqa: BLE001
                    st.sidebar.error(f"Recompute-fel: {e}")

            # ev. spara
            if save_cb is not None:
                try:
                    save_cb(df2)
                except Exception as e:  # noqa: BLE001
                    st.sidebar.error(f"Spara-fel: {e}")

            # töm kö efter körning
            st.session_state["_batch_queue"] = []
            return df2

    # Töm batch
    if st.sidebar.button("Töm batch", key=f"{key_prefix}_clear"):
        st.session_state["_batch_queue"] = []
        st.sidebar.info("Kön tömd.")

    # Visa logg
    with st.sidebar.expander("Batchlogg (senaste 200 rader)", expanded=False):
        if st.session_state["_batch_log"]:
            st.text("\n".join(st.session_state["_batch_log"][-200:]))
        else:
            st.caption("Ingen logg ännu.")

    return df
