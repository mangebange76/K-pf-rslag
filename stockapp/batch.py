# -*- coding: utf-8 -*-
"""
stockapp.batch
--------------
Sidopanel för batchuppdateringar.

Publik funktion:
- sidebar_batch_controls(df, user_rates, save_cb=None, recompute_cb=None, runner=None, key_prefix="batch")
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
                # Ny signatur: kan vara (df, t, user_rates) -> (df2, changed, msg)
                out = run_update_prices(df, t, user_rates)
                if isinstance(out, tuple) and len(out) == 3:
                    return out
                # fallback: antar bara df returneras
                return out, True, f"Pris uppdaterat: {t}"
            except Exception as e:
                return df, False, f"{t}: Fel (pris): {e}"

        # full uppdatering
        # försök combo -> full
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
) -> pd.DataFrame:
    """
    Visar kontroller i sidopanelen för att:
      1) välja ordning (A–Ö eller Äldst först (TS))
      2) välja batchstorlek
      3) skapa en batchkö (tickers)
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
        # om modulen inte får en explicit runner: titta i session_state, annars default
        runner = st.session_state.get("_runner") or _default_runner()

    # --- UI ---
    st.sidebar.markdown("### Batch-uppdatering")

    sort_mode = st.sidebar.radio(
        "Sortera",
        options=["A-Ö (Ticker)", "Äldst först (TS)"],
        index=1,
        key=f"{key_prefix}_sortmode",
        help="Välj ordning för hur tickers plockas till batchen.",
    )

    batch_size = st.sidebar.number_input(
        "Batchstorlek",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
        key=f"{key_prefix}_size",
    )

    # Uppdateringstyp
    upd_mode = st.sidebar.radio(
        "Uppdateringstyp",
        options=["Endast kurs", "Full uppdatering"],
        index=0,
        key=f"{key_prefix}_updmode",
    )
    mode_flag = "price" if upd_mode == "Endast kurs" else "full"

    # Bygg ordning & skapa batch
    if st.sidebar.button("Skapa batch", key=f"{key_prefix}_build"):
        order = _pick_order(df, sort_mode)
        # Om vi redan har en kö, försök skapa nästa fönster i ordningen
        # Ta tickers som inte redan ligger i kön
        existing = set(st.session_state["_batch_queue"])
        remaining = [t for t in order if t not in existing]
        if not remaining:
            remaining = order[:]  # om allt redan kört, börja om
        st.session_state["_batch_queue"] = remaining[: int(batch_size)]
        st.sidebar.success(f"Skapade batch med {len(st.session_state['_batch_queue'])} tickers.")

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
