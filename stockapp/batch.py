# stockapp/batch.py
# -*- coding: utf-8 -*-
"""
Batch & uppdateringsmotor.

Publika funktioner:
- runner_price(df, ticker, user_rates) -> (df_out, logmsg)
- runner_full(df, ticker, user_rates)  -> (df_out, logmsg)
- build_batch_order(df, mode)          -> [tickers]
- run_batch(df, queue, mode, user_rates, save_cb=None, save_every=5) -> (df_out, loglist)
- sidebar_batch_controls(df, user_rates, default_batch_size=10, save_cb=None, recompute_cb=None) -> df_out

Design:
- Robusta kolumnnamn: stöd för både "Kurs" och "Aktuell kurs".
- TS-stämpling via utils.stamp_fields_ts (skapar TS-kolumner om de saknas).
- Orchestrator (SEC/FMP/Yahoo) används om tillgänglig, annars fallback till Yahoo live price.
- Batch: progress 1/X i sidopanel, autospar var N:e (save_every) och efter avslut.
- Ingen experimental_rerun; endast st.rerun efter knappar.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .utils import (
    add_oldest_ts_col,
    parse_date,
    safe_float,
    stamp_fields_ts,
)

# Källor
try:
    from .fetchers.yahoo import get_live_price as _yahoo_price
except Exception:
    _yahoo_price = None  # type: ignore

try:
    # Förväntat API: run_update_full(df, ticker, user_rates) -> (df_out, logmsg) eller df_out
    from .fetchers.orchestrator import run_update_full as _run_update_full
except Exception:
    _run_update_full = None  # type: ignore


# ---------------------------- Hjälpare ----------------------------

def _find_row_index_for_ticker(df: pd.DataFrame, ticker: str) -> Optional[pd.Index]:
    if "Ticker" not in df.columns:
        return None
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    idx = df.index[mask]
    return idx if len(idx) > 0 else None


def _set_price_and_stamp(df: pd.DataFrame, row_idx: pd.Index, price: float) -> Tuple[pd.DataFrame, List[str]]:
    """
    Sätt pris i 'Kurs' om finns, annars 'Aktuell kurs'.
    Stämpla motsvarande TS-kolumn(er). Returnerar (df, stämplade fält).
    """
    stamped: List[str] = []
    if "Kurs" in df.columns:
        df.loc[row_idx, "Kurs"] = float(price)
        stamped.append("Kurs")
    elif "Aktuell kurs" in df.columns:
        df.loc[row_idx, "Aktuell kurs"] = float(price)
        stamped.append("Aktuell kurs")
    # Om båda kolumnerna finns – håll dem i synk
    if "Kurs" in df.columns and "Aktuell kurs" in df.columns:
        df.loc[row_idx, "Aktuell kurs"] = float(price)
        if "Aktuell kurs" not in stamped:
            stamped.append("Aktuell kurs")

    if stamped:
        df = stamp_fields_ts(df, stamped, ts_suffix=" TS")
    return df, stamped


# ----------------------- Publika uppdaterare ----------------------

def runner_price(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """
    Uppdatera ENDAST kurs för en ticker. Yahoo-fallback används om orchestrator saknas.
    """
    ridx = _find_row_index_for_ticker(df, ticker)
    if ridx is None:
        return df, f"{ticker}: Ticker finns inte i tabellen."

    if _yahoo_price is None:
        return df, f"{ticker}: Yahoo-källa saknas."

    try:
        price = _yahoo_price(str(ticker))
        if price is None or price <= 0:
            return df, f"{ticker}: Pris saknas."
        df2, stamped = _set_price_and_stamp(df, ridx, float(price))
        if not stamped:
            return df2, f"{ticker}: Pris uppdaterat (ingen TS-kolumn hittades)."
        return df2, f"{ticker}: Pris uppdaterat ({', '.join(stamped)})."
    except Exception as e:
        return df, f"{ticker}: Fel vid prisuppdatering – {e}"


def runner_full(df: pd.DataFrame, ticker: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """
    Full uppdatering via orchestrator om tillgänglig, annars fallback till runner_price.
    """
    if _run_update_full is None:
        return runner_price(df, ticker, user_rates)

    try:
        out = _run_update_full(df, ticker, user_rates)  # type: ignore
        if isinstance(out, tuple) and len(out) == 2:
            df2, msg = out
            return df2, str(msg)
        if isinstance(out, pd.DataFrame):
            return out, f"{ticker}: Full uppdatering OK."
        return df, f"{ticker}: Orchestrator returnerade oväntat svar."
    except Exception as e:
        df2, _ = runner_price(df, ticker, user_rates)
        return df2, f"{ticker}: Full uppdatering misslyckades ({e}). Använde pris-fallback."


# --------------------- Batch-ordning & körning --------------------

def build_batch_order(df: pd.DataFrame, mode: str = "Äldst först") -> List[str]:
    """
    Välj ordning på tickers:
      - "Äldst först": minsta (äldsta) TS först (NA först)
      - "A–Ö": alfabetiskt stigande
      - "Z–A": alfabetiskt fallande
    Robust om TS-kolumner saknas – fallbackar till "Ticker".
    """
    work = df.copy()
    if "Ticker" not in work.columns:
        return []

    if mode == "Äldst först":
        try:
            work = add_oldest_ts_col(work, dest_col="__oldest_ts__")
        except Exception:
            # Fallback: försök bygga min-TS manuellt
            ts_cols = [c for c in work.columns if c.endswith(" TS") or c.upper().startswith("TS_") or "TS" in c]
            if ts_cols:
                for c in ts_cols:
                    work[c] = work[c].apply(parse_date)
                work["__oldest_ts__"] = work[ts_cols].min(axis=1, skipna=True)
            else:
                work["__oldest_ts__"] = pd.NaT
        work = work.sort_values(by="__oldest_ts__", ascending=True, na_position="first")
    elif mode == "A–Ö":
        work = work.sort_values(by="Ticker", ascending=True)
    else:  # "Z–A"
        work = work.sort_values(by="Ticker", ascending=False)

    return work["Ticker"].astype(str).tolist()


def run_batch(
    df: pd.DataFrame,
    queue: List[str],
    mode: str,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    save_every: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Kör batch över given kö. Returnerar (df_out, loglist).
    - mode: "price" eller "full"
    - save_cb: callback som sparar df (t.ex. spara_data)
    - save_every: hur ofta vi autosparar
    """
    if not queue:
        return df, ["(tom kö)"]

    total = len(queue)
    bar = st.sidebar.progress(0, text=f"0/{total}")
    done = 0
    logs: List[str] = []
    work = df.copy()

    for tkr in list(queue):
        if mode == "price":
            work, msg = runner_price(work, tkr, user_rates)
        else:
            work, msg = runner_full(work, tkr, user_rates)

        done += 1
        bar.progress(done / total, text=f"{done}/{total}")
        logs.append(msg)

        # Poppa från session-kö om finns
        if "batch_queue" in st.session_state:
            st.session_state["batch_queue"] = [x for x in st.session_state["batch_queue"] if str(x).upper() != str(tkr).upper()]

        if save_cb and (done % max(1, int(save_every)) == 0):
            try:
                save_cb(work)
            except Exception as e:
                logs.append(f"[Autospara] Fel: {e}")

    # slut-spara
    if save_cb:
        try:
            save_cb(work)
        except Exception as e:
            logs.append(f"[Spara] Fel: {e}")

    return work, logs


# --------------------- Sidopanel / UI-komponent -------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    default_batch_size: int = 10,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Renderar batchpanelen i sidopanelen och returnerar ev. uppdaterat df.

    Parametrar:
    - default_batch_size: standardstorlek om ingen tidigare state finns
    - save_cb(df): funktion för att spara (t.ex. spara_data)
    - recompute_cb(df)->df: om du vill räkna om derivatkolumner efter batch
    """
    with st.sidebar.expander("⚙️ Batchuppdatering", expanded=True):
        # Initiera state
        st.session_state.setdefault("batch_queue", [])
        st.session_state.setdefault("batch_order_mode", "Äldst först")
        st.session_state.setdefault("batch_size", int(default_batch_size))

        st.session_state["batch_order_mode"] = st.selectbox(
            "Sortering", ["Äldst först", "A–Ö", "Z–A"],
            index=["Äldst först", "A–Ö", "Z–A"].index(st.session_state["batch_order_mode"])
        )
        st.session_state["batch_size"] = int(st.number_input(
            "Antal i batch", min_value=1, max_value=500, value=int(st.session_state["batch_size"])
        ))

        if st.button("Skapa batchkö"):
            order = build_batch_order(df, st.session_state["batch_order_mode"])
            queue = [t for t in order if t not in st.session_state["batch_queue"]]
            st.session_state["batch_queue"] = queue[: st.session_state["batch_size"]]
            st.toast(f"Skapade batch ({len(st.session_state['batch_queue'])} tickers).")
            st.rerun()

        if st.session_state["batch_queue"]:
            st.caption("Kö:")
            st.code(", ".join(st.session_state["batch_queue"]), language="text")

        col1, col2 = st.columns(2)
        out_df = df
        with col1:
            if st.button("Kör batch – endast kurs", use_container_width=True):
                out_df, log = run_batch(df, st.session_state["batch_queue"], mode="price", user_rates=user_rates, save_cb=save_cb)
                for ln in log:
                    st.write("• " + ln)
                if recompute_cb:
                    try:
                        out_df = recompute_cb(out_df)
                    except Exception as e:
                        st.warning(f"Recompute-fel: {e}")
                st.session_state["batch_queue"] = []
                st.rerun()
        with col2:
            if st.button("Kör batch – full", use_container_width=True):
                out_df, log = run_batch(df, st.session_state["batch_queue"], mode="full", user_rates=user_rates, save_cb=save_cb)
                for ln in log:
                    st.write("• " + ln)
                if recompute_cb:
                    try:
                        out_df = recompute_cb(out_df)
                    except Exception as e:
                        st.warning(f"Recompute-fel: {e}")
                st.session_state["batch_queue"] = []
                st.rerun()

    return out_df
