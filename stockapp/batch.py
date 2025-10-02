# stockapp/batch.py
# -*- coding: utf-8 -*-
"""
Batch-hantering:
- sidebar_batch_controls(df, user_rates, ...)
- run_batch_update(df, user_rates, tickers, mode="price", ...)

Stödjer:
- Sortering: "Äldst först" (kräver TS-kolumner), "A–Ö", "Z–A"
- Progressbar (i/X) i sidopanelen
- Logg i sidopanelen
- Sparar var 5:e ticker (eller via save_cb)

Runner-API (frivilligt att ge in):
  runner.price(df, ticker, user_rates) -> (df_out, logmsg)
  runner.full(df, ticker, user_rates)  -> (df_out, logmsg)

Fallbacks:
  - Pris: Yahoo (stockapp.fetchers.yahoo.get_live_price)
  - Full: Orchestrator (stockapp.fetchers.orchestrator.run_update_full) om finns,
          annars faller tillbaka till pris-runnern.
"""

from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import math
import pandas as pd
import streamlit as st

from .utils import add_oldest_ts_col, stamp_fields_ts, safe_float
from .storage import spara_data

# --- Frivilliga fetchers (robusta imports) -------------------------------
try:
    from .fetchers.yahoo import get_live_price as _yahoo_price
except Exception:  # pragma: no cover
    _yahoo_price = None  # type: ignore

try:
    from .fetchers.orchestrator import run_update_full as _orchestrator_full
except Exception:  # pragma: no cover
    _orchestrator_full = None  # type: ignore


# -------------------------------------------------------------------------
# Interna runners (fallback)
# -------------------------------------------------------------------------
def _fallback_price_runner(df: pd.DataFrame, tkr: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """Uppdatera ENDAST 'Kurs' via Yahoo som fallback."""
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns inte i tabellen"

    if _yahoo_price is None:
        return df, "Yahoo-priskälla saknas"

    try:
        px = _yahoo_price(str(tkr))
        if px and safe_float(px) > 0:
            df.loc[ridx, "Kurs"] = float(px)
            # stämpla TS-kolumn (generisk suffix " TS")
            df = stamp_fields_ts(df, ["Kurs"], ts_suffix=" TS")
            return df, "OK (pris)"
        return df, "Pris saknas"
    except Exception as e:  # pragma: no cover
        return df, f"Fel i prisuppdatering: {e}"


def _fallback_full_runner(df: pd.DataFrame, tkr: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """Uppdatera ALLT via orchestrator om den finns, annars pris."""
    if _orchestrator_full is None:
        # Ingen orkestrator – fallback till pris
        return _fallback_price_runner(df, tkr, user_rates)

    try:
        out = _orchestrator_full(df, tkr, user_rates)  # förväntas ge (df, msg) eller df
        if isinstance(out, tuple) and len(out) == 2:
            df2, msg = out
            return df2, str(msg)
        if isinstance(out, pd.DataFrame):
            return out, "OK (full)"
        return df, "Orchestrator: oväntat svar"
    except Exception as e:  # pragma: no cover
        # Faller tillbaka till pris om full hämtning fallerar
        df2, _ = _fallback_price_runner(df, tkr, user_rates)
        return df2, f"Full-uppdatering misslyckades: {e} → tog pris"


# -------------------------------------------------------------------------
# Publika API:t
# -------------------------------------------------------------------------
def run_batch_update(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    tickers: Iterable[str],
    mode: str = "price",
    save_every: int = 5,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    runner: Optional[object] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Kör batch mot angivna tickers.

    Args:
        df: DataFrame med minst kolumn 'Ticker'
        user_rates: valutakurser
        tickers: iterable med tickers
        mode: "price" eller "full"
        save_every: spara var n:te uppdatering
        save_cb: om du vill ta kontroll över hur sparning görs
        runner: objekt med metoder 'price' och/eller 'full' (se modul-docstring)

    Returns:
        (df_out, log_lines)
    """
    work = df.copy()
    log: List[str] = []
    tickers = [str(t).upper() for t in tickers if str(t).strip()]

    # välj faktiska funktionspekare
    if runner and hasattr(runner, "price"):
        price_fn = getattr(runner, "price")
    else:
        price_fn = _fallback_price_runner

    if runner and hasattr(runner, "full"):
        full_fn = getattr(runner, "full")
    else:
        full_fn = _fallback_full_runner

    total = len(tickers)
    done = 0
    prog = st.sidebar.progress(0.0, text=f"0/{total}")

    for tkr in tickers:
        if mode == "price":
            work, msg = price_fn(work, tkr, user_rates)
        else:
            work, msg = full_fn(work, tkr, user_rates)
        done += 1
        prog.progress(done / max(1, total), text=f"{done}/{total}")
        log.append(f"{tkr}: {msg}")

        # spara periodiskt
        if save_every and done % save_every == 0:
            _save(work, save_cb)

    # slutlig sparning
    _save(work, save_cb)

    # logga i sidopanel
    st.sidebar.write("**Batchlogg**")
    for ln in log:
        st.sidebar.write("• " + ln)

    return work, log


def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    default_batch_size: int = 10,
    default_sort: str = "Äldst först",
    runner: Optional[object] = None,
) -> pd.DataFrame:
    """
    Sidopanelens batch-kontroller. Returnerar ev. uppdaterad df.

    Parametrar som hanterar kompatibilitet med tidigare anrop:
    - save_cb: externt sätt att spara df
    - recompute_cb: om vissa beräkningar ska köras efter batch
    - default_batch_size, default_sort: förinställningar
    - runner: se run_batch_update
    """
    with st.sidebar.expander("⚙️ Batch", expanded=True):
        # init state
        st.session_state.setdefault("batch_sort", default_sort)
        st.session_state.setdefault("batch_size", int(default_batch_size))
        st.session_state.setdefault("batch_queue", [])

        st.session_state["batch_sort"] = st.selectbox(
            "Sortering",
            ["Äldst först", "A–Ö", "Z–A"],
            index=["Äldst först", "A–Ö", "Z–A"].index(st.session_state["batch_sort"]),
        )
        st.session_state["batch_size"] = int(
            st.number_input("Antal i batch", min_value=1, max_value=500, value=int(st.session_state["batch_size"]))
        )

        if st.button("Skapa batchkö"):
            order = _pick_order(df, st.session_state["batch_sort"])
            # undvik dubbletter i kön
            existing = set([str(t).upper() for t in st.session_state["batch_queue"]])
            new_queue: List[str] = []
            for t in order:
                u = str(t).upper()
                if u not in existing:
                    new_queue.append(u)
                if len(new_queue) >= st.session_state["batch_size"]:
                    break
            st.session_state["batch_queue"] = new_queue
            st.toast(f"Skapade batchkö: {len(new_queue)} tickers.")

        if st.session_state["batch_queue"]:
            st.write("**Kö:** " + ", ".join(st.session_state["batch_queue"]))

        c1, c2 = st.columns(2)
        out_df = df
        with c1:
            if st.button("Kör batch – endast kurs", use_container_width=True):
                out_df, _ = run_batch_update(
                    df,
                    user_rates,
                    st.session_state["batch_queue"],
                    mode="price",
                    save_every=5,
                    save_cb=save_cb,
                    runner=runner,
                )
                st.session_state["batch_queue"] = []
        with c2:
            if st.button("Kör batch – full", use_container_width=True):
                out_df, _ = run_batch_update(
                    df,
                    user_rates,
                    st.session_state["batch_queue"],
                    mode="full",
                    save_every=5,
                    save_cb=save_cb,
                    runner=runner,
                )
                st.session_state["batch_queue"] = []

        # recompute efter batch om önskat
        if out_df is not df and recompute_cb:
            try:
                out_df = recompute_cb(out_df)
            except Exception as e:  # pragma: no cover
                st.warning(f"⚠️ Kunde inte köra efterberäkningar: {e}")

        return out_df


# -------------------------------------------------------------------------
# Hjälpare (lokalt)
# -------------------------------------------------------------------------
def _save(df: pd.DataFrame, save_cb: Optional[Callable[[pd.DataFrame], None]]) -> None:
    """Spara df med angiven callback eller direkt till Google Sheet."""
    try:
        if save_cb:
            save_cb(df)
        else:
            spara_data(df)
    except Exception as e:  # pragma: no cover
        st.warning(f"⚠️ Kunde inte spara vid batch: {e}")


def _pick_order(df: pd.DataFrame, mode: str) -> List[str]:
    """
    Tar fram en lista av tickers i önskad ordning.
    - "Äldst först" använder add_oldest_ts_col (kräver TS_* fält för att bli meningsfull)
    - "A–Ö" / "Z–A" sorterar alfabetiskt på 'Ticker'
    """
    if "Ticker" not in df.columns or df.empty:
        return []

    work = df.copy()
    work["Ticker"] = work["Ticker"].astype(str)

    if mode == "Äldst först":
        # Skapar en kolumn '__oldest_ts__' av minsta TS-datum vi hittar.
        work = add_oldest_ts_col(work, dest_col="__oldest_ts__")
        work = work.sort_values(by="__oldest_ts__", ascending=True, na_position="first")
    elif mode == "A–Ö":
        work = work.sort_values(by="Ticker", ascending=True)
    else:  # "Z–A"
        work = work.sort_values(by="Ticker", ascending=False)

    return work["Ticker"].tolist()
