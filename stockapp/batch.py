# stockapp/batch.py
# -*- coding: utf-8 -*-
"""
Batch-körningar i sidopanelen:
- Skapa batch (A–Ö eller Äldst TS först), välj batch-storlek (N).
- Välj läge: Endast kurs eller Full uppdatering.
- Kör 'Nästa' för att processa 1 ticker i taget (minskar API-stryk).
- Visar 1/X-progress, körlogg, samt ändringsfält.
- Spara sker efter varje lyckad uppdatering (via save_cb).
"""

from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd

from .utils import apply_auto_updates_to_row, add_oldest_ts_col, uppdatera_berakningar
from .sources import fetch_full_ticker, fetch_price_only


# -------------------------------------------------------------------
# Hjälpare för urvalsordning
# -------------------------------------------------------------------
def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """
    Returnerar ticker-lista i vald ordning.
    sort_mode: "A–Ö (bolagsnamn)" eller "Äldst uppdaterade först (alla fält)"
    """
    if "Ticker" not in df.columns:
        return []
    if sort_mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn"], ascending=[True, True])
        return [str(t).upper() for t in work["Ticker"].tolist() if str(t).strip()]
    else:
        work = df.copy()
        if "Bolagsnamn" in work.columns:
            work = work.sort_values(by=["Bolagsnamn", "Ticker"])
        else:
            work = work.sort_values(by=["Ticker"])
        return [str(t).upper() for t in work["Ticker"].tolist() if str(t).strip()]


# -------------------------------------------------------------------
# Standard-runner (kan ersättas via parameter)
# -------------------------------------------------------------------
def _default_runner(ticker: str, price_only: bool = False) -> Tuple[Dict, Dict, str]:
    """
    Kör en uppdatering för en ticker.
    Returnerar (vals, debug, source_name)
    """
    if price_only:
        vals = fetch_price_only(ticker)
        dbg = {"mode": "price_only"}
        return vals, dbg, "Auto (Pris via Yahoo)"
    else:
        vals, dbg = fetch_full_ticker(ticker)
        return vals, dbg, "Auto (SEC/Yahoo + fallback)"


# -------------------------------------------------------------------
# UI + körning
# -------------------------------------------------------------------
def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Callable[[pd.DataFrame], None],
    recompute_cb: Callable[[pd.DataFrame, Dict[str, float]], pd.DataFrame],
    runner: Optional[Callable[[str, bool], Tuple[Dict, Dict, str]]] = None,
) -> pd.DataFrame:
    """
    Bygger sidopanel för batch-körningar.
    - df: DataFrame för bolagslistan (referens; uppdateras löpande).
    - user_rates: valutakurser (ej använda här men finns för future use).
    - save_cb(df): funktion som sparar df i Sheets.
    - recompute_cb(df, user_rates) -> df: räknar om derivat (riktkurser etc.).
    - runner(ticker, price_only) -> (vals, debug, source_name): egen uppdaterare.
      Om None används _default_runner.
    """
    st.sidebar.subheader("🛠️ Batch-uppdatering")

    # Init state
    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []      # list[str]
    if "batch_done" not in st.session_state:
        st.session_state.batch_done = 0        # antal körda i aktuell batch
    if "batch_total" not in st.session_state:
        st.session_state.batch_total = 0
    if "last_batch_log" not in st.session_state:
        st.session_state.last_batch_log = []   # list[dict]
    if "batch_mode_price_only" not in st.session_state:
        st.session_state.batch_mode_price_only = False

    sort_mode = st.sidebar.selectbox(
        "Ordning för batch",
        ["Äldst uppdaterade först (alla fält)", "A–Ö (bolagsnamn)"],
        index=0
    )
    batch_size = int(st.sidebar.number_input("Batch-storlek", min_value=1, max_value=200, value=20, step=1))
    st.sidebar.checkbox("🧪 Endast kurs (snabb)", key="batch_mode_price_only", help="Hämtar bara aktuell kurs/valuta/namn, inte P/S m.m.")

    col_b0, col_b1, col_b2 = st.sidebar.columns([1,1,1])
    with col_b0:
        if st.button("🧾 Skapa ny batch"):
            order = _pick_order(df, sort_mode)
            st.session_state.batch_queue = order[:batch_size]
            st.session_state.batch_done = 0
            st.session_state.batch_total = len(st.session_state.batch_queue)
            st.session_state.last_batch_log = []
            st.sidebar.success(f"Batch skapad ({st.session_state.batch_total} st).")
    with col_b1:
        if st.button("♻️ Återställ batch"):
            st.session_state.batch_queue = []
            st.session_state.batch_done = 0
            st.session_state.batch_total = 0
            st.session_state.last_batch_log = []
            st.sidebar.info("Batch återställd.")
    with col_b2:
        if st.button("➡️ Kör nästa"):
            _run_one_in_batch(df, user_rates, save_cb, recompute_cb, runner)

    # Progress + lista
    total = st.session_state.batch_total
    done = st.session_state.batch_done
    if total > 0:
        st.sidebar.progress(done / total)
        st.sidebar.caption(f"⏳ Körning: {done}/{total}")

    if st.session_state.batch_queue:
        st.sidebar.markdown("**I kö:** " + ", ".join(st.session_state.batch_queue[:10]) + (" ..." if len(st.session_state.batch_queue) > 10 else ""))
    else:
        st.sidebar.markdown("_Kön är tom._")

    # Körlogg
    if st.session_state.last_batch_log:
        st.sidebar.markdown("### 📒 Körlogg (senaste)")
        for item in st.session_state.last_batch_log[-8:][::-1]:
            # item: {"ticker":..., "status": "changed"/"nochange"/"error", "fields": [...], "msg": "..."}
            t = item.get("ticker", "?")
            stt = item.get("status", "?")
            if stt == "changed":
                st.sidebar.success(f"{t}: {stt} → {', '.join(item.get('fields', [])) or '(fält?)'}")
            elif stt == "nochange":
                st.sidebar.info(f"{t}: inga ändringar")
            else:
                st.sidebar.error(f"{t}: fel: {item.get('msg','okänt')}")

    return df


def _run_one_in_batch(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Callable[[pd.DataFrame], None],
    recompute_cb: Callable[[pd.DataFrame, Dict[str, float]], pd.DataFrame],
    runner: Optional[Callable[[str, bool], Tuple[Dict, Dict, str]]],
) -> None:
    """
    Plockar första ticker ur batch-kön, kör uppdateringen, skriver in värden,
    räknar om derivat och sparar. Loggar utfallet.
    """
    if not st.session_state.batch_queue:
        st.sidebar.info("Kön är tom.")
        return

    ticker = st.session_state.batch_queue.pop(0)
    price_only = bool(st.session_state.get("batch_mode_price_only", False))

    # Hitta radindex
    if "Ticker" not in df.columns:
        st.session_state.last_batch_log.append({"ticker": ticker, "status": "error", "msg": "Kolumnen 'Ticker' saknas."})
        return
    mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
    if not mask.any():
        st.session_state.last_batch_log.append({"ticker": ticker, "status": "error", "msg": "Ticker hittades inte i tabellen."})
        st.session_state.batch_done += 1
        return
    ridx = df.index[mask][0]

    # Runner
    run = runner if runner is not None else _default_runner

    try:
        vals, debug, source_name = run(ticker, price_only)
    except TypeError:
        # Om runnern har annan signatur, försök utan price_only
        try:
            tmp = run(ticker)  # kan vara (vals, debug) eller (vals, debug, source)
            if isinstance(tmp, tuple) and len(tmp) == 3:
                vals, debug, source_name = tmp
            elif isinstance(tmp, tuple) and len(tmp) == 2:
                vals, debug = tmp
                source_name = "Auto (custom runner)"
            else:
                vals, debug, source_name = (tmp or {}), {}, "Auto (custom runner)"
        except Exception as e:
            st.session_state.last_batch_log.append({"ticker": ticker, "status": "error", "msg": f"Runner-fel: {e}"})
            st.session_state.batch_done += 1
            return
    except Exception as e:
        st.session_state.last_batch_log.append({"ticker": ticker, "status": "error", "msg": f"Körning misslyckades: {e}"})
        st.session_state.batch_done += 1
        return

    # Skriv in vals → apply_auto_updates_to_row (sätter TS & källa om ändrat)
    try:
        changed_fields: Dict[str, List[str]] = {}
        changed = apply_auto_updates_to_row(
            df=df,
            row_idx=ridx,
            new_vals=vals or {},
            source=source_name,
            changes_map=changed_fields
        )
        # Om du vill stämpla "Senast auto-uppdaterad" även utan diff:
        #   sätt changed=True här under särskilda villkor, men då behöver
        #   även apply_auto_updates_to_row stödja "force_ts".

        # Recompute derivat
        df2 = recompute_cb(df, user_rates)

        # Spara endast om ändrat (skonsamt mot Sheets)
        if changed:
            save_cb(df2)
            st.session_state.last_batch_log.append({
                "ticker": ticker,
                "status": "changed",
                "fields": changed_fields.get(df2.at[ridx, "Ticker"], []),
            })
        else:
            st.session_state.last_batch_log.append({
                "ticker": ticker,
                "status": "nochange",
                "fields": list((vals or {}).keys()),
            })

        # Uppdatera räknare
        st.session_state.batch_done += 1

    except Exception as e:
        st.session_state.last_batch_log.append({"ticker": ticker, "status": "error", "msg": f"Skriv/spara-fel: {e}"})
        st.session_state.batch_done += 1
        return
