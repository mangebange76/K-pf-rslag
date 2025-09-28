# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import streamlit as st

from .sources import update_price_only, update_full_for_ticker

# Vilka TS-kolumner som räknas in i "äldst uppdaterade"
_TS_COLS = [
    "TS_Utestående aktier",
    "TS_P/S", "TS_P/S Q1", "TS_P/S Q2", "TS_P/S Q3", "TS_P/S Q4",
    "TS_Omsättning idag", "TS_Omsättning nästa år",
]


def _ensure_col(df: pd.DataFrame, col: str, default=""):
    if col not in df.columns:
        df[col] = default


def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for c in _TS_COLS:
        if c in row and str(row[c]).strip():
            d = pd.to_datetime(str(row[c]), errors="coerce")
            if pd.notna(d):
                dates.append(d)
    if not dates:
        return None
    return min(dates)


def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """
    Returnerar en ticker-lista i önskad sorteringsordning:
      - "A–Ö (bolagsnamn)" → sortera på Bolagsnamn, Ticker
      - "Äldst uppdaterade först (alla fält)" → sortera på äldsta TS bland spårade fält
    """
    if df.empty or "Ticker" not in df.columns:
        return []

    if sort_mode.startswith("Äldst"):
        work = df.copy()
        for c in _TS_COLS:
            _ensure_col(work, c, "")
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        # Saknade TS behandlas som “äldst” → lägg dem först
        work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("1900-01-01"))
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn", "Ticker"], ascending=[True, True, True])
        return [str(t).upper() for t in work["Ticker"].astype(str).tolist()]
    else:
        work = df.copy()
        _ensure_col(work, "Bolagsnamn", "")
        _ensure_col(work, "Ticker", "")
        work = work.sort_values(by=["Bolagsnamn", "Ticker"])
        return [str(t).upper() for t in work["Ticker"].astype(str).tolist()]


def _runner_from_choice(choice: str) -> Callable[[pd.DataFrame, str], Tuple[pd.DataFrame, Dict]]:
    """
    Mappar val i UI → runner-funktion.
    """
    if choice == "Kurs (snabb)":
        return update_price_only
    # default: Full auto
    return update_full_for_ticker


def run_batch_update(
    df: pd.DataFrame,
    tickers: List[str],
    *,
    runner: Callable[[pd.DataFrame, str], Tuple[pd.DataFrame, Dict]],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    commit_every: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Kör runner(df, ticker) för varje ticker i listan, visar progress i sidopanelen.
    - commit_every > 0 → spara till Sheets var N:te post via save_cb (för att skydda mot nätverksfel).
    Returnerar (df, log) där log innehåller 'ok', 'fail', 'details'.
    """
    total = len(tickers)
    if total == 0:
        return df, {"ok": [], "fail": [], "details": []}

    bar = st.sidebar.progress(0.0)
    stat = st.sidebar.empty()

    ok: List[str] = []
    fail: List[str] = []
    details: List[Dict] = []

    for i, tkr in enumerate(tickers, start=1):
        try:
            df, info = runner(df, tkr)
            ok.append(tkr)
            details.append({"ticker": tkr, "status": "ok", "info": info})
        except Exception as e:
            fail.append(tkr)
            details.append({"ticker": tkr, "status": "error", "error": str(e)})
        # progress + 1/X
        bar.progress(i / total)
        stat.write(f"**Batch:** {i}/{total} – {tkr}")

        # ev. del-skrivning för säkerhet
        if commit_every > 0 and (i % commit_every == 0) and callable(save_cb):
            try:
                save_cb(df)
            except Exception:
                # swallow; vi vill inte krascha hela batchen pga ett skrivfel mitt i
                pass

    bar.progress(1.0)
    stat.write("**Batch klar**")
    # slutlig skrivning
    if callable(save_cb):
        try:
            save_cb(df)
        except Exception:
            pass

    return df, {"ok": ok, "fail": fail, "details": details}


def sidebar_batch_controls(
    df: pd.DataFrame,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    *,
    default_sort: str = "Äldst uppdaterade först (alla fält)",
    default_runner_choice: str = "Full auto",
    default_batch_size: int = 10,
    commit_every: int = 0,
) -> pd.DataFrame:
    """
    Komplett batch-panel i sidopanelen:
      - Välj sortering (A–Ö / Äldst)
      - Skapa kö på topp-N från sorteringen
      - Välj runner (Kurs / Full auto)
      - Kör nästa N eller Kör alla
      - 1/X-indikator, progress-bar
      - Loggar ok/fail
    """
    st.sidebar.subheader("🛠️ Batch-körning")

    if "_batch_queue" not in st.session_state:
        st.session_state["_batch_queue"] = []  # list[str]
    if "_batch_pos" not in st.session_state:
        st.session_state["_batch_pos"] = 0
    if "_batch_log" not in st.session_state:
        st.session_state["_batch_log"] = {"ok": [], "fail": [], "details": []}
    if "_batch_sort" not in st.session_state:
        st.session_state["_batch_sort"] = default_sort
    if "_batch_runner_choice" not in st.session_state:
        st.session_state["_batch_runner_choice"] = default_runner_choice

    # 1) Välj sorteringsläge
    sort_mode = st.sidebar.selectbox(
        "Sortering för kö",
        ["Äldst uppdaterade först (alla fält)", "A–Ö (bolagsnamn)"],
        index=0 if default_sort.startswith("Äldst") else 1,
        key="_batch_sort",
    )

    # 2) Plocka ordning och låt användaren välja topp-N
    order = _pick_order(df, sort_mode)
    st.sidebar.caption(f"{len(order)} tickers tillgängliga i vald sortering.")
    top_n = st.sidebar.number_input("Hur många att lägga i kö?", min_value=1, max_value=max(1, len(order)), value=min(default_batch_size, max(1,len(order))), step=1)

    # 3) Runner-val
    runner_choice = st.sidebar.radio("Hämtningstyp", ["Full auto", "Kurs (snabb)"], index=0 if default_runner_choice=="Full auto" else 1, key="_batch_runner_choice")

    colb1, colb2 = st.sidebar.columns(2)
    with colb1:
        if st.button("➕ Skapa kö (från sortering)"):
            st.session_state["_batch_queue"] = order[:int(top_n)]
            st.session_state["_batch_pos"] = 0
            st.session_state["_batch_log"] = {"ok": [], "fail": [], "details": []}
            st.sidebar.success(f"Skapade kö med {len(st.session_state['_batch_queue'])} tickers.")
    with colb2:
        if st.button("🧹 Rensa kö"):
            st.session_state["_batch_queue"] = []
            st.session_state["_batch_pos"] = 0
            st.session_state["_batch_log"] = {"ok": [], "fail": [], "details": []}
            st.sidebar.info("Kö rensad.")

    # 4) Status för kö
    q = st.session_state["_batch_queue"]
    pos = st.session_state["_batch_pos"]
    remaining = max(0, len(q) - pos)
    st.sidebar.write(f"**Kö:** {pos}/{len(q)} (återstår {remaining})")

    # 5) Körkontroller
    run_n = st.sidebar.number_input("Kör nästa N", min_value=1, max_value=max(1, remaining) if remaining>0 else 1, value=min(default_batch_size, max(1, remaining)) if remaining>0 else 1, step=1)

    ckr1, ckr2 = st.sidebar.columns(2)
    with ckr1:
        run_next = st.button("▶️ Kör nästa N")
    with ckr2:
        run_all = st.button("⏩ Kör alla kvar")

    # Runner-funktion
    runner_fn = _runner_from_choice(runner_choice)

    # 6) Exekvera batch
    if (run_next or run_all) and remaining > 0:
        to_run = q[pos:(len(q) if run_all else pos+int(run_n))]
        df2, log = run_batch_update(
            df, to_run, runner=runner_fn, save_cb=save_cb, commit_every=commit_every
        )
        # uppdatera session-state
        st.session_state["_batch_pos"] += len(to_run)
        # ackumulera logg
        base = st.session_state["_batch_log"]
        base["ok"].extend(log.get("ok", []))
        base["fail"].extend(log.get("fail", []))
        base["details"].extend(log.get("details", []))
        st.session_state["_batch_log"] = base
        st.sidebar.success(f"Klar: +{len(to_run)} körda (totalt ok: {len(base['ok'])}, fail: {len(base['fail'])})")
        return df2  # returnera uppdaterad DF

    # 7) Liten loggvisning
    log = st.session_state["_batch_log"]
    if log["ok"] or log["fail"]:
        with st.sidebar.expander("📒 Körlogg (sammanfattning)"):
            st.write(f"✅ OK ({len(log['ok'])}): {', '.join(log['ok'][:20])}{'…' if len(log['ok'])>20 else ''}")
            st.write(f"⚠️ Fail ({len(log['fail'])}): {', '.join(log['fail'][:20])}{'…' if len(log['fail'])>20 else ''}")

    return df  # oförändrad DF om ingen körning gjordes
