# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# För runner-upplösning: dessa definieras i stockapp/sources.py om du har dem.
try:
    from .sources import get_runner_by_name  # returnerar en callable eller None
except Exception:
    get_runner_by_name = None

# ---- Hjälpare: äldst-ts, datumtolkning -------------------------------------

_TS_CAND_COLS = [
    "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
    "TS_Omsättning idag","TS_Omsättning nästa år",
]

def _parse_date(s: str) -> Optional[datetime]:
    try:
        s = (s or "").strip()
        if not s:
            return None
        return pd.to_datetime(s, errors="coerce").to_pydatetime()
    except Exception:
        return None

def _oldest_any_ts_value(row: pd.Series) -> Optional[datetime]:
    dates = []
    for c in _TS_CAND_COLS:
        if c in row and str(row[c]).strip():
            d = _parse_date(row[c])
            if d:
                dates.append(d)
    if dates:
        return min(dates)
    return None

def _normalize_ticker(x: str) -> str:
    return str(x or "").strip().upper()

# ---- Plocka körordning ------------------------------------------------------

def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    work = df.copy()
    work["Ticker"] = work["Ticker"].astype(str).str.upper().str.strip()
    work = work[work["Ticker"] != ""]

    if sort_mode.startswith("A–Ö"):
        order = list(work.sort_values(by=["Bolagsnamn","Ticker"], na_position="last")["Ticker"])
    elif sort_mode.startswith("Äldst"):
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts_value, axis=1)
        work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
        order = list(work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])["Ticker"])
    elif sort_mode.startswith("Slump"):
        order = list(work.sample(frac=1.0, random_state=int(time.time()))["Ticker"])
    else:
        order = list(work["Ticker"])

    # Order-bevarande avdubblering så vi aldrig kör samma ticker två gånger
    order = list(dict.fromkeys(order))
    return order

# ---- Runner-upplösning ------------------------------------------------------

def _resolve_runner(runner_choice: str) -> Callable:
    # 1) Försök använda registrerad runner i sources
    if get_runner_by_name is not None:
        fn = get_runner_by_name(runner_choice)
        if fn is not None:
            return fn

    # 2) Fallback: noop
    def _noop_runner(df: pd.DataFrame, ticker: str, **kwargs):
        return {
            "ticker": ticker,
            "attempted": True,
            "fetched_fields": [],
            "changed_fields": [],
            "saved": False,
            "error": None,
            "status": "inga data (noop runner)",
            "duration_s": 0.0,
        }
    return _noop_runner

# ---- Batch-UI + körning -----------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Optional[Dict[str,float]] = None,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    default_sort: str = "Äldst uppdaterade först (alla fält)",
    default_runner_choice: str = "Full auto",
    default_batch_size: int = 10,
    commit_every: int = 0,
) -> pd.DataFrame:
    """
    Sidopanel för batchkörning:
      • Sortering: Äldst / A–Ö / Slump
      • Välj runner: "Full auto", "Endast kurs", "Endast P/S + aktier" (kräver GET i sources)
      • Cooldown: exkludera tickers försöka senaste N timmar
      • 1/X-progress och status per ticker
      • Logg i st.session_state["_last_batch_log"]
      • Uppdaterar 'Senast auto-försök' + 'Senast auto-status' i DF
    """
    st.sidebar.subheader("🛠️ Batch-uppdatering")

    sort_mode = st.sidebar.selectbox(
        "Sortering",
        ["Äldst uppdaterade först (alla fält)", "A–Ö (bolagsnamn)", "Slumpvis"],
        index=0 if default_sort.startswith("Äldst") else 1 if default_sort.startswith("A–Ö") else 2,
        key="batch_sort_mode"
    )

    runner_choice = st.sidebar.selectbox(
        "Körläge",
        ["Full auto", "Endast kurs", "Endast P/S + aktier"],
        index=0 if default_runner_choice.startswith("Full") else 1 if "kurs" in default_runner_choice.lower() else 2,
        key="batch_runner_choice"
    )

    batch_size = st.sidebar.number_input(
        "Antal tickers i batch",
        min_value=1, max_value=200, value=int(default_batch_size), step=1, key="batch_size_input"
    )

    cool_hours = st.sidebar.number_input(
        "Exkludera tickers försökta inom (timmar)",
        min_value=0, max_value=168, value=24, step=1, key="batch_cooldown_h"
    )
    mark_attempt = st.sidebar.checkbox("Markera 'Senast auto-försök' även om inget ändras", value=True, key="batch_mark_attempt")
    show_preview = st.sidebar.checkbox("Visa förhandslista", value=False, key="batch_preview")

    # Bygg ordning
    tickers_ordered = _pick_order(df, sort_mode)

    # Exkludera nyligen försökta om så önskas
    if cool_hours > 0 and "Senast auto-försök" in df.columns:
        cutoff = datetime.now() - timedelta(hours=int(cool_hours))

        def _ok(tkr: str) -> bool:
            r = df[df["Ticker"].astype(str).str.upper().str.strip() == _normalize_ticker(tkr)]
            if r.empty:
                return True
            s = str(r.iloc[0].get("Senast auto-försök","")).strip()
            d = _parse_date(s)
            if not d:
                return True
            return d < cutoff

        tickers_ordered = [t for t in tickers_ordered if _ok(t)]

    to_run = tickers_ordered[: int(batch_size)]

    if show_preview:
        st.sidebar.write("**Körlista (förhandsvisning):**")
        st.sidebar.code(", ".join(to_run) if to_run else "(tom)")

    # Körning
    if st.sidebar.button("🔄 Kör batch nu", key="btn_run_batch"):
        runner = _resolve_runner(runner_choice)
        results: List[Dict] = []
        n = len(to_run)
        if n == 0:
            st.sidebar.info("Inget att köra (körlista tom).")
            st.session_state["_last_batch_log"] = []
            return df

        prog = st.sidebar.progress(0.0, text=f"Startar… 0/{n}")
        status_line = st.sidebar.empty()

        t_start = time.time()
        for i, tkr in enumerate(to_run):
            t0 = time.time()
            try:
                res = runner(df, tkr, user_rates=user_rates)
                if not isinstance(res, dict):
                    res = {"error": "runner returnerade ogiltigt resultat", "saved": False, "changed_fields": [], "fetched_fields": []}
            except Exception as e:
                res = {"error": str(e), "saved": False, "changed_fields": [], "fetched_fields": []}

            fetched = list(res.get("fetched_fields") or [])
            changed = list(res.get("changed_fields") or [])
            saved   = bool(res.get("saved", False))
            err     = res.get("error")

            if err:
                status = f"fel: {err}"
            else:
                if saved and changed:
                    status = "sparad"
                elif fetched and not changed:
                    status = "oförändrat"
                else:
                    status = res.get("status") or ("inga data" if not fetched else "oförändrat")

            # Markera försök & status
            if mark_attempt or saved:
                mask = (df["Ticker"].astype(str).str.upper().str.strip() == _normalize_ticker(tkr))
                if mask.any():
                    try:
                        df.loc[mask, "Senast auto-försök"] = datetime.now().strftime("%Y-%m-%d")
                        df.loc[mask, "Senast auto-status"] = status
                    except Exception:
                        pass

            dur = time.time() - t0
            results.append({
                "Ticker": tkr,
                "Status": status,
                "Fält hämtade": ", ".join(fetched) if fetched else "",
                "Fält ändrade": ", ".join(changed) if changed else "",
                "Ändringar (#)": len(changed),
                "Sparad?": "Ja" if saved else "Nej",
                "Tid (s)": round(dur,2),
            })

            prog.progress((i+1)/n, text=f"Kör… {i+1}/{n} ({tkr})")
            status_line.write(f"{i+1}/{n}: {tkr} — {status}")

            if save_cb and commit_every and ((i+1) % int(commit_every) == 0):
                try:
                    save_cb(df)
                except Exception as e:
                    st.sidebar.warning(f"Kunde inte del-spara efter {i+1} poster: {e}")

        tot_s = round(time.time() - t_start, 2)
        prog.progress(1.0, text=f"Klar! {n}/{n} — {tot_s}s")
        st.sidebar.success(f"Batch klar ({n} tickers, {tot_s}s).")

        # Spara rapport
        st.session_state["_last_batch_log"] = results

        # Summering
        try:
            df_log = pd.DataFrame(results)
            changed_n = int((df_log["Sparad?"] == "Ja").sum())
        except Exception:
            changed_n = sum(1 for r in results if r.get("Sparad?") == "Ja")
        st.sidebar.write(f"**Sparade poster:** {changed_n} / {n}")

        # Slutlig save
        if save_cb:
            try:
                save_cb(df)
            except Exception as e:
                st.sidebar.warning(f"Kunde inte spara slutligt: {e}")

    if st.sidebar.button("🧹 Återställ batchlogg (minne)", key="btn_reset_batch_hist"):
        st.session_state["_last_batch_log"] = []
        st.sidebar.info("Batchlogg i minnet nollställd.")

    return df
