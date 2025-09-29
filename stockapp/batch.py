# stockapp/batch.py
# -*- coding: utf-8 -*-
"""
Batch-körningar (sidopanel) + hjälpmetoder för att uppdatera enskilda rader.

Publik:
- sidebar_batch_controls(df, user_rates, save_cb=None, recompute_cb=None, runner_price=None, runner_full=None)
  Visar batch-panel i sidopanelen:
    * Skapa kö (A–Ö eller Äldst först), välj antal
    * Välj uppdateringstyp: Endast kurs eller Full uppdatering
    * Kör nästa / Kör alla i kö / Återställ kö
  Returnerar uppdaterad df (samma referens-objekt som skickades in)

Interna hjälpare:
- _pick_order(df, mode) -> List[str]
- _oldest_any_ts(row)   -> Optional[pd.Timestamp]
- _add_oldest_ts_col(df)
- _stamp_ts_for_field(df, idx, field, when=None)
- _note_auto_update(df, idx, source)
- _apply_updates_to_row(df, idx, new_vals, source, log_changed, always_stamp=False)

Runner default:
- Om runner_price/runner_full ej anges används stockapp.sources.run_update_price_only / run_update_full
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from .config import TS_FIELDS, FINAL_COLS
from .utils import now_stamp
from . import sources as _sources
from .storage import spara_data

# ---------------------------------------------------------------------------
# Hjälpare: TS & ordning
# ---------------------------------------------------------------------------

def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for c in TS_FIELDS.values():
        if c in row and str(row[c]).strip():
            try:
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None

def _add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_oldest_any_ts"] = out.apply(_oldest_any_ts, axis=1)
    out["_oldest_any_ts_fill"] = out["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return out

def _pick_order(df: pd.DataFrame, mode: str) -> List[str]:
    """
    Returnerar en lista med tickers i vald ordning.
    mode: "A–Ö (bolagsnamn)" eller "Äldst uppdaterade först"
    """
    work = df.copy()
    if mode.startswith("Äldst"):
        work = _add_oldest_ts_col(work)
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn"], ascending=[True, True])
    else:
        work = work.sort_values(by=["Bolagsnamn", "Ticker"], ascending=[True, True])
    # Endast unika tickers och icke-tomma
    tickers = [str(t).upper().strip() for t in work["Ticker"].tolist() if str(t).strip()]
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

# ---------------------------------------------------------------------------
# Hjälpare: stämpling & skrivning
# ---------------------------------------------------------------------------

def _stamp_ts_for_field(df: pd.DataFrame, row_idx: int, field: str, when: Optional[str] = None):
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    date_str = when if when else now_stamp()
    try:
        df.at[row_idx, ts_col] = date_str
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _apply_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: Dict[str, object],
    source: str,
    log_changed: Dict[str, List[str]],
    always_stamp: bool = False,
) -> bool:
    """
    Skriver endast meningsfulla fält från new_vals till df.
    - Numeriskt: >0 för P/S, P/S Qx, Utestående aktier, Aktuell kurs; 0 får skrivas för t.ex. utdelning om den uttryckligen finns med
    - Sträng: icke-tom
    - Stämplar TS_ för fält i TS_FIELDS när de skrivs (eller always_stamp=True om nyckeln finns i new_vals).
    - Sätter auto-uppdaterad och källa om något ändrades OCH/ELLER always_stamp=True och nyckeln finns (så datum hålls färskt).

    Returnerar True om någonting skrevs eller stämplades.
    """
    ticker = str(df.at[row_idx, "Ticker"]).upper().strip()
    changed_fields: List[str] = []
    stamped_only: List[str] = []

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f] if f in df.columns else None

        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # Fält som inte ska vara negativa/meningslösa
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier","Aktuell kurs"]:
                write_ok = (float(v) > 0)
            else:
                # ok att skriva noll i vissa fall (utdelning t.ex.)
                write_ok = (float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        if write_ok:
            # Uppdatera om skillnad
            if (old is None) or (str(old) != str(v)):
                df.at[row_idx, f] = v
                changed_fields.append(f)
                # Stämpla ev. TS_ för spårat fält
                if f in TS_FIELDS:
                    _stamp_ts_for_field(df, row_idx, f)
            else:
                # Identiskt — om always_stamp, stämpla TS_ ändå
                if always_stamp and f in TS_FIELDS:
                    _stamp_ts_for_field(df, row_idx, f)
                    stamped_only.append(f)
        else:
            # Inget skriv — men stämpla TS_ om always_stamp och f finns i new_vals
            if always_stamp and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                stamped_only.append(f)

    any_effect = bool(changed_fields or stamped_only)
    if any_effect:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            log_changed.setdefault(ticker, []).extend(changed_fields)
        # Om endast stämplar, logga TS_* också så att det syns i körlogg
        if stamped_only and not changed_fields:
            log_changed.setdefault(ticker, []).extend([f"(TS) {x}" for x in stamped_only])

    return any_effect

# ---------------------------------------------------------------------------
# Huvud-UI för batch i sidopanelen
# ---------------------------------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb=None,
    recompute_cb=None,
    runner_price=None,
    runner_full=None,
) -> pd.DataFrame:
    """
    Visar en sidopanelsektion för batch-körningar.
    - Skapa batch-kö utifrån ordning & max antal
    - Välj runner-typ: "Endast kurs" eller "Full uppdatering"
    - Kör nästa eller Kör alla
    - Progressbar + "i/X"-text
    - Logg över ändringar/missar

    Parametrar:
      save_cb(df) -> None             : om angiven, används vid spara; annars storage.spara_data
      recompute_cb(df) -> pd.DataFrame: om angiven, körs efter uppdatering (t.ex. uppdatera_berakningar)
      runner_price(tkr) -> (vals, source, err)
      runner_full(tkr, df=None, user_rates=None) -> (vals, source, err)
    """
    if runner_price is None:
        runner_price = _sources.run_update_price_only
    if runner_full is None:
        runner_full = _sources.run_update_full

    st.sidebar.subheader("🛠️ Batch-uppdatering")

    # Inställningar
    sort_mode = st.sidebar.selectbox("Sortera", ["Äldst uppdaterade först", "A–Ö (bolagsnamn)"], index=0)
    max_items = st.sidebar.number_input("Antal i kö", min_value=1, max_value=1000, value=20, step=1)
    run_type = st.sidebar.radio("Vad ska uppdateras?", ["Endast kurs", "Full uppdatering"], horizontal=True)
    always_stamp = st.sidebar.checkbox("Stämpla datum även om värdet ej ändras", value=True)

    # Skapa kö
    if st.sidebar.button("📋 Skapa batch-kö"):
        order = _pick_order(df, sort_mode)
        # Skär till max_items
        queue = order[: int(max_items)]
        # Om det redan finns en kö, ersätt med ny
        st.session_state["batch_queue"] = queue
        st.session_state["batch_done"] = []
        st.session_state["batch_sort"] = sort_mode
        st.session_state["batch_type"] = run_type
        st.sidebar.success(f"Kö skapad ({len(queue)} st).")

    queue: List[str] = st.session_state.get("batch_queue", [])
    done: List[str] = st.session_state.get("batch_done", [])
    batch_type = st.session_state.get("batch_type", run_type)
    # Om användaren ändrat radio efter skapad kö, respektera manuellt val nu:
    current_type = run_type

    if queue:
        st.sidebar.info(f"Aktuell kö: {len(queue)} tickers • Klara: {len(done)}")
        # Val: Kör nästa
        if st.sidebar.button("▶️ Kör nästa i kö"):
            tkr = None
            # Plocka första som inte är klar
            for cand in queue:
                if cand not in done:
                    tkr = cand
                    break
            if tkr:
                # Kör vald runner
                try:
                    if current_type == "Endast kurs":
                        vals, source, err = runner_price(tkr)
                    else:
                        vals, source, err = runner_full(tkr, df=df, user_rates=user_rates)
                except Exception as e:
                    vals, source, err = {}, "runner_error", str(e)

                # Loggar i session för senare visning
                log = st.session_state.get("last_auto_log", {"changed": {}, "misses": {}, "errors": {}})
                if "changed" not in log: log["changed"] = {}
                if "misses" not in log: log["misses"] = {}
                if "errors" not in log: log["errors"] = {}

                # Skriv rad om den finns
                ridxs = df.index[df["Ticker"].astype(str).str.upper().str.strip() == str(tkr).upper().strip()].tolist()
                if not ridxs:
                    log["errors"][tkr] = ["Ticker saknas i tabellen."]
                else:
                    idx = ridxs[0]
                    wrote = _apply_updates_to_row(df, idx, vals, source, log_changed=log["changed"], always_stamp=always_stamp)
                    if not wrote:
                        # registrera "miss"
                        log["misses"][tkr] = list(vals.keys()) if vals else ["(inga fält)"]

                    # Spara direkt?
                    if save_cb:
                        try:
                            save_cb(df)
                        except Exception as e:
                            log["errors"].setdefault(tkr, []).append(f"Spara-fel: {e}")
                    else:
                        try:
                            spara_data(df, do_snapshot=False)
                        except Exception as e:
                            log["errors"].setdefault(tkr, []).append(f"Spara-fel: {e}")

                    # Recompute?
                    if recompute_cb:
                        try:
                            df2 = recompute_cb(df)
                            if df2 is not None:
                                df[:] = df2  # in-place uppdatering
                        except Exception as e:
                            log["errors"].setdefault(tkr, []).append(f"Recompute-fel: {e}")

                # Markera klar
                if tkr not in done:
                    done.append(tkr)
                st.session_state["batch_done"] = done
                st.session_state["last_auto_log"] = log
                st.sidebar.success(f"Klar: {tkr}")

        # Kör alla
        if st.sidebar.button("⏩ Kör alla i kö"):
            total = len(queue)
            done_set = set(done)
            todo = [t for t in queue if t not in done_set]
            progress = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            log = st.session_state.get("last_auto_log", {"changed": {}, "misses": {}, "errors": {}})
            if "changed" not in log: log["changed"] = {}
            if "misses" not in log: log["misses"] = {}
            if "errors" not in log: log["errors"] = {}

            for i, tkr in enumerate(todo):
                status.write(f"Uppdaterar {i+1}/{len(todo)}: {tkr}")
                try:
                    if current_type == "Endast kurs":
                        vals, source, err = runner_price(tkr)
                    else:
                        vals, source, err = runner_full(tkr, df=df, user_rates=user_rates)
                except Exception as e:
                    vals, source, err = {}, "runner_error", str(e)

                ridxs = df.index[df["Ticker"].astype(str).str.upper().str.strip() == str(tkr).upper().strip()].tolist()
                if not ridxs:
                    log["errors"][tkr] = ["Ticker saknas i tabellen."]
                else:
                    idx = ridxs[0]
                    wrote = _apply_updates_to_row(df, idx, vals, source, log_changed=log["changed"], always_stamp=always_stamp)
                    if not wrote:
                        log["misses"][tkr] = list(vals.keys()) if vals else ["(inga fält)"]
                    # Spara efter varje
                    if save_cb:
                        try:
                            save_cb(df)
                        except Exception as e:
                            log["errors"].setdefault(tkr, []).append(f"Spara-fel: {e}")
                    else:
                        try:
                            spara_data(df, do_snapshot=False)
                        except Exception as e:
                            log["errors"].setdefault(tkr, []).append(f"Spara-fel: {e}")

                    # Recompute?
                    if recompute_cb:
                        try:
                            df2 = recompute_cb(df)
                            if df2 is not None:
                                df[:] = df2
                        except Exception as e:
                            log["errors"].setdefault(tkr, []).append(f"Recompute-fel: {e}")

                done.append(tkr)
                progress.progress((len(done) / max(total, 1.0)))
                status.write(f"Uppdaterar {len(done)}/{total}: {tkr}")

            st.session_state["batch_done"] = done
            st.session_state["last_auto_log"] = log
            st.sidebar.success("Batch körning klar.")

        # Återställ kö
        if st.sidebar.button("🧹 Återställ kö"):
            st.session_state.pop("batch_queue", None)
            st.session_state.pop("batch_done", None)
            st.sidebar.info("Kö återställd.")

        # Liten förhandsvisning av kö
        with st.sidebar.expander("Visa kö"):
            st.write(queue)

        # Visa logg
        with st.sidebar.expander("Senaste körlogg"):
            log = st.session_state.get("last_auto_log")
            if not log:
                st.write("Ingen körlogg ännu.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Ändringar**")
                    st.json(log.get("changed", {}))
                with col2:
                    st.markdown("**Missar**")
                    st.json(log.get("misses", {}))
                st.markdown("**Fel**")
                st.json(log.get("errors", {}))
    else:
        st.sidebar.info("Ingen aktiv kö. Skapa en via **📋 Skapa batch-kö**.")

    return df
