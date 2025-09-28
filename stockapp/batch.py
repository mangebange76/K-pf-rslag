# stockapp/batch.py
# -*- coding: utf-8 -*-
"""
Batch-körning (sidopanel):
- Välj sortering: A–Ö eller Äldst TS först (spårade TS-kolumner)
- Ange batch-storlek (t.ex. 10): "Kör nästa N"
- Visar progressbar + "i/X"-text
- Robust default-runner (SEC/Yahoo (+FMP fallback)) via stockapp.sources.fetch_all_fields_for_ticker
- Skriver EJ över manuella prognosfält ("Omsättning idag"/"Omsättning nästa år")
- Stämplar TS_ för spårade fält när de matas in (även om värdet råkar vara samma)
- Sätter "Senast auto-uppdaterad" + "Senast uppdaterad källa"
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any
import streamlit as st
import pandas as pd
import numpy as np

from .config import (
    TS_FIELDS,
    FINAL_COLS,
    säkerställ_kolumner,
    konvertera_typer,
    add_oldest_ts_col,
    now_stamp,
)

from .calc import update_calculations
from . import sources as _sources


# --------------------------------------------------------------------------------------
# Interna hjälp-funktioner
# --------------------------------------------------------------------------------------

MANUELL_PROGNOS_FALT = ("Omsättning idag", "Omsättning nästa år")

def _safe_float(x, default=0.0) -> float:
    try:
        f = float(x)
        if np.isnan(f):
            return default
        return f
    except Exception:
        return default

def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """Bygger ordningslista av tickers utifrån vald sortering."""
    if df is None or df.empty or "Ticker" not in df.columns:
        return []

    if sort_mode == "Äldst TS först":
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn", "Ticker"], ascending=[True, True, True])
        order = [str(t).upper().strip() for t in work["Ticker"].tolist()]
        return order

    # default A–Ö (bolagsnamn, ticker)
    work = df.sort_values(by=["Bolagsnamn", "Ticker"], ascending=[True, True]).copy()
    return [str(t).upper().strip() for t in work["Ticker"].tolist()]

def _ensure_session_defaults():
    """Initiera state-nycklar som används i batchpanelen."""
    ss = st.session_state
    ss.setdefault("batch_sort_mode", "Äldst TS först")  # eller "A–Ö"
    ss.setdefault("batch_size", 10)
    ss.setdefault("batch_order", [])
    ss.setdefault("batch_pointer", 0)
    ss.setdefault("batch_last_sort", ss["batch_sort_mode"])
    ss.setdefault("last_auto_log", {"changed": {}, "misses": {}, "debug": []})

def _apply_auto_fields_to_row(df: pd.DataFrame, row_idx: int, new_vals: Dict[str, Any], source_label: str, changes_map: Dict[str, List[str]]) -> bool:
    """
    Skriv in automatiskt hämtade fält i en rad.
    - Skriv EJ manuella prognosfält.
    - TS_ stämplas för spårade fält OM fältet finns i new_vals (även om lika).
    - 'Senast auto-uppdaterad' + 'Senast uppdaterad källa'
    Returnerar True om något värde faktiskt ändrades.
    """
    changed = False
    tkr = str(df.at[row_idx, "Ticker"]).strip().upper()
    changed_fields: List[str] = []

    for f, v in (new_vals or {}).items():
        if f in MANUELL_PROGNOS_FALT:
            continue  # lämna åt manuell uppdatering

        if f not in df.columns:
            # skapa kolumn om den saknas
            if any(x in f.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","score","market cap","kassa","skuld"]):
                df[f] = 0.0
            else:
                df[f] = ""

        old = df.at[row_idx, f]
        write_ok = True
        if isinstance(v, (int, float, np.floating)):
            # alla numeriska fält skrivs (även 0.0) för robusthet? Vi undviker att nolla befintliga bra värden:
            # strategi: skriv om v > 0, annars bara om kolumnen saknas/är NaN
            if f.lower().startswith("p/s") or "market cap" in f.lower() or f in ("Utestående aktier",):
                write_ok = (float(v) > 0)
            else:
                write_ok = (f not in ("P/S", "Utestående aktier") and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        if not write_ok:
            # stämpla ändå TS om spårat fält levererades (utan att skriva)
            ts_col = TS_FIELDS.get(f)
            if ts_col:
                df.at[row_idx, ts_col] = now_stamp()
            continue

        # skriv och markera ändrat om skillnad
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[row_idx, f] = v
            changed = True
            changed_fields.append(f)

        # TS-stämpel om spårat fält finns i nytt payload (alltid – även om lika)
        ts_col = TS_FIELDS.get(f)
        if ts_col:
            df.at[row_idx, ts_col] = now_stamp()

    # meta
    df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
    df.at[row_idx, "Senast uppdaterad källa"] = source_label

    if changed_fields:
        changes_map.setdefault(tkr, []).extend(changed_fields)

    return changed

def _default_runner(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Standard-runner: SEC/Yahoo (+FMP fallback) via sources-modulen.
    Returnerar (vals, debug) – 'vals' är en dict att skriva in i DF.
    """
    debug: Dict[str, Any] = {"src": "SEC/Yahoo/FMP"}
    vals = {}
    try:
        vals = _sources.fetch_all_fields_for_ticker(ticker)
        debug["keys"] = list(vals.keys())
    except Exception as e:
        debug["error"] = str(e)
    return vals or {}, debug

# --------------------------------------------------------------------------------------
# Publikt API – Batch-panel i sidopanelen
# --------------------------------------------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    runner: Optional[Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """
    Renderar batch-panel i sidopanelen och hanterar körningar.
    Returnerar (möjligen) uppdaterad DataFrame.
    """
    _ensure_session_defaults()
    ss = st.session_state

    st.sidebar.subheader("🛠️ Batch-uppdatering")

    # Välj ordning
    sort_mode = st.sidebar.selectbox("Ordning", ["Äldst TS först", "A–Ö"], index=0 if ss["batch_sort_mode"]=="Äldst TS först" else 1, key="batch_sort_mode")

    # Storlek
    batch_size = st.sidebar.number_input("Antal i nästa batch", min_value=1, max_value=200, value=int(ss["batch_size"]), step=1, key="batch_size")

    # Beräkna ordning om ny eller saknas
    if not ss["batch_order"] or ss["batch_last_sort"] != sort_mode:
        order = _pick_order(df, sort_mode)
        ss["batch_order"] = order
        ss["batch_pointer"] = 0
        ss["batch_last_sort"] = sort_mode

    total = len(ss["batch_order"])
    st.sidebar.caption(f"Kö: {total} tickers")
    st.sidebar.caption(f"Nästa startindex: {ss['batch_pointer']}")

    # Styra pointer manuellt?
    new_ptr = st.sidebar.number_input("Hoppa till index", min_value=0, max_value=max(0, total-1), value=int(ss["batch_pointer"]), step=1)
    if new_ptr != ss["batch_pointer"]:
        ss["batch_pointer"] = int(new_ptr)

    # Knappar
    colb1, colb2, colb3 = st.sidebar.columns([1,1,1])
    with colb1:
        reset_clicked = st.button("Återställ kö")
    with colb2:
        preview_clicked = st.button("Förhandsvisa")
    with colb3:
        run_clicked = st.button("Kör nästa N")

    if reset_clicked:
        ss["batch_order"] = _pick_order(df, sort_mode)
        ss["batch_pointer"] = 0
        st.sidebar.success("Kön återställd.")

    # Förhandsvisa vilka som körs
    if preview_clicked or not ss["batch_order"]:
        start = ss["batch_pointer"]
        end = min(total, start + int(batch_size))
        preview = ss["batch_order"][start:end]
        if preview:
            st.sidebar.write("**Nästa batch:**")
            st.sidebar.write(", ".join(preview))
        else:
            st.sidebar.info("Inget att köra (kön slut).")

    # Körning
    if run_clicked and ss["batch_order"]:
        start = ss["batch_pointer"]
        end = min(total, start + int(batch_size))
        to_run = ss["batch_order"][start:end]

        if not to_run:
            st.sidebar.info("Inget att köra (kön slut).")
            return df

        progress = st.sidebar.progress(0.0)
        step_txt = st.sidebar.empty()

        # För logg
        log = {"changed": {}, "misses": {}, "debug": []}
        any_changed = False

        run_fn = runner or _default_runner
        src_label = "Auto (SEC/Yahoo→FMP)"

        # Säkerställ schema/typer före skrivning
        df = säkerställ_kolumner(df)
        df = konvertera_typer(df)

        for i, tkr in enumerate(to_run):
            step_txt.write(f"Uppdaterar {i+1}/{len(to_run)} – **{tkr}**")
            try:
                vals, dbg = run_fn(tkr)
                log["debug"].append({tkr: dbg})

                # hitta rad
                mask = (df["Ticker"].astype(str).str.upper().str.strip() == str(tkr).upper().strip())
                if not mask.any():
                    log["misses"][tkr] = ["Ticker saknas i tabellen"]
                else:
                    idx = df.index[mask][0]
                    # skriv fält med TS och meta
                    changed = _apply_auto_fields_to_row(df, idx, vals, source_label=src_label, changes_map=log["changed"])
                    any_changed = any_changed or changed
            except Exception as e:
                log["misses"][tkr] = [f"error: {e}"]

            progress.progress((i+1)/max(1, len(to_run)))

        # Efter körning – uppdatera beräkningar & spara (om callbackar finns)
        if recompute_cb:
            try:
                df = recompute_cb(df)
            except Exception as e:
                st.sidebar.warning(f"Kunde inte räkna om formler: {e}")

        if save_cb and any_changed:
            try:
                save_cb(df)
                st.sidebar.success("Ändringar sparade.")
            except Exception as e:
                st.sidebar.error(f"Misslyckades spara: {e}")
        else:
            if not any_changed:
                st.sidebar.info("Inga faktiska ändringar – ingen skrivning.")

        # Flytta pointer
        ss["batch_pointer"] = end
        ss["last_auto_log"] = log

        # Summera
        n_ch = sum(len(v) for v in log["changed"].values())
        st.sidebar.write(f"Klart. Ändrade fält totalt: {n_ch}.")
        if log["misses"]:
            st.sidebar.warning(f"Missar på {len(log['misses'])} tickers.")

    return df


# --------------------------------------------------------------------------------------
# Hjälp-funktion för huvud-app: kör batch på given lista (utan UI)
# --------------------------------------------------------------------------------------

def run_batch_update(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    tickers: List[str],
    make_snapshot: bool = False,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    runner: Optional[Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Kör en batch programatiskt (utan att rita UI). Returnerar (df, log).
    """
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    progress = st.sidebar.progress(0.0)
    step_txt = st.sidebar.empty()

    log = {"changed": {}, "misses": {}, "debug": []}
    any_changed = False

    run_fn = runner or _default_runner
    src_label = "Auto (SEC/Yahoo→FMP)"

    for i, tkr in enumerate(tickers):
        step_txt.write(f"Uppdaterar {i+1}/{len(tickers)} – **{tkr}**")
        try:
            vals, dbg = run_fn(tkr)
            log["debug"].append({tkr: dbg})

            mask = (df["Ticker"].astype(str).str.upper().str.strip() == str(tkr).upper().strip())
            if not mask.any():
                log["misses"][tkr] = ["Ticker saknas i tabellen"]
            else:
                idx = df.index[mask][0]
                changed = _apply_auto_fields_to_row(df, idx, vals, source_label=src_label, changes_map=log["changed"])
                any_changed = any_changed or changed
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(1, len(tickers)))

    # Räkna om
    try:
        df = update_calculations(df, user_rates)
    except Exception as e:
        st.sidebar.warning(f"Kunde inte räkna om formler: {e}")

    # Spara
    if save_cb and any_changed:
        try:
            save_cb(df)
            st.sidebar.success("Ändringar sparade.")
        except Exception as e:
            st.sidebar.error(f"Misslyckades spara: {e}")
    else:
        if not any_changed:
            st.sidebar.info("Inga faktiska ändringar – ingen skrivning.")

    return df, log
