# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

from .sources import fetch_all_fields_for_ticker, fetch_price_only

# ------------------------------------------------------------
# Standard TS-fält (samma som i edit-vyn)
# ------------------------------------------------------------
DEFAULT_TS_FIELDS = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            if any(k in str(c).lower() for k in ["p/s","omsättning","kurs","marginal","utdelning","cagr","antal","riktkurs","värde","debt","cash","fcf","runway","market cap","gav"]):
                df[c] = 0.0
            else:
                df[c] = ""
    return df

def _stamp_ts_for_field(df: pd.DataFrame, ridx: int, field: str, ts_fields: Dict[str, str]):
    ts_col = ts_fields.get(field)
    if ts_col:
        df.at[ridx, ts_col] = _now_stamp()

def _note_auto_update(df: pd.DataFrame, ridx: int, source: str):
    _ensure_cols(df, ["Senast auto-uppdaterad","Senast uppdaterad källa"])
    df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = source

def _row_index_for_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    t = str(ticker).strip().upper()
    idx = df.index[df["Ticker"].astype(str).str.upper() == t].tolist()
    return idx[0] if idx else None

def _oldest_any_ts(row: pd.Series, ts_fields: Dict[str, str]) -> Optional[pd.Timestamp]:
    dates = []
    for c in ts_fields.values():
        if c in row and str(row[c]).strip():
            d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
            if pd.notna(d):
                dates.append(d)
    return min(dates) if dates else None

def _add_oldest_ts_col(df: pd.DataFrame, ts_fields: Dict[str, str]) -> pd.DataFrame:
    df["_oldest_any_ts"] = df.apply(lambda r: _oldest_any_ts(r, ts_fields), axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

def _ps_avg_from_row(row: pd.Series) -> float:
    vals = []
    for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
        try:
            v = float(row.get(k, 0.0) or 0.0)
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    return round(float(np.mean(vals)), 2) if vals else 0.0

def _recompute_locally(df: pd.DataFrame, ridx: int):
    """Minimal omräkning om ingen recompute_cb skickas in."""
    # P/S-snitt
    df.at[ridx, "P/S-snitt"] = _ps_avg_from_row(df.loc[ridx])
    # Riktkurser
    r = df.loc[ridx]
    ps_snitt = float(r.get("P/S-snitt", 0.0) or 0.0)
    shares_m = float(r.get("Utestående aktier", 0.0) or 0.0)
    if ps_snitt > 0 and shares_m > 0:
        for src, dst in [
            ("Omsättning idag",    "Riktkurs idag"),
            ("Omsättning nästa år","Riktkurs om 1 år"),
            ("Omsättning om 2 år", "Riktkurs om 2 år"),
            ("Omsättning om 3 år", "Riktkurs om 3 år"),
        ]:
            val = float(r.get(src, 0.0) or 0.0)
            if val > 0:
                df.at[ridx, dst] = round((val * ps_snitt) / shares_m, 2)

# ------------------------------------------------------------
# Apply fetched vals to df row (stämplar TS ALLTID på spårade fält)
# ------------------------------------------------------------

def _apply_vals_to_row(df: pd.DataFrame, ridx: int, vals: Dict, ts_fields: Dict[str, str], source_label: str) -> Tuple[bool, List[str]]:
    changed = False
    changed_fields = []
    _ensure_cols(df, list(vals.keys()) + ["Senast auto-uppdaterad","Senast uppdaterad källa"])
    for k, v in vals.items():
        if k not in df.columns:
            df[k] = np.nan
        old = df.at[ridx, k]
        df.at[ridx, k] = v
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            changed = True
            changed_fields.append(k)
        if k in ts_fields:
            _stamp_ts_for_field(df, ridx, k, ts_fields)
    _note_auto_update(df, ridx, source_label)
    return changed, changed_fields

# ------------------------------------------------------------
# Orderval (Äldst först / A–Ö)
# ------------------------------------------------------------

def _pick_order(df: pd.DataFrame, sort_mode: str, ts_fields: Dict[str, str]) -> List[str]:
    if sort_mode.startswith("Äldst"):
        work = _add_oldest_ts_col(df.copy(), ts_fields)
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])
    return [str(t).upper() for t in vis_df["Ticker"].astype(str).tolist() if str(t).strip()]

# ------------------------------------------------------------
# Batch core
# ------------------------------------------------------------

def run_batch_update(
    df: pd.DataFrame,
    user_rates: dict,
    tickers: List[str],
    *,
    ts_fields: Dict[str, str] = None,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    runner: Optional[Callable[[str], Tuple[Dict, Dict]]] = None,
    commit_every: int = 5,
    label: str = "Batch"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Kör batch-uppdatering för en given lista tickers.
    - runner: funktion som tar ticker -> (vals, debug). Default: fetch_all_fields_for_ticker.
    - commit_every: skriv till Sheets var N:e rad (skonsamt).
    Returnerar (df, log)
    """
    if ts_fields is None:
        ts_fields = DEFAULT_TS_FIELDS
    if runner is None:
        runner = fetch_all_fields_for_ticker

    total = len(tickers)
    changed_any = False
    log = {"changed": {}, "misses": {}, "errors": {}}

    prog = st.sidebar.progress(0.0, text=f"{label}: 0/{total}")
    status = st.sidebar.empty()

    for i, tkr in enumerate(tickers, start=1):
        status.write(f"Uppdaterar {i}/{total}: **{tkr}**")
        ridx = _row_index_for_ticker(df, tkr)
        if ridx is None:
            log["errors"][tkr] = "Ticker saknas i tabellen."
            prog.progress(i/total, text=f"{label}: {i}/{total}")
            continue

        try:
            vals, dbg = runner(tkr)
            changed, fields = _apply_vals_to_row(df, ridx, vals, ts_fields, source_label=f"Auto ({runner.__name__})")
            if changed:
                log["changed"][tkr] = fields
                changed_any = True
            else:
                log["misses"][tkr] = list(vals.keys()) if vals else ["(inga nya fält)"]
        except Exception as e:
            log["errors"][tkr] = str(e)

        # lokal eller global omräkning
        if recompute_cb:
            try:
                df2 = recompute_cb(df.copy())
                df[:] = df2
            except Exception as e:
                log["errors"][tkr] = f"recompute: {e}"
        else:
            _recompute_locally(df, ridx)

        # Spara skonsamt
        if save_cb and (i % max(1, commit_every) == 0):
            try:
                save_cb(df)
            except Exception as e:
                log["errors"][tkr] = f"save: {e}"

        prog.progress(i/total, text=f"{label}: {i}/{total}")

    # slutlig commit
    if save_cb:
        try:
            save_cb(df)
        except Exception as e:
            st.sidebar.error(f"Misslyckades spara slutligen: {e}")

    if changed_any:
        st.sidebar.success("Batch klar – ändringar sparade.")
    else:
        st.sidebar.info("Batch klar – inga faktiska ändringar upptäckta.")

    return df, log

# ------------------------------------------------------------
# Sidebar UI
# ------------------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: dict,
    *,
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    ts_fields: Dict[str, str] = None,
    runner: Optional[Callable[[str], Tuple[Dict, Dict]]] = None,
):
    """
    Visar en komplett batch-panel i sidopanelen:
    - välj sortering, välj full auto / endast kurs
    - bygg kö (storlek N)
    - kör/återuppta
    - återställ/skip
    Progressbar + 1/X-text ingår.
    """
    if ts_fields is None:
        ts_fields = DEFAULT_TS_FIELDS

    st.sidebar.subheader("🛠️ Batch-körning")

    # Val: sortering
    sort_mode = st.sidebar.radio("Ordning", ["Äldst uppdaterade först", "A–Ö"], horizontal=False, index=0)

    # Val: runner
    only_price = st.sidebar.toggle("Endast kurs (snabbt)", value=False)
    effective_runner = runner or st.session_state.get("_batch_runner")
    if effective_runner is None:
        effective_runner = fetch_price_only if only_price else fetch_all_fields_for_ticker

    # Batchstorlek
    n = int(st.sidebar.number_input("Batch-storlek", min_value=1, max_value=200, value=20, step=1))

    # init state
    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue: List[str] = []
    if "batch_idx" not in st.session_state:
        st.session_state.batch_idx = 0
    if "batch_log" not in st.session_state:
        st.session_state.batch_log = {}
    if "batch_sort" not in st.session_state:
        st.session_state.batch_sort = sort_mode

    # Knappar
    cols = st.sidebar.columns(3)
    with cols[0]:
        if st.button("🧱 Bygg kö"):
            order = _pick_order(df, sort_mode, ts_fields)
            # välj första n som INTE redan finns i kön (undvik duplicering)
            to_take = [t for t in order if t not in st.session_state.batch_queue][:n]
            st.session_state.batch_queue = to_take
            st.session_state.batch_idx = 0
            st.session_state.batch_sort = sort_mode
            st.sidebar.success(f"Kö byggd: {len(to_take)} tickers.")
    with cols[1]:
        if st.button("▶️ Kör batch"):
            if not st.session_state.batch_queue:
                st.sidebar.warning("Kö saknas. Klicka 'Bygg kö' först.")
            else:
                # processera från idx → slut
                total = len(st.session_state.batch_queue)
                remain = st.session_state.batch_queue[st.session_state.batch_idx:]
                df2, log = run_batch_update(
                    df, user_rates, remain,
                    ts_fields=ts_fields,
                    save_cb=save_cb,
                    recompute_cb=recompute_cb,
                    runner=effective_runner,
                    commit_every=5,
                    label="Batch"
                )
                df[:] = df2
                st.session_state.batch_log = log
                # när klart, hoppa idx till slut
                st.session_state.batch_idx = total
    with cols[2]:
        if st.button("♻️ Återställ"):
            st.session_state.batch_queue = []
            st.session_state.batch_idx = 0
            st.session_state.batch_log = {}
            st.sidebar.info("Kö återställd.")

    # Progressindikator
    total = len(st.session_state.batch_queue)
    current = min(st.session_state.batch_idx, total)
    if total > 0:
        st.sidebar.progress(0 if total == 0 else current/total, text=f"{current}/{total}")
        st.sidebar.caption("Kön (förhandsvisning): " + ", ".join(st.session_state.batch_queue[:min(10, total)]) + (" ..." if total > 10 else ""))

    # Skip-knapp om det finns en aktiv
    if total > 0 and current < total:
        tkr = st.session_state.batch_queue[current]
        st.sidebar.write(f"Aktuell: **{tkr}**")
        if st.sidebar.button("⏭️ Skippa aktuell"):
            st.session_state.batch_idx = min(total, st.session_state.batch_idx + 1)
            st.sidebar.info(f"Skippade {tkr}. Nästa blir index {st.session_state.batch_idx+1}/{total}.")

    # Kör nästa (en och en) – ibland praktiskt
    if total > 0 and current < total:
        if st.sidebar.button("➡️ Kör nästa (1 st)"):
            tkr = st.session_state.batch_queue[current]
            df2, log = run_batch_update(
                df, user_rates, [tkr],
                ts_fields=ts_fields,
                save_cb=save_cb,
                recompute_cb=recompute_cb,
                runner=effective_runner,
                commit_every=1,
                label="Batch 1/1"
            )
            df[:] = df2
            # flytta fram pekare
            st.session_state.batch_idx = current + 1
            # slå ihop logg
            for k in ["changed","misses","errors"]:
                st.session_state.batch_log.setdefault(k, {})
                st.session_state.batch_log[k].update(log.get(k, {}))

    # Visa senast körlogg
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Senaste batchlogg**")
    if st.session_state.batch_log:
        st.sidebar.json(st.session_state.batch_log)
    else:
        st.sidebar.caption("–")
