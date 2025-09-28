# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import pytz
except Exception:
    pytz = None

try:
    import streamlit as st
except Exception:
    st = None

# Vi använder dina runners från sources.py
from .sources import fetch_all_fields_for_ticker, fetch_price_only

# ------------------------------------------------------------
# Tidsstämplar & spårade fält
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

def _now_stamp() -> str:
    if pytz:
        try:
            tz = pytz.timezone("Europe/Stockholm")
            return datetime.now(tz).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.now().strftime("%Y-%m-%d")

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            # numeriska default för typiska siffervärden, annars str tom
            if any(k in c.lower() for k in ["p/s","omsättning","kurs","marginal","utdelning","cagr","antal","riktkurs","värde","debt","cash","kassa","fcf","runway","market cap","gav"]):
                df[c] = 0.0
            else:
                df[c] = ""
    return df

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

# ------------------------------------------------------------
# Kö-ordning och bygg kö
# ------------------------------------------------------------

def _pick_order(df: pd.DataFrame, sort_mode: str, ts_fields: Dict[str, str]) -> List[str]:
    """
    Returnerar lista med tickers i vald ordning.
    sort_mode: "A–Ö (bolagsnamn)" eller "Äldst först (spårade fält)"
    """
    if sort_mode.startswith("Äldst"):
        work = _add_oldest_ts_col(df.copy(), ts_fields)
        vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        work = df.copy()
        vis = work.sort_values(by=["Bolagsnamn","Ticker"])
    # Filtrera bort tomma tickers
    out = [str(t).strip().upper() for t in vis.get("Ticker", []) if str(t).strip()]
    return out

def _build_batch_queue(df: pd.DataFrame, sort_mode: str, batch_size: int, ts_fields: Dict[str, str]) -> List[str]:
    order = _pick_order(df, sort_mode, ts_fields)
    # Om användaren kör i omgångar: vi lägger INTE bara first N här,
    # utan hela listan i session_state och en cursor pekar var vi är.
    return order

# ------------------------------------------------------------
# Applicera värden på df (med tidsstämplar)
# ------------------------------------------------------------

def _apply_vals_to_df(df: pd.DataFrame, ticker: str, vals: Dict,
                      ts_fields: Dict[str, str], source_label: str = "Auto (batch)") -> Tuple[bool, List[str]]:
    """
    Skriver in alla nycklar i vals till df-raden för ticker.
    - Skapar kolumner vid behov
    - Stämplar TS-kolumn för spårade fält **alltid** (även vid samma värde)
    - Uppdaterar "Senast auto-uppdaterad" + "Senast uppdaterad källa"
    Returnerar (changed, changed_fields)
    """
    tkr = str(ticker).strip().upper()
    idx_list = df.index[df["Ticker"].astype(str).str.upper() == tkr].tolist()
    if not idx_list:
        return False, []
    ridx = idx_list[0]

    # Se till att kolumner finns
    df = _ensure_cols(df, list(vals.keys()))
    # Säkerställ meta-kolumner
    df = _ensure_cols(df, ["Senast auto-uppdaterad","Senast uppdaterad källa"])
    for ts_col in ts_fields.values():
        df = _ensure_cols(df, [ts_col])

    changed = False
    changed_fields = []
    for k, v in vals.items():
        if k not in df.columns:
            df[k] = np.nan
        old = df.at[ridx, k] if k in df.columns else None
        write_ok = True
        # skriv även samma värde (för att kunna stämpla TS)
        if write_ok:
            df.at[ridx, k] = v
            # track changed om faktiskt skillnad
            if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
                changed = True
                changed_fields.append(k)
            # TS-stämpel om spårat fält
            if k in ts_fields:
                df.at[ridx, ts_fields[k]] = _now_stamp()

    # Auto-meta alltid
    df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = source_label

    return changed, changed_fields

# ------------------------------------------------------------
# Körning
# ------------------------------------------------------------

def _get_runner(kind: str):
    """
    kind: "full" -> fetch_all_fields_for_ticker
          "price" -> fetch_price_only
    """
    if kind == "price":
        return fetch_price_only
    return fetch_all_fields_for_ticker

def run_batch_step(df: pd.DataFrame, tickers: List[str], cursor: int, step: int,
                   runner_kind: str, ts_fields: Dict[str, str]) -> Tuple[pd.DataFrame, Dict, int]:
    """
    Kör upp till 'step' tickers från cursor.
    Returnerar (df, log, new_cursor)
    """
    log = {"changed": {}, "misses": {}, "errors": {}}
    total = len(tickers)
    if total == 0:
        return df, log, cursor

    runner = _get_runner(runner_kind)

    # Progress UI
    prog = st.sidebar.progress(0.0) if st is not None else None
    txt = st.sidebar.empty() if st is not None else None

    count = 0
    i = cursor
    while i < total and count < step:
        tkr = tickers[i]
        if txt is not None:
            txt.write(f"Uppdaterar {i+1}/{total}: {tkr}")
        try:
            vals, dbg = runner(tkr)
            changed, fields = _apply_vals_to_df(df, tkr, vals, ts_fields, source_label=f"Auto ({runner_kind})")
            if changed:
                log["changed"][tkr] = fields
            else:
                # även om inget ändrades har vi stämplat TS och meta – visa nycklar som kom
                log["misses"][tkr] = list(vals.keys()) if vals else ["(inga fält)"]
        except Exception as e:
            log["errors"][tkr] = str(e)

        count += 1
        i += 1
        if prog is not None:
            prog.progress(min(1.0, (i - cursor) / max(1, step)))

    # Slut
    if txt is not None:
        txt.write("Klart för denna omgång.")
    return df, log, i

# ------------------------------------------------------------
# Sidopanel-UI
# ------------------------------------------------------------

def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: dict,
    save_cb,           # callable(df) -> None  (spara till Sheets)
    recompute_cb,      # callable(df) -> df    (räkna om riktkurser mm)
    ts_fields: Dict[str, str] = None,
    runner: str = None # "full" eller "price" (om None -> styrs via UI)
) -> pd.DataFrame:
    """
    Visar hela batchpanelen i sidopanelen och hanterar kölogik i st.session_state.
    Returnerar df (uppdaterat).
    """
    if ts_fields is None:
        ts_fields = DEFAULT_TS_FIELDS

    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []
    if "batch_cursor" not in st.session_state:
        st.session_state.batch_cursor = 0
    if "batch_sort_mode" not in st.session_state:
        st.session_state.batch_sort_mode = "Äldst först (spårade fält)"
    if "batch_last_log" not in st.session_state:
        st.session_state.batch_last_log = {}
    if "batch_runner_kind" not in st.session_state:
        st.session_state.batch_runner_kind = "full"

    with st.sidebar.expander("⚙️ Batch-uppdatering", expanded=False):
        sort_mode = st.selectbox("Sortera kö", ["Äldst först (spårade fält)", "A–Ö (bolagsnamn)"],
                                 index=0 if st.session_state.batch_sort_mode.startswith("Äldst") else 1)
        st.session_state.batch_sort_mode = sort_mode

        # Runner-val (om inte forcerat externt)
        if runner is None:
            runner_kind = st.radio("Vad ska uppdateras?", ["Full auto (alla fält)", "Endast kurs"], horizontal=False)
            st.session_state.batch_runner_kind = "full" if runner_kind.startswith("Full") else "price"
        else:
            st.session_state.batch_runner_kind = runner

        batch_size = st.number_input("Antal att köra nu", min_value=1, max_value=500, value=10, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Starta ny batch"):
                st.session_state.batch_queue = _build_batch_queue(df, st.session_state.batch_sort_mode, batch_size, ts_fields)
                st.session_state.batch_cursor = 0
                st.session_state.batch_last_log = {}
                st.success(f"Batch skapad: {len(st.session_state.batch_queue)} tickers i kö.")
        with col2:
            if st.button("Kör nästa"):
                q = st.session_state.batch_queue
                cur = st.session_state.batch_cursor
                if not q:
                    st.warning("Ingen kö. Klicka 'Starta ny batch' först.")
                elif cur >= len(q):
                    st.info("Kön är redan färdigkörd.")
                else:
                    df2, log, new_cur = run_batch_step(
                        df.copy(), q, cur, int(batch_size), st.session_state.batch_runner_kind, ts_fields
                    )
                    # Recompute & save
                    df2 = recompute_cb(df2)
                    try:
                        save_cb(df2)
                        st.success("Ändringar sparade.")
                    except Exception as e:
                        st.error(f"Kunde inte spara: {e}")
                    # Skriv tillbaka referens
                    df[:] = df2  # in-place
                    st.session_state.batch_cursor = new_cur
                    st.session_state.batch_last_log = log

        with col3:
            if st.button("Återställ"):
                st.session_state.batch_queue = []
                st.session_state.batch_cursor = 0
                st.session_state.batch_last_log = {}
                st.info("Batch-navigator återställd.")

        # Status
        q = st.session_state.batch_queue
        cur = st.session_state.batch_cursor
        if q:
            st.write(f"Köstatus: **{min(cur, len(q))}/{len(q)}** tickers körda.")
            # lista kort en försmak på nästa upp till 5
            nxt = q[cur:cur+5]
            if nxt:
                st.caption("Nästa i kö:")
                st.code(", ".join(nxt))
        else:
            st.write("Ingen aktiv kö.")

        # 1/X progress i sidopanelen (extra)
        if q:
            st.progress(0.0 if len(q) == 0 else min(1.0, cur / max(1, len(q))))

        # Senaste körlogg
        log = st.session_state.batch_last_log or {}
        if log:
            st.subheader("📒 Senaste körlogg")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Ändringar**")
                st.json(log.get("changed", {}))
            with c2:
                st.markdown("**Missar**")
                st.json(log.get("misses", {}))
            with c3:
                st.markdown("**Fel**")
                st.json(log.get("errors", {}))

    return df
