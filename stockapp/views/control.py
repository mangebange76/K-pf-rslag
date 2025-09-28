# -*- coding: utf-8 -*-
"""
stockapp/views/control.py
Kontroll-vy (överblick):
- Äldst uppdaterade (alla spårade fält)
- Kandidater för manuell hantering
- Senaste körlogg (om finns i session_state["last_auto_log"])
"""

from __future__ import annotations
from typing import Dict, Optional
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- TS-fält (håll i sync med övriga moduler) --------------------------------
TS_FIELDS: Dict[str, str] = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

# --- Hjälpare ----------------------------------------------------------------

def oldest_any_ts(row: pd.Series) -> pd.Timestamp:
    """
    Returnerar äldsta (minsta) tidsstämpeln bland alla TS_-kolumner på raden.
    Om inga TS finns returneras NaT.
    """
    dates = []
    for ts_col in TS_FIELDS.values():
        if ts_col in row and str(row.get(ts_col, "")).strip():
            try:
                d = pd.to_datetime(str(row[ts_col]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    if not dates:
        return pd.NaT
    return pd.Series(dates).min()


def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lägger in två hjälp-kolumner:
      - _oldest_any_ts (Timestamp eller NaT)
      - _oldest_any_ts_fill (NaT ersatt med långt fram i tiden för sortering)
    """
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(oldest_any_ts, axis=1)
    # För stabil sort: fyll NaT med ett sent datum så att poster utan TS hamnar sist
    filler = pd.Timestamp("2099-12-31")
    work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(filler)
    return work


def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Identifierar bolag som sannolikt kräver manuell hantering:
    - Saknar något kärnfältvärde (<= 0) ELLER saknar TS på spårade fält
    - ELLER där äldsta TS är äldre än 'older_than_days' dagar.

    Returnerar en tabell med Ticker, Bolagsnamn, Äldsta TS, flaggor.
    """
    # Kärnfält vi gärna vill ha > 0
    need_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år"
    ]
    # Relevanta TS-kolumner som finns i df
    ts_cols = [TS_FIELDS[c] for c in TS_FIELDS if c in need_cols and TS_FIELDS[c] in df.columns]

    # Cutoff
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=int(older_than_days)))

    out_rows = []
    for _, r in df.iterrows():
        # Saknade värden?
        missing_val = False
        for c in need_cols:
            if c in df.columns:
                try:
                    v = float(r.get(c, 0.0) or 0.0)
                except Exception:
                    v = 0.0
                if v <= 0.0:
                    missing_val = True
                    break

        # Saknar TS?
        missing_ts = False
        for ts in ts_cols:
            val = str(r.get(ts, "")).strip()
            if not val:
                missing_ts = True
                break

        # Äldsta TS
        oldest = oldest_any_ts(r)
        oldest_dt = oldest if isinstance(oldest, pd.Timestamp) else pd.NaT
        too_old = False
        if pd.notna(oldest_dt):
            try:
                too_old = oldest_dt < cutoff
            except Exception:
                too_old = False
        else:
            # Om helt utan TS – markera inte automatiskt som "för gammal",
            # men raden fångas ändå av 'missing_ts' ovan.
            too_old = False

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Äldsta TS": oldest_dt.strftime("%Y-%m-%d") if pd.notna(oldest_dt) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
                "För gammal TS?": "Ja" if too_old else "Nej",
            })

    return pd.DataFrame(out_rows)


# --- Själva vyn --------------------------------------------------------------

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df)
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)

    cols_show = ["Ticker","Bolagsnamn"]
    # visa alla relevanta TS-kolumner som faktiskt finns
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns:
            cols_show.append(k)
    cols_show.append("_oldest_any_ts")

    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Kräver manuell hantering?
    st.subheader("🛠️ Kräver manuell hantering")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30)
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga tydliga kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Senaste körlogg (om du nyss kört batch/auto i sidopanelen)
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log")
    if not log:
        st.info("Ingen auto-/batchkörning loggad i denna session ännu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ändringar** (ticker → fält)")
            if log.get("changed"):
                st.json(log["changed"])
            else:
                st.write("–")
        with col2:
            st.markdown("**Missar** (ticker → fält som ej uppdaterades)")
            if log.get("misses"):
                st.json(log["misses"])
            else:
                st.write("–")
        if "debug_first_20" in log:
            st.markdown("**Debug (första 20)**")
            st.json(log.get("debug_first_20", []))
