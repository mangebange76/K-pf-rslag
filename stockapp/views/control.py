# -*- coding: utf-8 -*-
"""
Kontroll-vy:
- Äldst uppdaterade (läser alla TS_-kolumner och visar topp 20)
- Kräver manuell hantering (saknade kärnfält eller mycket gamla TS)
- Senaste körlogg från st.session_state["last_auto_log"] om sådan finns
Robust mot saknade kolumner.
"""

from __future__ import annotations
from typing import List, Optional
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta


# ---------------- Hjälpare ----------------

def _parse_dt_safe(s: str) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        d = pd.to_datetime(str(s).strip(), errors="coerce")
        return d if pd.notna(d) else None
    except Exception:
        return None


def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """Minst datum över alla kolumner som börjar med 'TS_'."""
    ts_cols = [c for c in row.index if str(c).startswith("TS_")]
    dates: List[pd.Timestamp] = []
    for c in ts_cols:
        d = _parse_dt_safe(row.get(c, ""))
        if d is not None:
            dates.append(d)
    return min(dates) if dates else None


def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(oldest_any_ts, axis=1)
    # Fyll hjälpkolumn för sortering
    fill_val = pd.Timestamp("2099-12-31")
    work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(fill_val)
    return work


def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Kandidater för manuell hantering om:
      - saknar något kärnfält (om kolumnen finns) eller
      - saknar någon TS_ (om kolumnen finns) eller
      - äldsta TS äldre än 'older_than_days'
    """
    need_cols_base = ["Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
                      "Omsättning idag", "Omsättning nästa år"]
    need_cols = [c for c in need_cols_base if c in df.columns]

    # motsvarande TS_-kolumner om de finns
    ts_cols = [f"TS_{c}" for c in need_cols if f"TS_{c}" in df.columns]

    out_rows = []
    cutoff = datetime.now() - timedelta(days=int(older_than_days))

    for _, r in df.iterrows():
        # saknade numeric-värden (<= 0.0) på fält som finns
        missing_val = False
        for c in need_cols:
            try:
                v = float(r.get(c, 0.0) or 0.0)
                if v <= 0.0:
                    missing_val = True
                    break
            except Exception:
                # Om ej numeriskt, betrakta som saknat
                missing_val = True
                break

        # saknade tidsstämplar på motsvarande TS_-fält
        missing_ts = False
        for ts in ts_cols:
            if not str(r.get(ts, "")).strip():
                missing_ts = True
                break

        oldest = oldest_any_ts(r)
        oldest_dt = oldest.to_pydatetime() if isinstance(oldest, pd.Timestamp) else None
        too_old = (oldest_dt is not None and oldest_dt < cutoff)

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker", ""),
                "Bolagsnamn": r.get("Bolagsnamn", ""),
                "Äldsta TS": oldest.strftime("%Y-%m-%d") if isinstance(oldest, pd.Timestamp) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })

    return pd.DataFrame(out_rows)


# ---------------- Själva vyn ----------------

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla TS_-kolumner)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df)
    vis = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn"]).head(20)
    cols_show = ["Ticker", "Bolagsnamn"]
    # visa alla TS_-kolumner som finns
    cols_show += [c for c in vis.columns if str(c).startswith("TS_")]
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Kräver manuell hantering?
    st.subheader("🛠️ Kräver manuell hantering")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30)
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Senaste körlogg från Auto/Batch (om finns)
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log")
    if not log:
        st.info("Ingen auto- eller batchkörning loggad i denna session ännu.")
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
