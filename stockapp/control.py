# -*- coding: utf-8 -*-
"""
Kontrollvy: datakvalitet, dubbletter, åldrande prognoser m.m.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import streamlit as st

from .config import FINAL_COLS, TS_FIELDS
from .utils import ensure_schema, add_oldest_ts_col


def _build_requires_manual_df(df: pd.DataFrame, older_than_days: Optional[int] = None) -> pd.DataFrame:
    df = df.copy()
    for ts in TS_FIELDS:
        if ts not in df.columns:
            df[ts] = None
    need = df[["Ticker"] + TS_FIELDS].copy()
    need["Senaste TS (min av två)"] = need[TS_FIELDS].apply(
        lambda r: min([x for x in r.values.tolist() if x], default=None), axis=1
    )
    if older_than_days is not None and older_than_days > 0:
        # Filtrera på äldre än X dagar
        import datetime as _dt
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=int(older_than_days))
        need = need[
            need["Senaste TS (min av två)"].apply(
                lambda x: (x is None) or (pd.to_datetime(x, utc=True, errors="coerce") < cutoff)
            )
        ]
    return need.sort_values(by="Senaste TS (min av två)", ascending=True, na_position="first")


def kontrollvy(df: pd.DataFrame) -> None:
    df = ensure_schema(df, FINAL_COLS)
    st.subheader("Kontroll")

    # Dubbletter
    dups = df["Ticker"].astype(str).str.upper().duplicated(keep="first")
    if dups.any():
        st.warning("Dubbletter hittade – behåll första, ta bort resten:")
        st.dataframe(df.loc[dups, ["Ticker"]], hide_index=True, use_container_width=True)
    else:
        st.info("Inga dubbletter av tickers.")

    st.markdown("---")

    # Prognoser
    older_days = st.number_input("Visa manuella prognoser äldre än (dagar):", min_value=0, value=120, step=1)
    need = _build_requires_manual_df(df, older_than_days=int(older_days))
    st.write(f"Poster som kräver manuell prognos (äldre än {int(older_days)} dagar eller saknas):")
    st.dataframe(need, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Tidsstämplar (översikt)
    if "TS Full" in df.columns or "TS Kurs" in df.columns:
        work = add_oldest_ts_col(df.copy())
        st.write("Äldsta uppdaterings-tidsstämplar (översikt):")
        st.dataframe(
            work[["Ticker", "TS Kurs", "TS Full", "OldestTS"]].sort_values("OldestTS", ascending=True, na_position="first"),
            use_container_width=True,
            hide_index=True,
        )
