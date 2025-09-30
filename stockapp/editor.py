# -*- coding: utf-8 -*-
"""
UI: Lägg till / uppdatera bolag.
- Vänster: välj/skriv ticker, antal, GAV (SEK), manuella prognoser (omsättning i/ nästa år)
- Knappar: "Uppdatera kurs" och "Full uppdatering" (använder orchestrator-runner i session)
- Höger: "Manuell prognoslista" – bolag där de två prognosfälten saknar/har äldst TS
Returnerar ev. uppdaterad DataFrame (annars None).
"""

from __future__ import annotations
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from .config import FINAL_COLS, TS_FIELDS
from .utils import ensure_schema, now_stamp, stamp_fields_ts, add_oldest_ts_col
from .fetchers.orchestrator import run_update_full, run_update_price_only


def _put_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Infogar/uppdaterar rad efter 'Ticker'."""
    df = ensure_schema(df, FINAL_COLS)
    tkr = str(row.get("Ticker", "")).strip().upper()
    if not tkr:
        return df
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    mask = df["Ticker"].str.upper() == tkr
    if mask.any():
        idx = df.index[mask][0]
        for k, v in row.items():
            if k in df.columns:
                df.at[idx, k] = v
            else:
                df[k] = None
                df.at[idx, k] = v
    else:
        # komplettera med saknade kolumner
        for k in row.keys():
            if k not in df.columns:
                df[k] = None
        df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
    return df


def _manual_list(df: pd.DataFrame) -> pd.DataFrame:
    """Bygger listan på bolag där prognosfälten behöver manuell uppdatering."""
    df = df.copy()
    for ts in TS_FIELDS:
        if ts not in df.columns:
            df[ts] = None
    work = df[["Ticker"] + TS_FIELDS].copy()
    work["Senaste TS (min av två)"] = work[TS_FIELDS].apply(
        lambda r: min([x for x in r.values.tolist() if x], default=None), axis=1
    )
    return work.sort_values(by="Senaste TS (min av två)", ascending=True, na_position="first")


def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> Optional[pd.DataFrame]:
    df = ensure_schema(df, FINAL_COLS)
    st.subheader("Lägg till / uppdatera bolag")

    left, right = st.columns([1, 1])

    # ------------------- vänster – editera --------------------------------
    with left:
        existing = ["<ny>"] + sorted([x for x in df["Ticker"].astype(str).tolist() if x])
        pick = st.selectbox("Välj ticker", existing, index=0)
        if pick != "<ny>":
            row = df[df["Ticker"] == pick].iloc[0].to_dict()
        else:
            row = {"Ticker": ""}

        with st.form("edit_form", clear_on_submit=False):
            tkr = st.text_input("Ticker", value=str(row.get("Ticker", ""))).upper().strip()
            namn = st.text_input("Bolagsnamn", value=str(row.get("Bolagsnamn", "")))
            valuta = st.text_input("Valuta (USD/EUR/CAD/NOK/SEK)", value=str(row.get("Valuta", "USD")).upper())
            sektor = st.text_input("Sektor", value=str(row.get("Sektor", "")))
            antal = st.number_input("Antal du äger", min_value=0.0, value=float(row.get("Antal du äger", 0.0)), step=1.0)
            gav = st.number_input("GAV (SEK)", min_value=0.0, value=float(row.get("GAV (SEK)", 0.0)), step=0.01)

            st.markdown("**Manuella prognoser (M = miljoner i bolagets valuta)**")
            prog_i_ar = st.number_input(
                "Omsättning i år (M)", min_value=0.0, value=float(row.get("Omsättning i år (M)", 0.0)), step=1.0
            )
            prog_nasta = st.number_input(
                "Omsättning nästa år (M)", min_value=0.0, value=float(row.get("Omsättning nästa år (M)", 0.0)), step=1.0
            )

            c1, c2, c3 = st.columns(3)
            do_price = c1.form_submit_button("Uppdatera kurs")
            do_full = c2.form_submit_button("Full uppdatering")
            do_save = c3.form_submit_button("Spara rad")

        if do_price and tkr:
            try:
                upd, log = run_update_price_only(tkr, user_rates) if "_runner" not in st.session_state \
                    else st.session_state["_runner"](tkr, user_rates, "price")
                upd["Ticker"] = tkr
                df = _put_row(df, upd)
                st.success(f"Kurs uppdaterad för {tkr}")
                st.code(log)
            except Exception as e:
                st.error(f"Kunde inte uppdatera kurs: {e}")

        if do_full and tkr:
            try:
                upd, log = run_update_full(tkr, user_rates) if "_runner" not in st.session_state \
                    else st.session_state["_runner"](tkr, user_rates, "full")
                upd["Ticker"] = tkr
                df = _put_row(df, upd)
                st.success(f"Full uppdatering klar för {tkr}")
                st.code(log)
            except Exception as e:
                st.error(f"Kunde inte göra full uppdatering: {e}")

        if do_save:
            if not tkr:
                st.warning("Ange en ticker först.")
            else:
                newrow = {
                    "Ticker": tkr,
                    "Bolagsnamn": namn,
                    "Valuta": valuta,
                    "Sektor": sektor,
                    "Antal du äger": antal,
                    "GAV (SEK)": gav,
                    "Omsättning i år (M)": prog_i_ar,
                    "Omsättning nästa år (M)": prog_nasta,
                }
                # stämpla manuella fält om ändrade
                if prog_i_ar != row.get("Omsättning i år (M)", None):
                    newrow["TS Omsättning i år"] = now_stamp()
                if prog_nasta != row.get("Omsättning nästa år (M)", None):
                    newrow["TS Omsättning nästa år"] = now_stamp()

                df = _put_row(df, newrow)
                st.success("Rad sparad (lokalt). Glöm inte att byta vy för att skriva till Google Sheet via huvudlogik.")
                return df  # signal till app.py att skriva

    # ------------------- höger – manuell prognoslista ---------------------
    with right:
        st.subheader("Manuell prognoslista (äldst först)")
        need = _manual_list(df)
        st.dataframe(need, use_container_width=True, hide_index=True)

    return None
