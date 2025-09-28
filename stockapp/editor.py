# stockapp/editor.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Dict, Optional

from .sources import (
    update_price_only, update_full_for_ticker,
    _now_stamp, _ensure_col, _stamp_ts, _safe_float
)

MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

def _recalc_simple_row(row: pd.Series) -> pd.Series:
    """Beräkna P/S-snitt, Omsättning om 2/3 år (från 'Omsättning nästa år' + CAGR clamp) samt Riktkurser."""
    # P/S-snitt
    ps_clean = []
    for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
        v = _safe_float(row.get(k), 0.0)
        if v > 0: ps_clean.append(v)
    row["P/S-snitt"] = round(float(np.mean(ps_clean)), 2) if ps_clean else _safe_float(row.get("P/S-snitt"), 0.0)

    # CAGR clamp
    cagr = _safe_float(row.get("CAGR 5 år (%)"), 0.0)
    just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
    g = just_cagr / 100.0

    # Omsättning -> 2/3 år
    oms_next = _safe_float(row.get("Omsättning nästa år"), 0.0)
    if oms_next > 0:
        row["Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
        row["Omsättning om 3 år"] = round(oms_next * ((1.0 + g)**2), 2)

    # Riktkurser
    ps_avg = _safe_float(row.get("P/S-snitt"), 0.0)
    shares_m = _safe_float(row.get("Utestående aktier"), 0.0)  # i miljoner
    shares = shares_m * 1e6
    if ps_avg > 0 and shares > 0:
        row["Riktkurs idag"]    = round((_safe_float(row.get("Omsättning idag"), 0.0)     * ps_avg) / shares, 2)
        row["Riktkurs om 1 år"] = round((_safe_float(row.get("Omsättning nästa år"), 0.0) * ps_avg) / shares, 2)
        row["Riktkurs om 2 år"] = round((_safe_float(row.get("Omsättning om 2 år"), 0.0)  * ps_avg) / shares, 2)
        row["Riktkurs om 3 år"] = round((_safe_float(row.get("Omsättning om 3 år"), 0.0)  * ps_avg) / shares, 2)
    else:
        for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
            _ensure_col(row.to_frame().T, k)  # no-op för Series
            row[k] = _safe_float(row.get(k), 0.0)
    return row

def _recalc_simple(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    work = df.copy()
    for col in ["P/S-snitt","Omsättning om 2 år","Omsättning om 3 år","Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
        _ensure_col(work, col)
    work = work.apply(_recalc_simple_row, axis=1)
    return work

def _oldest_ts_pair(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for col in ["TS_Omsättning idag","TS_Omsättning nästa år"]:
        if col in row and str(row[col]).strip():
            d = pd.to_datetime(str(row[col]), errors="coerce")
            if pd.notna(d):
                dates.append(d)
    if not dates:
        return None
    return min(dates)

def _prognoslista(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år","Äldst TS"])
    work = df.copy()
    for c in ["TS_Omsättning idag","TS_Omsättning nästa år"]:
        _ensure_col(work, c)
    work["Äldst TS"] = work.apply(_oldest_ts_pair, axis=1)
    work = work.sort_values(by=["Äldst TS","Bolagsnamn"], ascending=[True, True])
    vis = work[["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år","Äldst TS"]].head(50)
    return vis

def editor_view(df: pd.DataFrame, user_rates: dict, save_cb=None) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    if df.empty:
        st.info("Databasen är tom. Lägg till ditt första bolag.")
    # sorteringsläge
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], index=0)
    if sort_val.startswith("Äldst"):
        # approximera via alla TS-kolumner
        def _oldest_any_ts(row: pd.Series):
            dts = []
            for c in row.index:
                if str(c).startswith("TS_") and str(row[c]).strip():
                    d = pd.to_datetime(str(row[c]), errors="coerce")
                    if pd.notna(d): dts.append(d)
            return min(dts) if dts else pd.NaT
        work = df.copy()
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        vis_df = work.sort_values(by=["_oldest_any_ts","Bolagsnamn"]).drop(columns=["_oldest_any_ts"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    idx_map = {lab: i for i, lab in enumerate(labels)}

    if "edit_idx" not in st.session_state: st.session_state.edit_idx = 0
    st.session_state.edit_idx = min(st.session_state.edit_idx, max(0, len(labels)-1))

    # navigator
    cols_nav = st.columns([1,3,1])
    with cols_nav[0]:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_idx = max(0, st.session_state.edit_idx - 1)
    with cols_nav[1]:
        st.write(f"Post {st.session_state.edit_idx+1}/{max(1,len(labels))}")
    with cols_nav[2]:
        if st.button("➡️ Nästa"):
            st.session_state.edit_idx = min(max(0, len(labels)-1), st.session_state.edit_idx + 1)

    selected = st.selectbox("Välj bolag", ["(nytt)"] + labels, index=(st.session_state.edit_idx+1 if labels else 0))
    if selected != "(nytt)":
        sel_row = vis_df.iloc[idx_map[selected]]
    else:
        sel_row = pd.Series({}, dtype=object)

    # Form
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            tkr   = st.text_input("Ticker (Yahoo-format)", value=str(sel_row.get("Ticker","") if not sel_row.empty else "")).upper().strip()
            name  = st.text_input("Bolagsnamn", value=str(sel_row.get("Bolagsnamn","") if not sel_row.empty else ""))
            valuta= st.text_input("Valuta", value=str(sel_row.get("Valuta","") if not sel_row.empty else "USD")).upper()
            sektor= st.text_input("Sektor", value=str(sel_row.get("Sektor","") if not sel_row.empty else ""))
            utest = st.number_input("Utestående aktier (miljoner)", value=_safe_float(sel_row.get("Utestående aktier"),0.0) if not sel_row.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=_safe_float(sel_row.get("Antal aktier"),0.0) if not sel_row.empty else 0.0)
            gav   = st.number_input("GAV SEK (per aktie)", value=_safe_float(sel_row.get("GAV SEK"),0.0) if not sel_row.empty else 0.0)
        with c2:
            kurs  = st.number_input("Aktuell kurs", value=_safe_float(sel_row.get("Aktuell kurs"),0.0) if not sel_row.empty else 0.0)
            utd   = st.number_input("Årlig utdelning (per aktie)", value=_safe_float(sel_row.get("Årlig utdelning"),0.0) if not sel_row.empty else 0.0)
            ps    = st.number_input("P/S", value=_safe_float(sel_row.get("P/S"),0.0) if not sel_row.empty else 0.0)
            ps1   = st.number_input("P/S Q1", value=_safe_float(sel_row.get("P/S Q1"),0.0) if not sel_row.empty else 0.0)
            ps2   = st.number_input("P/S Q2", value=_safe_float(sel_row.get("P/S Q2"),0.0) if not sel_row.empty else 0.0)
            ps3   = st.number_input("P/S Q3", value=_safe_float(sel_row.get("P/S Q3"),0.0) if not sel_row.empty else 0.0)
            ps4   = st.number_input("P/S Q4", value=_safe_float(sel_row.get("P/S Q4"),0.0) if not sel_row.empty else 0.0)
            oms_i = st.number_input("Omsättning idag (miljoner)", value=_safe_float(sel_row.get("Omsättning idag"),0.0) if not sel_row.empty else 0.0)
            oms_n = st.number_input("Omsättning nästa år (miljoner)", value=_safe_float(sel_row.get("Omsättning nästa år"),0.0) if not sel_row.empty else 0.0)

        cbtn = st.columns(3)
        with cbtn[0]:
            klik_save = st.form_submit_button("💾 Spara")
        with cbtn[1]:
            klik_px = st.form_submit_button("💹 Uppdatera **Kurs** (Yahoo)")
        with cbtn[2]:
            klik_auto = st.form_submit_button("🤖 Full auto (endast denna)")

    # Exekvera knappar
    if klik_px and tkr:
        df, info = update_price_only(df, tkr)
        st.success(f"Kurs uppdaterad för {tkr}.")
        return df

    if klik_auto and tkr:
        df, info = update_full_for_ticker(df, tkr)
        df = _recalc_simple(df)
        st.success(f"Full auto uppdaterad för {tkr}.")
        return df

    if klik_save and tkr:
        # skapa/uppdatera rad
        if "Ticker" not in df.columns:
            df["Ticker"] = ""
        mask = df["Ticker"].astype(str).str.upper() == tkr.upper()
        if not mask.any():
            # lägg ny
            row = {c: "" for c in df.columns}
            row.update({
                "Ticker": tkr, "Bolagsnamn": name, "Valuta": valuta, "Sektor": sektor,
                "Utestående aktier": utest, "Antal aktier": antal, "GAV SEK": gav,
                "Aktuell kurs": kurs, "Årlig utdelning": utd,
                "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
                "Omsättning idag": oms_i, "Omsättning nästa år": oms_n
            })
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            ridx = df.index[df["Ticker"].astype(str).str.upper()==tkr.upper()][0]
        else:
            ridx = df.index[mask][0]
            for k,v in {
                "Bolagsnamn": name, "Valuta": valuta, "Sektor": sektor,
                "Utestående aktier": utest, "Antal aktier": antal, "GAV SEK": gav,
                "Aktuell kurs": kurs, "Årlig utdelning": utd,
                "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
                "Omsättning idag": oms_i, "Omsättning nästa år": oms_n
            }.items():
                _ensure_col(df, k)
                df.at[ridx, k] = v

        # stämpla manuell uppdatering + TS för de fält som ingår
        _ensure_col(df, "Senast manuellt uppdaterad")
        df.at[ridx, "Senast manuellt uppdaterad"] = _now_stamp()
        for f in MANUELL_FALT_FOR_DATUM:
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]:
                _stamp_ts(df, ridx, f)

        # räkna om
        df = _recalc_simple(df)

        # spara via callback om finns
        if callable(save_cb):
            try:
                save_cb(df)
                st.success("Sparat.")
            except Exception as e:
                st.warning(f"Kunde inte spara: {e}")
        else:
            st.success("Uppdaterat (ej skrivet till Sheets i denna vy).")

        return df

    # Prognoslista (manuell) – de fält du alltid vill mata in själv
    st.subheader("📝 Manuell prognoslista (uppdatera omsättningar)")
    prog = _prognoslista(df)
    if prog.empty:
        st.info("Inget att visa ännu.")
    else:
        st.dataframe(prog, use_container_width=True, hide_index=True)

    return df
