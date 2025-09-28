# -*- coding: utf-8 -*-
"""
stockapp/views/edit.py

Lägg till / uppdatera-bolag:
- Välj bolag + bläddra
- Form med basfält + manuella prognosfält + GAV i SEK
- Knappar: Uppdatera kurs (runner_price_only), Full uppdatering (runner_full), Spara
- TS-stämplar MANUELLA prognosfält vid ändring: Omsättning idag / Omsättning nästa år
- Duplikat-skydd på ticker
- Visar "Manuell prognoslista" (äldsta TS för de två fälten)
"""

from __future__ import annotations
from typing import Dict, Optional, Callable
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# TS-fält
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

MANUELLA_PROGNOSFALT = ["Omsättning idag", "Omsättning nästa år"]

def _now_date_str() -> str:
    try:
        import pytz
        return datetime.now(pytz.timezone("Europe/Stockholm")).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ att viktiga kolumner finns (utan att förstöra df).
    """
    base_cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
        "Utestående aktier","Antal aktier","Årlig utdelning",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","P/S-snitt",
        # meta
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        # TS
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
        "TS_Omsättning idag","TS_Omsättning nästa år",
        # Portfölj
        "GAV SEK",
        # Info
        "Sektor","Industri","Utdelningsyield","Debt/Equity","Bruttomarginal","Nettomarginal","Kassa","EV/EBITDA"
    ]
    for c in base_cols:
        if c not in df.columns:
            if c.startswith("TS_") or c in ["Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Bolagsnamn","Valuta","Sektor","Industri"]:
                df[c] = ""
            elif c in ["Ticker"]:
                df[c] = ""
            elif c in ["GAV SEK"]:
                df[c] = 0.0
            else:
                df[c] = 0.0
    return df

def _stamp_manual(df: pd.DataFrame, ridx: int, fields_changed):
    # sätt "Senast manuellt uppdaterad" + TS för just ändrade manuella fält
    if fields_changed:
        if "Senast manuellt uppdaterad" in df.columns:
            df.at[ridx, "Senast manuellt uppdaterad"] = _now_date_str()
        for f in fields_changed:
            ts_col = TS_FIELDS.get(f)
            if ts_col in df.columns:
                df.at[ridx, ts_col] = _now_date_str()

def _manual_prognoslista(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabell för Omsättning idag / nästa år sorterad på äldsta TS för just dessa fält.
    """
    work = df.copy()
    def _pick_oldest_for_row(row):
        dates = []
        for f in MANUELLA_PROGNOSFALT:
            ts = TS_FIELDS.get(f, "")
            if ts and ts in row and str(row.get(ts,"")).strip():
                d = pd.to_datetime(str(row.get(ts,"")), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
        if not dates:
            return pd.NaT
        return pd.Series(dates).min()

    work["_oldest_manual"] = work.apply(_pick_oldest_for_row, axis=1)
    filler = pd.Timestamp("2099-12-31")
    work["_oldest_manual_fill"] = work["_oldest_manual"].fillna(filler)
    out = work.sort_values(by=["_oldest_manual_fill","Bolagsnamn","Ticker"])[
        ["Ticker","Bolagsnamn","Omsättning idag","Omsättning nästa år","TS_Omsättning idag","TS_Omsättning nästa år","_oldest_manual"]
    ].rename(columns={"_oldest_manual":"Äldsta TS (manuell)"})
    return out

def lagg_till_eller_uppdatera(df: pd.DataFrame,
                              user_rates: Dict[str, float],
                              runner_full: Optional[Callable] = None,
                              runner_price_only: Optional[Callable] = None,
                              save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
                              recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
    """
    Huvudvy för add/edit + on-demand uppdateringar för en specifik ticker.
    Returnerar ev. uppdaterad df (efter save / uppdateringar).
    """
    st.header("➕ Lägg till / uppdatera bolag")

    if df.empty:
        df = pd.DataFrame()
    df = _ensure_cols(df)

    # Sorteringsval för listan
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        # enkel äldsta TS över alla spårade, som i kontroll-vy
        def _oldest_any_ts(row):
            dates = []
            for ts_col in TS_FIELDS.values():
                if ts_col in row and str(row.get(ts_col,"")).strip():
                    d = pd.to_datetime(str(row.get(ts_col,"")), errors="coerce")
                    if pd.notna(d):
                        dates.append(d)
            if not dates: return pd.NaT
            return pd.Series(dates).min()
        work = df.copy()
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    labels = [f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})" for _, r in vis_df.iterrows()]
    idx_map = {i: vis_df.index[i] for i in range(len(vis_df))}
    if "edit_idx" not in st.session_state:
        st.session_state.edit_idx = 0
    st.session_state.edit_idx = max(0, min(st.session_state.edit_idx, len(vis_df)-1))

    # Val av befintlig eller nytt
    col_top1, col_top2 = st.columns([3,1])
    with col_top1:
        sel = st.selectbox("Välj bolag (lämna tomt för nytt):", options=["(nytt)"] + list(range(len(labels))), format_func=lambda x: "(nytt)" if x=="(nytt)" else labels[x], index=0 if len(labels)==0 else st.session_state.edit_idx+1)
    with col_top2:
        # Bläddring
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬅️ Föregående", use_container_width=True, disabled=len(labels)==0):
                st.session_state.edit_idx = max(0, st.session_state.edit_idx-1)
        with c2:
            if st.button("➡️ Nästa", use_container_width=True, disabled=len(labels)==0):
                st.session_state.edit_idx = min(len(labels)-1, st.session_state.edit_idx+1)

    is_new = (sel == "(nytt)")
    bef = pd.Series({}, dtype=object) if is_new else df.loc[idx_map[st.session_state.edit_idx]]

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker_in = st.text_input("Ticker (Yahoo-format)", value=str(bef.get("Ticker","")) if not is_new else "").upper().strip()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0) or 0.0))
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0) or 0.0))
            gavsek = st.number_input("GAV SEK", value=float(bef.get("GAV SEK",0.0) or 0.0))

            ps   = st.number_input("P/S", value=float(bef.get("P/S",0.0) or 0.0))
            ps1  = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0) or 0.0))
            ps2  = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0) or 0.0))
            ps3  = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0) or 0.0))
            ps4  = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0) or 0.0))
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner) – MANUELL", value=float(bef.get("Omsättning idag",0.0) or 0.0))
            oms_next = st.number_input("Omsättning nästa år (miljoner) – MANUELL", value=float(bef.get("Omsättning nästa år",0.0) or 0.0))

            st.markdown("**Vid Spara:**")
            st.write("- Uppdaterar raden och stämplar *manuella* prognosfält om ändrade.")
            st.write("- Kör beräkningar (om du skickar in `recompute_cb`).")
            st.write("- Sparar (om du skickar in `save_cb`).")

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        run_price = col_btn1.form_submit_button("🔁 Uppdatera kurs (vald ticker)")
        run_full  = col_btn2.form_submit_button("🔄 Full uppdatering (vald ticker)")
        saved     = col_btn3.form_submit_button("💾 Spara")

    # --- Knapplogik -----------------------------------------------------------
    df_out = df

    # Uppdatera KURS
    if run_price:
        if not ticker_in:
            st.error("Ange ticker först.")
        elif not callable(runner_price_only):
            st.error("runner_price_only saknas.")
        else:
            tkr = ticker_in.upper()
            if tkr not in df_out["Ticker"].astype(str).str.upper().values:
                st.error(f"{tkr} hittades inte i tabellen.")
            else:
                df_out, changed, src = runner_price_only(tkr, df_out, user_rates)
                if callable(recompute_cb):
                    try:
                        df_out = recompute_cb(df_out)
                    except Exception:
                        pass
                if callable(save_cb):
                    try:
                        save_cb(df_out)
                    except Exception as e:
                        st.warning(f"Kunde inte spara efter kursuppdatering: {e}")
                st.success(f"Kurs uppdaterad för {tkr}. Ändrade fält: {', '.join(changed) if changed else '(inga)'}")

    # Full uppdatering
    if run_full:
        if not ticker_in:
            st.error("Ange ticker först.")
        elif not callable(runner_full):
            st.error("runner_full saknas.")
        else:
            tkr = ticker_in.upper()
            if tkr not in df_out["Ticker"].astype(str).str.upper().values:
                st.error(f"{tkr} hittades inte i tabellen.")
            else:
                df_out, changed, src = runner_full(tkr, df_out, user_rates)
                if callable(recompute_cb):
                    try:
                        df_out = recompute_cb(df_out)
                    except Exception:
                        pass
                if callable(save_cb):
                    try:
                        save_cb(df_out)
                    except Exception as e:
                        st.warning(f"Kunde inte spara efter full uppdatering: {e}")
                st.success(f"Full uppdatering för {tkr}. Ändrade fält: {', '.join(changed) if changed else '(inga)'}")

    # Spara (lägg till eller uppdatera)
    if saved:
        if not ticker_in:
            st.error("Ticker måste anges.")
            return df_out
        tkr = ticker_in.upper()

        # Duplikatskydd (case-insensitive)
        exists_mask = df_out["Ticker"].astype(str).str.upper() == tkr
        if is_new and exists_mask.any():
            st.error(f"Ticker {tkr} finns redan.")
            return df_out

        # Skriv in
        new_vals = {
            "Ticker": tkr,
            "Utestående aktier": float(utest or 0.0),
            "Antal aktier": float(antal or 0.0),
            "GAV SEK": float(gavsek or 0.0),
            "P/S": float(ps or 0.0),
            "P/S Q1": float(ps1 or 0.0),
            "P/S Q2": float(ps2 or 0.0),
            "P/S Q3": float(ps3 or 0.0),
            "P/S Q4": float(ps4 or 0.0),
            # MANUELLA (stämplas separat om ändrade)
            "Omsättning idag": float(oms_idag or 0.0),
            "Omsättning nästa år": float(oms_next or 0.0),
        }

        if is_new:
            # skapa rad
            row = {c: ("" if (c.startswith("TS_") or c in ["Bolagsnamn","Valuta","Sektor","Industri","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]) else 0.0) for c in df_out.columns}
            row.update(new_vals)
            df_out = pd.concat([df_out, pd.DataFrame([row])], ignore_index=True)
            ridx = df_out.index[df_out["Ticker"].astype(str).str.upper() == tkr][0]
            # stämpla manuellt om fälten har satts != 0
            changed_manual = [f for f in MANUELLA_PROGNOSFALT if float(new_vals.get(f,0.0)) != 0.0]
            _stamp_manual(df_out, ridx, changed_manual)
            st.success(f"{tkr} tillagd.")
        else:
            ridx = df_out.index[exists_mask][0]
            # avgör vilka manuella fält som ändrats
            changed_manual = []
            for f in MANUELLA_PROGNOSFALT:
                try:
                    before = float(df_out.at[ridx, f] or 0.0)
                except Exception:
                    before = 0.0
                after = float(new_vals.get(f,0.0) or 0.0)
                if before != after:
                    changed_manual.append(f)
            # skriv alla new_vals
            for k, v in new_vals.items():
                df_out.at[ridx, k] = v
            _stamp_manual(df_out, ridx, changed_manual)
            st.success(f"{tkr} uppdaterad. {'(Manuella prognosfält stämplade.)' if changed_manual else ''}")

        # Recompute + Save
        if callable(recompute_cb):
            try:
                df_out = recompute_cb(df_out)
            except Exception:
                pass
        if callable(save_cb):
            try:
                save_cb(df_out)
            except Exception as e:
                st.warning(f"Kunde inte spara: {e}")

    # --- Manuell prognoslista (äldst TS) -------------------------------------
    st.markdown("### 📋 Manuell prognoslista – äldst TS (Omsättning idag / nästa år)")
    man = _manual_prognoslista(df_out)
    if man.empty:
        st.info("Inga poster att visa.")
    else:
        st.dataframe(man, use_container_width=True, hide_index=True)

    return df_out
