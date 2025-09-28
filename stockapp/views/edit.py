# stockapp/views/edit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable
from datetime import datetime

# Typer för callbacks/runners
RunnerFn = Callable[[str], tuple[Dict, Dict]]        # -> (vals, debug)
SaveFn   = Callable[[pd.DataFrame], None]
RecompFn = Callable[[pd.DataFrame], pd.DataFrame]

# Manuella fält som ska tidsstämplas när de ändras i formuläret
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# Hjälpare
def _now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _ensure_col(df: pd.DataFrame, col: str, default):
    if col not in df.columns:
        df[col] = default

def _stamp_ts_for_field(df: pd.DataFrame, ridx: int, field: str):
    ts_col = f"TS_{field}"
    if ts_col in df.columns:
        df.at[ridx, ts_col] = _now_stamp()

def _note_manual_update(df: pd.DataFrame, ridx: int):
    _ensure_col(df, "Senast manuellt uppdaterad", "")
    df.at[ridx, "Senast manuellt uppdaterad"] = _now_stamp()

def _note_auto_update(df: pd.DataFrame, ridx: int, source: str):
    _ensure_col(df, "Senast auto-uppdaterad", "")
    _ensure_col(df, "Senast uppdaterad källa", "")
    df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = source

def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for c in row.index:
        if str(c).startswith("TS_"):
            s = str(row.get(c, "")).strip()
            if s:
                d = pd.to_datetime(s, errors="coerce")
                if pd.notna(d):
                    dates.append(d)
    return min(dates) if dates else None

def _build_manual_forecast_list(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    # Fokusera ENBART på de två prognosfälten
    ts_cols = []
    for base in ["Omsättning idag","Omsättning nästa år"]:
        c = f"TS_{base}"
        if c in df.columns:
            ts_cols.append(c)

    out = []
    for _, r in df.iterrows():
        oldest = None
        for c in ts_cols:
            s = str(r.get(c, "")).strip()
            if not s:
                continue
            d = pd.to_datetime(s, errors="coerce")
            if pd.notna(d):
                if (oldest is None) or (d < oldest):
                    oldest = d
        out.append({
            "Ticker": r.get("Ticker",""),
            "Bolagsnamn": r.get("Bolagsnamn",""),
            "TS_Omsättning idag": str(r.get("TS_Omsättning idag","")),
            "TS_Omsättning nästa år": str(r.get("TS_Omsättning nästa år","")),
            "Äldsta TS": oldest.strftime("%Y-%m-%d") if oldest else "",
        })
    res = pd.DataFrame(out)
    if res.empty:
        return res
    # sortera med tomma sist
    res["_sort"] = res["Äldsta TS"].replace("", "9999-12-31")
    res = res.sort_values(by="_sort").drop(columns=["_sort"])
    return res

def _apply_vals_to_row(df: pd.DataFrame, ridx: int, vals: Dict, source_label: str):
    # skriv alltid och stämpla TS_ om finns
    for k, v in vals.items():
        if k not in df.columns:
            if isinstance(v, (int, float, np.floating)):
                df[k] = 0.0
            else:
                df[k] = ""
        df.at[ridx, k] = v
        _stamp_ts_for_field(df, ridx, k)
    _note_auto_update(df, ridx, source_label)

def _num(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0

# -----------------------------
# Huvudvy
# -----------------------------

def lagg_till_eller_uppdatera(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: SaveFn | None = None,
    recompute_cb: RecompFn | None = None,
    runner_full: RunnerFn | None = None,
    runner_price: RunnerFn | None = None,
) -> pd.DataFrame:
    """
    Lägg till / uppdatera-bolag med:
      - robust bläddring
      - enskild ticker-uppdatering (pris / full auto)
      - manuell prognoslista
    """
    st.header("➕ Lägg till / uppdatera bolag")

    # säkerställ viktiga kolumner
    must_num_cols = [
        "Utestående aktier","Antal aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Market Cap (nu)","FCF TTM (valuta)","Kassa (valuta)","Debt/Equity",
        "Bruttomarginal (%)","Nettomarginal (%)"
    ]
    for c in must_num_cols:
        _ensure_col(df, c, 0.0)
    _ensure_col(df, "Bolagsnamn", "")
    _ensure_col(df, "Valuta", "")
    _ensure_col(df, "Sektor", "")
    _ensure_col(df, "Senast manuellt uppdaterad", "")
    _ensure_col(df, "Senast auto-uppdaterad", "")
    _ensure_col(df, "Senast uppdaterad källa", "")
    # GAV i SEK/aktie
    _ensure_col(df, "GAV (SEK/aktie)", 0.0)

    # Sorteringsval
    sort_val = st.selectbox("Sortera för redigering", ["Äldst uppdaterade först (alla TS)","A–Ö (bolagsnamn)"])
    if sort_val.startswith("Äldst"):
        work = df.copy()
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # Urvalslista
    labels = ["— Nytt bolag —"] + [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    map_ticker = {f"{r['Bolagsnamn']} ({r['Ticker']})": str(r['Ticker']).upper().strip() for _, r in vis_df.iterrows()}

    # Robust index i session_state
    st.session_state.setdefault("edit_idx", 0)
    st.session_state["edit_idx"] = max(0, min(st.session_state["edit_idx"], len(labels)-1))

    # Välj via selectbox
    sel_label = st.selectbox("Välj bolag", options=labels, index=st.session_state["edit_idx"], key="edit_select_label")

    # Bläddring
    c_prev, c_pos, c_next = st.columns([1,2,1])
    with c_prev:
        if st.button("⬅️ Föregående"):
            st.session_state["edit_idx"] = max(0, st.session_state["edit_idx"] - 1)
    with c_pos:
        st.write(f"Post {st.session_state['edit_idx']+1}/{max(1,len(labels))}")
    with c_next:
        if st.button("➡️ Nästa"):
            st.session_state["edit_idx"] = min(len(labels)-1, st.session_state["edit_idx"] + 1)

    # Auto-nästa toggle
    auto_next = st.checkbox("Gå vidare till nästa efter uppdatering", value=True)

    # Avgör om nytt eller befintligt
    is_new = (sel_label == "— Nytt bolag —")
    if not is_new and sel_label in map_ticker:
        tkr = map_ticker[sel_label]
        ridxs = df.index[df["Ticker"].astype(str).str.upper().str.strip() == tkr]
        if len(ridxs) == 0:
            st.warning("Valt bolag hittades inte i tabellen (kan ha raderats). Välj igen eller skapa nytt.")
            is_new = True
        else:
            ridx = int(ridxs[0])
            bef = df.loc[ridx]
    else:
        bef = pd.Series({})

    # Visa statusbadges
    if not is_new and not bef.empty:
        colb1, colb2, colb3 = st.columns(3)
        with colb1:
            st.info(f"Senast manuellt: {bef.get('Senast manuellt uppdaterad','') or '–'}")
        with colb2:
            st.info(f"Senast auto: {bef.get('Senast auto-uppdaterad','') or '–'}")
        with colb3:
            st.info(f"Källa: {bef.get('Senast uppdaterad källa','') or '–'}")

    # Form
    with st.form("form_edit_company"):
        c1, c2 = st.columns(2)
        with c1:
            ticker_in = st.text_input("Ticker (Yahoo-format)", value=(bef.get("Ticker","") if not is_new else ""), disabled=(not is_new)).upper()
            namn_in   = st.text_input("Bolagsnamn", value=str(bef.get("Bolagsnamn","")) if not is_new else "")
            valuta_in = st.text_input("Valuta (t.ex. USD/EUR/SEK)", value=str(bef.get("Valuta","")) if not is_new else "USD")
            sektor_in = st.text_input("Sektor (valfritt)", value=str(bef.get("Sektor","")) if not is_new else "")

            utest_in  = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0) or 0.0))
            antal_in  = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0) or 0.0))
            gav_in    = st.number_input("GAV (SEK/aktie)", value=float(bef.get("GAV (SEK/aktie)",0.0) or 0.0), step=0.01, format="%.2f")

            pris_in   = st.number_input("Aktuell kurs", value=float(bef.get("Aktuell kurs",0.0) or 0.0), step=0.01)
            ps_in     = st.number_input("P/S", value=float(bef.get("P/S",0.0) or 0.0), step=0.01)
            ps1_in    = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0) or 0.0), step=0.01)
            ps2_in    = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0) or 0.0), step=0.01)
            ps3_in    = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0) or 0.0), step=0.01)
            ps4_in    = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0) or 0.0), step=0.01)

        with c2:
            oms_idag_in  = st.number_input("Omsättning idag (miljoner, MANUELL)", value=float(bef.get("Omsättning idag",0.0) or 0.0), step=1.0)
            oms_next_in  = st.number_input("Omsättning nästa år (miljoner, MANUELL)", value=float(bef.get("Omsättning nästa år",0.0) or 0.0), step=1.0)
            utd_in       = st.number_input("Årlig utdelning (per aktie, i bolagets valuta)", value=float(bef.get("Årlig utdelning",0.0) or 0.0), step=0.01)
            cagr_in      = st.number_input("CAGR 5 år (%)", value=float(bef.get("CAGR 5 år (%)",0.0) or 0.0), step=0.1)
            # Diagnosfält (visnings-/redigeringsbara)
            de_in        = st.number_input("Debt/Equity", value=float(bef.get("Debt/Equity",0.0) or 0.0), step=0.01)
            gm_in        = st.number_input("Bruttomarginal (%)", value=float(bef.get("Bruttomarginal (%)",0.0) or 0.0), step=0.1)
            nm_in        = st.number_input("Nettomarginal (%)", value=float(bef.get("Nettomarginal (%)",0.0) or 0.0), step=0.1)
            cash_in      = st.number_input("Kassa (valuta)", value=float(bef.get("Kassa (valuta)",0.0) or 0.0), step=1000.0, format="%.0f")
            fcf_in       = st.number_input("FCF TTM (valuta)", value=float(bef.get("FCF TTM (valuta)",0.0) or 0.0), step=1000.0, format="%.0f")

            # Riktkurser (räknas oftast om i recompute_cb, men låter dig skriva över)
            rk0_in = st.number_input("Riktkurs idag", value=float(bef.get("Riktkurs idag",0.0) or 0.0), step=0.01)
            rk1_in = st.number_input("Riktkurs om 1 år", value=float(bef.get("Riktkurs om 1 år",0.0) or 0.0), step=0.01)
            rk2_in = st.number_input("Riktkurs om 2 år", value=float(bef.get("Riktkurs om 2 år",0.0) or 0.0), step=0.01)
            rk3_in = st.number_input("Riktkurs om 3 år", value=float(bef.get("Riktkurs om 3 år",0.0) or 0.0), step=0.01)

        submitted = st.form_submit_button("💾 Spara")

    # Spara-knappen
    if submitted:
        if is_new:
            tkr = ticker_in.strip().upper()
            if not tkr:
                st.error("Ticker krävs för att skapa nytt bolag.")
                return df
            # om redan finns -> uppdatera den raden istället
            ridxs = df.index[df["Ticker"].astype(str).str.upper().str.strip() == tkr]
            if len(ridxs) == 0:
                # skapa ny rad med alla df-kolumner
                new_row = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Sektor","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in df.columns}
                new_row.update({
                    "Ticker": tkr, "Bolagsnamn": namn_in, "Valuta": valuta_in, "Sektor": sektor_in,
                    "Utestående aktier": utest_in, "Antal aktier": antal_in, "GAV (SEK/aktie)": gav_in,
                    "Aktuell kurs": pris_in,
                    "P/S": ps_in, "P/S Q1": ps1_in, "P/S Q2": ps2_in, "P/S Q3": ps3_in, "P/S Q4": ps4_in,
                    "Omsättning idag": oms_idag_in, "Omsättning nästa år": oms_next_in,
                    "Årlig utdelning": utd_in, "CAGR 5 år (%)": cagr_in,
                    "Debt/Equity": de_in, "Bruttomarginal (%)": gm_in, "Nettomarginal (%)": nm_in,
                    "Kassa (valuta)": cash_in, "FCF TTM (valuta)": fcf_in,
                    "Riktkurs idag": rk0_in, "Riktkurs om 1 år": rk1_in, "Riktkurs om 2 år": rk2_in, "Riktkurs om 3 år": rk3_in,
                })
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                ridx = df.index[df["Ticker"].astype(str).str.upper().str.strip()==tkr][0]
                # Manuell stämpel + TS för manuella fält (om värden != 0)
                set_any = False
                for f, v in [("P/S",ps_in),("P/S Q1",ps1_in),("P/S Q2",ps2_in),("P/S Q3",ps3_in),("P/S Q4",ps4_in),("Omsättning idag",oms_idag_in),("Omsättning nästa år",oms_next_in)]:
                    if _num(v) != 0.0:
                        _stamp_ts_for_field(df, ridx, f)
                        set_any = True
                if set_any:
                    _note_manual_update(df, ridx)
            else:
                # uppdatera befintlig
                ridx = int(ridxs[0])
                for k, v in {
                    "Bolagsnamn": namn_in, "Valuta": valuta_in, "Sektor": sektor_in,
                    "Utestående aktier": utest_in, "Antal aktier": antal_in, "GAV (SEK/aktie)": gav_in,
                    "Aktuell kurs": pris_in,
                    "P/S": ps_in, "P/S Q1": ps1_in, "P/S Q2": ps2_in, "P/S Q3": ps3_in, "P/S Q4": ps4_in,
                    "Omsättning idag": oms_idag_in, "Omsättning nästa år": oms_next_in,
                    "Årlig utdelning": utd_in, "CAGR 5 år (%)": cagr_in,
                    "Debt/Equity": de_in, "Bruttomarginal (%)": gm_in, "Nettomarginal (%)": nm_in,
                    "Kassa (valuta)": cash_in, "FCF TTM (valuta)": fcf_in,
                    "Riktkurs idag": rk0_in, "Riktkurs om 1 år": rk1_in, "Riktkurs om 2 år": rk2_in, "Riktkurs om 3 år": rk3_in,
                }.items():
                    df.at[ridx, k] = v
                # stämpla manuella fält (oavsett differens vill du få dagens datum)
                _note_manual_update(df, ridx)
                for f in MANUELL_FALT_FOR_DATUM:
                    _stamp_ts_for_field(df, ridx, f)

        else:
            # uppdatera vald befintlig rad
            tkr = str(bef.get("Ticker","")).upper().strip()
            ridx = int(df.index[df["Ticker"].astype(str).str.upper().str.strip()==tkr][0])
            for k, v in {
                "Bolagsnamn": namn_in, "Valuta": valuta_in, "Sektor": sektor_in,
                "Utestående aktier": utest_in, "Antal aktier": antal_in, "GAV (SEK/aktie)": gav_in,
                "Aktuell kurs": pris_in,
                "P/S": ps_in, "P/S Q1": ps1_in, "P/S Q2": ps2_in, "P/S Q3": ps3_in, "P/S Q4": ps4_in,
                "Omsättning idag": oms_idag_in, "Omsättning nästa år": oms_next_in,
                "Årlig utdelning": utd_in, "CAGR 5 år (%)": cagr_in,
                "Debt/Equity": de_in, "Bruttomarginal (%)": gm_in, "Nettomarginal (%)": nm_in,
                "Kassa (valuta)": cash_in, "FCF TTM (valuta)": fcf_in,
                "Riktkurs idag": rk0_in, "Riktkurs om 1 år": rk1_in, "Riktkurs om 2 år": rk2_in, "Riktkurs om 3 år": rk3_in,
            }.items():
                df.at[ridx, k] = v
            _note_manual_update(df, ridx)
            for f in MANUELL_FALT_FOR_DATUM:
                _stamp_ts_for_field(df, ridx, f)

        # Efter spara — räkna om + spara
        if recompute_cb:
            df = recompute_cb(df)
        if save_cb:
            save_cb(df)
        st.success("Sparat.")

    st.markdown("---")
    st.subheader("Enskild uppdatering (detta bolag)")
    c_run1, c_run2 = st.columns(2)

    # Ticker för runners
    cur_ticker = None
    if not is_new and not bef.empty:
        cur_ticker = str(bef.get("Ticker","")).upper().strip()

    with c_run1:
        if st.button("🔁 Uppdatera kurs"):
            if not cur_ticker:
                st.warning("Ingen ticker vald.")
            elif runner_price is None:
                st.warning("Pris-runner saknas i app.py-kopplingen.")
            else:
                with st.spinner(f"Hämtar kurs för {cur_ticker}..."):
                    try:
                        vals, dbg = runner_price(cur_ticker)
                    except Exception as e:
                        vals, dbg = {}, {"error": str(e)}
                ridxs = df.index[df["Ticker"].astype(str).str.upper().str.strip() == cur_ticker]
                if len(ridxs) == 0:
                    st.error(f"{cur_ticker} hittades inte i tabellen.")
                else:
                    ridx = int(ridxs[0])
                    _apply_vals_to_row(df, ridx, vals, source_label="Enkel pris-uppdatering")
                    if recompute_cb:
                        df = recompute_cb(df)
                    if save_cb:
                        save_cb(df)
                    st.success("Kurs uppdaterad.")
                    if auto_next:
                        st.session_state["edit_idx"] = min(st.session_state["edit_idx"]+1, len(labels)-1)

    with c_run2:
        if st.button("⚙️ Full auto för bolaget"):
            if not cur_ticker:
                st.warning("Ingen ticker vald.")
            elif runner_full is None:
                st.warning("Full-runner saknas i app.py-kopplingen.")
            else:
                with st.spinner(f"Kör full auto för {cur_ticker}..."):
                    try:
                        vals, dbg = runner_full(cur_ticker)
                    except Exception as e:
                        vals, dbg = {}, {"error": str(e)}
                ridxs = df.index[df["Ticker"].astype(str).str.upper().str.strip() == cur_ticker]
                if len(ridxs) == 0:
                    st.error(f"{cur_ticker} hittades inte i tabellen.")
                else:
                    ridx = int(ridxs[0])
                    _apply_vals_to_row(df, ridx, vals, source_label=f"Enkel full auto")
                    if recompute_cb:
                        df = recompute_cb(df)
                    if save_cb:
                        save_cb(df)
                    st.success("Full auto klar.")
                    if auto_next:
                        st.session_state["edit_idx"] = min(st.session_state["edit_idx"]+1, len(labels)-1)

    # ---- Manuell prognoslista (just här enligt din önskan) ----
    st.markdown("---")
    st.subheader("📝 Manuell prognoslista (uppdatera ‘Omsättning idag’ / ‘Omsättning nästa år’)")
    older_days = st.number_input("Flagga äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30)
    need = _build_manual_forecast_list(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Alla ser aktuella ut för prognosfälten.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell uppdatering.")
        st.dataframe(need, use_container_width=True, hide_index=True)

    return df
