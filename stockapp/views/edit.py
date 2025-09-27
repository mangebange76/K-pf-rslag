# stockapp/views/edit.py
import streamlit as st
import pandas as pd

_MANUAL_TS_FIELDS = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

def _stamp_ts_for_field(df: pd.DataFrame, row_idx: int, field: str):
    ts_col = f"TS_{field}"
    if ts_col in df.columns:
        df.at[row_idx, ts_col] = pd.Timestamp.now().strftime("%Y-%m-%d")

def _note_manual_update(df: pd.DataFrame, row_idx: int):
    if "Senast manuellt uppdaterad" in df.columns:
        df.at[row_idx, "Senast manuellt uppdaterad"] = pd.Timestamp.now().strftime("%Y-%m-%d")

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates=None) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")
    if df is None:
        df = pd.DataFrame()

    vis_df = df.sort_values(by=[c for c in ["Bolagsnamn","Ticker"] if c in df.columns])
    namn_map = {f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})": r.get('Ticker','') for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map and not df.empty:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            valuta = st.text_input("Valuta (t.ex. USD/EUR/SEK)", value=bef.get("Valuta","") if not bef.empty else "")
            kurs   = st.number_input("Aktuell kurs", value=float(bef.get("Aktuell kurs",0.0)) if not bef.empty else 0.0)

        spar = st.form_submit_button("💾 Spara")

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "Valuta": valuta, "Aktuell kurs": kurs,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        if not df.empty and (df["Ticker"] == ticker).any():
            ridx = df.index[df["Ticker"] == ticker][0]
            before = {f: float(df.at[ridx, f]) if f in df.columns else 0.0 for f in _MANUAL_TS_FIELDS}
            for k,v in ny.items():
                if k in df.columns:
                    df.at[ridx, k] = v
                else:
                    df[k] = 0.0
                    df.at[ridx, k] = v
            after  = {f: float(df.at[ridx, f]) if f in df.columns else 0.0 for f in _MANUAL_TS_FIELDS}
            changed = [k for k in _MANUAL_TS_FIELDS if before.get(k,0.0) != after.get(k,0.0)]
            if changed:
                _note_manual_update(df, ridx)
                for f in changed:
                    _stamp_ts_for_field(df, ridx, f)
        else:
            # Ny rad
            base = {c: (0.0 if not str(c).startswith(("TS_","Senast")) and c not in ["Ticker","Bolagsnamn","Valuta"] else "") for c in set(list(df.columns)+list(ny.keys()))}
            base.update(ny)
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
            ridx = df.index[-1]
            _note_manual_update(df, ridx)
            for f in _MANUAL_TS_FIELDS:
                if float(base.get(f, 0.0) or 0.0) != 0.0:
                    _stamp_ts_for_field(df, ridx, f)

        st.success("Sparat (i minnet). Glöm inte spara till Google Sheet i app.py.")
    return df
