# stockapp/views/analysis.py
import streamlit as st
import pandas as pd

# Vilka kolumner vi försöker visa om de finns
_SHOW_COLS = [
    "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
    "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
]

def analysvy(df: pd.DataFrame, user_rates=None):
    st.header("📈 Analys")
    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = df.copy()
    vis_df = vis_df.sort_values(by=[c for c in ["Bolagsnamn","Ticker"] if c in vis_df.columns])
    labels = [f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})" for _, r in vis_df.iterrows()]
    if not labels:
        st.info("Inga bolag i databasen ännu.")
        return

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    st.session_state.analys_idx = st.number_input(
        "Visa bolag #", min_value=0, max_value=max(0, len(labels)-1),
        value=st.session_state.analys_idx, step=1
    )
    st.selectbox("Eller välj i lista", labels, index=st.session_state.analys_idx, key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(labels)-1, st.session_state.analys_idx+1)

    st.write(f"Post {st.session_state.analys_idx+1}/{len(labels)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    cols = [c for c in _SHOW_COLS if c in vis_df.columns]
    # Lägg till alla TS_-kolumner om de finns
    cols += [c for c in vis_df.columns if str(c).startswith("TS_")]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)
