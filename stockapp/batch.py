# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

def sidebar_batch_controls(df: pd.DataFrame, user_rates: dict, save_cb, recompute_cb):
    st.sidebar.subheader("🛠️ Batch-uppdatering")

    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []

    add = st.sidebar.text_input("Lägg till ticker (komma-separerat)")
    if st.sidebar.button("➕ Lägg till i kö"):
        if add.strip():
            to_add = [t.strip().upper() for t in add.replace(";", ",").split(",") if t.strip()]
            st.session_state.batch_queue.extend(to_add)
            st.sidebar.success(f"Lagt till: {', '.join(to_add)}")

    if st.session_state.batch_queue:
        st.sidebar.write("Kö:", ", ".join(st.session_state.batch_queue))

        if st.sidebar.button("🚀 Kör batch"):
            total = len(st.session_state.batch_queue)
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            # Här kopplar vi på riktig logik i Del 2 (fetchers + apply)
            for i, t in enumerate(list(st.session_state.batch_queue), start=1):
                status.write(f"Kör {i}/{total}: {t}")
                prog.progress(i/total)
            st.session_state.batch_queue = []
            st.sidebar.success("Batch (mock) klar.")

    return df
