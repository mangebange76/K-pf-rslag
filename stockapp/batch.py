# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

from .fetchers import auto_fetch_for_ticker
from .calc import apply_auto_updates_to_row, recompute_all

def _pick_order(df: pd.DataFrame, mode: str):
    if mode == "Äldst uppdaterade först (alla fält)":
        # grovt: sortera på äldsta TS över spårade fält
        from .utils import add_oldest_ts_col
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"], ascending=[True, True])
        return list(work["Ticker"].astype(str))
    else:
        return list(df.sort_values(by=["Bolagsnamn","Ticker"])["Ticker"].astype(str))

def sidebar_batch_controls(df: pd.DataFrame, user_rates: dict, save_cb, recompute_cb):
    st.sidebar.subheader("🛠️ Batch-uppdatering")

    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []

    # snabbval: sortering att ladda in i kö
    sort_mode = st.sidebar.selectbox("Ordning för +Lägg till 20", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if st.sidebar.button("📥 Lägg till 20 i kö"):
        order = _pick_order(df, sort_mode)
        already = set(st.session_state.batch_queue)
        to_add = [t for t in order if t and t not in already][:20]
        st.session_state.batch_queue.extend(to_add)
        st.sidebar.success(f"Lagt till: {', '.join(to_add)}")

    add = st.sidebar.text_input("Lägg till ticker (komma-separerat)")
    if st.sidebar.button("➕ Lägg till manuellt"):
        if add.strip():
            to_add = [t.strip().upper() for t in add.replace(";", ",").split(",") if t.strip()]
            st.session_state.batch_queue.extend(to_add)
            st.sidebar.success(f"Lagt till: {', '.join(to_add)}")

    if st.session_state.batch_queue:
        st.sidebar.write("Kö:", ", ".join(st.session_state.batch_queue))

        make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True)
        if st.sidebar.button("🚀 Kör batch (Auto: SEC/Yahoo→Yahoo→FMP)"):
            tickers = list(st.session_state.batch_queue)
            total = len(tickers)
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            change_log = {}
            misses = {}

            # snapshot?
            if make_snapshot and save_cb:
                try:
                    save_cb(df)  # enkla snapshot: Cloud-versionen har ej flik-skapande här; kan utökas.
                    st.sidebar.info("Snapshot sparad (full skrivning).")
                except Exception as e:
                    st.sidebar.warning(f"Kunde inte skapa snapshot: {e}")

            # körning
            for i, t in enumerate(tickers, start=1):
                status.write(f"Kör {i}/{total}: {t}")
                try:
                    vals, debug = auto_fetch_for_ticker(t)
                    # hitta rad
                    mask = (df["Ticker"].astype(str).str.upper() == t.upper())
                    if not mask.any():
                        misses[t] = ["Ticker hittades inte i bladet"]
                    else:
                        ridx = df.index[mask][0]
                        changed = apply_auto_updates_to_row(df, ridx, vals, source="Auto (SEC/Yahoo→Yahoo→FMP)", force_stamp_ts=True)
                        if changed:
                            change_log.setdefault(t, []).extend(list(vals.keys()))
                except Exception as e:
                    misses[t] = [f"error: {e}"]

                if i % 5 == 0 and save_cb:
                    try:
                        save_cb(df)
                    except Exception as e:
                        st.sidebar.warning(f"Delvis skrivning misslyckades vid {i}: {e}")

                prog.progress(i/total)

            # skriv slutresultat
            if save_cb:
                try:
                    save_cb(df)
                    st.sidebar.success("Batch klar – ändringar sparade.")
                except Exception as e:
                    st.sidebar.error(f"Slutsparning misslyckades: {e}")

            # recompute (i minnet)
            df = recompute_cb(df, user_rates) if recompute_cb else df
            st.session_state.batch_queue = []
            with st.sidebar.expander("📒 Körlogg", expanded=False):
                st.write("**Ändringar**", change_log if change_log else "–")
                st.write("**Missar**", misses if misses else "–")

    return df
