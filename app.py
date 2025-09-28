# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict

# --- Lokal Stockholm-tid om pytz finns (annars systemtid) ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
    def now_dt():
        return datetime.now(TZ_STHLM)
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")
    def now_dt():
        return datetime.now()

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ========== Importera modul-funktioner ==========
# Valutapanel (säker state-hantering)
from stockapp.rates import sidebar_rates

# I/O mot Google Sheets
from stockapp.storage import hamta_data, spara_data

# Beräkningar & schemastöd
from stockapp.utils import (
    ensure_schema,
    migrera_gamla_riktkurskolumner,
    konvertera_typer,
    uppdatera_berakningar,
)

# Vyer
from stockapp.views import (
    kontrollvy,
    analysvy,
    lagg_till_eller_uppdatera,
    visa_investeringsforslag,
    visa_portfolj,
)

# Batchpanel (köläge, 1/X-uppdateringar, loggar)
from stockapp.batch import sidebar_batch_controls

# Datakällor/runners (full uppdatering respektive endast pris)
from stockapp.sources import run_update_full, run_update_price_only


# ======= Liten helper så övrig kod kan fortsätta anropa växelkursen på samma sätt =======
def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    """Enkel helper: returnerar växelkursen för 'valuta' till SEK."""
    if not valuta:
        return 1.0
    try:
        return float(user_rates.get(str(valuta).upper(), 1.0))
    except Exception:
        return 1.0


# ======= Callbacks som batchpanelen kan använda =======
def _save(df: pd.DataFrame):
    """Skriv hela DataFrame till Google Sheets (utan snapshot här – styrs i batchpanelen)."""
    spara_data(df)


def _recompute(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """Kör om alla härledda kolumner."""
    return uppdatera_berakningar(df, user_rates)


# ============================== MAIN ==============================

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Sidopanel: valutakurser (ny robust modul) ---
    user_rates = sidebar_rates()  # {"USD":..., "EUR":..., "NOK":..., "CAD":..., "SEK": 1.0}

    # --- Sidopanel: utility-knappar för cache & laddning ---
    st.sidebar.markdown("---")
    col_reload, col_clear = st.sidebar.columns(2)
    with col_reload:
        if st.button("↻ Läs om data från Google Sheets", key="btn_reload_sheet"):
            st.cache_data.clear()
    with col_clear:
        if st.button("🧹 Töm cache (lokalt)", key="btn_clear_cache"):
            st.cache_data.clear()
            st.sidebar.info("Cache rensad.")

    # --- Läs data från Google Sheets ---
    #    (Modulens hamta_data tar hand om kopplingen; vi säkerställer schema & typer här.)
    df = hamta_data()
    df = ensure_schema(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Behåll en referens i session_state om batchpanelen vill jobba mot samma DF-objekt
    st.session_state["_df_ref"] = df

    # --- Sidopanel: batch-körning (köläge, 1/X text, loggar)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Uppdateringar")
    # - runner för FULL uppdatering (allt för en ticker)
    st.session_state.setdefault("runner_full", run_update_full)
    # - runner för ENDAST PRIS (snabbknapp)
    st.session_state.setdefault("runner_price", run_update_price_only)

    # Visa batchkontrollerna; funktionen kan returnera samma df eller ett uppdaterat df efter körning.
    # Den här panelen erbjuder:
    # - Skapa batchkö (ex. 10–50 st, A–Ö eller äldst-uppdaterade först)
    # - Kör batch (med progressbar och “i/X” text)
    # - Loggar: changed/misses och senaste körning
    df = sidebar_batch_controls(
        df,
        user_rates,
        save_cb=_save,
        recompute_cb=_recompute,
        runner=st.session_state["runner_full"],
        price_runner=st.session_state["runner_price"],
    )

    # --- Router för vy-val ---
    st.sidebar.markdown("---")
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Kontroll", "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"],
        index=0,
        key="main_view_choice",
        horizontal=False
    )

    # Säkerställ att vi räknar om härledda kolumner när det behövs
    df_calc = uppdatera_berakningar(df.copy(), user_rates)

    if meny == "Kontroll":
        kontrollvy(df_calc)
    elif meny == "Analys":
        analysvy(df_calc, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        # Den här vyn hanterar även "Manuell prognoslista" enligt specifikationen.
        df2 = lagg_till_eller_uppdatera(df_calc, user_rates)
        # Spara om vyn har ändrat något (vyn returnerar ev. uppdaterad df)
        if df2 is not None and not df2.equals(df):
            spara_data(df2)
            df = df2
            st.success("Ändringar sparade.")
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df_calc, user_rates)

    # --- Visa senaste batch-logg längst ned (om satt av batchpanelen) ---
    if "last_batch_log" in st.session_state and st.session_state["last_batch_log"]:
        st.divider()
        with st.expander("📒 Senaste batch-logg", expanded=False):
            log = st.session_state["last_batch_log"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Ändringar (ticker → fält)**")
                st.json(log.get("changed", {}))
            with col2:
                st.markdown("**Missar (ticker → skäl)**")
                st.json(log.get("misses", {}))
            if "queue_info" in log:
                st.markdown("**Kö-information**")
                st.json(log["queue_info"])


if __name__ == "__main__":
    main()
