# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd

# --- StockApp-moduler ---
from stockapp.config import STANDARD_VALUTAKURSER
from stockapp.storage import hamta_data, spara_data, backup_snapshot_sheet
from stockapp.rates import las_sparade_valutakurser, spara_valutakurser, hamta_valutakurser_auto
from stockapp.utils import ensure_schema, konvertera_typer, uppdatera_berakningar
from stockapp.batch import sidebar_batch_controls
from stockapp.views import (
    kontrollvy,
    analysvy,
    lagg_till_eller_uppdatera,
    visa_investeringsforslag,
    visa_portfolj,
)
from stockapp.sources import run_update_full, update_price_only_yf


st.set_page_config(page_title="📊 Aktieanalys och investeringsförslag", layout="wide")


# -------------------------------------------------------------
# Små hjälpare
# -------------------------------------------------------------
def _recompute(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Säkerställ schema + typer och kör härledda beräkningar (P/S-snitt, riktkurser m.m.).
    Använd denna före visningar/analys/sparning.
    """
    df = ensure_schema(df)
    df = konvertera_typer(df)
    df = uppdatera_berakningar(df, user_rates)
    return df


def _save(df: pd.DataFrame, make_snapshot: bool = False):
    """Spara till Google Sheet, ev. med snapshot före."""
    try:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("✅ Ändringar sparade till Google Sheets.")
    except Exception as e:
        st.sidebar.error(f"❌ Fel vid sparning: {e}")


# -------------------------------------------------------------
# Sidopanel: valutakurser (utan experimental_rerun)
# -------------------------------------------------------------
def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # 1) Ladda sparade eller standard (engångs-init i session_state)
    try:
        saved = las_sparade_valutakurser()
    except Exception:
        saved = {}

    defaults = {
        "USD": float(saved.get("USD", STANDARD_VALUTAKURSER["USD"])),
        "NOK": float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
        "CAD": float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
        "EUR": float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
        "SEK": 1.0,
    }

    # Initiera session state en gång
    for k_sym, key in [("USD","rate_usd_input"),("NOK","rate_nok_input"),
                       ("CAD","rate_cad_input"),("EUR","rate_eur_input")]:
        if key not in st.session_state:
            st.session_state[key] = float(defaults[k_sym])

    # Knappar FÖRE fälten så vi kan uppdatera session_state tryggt
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("🌐 Hämta kurser autom."):
            try:
                auto_rates, misses, provider = hamta_valutakurser_auto()
                # Sätt in i session_state INNAN widgets renderas
                st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
                st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
                st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
                st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
                st.sidebar.success(f"Kurser uppdaterade (källa: {provider}).")
                if misses:
                    st.sidebar.warning("Missar:\n- " + "\n- ".join(misses))
            except Exception as e:
                st.sidebar.error(f"Kunde inte hämta kurser automatiskt: {e}")
    with col_btn2:
        if st.button("💾 Spara kurser"):
            try:
                spara_valutakurser({
                    "USD": st.session_state.rate_usd_input,
                    "NOK": st.session_state.rate_nok_input,
                    "CAD": st.session_state.rate_cad_input,
                    "EUR": st.session_state.rate_eur_input,
                    "SEK": 1.0,
                })
                st.sidebar.success("Valutakurser sparade.")
            except Exception as e:
                st.sidebar.error(f"Fel vid sparning: {e}")

    # Fält (kopplade till session_state-keys)
    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd_input), step=0.01, format="%.4f", key="rate_usd_input")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok_input), step=0.01, format="%.4f", key="rate_nok_input")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad_input), step=0.01, format="%.4f", key="rate_cad_input")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur_input), step=0.01, format="%.4f", key="rate_eur_input")

    # Läs sparade (återställ) – uppdaterar session_state värden, nästa körning speglar i fälten
    if st.sidebar.button("↻ Läs sparade"):
        try:
            sr = las_sparade_valutakurser()
            st.session_state.rate_usd_input = float(sr.get("USD", st.session_state.rate_usd_input))
            st.session_state.rate_nok_input = float(sr.get("NOK", st.session_state.rate_nok_input))
            st.session_state.rate_cad_input = float(sr.get("CAD", st.session_state.rate_cad_input))
            st.session_state.rate_eur_input = float(sr.get("EUR", st.session_state.rate_eur_input))
            st.sidebar.info("Inlästa sparade kurser.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte läsa sparade kurser: {e}")

    return {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}


# -------------------------------------------------------------
# Sidopanel: batch & actions
# -------------------------------------------------------------
def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame | None:
    """
    Visar batchpanelen och returnerar ev. uppdaterad df (annars None).
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Batch & åtgärder")

    # Runner-funktioner (signatur: (df, user_rates, ticker) -> (df_out, info_str))
    def _full_runner(df_in: pd.DataFrame, ur: dict, tkr: str):
        return run_update_full(df_in, ur, tkr)

    def _price_runner(df_in: pd.DataFrame, ur: dict, tkr: str):
        return update_price_only_yf(df_in, ur, tkr)

    # Batch-UI & körning
    updated_df = sidebar_batch_controls(
        df=df,
        user_rates=user_rates,
        save_cb=_save,
        recompute_cb=_recompute,
        runner=_full_runner,
        price_runner=_price_runner,
    )
    return updated_df


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    # 1) Sidopanel: valutakurser
    user_rates = _sidebar_rates()

    # 2) Läs data från Google Sheet
    try:
        df_raw = hamta_data()
    except Exception as e:
        st.error(f"❌ Kunde inte läsa data från Google Sheets: {e}")
        df_raw = pd.DataFrame({})

    # 3) Robust: schema + typer och en minsta tom-df om helt tomt
    if df_raw is None or df_raw.empty:
        df_raw = pd.DataFrame({})
    df = ensure_schema(df_raw)
    df = konvertera_typer(df)

    # 4) Beräkningar (P/S-snitt, riktkurser m.m.) och spara referenser
    df_calc = _recompute(df.copy(), user_rates)
    st.session_state["_df_ref"] = df  # “rå” (utan mellanberäkningar) om du föredrar
    st.session_state["_df_calc"] = df_calc

    # 5) Batch & åtgärder i sidopanelen (kan returnera uppdaterad df)
    maybe_updated = _sidebar_batch_and_actions(df_calc, user_rates)
    if maybe_updated is not None:
        df_calc = _recompute(maybe_updated, user_rates)
        st.session_state["_df_ref"] = ensure_schema(maybe_updated)
        st.session_state["_df_calc"] = df_calc

    # 6) Meny och vyer
    st.sidebar.markdown("---")
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Kontroll", "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"],
        index=0
    )

    if meny == "Kontroll":
        # Skicka beräknad df för smidigare visning
        kontrollvy(df_calc)
    elif meny == "Analys":
        analysvy(df_calc, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df_calc, user_rates)
        if df2 is not None:
            df_calc = _recompute(df2, user_rates)
            st.session_state["_df_ref"] = ensure_schema(df2)
            st.session_state["_df_calc"] = df_calc
            # Spara direkt vid ändringar (utan snapshot som default)
            _save(df_calc, make_snapshot=False)
    elif meny == "Investeringsförslag":
        df_calc = _recompute(df_calc, user_rates)
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        df_calc = _recompute(df_calc, user_rates)
        visa_portfolj(df_calc, user_rates)


if __name__ == "__main__":
    main()
