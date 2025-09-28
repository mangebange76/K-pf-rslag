# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd

# ---- Viktigt: absoluta importer (app.py ligger utanför paketet) -------------
from stockapp.config import (
    FINAL_COLS, TS_FIELDS, STANDARD_VALUTAKURSER,
    SHEET_NAME, RATES_SHEET_NAME,
)
from stockapp.storage import hamta_data, spara_data
from stockapp.utils import (
    säkerställ_kolumner, migrera_gamla_riktkurskolumner, konvertera_typer,
    add_oldest_ts_col, now_stamp, now_dt,
)
from stockapp.rates import (
    las_sparade_valutakurser, spara_valutakurser, hamta_valutakurser_auto,
)
from stockapp.batch import sidebar_batch_controls
from stockapp.sources import run_update_price_only, run_update_full
from stockapp.views import (
    kontrollvy, analysvy, lagg_till_eller_uppdatera,
    visa_investeringsforslag, visa_portfolj,
)

# -----------------------------------------------------------------------------


st.set_page_config(page_title="📊 Aktieanalys och investeringsförslag", layout="wide")


# ============= Små hjälp-funktioner i app.py =================================

def _ensure_state_defaults():
    """Initiera state-nycklar som UI:n förlitar sig på."""
    ss = st.session_state
    # Dataframe-referens (håll en "master" i state)
    if "_df_ref" not in ss:
        try:
            df = hamta_data()
        except Exception:
            df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        df = migrera_gamla_riktkurskolumner(df)
        df = konvertera_typer(df)
        ss["_df_ref"] = df

    # Valutarates
    if "rates_inited" not in ss:
        saved = {}
        try:
            saved = las_sparade_valutakurser()
        except Exception:
            saved = {}
        ss.rate_usd_input = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
        ss.rate_nok_input = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        ss.rate_cad_input = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        ss.rate_eur_input = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
        ss.rates_inited = True

    # Batch-runner default
    if "runner_choice" not in ss:
        ss.runner_choice = "Full auto"

    # Snabb-uppdateringsticker
    if "quick_ticker" not in ss:
        ss.quick_ticker = ""


def _build_user_rates_from_state() -> dict:
    """Bygg user_rates-dict av sidopanelens input-värden (utan att skriva tillbaka till state)."""
    return {
        "USD": float(st.session_state.rate_usd_input),
        "NOK": float(st.session_state.rate_nok_input),
        "CAD": float(st.session_state.rate_cad_input),
        "EUR": float(st.session_state.rate_eur_input),
        "SEK": 1.0,
    }


def _save_df(df: pd.DataFrame, do_snapshot: bool = False):
    """Säkert spara df via storage (hela bladet)."""
    spara_data(df, do_snapshot=do_snapshot)
    # Uppdatera master-referensen
    st.session_state["_df_ref"] = df


def _normalize_runner_result_app(res, df_fallback: pd.DataFrame):
    """Normalisera runner-retur i appens snabbknappar."""
    df_out = df_fallback
    changed = []
    msg = ""
    if isinstance(res, tuple):
        if len(res) == 3:
            df_out, changed, msg = res
        elif len(res) == 2:
            df_out, msg = res
        elif len(res) == 1:
            df_out = res[0]
    elif isinstance(res, pd.DataFrame):
        df_out = res
    else:
        msg = str(res)
    return df_out, changed, msg


# ============= Sidopanel: Valutor + Batch + Snabb-uppdatering ================

def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # Actions FÖRE widgets: auto-hämtning
    auto_click = st.sidebar.button("🌐 Hämta kurser automatiskt")
    if auto_click:
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
        st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
        st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
        st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
        st.session_state["_rates_provider"] = provider
        st.session_state["_rates_misses"] = misses
        st.rerun()

    st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd_input), step=0.01, format="%.4f", key="rate_usd_input")
    st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok_input), step=0.01, format="%.4f", key="rate_nok_input")
    st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad_input), step=0.01, format="%.4f", key="rate_cad_input")
    st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur_input), step=0.01, format="%.4f", key="rate_eur_input")

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            to_save = {
                "USD": float(st.session_state.rate_usd_input),
                "NOK": float(st.session_state.rate_nok_input),
                "CAD": float(st.session_state.rate_cad_input),
                "EUR": float(st.session_state.rate_eur_input),
                "SEK": 1.0
            }
            spara_valutakurser(to_save)
            st.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↺ Läs sparade kurser"):
            saved = las_sparade_valutakurser()
            st.session_state.rate_usd_input = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
            st.session_state.rate_nok_input = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
            st.session_state.rate_cad_input = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
            st.session_state.rate_eur_input = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
            st.info("Kurser återställda från lagrat värde.")
            st.rerun()

    if "_rates_provider" in st.session_state:
        prov = st.session_state.get("_rates_provider")
        misses = st.session_state.get("_rates_misses") or []
        st.sidebar.caption(f"Källa: {prov}" + ("" if not misses else f" • Missar: {', '.join(misses)}"))

    return _build_user_rates_from_state()


def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Snabb-uppdatering (en ticker)")

    st.sidebar.text_input("Ticker (Yahoo-format)", key="quick_ticker")

    # Välj runner för knappar & batch
    st.sidebar.radio("Val av uppdaterare", ["Full auto", "Endast kurs"], index=0, key="runner_choice")
    runner = run_update_full if st.session_state.runner_choice == "Full auto" else run_update_price_only

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("⚡ Uppdatera vald ticker"):
            tkr = (st.session_state.quick_ticker or "").strip().upper()
            if not tkr:
                st.sidebar.warning("Ange en ticker.")
            else:
                res = runner(df.copy(), user_rates, tkr)
                df2, changed, msg = _normalize_runner_result_app(res, df.copy())
                _save_df(df2, do_snapshot=False)
                if changed:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.info(msg)

    with c2:
        if st.button("⟳ Endast kurs (fallback)"):
            tkr = (st.session_state.quick_ticker or "").strip().upper()
            if not tkr:
                st.sidebar.warning("Ange en ticker.")
            else:
                res = run_update_price_only(df.copy(), user_rates, tkr)
                df2, changed, msg = _normalize_runner_result_app(res, df.copy())
                _save_df(df2, do_snapshot=False)
                if changed:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.info(msg)

    # Batchkontroller
    st.sidebar.markdown("---")
    st.sidebar.subheader("📦 Batch-körning")

    df_out = sidebar_batch_controls(
        df=df,
        user_rates=user_rates,
        save_cb=lambda d, snap: _save_df(d, do_snapshot=snap),
        recompute_cb=None,      # runners beräknar P/S-snitt själva; vill du räkna centralt, skicka funktion
        runner=runner
    )
    if df_out is not None:
        _save_df(df_out, do_snapshot=False)
        return df_out
    return df


# =============================== MAIN ========================================

def main():
    _ensure_state_defaults()

    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs master-DF från state
    df = st.session_state["_df_ref"]

    # Sidopanel: valutakurser (med auto/spara) → ger user_rates-dict
    user_rates = _sidebar_rates()

    # Sidopanel: snabb-uppdatering + batchpanel
    df_after_sidebar = _sidebar_batch_and_actions(df, user_rates)
    if df_after_sidebar is not None:
        df = df_after_sidebar  # för visning i denna körning

    # Meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy",
                            ["Kontroll", "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"], index=0)

    # Visa vyer
    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        if df2 is not None and not df2.equals(df):
            _save_df(df2, do_snapshot=False)
            st.success("Ändringar sparade.")
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df, user_rates)

    st.caption(f"Datablad: {SHEET_NAME} • Senast läst: {now_stamp()}")


if __name__ == "__main__":
    main()
