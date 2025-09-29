# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------
# Importera modulära delar
# -------------------------------------------------------
from stockapp.storage import hamta_data, spara_data
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)
from stockapp.utils import (
    ensure_schema,
)

# Invest-vyn (rankning/poäng + expander med nyckeltal)
from stockapp.invest import visa_investeringsforslag

# Dessa vyer är valfria – visas bara om moduler finns
try:
    from stockapp.views import (
        kontrollvy,
        analysvy,
        lagg_till_eller_uppdatera,
        visa_portfolj,
    )
    _HAVE_VIEWS = True
except Exception:
    _HAVE_VIEWS = False

# Valfria uppdaterings- & batch-funktioner
_HAVE_UPDATE = False
try:
    from stockapp.update import run_update_price_only, run_update_full
    _HAVE_UPDATE = True
except Exception:
    _HAVE_UPDATE = False

_HAVE_BATCH = False
try:
    from stockapp.batch import sidebar_batch_controls
    _HAVE_BATCH = True
except Exception:
    _HAVE_BATCH = False

# -------------------------------------------------------
# Streamlit bas
# -------------------------------------------------------
st.set_page_config(page_title="Aktieanalys & Investeringsförslag", layout="wide")

# Kompatibilitet: om någon modul använder experimental_rerun men din Streamlit har st.rerun
if not hasattr(st, "experimental_rerun") and hasattr(st, "rerun"):
    setattr(st, "experimental_rerun", st.rerun)

# -------------------------------------------------------
# Sidopanel – Valutakurser (robust utan förbjudna writes)
# -------------------------------------------------------
def _sidebar_rates() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")

    # 1) Läs sparade kurser
    saved = las_sparade_valutakurser()  # dict {"USD":..., "EUR":..., "CAD":..., "NOK":..., "SEK":1.0}

    # 2) Init defaults EN gång (innan widgets renderas)
    defaults = {
        "form_rate_usd": float(saved.get("USD", 10.0)),
        "form_rate_eur": float(saved.get("EUR", 11.0)),
        "form_rate_cad": float(saved.get("CAD", 7.0)),
        "form_rate_nok": float(saved.get("NOK", 1.0)),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # 3) Auto-hämtning – skriv till session_state INNAN form byggs
    auto_col1, auto_col2 = st.sidebar.columns([1, 1])
    with auto_col1:
        do_auto = st.button("🌐 Hämta automatiskt")
    with auto_col2:
        do_reload = st.button("↻ Läs sparade")

    if do_auto:
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.sidebar.success(f"Hämtade från {provider}.")
        if misses:
            st.sidebar.warning("Kunde inte hämta för:\n- " + "\n- ".join(misses))
        st.session_state["form_rate_usd"] = float(auto_rates.get("USD", st.session_state["form_rate_usd"]))
        st.session_state["form_rate_eur"] = float(auto_rates.get("EUR", st.session_state["form_rate_eur"]))
        st.session_state["form_rate_cad"] = float(auto_rates.get("CAD", st.session_state["form_rate_cad"]))
        st.session_state["form_rate_nok"] = float(auto_rates.get("NOK", st.session_state["form_rate_nok"]))

    if do_reload:
        sv = las_sparade_valutakurser()
        st.session_state["form_rate_usd"] = float(sv.get("USD", st.session_state["form_rate_usd"]))
        st.session_state["form_rate_eur"] = float(sv.get("EUR", st.session_state["form_rate_eur"]))
        st.session_state["form_rate_cad"] = float(sv.get("CAD", st.session_state["form_rate_cad"]))
        st.session_state["form_rate_nok"] = float(sv.get("NOK", st.session_state["form_rate_nok"]))
        st.sidebar.info("Sparade kurser inlästa.")

    # 4) Form (vi ändrar inte keys efter render)
    with st.sidebar.form("rates_form", clear_on_submit=False):
        usd = st.number_input("USD → SEK", key="form_rate_usd", step=0.01, format="%.4f")
        eur = st.number_input("EUR → SEK", key="form_rate_eur", step=0.01, format="%.4f")
        cad = st.number_input("CAD → SEK", key="form_rate_cad", step=0.01, format="%.4f")
        nok = st.number_input("NOK → SEK", key="form_rate_nok", step=0.01, format="%.4f")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            save_click = st.form_submit_button("💾 Spara")
        with col_s2:
            cancel_click = st.form_submit_button("Ångra ändringar")

    if save_click:
        try:
            to_save = {"USD": float(usd), "EUR": float(eur), "CAD": float(cad), "NOK": float(nok), "SEK": 1.0}
            spara_valutakurser(to_save)
            st.sidebar.success("Valutakurser sparade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")

    if cancel_click:
        sv = las_sparade_valutakurser()
        st.session_state["form_rate_usd"] = float(sv.get("USD", st.session_state["form_rate_usd"]))
        st.session_state["form_rate_eur"] = float(sv.get("EUR", st.session_state["form_rate_eur"]))
        st.session_state["form_rate_cad"] = float(sv.get("CAD", st.session_state["form_rate_cad"]))
        st.session_state["form_rate_nok"] = float(sv.get("NOK", st.session_state["form_rate_nok"]))
        st.sidebar.info("Återställde till sparade kurser.")

    return {
        "USD": float(st.session_state["form_rate_usd"]),
        "EUR": float(st.session_state["form_rate_eur"]),
        "CAD": float(st.session_state["form_rate_cad"]),
        "NOK": float(st.session_state["form_rate_nok"]),
        "SEK": 1.0,
    }

# -------------------------------------------------------
# Sidopanel – ⚙️ Uppdatera data (ticker, kurs, full auto & ev. batch)
# -------------------------------------------------------
def _sidebar_updates(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Uppdatera data")

    if df is None or df.empty or "Ticker" not in df.columns:
        st.sidebar.info("Ingen data inläst ännu.")
        return df

    tickers = sorted([str(t).upper() for t in df["Ticker"].dropna().astype(str).unique() if str(t).strip()])
    if "upd_selected_ticker" not in st.session_state:
        st.session_state["upd_selected_ticker"] = tickers[0] if tickers else ""

    tkr = st.sidebar.selectbox("Välj ticker", tickers, index=max(0, tickers.index(st.session_state["upd_selected_ticker"])) if tickers else 0, key="upd_selected_ticker")

    col_u1, col_u2 = st.sidebar.columns(2)
    do_price = col_u1.button("📉 Uppdatera kurs (snabb)")
    do_full  = col_u2.button("🔄 Full auto (endast denna)")

    df_out = df

    if do_price:
        if not _HAVE_UPDATE:
            st.sidebar.warning("Modulen 'stockapp.update' saknas – kan inte uppdatera kurs.")
        else:
            try:
                # Vi tillåter både (df, user_rates, ticker) och (df, ticker) och (ticker, df)
                changed = False
                try:
                    df2, msg = run_update_price_only(df.copy(), user_rates, tkr)
                    changed = True
                except TypeError:
                    try:
                        df2, msg = run_update_price_only(df.copy(), tkr)
                        changed = True
                    except TypeError:
                        df2, msg = run_update_price_only(tkr, df.copy())
                        changed = True

                if changed and isinstance(df2, pd.DataFrame):
                    try:
                        spara_data(df2)
                        df_out = df2
                        st.sidebar.success(f"Kurs uppdaterad för {tkr}.")
                        if msg:
                            st.sidebar.caption(str(msg))
                    except Exception as e:
                        st.sidebar.error(f"Kunde inte spara uppdaterad kurs: {e}")
                else:
                    st.sidebar.info("Ingen förändring upptäcktes.")
            except Exception as e:
                st.sidebar.error(f"Fel vid kursuppdatering: {e}")

    if do_full:
        if not _HAVE_UPDATE:
            st.sidebar.warning("Modulen 'stockapp.update' saknas – kan inte göra full auto-uppdatering.")
        else:
            try:
                changed = False
                # Tillåt olika signaturer beroende på modulens version
                try:
                    df2, log, note = run_update_full(df.copy(), user_rates, tkr)
                    changed = True
                    extra = note
                except TypeError:
                    try:
                        df2, log, note = run_update_full(df.copy(), tkr, user_rates)
                        changed = True
                        extra = note
                    except TypeError:
                        df2, log = run_update_full(df.copy(), tkr)
                        changed = True
                        extra = None

                if changed and isinstance(df2, pd.DataFrame):
                    try:
                        spara_data(df2)
                        df_out = df2
                        st.sidebar.success(f"Full auto uppdaterad: {tkr}")
                        if extra:
                            st.sidebar.caption(str(extra))
                        if log:
                            st.sidebar.caption("Logg (kort):")
                            st.sidebar.json(log, expanded=False)
                    except Exception as e:
                        st.sidebar.error(f"Kunde inte spara: {e}")
                else:
                    st.sidebar.info("Ingen förändring upptäcktes.")
            except Exception as e:
                st.sidebar.error(f"Fel vid full uppdatering: {e}")

    # Batch-panel om modul finns
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧰 Batch")
    if _HAVE_BATCH:
        try:
            # sidebar_batch_controls förväntas själv hantera UI och körning.
            # Den kan ev. returnera nytt df om något ändrats, annars None.
            df2 = sidebar_batch_controls(df_out, user_rates)
            if isinstance(df2, pd.DataFrame) and not df2.equals(df_out):
                try:
                    spara_data(df2)
                    st.sidebar.success("Batch: ändringar sparade.")
                    df_out = df2
                except Exception as e:
                    st.sidebar.error(f"Batch: kunde inte spara: {e}")
        except Exception as e:
            st.sidebar.error(f"Batch-panel fel: {e}")
    else:
        st.sidebar.info("Batch-modul saknas (stockapp.batch).")

    return df_out

# -------------------------------------------------------
# Huvudkörning
# -------------------------------------------------------
def main():
    st.title("📊 Aktieanalys & Investeringsförslag")

    # Ladda data
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        df = pd.DataFrame()

    # Säkerställ schema
    df = ensure_schema(df)

    # Sidopanel: valutakurser
    user_rates = _sidebar_rates()

    # Sidopanel: uppdateringsverktyg (kan returnera nytt df)
    df = _sidebar_updates(df, user_rates)

    # Behåll referens
    st.session_state["_df_ref"] = df

    # Meny
    if _HAVE_VIEWS:
        val = st.sidebar.radio(
            "📌 Välj vy",
            ["Investeringsförslag", "Kontroll", "Analys", "Lägg till / uppdatera bolag", "Portfölj"],
            index=0,
        )
    else:
        val = st.sidebar.radio(
            "📌 Välj vy",
            ["Investeringsförslag"],
            index=0,
        )

    # Kör vald vy
    if val == "Investeringsförslag":
        # default_style: "growth" eller "income" beroende på vad du vill ha som standard
        visa_investeringsforslag(st.session_state["_df_ref"], user_rates, default_style="growth")

    elif val == "Kontroll" and _HAVE_VIEWS:
        kontrollvy(st.session_state["_df_ref"])

    elif val == "Analys" and _HAVE_VIEWS:
        analysvy(st.session_state["_df_ref"], user_rates)

    elif val == "Lägg till / uppdatera bolag" and _HAVE_VIEWS:
        df2 = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates)
        if isinstance(df2, pd.DataFrame) and not df2.equals(st.session_state["_df_ref"]):
            try:
                spara_data(df2)
                st.session_state["_df_ref"] = df2
                st.success("Ändringar sparade.")
            except Exception as e:
                st.error(f"Kunde inte spara ändringar: {e}")

    elif val == "Portfölj" and _HAVE_VIEWS:
        visa_portfolj(st.session_state["_df_ref"], user_rates)

if __name__ == "__main__":
    main()
