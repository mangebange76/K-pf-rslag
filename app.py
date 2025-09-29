# app.py — huvudfil som kopplar ihop modulerna
# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Egna moduler
from stockapp.config import FINAL_COLS, TS_FIELDS, STANDARD_VALUTAKURSER
from stockapp.storage import hamta_data, spara_data
from stockapp.utils import ensure_schema, now_stamp, stamp_fields_ts
from stockapp.rates import las_sparade_valutakurser, spara_valutakurser, hamta_valutakurser_auto, hamta_valutakurs
from stockapp.sources import run_update_price_only, run_update_full
from stockapp.views import kontrollvy, analysvy, lagg_till_eller_uppdatera, visa_investeringsforslag, visa_portfolj

st.set_page_config(page_title="Aktieanalys & investeringsförslag", layout="wide")


# ------------------------------------------------------------
# Beräkningar (P/S-snitt, riktkurser m.m.) — samma anda som förr
# ------------------------------------------------------------
def _recompute_derived(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    - P/S-snitt = snitt av positiva P/S Q1..Q4
    - CAGR clamp ( >100 => 50, <0 => 2 ) för extrapolation av omsättning 2/3 år från 'Omsättning nästa år'
    - Riktkurs idag/1/2/3 år = (Omsättning * P/S-snitt) / Utestående aktier
    """
    if df.empty:
        return df

    df = df.copy()
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_pos = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(float(np.mean(ps_pos)), 2) if ps_pos else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp
        cagr = float(rad.get("CAGR 5 år (%)", 0.0) or 0.0)
        if cagr > 100.0:
            just_cagr = 50.0
        elif cagr < 0.0:
            just_cagr = 2.0
        else:
            just_cagr = cagr
        g = just_cagr / 100.0

        # Omsättning om 2/3 år från "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0) or 0.0)
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0) or 0.0)
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0) or 0.0)

        # Riktkurser (kräver utestående aktier > 0 & P/S-snitt > 0)
        aktier_ut_milj = float(rad.get("Utestående aktier", 0.0) or 0.0)  # miljoner
        aktier_ut = aktier_ut_milj * 1_000_000.0
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0) or 0.0)       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0) or 0.0)   * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])               * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])               * ps_snitt) / aktier_ut, 2)
        else:
            for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i, k] = 0.0
    return df


# ------------------------------------------------------------
# Sidopanel: Valutor + Batch + actions
# ------------------------------------------------------------
def _sidebar_rates() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")

    # Init de "lagrade" input-värdena innan widgets skapas
    saved = las_sparade_valutakurser()
    if "rate_usd_input" not in st.session_state:
        st.session_state.rate_usd_input = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
    if "rate_nok_input" not in st.session_state:
        st.session_state.rate_nok_input = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
    if "rate_cad_input" not in st.session_state:
        st.session_state.rate_cad_input = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
    if "rate_eur_input" not in st.session_state:
        st.session_state.rate_eur_input = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

    # Auto-knapp (sätt state **innan** widgets ritas)
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
        st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
        st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
        st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
        st.session_state["_fx_provider"] = provider
        st.session_state["_fx_misses"] = misses
        st.rerun()

    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd_input), step=0.01, format="%.4f", key="usd_widget")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok_input), step=0.01, format="%.4f", key="nok_widget")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad_input), step=0.01, format="%.4f", key="cad_widget")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur_input), step=0.01, format="%.4f", key="eur_widget")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara kurser"):
            rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}
            spara_valutakurser(rates)
            st.success("Valutakurser sparade.")
    with c2:
        if st.button("↻ Läs sparade kurser"):
            sr = las_sparade_valutakurser()
            st.session_state.rate_usd_input = float(sr.get("USD", st.session_state.rate_usd_input))
            st.session_state.rate_nok_input = float(sr.get("NOK", st.session_state.rate_nok_input))
            st.session_state.rate_cad_input = float(sr.get("CAD", st.session_state.rate_cad_input))
            st.session_state.rate_eur_input = float(sr.get("EUR", st.session_state.rate_eur_input))
            st.rerun()

    if "_fx_provider" in st.session_state:
        st.sidebar.caption(f"Källa: {st.session_state['_fx_provider']}")
        m = st.session_state.get("_fx_misses") or []
        if m:
            st.sidebar.warning("Missade par:\n- " + "\n- ".join(m))

    # Returnera rates för beräkningar
    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}


def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    if sort_mode == "Äldst TS":
        # beräkna äldsta TS bland spårade fält
        def _oldest_any_ts(row: pd.Series):
            dates = []
            for c in TS_FIELDS.values():
                if c in row and str(row[c]).strip():
                    d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                    if pd.notna(d):
                        dates.append(d)
            return min(dates) if dates else pd.NaT
        work = df.copy()
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        work = work.sort_values(by=["_oldest_any_ts","Bolagsnamn"], ascending=[True, True])
        return [t for t in work["Ticker"].astype(str).tolist() if t.strip()]
    # default: A–Ö bolagsnamn
    w = df.sort_values(by=["Bolagsnamn","Ticker"])
    return [t for t in w["Ticker"].astype(str).tolist() if t.strip()]


def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Batch & åtgärder")

    if df.empty:
        st.sidebar.info("Inga bolag i databasen ännu.")
        return df

    sort_mode = st.sidebar.selectbox("Sortering", ["A–Ö","Äldst TS"], index=0)
    batch_size = st.sidebar.number_input("Batch-storlek", min_value=1, max_value=200, value=10, step=1)

    # Pris för alla
    if st.sidebar.button("🔁 Uppdatera **kurs** för ALLA"):
        tickers = [t for t in df["Ticker"].astype(str) if t.strip()]
        n = len(tickers)
        prog = st.sidebar.progress(0.0, text="Startar…")
        changed_total: Dict[str, List[str]] = {}
        for i, tkr in enumerate(tickers, start=1):
            try:
                df, changed, err = run_update_price_only(tkr, df, user_rates)
                if changed:
                    # sätt enkel auto-stämpel
                    ridx = df.index[df["Ticker"] == tkr][0]
                    df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
                    df.at[ridx, "Senast uppdaterad källa"] = "Yahoo pris"
                    changed_total[tkr] = changed
            except Exception as e:
                pass
            prog.progress(min(i/n, 1.0), text=f"Uppdaterar pris {i}/{n}")
        df = _recompute_derived(df, user_rates)
        spara_data(df)
        st.sidebar.success(f"Klar. Ändringar: {len(changed_total)} tickers.")
        return df

    # Full auto (batch)
    if st.sidebar.button("🚀 Kör **full auto** (batch)"):
        order = _pick_order(df, sort_mode)
        to_run = order[: int(batch_size)]
        if not to_run:
            st.sidebar.info("Inget att köra.")
            return df

        n = len(to_run)
        prog = st.sidebar.progress(0.0, text="Startar…")
        changed_total: Dict[str, List[str]] = {}
        for i, tkr in enumerate(to_run, start=1):
            try:
                df, changed, err = run_update_full(tkr, df, user_rates)
                if changed:
                    ridx = df.index[df["Ticker"] == tkr][0]
                    df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
                    df.at[ridx, "Senast uppdaterad källa"] = "Auto (SEC/Yahoo→Yahoo→FMP/Finnhub)"
                    changed_total[tkr] = changed
            except Exception:
                pass
            prog.progress(min(i/n, 1.0), text=f"Kör {i}/{n}")
        df = _recompute_derived(df, user_rates)
        spara_data(df)
        st.sidebar.success(f"Klar. Ändringar: {len(changed_total)} tickers.")
        return df

    return df


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.title("📊 Aktieanalys & investeringsförslag")

    # 1) Valutor i sidopanelen
    user_rates = _sidebar_rates()

    # 2) Läs data
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    df = ensure_schema(df)

    # 3) Batch & åtgärder (kan returnera uppdaterad df)
    df = _sidebar_batch_and_actions(df, user_rates)

    # 4) Meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    # 5) Kör vald vy
    if meny == "Kontroll":
        df_calc = _recompute_derived(df.copy(), user_rates)
        kontrollvy(df_calc)

    elif meny == "Analys":
        df_calc = _recompute_derived(df.copy(), user_rates)
        analysvy(df_calc, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df.copy(), user_rates)
        if df2 is not None and not df2.equals(df):
            # Recompute & spara
            df2 = _recompute_derived(df2, user_rates)
            spara_data(df2)
            st.success("Ändringar sparade.")
            # Uppdatera lokalt df
            df = df2

    elif meny == "Investeringsförslag":
        df_calc = _recompute_derived(df.copy(), user_rates)
        visa_investeringsforslag(df_calc, user_rates)

    elif meny == "Portfölj":
        df_calc = _recompute_derived(df.copy(), user_rates)
        visa_portfolj(df_calc, user_rates)


if __name__ == "__main__":
    main()
