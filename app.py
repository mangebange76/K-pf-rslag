# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import time
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd

# ---- Våra moduler -----------------------------------------------------------
from stockapp.config import (
    APP_TITLE,
    STANDARD_VALUTAKURSER,
    TS_FIELDS,
)
from stockapp.utils import ensure_schema, now_stamp
from stockapp.storage import hamta_data, spara_data
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)
from stockapp.update import (
    update_price_for_all,
    update_price_for_ticker,
    update_full_for_ticker,
)
from stockapp.views import (
    kontrollvy,
    analysvy,
    lagg_till_eller_uppdatera,
    visa_portfolj,
)
from stockapp.invest import visa_investeringsforslag


# -----------------------------------------------------------------------------
# Streamlit grund
# -----------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")


# -----------------------------------------------------------------------------
# Små hjälpare lokalt (endast i app)
# -----------------------------------------------------------------------------
def _init_session_defaults():
    ss = st.session_state
    ss.setdefault("_last_log", {})            # senaste körlogg (ändringar/missar)
    ss.setdefault("_batch_cursor", 0)         # pekare för nästa batch-slice
    ss.setdefault("_batch_order_cache", [])   # cachenad ordning (tickers) för batch
    ss.setdefault("_batch_sort_mode", "Äldst först")
    ss.setdefault("_rates_display", None)     # vad vi visar i inputs
    ss.setdefault("_rates_saved_nonce", 0)    # bumpas när vi sparar kurser
    ss.setdefault("_df_ref", None)            # den DF vi jobbar med i sessionen


def _oldest_any_ts(row: pd.Series) -> pd.Timestamp | None:
    dates = []
    for c in row.index:
        if str(c).startswith("TS_"):
            val = str(row.get(c, "")).strip()
            if val:
                try:
                    d = pd.to_datetime(val, errors="coerce")
                    if pd.notna(d):
                        dates.append(d)
                except Exception:
                    pass
    if not dates:
        return None
    return min(dates)


def _sorted_tickers(df: pd.DataFrame, mode: str) -> List[str]:
    """
    Bygger en ordning på tickers för batch-körning.
    - 'Äldst först' sorterar på äldsta TS_* (min-date). Tomma → längst upp.
    - 'A–Ö' sorterar alfabetiskt på Bolagsnamn, sedan Ticker.
    """
    work = df.copy()
    work["_oldest"] = work.apply(_oldest_any_ts, axis=1)
    if mode == "Äldst först":
        # tom/NaT först:
        work["_oldest_fill"] = work["_oldest"].fillna(pd.Timestamp("1900-01-01"))
        work = work.sort_values(by=["_oldest_fill", "Bolagsnamn", "Ticker"], ascending=[True, True, True])
    else:
        work = work.sort_values(by=["Bolagsnamn", "Ticker"], ascending=[True, True])
    return list(work["Ticker"].astype(str))


def _progress_bar(i: int, n: int, label: str = "Körning"):
    pct = (i / max(1, n))
    st.sidebar.progress(pct, text=f"{label}: {i}/{n}")


# -----------------------------------------------------------------------------
# Sidopanel: Valutor & uppdateringsknappar
# -----------------------------------------------------------------------------
def _sidebar_rates() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")

    # 1) Initiera "display"-värden bara en gång
    if st.session_state["_rates_display"] is None:
        try:
            saved = las_sparade_valutakurser()
        except Exception:
            saved = STANDARD_VALUTAKURSER.copy()
        # Spara som floats
        st.session_state["_rates_display"] = {
            "USD": float(saved.get("USD", STANDARD_VALUTAKURSER["USD"])),
            "NOK": float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
            "CAD": float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
            "EUR": float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
            "SEK": 1.0,
        }

    # 2) Rita inputs mot display-state (vi läser av värdena direkt ur widget-resultaten,
    #    vi rör aldrig widget-keys i session_state → undviker StreamlitAPIException)
    disp = st.session_state["_rates_display"]
    colA, colB = st.sidebar.columns(2)
    with colA:
        usd_val = st.number_input("USD → SEK", value=float(disp["USD"]), step=0.01, format="%.4f")
        cad_val = st.number_input("CAD → SEK", value=float(disp["CAD"]), step=0.01, format="%.4f")
        sek_val = st.number_input("SEK → SEK", value=1.0, step=0.01, format="%.2f", disabled=True)
    with colB:
        eur_val = st.number_input("EUR → SEK", value=float(disp["EUR"]), step=0.01, format="%.4f")
        nok_val = st.number_input("NOK → SEK", value=float(disp["NOK"]), step=0.01, format="%.4f")

    # 3) Knapp: Auto-hämtning
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        # Uppdatera display-värdena (inte widget keys)
        st.session_state["_rates_display"].update({
            "USD": float(auto_rates.get("USD", usd_val)),
            "NOK": float(auto_rates.get("NOK", nok_val)),
            "CAD": float(auto_rates.get("CAD", cad_val)),
            "EUR": float(auto_rates.get("EUR", eur_val)),
        })
        st.sidebar.success(f"Valutakurser uppdaterade (källa: {provider}).")
        if misses:
            st.sidebar.warning("Kunde inte hämta för:\n- " + "\n- ".join(misses))

        # Skriv över lokala variabler så retur blir synkad med display
        usd_val = st.session_state["_rates_display"]["USD"]
        nok_val = st.session_state["_rates_display"]["NOK"]
        cad_val = st.session_state["_rates_display"]["CAD"]
        eur_val = st.session_state["_rates_display"]["EUR"]

    # 4) Spara/Läs
    colS1, colS2 = st.sidebar.columns(2)
    with colS1:
        if st.button("💾 Spara kurser"):
            rates_to_save = {
                "USD": float(usd_val),
                "NOK": float(nok_val),
                "CAD": float(cad_val),
                "EUR": float(eur_val),
                "SEK": 1.0,
            }
            try:
                spara_valutakurser(rates_to_save)
                st.session_state["_rates_display"].update(rates_to_save)
                st.sidebar.success("Sparat.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte spara kurser: {e}")

    with colS2:
        if st.button("↻ Läs sparade kurser"):
            try:
                saved = las_sparade_valutakurser()
                st.session_state["_rates_display"].update({
                    "USD": float(saved.get("USD", STANDARD_VALUTAKURSER["USD"])),
                    "NOK": float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
                    "CAD": float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
                    "EUR": float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
                    "SEK": 1.0,
                })
                st.sidebar.success("Inlästa.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte läsa sparade kurser: {e}")

    # 5) Retur som user_rates
    return {
        "USD": float(usd_val),
        "NOK": float(nok_val),
        "CAD": float(cad_val),
        "EUR": float(eur_val),
        "SEK": 1.0,
    }


def _sidebar_updates(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Snabb-uppdatering")

    # Alla kurser
    if st.sidebar.button("📈 Uppdatera kurser (alla)"):
        try:
            # Progress
            tickers = list(df["Ticker"].astype(str))
            n = len(tickers)
            log_changed = {}
            log_miss = {}

            prog = st.sidebar.empty()
            bar = st.sidebar.progress(0.0, text="Startar...")

            df2 = df.copy()
            for i, tkr in enumerate(tickers, start=1):
                try:
                    df2, changed, _ = update_price_for_ticker(df2, tkr)
                    if changed:
                        log_changed[tkr] = changed
                    else:
                        log_miss[tkr] = ["(ingen förändring)"]
                except Exception as e:
                    log_miss[tkr] = [f"error: {e}"]
                # progress
                bar.progress(i / n, text=f"Uppdaterar kurser: {i}/{n}")

            spara_data(df2, do_snapshot=False)
            st.session_state["_df_ref"] = df2
            st.session_state["_last_log"] = {"changed": log_changed, "misses": log_miss}
            st.sidebar.success("Klart – kurser uppdaterade.")
            return df2
        except Exception as e:
            st.sidebar.error(f"Fel vid uppdatering: {e}")

    # Enskild ticker
    tkr = st.sidebar.text_input("Ticker (Yahoo-format) för enskild uppdatering", value="")
    colU1, colU2 = st.sidebar.columns(2)
    if tkr:
        with colU1:
            if st.button("🔹 Endast kurs (ticker)"):
                try:
                    df2, changed, _ = update_price_for_ticker(df.copy(), tkr.strip().upper())
                    spara_data(df2, do_snapshot=False)
                    st.session_state["_df_ref"] = df2
                    st.session_state["_last_log"] = {"changed": {tkr: changed} if changed else {}, "misses": {}}
                    st.sidebar.success(f"Klar ({tkr}).")
                    return df2
                except Exception as e:
                    st.sidebar.error(f"{tkr}: {e}")

        with colU2:
            if st.button("🔶 Full uppdatering (ticker)"):
                try:
                    df2, changed, _ = update_full_for_ticker(df.copy(), tkr.strip().upper())
                    spara_data(df2, do_snapshot=False)
                    st.session_state["_df_ref"] = df2
                    st.session_state["_last_log"] = {"changed": {tkr: changed} if changed else {}, "misses": {}}
                    st.sidebar.success(f"Klar ({tkr}).")
                    return df2
                except Exception as e:
                    st.sidebar.error(f"{tkr}: {e}")

    return df


def _sidebar_batch(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧺 Batch-körning")

    sort_mode = st.sidebar.radio("Ordning", ["Äldst först", "A–Ö"], index=0, horizontal=True)
    size = st.sidebar.number_input("Batch-storlek", min_value=1, max_value=max(1, len(df)), value=min(10, max(1, len(df))), step=1)

    # Bygg/uppdatera ordning vid behov
    if (sort_mode != st.session_state["_batch_sort_mode"]) or not st.session_state["_batch_order_cache"]:
        st.session_state["_batch_sort_mode"] = sort_mode
        st.session_state["_batch_order_cache"] = _sorted_tickers(df, sort_mode)
        st.session_state["_batch_cursor"] = 0

    # Visa info
    total = len(st.session_state["_batch_order_cache"])
    cur = st.session_state["_batch_cursor"]
    st.sidebar.write(f"Vald ordning innehåller **{total}** tickers. Nästa startindex: **{cur}**")

    colB1, colB2, colB3 = st.sidebar.columns(3)
    with colB1:
        if st.button("↺ Återställ kö"):
            st.session_state["_batch_cursor"] = 0
            st.sidebar.info("Batch-kö återställd.")
    with colB2:
        pass
    with colB3:
        pass

    # Kör batch
    if st.sidebar.button(f"▶ Kör batch ({int(size)})"):
        order = st.session_state["_batch_order_cache"]
        start = st.session_state["_batch_cursor"]
        end = min(start + int(size), total)
        subset = order[start:end]
        if not subset:
            st.sidebar.warning("Inget mer att köra i denna ordning. Återställ kö om du vill börja om.")
            return df

        df2 = df.copy()
        log_changed = {}
        log_miss = {}

        n = len(subset)
        bar = st.sidebar.progress(0.0, text=f"Kör batch: 0/{n}")
        for i, tkr in enumerate(subset, start=1):
            try:
                df2, changed, _ = update_full_for_ticker(df2, tkr)
                if changed:
                    log_changed[tkr] = changed
                else:
                    log_miss[tkr] = ["(inga nya fält)"]
            except Exception as e:
                log_miss[tkr] = [f"error: {e}"]
            bar.progress(i / n, text=f"Kör batch: {i}/{n}")

        spara_data(df2, do_snapshot=False)
        st.session_state["_df_ref"] = df2
        st.session_state["_last_log"] = {"changed": log_changed, "misses": log_miss}
        st.session_state["_batch_cursor"] = end
        st.sidebar.success(f"Klar. Körde {n} tickers (index {start}–{end-1}).")
        return df2

    return df


def _sidebar_logs():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📒 Senaste körlogg")
    log = st.session_state.get("_last_log", {})
    if not log:
        st.sidebar.info("Ingen körlogg ännu.")
        return
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.sidebar.markdown("**Ändringar**")
        st.sidebar.json(log.get("changed", {}))
    with col2:
        st.sidebar.markdown("**Missar**")
        st.sidebar.json(log.get("misses", {}))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    _init_session_defaults()
    st.title(APP_TITLE)

    # Läs data (en gång per session) & säkerställ schema
    if st.session_state["_df_ref"] is None:
        df = hamta_data()
        df = ensure_schema(df)
        st.session_state["_df_ref"] = df
    else:
        df = st.session_state["_df_ref"]

    # Sidopanel – valutakurser först (innan andra knappar)
    user_rates = _sidebar_rates()

    # Sidopanel – snabba uppdateringar & batch
    df = _sidebar_updates(df)
    df = _sidebar_batch(df)
    _sidebar_logs()

    # Meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll", "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"], index=0)

    # Vyer
    if meny == "Kontroll":
        kontrollvy(st.session_state["_df_ref"])
    elif meny == "Analys":
        analysvy(st.session_state["_df_ref"], user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates)
        # Spara ev. ändringar från formen
        if df2 is not None and not df2.equals(st.session_state["_df_ref"]):
            spara_data(df2, do_snapshot=False)
            st.session_state["_df_ref"] = df2
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(st.session_state["_df_ref"], user_rates)
    elif meny == "Portfölj":
        visa_portfolj(st.session_state["_df_ref"], user_rates)


if __name__ == "__main__":
    main()
