# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import gspread
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google.oauth2.service_account import Credentials

# --- Våra moduler
from stockapp.views import (
    kontrollvy,
    analysvy,
    lagg_till_eller_uppdatera,
    visa_investeringsforslag,
    visa_portfolj,
)
from stockapp.batch import sidebar_batch_controls
from stockapp.sources import _safe_float  # nyttjas för robusthet i vissa hjälpare

# -------------------------------------------------------------------------------------
# Grund-inställningar
# -------------------------------------------------------------------------------------
st.set_page_config(page_title="Aktieanalys & Investeringsförslag", layout="wide")

# Lokal Stockholm-tid om pytz finns (annars systemtid)
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

# -------------------------------------------------------------------------------------
# Google Sheets-koppling
# -------------------------------------------------------------------------------------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def _sheet_main():
    return get_spreadsheet().worksheet(SHEET_NAME)

def _rates_sheet():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        _with_backoff(ws.update, [["Valuta","Kurs"]])
        return ws

def hamta_data() -> pd.DataFrame:
    try:
        ws = _sheet_main()
        data = _with_backoff(ws.get_all_records)
        df = pd.DataFrame(data)
    except Exception:
        df = pd.DataFrame()
    return df

def spara_data(df: pd.DataFrame):
    """Skriv hela DataFrame till huvudbladet."""
    if df is None or df.empty:
        # Skydd mot att råka skriva tomt av misstag
        st.warning("Sparning avbruten: DF tom.")
        return
    ws = _sheet_main()
    _with_backoff(ws.clear)
    _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# -------------------------------------------------------------------------------------
# Valutor
# -------------------------------------------------------------------------------------
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def _las_sparade_valutakurser_cached(nonce: int):
    ws = _rates_sheet()
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        c = str(r.get("Valuta","")).upper().strip()
        v = str(r.get("Kurs","")).replace(",", ".").strip()
        try:
            out[c] = float(v)
        except Exception:
            pass
    return out

def las_sparade_valutakurser() -> Dict[str, float]:
    return _las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: Dict[str,float]):
    ws = _rates_sheet()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurser_auto() -> Tuple[Dict[str,float], list, str]:
    """FMP→Frankfurter→exchangerate.host (fallback). Returnerar (rates, misses, provider)."""
    misses = []
    rates = {}
    provider = None
    # 1) FMP om key finns
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    base = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
    if fmp_key:
        try:
            def _pair(pair):
                url = f"{base}/api/v3/fx/{pair}"
                r = requests.get(url, params={"apikey": fmp_key}, timeout=15)
                if r.status_code != 200:
                    return None, r.status_code
                j = r.json() or {}
                return (float(j.get("price")) if j.get("price") is not None else None), 200
            provider = "FMP"
            for pair in ("USDSEK","NOKSEK","CADSEK","EURSEK"):
                v, sc = _pair(pair)
                if v and v > 0:
                    rates[pair[:3]] = float(v)
                else:
                    misses.append(f"{pair} (HTTP {sc if sc else '??'})")
        except Exception:
            pass
    # 2) Frankfurter
    if len(rates) < 4:
        provider = "Frankfurter"
        for ccy in ("USD","EUR","CAD","NOK"):
            try:
                r = requests.get("https://api.frankfurter.app/latest", params={"from": ccy, "to": "SEK"}, timeout=10)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[ccy] = float(v)
            except Exception:
                pass
    # 3) exchangerate.host
    if len(rates) < 4:
        provider = "exchangerate.host"
        for ccy in ("USD","EUR","CAD","NOK"):
            try:
                r = requests.get("https://api.exchangerate.host/latest", params={"base": ccy, "symbols": "SEK"}, timeout=10)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[ccy] = float(v)
            except Exception:
                pass
    # Fyll luckor från sparat/standard
    saved = las_sparade_valutakurser()
    for ccy in ("USD","EUR","CAD","NOK"):
        if ccy not in rates:
            rates[ccy] = float(saved.get(ccy, STANDARD_VALUTAKURSER.get(ccy, 1.0)))
    rates["SEK"] = 1.0
    return rates, misses, (provider or "okänd")

# -------------------------------------------------------------------------------------
# Schema-säkring (minimi-kolumner så vyer fungerar)
# -------------------------------------------------------------------------------------
MIN_COLS = [
    "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs",
    "Utestående aktier","Market Cap (nu)","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)",
    "Antal aktier","Årlig utdelning","GAV SEK",
    "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    # TS-kolumner
    "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år",
]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame({c: [] for c in MIN_COLS})
    for c in MIN_COLS:
        if c not in df.columns:
            # numeriska default för mätvärden
            if any(x in c.lower() for x in ["p/s","omsättning","riktkurs","aktier","marginal","debt","kassa","fcf","runway","mcap","kurs","andel","utdelning","gav"]):
                df[c] = 0.0
            elif c.startswith("TS_") or c in ["Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sektor","Valuta","Bolagsnamn","Ticker"]:
                df[c] = ""
            else:
                df[c] = ""
    # Duplicats-fix
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

# -------------------------------------------------------------------------------------
# Sidopanel: Valutakurser
# -------------------------------------------------------------------------------------
def _init_rate_state():
    # Första körningen → initiera från sparat eller standard
    if "rate_usd_input" not in st.session_state:
        saved = las_sparade_valutakurser()
        st.session_state.rate_usd_input = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
        st.session_state.rate_nok_input = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        st.session_state.rate_cad_input = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        st.session_state.rate_eur_input = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

def _sidebar_rates() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")
    _init_rate_state()

    # Hantera “Hämta automatiskt” FÖRE widgets renderas
    auto_fetch = st.sidebar.button("🌐 Hämta kurser automatiskt")
    if auto_fetch:
        auto_rates, misses, provider = hamta_valutakurser_auto()
        # Skriv endast till *egna* state-nycklar innan widgetarna skapas
        st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
        st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
        st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
        st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
        st.sidebar.success(f"Valutakurser hämtade (källa: {provider}).")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))

    # Widgets – använder session_state-nycklar som kontrollerat sätts ovan
    usd = st.sidebar.number_input("USD → SEK", key="rate_usd_input", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok_input", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad_input", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur_input", step=0.01, format="%.4f")

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            rates = {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}
            spara_valutakurser(rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear()
            saved = las_sparade_valutakurser()
            st.session_state.rate_usd_input = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
            st.session_state.rate_nok_input = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
            st.session_state.rate_cad_input = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
            st.session_state.rate_eur_input = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
            st.sidebar.info("Inlästa sparade kurser.")

    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}

# -------------------------------------------------------------------------------------
# Sidopanel: Batch-kontroller
# -------------------------------------------------------------------------------------
def _sidebar_batch_and_actions(df: pd.DataFrame):
    # Batchpanel som även sparar till Sheet efter körning
    def _save(d: pd.DataFrame):
        spara_data(d)

    df2 = sidebar_batch_controls(
        df,
        save_cb=_save,
        default_sort="Äldst uppdaterade först (alla fält)",
        default_runner_choice="Full auto",
        default_batch_size=10,
        commit_every=0,   # sätt >0 om du vill del-spara var N:te
    )
    return df2

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------
def main():
    st.title("📊 Aktieanalys & investeringsförslag")

    # Valutakurser i sidopanel
    user_rates = _sidebar_rates()

    # Läs data
    df = hamta_data()
    df = ensure_schema(df)

    # Batch-panel i sidopanelen
    df = _sidebar_batch_and_actions(df)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates, save_cb=lambda d: spara_data(d))
        # skriv endast om ändringar skett (editor returnerar ny DF om förändrat)
        if df2 is not None and df2 is not df and not df2.equals(df):
            spara_data(df2)
            df = df2
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
