# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials

# ---- Våra moduler
from stockapp.batch import sidebar_batch_controls
from stockapp.views import kontrollvy, analysvy, lagg_till_eller_uppdatera, visa_investeringsforslag, visa_portfolj
from stockapp.utils import ensure_ticker_col, find_duplicate_tickers

# -----------------------------------------------------------------------------
# Bas: tidszon (Stockholm), sidkonfig
# -----------------------------------------------------------------------------
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp(): return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp(): return datetime.now().strftime("%Y-%m-%d")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# -----------------------------------------------------------------------------
# Google Sheets-koppling
# -----------------------------------------------------------------------------
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
        if d: time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def _spreadsheet():
    return client.open_by_url(SHEET_URL)

def _sheet_main():
    return _spreadsheet().worksheet(SHEET_NAME)

def _sheet_rates():
    ss = _spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        _with_backoff(ws.update, [["Valuta","Kurs"]])
        return ws

def hamta_data() -> pd.DataFrame:
    ws = _sheet_main()
    rows = _with_backoff(ws.get_all_records)
    return pd.DataFrame(rows)

def spara_data(df: pd.DataFrame):
    """
    Säkrare skrivning: vägra spara om df är tom eller innehåller dubbletter.
    """
    if df is None or df.empty:
        st.warning("Sparning avbruten: DataFrame är tom.")
        return
    dups = find_duplicate_tickers(df)
    if not dups.empty:
        st.error("Sparning avbruten: Dubblett-tickers hittades. Rensa dubbletter först.")
        return
    ws = _sheet_main()
    _with_backoff(ws.clear)
    _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# -----------------------------------------------------------------------------
# Schema & defaultkolumner (minsta uppsättning som appen använder)
# -----------------------------------------------------------------------------
FINAL_COLS = [
    # Grund
    "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","Utestående aktier","Market Cap (nu)",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Årlig utdelning","GAV SEK",
    # Analys/nyckeltal (kan vara tomma om du inte hämtar dem ännu)
    "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)",
    # Meta
    "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    "Senast auto-försök","Senast auto-status",
    # TS-kolumner
    "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
    "TS_Omsättning idag","TS_Omsättning nästa år",
]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # lägg in saknade kolumner
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","market","marginal","debt","kassa","fcf","runway","gav"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""
            else:
                df[kol] = ""
    # typer
    num_cols = [
        "Aktuell kurs","Utestående aktier","Market Cap (nu)",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Antal aktier","Årlig utdelning","GAV SEK",
        "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Senast auto-försök","Senast auto-status"]:
        df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)
    return df

# -----------------------------------------------------------------------------
# Valutakurser
# -----------------------------------------------------------------------------
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    ws = _sheet_rates()
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        val = str(r.get("Kurs","")).replace(",",".").strip()
        try:
            out[cur] = float(val)
        except:
            pass
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = _sheet_rates()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurser_auto():
    """
    Enkel automatisk hämtning: Frankfurter -> exchangerate.host
    Returnerar (rates, misses, provider)
    """
    misses, rates, provider = [], {}, None
    # Frankfurter
    provider = "Frankfurter"
    for base_ccy in ("USD","EUR","CAD","NOK"):
        try:
            r = requests.get("https://api.frankfurter.app/latest",
                             params={"from": base_ccy, "to": "SEK"}, timeout=12)
            if r.status_code == 200:
                v = (r.json() or {}).get("rates", {}).get("SEK")
                if v:
                    rates[base_ccy] = float(v)
        except Exception:
            pass
    # fallback
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base_ccy in ("USD","EUR","CAD","NOK"):
            try:
                r = requests.get("https://api.exchangerate.host/latest",
                                 params={"base": base_ccy, "symbols": "SEK"}, timeout=12)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass
    # komplettera
    saved = las_sparade_valutakurser()
    for base_ccy in ("USD","EUR","CAD","NOK"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))
    return rates, misses, (provider or "okänd")

# -----------------------------
# Sidopanel: valutakurser + batch
# -----------------------------

def _sidebar_rates() -> Dict[str, float]:
    """
    Stabil sidopanel för valutakurser utan experimental_rerun och utan att
    skriva till widget-keys efter att de skapats.
    """
    st.sidebar.header("💱 Valutakurser → SEK")

    # Initiera state en gång
    for k, v in {
        "rate_usd_input": STANDARD_VALUTAKURSER["USD"],
        "rate_nok_input": STANDARD_VALUTAKURSER["NOK"],
        "rate_cad_input": STANDARD_VALUTAKURSER["CAD"],
        "rate_eur_input": STANDARD_VALUTAKURSER["EUR"],
        "rates_reload": 0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Om vi har en "apply-buffer" från auto-hämtning så skriv in i state
    if st.session_state.get("_apply_rates_flag"):
        buf = st.session_state.get("_apply_rates_buffer", {})
        for ckey, skey in [("USD","rate_usd_input"), ("NOK","rate_nok_input"),
                           ("CAD","rate_cad_input"), ("EUR","rate_eur_input")]:
            if ckey in buf:
                try:
                    st.session_state[skey] = float(buf[ckey])
                except Exception:
                    pass
        st.session_state["_apply_rates_flag"] = False
        st.session_state["_apply_rates_buffer"] = {}

    # Läs sparade (för info)
    saved = las_sparade_valutakurser()
    with st.sidebar.expander("📦 Sparade kurser (Sheet)", expanded=False):
        if saved:
            st.write(saved)
        else:
            st.caption("Inget sparat hittades, använder standardvärden.")

    # Knapp: hämta automatiskt (Frankfurter → exchangerate.host)
    if st.sidebar.button("🌐 Hämta kurser automatiskt", key="btn_rates_auto"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.sidebar.success(f"Valutakurser (källa: {provider}) hämtade.")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        # Lägg i buffer och signalera att vi vill applicera innan widgets ritas nästa gång
        st.session_state["_apply_rates_buffer"] = auto_rates
        st.session_state["_apply_rates_flag"] = True

    # Själva input-widgets
    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state["rate_usd_input"]), step=0.01, format="%.4f", key="rate_usd_input")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state["rate_nok_input"]), step=0.01, format="%.4f", key="rate_nok_input")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state["rate_cad_input"]), step=0.01, format="%.4f", key="rate_cad_input")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state["rate_eur_input"]), step=0.01, format="%.4f", key="rate_eur_input")

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser", key="btn_rates_save"):
            try:
                spara_valutakurser({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
                st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
                st.sidebar.success("Valutakurser sparade.")
            except Exception as e:
                st.sidebar.warning(f"Kunde inte spara kurser: {e}")
    with col_rates2:
        if st.button("↻ Läs sparade kurser", key="btn_rates_reload"):
            # Lägg in sparade som buffer som appliceras innan widgets skapas nästa gång
            sr = las_sparade_valutakurser()
            st.session_state["_apply_rates_buffer"] = {
                "USD": float(sr.get("USD", STANDARD_VALUTAKURSER["USD"])),
                "NOK": float(sr.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
                "CAD": float(sr.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
                "EUR": float(sr.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
            }
            st.session_state["_apply_rates_flag"] = True
            st.sidebar.info("Sparade kurser läses in vid nästa render.")

    user_rates = {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}
    return user_rates


def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: Dict[str,float]) -> pd.DataFrame:
    """
    Kapslar batchpanelen och spara-funktion.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧰 Åtgärder")

    def _save(d: pd.DataFrame):
        d2 = ensure_schema(ensure_ticker_col(d))
        spara_data(d2)

    df2 = sidebar_batch_controls(df, user_rates, save_cb=_save, default_sort="Äldst uppdaterade först (alla fält)")
    return df2


# -----------------------------
# MAIN
# -----------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs data
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = ensure_schema(df)
        try:
            spara_data(df)
        except Exception as e:
            st.warning(f"Kunde inte init-spara tomt schema: {e}")

    df = ensure_schema(df)
    df = ensure_ticker_col(df)

    # Valutakurser i sidopanelen
    user_rates = _sidebar_rates()

    # Batch & åtgärder i sidopanelen
    df = _sidebar_batch_and_actions(df, user_rates)

    # Meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    # Save-callback att skicka in till vyer som ska få spara
    def _save_df(d: pd.DataFrame):
        d2 = ensure_schema(ensure_ticker_col(d))
        spara_data(d2)

    # Visa vald vy
    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates, save_cb=_save_df)
        # Uppdatera referensen om något förändrats lokalt i vyn
        if isinstance(df2, pd.DataFrame):
            df = df2
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
