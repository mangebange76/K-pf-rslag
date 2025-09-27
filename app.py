# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import requests
import time
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials

# Vyer + Batch
from stockapp.views import (
    kontrollvy, analysvy, lagg_till_eller_uppdatera, visa_investeringsforslag, visa_portfolj
)
from stockapp.batch import sidebar_batch_controls

# Om du har en calc-modul – använd den. Annars fallback.
try:
    from stockapp.calc import uppdatera_berakningar
except Exception:
    def uppdatera_berakningar(df, user_rates):  # minimal fallback
        if "P/S Q1" in df and "P/S Q2" in df and "P/S Q3" in df and "P/S Q4" in df:
            ps_cols = ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]
            vals = df[ps_cols].replace({None:0}).astype(float)
            df["P/S-snitt"] = vals[vals>0].mean(axis=1).fillna(0.0).round(2)
        return df

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

# --- Google Sheets-koppling ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    """Liten backoff-hjälpare för att mildra 429/kvotfel."""
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

def skapa_koppling():
    return get_spreadsheet().worksheet(SHEET_NAME)

def skapa_rates_sheet_if_missing():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        _with_backoff(ws.update, [["Valuta","Kurs"]])
        return ws

def hamta_data():
    sheet = skapa_koppling()
    data = _with_backoff(sheet.get_all_records)
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame, do_snapshot: bool = False):
    """Skriv hela DataFrame till huvudbladet. (Snapshot valfritt – av som default i batch.)"""
    if do_snapshot:
        try:
            backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)
        except Exception as e:
            st.warning(f"Kunde inte skapa snapshot före skrivning: {e}")
    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    _with_backoff(sheet.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# --- Tids-/utility ---
def _ts_str():
    return now_dt().strftime("%Y%m%d-%H%M%S")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# --- Standard valutakurser till SEK (fallback/startvärden) ---
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    ws = skapa_rates_sheet_if_missing()
    rows = _with_backoff(ws.get_all_records)  # [{'Valuta': 'USD', 'Kurs': '9.46'}, ...]
    out = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        val = str(r.get("Kurs", "")).replace(",", ".").strip()
        try:
            out[cur] = float(val)
        except:
            pass
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = skapa_rates_sheet_if_missing()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, str(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurser_auto():
    misses = []
    rates = {}
    provider = None

    # 1) Frankfurter (ECB)
    if len(rates) < 4:
        provider = "Frankfurter"
        for base_ccy in ("USD","EUR","CAD","NOK"):
            try:
                r2 = requests.get("https://api.frankfurter.app/latest",
                                  params={"from": base_ccy, "to": "SEK"}, timeout=10)
                if r2.status_code == 200:
                    rr = r2.json() or {}
                    v = (rr.get("rates") or {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass

    # 2) exchangerate.host (fallback)
    if len(rates) < 4:
        provider = "exchangerate.host"
        for base_ccy in ("USD","EUR","CAD","NOK"):
            try:
                r = requests.get("https://api.exchangerate.host/latest",
                                 params={"base": base_ccy, "symbols": "SEK"}, timeout=10)
                if r.status_code == 200:
                    v = (r.json() or {}).get("rates", {}).get("SEK")
                    if v:
                        rates[base_ccy] = float(v)
            except Exception:
                pass

    # Fyll luckor med sparade/standard
    saved = las_sparade_valutakurser()
    for base_ccy in ("USD","EUR","CAD","NOK"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))

    return rates, misses, (provider or "okänd")

# --- Kolumnschema & TS-fält --------------------------------------------------
TS_FIELDS = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
    # TS
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"):
                df[kol] = ""
            else:
                df[kol] = ""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Riktkurs 2026": "Riktkurs om 1 år",
        "Riktkurs 2027": "Riktkurs om 2 år",
        "Riktkurs 2028": "Riktkurs om 3 år",
        "Riktkurs om idag": "Riktkurs idag",
    }
    for old, new in mapping.items():
        if old in df.columns:
            if new not in df.columns:
                df[new] = 0.0
            new_vals = pd.to_numeric(df[new], errors="coerce").fillna(0.0)
            old_vals = pd.to_numeric(df[old], errors="coerce").fillna(0.0)
            mask = (new_vals == 0.0) & (old_vals > 0.0)
            df.loc[mask, new] = old_vals[mask]
            df = df.drop(columns=[old])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)
    return df

# --- Snapshot (valfri) -------------------------------------------------------
def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    ss = get_spreadsheet()
    snap_name = f"Snapshot-{_ts_str()}"
    try:
        ss.add_worksheet(title=snap_name, rows=max(1000, len(df)+10), cols=max(50, len(df.columns)+2))
        ws = ss.worksheet(snap_name)
        _with_backoff(ws.clear)
        _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())
        st.success(f"Snapshot skapad: {snap_name}")
    except Exception as e:
        st.warning(f"Misslyckades skapa snapshot-flik: {e}")

# ------------------------------
# SIDOPANEL: Valutakurser (säker)
# ------------------------------
def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # initera state en gång
    if "rate_usd_input" not in st.session_state: st.session_state.rate_usd_input = STANDARD_VALUTAKURSER["USD"]
    if "rate_eur_input" not in st.session_state: st.session_state.rate_eur_input = STANDARD_VALUTAKURSER["EUR"]
    if "rate_cad_input" not in st.session_state: st.session_state.rate_cad_input = STANDARD_VALUTAKURSER["CAD"]
    if "rate_nok_input" not in st.session_state: st.session_state.rate_nok_input = STANDARD_VALUTAKURSER["NOK"]

    # widgets (bara key – inga direkta .value-assigns efter att widgets skapats)
    st.sidebar.number_input("USD → SEK", min_value=0.0, step=0.0001, format="%.4f", key="rate_usd_input")
    st.sidebar.number_input("EUR → SEK", min_value=0.0, step=0.0001, format="%.4f", key="rate_eur_input")
    st.sidebar.number_input("CAD → SEK", min_value=0.0, step=0.0001, format="%.4f", key="rate_cad_input")
    st.sidebar.number_input("NOK → SEK", min_value=0.0, step=0.0001, format="%.4f", key="rate_nok_input")

    col_rates1, col_rates2 = st.sidebar.columns(2)

    with col_rates1:
        if st.button("🌐 Hämta kurser automatiskt"):
            auto_rates, misses, provider = hamta_valutakurser_auto()
            # uppdatera widget-state (tillåtet)
            st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
            st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
            st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
            st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
            st.sidebar.success(f"Valutakurser (källa: {provider}) hämtade.")
            if misses:
                st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))

    with col_rates2:
        if st.button("💾 Spara kurser"):
            rates = {
                "USD": float(st.session_state.rate_usd_input),
                "EUR": float(st.session_state.rate_eur_input),
                "CAD": float(st.session_state.rate_cad_input),
                "NOK": float(st.session_state.rate_nok_input),
                "SEK": 1.0,
            }
            spara_valutakurser(rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om sparade kurser"):
        saved = las_sparade_valutakurser()
        st.session_state.rate_usd_input = float(saved.get("USD", st.session_state.rate_usd_input))
        st.session_state.rate_eur_input = float(saved.get("EUR", st.session_state.rate_eur_input))
        st.session_state.rate_cad_input = float(saved.get("CAD", st.session_state.rate_cad_input))
        st.session_state.rate_nok_input = float(saved.get("NOK", st.session_state.rate_nok_input))
        st.sidebar.success("Kurser laddade.")

    return {
        "USD": float(st.session_state.rate_usd_input),
        "EUR": float(st.session_state.rate_eur_input),
        "CAD": float(st.session_state.rate_cad_input),
        "NOK": float(st.session_state.rate_nok_input),
        "SEK": 1.0,
    }

# --------------------------------------------
# Sidopanel: Batch (koppling till batch-modul)
# --------------------------------------------
def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    # callbacks som batch-panelen använder
    def _save_cb(df_new: pd.DataFrame):
        spara_data(df_new, do_snapshot=False)

    def _recompute_cb(df_new: pd.DataFrame) -> pd.DataFrame:
        return uppdatera_berakningar(df_new, user_rates)

    df_out = sidebar_batch_controls(df, user_rates, save_cb=_save_cb, recompute_cb=_recompute_cb)
    return df_out

# -------------
# MAIN-LOOPEN
# -------------
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Sidopanel valutakurser
    user_rates = _sidebar_rates()

    # 2) Läs data från Google Sheets
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa data: {e}")
        df = pd.DataFrame({})
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    # 3) Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 4) Batch-panel i sidopanelen (kan returnera uppdaterat df)
    df = _sidebar_batch_and_actions(df, user_rates)

    # 5) Meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        if df2 is not None and isinstance(df2, pd.DataFrame) and not df2.equals(df):
            # Spara direkt om användaren ändrade och sparade
            spara_data(df2, do_snapshot=False)
            df = df2
    elif meny == "Investeringsförslag":
        df_calc = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        df_calc = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df_calc, user_rates)

if __name__ == "__main__":
    main()
