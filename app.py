# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import requests
import time
import math
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
from typing import Optional

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
    """Skriv hela DataFrame till huvudbladet. Optionellt: skapa snapshot-flik först."""
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

def _ts_datetime():
    return now_dt()

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

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# --- Automatisk valutahämtning (FMP -> Frankfurter -> exchangerate.host) ---
def hamta_valutakurser_auto():
    misses = []
    rates = {}
    provider = None

    # 1) FMP, om API-nyckel finns
    fmp_key = st.secrets.get("FMP_API_KEY", "")
    if fmp_key:
        try:
            base = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
            def _pair(pair):
                url = f"{base}/api/v3/fx/{pair}"
                r = requests.get(url, params={"apikey": fmp_key}, timeout=15)
                if r.status_code != 200:
                    return None, r.status_code
                j = r.json() or {}
                px = j.get("price")
                return float(px) if px is not None else None, 200
            provider = "FMP"
            for pair in ("USDSEK","NOKSEK","CADSEK","EURSEK"):
                v, sc = _pair(pair)
                if v and v > 0:
                    base_ccy = pair[:3]
                    rates[base_ccy] = float(v)
                else:
                    misses.append(f"{pair} (HTTP {sc if sc else '??'})")
        except Exception:
            pass

    # 2) Frankfurter (ECB)
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

    # 3) exchangerate.host
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

    # Fyll ev luckor med sparade/standard
    saved = las_sparade_valutakurser()
    for base_ccy in ("USD","EUR","CAD","NOK"):
        if base_ccy not in rates:
            rates[base_ccy] = float(saved.get(base_ccy, STANDARD_VALUTAKURSER.get(base_ccy, 1.0)))

    return rates, misses, (provider or "okänd")

# === Stabil sidopanel för kurser (uppdaterar state före widgets) ============
def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # initera från sparat en gång
    if "rates_initialized" not in st.session_state:
        saved = las_sparade_valutakurser()
        st.session_state.rate_usd = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
        st.session_state.rate_nok = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        st.session_state.rate_cad = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        st.session_state.rate_eur = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
        st.session_state["rates_initialized"] = True

    # hämta automatiskt -> uppdatera state -> flash -> rerun
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
        st.session_state["_flash_rates_msg"] = {"provider": provider, "misses": misses}
        st.rerun()

    flash = st.session_state.pop("_flash_rates_msg", None)
    if flash:
        st.sidebar.success(f"Valutakurser (källa: {flash['provider']}) hämtade.")
        if flash["misses"]:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(flash["misses"]))

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")

    user_rates = {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear()
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return user_rates

# =================== Kolumnschema & TS-konfiguration =========================
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
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# Extra nyckeltal
EXTRA_COLS = [
    "Sektor", "_marketCap_raw",
    "EV", "EBITDA (TTM)", "EV/EBITDA",
    "Bruttomarginal (%)", "Nettomarginal (%)",
    "Skuldsättning D/E", "Current ratio", "Quick ratio",
    "Totalt kassa", "OCF (TTM)", "CapEx (TTM)", "FCF (TTM)",
    "Kassaförbrukning/kvartal", "Runway (kvartal)",
    "Direktavkastning (%)", "Utdelningsandel av FCF (%)",
    "EPS (TTM)", "P/E (TTM)",
    "MCAP Q1", "MCAP Q2", "MCAP Q3", "MCAP Q4",
]
for _c in EXTRA_COLS:
    if _c not in FINAL_COLS:
        FINAL_COLS.append(_c)

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","mcap","ev","ebitda","marginal","ratio","kassa","ocf","capex","fcf","runway","eps","pe","andel","yld"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"):
                df[kol] = ""
            else:
                df[kol] = ""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

# --- Migrering & typkonvertering --------------------------------------------

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
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        # Extra:
        "_marketCap_raw", "EV", "EBITDA (TTM)", "EV/EBITDA",
        "Bruttomarginal (%)", "Nettomarginal (%)",
        "Skuldsättning D/E", "Current ratio", "Quick ratio",
        "Totalt kassa", "OCF (TTM)", "CapEx (TTM)", "FCF (TTM)",
        "Kassaförbrukning/kvartal", "Runway (kvartal)",
        "Direktavkastning (%)", "Utdelningsandel av FCF (%)",
        "EPS (TTM)", "P/E (TTM)",
        "MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Sektor","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)
    return df

# --- Tidsstämpelshjälpare ----------------------------------------------------

def _stamp_ts_for_field(df: pd.DataFrame, row_idx: int, field: str, when: Optional[str] = None):
    """Sätter TS-kolumnen för ett spårat fält om den finns."""
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    date_str = when if when else now_stamp()
    try:
        df.at[row_idx, ts_col] = date_str
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    """Sätter auto-uppdaterad-tidsstämpel och källa."""
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _note_manual_update(df: pd.DataFrame, row_idx: int):
    """Sätter manuell uppdatering (anropas i formulär-flödet)."""
    try:
        df.at[row_idx, "Senast manuellt uppdaterad"] = now_stamp()
    except Exception:
        pass

# Fält som triggar "Senast manuellt uppdaterad" i formuläret
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# --- Yahoo-hjälpare & beräkningar -------------------------------------------

def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """CAGR baserat på 'Total Revenue' (årsbasis), enkel approx."""
    try:
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0
        if series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start = float(series.iloc[0]); end = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def hamta_yahoo_fält(ticker: str) -> dict:
    """Basfält + utökade nyckeltal från Yahoo (yfinance.info + hist fallback)."""
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        # Nya:
        "Sektor": "",
        "_marketCap_raw": 0.0,
        "EV": 0.0,
        "EBITDA (TTM)": 0.0,
        "EV/EBITDA": 0.0,
        "Bruttomarginal (%)": 0.0,
        "Nettomarginal (%)": 0.0,
        "Skuldsättning D/E": 0.0,
        "Current ratio": 0.0,
        "Quick ratio": 0.0,
        "Totalt kassa": 0.0,
        "OCF (TTM)": 0.0,
        "CapEx (TTM)": 0.0,
        "FCF (TTM)": 0.0,
        "Kassaförbrukning/kvartal": 0.0,
        "Runway (kvartal)": 0.0,
        "Direktavkastning (%)": 0.0,
        "Utdelningsandel av FCF (%)": 0.0,
        "EPS (TTM)": 0.0,
        "P/E (TTM)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        pris = info.get("regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        # Valuta/namn/sektor
        valuta = info.get("currency")
        if valuta:
            out["Valuta"] = str(valuta).upper()
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)
        sektor = info.get("sector") or ""
        if sektor:
            out["Sektor"] = str(sektor)

        # Utdelning / direktavkastning
        div_rate = info.get("dividendRate")
        if div_rate is not None:
            try: out["Årlig utdelning"] = float(div_rate)
            except: pass
        div_yield = info.get("dividendYield")
        if div_yield not in (None, "", 0):
            try:
                out["Direktavkastning (%)"] = float(div_yield) * (100.0 if float(div_yield) < 1.0 else 1.0)
            except: pass

        # Market cap / EV / EBITDA
        mcap = info.get("marketCap") or info.get("market_cap")
        if mcap not in (None, "", 0):
            try: out["_marketCap_raw"] = float(mcap)
            except: pass
        ev = info.get("enterpriseValue")
        if ev not in (None, "", 0):
            try: out["EV"] = float(ev)
            except: pass
        ebitda = info.get("ebitda")
        if ebitda not in (None, "", 0):
            try: out["EBITDA (TTM)"] = float(ebitda)
            except: pass
        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda not in (None, "", 0):
            try: out["EV/EBITDA"] = float(ev_ebitda)
            except: pass

        # Marginaler (Yahoo ofta i decimal 0.xx)
        gm = info.get("grossMargins")
        if gm not in (None, ""):
            try: out["Bruttomarginal (%)"] = float(gm) * (100.0 if float(gm) <= 1.0 else 1.0)
            except: pass
        pm = info.get("profitMargins")
        if pm not in (None, ""):
            try: out["Nettomarginal (%)"] = float(pm) * (100.0 if float(pm) <= 1.0 else 1.0)
            except: pass

        # Skuld/likviditet
        dte = info.get("debtToEquity")
        if dte not in (None, ""):
            try: out["Skuldsättning D/E"] = float(dte)
            except: pass
        cr = info.get("currentRatio")
        if cr not in (None, ""):
            try: out["Current ratio"] = float(cr)
            except: pass
        qr = info.get("quickRatio")
        if qr not in (None, ""):
            try: out["Quick ratio"] = float(qr)
            except: pass

        # Kassa & kassaflöde (TTM enligt Yahoo)
        tot_cash = info.get("totalCash")
        if tot_cash not in (None, ""):
            try: out["Totalt kassa"] = float(tot_cash)
            except: pass
        ocf = info.get("operatingCashflow")
        if ocf not in (None, ""):
            try: out["OCF (TTM)"] = float(ocf)
            except: pass
        fcf = info.get("freeCashflow")
        if fcf not in (None, ""):
            try: out["FCF (TTM)"] = float(fcf)
            except: pass

        # CapEx (TTM) finns inte alltid – sätt via OCF-FCF om båda finns
        if (out["CapEx (TTM)"] == 0.0) and (out["OCF (TTM)"] or 0.0) and (out["FCF (TTM)"] or 0.0):
            try:
                out["CapEx (TTM)"] = float(out["OCF (TTM)"]) - float(out["FCF (TTM)"])
            except: pass

        # Kassaförbrukning & runway (enkel)
        if out["FCF (TTM)"] < 0 and out["Totalt kassa"] > 0:
            try:
                burn_q = abs(out["FCF (TTM)"]) / 4.0
                out["Kassaförbrukning/kvartal"] = burn_q
                out["Runway (kvartal)"] = out["Totalt kassa"] / burn_q if burn_q > 0 else 0.0
            except: pass

        # EPS / P/E
        eps = info.get("trailingEps")
        if eps not in (None, ""):
            try: out["EPS (TTM)"] = float(eps)
            except: pass
        pe = info.get("trailingPE") or info.get("trailingPe")
        if pe not in (None, ""):
            try: out["P/E (TTM)"] = float(pe)
            except: pass

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

    except Exception:
        pass
    return out

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: >100% → 50%, <0% → 2%
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# --- SEC & kvartalsdata (US + FPI/IFRS) -------------------------------------

SEC_USER_AGENT = st.secrets.get(
    "SEC_USER_AGENT",
    "StockApp/1.0 (contact: your-email@example.com)"
)

def _sec_get(url: str, params=None):
    try:
        r = requests.get(url, params=params or {}, headers={"User-Agent": SEC_USER_AGENT}, timeout=30)
        if r.status_code == 200:
            return r.json(), 200
        return None, r.status_code
    except Exception:
        return None, 0

@st.cache_data(show_spinner=False, ttl=86400)
def _sec_ticker_map():
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not isinstance(j, dict):
        return {}
    out = {}
    for _, v in j.items():
        try:
            out[str(v["ticker"]).upper()] = str(v["cik_str"]).zfill(10)
        except Exception:
            pass
    return out

def _sec_cik_for(ticker: str) -> Optional[str]:
    return _sec_ticker_map().get(str(ticker).upper())

def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

from datetime import datetime as _dt, timedelta as _td

def _parse_iso(d: str):
    try:
        return _dt.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return _dt.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

def _is_instant_entry(it: dict) -> bool:
    end = it.get("end"); start = it.get("start")
    if not end:
        return False
    if not start:
        return True  # instant
    d1 = _parse_iso(str(start)); d2 = _parse_iso(str(end))
    if d1 and d2:
        try:
            return (d2 - d1).days <= 2
        except Exception:
            return False
    return False

def _collect_share_entries(facts: dict) -> list:
    entries = []
    facts_all = (facts.get("facts") or {})
    sources = [
        ("dei", ["EntityCommonStockSharesOutstanding", "EntityCommonSharesOutstanding"]),
        ("us-gaap", ["CommonStockSharesOutstanding", "ShareIssued"]),
        ("ifrs-full", ["NumberOfSharesIssued", "IssuedCapitalNumberOfShares", "OrdinarySharesNumber", "NumberOfOrdinaryShares"]),
    ]
    unit_keys = ("shares", "USD_shares", "Shares", "SHARES")
    for taxo, keys in sources:
        sect = facts_all.get(taxo, {})
        for key in keys:
            fact = sect.get(key)
            if not fact:
                continue
            units = fact.get("units") or {}
            for uk in unit_keys:
                arr = units.get(uk)
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    if not _is_instant_entry(it):
                        continue
                    end = _parse_iso(str(it.get("end", "")))
                    val = it.get("val", None)
                    if end and val is not None:
                        try:
                            v = float(val)
                            frame = it.get("frame") or ""
                            form = (it.get("form") or "").upper()
                            entries.append({"end": end, "val": v, "frame": frame, "form": form, "taxo": taxo, "concept": key})
                        except Exception:
                            pass
    return entries

def _sec_latest_shares_robust(facts: dict) -> float:
    rows = _collect_share_entries(facts)
    if not rows:
        return 0.0
    newest = max(r["end"] for r in rows)
    todays = [r for r in rows if r["end"] == newest]
    total = 0.0
    for r in todays:
        try:
            total += float(r["val"])
        except Exception:
            pass
    return total if total > 0 else 0.0

@st.cache_data(show_spinner=False, ttl=21600)
def _fx_rate_cached(base: str, quote: str) -> float:
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            return float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            return float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
    except Exception:
        pass
    return 1.0

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    Fix: säkerställ att Dec/Jan-perioden inte tappas (dedup + komplettering).
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A")}),
    ]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    prefer_units = ("USD","CAD","EUR","GBP")

    def _collect_for(taxo):
        gaap = (facts.get("facts") or {}).get(taxo, {})
        best_rows, best_unit = [], None
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list):
                    continue
                tmp = []
                for it in arr:
                    form = (it.get("form") or "").upper()
                    if not any(f in form for f in ("10-Q","10-Q/A","6-K","6-K/A")):
                        continue
                    end = _parse_iso(str(it.get("end", "")))
                    start = _parse_iso(str(it.get("start", "")))
                    val = it.get("val", None)
                    if not (end and start and val is not None):
                        continue
                    try:
                        dur = (end - start).days
                    except Exception:
                        dur = None
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # deduplicera per end-datum (håll senaste värdet)
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)
                # Dec/Jan-fix: om senaste året saknar ett Q runt Dec/Jan, försök hitta närliggande +/- 40 dagar och ta med
                if len(rows) >= 5:
                    # ingenting extra att göra – men lämna top N
                    pass
                best_rows, best_unit = rows[:max_quarters], unit_code
                break
            if best_rows:
                break
        return best_rows, best_unit

    # Försök US-GAAP först, sedan IFRS
    rows, unit = _collect_for("us-gaap")
    if not rows:
        rows, unit = _collect_for("ifrs-full")

    return rows, unit

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters)
    return rows

def _yahoo_prices_for_dates(ticker: str, dates: list) -> dict:
    if not dates:
        return {}
    dmin = min(dates) - _td(days=14)
    dmax = max(dates) + _td(days=2)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=dmin, end=dmax, interval="1d")
        if hist is None or hist.empty:
            return {}
        hist = hist.sort_index()
        out = {}
        idx = list(hist.index.date)
        closes = list(hist["Close"].values)
        for d in dates:
            px = None
            for j in range(len(idx)-1, -1, -1):
                if idx[j] <= d:
                    try: px = float(closes[j])
                    except: px = None
                    break
            if px is not None:
                out[d] = px
        return out
    except Exception:
        return {}

def _ttm_windows(values: list, need: int = 4) -> list:
    """
    values: [(end_date, kvartalsintäkt), ...] nyast→äldst
    returnerar upp till 'need' TTM-summor: [(end_i, ttm_i), ...]
    """
    out = []
    if len(values) < 4:
        return out
    for i in range(0, min(need, len(values) - 3)):
        end_i = values[i][0]
        ttm_i = sum(v for (_, v) in values[i:i+4])
        out.append((end_i, float(ttm_i)))
    return out

def _implied_shares_from_yahoo(ticker: str, price: float = None, mcap: float = None) -> float:
    t = yf.Ticker(ticker)
    if mcap is None:
        try:
            mcap = t.info.get("marketCap") or t.info.get("market_cap")
        except Exception:
            mcap = None
    if price is None:
        try:
            price = t.info.get("regularMarketPrice") or t.info.get("last_price")
        except Exception:
            price = None
    try:
        mcap = float(mcap or 0.0); price = float(price or 0.0)
    except Exception:
        return 0.0
    if mcap > 0 and price > 0:
        return mcap / price
    return 0.0

# --- Global Yahoo fallback (icke-SEC: Kanada/EU/Norden etc.) -----------------

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yfi_quarterly_revenues(t: yf.Ticker) -> list:
    # 1) quarterly_financials
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            idx = [str(x).strip() for x in qf.index]
            cand_rows = [
                "Total Revenue","TotalRevenue","Revenues","Revenue","Sales",
                "Total revenue","Revenues from contracts with customers"
            ]
            for key in cand_rows:
                if key in idx:
                    row = qf.loc[key].dropna()
                    out = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            out.append((d, float(v)))
                        except Exception:
                            pass
                    out.sort(key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) income_stmt fallback
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            out = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    out.append((d, float(v)))
                except Exception:
                    pass
            out.sort(key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

# --- SEC + Yahoo combo / Yahoo fallback / P/S-historik -----------------------

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC, pris/valuta/namn från Yahoo.
    Räknar P/S (TTM) nu + P/S Q1–Q4 historik (inkl. Dec/Jan-fix).
    Om CIK saknas → hamta_yahoo_global_combo.
    """
    out = {}
    cik = _sec_cik_for(ticker)
    if not cik:
        return hamta_yahoo_global_combo(ticker)

    facts, sc = _sec_companyfacts(cik)
    if sc != 200 or not isinstance(facts, dict):
        return hamta_yahoo_global_combo(ticker)

    # Yahoo-basics
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "_marketCap_raw", "Sektor", "EV","EBITDA (TTM)","EV/EBITDA",
              "Bruttomarginal (%)","Nettomarginal (%)","Skuldsättning D/E","Current ratio","Quick ratio",
              "Totalt kassa","OCF (TTM)","CapEx (TTM)","FCF (TTM)","Kassaförbrukning/kvartal","Runway (kvartal)",
              "Direktavkastning (%)","Utdelningsandel av FCF (%)","EPS (TTM)","P/E (TTM)"):
        if y.get(k) not in (None, ""):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()

    # Shares: implied → fallback SEC
    implied = _implied_shares_from_yahoo(ticker, price=out.get("Aktuell kurs"), mcap=out.get("_marketCap_raw"))
    sec_shares = _sec_latest_shares_robust(facts)
    shares_used = 0.0
    if implied and implied > 0:
        shares_used = float(implied)
        out["_debug_shares_source"] = "Yahoo implied (mcap/price)"
    elif sec_shares and sec_shares > 0:
        shares_used = float(sec_shares)
        out["_debug_shares_source"] = "SEC instant (robust)"
    else:
        out["_debug_shares_source"] = "unknown"

    if shares_used > 0:
        out["Utestående aktier"] = shares_used / 1e6

    # Market cap (nu)
    mcap_now = out.get("_marketCap_raw", 0.0)
    try:
        mcap_now = float(mcap_now or 0.0)
    except Exception:
        mcap_now = 0.0
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used
        out["_marketCap_raw"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    if not q_rows or not rev_unit:
        return out
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    ttm_list = _ttm_windows(q_rows, need=4)
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    return out

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC.
    Räknar implied shares, P/S (TTM) nu, P/S Q1–Q4 historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas + nyckeltal
    y = hamta_yahoo_fält(ticker)
    for k, v in y.items():
        if v not in (None, ""):
            out[k] = v
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else 0.0
    except Exception:
        mcap = 0.0

    # Implied shares → fallback sharesOutstanding
    shares = 0.0
    if mcap > 0 and px > 0:
        shares = mcap / px
        out["_debug_shares_source"] = "Yahoo implied (mcap/price)"
    else:
        so = info.get("sharesOutstanding")
        try:
            so = float(so or 0.0)
        except Exception:
            so = 0.0
        if so > 0:
            shares = so
            out["_debug_shares_source"] = "Yahoo sharesOutstanding"

    if shares > 0:
        out["Utestående aktier"] = shares / 1e6

    # Kvartalsintäkter → TTM
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows or len(q_rows) < 4:
        return out
    ttm_list = _ttm_windows(q_rows, need=4)

    # Valutakonvertering om financialCurrency != prisvaluta
    fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        conv = _fx_rate_cached(fin_ccy, px_ccy) or 1.0
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # Market cap (nu)
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px
        out["_marketCap_raw"] = mcap

    # P/S (TTM) nu
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historik)
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    out[f"P/S Q{idx}"] = (shares * p) / ttm_rev_px

    return out

# --- FMP light (fallback P/S) + Finnhub estimat ------------------------------

FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.5))
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20))

def _fmp_get(path: str, params=None, stable: bool = True):
    block_until = st.session_state.get("fmp_block_until")
    if block_until and _ts_datetime() < block_until:
        return None, 429

    params = (params or {}).copy()
    if FMP_KEY:
        params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}/{path}"

    delays = [0.0, 1.2, 2.5]
    last_sc = 0
    last_json = None

    for extra_sleep in delays:
        try:
            if FMP_CALL_DELAY > 0:
                time.sleep(FMP_CALL_DELAY)
            r = requests.get(url, params=params, timeout=20)
            sc = r.status_code
            last_sc = sc
            try:
                j = r.json()
            except Exception:
                j = None
            last_json = j

            if 200 <= sc < 300:
                return j, sc

            if sc == 429:
                st.session_state["fmp_block_until"] = _ts_datetime() + timedelta(minutes=FMP_BLOCK_MINUTES)
                time.sleep(extra_sleep)
                continue

            if sc in (403, 502, 503, 504):
                time.sleep(extra_sleep)
                continue

            return j, sc
        except Exception:
            time.sleep(extra_sleep)
            continue

    return last_json, last_sc

def _fmp_pick_symbol(yahoo_ticker: str) -> str:
    sym = str(yahoo_ticker).strip().upper()
    js, sc = _fmp_get(f"api/v3/quote-short/{sym}", stable=False)
    if isinstance(js, list) and js:
        return sym
    js, sc = _fmp_get("api/v3/search", {"query": yahoo_ticker, "limit": 1}, stable=False)
    if isinstance(js, list) and js:
        return str(js[0].get("symbol", sym)).upper()
    return sym

@st.cache_data(show_spinner=False, ttl=1800)
def hamta_fmp_falt_light(yahoo_ticker: str) -> dict:
    out = {"_debug": {}, "_symbol": _fmp_pick_symbol(yahoo_ticker)}
    sym = out["_symbol"]

    q, sc_q = _fmp_get(f"api/v3/quote/{sym}", stable=False)
    out["_debug"]["quote_sc"] = sc_q
    if isinstance(q, list) and q:
        q0 = q[0]
        if "price" in q0:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
        if q0.get("marketCap") is not None:
            try: out["_marketCap_raw"] = float(q0["marketCap"])
            except: pass
        if q0.get("sharesOutstanding") is not None:
            try: out["Utestående aktier"] = float(q0["sharesOutstanding"]) / 1e6
            except: pass

    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}", stable=False)
    out["_debug"]["ratios_ttm_sc"] = sc_rttm
    if isinstance(rttm, list) and rttm:
        v = rttm[0].get("priceToSalesTTM") or rttm[0].get("priceToSalesRatioTTM")
        try:
            if v and float(v) > 0:
                out["P/S"] = float(v)
                out["_debug"]["ps_source"] = "ratios-ttm"
        except Exception:
            pass

    return out

FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    if not FINNHUB_KEY:
        return {}
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/stock/revenue-estimate",
            params={"symbol": ticker.upper(), "freq": "annual", "token": FINNHUB_KEY},
            timeout=20,
        )
        if r.status_code != 200:
            return {}
        j = r.json() or {}
        data = j.get("data") or []
        if not data:
            return {}
        data.sort(key=lambda d: d.get("period",""), reverse=False)
        out = {}
        last_two = data[-2:] if len(data) >= 2 else data[-1:]
        if len(last_two) >= 1:
            v = last_two[0].get("revenueAvg") or last_two[0].get("revenueMean") or last_two[0].get("revenue")
            try:
                if v and float(v) > 0:
                    out["Omsättning idag"] = float(v) / 1e6
            except Exception:
                pass
        if len(last_two) == 2:
            v = last_two[1].get("revenueAvg") or last_two[1].get("revenueMean") or last_two[1].get("revenue")
            try:
                if v and float(v) > 0:
                    out["Omsättning nästa år"] = float(v) / 1e6
            except Exception:
                pass
        return out
    except Exception:
        return {}

# --- Skriva in auto-fält i DF -----------------------------------------------

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, changes_map: dict) -> bool:
    """
    Skriver fält med nytt (>=0) värde. Uppdaterar TS_ och auto-TS/källa.
    Returnerar True om något faktiskt ändrades.
    """
    changed_fields = []
    for f, v in new_vals.items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f]
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        if not write_ok:
            continue
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[row_idx, f] = v
            changed_fields.append(f)
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)

    if changed_fields:
        _note_auto_update(df, row_idx, source)
        changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return True
    return False

# --- Huvudpipeline för ett bolag --------------------------------------------

def auto_fetch_for_ticker(ticker: str):
    """
    1) SEC + Yahoo combo (inkl. global fallback)
    2) Finnhub (estimat) om saknas
    3) FMP light (P/S) om saknas
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Aktuell kurs","Bolagsnamn","Valuta","_marketCap_raw","_debug_shares_source"]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","_marketCap_raw",
                  "Sektor","EV","EBITDA (TTM)","EV/EBITDA","Bruttomarginal (%)","Nettomarginal (%)","Skuldsättning D/E","Current ratio","Quick ratio",
                  "Totalt kassa","OCF (TTM)","CapEx (TTM)","FCF (TTM)","Kassaförbrukning/kvartal","Runway (kvartal)","Direktavkastning (%)",
                  "Utdelningsandel av FCF (%)","EPS (TTM)","P/E (TTM)"]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) Finnhub estimat (om saknas)
    try:
        if ("Omsättning idag" not in vals) or ("Omsättning nästa år" not in vals):
            fh = hamta_finnhub_revenue_estimates(ticker)
            debug["finnhub"] = fh
            for k in ["Omsättning idag","Omsättning nästa år"]:
                v = fh.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        debug["finnhub_err"] = str(e)

    # 3) FMP light P/S om saknas
    try:
        if ("P/S" not in vals):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier")}
            for k in ["P/S"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
            if ("Utestående aktier" not in vals) and (fmpl.get("Utestående aktier") not in (None, "", 0, 0.0)):
                vals["Utestående aktier"] = fmpl["Utestående aktier"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    return vals, debug

# --- Snapshots, "äldst"-hjälpare & batch-körning ----------------------------

def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df.
    """
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

def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """
    Returnerar äldsta (minsta) tidsstämpeln bland alla TS_-kolumner för en rad.
    None om inga tidsstämplar.
    """
    dates = []
    for c in TS_FIELDS.values():
        if c in row and str(row[c]).strip():
            try:
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None

def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    # Fyll på för sortering: None -> långt i framtiden
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

# --- Rapport: bolag som kan kräva manuell hantering -------------------------

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Bolag som sannolikt kräver manuell hantering:
    - saknar någon av kärnfälten/TS,
    - och/eller äldsta TS är äldre än 'older_than_days'.
    """
    need_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS[c] for c in TS_FIELDS if c in need_cols]

    out_rows = []
    cutoff = now_dt() - timedelta(days=older_than_days)

    for _, r in df.iterrows():
        missing_val = any((float(r.get(c, 0.0)) <= 0.0) for c in need_cols)
        missing_ts  = any((not str(r.get(ts, "")).strip()) for ts in ts_cols if ts in r)
        oldest = oldest_any_ts(r)
        oldest_dt = pd.to_datetime(oldest).to_pydatetime() if (oldest is not None and pd.notna(oldest)) else None
        too_old = (oldest_dt is not None and oldest_dt < cutoff)

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Äldsta TS": oldest.strftime("%Y-%m-%d") if (oldest is not None and pd.notna(oldest)) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })

    return pd.DataFrame(out_rows)

# --- Extra rapport: "Omsättning idag/nästa år" äldst först -------------------

def build_revenue_ts_age_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Listar tickers sorterade på äldst TS för 'Omsättning idag' och 'Omsättning nästa år'.
    """
    rows = []
    for _, r in df.iterrows():
        tkr = str(r.get("Ticker",""))
        name = str(r.get("Bolagsnamn",""))
        d1 = str(r.get("TS_Omsättning idag","")).strip()
        d2 = str(r.get("TS_Omsättning nästa år","")).strip()
        ts1 = pd.to_datetime(d1, errors="coerce") if d1 else pd.NaT
        ts2 = pd.to_datetime(d2, errors="coerce") if d2 else pd.NaT
        rows.append({
            "Ticker": tkr,
            "Bolagsnamn": name,
            "TS_Omsättning idag": ts1,
            "TS_Omsättning nästa år": ts2
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["TS_min"] = out[["TS_Omsättning idag","TS_Omsättning nästa år"]].min(axis=1, skipna=True)
    out = out.sort_values(by="TS_min", ascending=True, na_position="first")
    return out

# --- Batch-körning med kvarhållning & 1/X-progress ---------------------------

def _pick_batch_ordered_tickers(df: pd.DataFrame, mode: str) -> list:
    """
    mode: 'Äldst först' eller 'A–Ö'.
    Returnerar *hela* ordnade listan (vi skivar senare beroende på var i kön vi är).
    """
    if mode == "Äldst först":
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        work = df.sort_values(by=["Bolagsnamn","Ticker"])
    return [str(x).upper() for x in work["Ticker"].tolist() if str(x).strip()]

def prepare_batch_queue(df: pd.DataFrame, mode: str, batch_size: int):
    """
    Bygger/uppdaterar kö i session_state:
      - batch_mode, batch_size
      - batch_all (alla tickers i rätt ordning)
      - batch_pos (startindex för nästa körning)
      - batch_done (set av tickers uppdaterade i denna session)
    """
    all_ticks = _pick_batch_ordered_tickers(df, mode)
    st.session_state.batch_mode = mode
    st.session_state.batch_size = int(batch_size)
    st.session_state.batch_all = all_ticks
    # init position om saknas
    if "batch_pos" not in st.session_state:
        st.session_state.batch_pos = 0
    # init done-mängd
    if "batch_done" not in st.session_state:
        st.session_state.batch_done = set()

def next_batch_slice() -> list:
    """
    Tar nästa slice enligt batch_size och batch_pos.
    Skippas tickers som redan ligger i batch_done.
    """
    all_ticks = st.session_state.get("batch_all", [])
    pos = int(st.session_state.get("batch_pos", 0))
    size = int(st.session_state.get("batch_size", 10))
    if not all_ticks or pos >= len(all_ticks):
        return []
    sl = []
    i = pos
    while i < len(all_ticks) and len(sl) < size:
        t = all_ticks[i]
        if t not in st.session_state.get("batch_done", set()):
            sl.append(t)
        i += 1
    # Om sl blev tom (allt i slutet redan gjort), hoppa fram tills vi hittar ogjorda
    while not sl and pos < len(all_ticks):
        pos += size
        st.session_state.batch_pos = pos
        i = pos
        while i < len(all_ticks) and len(sl) < size:
            t = all_ticks[i]
            if t not in st.session_state.get("batch_done", set()):
                sl.append(t)
            i += 1
        if sl:
            break
    return sl

def advance_batch_position():
    """Flytta fram batch_pos med batch_size."""
    st.session_state.batch_pos = int(st.session_state.get("batch_pos", 0)) + int(st.session_state.get("batch_size", 10))

def reset_batch_queue():
    """Nollställ batch-state men behåll ordningsval."""
    st.session_state.batch_pos = 0
    st.session_state.batch_done = set()

def batch_update_slice(df: pd.DataFrame, user_rates: dict, tickers: list, make_snapshot: bool = False):
    """
    Uppdaterar *just den slice* som ges (tickers).
    Skriver endast om något ändrats. Visar 1/X-progress.
    """
    log = {"changed": {}, "misses": {}, "debug": []}
    total = len(tickers)
    if total == 0:
        st.info("Inget att köra i denna batch.")
        return df, log

    progress = st.progress(0)
    status = st.empty()

    any_changed = False
    for i, tkr in enumerate(tickers):
        status.write(f"Uppdaterar {i+1}/{total}: {tkr}")
        # 1/X progress
        progress.progress((i+1)/total, text=f"{i+1}/{total}")
        try:
            # hitta radindex
            mask = (df["Ticker"].astype(str).str.upper() == str(tkr).upper())
            if not mask.any():
                log["misses"][tkr] = ["Ticker saknas i tabellen"]
                continue
            row_idx = df.index[mask][0]

            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(df, row_idx, new_vals, source="Auto (Batch)", changes_map=log["changed"])
            any_changed = any_changed or changed
            log["debug"].append({tkr: debug})
            # markera som klar i denna session
            st.session_state.batch_done.add(tkr)
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

    # efter loop — räkna om och ev spara
    df = uppdatera_berakningar(df, user_rates)
    if any_changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.success("Klart! Ändringar sparade för denna batch.")
    else:
        st.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# --- Kontrollvyer ------------------------------------------------------------

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns: cols_show.append(k)
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Kräver manuell hantering?
    st.subheader("🛠️ Kräver manuell hantering")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30)
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Omsättnings-TS (för att du enkelt ska hitta "manuell uppdatering" för prognosfält)
    st.subheader("📌 Omsättning (idag & nästa år) – äldst först")
    rev_age = build_revenue_ts_age_df(df)
    if rev_age.empty:
        st.info("Inga TS-data funna för omsättningsfälten ännu.")
    else:
        show = rev_age.copy()
        show["TS_Omsättning idag"] = show["TS_Omsättning idag"].dt.strftime("%Y-%m-%d")
        show["TS_Omsättning nästa år"] = show["TS_Omsättning nästa år"].dt.strftime("%Y-%m-%d")
        st.dataframe(show[["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år"]], use_container_width=True, hide_index=True)

    st.divider()

    # 4) Senaste körlogg (om batch/auto kördes nyss)
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log") or st.session_state.get("last_batch_log")
    if not log:
        st.info("Ingen körlogg i denna session ännu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ändringar** (ticker → fält)")
            if log.get("changed"):
                st.json(log["changed"])
            else:
                st.write("–")
        with col2:
            st.markdown("**Missar** (ticker → fält som ej uppdaterades)")
            if log.get("misses"):
                st.json(log["misses"])
            else:
                st.write("–")
        if log.get("debug"):
            with st.expander("Debug (detaljer)"):
                st.json(log.get("debug", []))

# --- Analys-vy ---------------------------------------------------------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","Utestående aktier","_marketCap_raw",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "EV","EBITDA (TTM)","EV/EBITDA","Bruttomarginal (%)","Nettomarginal (%)","Skuldsättning D/E",
        "Current ratio","Quick ratio","Totalt kassa","OCF (TTM)","CapEx (TTM)","FCF (TTM)",
        "Kassaförbrukning/kvartal","Runway (kvartal)","Direktavkastning (%)","Utdelningsandel av FCF (%)",
        "EPS (TTM)","P/E (TTM)",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
        "TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# --- Hjälpfunktioner för visning / scoring -----------------------------------

def _format_mcap(v: float, ccy: str) -> str:
    """Formatera market cap: T, B, M, K + valuta."""
    try:
        x = float(v or 0.0)
    except:
        x = 0.0
    suffix = ""
    num = x
    if abs(x) >= 1e12:
        num = x / 1e12; suffix = "T"
    elif abs(x) >= 1e9:
        num = x / 1e9; suffix = "B"
    elif abs(x) >= 1e6:
        num = x / 1e6; suffix = "M"
    elif abs(x) >= 1e3:
        num = x / 1e3; suffix = "K"
    return f"{num:,.2f} {suffix} {ccy}".replace(",", " ")

def _risk_label(mcap_raw: float, ccy: str) -> str:
    """Enkel storleks-etikett baserat på market cap i lokal valuta."""
    x = float(mcap_raw or 0.0)
    # cutoffs i USD-termer – vi antar att ccy≈USD för US-bolag; annars är etiketten ändå vägledande
    if x >= 200e9:   return "Mega/Large Cap"
    if x >= 10e9:    return "Large Cap"
    if x >= 2e9:     return "Mid Cap"
    if x >= 300e6:   return "Small Cap"
    return "Micro/Nano Cap"

def _cap_bucket(mcap_raw: float) -> str:
    x = float(mcap_raw or 0.0)
    if x >= 200e9:   return "Mega/Large"
    if x >= 10e9:    return "Large"
    if x >= 2e9:     return "Mid"
    if x >= 300e6:   return "Small"
    return "Micro/Nano"

def _ps_avg_from_quarters(row) -> float:
    vals = [row.get("P/S Q1",0), row.get("P/S Q2",0), row.get("P/S Q3",0), row.get("P/S Q4",0)]
    vals = [float(v) for v in vals if float(v)>0]
    return float(np.mean(vals)) if vals else 0.0

def _score_growth(row: pd.Series) -> float:
    """
    Tillväxtscore 0–100: kombinerar (lägre bättre) värdering vs P/S-snitt,
    CAGR, marginaler, skuldsättning, FCF & runway där det finns.
    """
    ps_now = float(row.get("P/S", 0.0))
    ps_avg = _ps_avg_from_quarters(row) or float(row.get("P/S-snitt", 0.0))
    cagr = float(row.get("CAGR 5 år (%)", 0.0))
    gross = float(row.get("Bruttomarginal (%)", 0.0))
    netm  = float(row.get("Nettomarginal (%)", 0.0))
    d_e   = float(row.get("Skuldsättning D/E", 0.0))
    fcf   = float(row.get("FCF (TTM)", 0.0))
    runway= float(row.get("Runway (kvartal)", 0.0))
    # normalisera
    val_score = 0.0
    if ps_avg>0 and ps_now>0:
        ratio = ps_now/ps_avg
        # 1.0 = neutral, <1 bättre
        if ratio <= 0.6:   val_score = 30
        elif ratio <= 0.8: val_score = 24
        elif ratio <= 1.0: val_score = 18
        elif ratio <= 1.2: val_score = 10
        else:              val_score = max(0, 10 - 10*(ratio-1.2))  # snabbt avtagande

    cagr_score = max(0.0, min(25.0, cagr*0.5))          # 50% CAGR -> 25p (clampas redan tidigare i beräkningar)
    margin_score = max(0.0, min(20.0, (gross*0.2) + (netm*0.2)))  # godtycklig mix
    leverage_pen = 0.0
    if d_e>0:
        if d_e <= 0.5: leverage_pen = 0
        elif d_e <= 1.0: leverage_pen = -5
        elif d_e <= 2.0: leverage_pen = -10
        else: leverage_pen = -15
    cash_score = 0.0
    if fcf>0:
        cash_score += 10
    if runway>8:
        cash_score += 5
    elif runway>4:
        cash_score += 2

    score = max(0.0, min(100.0, val_score + cagr_score + margin_score + leverage_pen + cash_score))
    return round(score,1)

def _score_dividend(row: pd.Series) -> float:
    """
    Utdelningsscore 0–100: yield, FCF-täckning, stabilitet (D/E låg, current/quick), marginaler, P/S relativt snitt.
    """
    dy   = float(row.get("Direktavkastning (%)", 0.0))
    fcf  = float(row.get("FCF (TTM)", 0.0))
    div_rate = float(row.get("Årlig utdelning", 0.0))
    price    = float(row.get("Aktuell kurs", 0.0))
    d_e  = float(row.get("Skuldsättning D/E", 0.0))
    cr   = float(row.get("Current ratio", 0.0))
    qr   = float(row.get("Quick ratio", 0.0))
    gross= float(row.get("Bruttomarginal (%)", 0.0))
    ps_now = float(row.get("P/S", 0.0))
    ps_avg = _ps_avg_from_quarters(row) or float(row.get("P/S-snitt", 0.0))

    yield_score = min(35.0, dy*3.5)  # ~10% yield -> 35p
    cover_score = 0.0
    if price>0 and div_rate>0:
        annual_cash = div_rate  # per aktie i källvaluta
        # approximera FCF-täckning: positiv fcf premieras kraftigt
        if fcf > 0:
            cover_score = 25.0
        elif fcf == 0:
            cover_score = 5.0
        else:
            cover_score = 0.0
    safety_score = 0.0
    if d_e <= 0.5: safety_score += 15
    elif d_e <= 1.0: safety_score += 8
    if cr >= 1.5: safety_score += 7
    elif cr >= 1.0: safety_score += 3
    if qr >= 1.0: safety_score += 3

    margin_score = max(0.0, min(10.0, (gross*0.1)))  # 50% -> 5p etc

    val_score = 0.0
    if ps_avg>0 and ps_now>0:
        ratio = ps_now/ps_avg
        if ratio <= 0.8:   val_score = 10
        elif ratio <= 1.0: val_score = 7
        elif ratio <= 1.2: val_score = 3

    score = max(0.0, min(100.0, yield_score + cover_score + safety_score + margin_score + val_score))
    return round(score,1)

def _label_by_score(score: float) -> str:
    if score >= 85: return "Mycket bra (Köp)"
    if score >= 70: return "Bra (Köp/Öka)"
    if score >= 55: return "Okej (Behåll)"
    if score >= 40: return "Övervärderad (Trimma)"
    return "Sälj/Undvik"

# --- Investeringsförslag -----------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Läge: Tillväxt/Utdelning
    mode = st.radio("Fokus", ["Tillväxt", "Utdelning"], horizontal=True)

    # Filter: sektor + storleksklass
    sectors = sorted([s for s in df.get("Sektor", pd.Series([])).astype(str).unique() if s])
    sector_sel = st.multiselect("Filtrera sektor(er)", sectors, default=[])
    cap_sel = st.multiselect("Filtrera storleksklass(er)", ["Mega/Large","Large","Mid","Small","Micro/Nano"], default=[])

    # Kapital & riktkursval
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas för potential?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    # Bas: endast bolag med pris & vald riktkurs
    base = df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Risklabel/cap bucket + filter
    if "_marketCap_raw" not in base.columns:
        base["_marketCap_raw"] = 0.0
    base["Cap bucket"] = base["_marketCap_raw"].apply(_cap_bucket)
    if sector_sel:
        base = base[base["Sektor"].isin(sector_sel)]
    if cap_sel:
        base = base[base["Cap bucket"].isin(cap_sel)]

    if base.empty:
        st.warning("Inga bolag efter filtrering.")
        return

    # Score & potential
    if mode == "Tillväxt":
        base["Score"] = base.apply(_score_growth, axis=1)
    else:
        base["Score"] = base.apply(_score_dividend, axis=1)

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["P/S-snitt (4q)"] = base.apply(_ps_avg_from_quarters, axis=1)

    # Sortering: primärt score, sekundärt potential
    base = base.sort_values(by=["Score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # Robust bläddring
    key_ns = "forslag"
    if f"{key_ns}_idx" not in st.session_state:
        st.session_state[f"{key_ns}_idx"] = 0
    st.session_state[f"{key_ns}_idx"] = min(st.session_state[f"{key_ns}_idx"], len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state[f"{key_ns}_idx"] = max(0, st.session_state[f"{key_ns}_idx"] - 1)
    with col_mid:
        i_now = st.session_state[f"{key_ns}_idx"]
        st.write(f"Förslag {i_now+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state[f"{key_ns}_idx"] = min(len(base)-1, st.session_state[f"{key_ns}_idx"] + 1)

    rad = base.iloc[st.session_state[f"{key_ns}_idx"]]

    # Växelkurs och köpberäkning
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = float(rad["Aktuell kurs"]) * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    # Market cap & etiketter
    mcap_raw = float(rad.get("_marketCap_raw", 0.0))
    mcap_txt = _format_mcap(mcap_raw, rad["Valuta"])
    risk = _risk_label(mcap_raw, rad["Valuta"])

    # P/S-info
    ps_now = float(rad.get("P/S", 0.0))
    ps_avg = float(rad.get("P/S-snitt (4q)", 0.0))
    ps_avg = ps_avg if ps_avg>0 else float(rad.get("P/S-snitt", 0.0))
    ps_avg = round(ps_avg, 2) if ps_avg else 0.0

    # UI
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']}) — {risk}")
    st.markdown(
        f"- **Aktuell kurs:** {rad['Aktuell kurs']:.2f} {rad['Valuta']}  "
        f"- **Riktkurs ({riktkurs_val.lower()}):** {rad[riktkurs_val]:.2f} {rad['Valuta']}  "
        f"- **Uppsida:** {rad['Potential (%)']:.1f}%"
    )
    st.markdown(
        f"- **Market Cap (nu):** {mcap_txt}  "
        f"- **P/S (nu):** {ps_now:.2f}  "
        f"- **P/S-snitt (4 senaste):** {ps_avg:.2f}"
    )
    st.markdown(
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st  "
        f"- **Score ({mode}):** {rad['Score']:.1f} → **{_label_by_score(rad['Score'])}**"
    )

    with st.expander("Detaljer & nyckeltal"):
        # MCAP Q1–Q4 om de finns
        m1 = rad.get("MCAP Q1", 0.0); m2 = rad.get("MCAP Q2", 0.0)
        m3 = rad.get("MCAP Q3", 0.0); m4 = rad.get("MCAP Q4", 0.0)
        mcaps = []
        if m1: mcaps.append(("MCAP Q1", _format_mcap(m1, rad["Valuta"])))
        if m2: mcaps.append(("MCAP Q2", _format_mcap(m2, rad["Valuta"])))
        if m3: mcaps.append(("MCAP Q3", _format_mcap(m3, rad["Valuta"])))
        if m4: mcaps.append(("MCAP Q4", _format_mcap(m4, rad["Valuta"])))

        lines = []
        if mcaps:
            lines.append("**Historisk MCAP (TTM-ändar):** " + ", ".join([f"{k}: {v}" for (k,v) in mcaps]))
        # ytterligare nyckeltal om tillgängligt
        add_keys = [
            ("Sektor", "Sektor"),
            ("Bruttomarginal (%)","Bruttomarginal (%)"),
            ("Nettomarginal (%)","Nettomarginal (%)"),
            ("Skuldsättning D/E","Skuldsättning D/E"),
            ("Current ratio","Current ratio"),
            ("Quick ratio","Quick ratio"),
            ("Totalt kassa","Totalt kassa"),
            ("OCF (TTM)","OCF (TTM)"),
            ("CapEx (TTM)","CapEx (TTM)"),
            ("FCF (TTM)","FCF (TTM)"),
            ("Runway (kvartal)","Runway (kvartal)"),
            ("Direktavkastning (%)","Direktavkastning (%)"),
            ("EPS (TTM)","EPS (TTM)"),
            ("P/E (TTM)","P/E (TTM)"),
        ]
        for label, key in add_keys:
            val = rad.get(key, None)
            if val not in (None, "", 0, 0.0):
                if "kassa" in label.lower() or "ocf" in label.lower() or "capex" in label.lower() or "fcf" in label.lower():
                    # belopp – formatera i valuta
                    lines.append(f"- **{label}:** {_format_mcap(float(val), rad['Valuta'])}")
                elif "%" in label:
                    lines.append(f"- **{label}:** {float(val):.2f} %")
                else:
                    lines.append(f"- **{label}:** {float(val):.2f}")
        if not lines:
            st.write("–")
        else:
            st.markdown("\n".join(lines))

# --- Portföljvy & Säljvakt ---------------------------------------------------

def _value_sek(row, user_rates: dict) -> float:
    vx = hamta_valutakurs(row.get("Valuta",""), user_rates)
    return float(row.get("Antal aktier",0.0)) * float(row.get("Aktuell kurs",0.0)) * vx

def _ps_ratio_vs_avg(row) -> float:
    ps_now = float(row.get("P/S",0.0))
    ps_avg = _ps_avg_from_quarters(row) or float(row.get("P/S-snitt",0.0))
    return ps_now/ps_avg if (ps_now>0 and ps_avg>0) else 0.0

def _overvaluation_flag(row, mode: str) -> str:
    """Returnerar etikett vid övervärdering (för säljvakt)."""
    ratio = _ps_ratio_vs_avg(row)
    pot   = 0.0
    try:
        pot = (float(row.get("Riktkurs om 1 år",0))-float(row.get("Aktuell kurs",0))) / max(1e-9,float(row.get("Aktuell kurs",0))) * 100.0
    except:
        pass

    if ratio == 0.0:
        return ""
    # Aggressiva trösklar för varning
    if ratio >= 1.5 and pot < 5:
        return "Sälj"
    if ratio >= 1.3 and pot < 10:
        return "Trimma"
    return ""

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")

    # Försäkra kolumner
    if "GAV (SEK)" not in df.columns:
        df["GAV (SEK)"] = 0.0
    if "Sektor" not in df.columns:
        df["Sektor"] = ""

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Värde (SEK)"] = port.apply(lambda r: _value_sek(r, user_rates), axis=1)
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)

    # Avkastning mot GAV (SEK)
    port["GAV (SEK)"] = pd.to_numeric(port["GAV (SEK)"], errors="coerce").fillna(0.0)
    port["Inköpsvärde (SEK)"] = port["GAV (SEK)"] * port["Antal aktier"]
    port["Orealiserad P/L (SEK)"] = port["Värde (SEK)"] - port["Inköpsvärde (SEK)"]
    port["P/L (%)"] = np.where(port["Inköpsvärde (SEK)"]>0, port["Orealiserad P/L (SEK)"] / port["Inköpsvärde (SEK)"] * 100.0, 0.0).round(2)

    # Sektorvikt
    sektor_agg = port.groupby("Sektor", dropna=False)["Värde (SEK)"].sum().reset_index()
    sektor_agg["Andel (%)"] = np.where(total_värde>0, sektor_agg["Värde (SEK)"]/total_värde*100.0, 0.0).round(2)

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2):,.2f} SEK".replace(",", " "))
    st.dataframe(
        port[["Ticker","Bolagsnamn","Sektor","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","GAV (SEK)","Inköpsvärde (SEK)","Orealiserad P/L (SEK)","P/L (%)"]],
        use_container_width=True, hide_index=True
    )

    st.subheader("Sektorbalans")
    st.dataframe(sektor_agg, use_container_width=True, hide_index=True)

    # --- Säljvakt (trim/sälj-förslag) ---
    st.subheader("🚨 Säljvakt (signaler)")
    mode = st.radio("Värderingsprofil för säljvakt", ["Tillväxt","Utdelning"], horizontal=True, key="sv_mode")

    # beräkna flagga
    port["Varning"] = port.apply(lambda r: _overvaluation_flag(r, mode), axis=1)

    # extra regler: övervikt i sektor, stor vinst vs GAV
    # Overweight flag (default > 25% av portföljen i en sektor)
    sektor_over = set(sektor_agg[sektor_agg["Andel (%)"] > 25.0]["Sektor"].tolist())
    port["Övervikt sektor?"] = port["Sektor"].apply(lambda s: "Ja" if s in sektor_over else "Nej")

    # Stor vinst: > 80% upp mot GAV
    port["Stor vinst?"] = np.where(port["P/L (%)"] > 80.0, "Ja", "Nej")

    # Prioritera varningar: Sälj > Trimma; inom samma etikett, störst andel först
    candidates = port[(port["Varning"] != "") | (port["Övervikt sektor?"]=="Ja") | (port["Stor vinst?"]=="Ja")].copy()
    if candidates.empty:
        st.success("Inga tydliga sälj-/trim-signaler just nu.")
        return

    candidates["prio"] = candidates["Varning"].map({"Sälj":2, "Trimma":1}).fillna(0)
    candidates = candidates.sort_values(by=["prio","Andel (%)","P/L (%)"], ascending=[False,False,False])

    # Rekommenderad trim-storlek (helt heuristiskt):
    # Baseras på etikett, sektorövervikt och andel i portföljen
    def _trim_size_pct(row):
        base = 0.0
        if row["Varning"] == "Sälj":
            base = 30.0
        elif row["Varning"] == "Trimma":
            base = 15.0
        # öka om sektorövervikt
        if row["Övervikt sektor?"] == "Ja":
            base += 10.0
        # öka lite om väldigt stor andel
        if row["Andel (%)"] > 15:
            base += 5.0
        return min(50.0, base)  # max 50% trim

    candidates["Föreslagen trim (%)"] = candidates.apply(_trim_size_pct, axis=1).round(0)

    st.dataframe(
        candidates[["Ticker","Bolagsnamn","Sektor","Andel (%)","P/L (%)","Varning","Övervikt sektor?","Stor vinst?","Föreslagen trim (%)"]],
        use_container_width=True, hide_index=True
    )

# --- Lägg till / uppdatera bolag --------------------------------------------

def _ts_badge(ts: str, label: str) -> str:
    ts = (ts or "").strip()
    return f"**{label}:** {ts if ts else '–'}"

def _render_ts_section(row: pd.Series):
    # Övergripande
    left, right = st.columns(2)
    with left:
        st.markdown(_ts_badge(row.get("Senast manuellt uppdaterad",""), "🖐️ Manuell"), help="Senaste gång du manuellt ändrade något av kärnfälten.")
    with right:
        src = row.get("Senast uppdaterad källa","")
        text = f"{row.get('Senast auto-uppdaterad','')}"
        if src:
            text += f"  \n_källa: {src}_"
        st.markdown(f"**🤖 Auto:** {text if text.strip() else '–'}")

    # Per-fält TS
    st.caption("Tidsstämplar per fält")
    cols = st.columns(4)
    ts_map = {
        "Utestående aktier": "TS_Utestående aktier",
        "P/S": "TS_P/S",
        "P/S Q1": "TS_P/S Q1",
        "P/S Q2": "TS_P/S Q2",
        "P/S Q3": "TS_P/S Q3",
        "P/S Q4": "TS_P/S Q4",
        "Omsättning idag": "TS_Omsättning idag",
        "Omsättning nästa år": "TS_Omsättning nästa år",
    }
    items = list(ts_map.items())
    for i, (lbl, colname) in enumerate(items):
        with cols[i % 4]:
            st.markdown(_ts_badge(row.get(colname, ""), lbl))

def _apply_manual_ts_if_changed(df: pd.DataFrame, ticker: str, before: dict, after: dict):
    ridx = df.index[df["Ticker"].astype(str).str.upper()==ticker.upper()]
    if len(ridx)==0:
        return
    ridx = ridx[0]
    changed_any = False
    changed_fields = []
    for k in MANUELL_FALT_FOR_DATUM:
        if float(before.get(k,0.0)) != float(after.get(k,0.0)):
            changed_any = True
            changed_fields.append(k)
    if changed_any:
        _note_manual_update(df, ridx)
        for f in changed_fields:
            _stamp_ts_for_field(df, ridx, f)

def _update_one_price(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, dict]:
    """Hämta endast pris/valuta/namn/MCAP från Yahoo och skriv in om ändrat."""
    mask = (df["Ticker"].astype(str).str.upper()==ticker.upper())
    if not mask.any():
        return df, {"error": f"{ticker} hittades inte i tabellen."}
    ridx = df.index[mask][0]

    base = hamta_yahoo_fält(ticker)
    # Lägg till rå MCAP om möjligt
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        base["_marketCap_raw"] = float(info.get("marketCap") or 0.0)
    except Exception:
        pass

    chmap = {}
    changed = apply_auto_updates_to_row(
        df, ridx,
        {
            "Bolagsnamn": base.get("Bolagsnamn",""),
            "Valuta": base.get("Valuta",""),
            "Aktuell kurs": float(base.get("Aktuell kurs",0.0) or 0.0),
            "_marketCap_raw": float(base.get("_marketCap_raw",0.0) or 0.0),
        },
        source="Kurs (Yahoo)",
        changes_map={"_tmp": []}
    )
    if changed:
        _note_auto_update(df, ridx, "Kurs (Yahoo)")
        df = uppdatera_berakningar(df, las_sparade_valutakurser())
        spara_data(df)  # skriv säkert – INTE clear+skapa
        msg = "Kurs/valuta/namn uppdaterade."
    else:
        msg = "Ingen ändring på kurs/valuta/namn."

    return df, {"ok": msg}

def _update_one_full(df: pd.DataFrame, user_rates: dict, ticker: str) -> tuple[pd.DataFrame, dict]:
    """Full auto för en ticker med 1/1-progress."""
    mask = (df["Ticker"].astype(str).str.upper()==ticker.upper())
    if not mask.any():
        return df, {"error": f"{ticker} hittades inte i tabellen."}
    ridx = df.index[mask][0]

    prog = st.progress(0, text="0/1")
    status = st.empty()
    status.write(f"Uppdaterar 1/1: {ticker}")
    vals, debug = auto_fetch_for_ticker(ticker)
    prog.progress(0.6, text="0.6/1")

    changed = apply_auto_updates_to_row(df, ridx, vals, source="Auto (Enskild)", changes_map={"_tmp": []})
    df = uppdatera_berakningar(df, user_rates)
    prog.progress(1.0, text="1/1")

    if changed:
        spara_data(df)
        status.success("Klart! Ändringar sparade.")
        return df, {"changed": True, "debug": debug}
    else:
        status.info("Inga ändringar hittades vid auto-uppdatering.")
        return df, {"changed": False, "debug": debug}

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sort-val inkl. "Äldst först"
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # Etiketter + robust bläddring
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    key_ns = "edit"
    if f"{key_ns}_index" not in st.session_state:
        st.session_state[f"{key_ns}_index"] = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state[f"{key_ns}_index"], len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state[f"{key_ns}_index"] = max(0, st.session_state[f"{key_ns}_index"] - 1)
    with col_pos:
        st.write(f"Post {st.session_state[f'{key_ns}_index']}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state[f"{key_ns}_index"] = min(len(val_lista)-1, st.session_state[f"{key_ns}_index"] + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    # ========== FORM ==========
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav   = st.number_input("GAV (SEK) per aktie", value=float(bef.get("GAV (SEK)",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över manuella 0-värden):**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

        spar = st.form_submit_button("💾 Spara")

    # Efter form – TS-etiketter
    if not bef.empty:
        _render_ts_section(bef)

    # ======= Knappar: Kurs / Full auto för vald ticker =======
    if ticker:
        cols_btn = st.columns([1,1,2])
        with cols_btn[0]:
            if st.button("💹 Uppdatera kurs (Yahoo)"):
                df2, info = _update_one_price(df, ticker)
                if "error" in info:
                    st.error(info["error"])
                else:
                    st.success(info.get("ok","Uppdaterat."))
                return df2  # lämna funktionen så UI ritas om fräscht
        with cols_btn[1]:
            if st.button("🤖 Full auto för denna"):
                df2, info = _update_one_full(df, user_rates, ticker)
                if info.get("changed", False):
                    st.success("Data uppdaterade.")
                else:
                    st.info("Inga faktiska ändringar.")
                return df2

    # ======= Spara manuella ändringar =======
    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV (SEK)": gav,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Bestäm om manuell-ts ska sättas + vilka TS-fält som ska stämplas
        before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM} if not bef.empty else {f: 0.0 for f in MANUELL_FALT_FOR_DATUM}
        after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
        # Skriv in nya fält
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        _apply_manual_ts_if_changed(df, ticker, before, after)

        # Hämta basfält från Yahoo (pris/namn/valuta/utdelning/CAGR)
        data = hamta_yahoo_fält(ticker)
        ridx = df.index[df["Ticker"].astype(str).str.upper()==ticker.upper()][0]
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # Snabb-lista över äldst uppdaterade – topp 10
    st.markdown("### ⏱️ Äldst uppdaterade (alla spårade fält, topp 10)")
    work = add_oldest_ts_col(df.copy())
    topp = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"], ascending=[True, True]).head(10)

    visa_kol = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
              "TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in df.columns:
            visa_kol.append(k)
    visa_kol.append("_oldest_any_ts")

    st.dataframe(topp[visa_kol], use_container_width=True, hide_index=True)

    return df

# --- Extra nyckeltal från Yahoo (EV/EBITDA, marginaler, cashflow etc.) -------

def compute_extra_yahoo_metrics(ticker: str) -> dict:
    """
    Hämtar 'best effort' nyckeltal via Yahoo:
    EV, EBITDA (TTM), EV/EBITDA, bruttomarginal, nettomarginal, D/E,
    current/quick ratio, total cash, OCF/CapEx/FCF (TTM), runway (kvartal),
    dividend yield, EPS (TTM), P/E (TTM), sektor, market cap (raw).
    """
    out = {}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Market cap & sektor
        mc = float(info.get("marketCap") or 0.0)
        if mc > 0:
            out["_marketCap_raw"] = mc
        if info.get("sector"):
            out["Sektor"] = str(info.get("sector"))

        # EV/EBITDA
        ev = float(info.get("enterpriseValue") or 0.0)
        ebitda = float(info.get("ebitda") or 0.0)
        if ev > 0:
            out["EV"] = ev
        if ebitda != 0.0:
            out["EBITDA (TTM)"] = ebitda
        if ev > 0 and ebitda and abs(ebitda) > 1e-9:
            out["EV/EBITDA"] = ev / ebitda

        # Marginaler (andelar i Yahoo.info)
        gm = info.get("grossMargins", None)  # 0.65 -> 65%
        nm = info.get("profitMargins", None)
        if gm is not None:
            out["Bruttomarginal (%)"] = float(gm) * 100.0
        if nm is not None:
            out["Nettomarginal (%)"] = float(nm) * 100.0

        # Leverage & likviditet
        if info.get("debtToEquity") is not None:
            out["Skuldsättning D/E"] = float(info.get("debtToEquity"))
        if info.get("currentRatio") is not None:
            out["Current ratio"] = float(info.get("currentRatio"))
        if info.get("quickRatio") is not None:
            out["Quick ratio"] = float(info.get("quickRatio"))

        # Kassa & kassaflöde
        total_cash = float(info.get("totalCash") or 0.0)
        if total_cash:
            out["Totalt kassa"] = total_cash

        ocf = float(info.get("operatingCashflow") or 0.0)
        capex = float(info.get("capitalExpenditures") or 0.0)
        if ocf != 0.0:
            out["OCF (TTM)"] = ocf
        if capex != 0.0:
            out["CapEx (TTM)"] = capex
        if ocf or capex:
            fcf = ocf - capex
            out["FCF (TTM)"] = fcf
            # runway (kvartal)
            if fcf < 0 and total_cash > 0:
                quarterly_burn = abs(fcf) / 4.0
                out["Kassaförbrukning/kvartal"] = quarterly_burn
                out["Runway (kvartal)"] = total_cash / quarterly_burn if quarterly_burn > 0 else 0.0
            elif fcf >= 0 and total_cash > 0:
                out["Kassaförbrukning/kvartal"] = 0.0
                out["Runway (kvartal)"] = 999.0

        # Utdelning & värdering
        if info.get("dividendYield") is not None:
            out["Direktavkastning (%)"] = float(info.get("dividendYield") or 0.0) * 100.0
        if info.get("trailingEps") is not None:
            out["EPS (TTM)"] = float(info.get("trailingEps") or 0.0)
        if info.get("trailingPE") is not None:
            out["P/E (TTM)"] = float(info.get("trailingPE") or 0.0)

    except Exception:
        pass
    return out

# --- Sektorsmart scoring (överskriver tidigare versioner) --------------------

def _sector_adjustments(row: pd.Series, mode: str) -> dict:
    """
    Returnerar multiplikatorer/bonusar för komponenter beroende på sektor.
    Enkel heuristik som ger 'rätt riktning' utan att bli för dogmatisk.
    Nycklar: cagr_w, margin_w, val_w, leverage_pen_w, safety_w, yield_w, bonus
    """
    sector = (row.get("Sektor","") or "").lower()
    adj = dict(cagr_w=1.0, margin_w=1.0, val_w=1.0, leverage_pen_w=1.0, safety_w=1.0, yield_w=1.0, bonus=0.0)

    if mode == "Tillväxt":
        if "technology" in sector or "communication" in sector or "software" in sector or "semiconductor" in sector:
            adj["cagr_w"] = 1.25
            adj["margin_w"] = 1.15
            adj["leverage_pen_w"] = 0.85
            if float(row.get("Bruttomarginal (%)",0)) > 55:
                adj["bonus"] += 2.0
        elif "utilities" in sector or "real estate" in sector:
            adj["cagr_w"] = 0.8
            adj["margin_w"] = 0.9
            adj["val_w"] = 0.9
            adj["leverage_pen_w"] = 1.15
        elif "energy" in sector or "materials" in sector or "industrial" in sector:
            adj["val_w"] = 0.85  # P/S mindre informativt här
    else:  # Utdelning
        if "utilities" in sector or "consumer defensive" in sector or "real estate" in sector:
            adj["yield_w"] = 1.2
            adj["safety_w"] = 1.2
        if "technology" in sector:
            adj["yield_w"] = 0.8
            # FCF-täckning viktigare
            adj["safety_w"] = 1.1
    return adj

def _score_growth(row: pd.Series) -> float:
    ps_now = float(row.get("P/S", 0.0))
    ps_avg = _ps_avg_from_quarters(row) or float(row.get("P/S-snitt", 0.0))
    cagr = float(row.get("CAGR 5 år (%)", 0.0))
    gross = float(row.get("Bruttomarginal (%)", 0.0))
    netm  = float(row.get("Nettomarginal (%)", 0.0))
    d_e   = float(row.get("Skuldsättning D/E", 0.0))
    fcf   = float(row.get("FCF (TTM)", 0.0))
    runway= float(row.get("Runway (kvartal)", 0.0))

    # Bas-komponenter
    val_score = 0.0
    if ps_avg>0 and ps_now>0:
        ratio = ps_now/ps_avg
        if ratio <= 0.6:   val_score = 30
        elif ratio <= 0.8: val_score = 24
        elif ratio <= 1.0: val_score = 18
        elif ratio <= 1.2: val_score = 10
        else:              val_score = max(0, 10 - 10*(ratio-1.2))

    cagr_score = max(0.0, min(25.0, cagr*0.5))
    margin_score = max(0.0, min(20.0, (gross*0.2) + (netm*0.2)))
    leverage_pen = 0.0
    if d_e>0:
        if d_e <= 0.5: leverage_pen = 0
        elif d_e <= 1.0: leverage_pen = -5
        elif d_e <= 2.0: leverage_pen = -10
        else: leverage_pen = -15
    cash_score = 0.0
    if fcf>0: cash_score += 10
    if runway>8: cash_score += 5
    elif runway>4: cash_score += 2

    # Sektorsjusteringar
    adj = _sector_adjustments(row, "Tillväxt")
    score = (
        val_score*adj["val_w"] +
        cagr_score*adj["cagr_w"] +
        margin_score*adj["margin_w"] +
        leverage_pen*adj["leverage_pen_w"] +
        cash_score + adj["bonus"]
    )
    return round(max(0.0, min(100.0, score)), 1)

def _score_dividend(row: pd.Series) -> float:
    dy   = float(row.get("Direktavkastning (%)", 0.0))
    fcf  = float(row.get("FCF (TTM)", 0.0))
    d_e  = float(row.get("Skuldsättning D/E", 0.0))
    cr   = float(row.get("Current ratio", 0.0))
    qr   = float(row.get("Quick ratio", 0.0))
    gross= float(row.get("Bruttomarginal (%)", 0.0))
    ps_now = float(row.get("P/S", 0.0))
    ps_avg = _ps_avg_from_quarters(row) or float(row.get("P/S-snitt", 0.0))

    yield_score = min(35.0, dy*3.5)
    cover_score = 25.0 if fcf>0 else (5.0 if fcf==0 else 0.0)
    safety_score = 0.0
    if d_e <= 0.5: safety_score += 15
    elif d_e <= 1.0: safety_score += 8
    if cr >= 1.5: safety_score += 7
    elif cr >= 1.0: safety_score += 3
    if qr >= 1.0: safety_score += 3
    margin_score = max(0.0, min(10.0, (gross*0.1)))
    val_score = 0.0
    if ps_avg>0 and ps_now>0:
        ratio = ps_now/ps_avg
        if ratio <= 0.8:   val_score = 10
        elif ratio <= 1.0: val_score = 7
        elif ratio <= 1.2: val_score = 3

    # Sektorsjusteringar
    adj = _sector_adjustments(row, "Utdelning")
    score = (
        yield_score*adj["yield_w"] +
        cover_score*adj["safety_w"] +  # täckning betraktas som "säkerhet"
        safety_score*adj["safety_w"] +
        margin_score +
        val_score*adj["val_w"] +
        adj["bonus"]
    )
    return round(max(0.0, min(100.0, score)), 1)

# --- Överskugga batch_update_slice för att injicera extra nyckeltal ----------

def batch_update_slice(df: pd.DataFrame, user_rates: dict, tickers: list, make_snapshot: bool = False):
    """
    Uppdaterar *just den slice* som ges (tickers) – nu även med extra Yahoo-nyckeltal.
    Visar 1/X-progress.
    """
    log = {"changed": {}, "misses": {}, "debug": []}
    total = len(tickers)
    if total == 0:
        st.info("Inget att köra i denna batch.")
        return df, log

    progress = st.progress(0)
    status = st.empty()

    any_changed = False
    for i, tkr in enumerate(tickers):
        status.write(f"Uppdaterar {i+1}/{total}: {tkr}")
        progress.progress((i+1)/total, text=f"{i+1}/{total}")
        try:
            mask = (df["Ticker"].astype(str).str.upper() == str(tkr).upper())
            if not mask.any():
                log["misses"][tkr] = ["Ticker saknas i tabellen"]
                continue
            row_idx = df.index[mask][0]

            # Basflöde
            new_vals, debug = auto_fetch_for_ticker(tkr)
            # Extra nyckeltal
            extra = compute_extra_yahoo_metrics(tkr)
            new_vals.update(extra)

            changed = apply_auto_updates_to_row(df, row_idx, new_vals, source="Auto (Batch)", changes_map=log["changed"])
            any_changed = any_changed or changed
            log["debug"].append({tkr: debug})
            st.session_state.batch_done.add(tkr)
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

    df = uppdatera_berakningar(df, user_rates)
    if any_changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.success("Klart! Ändringar sparade för denna batch.")
    else:
        st.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")
    return df, log

# --- Sidopanel: Valutakurser + Batch-kö -------------------------------------

def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # Initera state om saknas
    if "rate_usd" not in st.session_state: st.session_state.rate_usd = STANDARD_VALUTAKURSER["USD"]
    if "rate_nok" not in st.session_state: st.session_state.rate_nok = STANDARD_VALUTAKURSER["NOK"]
    if "rate_cad" not in st.session_state: st.session_state.rate_cad = STANDARD_VALUTAKURSER["CAD"]
    if "rate_eur" not in st.session_state: st.session_state.rate_eur = STANDARD_VALUTAKURSER["EUR"]

    saved_rates = las_sparade_valutakurser()
    if saved_rates:
        st.session_state.rate_usd = float(saved_rates.get("USD", st.session_state.rate_usd))
        st.session_state.rate_nok = float(saved_rates.get("NOK", st.session_state.rate_nok))
        st.session_state.rate_cad = float(saved_rates.get("CAD", st.session_state.rate_cad))
        st.session_state.rate_eur = float(saved_rates.get("EUR", st.session_state.rate_eur))

    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd), step=0.01, format="%.4f", key="usd_input")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok), step=0.01, format="%.4f", key="nok_input")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad), step=0.01, format="%.4f", key="cad_input")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur), step=0.01, format="%.4f", key="eur_input")

    # Auto-hämtning
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.session_state.rate_usd = float(auto_rates.get("USD", usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", eur))
        st.sidebar.success(f"Valutakurser hämtade (källa: {provider}).")
        if misses:
            st.sidebar.warning("Kunde inte hämta:\n- " + "\n- ".join(misses))
        # Sätt widgetvärden och rerun så de syns direkt
        st.session_state.usd_input = st.session_state.rate_usd
        st.session_state.nok_input = st.session_state.rate_nok
        st.session_state.cad_input = st.session_state.rate_cad
        st.session_state.eur_input = st.session_state.rate_eur
        st.rerun()

    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear()
            st.rerun()

    st.sidebar.markdown("---")
    return user_rates

def _sidebar_batch_controls(df: pd.DataFrame):
    st.sidebar.subheader("🛠️ Batch-uppdatering")
    mode = st.sidebar.selectbox("Ordning", ["Äldst först","A–Ö"])
    size = st.sidebar.number_input("Batch-storlek", min_value=1, max_value=100, value=10, step=1)

    if st.sidebar.button("📦 Förbered kö"):
        prepare_batch_queue(df, mode, int(size))
        st.sidebar.success("Kö förberedd.")

    if st.sidebar.button("▶️ Kör nästa batch"):
        sl = next_batch_slice()
        if not sl:
            st.sidebar.info("Inget kvar i kö-slicen. Klicka 'Förbered kö' eller 'Återställ kö'.")
        else:
            st.sidebar.write(f"Kör: {', '.join(sl)}")
            make_snapshot = st.sidebar.checkbox("Snapshot före skrivning", value=True, key="batch_snap_ck")
            df2, log = batch_update_slice(st.session_state.get("_df_ref", df), las_sparade_valutakurser(), sl, make_snapshot=make_snapshot)
            st.session_state["_df_ref"] = df2
            st.session_state["last_batch_log"] = log

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("⏭️ Hoppa fram (öka pos)"):
            advance_batch_position()
            st.sidebar.info(f"Pos = {st.session_state.get('batch_pos',0)}")
    with c2:
        if st.button("♻️ Återställ kö"):
            reset_batch_queue()
            st.sidebar.success("Batch-kö återställd.")

    # Visning av status
    all_ticks = st.session_state.get("batch_all", [])
    pos = int(st.session_state.get("batch_pos", 0))
    done = len(st.session_state.get("batch_done", set()))
    if all_ticks:
        st.sidebar.caption(f"Kö-status: pos {pos} av {len(all_ticks)}, klara {done} st.")

# --- MAIN --------------------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: kurser + batchkontroller
    user_rates = _sidebar_rates()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # Läs data
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    # Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Håll en ref i session (för batch som skriver under körning)
    st.session_state["_df_ref"] = df

    # Batch-kontroller i sidopanel (använder _df_ref vid körning)
    _sidebar_batch_controls(df)

    st.sidebar.markdown("---")
    # Snabb-knapp: Auto-uppdatera ALLA (tung) – låt finnas kvar
    make_snapshot = st.sidebar.checkbox("Snapshot före skrivning (vid 'Auto alla')", value=True)
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → Finnhub → FMP)"):
        df_all, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot)
        st.session_state["_df_ref"] = df_all
        st.session_state["last_auto_log"] = log

    # Navigation
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Kontroll":
        kontrollvy(st.session_state["_df_ref"])
    elif meny == "Analys":
        analysvy(st.session_state["_df_ref"], user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates)
        st.session_state["_df_ref"] = df2
    elif meny == "Investeringsförslag":
        df_calc = uppdatera_berakningar(st.session_state["_df_ref"], user_rates)
        st.session_state["_df_ref"] = df_calc
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        df_calc = uppdatera_berakningar(st.session_state["_df_ref"], user_rates)
        st.session_state["_df_ref"] = df_calc
        visa_portfolj(df_calc, user_rates)

if __name__ == "__main__":
    main()
