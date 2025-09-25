# app.py — Del 1/8
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
from typing import Optional, Tuple

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

# --- Kolumnschema & tidsstämplar --------------------------------------------

# Spårade fält → respektive TS-kolumn (uppdateras när fältet ändras automatiskt eller manuellt)
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

# Slutlig kolumnlista i databasen (utökad med kassaflöden, BS/IS, mcap-historik mm)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn",
    "Utestående aktier",
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier",
    "Valuta", "Finansiell valuta",
    "Årlig utdelning", "Aktuell kurs",
    "MCap (nu)",
    "CAGR 5 år (%)", "P/S-snitt",
    "Sektor",
    # Kassaflöden/kostnader (Q)
    "Operativt kassaflöde (Q)", "CapEx (Q)", "Fritt kassaflöde (Q)",
    "Operating Expense (Q)", "FoU (Q)", "SG&A (Q)",
    # Balans/IS (TTM) + övrigt
    "Kassa", "Total skuld", "EBITDA (TTM)", "Räntekostnad (TTM)",
    "Current assets", "Current liabilities", "Beta",
    "Burn rate (Q)", "Runway (kvartal)",
    # MCAP-historik för fyra TTM-fönster
    "MCap Datum Q1","MCap Q1","MCap Datum Q2","MCap Q2",
    "MCap Datum Q3","MCap Q3","MCap Datum Q4","MCap Q4",
    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    for kol in FINAL_COLS:
        if kol not in df.columns:
            # Numeriska fält
            if any(x in kol for x in [
                "kurs","Omsättning","P/S","utdelning","CAGR","Antal","Riktkurs","aktier","snitt",
                "MCap","Kassa","skuld","EBITDA","Räntekostnad","assets","liabilities","Beta",
                "kassaflöde","CapEx","Fritt","Burn","Runway"
            ]) and not kol.startswith("TS_") and "Datum" not in kol:
                df[kol] = 0.0
            # Datum/strängfält
            elif kol.startswith("TS_") or "Datum" in kol or kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Ticker","Bolagsnamn","Valuta","Finansiell valuta","Sektor"):
                df[kol] = ""
            else:
                df[kol] = 0.0
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
        "Utestående aktier", "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "MCap (nu)",
        "Operativt kassaflöde (Q)", "CapEx (Q)", "Fritt kassaflöde (Q)",
        "Operating Expense (Q)", "FoU (Q)", "SG&A (Q)",
        "Kassa", "Total skuld", "EBITDA (TTM)", "Räntekostnad (TTM)",
        "Current assets", "Current liabilities", "Beta",
        "Burn rate (Q)", "Runway (kvartal)",
        "MCap Q1","MCap Q2","MCap Q3","MCap Q4"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    str_cols = ["Ticker","Bolagsnamn","Valuta","Finansiell valuta","Sektor",
                "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
                "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]
    for c in str_cols:
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

# app.py — Del 2/8
# --- Yahoo-hjälpare & beräkningar & robust skrivning ------------------------

def _yfi_get(tkr: yf.Ticker, *keys):
    """Säker hämtning ur yfinance.info med fallback på hist."""
    try:
        info = tkr.info or {}
        for k in keys:
            if k in info and info[k] is not None:
                return info[k]
    except Exception:
        pass
    return None

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
    """
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Finansiell valuta, MCap, P/S (Yahoo), Sektor, Beta,
    Årlig utdelning, CAGR 5 år (%).
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Finansiell valuta": "",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "P/S (Yahoo)": 0.0,
        "MCap (nu)": 0.0,
        "Sektor": "",
        "Beta": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        # Valutor
        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()
        fin_ccy = info.get("financialCurrency", None)
        if fin_ccy:
            out["Finansiell valuta"] = str(fin_ccy).upper()

        # Namn, sektor
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)
        if info.get("sector"):
            out["Sektor"] = str(info.get("sector"))

        # Dividend, beta
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass
        beta = info.get("beta") or info.get("beta3Year") or info.get("beta5Year")
        try:
            if beta is not None:
                out["Beta"] = float(beta)
        except Exception:
            pass

        # MCap & P/S (Yahoo)
        mcap = info.get("marketCap")
        try:
            if mcap is not None:
                out["MCap (nu)"] = float(mcap)
        except Exception:
            pass
        ps_y = info.get("priceToSalesTrailing12Months") or info.get("priceToSalesTTM")
        try:
            if ps_y is not None and float(ps_y) > 0:
                out["P/S (Yahoo)"] = float(ps_y)
        except Exception:
            pass

        # CAGR
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

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, changes_map: dict, force_ts: bool = False) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde.
    Uppdaterar TS_ för spårade fält när värde ändras.
    Sätter 'Senast auto-uppdaterad' + källa om något ändrats ELLER om force_ts=True.
    Returnerar True om något fält faktiskt ändrades.
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

    if changed_fields or force_ts:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return bool(changed_fields)
    return False

# app.py — Del 3/8
# --- SEC (US + FPI/IFRS), kvartalsdata, TTM, MCAP-historik -------------------

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
    # ~9–10MB JSON med ALLA tickers → cachea 24h
    j, sc = _sec_get("https://www.sec.gov/files/company_tickers.json")
    if not isinstance(j, dict):
        return {}
    out = {}
    # format: {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
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

# ---------- datumhjälpare ----------------------------------------------------
from datetime import datetime as _dt, timedelta as _td, date as _date

def _parse_iso(d: str):
    try:
        return _dt.fromisoformat(str(d).replace("Z", "+00:00")).date()
    except Exception:
        try:
            return _dt.strptime(str(d), "%Y-%m-%d").date()
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
            return (d2 - d1).days <= 2  # ≈ instant
        except Exception:
            return False
    return False

# ---------- shares (instant; multi-class) ------------------------------------
def _collect_share_entries(facts: dict) -> list:
    """
    Hämtar alla 'instant' aktieposter från dei/us-gaap/ifrs-full (unit='shares' m.fl.).
    Returnerar [{"end": date, "val": float, "frame": "...", "form": "..."}].
    """
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
    """
    Summerar multi-class per senaste 'end' (instant). Om flera olika 'end' finns,
    välj det senaste datumet och summera alla frames för det datumet.
    Returnerar aktier (styck), 0 om ej hittat.
    """
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

# ---------- FX för att matcha prisvaluta -------------------------------------
@st.cache_data(show_spinner=False, ttl=21600)
def _fx_rate_cached(base: str, quote: str) -> float:
    """Enkel FX (dagens) via Frankfurter → exchangerate.host fallback."""
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

# ---------- kvartalsintäkter (SEC) + Q4-backfill -----------------------------
def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    Om Q4 saknas för ett visst år försöker vi backfilla Q4 = Årsintäkt - Q1-Q3.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A", "6-K", "6-K/A")}),
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

    rows = []
    unit_code = None

    # 1) Läs kvartal ur 10-Q/6-K
    for taxo, cfg in taxos:
        gaap = (facts.get("facts") or {}).get(taxo, {})
        got = False
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})
            for uc in prefer_units:
                arr = units.get(uc)
                if not isinstance(arr, list):
                    continue
                tmp = []
                for it in arr:
                    form = (it.get("form") or "").upper()
                    if not any(f in form for f in cfg["forms"]):
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
                        tmp.append((end, v, start.year, end.year))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # deduplicera per end
                ded = {}
                for end, v, sy, ey in tmp:
                    ded[end] = v
                out = sorted(ded.items(), key=lambda t: t[0], reverse=True)
                rows = out[:max_quarters]
                unit_code = uc
                got = True
                break
            if got:
                break
        if got:
            break

    # 2) Q4-backfill om lucka (använd årsintäkt från 10-K/20-F)
    #    Hitta per-år grupper av Q1-Q3 och fyll Q4 om årssumma finns.
    if rows:
        # Bygg karta: år -> lista[(end, value)]
        per_year = {}
        for end, val in rows:
            y = end.year
            per_year.setdefault(y, []).append((end, val))
        # Hämta årsintäkter
        ann_val, ann_unit = _sec_annual_revenue_with_unit(facts)
        if ann_val and ann_unit and (unit_code is None or ann_unit == unit_code):
            # För varje år där vi har exakt 3 kvartal – försök backfilla Q4 = År - (Q1+Q2+Q3)
            add_rows = []
            for y, lst in per_year.items():
                if len(lst) == 3 and y in ann_val:
                    qsum = sum(v for (_, v) in lst)
                    ysum = float(ann_val[y])
                    q4 = ysum - qsum
                    # Enddate för Q4 är normalt fiskalt årsslut. Vi gissar end = max(end i året)
                    end_guess = max(e for (e, _) in lst) + _td(days=90)  # approx.
                    # clamp: rimlig > 0
                    if q4 > 0:
                        add_rows.append((end_guess, q4))
            # slå ihop och sortera om
            if add_rows:
                pool = dict(rows)
                for e, v in add_rows:
                    # bara lägg till om vi inte redan har en datapunkt nära detta
                    if e not in pool:
                        pool[e] = v
                rows = sorted(pool.items(), key=lambda t: t[0], reverse=True)[:max_quarters]

    return rows, unit_code

def _sec_annual_revenue_with_unit(facts: dict):
    """
    Returnerar ({år:intäkter}, unit_code) för US-GAAP/IFRS årsintäkter (10-K/20-F).
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-K","10-K/A","20-F","20-F/A")}),
        ("ifrs-full", {"forms": ("20-F","20-F/A","10-K","10-K/A")}),
    ]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
    ]
    prefer_units = ("USD","CAD","EUR","GBP")
    best = {}
    best_unit = None
    for taxo, cfg in taxos:
        gaap = (facts.get("facts") or {}).get(taxo, {})
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})
            for uc in prefer_units:
                arr = units.get(uc)
                if not isinstance(arr, list):
                    continue
                tmp = {}
                for it in arr:
                    form = (it.get("form") or "").upper()
                    if not any(f in form for f in cfg["forms"]):
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
                    # årsperiod ≈ 350–370 dagar
                    if dur is None or dur < 330 or dur > 400:
                        continue
                    try:
                        v = float(val)
                        tmp[end.year] = v
                    except Exception:
                        pass
                if tmp:
                    # Välj det mest kompletta
                    if len(tmp) > len(best):
                        best = tmp
                        best_unit = uc
    return best, best_unit

# ---------- Yahoo pris & implied shares & TTM-fönster ------------------------
def _yahoo_prices_for_dates(ticker: str, dates: list) -> dict:
    """
    Hämtar dagliga priser i ett fönster som täcker alla 'dates' och returnerar
    'Close' på eller närmast FÖRE respektive datum.
    """
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
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), osv.
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
        mcap = _yfi_get(t, "market_cap", "marketCap")
    if price is None:
        price = _yfi_get(t, "last_price", "regularMarketPrice")
    try:
        mcap = float(mcap or 0.0); price = float(price or 0.0)
    except Exception:
        return 0.0
    if mcap > 0 and price > 0:
        return mcap / price
    return 0.0

# ---------- Global Yahoo fallback (icke-SEC: .TO/.V/.CN + EU/Norden) ---------
def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yfi_quarterly_revenues(t: yf.Ticker) -> list:
    """
    Försöker läsa kvartalsintäkter från Yahoo.
    Returnerar [(period_end_date, value), ...] sorterat nyast→äldst.
    """
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

    # 2) fallback: income_stmt quarterly via v1-api (kan vara tomt)
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

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik och MCAP-historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/mm
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","MCap (nu)","P/S (Yahoo)","Sektor","Beta"):
        if y.get(k): out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = out.get("MCap (nu)") or info.get("marketCap") or 0.0
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
    if mcap > 0:
        out["MCap (nu)"] = mcap

    # P/S (TTM) nu
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historisk) + MCAP-historik via historiska priser
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares * float(p)
                    out[f"P/S Q{idx}"] = (mcap_hist / ttm_rev_px)
                    out[f"MCap Datum Q{idx}"] = d_end.isoformat()
                    out[f"MCap Q{idx}"] = float(mcap_hist)

    return out

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn/mcap/ps_yahoo/sektor/beta från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik
    med robust Q4-backfill. Sparar även MCAP-historik för Q1–Q4.
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
    for k in ("Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","MCap (nu)","P/S (Yahoo)","Sektor","Beta"):
        if y.get(k): out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()
    price_now = float(out.get("Aktuell kurs") or 0.0)

    # Shares: implied → fallback SEC robust
    implied = _implied_shares_from_yahoo(ticker, price=out.get("Aktuell kurs"), mcap=out.get("MCap (nu)"))
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
    mcap_now = float(out.get("MCap (nu)") or 0.0)
    if mcap_now <= 0 and price_now > 0 and shares_used > 0:
        mcap_now = price_now * shares_used
        out["MCap (nu)"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering (med Q4-backfill)
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=24)
    if not q_rows or not rev_unit:
        return out
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    q_rows_px = [(d, v * conv) for (d, v) in q_rows]
    ttm_list = _ttm_windows(q_rows_px, need=4)

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik + MCAP-historik via historiska priser
    if shares_used > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCap Datum Q{idx}"] = d_end.isoformat()
                    out[f"MCap Q{idx}"] = float(mcap_hist)

    return out

# app.py — Del 4/8
# --- FMP, Finnhub, Yahoo CF/BS, auto-pipelines --------------------------------

# =============== FMP =========================================================
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.0))      # skonsam default
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20)) # paus efter 429

def _fmp_get(path: str, params=None, stable: bool = True):
    """
    Throttlad GET med enkel backoff + 'circuit breaker' vid 429.
    Returnerar (json, statuscode). Anropa med path t.ex. 'api/v3/quote/AAPL'.
    """
    block_until = st.session_state.get("fmp_block_until")
    if block_until and _ts_datetime() < block_until:
        return None, 429

    params = (params or {}).copy()
    if FMP_KEY:
        params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}/{path}"

    delays = [0.0, 1.0, 2.2]
    last_sc = 0
    last_json = None

    for extra_sleep in delays:
        try:
            if FMP_CALL_DELAY > 0:
                time.sleep(FMP_CALL_DELAY)
            r = requests.get(url, params=params, timeout=25)
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
    """Validera symbol via quote-short, annars search. Faller tillbaka till upper()."""
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
    """
    Lätt variant: quote (pris/mcap/shares) + ratios-ttm (P/S).
    Namn/valuta fylls via Yahoo längre ned om saknas.
    """
    out = {"_debug": {}, "_symbol": _fmp_pick_symbol(yahoo_ticker)}
    sym = out["_symbol"]

    # pris, marketCap, sharesOutstanding
    q, sc_q = _fmp_get(f"api/v3/quote/{sym}", stable=False)
    out["_debug"]["quote_sc"] = sc_q
    if isinstance(q, list) and q:
        q0 = q[0]
        if q0.get("price") is not None:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
        if q0.get("marketCap") is not None:
            try: out["MCap (nu)"] = float(q0["marketCap"])
            except: pass
        if q0.get("sharesOutstanding") is not None:
            try: out["Utestående aktier"] = float(q0["sharesOutstanding"]) / 1e6
            except: pass

    # P/S TTM
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

@st.cache_data(show_spinner=False, ttl=1800)
def hamta_fmp_falt(yahoo_ticker: str) -> dict:
    """
    Fullare variant: försöker hämta namn/valuta/pris/shares, P/S (TTM, key-metrics, beräkning),
    P/S Q1–Q4 (ratios quarterly) samt analytikerestimat (om plan tillåter).
    """
    out = {"_debug": {}}
    sym = _fmp_pick_symbol(yahoo_ticker)
    out["_symbol"] = sym

    # Profile (namn/valuta/price/shares)
    prof, sc_prof = _fmp_get(f"api/v3/profile/{sym}", stable=False)
    out["_debug"]["profile_sc"] = sc_prof
    if isinstance(prof, list) and prof:
        p0 = prof[0]
        if p0.get("companyName"): out["Bolagsnamn"] = p0["companyName"]
        if p0.get("currency"):    out["Valuta"]     = str(p0["currency"]).upper()
        if p0.get("price") is not None:
            try: out["Aktuell kurs"] = float(p0["price"])
            except: pass
        if p0.get("sharesOutstanding"):
            try: out["Utestående aktier"] = float(p0["sharesOutstanding"]) / 1e6
            except: pass

    # Quote (pris + marketCap)
    qfull, sc_qfull = _fmp_get(f"api/v3/quote/{sym}", stable=False)
    out["_debug"]["quote_sc"] = sc_qfull
    market_cap = 0.0
    if isinstance(qfull, list) and qfull:
        q0 = qfull[0]
        if "price" in q0 and "Aktuell kurs" not in out:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
        if q0.get("marketCap") is not None:
            try: market_cap = float(q0["marketCap"])
            except: pass
    if market_cap > 0:
        out["MCap (nu)"] = market_cap

    # Shares fallback
    if "Utestående aktier" not in out:
        flo, sc_flo = _fmp_get(f"api/v4/shares_float/{sym}", stable=False)
        out["_debug"]["shares_float_sc"] = sc_flo
        if isinstance(flo, list):
            for it in flo:
                n = it.get("outstandingShares") or it.get("sharesOutstanding")
                if n:
                    try:
                        out["Utestående aktier"] = float(n) / 1e6
                        break
                    except:
                        pass

    # P/S via ratios-ttm
    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}", stable=False)
    out["_debug"]["ratios_ttm_sc"] = sc_rttm
    ps_from_ratios = None
    if isinstance(rttm, list) and rttm:
        try:
            v = rttm[0].get("priceToSalesTTM") or rttm[0].get("priceToSalesRatioTTM")
            if v is not None:
                ps_from_ratios = float(v)
        except Exception:
            pass
    if ps_from_ratios and ps_from_ratios > 0:
        out["P/S"] = ps_from_ratios
        out["_debug"]["ps_source"] = "ratios-ttm"

    # key-metrics-ttm
    if "P/S" not in out:
        kttm, sc_kttm = _fmp_get(f"api/v3/key-metrics-ttm/{sym}", stable=False)
        out["_debug"]["key_metrics_ttm_sc"] = sc_kttm
        if isinstance(kttm, list) and kttm:
            try:
                v = kttm[0].get("priceToSalesRatioTTM") or kttm[0].get("priceToSalesTTM")
                if v and float(v) > 0:
                    out["P/S"] = float(v)
                    out["_debug"]["ps_source"] = "key-metrics-ttm"
            except Exception:
                pass

    # P/S = marketCap / revenueTTM
    if "P/S" not in out and market_cap > 0:
        isttm, sc_isttm = _fmp_get(f"api/v3/income-statement-ttm/{sym}", stable=False)
        out["_debug"]["income_ttm_sc"] = sc_isttm
        revenue_ttm = 0.0
        if isinstance(isttm, list) and isttm:
            cand = istttm[0] if len(isttm) else {}
            for k in ("revenueTTM", "revenue"):
                if cand.get(k) is not None:
                    try:
                        revenue_ttm = float(cand[k]); break
                    except Exception:
                        pass
        if revenue_ttm > 0:
            try:
                ps_calc = market_cap / revenue_ttm
                if ps_calc > 0:
                    out["P/S"] = float(ps_calc)
                    out["_debug"]["ps_source"] = "calc(marketCap/revenueTTM)"
            except Exception:
                pass

    # P/S Q1–Q4 (FMP ratios-quarterly om tillgängligt)
    rq, sc_rq = _fmp_get(f"api/v3/ratios/{sym}", {"period": "quarter", "limit": 4}, stable=False)
    out["_debug"]["ratios_quarter_sc"] = sc_rq
    if isinstance(rq, list) and rq:
        for i, row in enumerate(rq[:4], start=1):
            ps = row.get("priceToSalesRatio")
            if ps is not None:
                try:
                    out[f"P/S Q{i}"] = float(ps)
                except:
                    pass

    return out

# =============== Finnhub (valfritt, estimat) =================================
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets. Hämtar annual revenue estimates:
    current FY + next FY (om finns). Returnerar i miljoner.
    """
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
        j = r.json() | {}
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

# =============== Yahoo CF/BS (kassaflöde/balans/IS) ==========================
def _safe_row_val(df: pd.DataFrame, key: str):
    try:
        if isinstance(df, pd.DataFrame) and not df.empty and key in df.index:
            ser = df.loc[key].dropna()
            if not ser.empty:
                # Senaste kolumnen
                return float(ser.iloc[0])
    except Exception:
        pass
    return 0.0

def _safe_row_ttm(df: pd.DataFrame, key: str):
    """Summera upp till senaste 4 kvartal för ett nyckeltal (kan vara negativt)."""
    try:
        if isinstance(df, pd.DataFrame) and not df.empty and key in df.index:
            ser = df.loc[key].dropna()
            vals = [float(x) for x in list(ser.values)[:4] if pd.notna(x)]
            if vals:
                return float(sum(vals))
    except Exception:
        pass
    return 0.0

def hamta_cashflow_balance_yahoo(ticker: str) -> dict:
    """
    Hämtar senaste kvartalets kassaflöden + TTM för EBITDA/ränta och balansposter.
    Returnerar fält som matchar FINAL_COLS och derivat (FCF, Burn, Runway).
    """
    out = {}
    try:
        t = yf.Ticker(ticker)

        # Kvartalsvisa statements
        q_cf = getattr(t, "quarterly_cashflow", None)
        q_bs = getattr(t, "quarterly_balance_sheet", None)
        q_is = getattr(t, "quarterly_income_stmt", None)

        # Operating CF, CapEx, FCF (Q)
        op_cf = _safe_row_val(q_cf, "Total Cash From Operating Activities")
        capex = _safe_row_val(q_cf, "Capital Expenditures")
        fcf = float(op_cf) + float(capex)  # capex brukar vara negativt
        out["Operativt kassaflöde (Q)"] = float(op_cf)
        out["CapEx (Q)"] = float(capex)
        out["Fritt kassaflöde (Q)"] = float(fcf)

        # Opex / FoU / SG&A (Q)
        opex = _safe_row_val(q_is, "Operating Expense")
        rdn  = _safe_row_val(q_is, "Research Development")
        sga  = _safe_row_val(q_is, "Selling General Administrative")
        out["Operating Expense (Q)"] = float(opex)
        out["FoU (Q)"] = float(rdn)
        out["SG&A (Q)"] = float(sga)

        # Kassa & skulder (BS, Q)
        cash = _safe_row_val(q_bs, "Cash")
        if cash == 0.0:
            cash = _safe_row_val(q_bs, "Cash And Cash Equivalents")
        debt = _safe_row_val(q_bs, "Total Debt")
        cur_assets = _safe_row_val(q_bs, "Total Current Assets")
        cur_liab   = _safe_row_val(q_bs, "Total Current Liabilities")
        out["Kassa"] = float(cash)
        out["Total skuld"] = float(debt)
        out["Current assets"] = float(cur_assets)
        out["Current liabilities"] = float(cur_liab)

        # EBITDA (TTM) & Räntekostnad (TTM)
        ebitda_ttm = _safe_row_ttm(q_is, "Ebitda")
        interest_ttm = _safe_row_ttm(q_is, "Interest Expense")
        out["EBITDA (TTM)"] = float(ebitda_ttm)
        out["Räntekostnad (TTM)"] = float(interest_ttm)

        # Burn & runway (kvartalsvis burn ≈ -FCF om FCF<0, annars 0)
        burn = -fcf if fcf < 0 else 0.0
        out["Burn rate (Q)"] = float(burn)
        out["Runway (kvartal)"] = float((cash / (-burn)) if burn < 0 else 0.0)
    except Exception:
        pass
    return out

# =============== Auto pipelines ==============================================
def auto_fetch_for_ticker(ticker: str) -> Tuple[dict, dict]:
    """
    Pipeline:
      1) SEC + Yahoo (implied shares, TTM/Q-PS, MCAP-historik) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S/mcap/shares) om saknas
      4) Yahoo CF/BS (kassa, skuld, FCF, burn, runway, opex/FoU/SG&A, EBITDA/Interest TTM)
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","Finansiell valuta","MCap (nu)",
            "_debug_shares_source","P/S (Yahoo)","Sektor","Beta",
            "MCap Datum Q1","MCap Q1","MCap Datum Q2","MCap Q2","MCap Datum Q3","MCap Q3","MCap Datum Q4","MCap Q4"
        ]}
        for k in [
            "Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","MCap (nu)","P/S (Yahoo)","Sektor","Beta",
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "MCap Datum Q1","MCap Q1","MCap Datum Q2","MCap Q2","MCap Datum Q3","MCap Q3","MCap Datum Q4","MCap Q4"
        ]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) Finnhub estimat om saknas
    try:
        need_cur = ("Omsättning idag" not in vals) or (vals.get("Omsättning idag", 0.0) <= 0.0)
        need_nxt = ("Omsättning nästa år" not in vals) or (vals.get("Omsättning nästa år", 0.0) <= 0.0)
        if need_cur or need_nxt:
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
        if ("P/S" not in vals) or (float(vals.get("P/S",0.0)) <= 0.0):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier"), "MCap (nu)": fmpl.get("MCap (nu)")}
            for k in ["P/S","Utestående aktier","MCap (nu)","Aktuell kurs"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # 4) Yahoo CF/BS (kassaflöden & balans)
    try:
        cfbs = hamta_cashflow_balance_yahoo(ticker)
        debug["yahoo_cfbs"] = cfbs
        for k, v in cfbs.items():
            if v not in (None, "",):
                vals[k] = v
    except Exception as e:
        debug["yahoo_cfbs_err"] = str(e)

    return vals, debug

def _find_row_idx_by_ticker(df: pd.DataFrame, tkr: str):
    if "Ticker" not in df.columns:
        return None
    m = df["Ticker"].astype(str).str.upper().str.strip() == (tkr or "").upper().strip()
    idxs = df.index[m].tolist()
    return idxs[0] if idxs else None

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, force_ts: bool = True):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla nya värden.
    Stämplar TS_ per fält, samt 'Senast auto-uppdaterad' + källa.
    """
    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()

    total = len(df)
    any_changed = False

    for i, row in df.reset_index(drop=True).iterrows():
        tkr = str(row.get("Ticker","")).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1))
            continue

        status.write(f"Uppdaterar {i+1}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            ridx = _find_row_idx_by_ticker(df, tkr)
            if ridx is None:
                log["misses"][tkr] = ["ticker saknas i df"]
                progress.progress((i+1)/max(total,1))
                continue
            changed = apply_auto_updates_to_row(
                df, ridx, new_vals,
                source="Auto (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)",
                changes_map=log["changed"],
                force_ts=force_ts
            )
            any_changed = any_changed or changed
            if not changed:
                log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
            if i < 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(total,1))

    # Efter loop — räkna om & spara (en gång)
    df = uppdatera_berakningar(df, user_rates)

    if any_changed or force_ts:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Data sparad.")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# app.py — Del 5/8
# --- Snapshots, kontroll & debug --------------------------------------------

def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df. Kräver endast Sheets (inte Drive).
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
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Bolag som sannolikt kräver manuell hantering:
    - saknar någon av kärnfälten eller TS,
    - och äldsta TS är äldre än 'older_than_days'.
    """
    need_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS[c] for c in TS_FIELDS if c in need_cols]

    out_rows = []
    cutoff = now_dt() - timedelta(days=older_than_days)

    for _, r in df.iterrows():
        missing_val = any((float(r.get(c, 0.0)) <= 0.0) for c in need_cols)
        missing_ts  = any((not str(r.get(ts, "")).strip()) for ts in ts_cols if ts in r)
        oldest = oldest_any_ts(r)
        oldest_dt = pd.to_datetime(oldest).to_pydatetime() if pd.notna(oldest) else None
        too_old = (oldest_dt is not None and oldest_dt < cutoff)

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Äldsta TS": oldest.strftime("%Y-%m-%d") if pd.notna(oldest) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })

    return pd.DataFrame(out_rows)

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"]).head(20)
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

    # 3) Senaste körlogg (om du nyss körde Auto)
    st.subheader("📒 Senaste körlogg (Auto)")
    log = st.session_state.get("last_auto_log")
    if not log:
        st.info("Ingen auto-körning körd i denna session ännu.")
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
        st.markdown("**Debug (första 20)**")
        st.json(log.get("debug_first_20", []))

def debug_test_single_ticker(ticker: str):
    """Visar vad källorna levererar för en ticker, för felsökning."""
    st.markdown(f"### Testa datakällor för: **{ticker}**")
    cols = st.columns(2)

    with cols[0]:
        st.write("**SEC/Yahoo combo (inkl. Yahoo fallback)**")
        try:
            v = hamta_sec_yahoo_combo(ticker)
            st.json(v)
        except Exception as e:
            st.error(f"SEC/Yahoo fel: {e}")

        st.write("**Yahoo global fallback direkt**")
        try:
            y = hamta_yahoo_global_combo(ticker)
            st.json(y)
        except Exception as e:
            st.error(f"Yahoo global fel: {e}")

    with cols[1]:
        st.write("**FMP light**")
        try:
            f = hamta_fmp_falt_light(ticker)
            st.json(f)
        except Exception as e:
            st.error(f"FMP fel: {e}")

        st.write("**Finnhub estimat**")
        try:
            fh = hamta_finnhub_revenue_estimates(ticker)
            st.json(fh)
        except Exception as e:
            st.error(f"Finnhub fel: {e}")

# app.py — Del 6/8
# --- Analys, Portfölj & Investeringsförslag ----------------------------------

# ===== Hjälpformattering & risklabel =========================================
def _fmt_large(n: float) -> str:
    try:
        n = float(n)
    except Exception:
        return str(n)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12:
        return f"{sign}{n/1e12:.2f} T"
    if n >= 1e9:
        return f"{sign}{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{sign}{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{sign}{n/1e3:.2f} K"
    return f"{sign}{n:.0f}"

def _fmt_curr(n: float, ccy: str) -> str:
    return f"{_fmt_large(n)} {ccy or ''}".strip()

def _to_sek(n: float, valuta: str, rates: dict) -> float:
    try:
        r = float(hamta_valutakurs(valuta, rates))
        return float(n) * (r if r > 0 else 1.0)
    except Exception:
        return float(n or 0.0)

def _to_usd(n: float, valuta: str, rates: dict) -> float:
    """Konvertera belopp i 'valuta' till USD via SEK som pivot (USD→SEK finns alltid i rates)."""
    try:
        sek = _to_sek(n, valuta, rates)
        usd_rate = float(rates.get("USD", 10.0)) or 10.0
        return sek / usd_rate
    except Exception:
        return 0.0

def classify_market_cap_band_usd(mcap_val: float, valuta: str, rates: dict) -> tuple[str, float]:
    """
    Returnerar (band, mcap_usd).
      Micro: < $300M
      Small: $300M–$2B
      Mid:   $2B–$10B
      Large: $10B–$200B
      Mega:  > $200B
    """
    usd = _to_usd(mcap_val or 0.0, valuta or "USD", rates)
    if usd <= 0:
        return "Okänd", 0.0
    if usd >= 200e9:
        return "Mega", usd
    if usd >= 10e9:
        return "Large", usd
    if usd >= 2e9:
        return "Mid", usd
    if usd >= 300e6:
        return "Small", usd
    return "Micro", usd

# ===== Robust nav för Investeringsförslag ====================================
def _prop_nav_sync(order_tickers: list[str]):
    """
    Håller förslags-index i synk när filtren ändras.
    Om ordningen byts ut försöker vi behålla nuvarande ticker.
    """
    if "prop_tickers" not in st.session_state:
        st.session_state.prop_tickers = order_tickers
        st.session_state.forslags_index = 0
        return

    if order_tickers != st.session_state.prop_tickers:
        cur_idx = int(st.session_state.get("forslags_index", 0) or 0)
        cur_tkr = None
        if 0 <= cur_idx < len(st.session_state.prop_tickers):
            cur_tkr = st.session_state.prop_tickers[cur_idx]

        if cur_tkr in order_tickers:
            st.session_state.forslags_index = order_tickers.index(cur_tkr)
        else:
            st.session_state.forslags_index = 0

        st.session_state.prop_tickers = order_tickers

def _prop_nav_prev(list_len: int, wrap: bool = True):
    idx = int(st.session_state.get("forslags_index", 0) or 0)
    if list_len <= 0:
        st.session_state.forslags_index = 0
    else:
        st.session_state.forslags_index = (idx - 1) % list_len if wrap else max(0, idx - 1)

def _prop_nav_next(list_len: int, wrap: bool = True):
    idx = int(st.session_state.get("forslags_index", 0) or 0)
    if list_len <= 0:
        st.session_state.forslags_index = 0
    else:
        st.session_state.forslags_index = (idx + 1) % list_len if wrap else min(list_len - 1, idx + 1)

def _prop_safe_index(list_len: int) -> int:
    """Garantier: alltid ett giltigt index (0..len-1), även om state är trasigt."""
    if list_len <= 0:
        st.session_state.forslags_index = 0
        return 0
    try:
        idx = int(st.session_state.get("forslags_index", 0) or 0)
    except Exception:
        idx = 0
    if idx < 0: idx = 0
    if idx >= list_len: idx = list_len - 1
    st.session_state.forslags_index = idx
    return idx

# ===== Analysvy ===============================================================
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
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","MCap (nu)","Sektor","Beta",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# ===== Portfölj ===============================================================
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)
    port["Andel (%)"] = port["Andel (%)"].round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True, hide_index=True
    )

# ===== Investeringsförslag ====================================================
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    # Filtrering: sektor + cap-band
    sectors = sorted([s for s in df.get("Sektor", pd.Series(dtype=str)).dropna().unique().tolist() if str(s).strip()])
    sektor_val = st.multiselect("Filtrera på sektor", ["Alla"] + sectors, default=["Alla"])
    cap_filter = st.selectbox("Filtrera på börsvärde", ["Alla","Micro","Small","Mid","Large","Mega"], index=0)

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()

    # Sektorfiler
    if "Alla" not in sektor_val:
        base = base[base["Sektor"].isin(sektor_val)].copy()

    # Cap-band kolumn
    def _band_for_row(row):
        return classify_market_cap_band_usd(row.get("MCap (nu)", 0.0), row.get("Valuta","USD"), user_rates)[0]
    if not base.empty:
        base["CapBand"] = base.apply(_band_for_row, axis=1)
        if cap_filter != "Alla":
            base = base[base["CapBand"] == cap_filter].copy()

    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Potential & diff
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    # Sortering
    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # --- ROBUST NAVIGERING ---------------------------------------------------
    order_tickers = base["Ticker"].astype(str).str.upper().str.strip().tolist()
    _prop_nav_sync(order_tickers)  # synka index mot nytt urval

    # Wrap-around som val
    if "prop_wrap" not in st.session_state:
        st.session_state.prop_wrap = True
    st.session_state.prop_wrap = st.checkbox("Låt bläddring gå runt (wrap-around)", value=st.session_state.prop_wrap)

    # Gå-till (select)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in base.iterrows()]
    safe_idx = _prop_safe_index(len(base))
    goto = st.selectbox("Välj bolag bland urvalet", labels, index=safe_idx, key="prop_select")
    if labels:
        chosen_idx = labels.index(goto)
        if chosen_idx != st.session_state.forslags_index:
            st.session_state.forslags_index = chosen_idx
            safe_idx = _prop_safe_index(len(base))

    # Föregående / status / Nästa
    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        st.button("⬅️ Föregående", on_click=_prop_nav_prev, args=(len(base), st.session_state.prop_wrap), disabled=(len(base) <= 1), key="prop_btn_prev")
    with col_mid:
        st.write(f"Förslag {safe_idx+1}/{len(base)}")
    with col_next:
        st.button("➡️ Nästa", on_click=_prop_nav_next, args=(len(base), st.session_state.prop_wrap), disabled=(len(base) <= 1), key="prop_btn_next")

    # Sista guard innan iloc
    safe_idx = _prop_safe_index(len(base))
    rad = base.iloc[safe_idx]

    # Portföljvärde för andel-beräkning
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    # Köp-beräkning
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r2 = port[port["Ticker"] == rad["Ticker"]]
        if not r2.empty:
            nuv_innehav = float(r2["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    # Header-rad (snabböversikt)
    mcap_now = float(rad.get("MCap (nu)", 0.0))
    mcap_txt = _fmt_curr(mcap_now, rad.get("Valuta",""))
    cap_band, mcap_usd = classify_market_cap_band_usd(mcap_now, rad.get("Valuta","USD"), user_rates)
    st.caption(f"Risklabel: **{cap_band}cap** (≈ {_fmt_large(mcap_usd)} USD)  •  Sektor: **{rad.get('Sektor','–')}**")

    # Basrad om köp
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
    ]
    st.markdown("\n".join(lines))

    # Nyckeltal & historik (expander)
    with st.expander("📊 Nyckeltal & historik", expanded=True):
        # MCap + P/S
        ps_now = float(rad.get("P/S", 0.0))
        ps_y   = float(rad.get("P/S (Yahoo)", 0.0))
        ps_s   = float(rad.get("P/S-snitt", 0.0))
        st.markdown(
            f"""
- **Börsvärde (nu):** {mcap_txt}  
- **P/S (nu):** {ps_now:.2f}   {'(Yahoo: '+str(round(ps_y,2))+')' if ps_y>0 else ''}  
- **Snitt P/S (senaste 4 TTM):** {ps_s:.2f}
            """.strip()
        )

        # MCAP-historik Q1–Q4
        mh = []
        for i in range(1,5):
            dt = str(rad.get(f"MCap Datum Q{i}",""))
            mv = float(rad.get(f"MCap Q{i}", 0.0))
            if dt or mv:
                mh.append(f"- **MCap Q{i}** ({dt or '–'}): {_fmt_curr(mv, rad.get('Valuta',''))}")
        if mh:
            st.markdown("**MCAP-historik (TTM-punkter):**\n" + "\n".join(mh))

        # CF/BS/IS
        cash = float(rad.get("Kassa",0.0))
        debt = float(rad.get("Total skuld",0.0))
        fcfq = float(rad.get("Fritt kassaflöde (Q)",0.0))
        burn = float(rad.get("Burn rate (Q)",0.0))
        runway = float(rad.get("Runway (kvartal)",0.0))
        opex = float(rad.get("Operating Expense (Q)",0.0))
        rdn  = float(rad.get("FoU (Q)",0.0))
        sga  = float(rad.get("SG&A (Q)",0.0))
        ebitda_ttm = float(rad.get("EBITDA (TTM)",0.0))
        int_ttm    = float(rad.get("Räntekostnad (TTM)",0.0))
        cur_assets = float(rad.get("Current assets",0.0))
        cur_liab   = float(rad.get("Current liabilities",0.0))
        cur_ratio  = (cur_assets / cur_liab) if cur_liab > 0 else 0.0

        ccy = rad.get("Valuta","")
        st.markdown(
            f"""
**Kassa/skuld & kassaflöden**
- **Kassa:** {_fmt_curr(cash, ccy)}   •   **Skuld:** {_fmt_curr(debt, ccy)}
- **Operativt kassaflöde (Q):** {_fmt_curr(rad.get('Operativt kassaflöde (Q)',0.0), ccy)}
- **CapEx (Q):** {_fmt_curr(rad.get('CapEx (Q)',0.0), ccy)}
- **Fritt kassaflöde (Q):** {_fmt_curr(fcfq, ccy)}
- **Burn rate (Q):** {_fmt_curr(burn, ccy)}   •   **Runway:** {runway:.1f} kvartal

**Kostnader (Q)**
- **Operating Expense:** {_fmt_curr(opex, ccy)}   •   **FoU:** {_fmt_curr(rdn, ccy)}   •   **SG&A:** {_fmt_curr(sga, ccy)}

**Lönsamhet & likviditet**
- **EBITDA (TTM):** {_fmt_curr(ebitda_ttm, ccy)}   •   **Räntekostnad (TTM):** {_fmt_curr(int_ttm, ccy)}
- **Current ratio:** {cur_ratio:.2f}
            """.strip()
        )

# app.py — Del 7/8
# --- Lägg till/uppdatera bolag: enskild & batch --------------------------------

def _ts_badges(row: pd.Series) -> str:
    auto = str(row.get("Senast auto-uppdaterad","")).strip()
    src  = str(row.get("Senast uppdaterad källa","")).strip()
    manu = str(row.get("Senast manuellt uppdaterad","")).strip()
    parts = []
    if auto:
        parts.append(f"🟦 Auto: **{auto}**" + (f"  _(källa: {src})_" if src else ""))
    if manu:
        parts.append(f"🟧 Manuell: **{manu}**")
    if not parts:
        return "–"
    return " • ".join(parts)

def _price_only_update(df: pd.DataFrame, tkr: str) -> tuple[bool, str]:
    """Uppdatera ENDAST 'Aktuell kurs' + MCap (nu) om möjligt. Stämpla auto-källa."""
    ridx = _find_row_idx_by_ticker(df, tkr)
    if ridx is None:
        return False, f"{tkr} hittades inte i tabellen."
    try:
        y = hamta_yahoo_fält(tkr)
        changed = False
        if float(y.get("Aktuell kurs", 0.0)) > 0:
            df.at[ridx, "Aktuell kurs"] = float(y["Aktuell kurs"]); changed = True
        if float(y.get("MCap (nu)", 0.0)) > 0:
            df.at[ridx, "MCap (nu)"] = float(y["MCap (nu)"]); changed = True
        if y.get("Valuta"): df.at[ridx, "Valuta"] = y["Valuta"]
        _note_auto_update(df, ridx, source="Auto (pris/Yahoo)")
        return changed, "Kurs uppdaterad." if changed else "Ingen kursförändring hittades."
    except Exception as e:
        return False, f"Fel vid prisuppdatering: {e}"

def _auto_full_one(df: pd.DataFrame, user_rates: dict, tkr: str) -> tuple[bool, dict, str]:
    """Full auto för EN ticker med force_ts=True (stämplar auto-tid även om värdena är oförändrade)."""
    ridx = _find_row_idx_by_ticker(df, tkr)
    if ridx is None:
        return False, {}, f"{tkr} hittades inte i tabellen."
    vals, debug = auto_fetch_for_ticker(tkr)
    changed = apply_auto_updates_to_row(
        df, ridx, vals, source="Auto (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)", changes_map={}, force_ts=True
    )
    # Räkna om beräkningar för hela df (lätt att hålla konsistens)
    uppdatera_berakningar(df, user_rates)
    return changed, debug, ("Fält uppdaterade." if changed else "Inga ändringar hittades vid auto-uppdatering.")

def _ts_table_for_row(row: pd.Series) -> pd.DataFrame:
    """Liten tabell med de spårade TS-fälten."""
    items = []
    for f, ts_col in TS_FIELDS.items():
        if ts_col in row.index:
            items.append({"Fält": f, "Senast (TS)": str(row.get(ts_col,""))})
    return pd.DataFrame(items)

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # --- Sorteringsläge för listan
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # --- Välj post + robust bläddring
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": str(r['Ticker']).upper().strip() for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    # --- Befintlig rad?
    if valt_label and valt_label in namn_map:
        cur_tkr = namn_map[valt_label]
        sel = vis_df[vis_df["Ticker"].astype(str).str.upper().str.strip() == cur_tkr]
        bef = sel.iloc[0] if not sel.empty else pd.Series({}, dtype=object)
    else:
        cur_tkr = ""
        bef = pd.Series({}, dtype=object)

    # --- TS-etiketter (överst)
    if not bef.empty:
        st.markdown(f"**Uppdateringsstatus:** {_ts_badges(bef)}")
        with st.expander("Visa tidsstämplar per fält (TS_)", expanded=False):
            st.dataframe(_ts_table_for_row(bef), use_container_width=True, hide_index=True)

    # --- Enskilda snabbknappar (om valt befintligt)
    if not bef.empty:
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            if st.button("📈 Uppdatera **kurs** (endast)", key="btn_price_only"):
                ok, msg = _price_only_update(df, cur_tkr)
                if ok:
                    spara_data(df)   # spara direkt, billigt
                    st.success(msg)
                else:
                    st.warning(msg)
        with c2:
            if st.button("🧠 Full auto – **bara denna**", key="btn_full_auto_one"):
                changed, debug, msg = _auto_full_one(df, user_rates, cur_tkr)
                spara_data(df, do_snapshot=False)  # spara efter en-post-körning
                st.success(msg)
                with st.expander("Debug för denna körning"):
                    st.json(debug)
        with c3:
            st.caption("Tips: Använd batch-sektionen längre ned för att köra flera i följd.")

    # --- Formulär för skapa/ändra fält
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=str(bef.get("Ticker","")).upper() if not bef.empty else "").upper().strip()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

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

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Bestäm om manuell-ts ska sättas + vilka TS-fält som ska stämplas
        datum_sätt = False
        changed_manual_fields = []
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            for k in MANUELL_FALT_FOR_DATUM:
                if before[k] != after[k]:
                    datum_sätt = True
                    changed_manual_fields.append(k)
        else:
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
                changed_manual_fields = [f for f in MANUELL_FALT_FOR_DATUM if float(ny.get(f,0.0)) != 0.0]

        # Skriv in nya fält (skapa om behövs)
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"].astype(str).str.upper().str.strip()==ticker.upper().strip(), k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Sätt manuell TS + TS_ per fält
        ridx = _find_row_idx_by_ticker(df, ticker)
        if ridx is not None and datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo
        data = hamta_yahoo_fält(ticker)
        ridx = _find_row_idx_by_ticker(df, ticker)
        if ridx is not None:
            if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
            if data.get("Valuta"):     df.loc[ridx, "Valuta"]     = data["Valuta"]
            if float(data.get("Aktuell kurs",0))>0: df.loc[ridx, "Aktuell kurs"] = float(data["Aktuell kurs"])
            if data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            if data.get("CAGR 5 år (%)")   is not None: df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # --- Äldst uppdaterade (alla spårade fält) ---
    st.markdown("### ⏱️ Äldst uppdaterade (alla spårade fält, topp 10)")
    work = add_oldest_ts_col(df.copy())
    topp = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"], ascending=[True, True, True]).head(10)

    visa_kol = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
              "TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in df.columns:
            visa_kol.append(k)
    visa_kol.append("_oldest_any_ts")
    st.dataframe(topp[visa_kol], use_container_width=True, hide_index=True)

    # ===================== BATCH-KÖRNING =========================
    st.divider()
    st.subheader("⚙️ Batchkörning (delkörningar med minne)")

    # Bygg eller uppdatera batchlista
    colb1, colb2 = st.columns([2,1])
    with colb1:
        batch_order = st.selectbox("Ordning", ["A–Ö (bolagsnamn)","Äldst uppdaterade först"], index=1)
    with colb2:
        chunk = st.number_input("Antal per körning", min_value=1, max_value=100, value=int(st.session_state.get("batch_chunk", 10)), step=1)
        st.session_state.batch_chunk = int(chunk)

    def _build_queue() -> list[str]:
        if batch_order.startswith("Äldst"):
            w = add_oldest_ts_col(df.copy())
            lst = w.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"]).loc[:, "Ticker"].astype(str).str.upper().str.strip().tolist()
        else:
            w = df.sort_values(by=["Bolagsnamn","Ticker"])
            lst = w["Ticker"].astype(str).str.upper().str.strip().tolist()
        # filtrera ut tomma
        return [t for t in lst if t]

    # Initiera kö vid behov
    if "batch_queue" not in st.session_state or "batch_ptr" not in st.session_state:
        st.session_state.batch_queue = _build_queue()
        st.session_state.batch_ptr = 0

    # Om användaren ändrar ordning – bygg ny kö men försök behålla position
    if st.button("🔃 Bygg/uppdatera batchlista utifrån vald ordning"):
        st.session_state.batch_queue = _build_queue()
        st.session_state.batch_ptr = 0
        st.success(f"Batchlista klar ({len(st.session_state.batch_queue)} tickers).")

    colc1, colc2, colc3 = st.columns([1,1,2])
    with colc1:
        if st.button("🚀 Kör nästa batch"):
            q = st.session_state.batch_queue
            p = int(st.session_state.batch_ptr or 0)
            n = int(st.session_state.batch_chunk or 10)
            if p >= len(q):
                st.info("Inget kvar i kön. Bygg om listan eller återställ.")
            else:
                till = min(p + n, len(q))
                run = q[p:till]
                progress = st.progress(0.0)
                misses = {}
                changed_any = False
                for i, tkr in enumerate(run, start=1):
                    ridx = _find_row_idx_by_ticker(df, tkr)
                    if ridx is None:
                        misses[tkr] = ["ticker saknas i df"]
                        progress.progress(i/len(run))
                        continue
                    try:
                        new_vals, debug = auto_fetch_for_ticker(tkr)
                        ch = apply_auto_updates_to_row(
                            df, ridx, new_vals,
                            source="Batch auto (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)",
                            changes_map={},
                            force_ts=True
                        )
                        changed_any = changed_any or ch
                    except Exception as e:
                        misses[tkr] = [f"error: {e}"]
                    progress.progress(i/len(run))

                # räkna om & spara EN gång
                uppdatera_berakningar(df, user_rates)
                spara_data(df, do_snapshot=False)

                st.session_state.batch_ptr = till
                if misses:
                    st.warning(f"Klar med batch ({p+1}–{till}). Några missar:")
                    st.json(misses)
                else:
                    st.success(f"Klar med batch ({p+1}–{till}).")
    with colc2:
        if st.button("↩️ Återställ batch"):
            st.session_state.batch_queue = _build_queue()
            st.session_state.batch_ptr = 0
            st.info("Batchen återställd till start.")
    with colc3:
        q = st.session_state.batch_queue
        p = int(st.session_state.batch_ptr or 0)
        st.caption(f"Kvar i kö: **{max(0, len(q)-p)}**  •  Total: **{len(q)}**  •  Position: **{p}**")
        if p < len(q):
            nxt = q[p:p+min(5, len(q)-p)]
            if nxt:
                st.caption("Nästa upp till 5 tickers: " + ", ".join(nxt))

    return df

# app.py — Del 8/8
# --- MAIN --------------------------------------------------------------------

def _init_rate_state():
    # Läs sparade (Sheets) och initiera session_state första gången
    saved = las_sparade_valutakurser()
    for c, default in [("USD", STANDARD_VALUTAKURSER["USD"]),
                       ("NOK", STANDARD_VALUTAKURSER["NOK"]),
                       ("CAD", STANDARD_VALUTAKURSER["CAD"]),
                       ("EUR", STANDARD_VALUTAKURSER["EUR"])]:
        key = f"rate_{c.lower()}"
        if key not in st.session_state:
            st.session_state[key] = float(saved.get(c, default))
    if "rates_reload" not in st.session_state:
        st.session_state["rates_reload"] = 0

def _build_user_rates_from_state() -> dict:
    return {
        "USD": float(st.session_state.get("rate_usd", STANDARD_VALUTAKURSER["USD"])),
        "NOK": float(st.session_state.get("rate_nok", STANDARD_VALUTAKURSER["NOK"])),
        "CAD": float(st.session_state.get("rate_cad", STANDARD_VALUTAKURSER["CAD"])),
        "EUR": float(st.session_state.get("rate_eur", STANDARD_VALUTAKURSER["EUR"])),
        "SEK": 1.0,
    }

def _snabb_uppdatera_alla_kurser(df: pd.DataFrame):
    if df.empty:
        st.warning("Inga bolag i tabellen.")
        return df
    st.info("Startar snabb kursuppdatering (pris + MCap via Yahoo).")
    prog = st.progress(0.0)
    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        tkr = str(row.get("Ticker","")).strip().upper()
        if not tkr:
            prog.progress(i/max(total,1.0)); continue
        try:
            y = hamta_yahoo_fält(tkr)
            if float(y.get("Aktuell kurs", 0.0)) > 0:
                df.at[idx, "Aktuell kurs"] = float(y["Aktuell kurs"])
            if float(y.get("MCap (nu)", 0.0)) > 0:
                df.at[idx, "MCap (nu)"] = float(y["MCap (nu)"])
            if y.get("Valuta"):
                df.at[idx, "Valuta"] = str(y["Valuta"])
            _note_auto_update(df, idx, source="Auto (pris/Yahoo)")
        except Exception:
            pass
        # Liten paus för att vara snäll
        time.sleep(0.05)
        prog.progress(i/max(total,1.0))
    st.success("Kursuppdatering klar (alla tickers).")
    return df

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Sidopanel: valutakurser
    st.sidebar.header("💱 Valutakurser → SEK")
    _init_rate_state()

    st.session_state.rate_usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd), step=0.01, format="%.4f")
    st.session_state.rate_nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok), step=0.01, format="%.4f")
    st.session_state.rate_cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad), step=0.01, format="%.4f")
    st.session_state.rate_eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur), step=0.01, format="%.4f")

    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        # Sätt i state så att inputs uppdateras direkt
        st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
        st.sidebar.success(f"Valutakurser hämtade (källa: {provider}).")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))

    user_rates = _build_user_rates_from_state()

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            # Läs in igen och lägg i state
            saved = las_sparade_valutakurser()
            st.session_state.rate_usd = float(saved.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(saved.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(saved.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(saved.get("EUR", st.session_state.rate_eur))
            st.sidebar.info("Hämtade sparade kurser.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()

    # --- Läs data (Sheets) & säkra schema
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Auto-uppdatering (tung) och snabb kurs-körning
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Uppdateringar")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före tung körning", value=True)
    if st.sidebar.button("⚡ Uppdatera alla **kurser** (snabb)"):
        df = _snabb_uppdatera_alla_kurser(df)
        # Räkna om derivat som kan bero på kurs (riktkurser beror inte på kurs, men lämnar för konsekvens)
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df, do_snapshot=False)

    if st.sidebar.button("🔄 Auto-uppdatera **alla** (tung)"):
        df, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot, force_ts=True)
        st.session_state["last_auto_log"] = log

    # --- Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # Om funktionen expanderade df (ny rad etc), spara tillbaka och uppdatera df för fortsatt navigering
        if not df2.equals(df):
            df = df2
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
