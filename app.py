# -*- coding: utf-8 -*-

# app.py — Del 1/? — Imports, tidszon, Google Sheets, valutakurser

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
from typing import Optional, Tuple, List, Dict, Any

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

def hamta_data() -> pd.DataFrame:
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

# app.py — Del 2/? — Kolumnschema, migrering, typkonvertering, TS-hjälpare

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

# Slutlig kolumnlista i databasen
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Metadata / extra
    "_marketCap_raw", "Sektor",

    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    if df is None or df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Bolagsnamn","Valuta","Sektor","Ticker"):
                df[kol] = ""
            elif any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","_marketcap_raw"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    # inga dubbletter
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
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "_marketCap_raw"
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

# app.py — Del 3/? — Yahoo-hjälpare & beräkningar & merge-hjälpare

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
    """Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR, Sektor, Mcap."""
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "Sektor": "",
        "_marketCap_raw": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency")
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        sect = info.get("sector") or ""
        out["Sektor"] = str(sect)

        mcap = info.get("marketCap")
        try:
            if mcap is not None:
                out["_marketCap_raw"] = float(mcap)
        except Exception:
            pass

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
        try:
            ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        except Exception:
            ps_clean = []
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: >100% → 50%, <0% → 2%
        try:
            cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        except Exception:
            cagr = 0.0
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år"
        try:
            oms_next = float(rad.get("Omsättning nästa år", 0.0))
        except Exception:
            oms_next = 0.0
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0) or 0.0)
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0) or 0.0)

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        try:
            aktier_ut_m = float(rad.get("Utestående aktier", 0.0) or 0.0)  # i miljoner
        except Exception:
            aktier_ut_m = 0.0
        aktier_ut = aktier_ut_m * 1_000_000.0
        if aktier_ut > 0 and ps_snitt > 0:
            try:
                oms_idag = float(rad.get("Omsättning idag", 0.0) or 0.0) * 1_000_000.0
                oms_nxt  = float(rad.get("Omsättning nästa år", 0.0) or 0.0) * 1_000_000.0
                oms_2y   = float(df.at[i, "Omsättning om 2 år"] or 0.0) * 1_000_000.0
                oms_3y   = float(df.at[i, "Omsättning om 3 år"] or 0.0) * 1_000_000.0
            except Exception:
                oms_idag = oms_nxt = oms_2y = oms_3y = 0.0

            df.at[i, "Riktkurs idag"]    = round((oms_idag * ps_snitt) / aktier_ut, 2) if oms_idag>0 else 0.0
            df.at[i, "Riktkurs om 1 år"] = round((oms_nxt  * ps_snitt) / aktier_ut, 2) if oms_nxt >0 else 0.0
            df.at[i, "Riktkurs om 2 år"] = round((oms_2y   * ps_snitt) / aktier_ut, 2) if oms_2y >0 else 0.0
            df.at[i, "Riktkurs om 3 år"] = round((oms_3y   * ps_snitt) / aktier_ut, 2) if oms_3y >0 else 0.0
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, changes_map: dict, always_stamp: bool=False) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde.
    Uppdaterar TS_ för spårade fält, sätter 'Senast auto-uppdaterad' + källa.
    Om always_stamp=True stämplas TS även om värdet råkar bli identiskt.
    Returnerar True om något fält faktiskt ändrades (eller om always_stamp och fält finns).
    """
    changed_fields = []
    wrote_any = False
    for f, v in new_vals.items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f] if f in df.columns else None

        # skrivregler
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        if not write_ok and not always_stamp:
            continue

        # skriv
        if always_stamp:
            df.at[row_idx, f] = v
            wrote_any = True
        else:
            if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
                df.at[row_idx, f] = v
                changed_fields.append(f)
                wrote_any = True

        # TS för spårade fält
        if f in TS_FIELDS:
            _stamp_ts_for_field(df, row_idx, f)

    if wrote_any:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return True
    return False

# app.py — Del 4/? — Datakällor: FMP, SEC, Yahoo fallback, Finnhub & auto-pipeline

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

    delays = [0.0, 1.0, 2.0]
    last_sc, last_json = 0, None
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
        if "price" in q0:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
        if q0.get("marketCap") is not None:
            try: out["_marketCap_raw"] = float(q0["marketCap"])
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

# =============== SEC (US + FPI/IFRS) ========================================

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

# ---------- datum & kvarts-utils --------------------------------------------
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
            return (d2 - d1).days <= 2  # ≈ instant
        except Exception:
            return False
    return False

def _fiscal_quarter_from_date(d):
    """Returnerar en (FY, Qn)-tuppel utifrån kalenderkvartal – används för unique-filter."""
    if not d: return (None, None)
    m = d.month
    q = 1 if m in (1,2,3) else 2 if m in (4,5,6) else 3 if m in (7,8,9) else 4
    fy = d.year
    return (fy, q)

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

# ---------- IFRS/GAAP kvartalsintäkter + valuta (årsskifte-fix) --------------

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
    return 0.0

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    Inkl. tolerans kring årsskifte: 75–110 dagar duration accepteras.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A", "10-K", "10-K/A")}),  # vissa Q4-siffror rapporteras i 10-K som ~90d segment
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A", "10-K")}),
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

    # samla alla kandidater
    candidates = []
    pick_unit = None
    for taxo, cfg in taxos:
        gaap = (facts.get("facts") or {}).get(taxo, {})
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list):
                    continue
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
                    # tillåt 75–110 dagar → fånga Q4 runt årsskifte
                    if dur is None or dur < 75 or dur > 110:
                        continue
                    try:
                        v = float(val)
                        candidates.append((end, v, unit_code))
                    except Exception:
                        pass

    if not candidates:
        return [], None

    # unika per (FY, Q) → plocka senaste för varje kvartsnyckel
    by_fyq = {}
    for end, v, unit in candidates:
        fy, q = _fiscal_quarter_from_date(end)
        if fy is None:
            continue
        key = (fy, q)
        if key not in by_fyq or end > by_fyq[key][0]:
            by_fyq[key] = (end, v, unit)

    # sortera nyast→äldst och kapa
    rows = sorted(by_fyq.values(), key=lambda t: t[0], reverse=True)
    if not rows:
        return [], None
    unit = rows[0][2]
    rows = [(d, v) for (d, v, _) in rows[:max_quarters]]
    return rows, unit

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters)
    return rows

def _sec_quarterly_revenues(facts: dict):
    rows = _sec_quarterly_revenues_dated(facts, max_quarters=4)
    return [v for (_, v) in rows]

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

    # 2) fallback: income_stmt quarterly
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

def _merge_sec_yahoo_quarters(sec_rows: list, y_rows: list) -> list:
    """
    Slår ihop SEC- och Yahoo-kvartal och säkerställer att senaste 4 kvartalen finns.
    Deduplicerar per (FY,Q). Fyller luckor (ex. dec/jan) från Yahoo om SEC saknas.
    """
    all_rows = []
    for (src_rows, tag) in ((sec_rows, "sec"), (y_rows, "y")):
        for d, v in (src_rows or []):
            fy, q = _fiscal_quarter_from_date(d)
            if fy is None:
                continue
            all_rows.append({"fy": fy, "q": q, "end": d, "rev": float(v), "src": tag})

    if not all_rows:
        return []

    # välj senaste datapunkten per (fy,q)
    by_fyq = {}
    for r in all_rows:
        key = (r["fy"], r["q"])
        if key not in by_fyq or r["end"] > by_fyq[key]["end"]:
            by_fyq[key] = r

    merged = sorted(by_fyq.values(), key=lambda r: r["end"], reverse=True)
    return [(r["end"], r["rev"]) for r in merged]

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn/mcap/sector från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik, MCAP Q1–Q4.
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "_marketCap_raw", "Sektor", "Årlig utdelning", "CAGR 5 år (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()

    # Shares: implied → fallback SEC robust
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
    mcap_now = float(out.get("_marketCap_raw", 0.0) or 0.0)
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used
        out["_marketCap_raw"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering; merge med Yahoo för årsskifte
    sec_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    y_rows = _yfi_quarterly_revenues(yf.Ticker(ticker))
    merged_rows = _merge_sec_yahoo_quarters(sec_rows, y_rows)
    if not merged_rows:
        return out

    # konvertera till prisvaluta
    conv = 1.0
    if rev_unit and rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    rows_px = [(d, v * conv) for (d, v) in merged_rows]

    # bygg TTM-fönster (4 st)
    ttm_list = _ttm_windows(rows_px, need=4)

    # P/S (TTM) nu + P/S Q1–Q4 & MCAP Q1–Q4 (baserat på samma shares_used)
    if mcap_now > 0 and ttm_list:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    if shares_used > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCAP Q{idx}"] = float(mcap_hist)

    return out

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 + MCAP Q1–Q4 historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/sector/mcap
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","Sektor","_marketCap_raw","Årlig utdelning","CAGR 5 år (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(out.get("_marketCap_raw", 0.0) or 0.0)

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

    # Kvartalsintäkter → TTM (merge via Yahoo – SEC saknas här)
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows or len(q_rows) < 4:
        return out
    ttm_list = _ttm_windows(q_rows, need=4)

    # Market cap (nu)
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px
        out["_marketCap_raw"] = mcap

    # P/S (TTM) nu
    if mcap > 0 and ttm_list:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historisk) + MCAP Q1–Q4
    if shares > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
            if ttm_rev and ttm_rev > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares * p
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev)
                    out[f"MCAP Q{idx}"] = float(mcap_hist)

    return out

# =============== Finnhub (valfritt, estimat) =================================
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets. Hämtar annual revenue estimates:
    current FY + next FY (om finns).
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

# =============== Auto-pipeline (per ticker) ==================================

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares + kvartal merge) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S/mcap/shares) om saknas
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4","Aktuell kurs","Bolagsnamn","Valuta","Sektor","_marketCap_raw","_debug_shares_source"]}
        for k in ["Bolagsnamn","Valuta","Sektor","Aktuell kurs","_marketCap_raw","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4","Årlig utdelning","CAGR 5 år (%)"]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) Finnhub estimat om saknas
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

    # 3) FMP light P/S/mcap/shares om saknas
    try:
        need = any(k not in vals for k in ["P/S","_marketCap_raw","Utestående aktier"])
        if need:
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "_marketCap_raw": fmpl.get("_marketCap_raw"), "Utestående aktier": fmpl.get("Utestående aktier")}
            for k in ["P/S","_marketCap_raw","Utestående aktier","Aktuell kurs"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    return vals, debug

# app.py — Del 5/? — Snapshots, TS-analys, batch-uppdatering, Kontroll-vy

# --- Snapshots ---------------------------------------------------------------

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

# --- Äldsta TS-analys --------------------------------------------------------

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
        try:
            missing_val = any((float(r.get(c, 0.0) or 0.0) <= 0.0) for c in need_cols)
        except Exception:
            missing_val = True
        missing_ts  = any((not str(r.get(ts, "") or "").strip()) for ts in ts_cols if ts in r)
        oldest = oldest_any_ts(r)
        oldest_dt = oldest.to_pydatetime() if isinstance(oldest, pd.Timestamp) and pd.notna(oldest) else None
        too_old = (oldest_dt is not None and oldest_dt < cutoff)

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Äldsta TS": oldest.strftime("%Y-%m-%d") if isinstance(oldest, pd.Timestamp) and pd.notna(oldest) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })

    return pd.DataFrame(out_rows)

# --- Auto-uppdatera: en rad & batch -----------------------------------------

def auto_update_one(df: pd.DataFrame, idx: int, ticker: str, *, variant: str = "full", changes_map: dict = None, always_stamp: bool=False) -> Tuple[pd.DataFrame, bool, dict]:
    """
    Uppdaterar en (1) ticker.
    variant: "full" (SEC/Yahoo→Finnhub→FMP) eller "pris" (endast Yahoo pris/mcap/valuta/namn/utdelning/sector).
    always_stamp: stämpla TS även om värdena blir oförändrade.
    """
    changes_map = changes_map if isinstance(changes_map, dict) else {}
    debug = {"ticker": ticker, "variant": variant}
    changed = False

    if variant == "pris":
        base = hamta_yahoo_fält(ticker)
        vals = {}
        for k in ("Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","Sektor","_marketCap_raw"):
            v = base.get(k)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
        if vals:
            changed = apply_auto_updates_to_row(df, idx, vals, source="Auto (pris/valuta från Yahoo)", changes_map=changes_map, always_stamp=always_stamp)
    else:
        vals, dbg = auto_fetch_for_ticker(ticker)
        debug["pipeline"] = dbg
        if vals:
            changed = apply_auto_updates_to_row(df, idx, vals, source="Auto (SEC/Yahoo→Finnhub→FMP)", changes_map=changes_map, always_stamp=always_stamp)

    return df, changed, debug

def _next_alpha_batch_indices(df: pd.DataFrame, batch_size: int) -> List[int]:
    """
    Håller en roterande pekare (session_state.alpha_cursor) genom A–Ö-listan.
    Returnerar index-lista för nästa batch.
    """
    if "alpha_cursor" not in st.session_state:
        st.session_state.alpha_cursor = 0
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index()
    start = st.session_state.alpha_cursor
    end = min(start + batch_size, len(vis))
    rows = vis.iloc[start:end]
    st.session_state.alpha_cursor = 0 if end >= len(vis) else end
    return list(rows["index"].values)

def _oldest_batch_indices(df: pd.DataFrame, batch_size: int) -> List[int]:
    """
    Tar de äldsta baserat på _oldest_any_ts_fill. Dynamiskt (ny beräkning varje gång).
    """
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).reset_index()
    rows = vis.iloc[:batch_size]
    return list(rows["index"].values)

def auto_update_batch(df: pd.DataFrame, user_rates: dict, *, picker: str = "oldest", batch_size: int = 10, variant: str = "full", make_snapshot: bool = True, always_stamp: bool=False):
    """
    Kör auto-uppdatering i batch:
      picker: "oldest" eller "alpha"
      variant: "full" eller "pris"
      always_stamp: stämpla TS även om inget ändras värdemässigt
    Visar progress + texten i/X.
    """
    n = len(df)
    if n == 0:
        st.warning("Ingen data i tabellen.")
        return df, {"changed": {}, "misses": {}, "debug": []}

    # välj indices
    if picker == "alpha":
        indices = _next_alpha_batch_indices(df, batch_size)
    else:
        indices = _oldest_batch_indices(df, batch_size)

    total = len(indices)
    progress = st.progress(0.0)
    status = st.empty()

    log = {"changed": {}, "misses": {}, "debug": []}
    any_changed = False

    for i, ridx in enumerate(indices, start=1):
        try:
            tkr = str(df.at[ridx, "Ticker"]).strip().upper()
        except Exception:
            tkr = ""
        if not tkr:
            status.write(f"{i}/{total} – tom ticker, hoppar över…")
            progress.progress(i/total)
            continue

        status.write(f"{i}/{total} – uppdaterar {tkr} …")
        try:
            df, changed, dbg = auto_update_one(df, ridx, tkr, variant=variant, changes_map=log["changed"], always_stamp=always_stamp)
            if not changed:
                log["misses"][tkr] = ["(inga nya/meningsfulla fält)"]
            any_changed = any_changed or changed
            log["debug"].append({tkr: dbg})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress(i/total)

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)
    if any_changed and make_snapshot:
        backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)
    if any_changed:
        spara_data(df, do_snapshot=False)
        st.success("Klart! Ändringar sparade.")
    else:
        st.info("Ingen faktisk ändring upptäcktes – ingen skrivning gjordes.")

    return df, log

# --- Debug: single ticker ----------------------------------------------------

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

# --- Kontroll-vy -------------------------------------------------------------

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

    # 3) Snabb batch-körning
    st.subheader("⚙️ Snabb batch-körning")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        picker = st.selectbox("Urval", ["Äldst","A–Ö"], index=0)
    with c2:
        batch_size = st.number_input("Antal i batch", min_value=1, max_value=100, value=10, step=1)
    with c3:
        variant = st.selectbox("Variant", ["full","pris"], index=0)
    with c4:
        make_snapshot = st.checkbox("Snapshot före skrivning", value=True)

    always_stamp = st.checkbox("Stämpla TS även vid oförändrat värde", value=True, help="Bra för att markera att posten kontrollerats idag.")

    if st.button("🚀 Kör batch nu"):
        df2, log = auto_update_batch(
            df.copy(),  # jobba på kopia för säkerhets skull
            user_rates={"USD": st.session_state.get("rate_usd", STANDARD_VALUTAKURSER["USD"]),
                        "NOK": st.session_state.get("rate_nok", STANDARD_VALUTAKURSER["NOK"]),
                        "CAD": st.session_state.get("rate_cad", STANDARD_VALUTAKURSER["CAD"]),
                        "EUR": st.session_state.get("rate_eur", STANDARD_VALUTAKURSER["EUR"]),
                        "SEK": 1.0},
            picker="alpha" if picker=="A–Ö" else "oldest",
            batch_size=int(batch_size),
            variant=variant,
            make_snapshot=bool(make_snapshot),
            always_stamp=bool(always_stamp),
        )
        # ersätt df i session?
        try:
            spara_data(df2, do_snapshot=False)  # redundans – auto_update_batch sparar redan när ändringar finns
        except Exception:
            pass
        st.session_state["last_auto_log"] = log
        st.success("Batch klar.")

    # 4) Senaste körlogg
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log")
    if not log:
        st.info("Ingen batch/auto-körning i denna session ännu.")
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
        st.markdown("**Debug (första)**")
        st.json(log.get("debug", [])[:20])

# app.py — Del 6/? — Analys-vy + format-helpers

# --------- Små hjälpfunktioner för snygg visning ----------------------------

def _fmt_large_number(n: float) -> str:
    """Formatera marknadsvärde m.m. i K / M / Md / T."""
    try:
        x = float(n)
    except Exception:
        return "–"
    neg = x < 0
    x = abs(x)
    if x >= 1e12:
        s = f"{x/1e12:.2f} T"
    elif x >= 1e9:
        s = f"{x/1e9:.2f} Md"
    elif x >= 1e6:
        s = f"{x/1e6:.2f} M"
    elif x >= 1e3:
        s = f"{x/1e3:.2f} K"
    else:
        s = f"{x:.0f}"
    return ("-" if neg else "") + s

def _badge_ts(row: pd.Series) -> str:
    """Returnerar en liten etiketttext som visar senast uppdaterad & källa."""
    auto = str(row.get("Senast auto-uppdaterad","") or "").strip()
    manu = str(row.get("Senast manuellt uppdaterad","") or "").strip()
    src  = str(row.get("Senast uppdaterad källa","") or "").strip()
    bits = []
    if auto:
        bits.append(f"Auto: {auto}")
    if manu:
        bits.append(f"Manuellt: {manu}")
    if src:
        bits.append(f"Källa: {src}")
    return " | ".join(bits) if bits else "–"

def _ps_hist_snitt(row: pd.Series) -> float:
    vals = []
    for k in ("P/S Q1","P/S Q2","P/S Q3","P/S Q4"):
        try:
            v = float(row.get(k, 0.0) or 0.0)
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    return round(float(np.mean(vals)) if vals else 0.0, 2)

# --------- Analys-vy --------------------------------------------------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # säkerställ beräkningar
    df = uppdatera_berakningar(df.copy(), user_rates)

    # Välj bolag
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    idx_num = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=int(st.session_state.analys_idx), step=1)
    st.session_state.analys_idx = int(idx_num)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    col_a, col_b, col_c = st.columns([1,1,2])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    with col_c:
        st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]

    # Header + statusrad
    st.subheader(f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})")
    st.caption(_badge_ts(r))

    # Nyckeltal
    left, right = st.columns([1,1])
    with left:
        st.metric("Aktuell kurs", f"{round(float(r.get('Aktuell kurs',0.0) or 0.0),2)} {r.get('Valuta','')}")
        st.metric("P/S TTM", round(float(r.get("P/S",0.0) or 0.0), 2))
        psn = _ps_hist_snitt(r)
        st.metric("P/S-snitt (Q1–Q4)", psn)
        mcap = float(r.get("_marketCap_raw", 0.0) or 0.0)
        st.metric("Market Cap", _fmt_large_number(mcap))
        st.write(f"**Sektor:** {r.get('Sektor','–')}")
        st.write(f"**Utestående aktier:** {round(float(r.get('Utestående aktier',0.0) or 0.0),2)} M")
    with right:
        st.write("**Riktkurser**")
        st.write(f"- Idag: **{round(float(r.get('Riktkurs idag',0.0) or 0.0),2)}** {r.get('Valuta','')}")
        st.write(f"- Om 1 år: **{round(float(r.get('Riktkurs om 1 år',0.0) or 0.0),2)}** {r.get('Valuta','')}")
        st.write(f"- Om 2 år: **{round(float(r.get('Riktkurs om 2 år',0.0) or 0.0),2)}** {r.get('Valuta','')}")
        st.write(f"- Om 3 år: **{round(float(r.get('Riktkurs om 3 år',0.0) or 0.0),2)}** {r.get('Valuta','')}")
        st.write(f"**CAGR 5 år:** {round(float(r.get('CAGR 5 år (%)',0.0) or 0.0),2)} %")
        st.write(f"**Årlig utdelning/aktie:** {round(float(r.get('Årlig utdelning',0.0) or 0.0),2)} {r.get('Valuta','')}")

    # Tabell med rådata för transparens
    show_cols = [
        "Ticker","Bolagsnamn","Valuta","Sektor","Aktuell kurs","_marketCap_raw",
        "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    vis = pd.DataFrame([r[show_cols].to_dict()])
    if "_marketCap_raw" in vis.columns:
        vis["_marketCap_fmt"] = vis["_marketCap_raw"].apply(_fmt_large_number)
    st.dataframe(vis, use_container_width=True, hide_index=True)

# app.py — Del 7/? — Portfölj, Investeringsförslag, Lägg till/uppdatera, Sidebar-kurser, MAIN

# --------- Investerings-helpers ---------------------------------------------

def _risk_label_from_mcap(mcap_raw: float) -> str:
    x = float(mcap_raw or 0.0)
    if x >= 200e9:   return "Mega"
    if x >= 10e9:    return "Large"
    if x >= 2e9:     return "Mid"
    if x >= 300e6:   return "Small"
    if x > 0:        return "Micro"
    return "–"

def _potential_pct(curr_px: float, target_px: float) -> float:
    try:
        curr, tgt = float(curr_px or 0.0), float(target_px or 0.0)
        if curr <= 0 or tgt <= 0: return 0.0
        return (tgt - curr) / curr * 100.0
    except Exception:
        return 0.0

# --------- Portfölj ----------------------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    if df is None or df.empty:
        st.info("Ingen data.")
        return

    port = df[df.get("Antal aktier", 0) > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(float(port['Total årlig utdelning (SEK)'].sum()),2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(float(port['Total årlig utdelning (SEK)'].sum())/12.0,2)} SEK")

    show_cols = ["Ticker","Bolagsnamn","Sektor","Antal aktier","Aktuell kurs","Valuta","_marketCap_raw","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    if "_marketCap_raw" in show_cols:
        port = port.copy()
        port["_MarketCap"] = port["_marketCap_raw"].apply(_fmt_large_number)
        # visa båda för transparens
        if "_MarketCap" not in show_cols:
            show_cols.insert(show_cols.index("_marketCap_raw")+1, "_MarketCap")

    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

# --------- Investeringsförslag ----------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    if df is None or df.empty:
        st.info("Ingen data.")
        return

    df = uppdatera_berakningar(df.copy(), user_rates)

    # Filtrering
    risk_filter = st.selectbox("Risklabel", ["Alla","Mega","Large","Mid","Small","Micro"], index=0)
    sector_filter = st.selectbox("Sektor", ["Alla"] + sorted(list({s for s in df.get("Sektor","").tolist() if str(s).strip()})), index=0)

    # välj riktkurs
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    base = df[(df[riktkurs_val] > 0) & (df["Aktuell kurs"] > 0)].copy()
    if risk_filter != "Alla":
        base["risklabel"] = base["_marketCap_raw"].apply(_risk_label_from_mcap)
        base = base[base["risklabel"] == risk_filter]
    if sector_filter != "Alla":
        base = base[base["Sektor"] == sector_filter]

    if base.empty:
        st.info("Inga bolag matchar filtren.")
        return

    # beräkningar
    base["Potential (%)"] = base.apply(lambda r: _potential_pct(r["Aktuell kurs"], r[riktkurs_val]), axis=1)
    base["P/S-snitt_4Q"] = base.apply(_ps_hist_snitt, axis=1)

    # sortering
    sort_choice = st.radio("Sortera efter", ["Störst potential", "Lägst P/S-snitt_4Q"], horizontal=True)
    if sort_choice == "Störst potential":
        base = base.sort_values(by=["Potential (%)","P/S-snitt_4Q"], ascending=[False, True]).reset_index(drop=True)
    else:
        base = base.sort_values(by=["P/S-snitt_4Q","Potential (%)"], ascending=[True, False]).reset_index(drop=True)

    # Navigering (robust)
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # visning
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    with st.expander("Detaljer", expanded=True):
        st.write(f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}")
        st.write(f"- **Utestående aktier:** {round(float(rad.get('Utestående aktier',0.0) or 0.0),2)} M")
        st.write(f"- **Market Cap (nu):** {_fmt_large_number(rad.get('_marketCap_raw',0.0))}")
        st.write(f"- **P/S TTM:** {round(float(rad.get('P/S',0.0) or 0.0),2)}")
        st.write(f"- **P/S-snitt (Q1–Q4):** {round(rad['P/S-snitt_4Q'],2)}")
        for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
            if k in rad:
                st.write(f"- **{k}:** {round(rad[k],2)} {rad['Valuta']}")
        # MCAP Q1–Q4 om finns
        mrows = []
        for i in range(1,5):
            v = rad.get(f"MCAP Q{i}", None)
            if v is not None and float(v) > 0:
                mrows.append((f"MCAP Q{i}", _fmt_large_number(v)))
        if mrows:
            st.write("**Historik (MCAP):** " + ", ".join([f"{k}: {v}" for k,v in mrows]))

    # enkel “köp-signal” på potential
    pot = float(rad["Potential (%)"] or 0.0)
    if pot >= 30:
        st.success(f"Indikation: Hög uppsida ({round(pot,1)}%).")
    elif pot >= 10:
        st.info(f"Indikation: Måttlig uppsida ({round(pot,1)}%).")
    else:
        st.warning(f"Indikation: Begränsad uppsida ({round(pot,1)}%).")

# --------- Lägg till / uppdatera --------------------------------------------

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
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

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
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
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%), Sektor, MarketCap via Yahoo")
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

        # Skriv in nya fält
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Sektor","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Sätt manuell TS + TS_ per fält
        ridx = df.index[df["Ticker"]==ticker][0]
        if datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Sektor"):     df.loc[ridx, "Sektor"] = data["Sektor"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if "_marketCap_raw" in data and data.get("_marketCap_raw") is not None:   df.loc[ridx, "_marketCap_raw"]  = float(data.get("_marketCap_raw") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # --- Äldst uppdaterade (alla spårade fält) ---
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

# --------- Sidebar – valutakurser -------------------------------------------

def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # init nycklar en (1) gång
    for k, v in (("rate_usd", STANDARD_VALUTAKURSER["USD"]),
                 ("rate_nok", STANDARD_VALUTAKURSER["NOK"]),
                 ("rate_cad", STANDARD_VALUTAKURSER["CAD"]),
                 ("rate_eur", STANDARD_VALUTAKURSER["EUR"])):
        if k not in st.session_state:
            st.session_state[k] = float(las_sparade_valutakurser().get(k.split("_")[1].upper(), v))

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", value=float(st.session_state.rate_usd), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", value=float(st.session_state.rate_nok), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", value=float(st.session_state.rate_cad), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", value=float(st.session_state.rate_eur), step=0.01, format="%.4f")

    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.sidebar.success(f"Valutakurser (källa: {provider}) hämtade.")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        # uppdatera via widgets (tilldela inte session_state direkt efter init av widgetar)
        st.session_state.rate_usd = float(auto_rates.get("USD", usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", eur))

    user_rates = {"USD": float(st.session_state.rate_usd),
                  "NOK": float(st.session_state.rate_nok),
                  "CAD": float(st.session_state.rate_cad),
                  "EUR": float(st.session_state.rate_eur),
                  "SEK": 1.0}

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
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return user_rates

# --------- MAIN --------------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Ladda data
    try:
        df = hamta_data()
    except Exception:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    # Säkerställ schema & typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Sidopanel: valutakurser
    user_rates = _sidebar_rates()

    st.sidebar.subheader("🛠️ Auto-uppdatering")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True)

    # Snabbknappar i sidebar
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("🔄 Uppdatera kurs (alla)"):
            # lätt variant – bara pris/valuta/mcap
            df2, log = auto_update_batch(df.copy(), user_rates, picker="alpha", batch_size=25, variant="pris", make_snapshot=make_snapshot, always_stamp=True)
            st.session_state["last_auto_log"] = log
            df = df2
    with c2:
        if st.button("🚀 Full auto (batch 10 äldst)"):
            df2, log = auto_update_batch(df.copy(), user_rates, picker="oldest", batch_size=10, variant="full", make_snapshot=make_snapshot, always_stamp=True)
            st.session_state["last_auto_log"] = log
            df = df2

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
