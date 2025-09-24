# app.py — Del 1/7
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
    """
    Skriv hela DataFrame till huvudbladet. Optionellt: skapa snapshot-flik först.
    Skydd: om df är tomt och "Tillåt tom skrivning" EJ är påslaget i sidopanelen -> skriv INTE.
    """
    if df is None or df.empty:
        if not st.session_state.get("destructive_ok", False):
            st.error("Blockerat: Försök att spara en TOM tabell. Aktivera 'Tillåt tom skrivning' i sidopanelen om du verkligen vill.")
            return
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

# app.py — Del 2/7
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

# Slutlig kolumnlista i databasen (inkl. nya mätare/nyckeltal och MCap-historik)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Nyckeltal (valfria)
    "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa", "Finansiell valuta",

    # Market cap nu + historik (lokal valuta)
    "MCap (nu)", "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
    "MCap Datum Q1", "MCap Datum Q2", "MCap Datum Q3", "MCap Datum Q4",

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
            if kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Ticker","Bolagsnamn","Valuta","Finansiell valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
                         "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"):
                df[kol] = ""
            else:
                # numeriska default
                df[kol] = 0.0
    # rensa ev. dubbletter i kolumnnamn
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
        "Utestående aktier",
        "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa",
        "MCap (nu)", "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    str_cols = ["Ticker","Bolagsnamn","Valuta","Finansiell valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
                "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Alla TS_ som sträng
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

# app.py — Del 3/7
# --- Yahoo-hjälpare & beräkningar & merge-hjälpare ---------------------------

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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR.
    Utökat: P/S (Yahoo) samt Market Cap (nu) om tillgängligt.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "P/S (Yahoo)": 0.0,
        "MCap (nu)": 0.0,
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
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency", None)
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

        # Extra fält
        ps_y = info.get("priceToSalesTrailing12Months", None)
        try:
            if ps_y is not None and float(ps_y) > 0:
                out["P/S (Yahoo)"] = float(ps_y)
        except Exception:
            pass

        mcap = info.get("marketCap", None)
        try:
            if mcap is not None and float(mcap) > 0:
                out["MCap (nu)"] = float(mcap)
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

def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    *,
    force_ts: bool = False
) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde.
    Uppdaterar TS_ för spårade fält, sätter 'Senast auto-uppdaterad' + källa.
    Returnerar True om något fält faktiskt ändrades.
    När force_ts=True stämplas datum/källa samt relevanta TS_-kolumner även om värdena inte ändras.
    """
    changed_fields = []
    touched_fields = []  # för TS när force_ts=True

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue

        # avgör om v är "meningsfullt"
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # siffror: tillåt >0; för vissa 0-vänliga fält tillåt >=0
            zero_ok_fields = {"Årlig utdelning","CAGR 5 år (%)","P/S (Yahoo)","MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4"}
            write_ok = (float(v) > 0) or (f in zero_ok_fields and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        if not write_ok:
            # även om vi inte skriver värdet kan det vara intressant för TS när force_ts=True
            if force_ts and f in TS_FIELDS:
                touched_fields.append(f)
            continue

        old = df.at[row_idx, f]
        # skriv endast om värdet faktiskt ändras
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[row_idx, f] = v
            changed_fields.append(f)
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
        else:
            # oförändrat värde – men om force_ts=True och f är TS-spårat, stämpla ändå
            if force_ts and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)

    # Om några fält skrevs – notera auto
    if changed_fields:
        _note_auto_update(df, row_idx, source)
        changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return True

    # Inga fält ändrades – men om force_ts=True och vi hade berörda fält eller vi vill stämpla ändå:
    if force_ts:
        # Stämpla TS för alla touched_fields som har TS-kolumn
        for f in touched_fields:
            _stamp_ts_for_field(df, row_idx, f)
        # Alltid stämpla auto-uppdaterad/källa vid force_ts
        _note_auto_update(df, row_idx, source)
        # Signalerar "ingen data ändrad" men TS/källa uppdaterad
        return False

    return False

# app.py — Del 4/7
# --- Datakällor: FMP, SEC (US + IFRS/6-K), Yahoo global fallback, Finnhub ----

# =============== FMP =========================================================
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.5))      # skonsam default
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

# (Valfri, fullare FMP – kvar om du vill använda i framtiden)
@st.cache_data(show_spinner=False, ttl=1800)
def hamta_fmp_falt(yahoo_ticker: str) -> dict:
    out = {"_debug": {}}
    sym = _fmp_pick_symbol(yahoo_ticker)

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

    qfull, sc_qfull = _fmp_get(f"api/v3/quote/{sym}", stable=False)
    out["_debug"]["quote_sc"] = sc_qfull
    market_cap = 0.0
    if isinstance(qfull, list) and qfull:
        q0 = qfull[0]
        if "price" in q0 and "Aktuell kurs" not in out:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
        if q0.get("marketCap") is not None:
            try: market_cap = float(q0["marketCap"]); out["MCap (nu)"] = market_cap
            except: pass

    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}", stable=False)
    out["_debug"]["ratios_ttm_sc"] = sc_rttm
    if isinstance(rttm, list) and rttm:
        try:
            v = rttm[0].get("priceToSalesTTM") or rttm[0].get("priceToSalesRatioTTM")
            if v is not None and float(v) > 0:
                out["P/S"] = float(v); out["_debug"]["ps_source"] = "ratios-ttm"
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

# ---------- robusta "instant" aktier (multi-class + IFRS) -------------------
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

# ---------- IFRS/GAAP kvartalsintäkter + valuta ------------------------------
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
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters)
    return rows

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
    Räknar implied shares, P/S (TTM) nu, P/S Q1–Q4 historik, samt MCap-historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price + extra
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","P/S (Yahoo)","MCap (nu)"):
        if y.get(k): out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(info.get("marketCap") or 0.0)
    if mcap <= 0 and float(out.get("MCap (nu)", 0.0)) > 0:
        mcap = float(out["MCap (nu)"])

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
    if q_rows and len(q_rows) >= 4:
        ttm_list = _ttm_windows(q_rows, need=4)
    else:
        ttm_list = []

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

    # P/S & MCap Q1–Q4 (historik)
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares * p
                    out[f"P/S Q{idx}"] = (mcap_hist / ttm_rev_px)
                    out[f"MCap Q{idx}"] = mcap_hist
                    out[f"MCap Datum Q{idx}"] = str(d_end)

    return out

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik + MCap-historik.
    Om CIK saknas → hamta_yahoo_global_combo.
    """
    out = {}
    cik = _sec_cik_for(ticker)
    if not cik:
        return hamta_yahoo_global_combo(ticker)

    facts, sc = _sec_companyfacts(cik)
    if sc != 200 or not isinstance(facts, dict):
        return hamta_yahoo_global_combo(ticker)

    # Yahoo-basics (+ P/S (Yahoo) + MCap (nu))
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "P/S (Yahoo)", "MCap (nu)"):
        if y.get(k): out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()

    # Shares: implied → fallback SEC robust
    implied = _implied_shares_from_yahoo(ticker, price=out.get("Aktuell kurs"), mcap=out.get("MCap (nu)", None))
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
    mcap_now = float(out.get("MCap (nu)", 0.0) or 0.0)
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used
        out["MCap (nu)"] = mcap_now

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

    # P/S & MCap Q1–Q4 historik
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCap Q{idx}"] = float(mcap_hist)
                    out[f"MCap Datum Q{idx}"] = str(d_end)

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

# =============== Auto-fetch pipeline (inkl. nyckeltal & MCap-historik) =======

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares, P/S TTM, P/S Q1–Q4, MCap-hist) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S, MCap) om saknas
      4) (valfritt) läs nyckeltal via Yahoo (Debt/Equity, marginaler, kassa)
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","_debug_shares_source",
            "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
            "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S (Yahoo)",
                  "P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                  "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
                  "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]:
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

    # 3) FMP light P/S/MCap om saknas
    try:
        if ("P/S" not in vals) or ("MCap (nu)" not in vals):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier"), "MCap (nu)": fmpl.get("MCap (nu)")}
            for k in ["P/S","MCap (nu)"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
            if ("Utestående aktier" not in vals) and (fmpl.get("Utestående aktier") not in (None, "", 0, 0.0)):
                vals["Utestående aktier"] = fmpl["Utestående aktier"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # 4) (valfritt) Yahoo nyckeltal
    try:
        yk = hamta_yahoo_nyckeltal(ticker) if 'hamta_yahoo_nyckeltal' in globals() else {}
        debug["yahoo_keys"] = yk
        for k in ["Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa","Finansiell valuta"]:
            v = yk.get(k)
            if v is not None:
                vals[k] = v
    except Exception as e:
        debug["yahoo_keys_err"] = str(e)

    return vals, debug

# app.py — Del 5/7
# --- Snapshots, auto-uppdatering, batch/våg + kontrollvy ---------------------

# ===== Snapshot till samma Google Sheet ======================================

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

# ===== Äldsta TS per rad (för sortering/kontroll) ============================

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

# ===== Robust radmatchning på ticker =========================================

def _find_row_index_by_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    """Returnera radindex för en ticker (case/whitespace-okänslig)."""
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return None
    mask = df["Ticker"].astype(str).str.strip().str.upper() == tkr
    hits = df.index[mask].tolist()
    return hits[0] if hits else None

# ===== Pris-only cachead hämtning ============================================

@st.cache_data(show_spinner=False, ttl=900)
def _price_only_fetch_cached(ticker: str) -> dict:
    """
    Hämtar endast: Aktuell kurs, Valuta, Bolagsnamn via Yahoo.
    Cache i 15 min för att kunna uppdatera alla snabbt utan spam.
    """
    out = {}
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info_dict(t)
        # pris
        px = info.get("regularMarketPrice")
        if px is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        if px is not None:
            out["Aktuell kurs"] = float(px)
        # valuta & namn
        if info.get("currency"):
            out["Valuta"] = str(info.get("currency")).upper()
        namn = info.get("shortName") or info.get("longName")
        if namn:
            out["Bolagsnamn"] = str(namn)
    except Exception:
        pass
    return out

# ===== Uppdatera endast pris för en ticker ===================================

def update_price_for_ticker(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, bool, str]:
    """Uppdaterar endast aktuell kurs/valuta/namn för ett enskilt bolag (stämplar alltid auto/källa)."""
    tkr = (ticker or "").upper().strip()
    if not tkr:
        return df, False, "Ingen ticker"
    ridx = _find_row_index_by_ticker(df, tkr)
    if ridx is None:
        return df, False, f"{tkr} hittades inte i tabellen."
    vals = _price_only_fetch_cached(tkr)
    if not vals:
        # stämpla ändå att försök gjorts
        _note_auto_update(df, ridx, source="Pris-only (Yahoo)")
        return df, False, f"Inga kursdata för {tkr}."
    changed = False
    if "Aktuell kurs" in vals and float(vals["Aktuell kurs"]) > 0:
        if float(df.at[ridx, "Aktuell kurs"] or 0.0) != float(vals["Aktuell kurs"]):
            df.at[ridx, "Aktuell kurs"] = float(vals["Aktuell kurs"])
            changed = True
    if vals.get("Valuta") and (not str(df.at[ridx, "Valuta"]).strip()):
        df.at[ridx, "Valuta"] = vals["Valuta"]
        changed = True
    if vals.get("Bolagsnamn") and (not str(df.at[ridx, "Bolagsnamn"]).strip()):
        df.at[ridx, "Bolagsnamn"] = vals["Bolagsnamn"]
        changed = True
    # Stämpla alltid auto + källa även om värdet blev oförändrat
    _note_auto_update(df, ridx, source="Pris-only (Yahoo)")
    return df, changed, "OK"

# ===== Uppdatera pris för ALLA tickers =======================================

def update_prices_all(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"priced": [], "miss": []}
    tickers = [str(t).strip().upper() for t in df["Ticker"].astype(str).tolist() if str(t).strip()]
    if not tickers:
        return df, log
    progress = st.sidebar.progress(0, text="Hämtar priser…")
    for i, tkr in enumerate(tickers):
        ridx = _find_row_index_by_ticker(df, tkr)
        if ridx is None:
            log["miss"].append(tkr)
            progress.progress((i+1)/len(tickers))
            continue
        df, changed, _ = update_price_for_ticker(df, tkr)
        if changed:
            log["priced"].append(tkr)
        progress.progress((i+1)/len(tickers))
    progress.empty()
    return df, log

# ===== Enskild full auto-uppdatering ========================================

def update_full_for_ticker(df: pd.DataFrame, ticker: str, user_rates: dict, *, force_ts: bool = True) -> tuple[pd.DataFrame, bool, dict]:
    """
    Kör full auto_fetch_for_ticker + apply_auto_updates_to_row för ett enskilt bolag.
    force_ts=True => stämpla datum/källa/TS även om värdena är oförändrade.
    """
    tkr = (ticker or "").upper().strip()
    if not tkr:
        return df, False, {"err": "Ingen ticker"}
    ridx = _find_row_index_by_ticker(df, tkr)
    if ridx is None:
        return df, False, {"err": f"{tkr} hittades inte"}
    new_vals, debug = auto_fetch_for_ticker(tkr)
    changed = apply_auto_updates_to_row(
        df, ridx, new_vals,
        source="Auto (Enskild SEC/Yahoo→Finnhub→FMP→Nyckeltal)",
        changes_map={},
        force_ts=force_ts
    )
    # Alltid räkna om efter en uppdatering (changed kan vara False om bara TS/källa stämplades)
    df = uppdatera_berakningar(df, user_rates)
    return df, changed, debug

# ===== Klassisk auto-uppdatering för hela listan =============================

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, *, force_ts: bool = True):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla nya värden.
    Stämplar TS_ per fält, samt 'Senast auto-uppdaterad' + källa.
    force_ts=True => stämplar datum/källa/TS även om värdena inte ändras.
    """
    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    progress = st.sidebar.progress(0, text="Auto-uppdaterar…")
    status = st.sidebar.empty()

    total = len(df)
    any_changed = False

    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1))
            continue

        status.write(f"Uppdaterar {i+1}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(
                df, idx, new_vals,
                source="Auto (SEC/Yahoo→Finnhub→FMP→Nyckeltal)",
                changes_map=log["changed"],
                force_ts=force_ts
            )
            if not changed:
                log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
            any_changed = any_changed or changed
            if i < 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(total,1))

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    if any_changed or force_ts:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Uppdateringar/TS sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# ===== Vågvis körning (kö / batch) ===========================================

def start_wave(df: pd.DataFrame, mode: str = "oldest"):
    """
    Skapar en vågkö av tickers enligt valt mode.
      - 'oldest': sortera efter äldsta TS (alla spårade fält)
      - 'alphabetic': A–Ö efter Bolagsnamn/Ticker
    """
    work = df.copy()
    if mode == "oldest":
        work = add_oldest_ts_col(work)
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        work = work.sort_values(by=["Bolagsnamn","Ticker"])
    queue = [str(t).strip().upper() for t in work["Ticker"].tolist() if str(t).strip()]
    st.session_state["wave_queue"] = queue
    st.session_state["wave_done"] = []
    st.session_state["wave_changed"] = []
    st.session_state["wave_miss"] = []
    st.session_state["wave_started_at"] = _ts_datetime()
    st.session_state["wave_mode"] = mode

def run_wave_step(df: pd.DataFrame, user_rates: dict, *, batch_size: int = 10, make_snapshot: bool = True, force_ts: bool = True):
    """
    Kör nästa batch i vågkön. force_ts=True => stämpla TS/källa även om värdena inte ändras.
    """
    queue = st.session_state.get("wave_queue", []) or []
    done = st.session_state.get("wave_done", []) or []
    changed_list = st.session_state.get("wave_changed", []) or []
    miss = st.session_state.get("wave_miss", []) or []

    if not queue:
        st.info("Ingen aktiv vågkö. Tryck 'Starta våg' först.")
        return df, {"processed": 0, "remaining": 0}

    to_process = queue[:batch_size]
    remaining_after = queue[batch_size:]

    progress = st.sidebar.progress(0, text="Kör batch…")
    for i, tkr in enumerate(to_process):
        ridx = _find_row_index_by_ticker(df, tkr)
        if ridx is None:
            st.warning(f"{tkr}: fanns inte i DataFrame – hoppar.")
            miss.append(tkr)
            progress.progress((i+1)/len(to_process))
            continue
        try:
            new_vals, dbg = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(
                df, ridx, new_vals,
                source="Auto (Våg SEC/Yahoo→Finnhub→FMP→Nyckeltal)",
                changes_map={},
                force_ts=force_ts
            )
            # räkna om efter varje (säkert men lite långsammare)
            df = uppdatera_berakningar(df, user_rates)
            done.append(tkr)
            if changed:
                changed_list.append(tkr)
        except Exception as e:
            miss.append(tkr)
        progress.progress((i+1)/len(to_process))
    progress.empty()

    # spara efter varje batch
    spara_data(df, do_snapshot=make_snapshot)

    # uppdatera state
    st.session_state["wave_queue"] = remaining_after
    st.session_state["wave_done"] = done
    st.session_state["wave_changed"] = changed_list
    st.session_state["wave_miss"] = miss

    info = {
        "processed": len(to_process),
        "remaining": len(remaining_after),
        "changed_in_batch": len(changed_list),
        "miss_total": len(miss),
    }
    return df, info

# ===== Debugvy (enstaka ticker) ==============================================

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

# ===== Hjälplistor & Kontroll-vy =============================================

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

    # 3) Senaste körlogg (Auto)
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

# app.py — Del 6/7
# --- Analys, Portfölj, Investeringsförslag, Form + hjälpare ------------------

# === Hjälpare: snyggt formatera stora tal & risklabel ========================

def format_big_number_sv(n: float, valuta: str | None = None, decimals: int = 1) -> str:
    """Formaterar stora tal som '1,2 miljoner/miljarder/biljoner [VALUTA]'. Returnerar '–' vid ogiltigt."""
    try:
        n = float(n)
    except Exception:
        return "–"
    absn = abs(n)
    suf = f" {valuta}" if valuta else ""
    if absn >= 1e12:
        return f"{n/1e12:.{decimals}f} biljoner{suf}"
    if absn >= 1e9:
        return f"{n/1e9:.{decimals}f} miljarder{suf}"
    if absn >= 1e6:
        return f"{n/1e6:.{decimals}f} miljoner{suf}"
    if absn >= 1e3:
        return f"{n/1e3:.{decimals}f} tusen{suf}"
    return f"{n:,.0f}{suf}".replace(",", " ")

def risk_bucket_from_mcap_sek(mcap_sek: float) -> str:
    """
    Grov riskklassning baserat på marknadsvärde i SEK.
    Trösklar (ungefärliga):
      Micro < 3 mdr, Small 3–25, Mid 25–100, Large 100–500, Mega > 500.
    """
    try:
        x = float(mcap_sek or 0.0)
    except Exception:
        x = 0.0
    if x <= 0:
        return "Okänd"
    if x < 3e9:
        return "Micro"
    if x < 25e9:
        return "Small"
    if x < 100e9:
        return "Mid"
    if x < 500e9:
        return "Large"
    return "Mega"

# === (Ny) Yahoo-nyckeltal: D/E, marginaler, kassa ============================

def hamta_yahoo_nyckeltal(ticker: str) -> dict:
    """
    Hämtar Debt/Equity, bruttomarginal, nettomarginal, kassa samt finansiell valuta via Yahoo.
    Notera: marginaler kommer i andel (0.45) → konverteras till procent (45.0).
    """
    out = {"Debt/Equity": 0.0, "Bruttomarginal (%)": 0.0, "Nettomarginal (%)": 0.0, "Kassa": 0.0, "Finansiell valuta": ""}
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info_dict(t)
        de = info.get("debtToEquity")
        if de is not None:
            try: out["Debt/Equity"] = float(de)
            except: pass
        gm = info.get("grossMargins")
        if gm is not None:
            try: out["Bruttomarginal (%)"] = round(float(gm)*100.0, 2)
            except: pass
        pm = info.get("profitMargins")
        if pm is not None:
            try: out["Nettomarginal (%)"] = round(float(pm)*100.0, 2)
            except: pass
        cash = info.get("totalCash")
        if cash is not None:
            try: out["Kassa"] = float(cash)
            except: pass
        fccy = info.get("financialCurrency")
        if fccy:
            out["Finansiell valuta"] = str(fccy).upper()
    except Exception:
        pass
    return out

# === Analysvy ================================================================

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

    # extra: visa market cap snyggt + risk
    vx = hamta_valutakurs(r.get("Valuta",""), user_rates)
    mcap_now = float(r.get("MCap (nu)", 0.0) or 0.0)
    mcap_str = format_big_number_sv(mcap_now, r.get("Valuta",""))
    mcap_sek_str = format_big_number_sv(mcap_now * vx, "SEK")
    label = risk_bucket_from_mcap_sek(mcap_now * vx)

    st.markdown(f"**Market cap (nu):** {mcap_str} ({mcap_sek_str}) • **Risk:** `{label}`")

    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
        "P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa","Finansiell valuta",
        "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# === Portföljvy ==============================================================

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True, hide_index=True
    )

# === Investeringsförslag =====================================================

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    # Ny: Riskfilter
    risk_filter = st.selectbox("Riskklass (market cap, SEK)", ["Alla","Micro","Small","Mid","Large","Mega"], index=0)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Beräkna potential, risklabel, P/S 4Q-snitt, MCap SEK
    def _ps_avg4(row):
        arr = [float(row.get(f"P/S Q{i}", 0.0) or 0.0) for i in range(1,5)]
        arr = [x for x in arr if x > 0]
        return round(float(np.mean(arr)) if arr else 0.0, 2)

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0
    base["P/S 4Q-snitt"] = base.apply(_ps_avg4, axis=1)

    # mcap SEK och risklabel
    base["vx"] = base["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    base["MCap SEK"] = base["MCap (nu)"] * base["vx"]
    base["Risklabel"] = base["MCap SEK"].apply(risk_bucket_from_mcap_sek)

    # Filtrera på risk
    if risk_filter != "Alla":
        base = base[base["Risklabel"] == risk_filter]
        if base.empty:
            st.info("Inga bolag matchar riskfiltret.")
            return

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

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

    # Portföljandelar & köpberäkning
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r0 = port[port["Ticker"] == rad["Ticker"]]
        if not r0.empty:
            nuv_innehav = float(r0["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Snygg MCap-strängar
    mcap_now_local = float(rad.get("MCap (nu)", 0.0) or 0.0)
    mcap_now_sek   = mcap_now_local * vx
    mcap_local_str = format_big_number_sv(mcap_now_local, str(rad['Valuta']))
    mcap_sek_str   = format_big_number_sv(mcap_now_sek, "SEK")

    # Varningshint om P/S sticker från snittet
    ps_now = float(rad.get("P/S", 0.0) or 0.0)
    ps_avg = float(rad.get("P/S 4Q-snitt", rad.get("P/S 4Q-snitt", 0.0)) or 0.0)
    if ps_now > 0 and ps_avg > 0 and ps_now > ps_avg * 10:
        st.warning("P/S (nu) är >10× över 4Q-snittet – kontrollera datakällor eller extraordinära händelser.")

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    lines = [
        f"- **Riskklass:** `{rad['Risklabel']}`",
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']} ({round(kurs_sek,2)} SEK)",
        f"- **Market cap (nu):** {mcap_local_str} ({mcap_sek_str})",
        f"- **P/S (nu):** {round(ps_now,2)}",
        f"- **P/S (4Q-snitt):** {round(rad['P/S 4Q-snitt'],2)}",
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

    with st.expander("📜 Detaljer: MCap-historik & nyckeltal"):
        # MCap-historik tabell
        hist_rows = []
        for q in range(1,5):
            mv = float(rad.get(f"MCap Q{q}", 0.0) or 0.0)
            dt = str(rad.get(f"MCap Datum Q{q}", "") or "")
            if mv > 0 or dt:
                hist_rows.append({
                    "Kvartal": f"Q{q}",
                    "Datum": dt,
                    f"MCap ({rad['Valuta']})": format_big_number_sv(mv, str(rad['Valuta'])),
                    "MCap (SEK)": format_big_number_sv(mv * vx, "SEK")
                })
        if hist_rows:
            st.table(pd.DataFrame(hist_rows))
        else:
            st.write("Ingen MCap-historik tillgänglig.")

        # Nyckeltal
        st.write("**Nyckeltal (senaste):**")
        st.json({
            "Debt/Equity": float(rad.get("Debt/Equity", 0.0) or 0.0),
            "Bruttomarginal (%)": float(rad.get("Bruttomarginal (%)", 0.0) or 0.0),
            "Nettomarginal (%)": float(rad.get("Nettomarginal (%)", 0.0) or 0.0),
            "Kassa": format_big_number_sv(float(rad.get("Kassa", 0.0) or 0.0), rad.get("Finansiell valuta","")),
            "Finansiell valuta": str(rad.get("Finansiell valuta","") or "")
        })

# === Lägg till / uppdatera (enskild) ========================================

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["-_dummy" if "_oldest_any_ts_fill" not in work.columns else "_oldest_any_ts_fill","Bolagsnamn"])
        if "_oldest_any_ts_fill" in work.columns:
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

    # Snabbetiketter för datum/källa (återkommande önskemål)
    if not bef.empty:
        manu = str(bef.get("Senast manuellt uppdaterad","") or "")
        auto = str(bef.get("Senast auto-uppdaterad","") or "")
        src  = str(bef.get("Senast uppdaterad källa","") or "")
        st.caption(f"🕒 **Manuellt:** {manu or '–'}  •  🤖 **Auto:** {auto or '–'}  •  🔗 **Källa:** {src or '–'}")

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            st.caption(f"TS: {str(bef.get('TS_P/S','') or '–')}")
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            st.caption(f"TS: {str(bef.get('TS_P/S Q1','') or '–')}")
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            st.caption(f"TS: {str(bef.get('TS_P/S Q2','') or '–')}")
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            st.caption(f"TS: {str(bef.get('TS_P/S Q3','') or '–')}")
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
            st.caption(f"TS: {str(bef.get('TS_P/S Q4','') or '–')}")
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            st.caption(f"TS: {str(bef.get('TS_Omsättning idag','') or '–')}")
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            st.caption(f"TS: {str(bef.get('TS_Omsättning nästa år','') or '–')}")

            st.markdown("**Vid spara uppdateras också automatiskt:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")
            st.write("- Datumstämplar uppdateras även om värdet inte ändrats")

        col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
        with col_btn1:
            pris_only = st.form_submit_button("💹 Uppdatera kurs")
        with col_btn2:
            full_one  = st.form_submit_button("🤖 Full auto (endast denna)")
        with col_btn3:
            spar = st.form_submit_button("💾 Spara")

    # Knapp: Kurs-only
    if pris_only and (ticker or "").strip():
        ridx_tmp = _find_row_index_by_ticker(df, ticker)
        if ridx_tmp is None and not bef.empty:
            # fallback till befintlig rad
            ticker = bef.get("Ticker","")
        df, changed, msg = update_price_for_ticker(df, ticker)
        if changed:
            df = uppdatera_berakningar(df, user_rates)
            spara_data(df)
            st.success(f"Kurs uppdaterad för {ticker}.")
        else:
            spara_data(df)  # stämplar auto även om oförändrat
            st.info(f"Inga kursändringar för {ticker}. ({msg})")

    # Knapp: Full auto enskild
    if full_one and (ticker or "").strip():
        df, changed, dbg = update_full_for_ticker(df, ticker, user_rates, force_ts=True)
        spara_data(df)
        if changed:
            st.success(f"Auto-uppdaterade {ticker}.")
        else:
            st.info(f"Inga ändringar hittades vid auto-uppdatering av {ticker} (TS/källa uppdaterad).")
        with st.expander("Debug-data (källor)"):
            st.json(dbg)

    # Knapp: Spara (manuellt)
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
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Sätt manuell TS + TS_ per fält (även om värdet inte ändras sätter vi datum om datum_sätt)
        ridx = _find_row_index_by_ticker(df, ticker)
        if ridx is None:
            ridx = df.index[df["Ticker"]==ticker][0]
        if datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if "P/S (Yahoo)" in data and data.get("P/S (Yahoo)") is not None:         df.loc[ridx, "P/S (Yahoo)"]     = float(data.get("P/S (Yahoo)") or 0.0)
        if "MCap (nu)" in data and data.get("MCap (nu)") is not None:             df.loc[ridx, "MCap (nu)"]       = float(data.get("MCap (nu)") or 0.0)

        # Nyckeltal
        yk = hamta_yahoo_nyckeltal(ticker)
        for k in ["Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa","Finansiell valuta"]:
            if yk.get(k) is not None:
                df.loc[ridx, k] = yk[k]

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

# app.py — Del 7/7
# --- MAIN --------------------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # ====== Sidopanel: säkerhet & inställningar ==============================
    st.sidebar.header("🛡️ Säkerhet & skrivskydd")
    st.sidebar.checkbox("💥 Tillåt tom skrivning", value=st.session_state.get("destructive_ok", False), key="destructive_ok",
                        help="Avbockad = appen blockerar alla försök att skriva en TOM tabell till Google Sheets.")

    st.sidebar.markdown("---")
    st.sidebar.header("💱 Valutakurser → SEK")

    # Hantera state-nycklar för att kunna uppdatera number_input visuellt
    if "usd_rate" not in st.session_state or "nok_rate" not in st.session_state \
       or "cad_rate" not in st.session_state or "eur_rate" not in st.session_state:
        saved_rates_boot = las_sparade_valutakurser()
        st.session_state.usd_rate = float(saved_rates_boot.get("USD", STANDARD_VALUTAKURSER["USD"]))
        st.session_state.nok_rate = float(saved_rates_boot.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        st.session_state.cad_rate = float(saved_rates_boot.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        st.session_state.eur_rate = float(saved_rates_boot.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

    usd = st.sidebar.number_input("USD → SEK", value=st.session_state.usd_rate, step=0.01, format="%.4f", key="usd_rate")
    nok = st.sidebar.number_input("NOK → SEK", value=st.session_state.nok_rate, step=0.01, format="%.4f", key="nok_rate")
    cad = st.sidebar.number_input("CAD → SEK", value=st.session_state.cad_rate, step=0.01, format="%.4f", key="cad_rate")
    eur = st.sidebar.number_input("EUR → SEK", value=st.session_state.eur_rate, step=0.01, format="%.4f", key="eur_rate")

    col_rates1, col_rates2, col_rates3 = st.sidebar.columns(3)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade"):
            st.cache_data.clear()
            saved = las_sparade_valutakurser()
            st.session_state.usd_rate = float(saved.get("USD", usd))
            st.session_state.nok_rate = float(saved.get("NOK", nok))
            st.session_state.cad_rate = float(saved.get("CAD", cad))
            st.session_state.eur_rate = float(saved.get("EUR", eur))
            st.experimental_rerun()
    with col_rates3:
        if st.button("🌐 Hämta auto"):
            auto_rates, misses, provider = hamta_valutakurser_auto()
            st.session_state.usd_rate = float(auto_rates.get("USD", usd))
            st.session_state.nok_rate = float(auto_rates.get("NOK", nok))
            st.session_state.cad_rate = float(auto_rates.get("CAD", cad))
            st.session_state.eur_rate = float(auto_rates.get("EUR", eur))
            if misses:
                st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
            st.sidebar.success(f"Valutakurser hämtade (källa: {provider}).")
            st.experimental_rerun()

    user_rates = {"USD": st.session_state.usd_rate, "NOK": st.session_state.nok_rate,
                  "CAD": st.session_state.cad_rate, "EUR": st.session_state.eur_rate, "SEK": 1.0}

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.experimental_rerun()

    # ====== Läs data =========================================================
    df = hamta_data()
    if df.empty:
        # Initial säkerställning – spara INTE direkt (skydd kan blocka).
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        st.warning("Google Sheet verkar sakna data eller är tomt. Lägger upp vyer men sparar inte tom tabell.")
    else:
        # Säkerställ schema, migrera och typer
        df = säkerställ_kolumner(df)
        df = migrera_gamla_riktkurskolumner(df)
        df = konvertera_typer(df)

    # ====== Auto-uppdatering & vågor i sidopanel =============================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Auto-uppdatering")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True, help="Säkerhetskopia skapas innan skrivning till huvudbladet.")

    # Pris-only för alla tickers
    if st.sidebar.button("💹 Uppdatera priser (alla)"):
        df, logp = update_prices_all(df)
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success(f"Priser uppdaterade. Ändrade: {len(logp.get('priced', []))}, missar: {len(logp.get('miss', []))}")

    # Full auto alla
    if st.sidebar.button("🔄 Full auto (alla)"):
        df, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot, force_ts=True)
        st.session_state["last_auto_log"] = log

    # Våg-körning (batch)
    st.sidebar.subheader("🌊 Vågvis uppdatering")
    mode = st.sidebar.selectbox("Kö-ordning", ["oldest","alphabetic"], index=0, help="Välj ordning för kön.")
    batch_sz = st.sidebar.number_input("Batch-storlek", min_value=1, max_value=100, value=10, step=1)
    col_w1, col_w2, col_w3 = st.sidebar.columns(3)
    with col_w1:
        if st.button("Starta våg"):
            start_wave(df, mode=mode)
            st.sidebar.success("Vågkö skapad.")
    with col_w2:
        if st.button("Kör nästa batch"):
            df, info = run_wave_step(df, user_rates, batch_size=int(batch_sz), make_snapshot=make_snapshot, force_ts=True)
            st.sidebar.info(f"Bearbetade {info['processed']} • Kvar {info['remaining']}")
    with col_w3:
        if st.button("Rensa kö"):
            for k in ["wave_queue","wave_done","wave_changed","wave_miss","wave_started_at","wave_mode"]:
                st.session_state.pop(k, None)
            st.sidebar.success("Kön återställd.")

    # Visuell status för vågen
    q = st.session_state.get("wave_queue", []) or []
    d = st.session_state.get("wave_done", []) or []
    ch = st.session_state.get("wave_changed", []) or []
    mi = st.session_state.get("wave_miss", []) or []
    if q or d or ch or mi:
        st.sidebar.markdown(f"**Köstatus:** kvar: {len(q)} • klara: {len(d)} • ändrade: {len(ch)} • miss: {len(mi)}")

    # ====== Meny =============================================================
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj","Debug"])

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        df = uppdatera_berakningar(df, user_rates)
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)
    elif meny == "Debug":
        st.subheader("🔍 Debug – testa datakällor för en ticker")
        t_inp = st.text_input("Ticker (t.ex. AAPL, SHOP, NVDA, VOLV-B.ST)")
        if st.button("Visa källdata") and t_inp.strip():
            debug_test_single_ticker(t_inp.strip().upper())

if __name__ == "__main__":
    main()
