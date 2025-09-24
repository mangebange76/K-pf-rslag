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
    """Skriv hela DataFrame till huvudbladet. Optionellt: skapa snapshot-flik först."""
    # skydd: om df är tom och användaren inte uttryckligen tillåtit tom skrivning – blockera
    if df is None or df.empty:
        if not st.session_state.get("destructive_ok", False):
            st.error("Skrivning av TOM tabell blockerades (säkerhet). Bocka i 'Tillåt tom skrivning' om du vill skriva ändå.")
            return
    if do_snapshot and df is not None and not df.empty:
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

# Slutlig kolumnlista i databasen
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Market cap (nu + historik)
    "MCap (nu)", "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
    "MCap Datum Q1", "MCap Datum Q2", "MCap Datum Q3", "MCap Datum Q4",

    # Nyckeltal
    "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa", "Finansiell valuta",

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
            if any(x in kol.lower() for x in [
                "kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier",
                "snitt","mcap","debt","marginal","kassa"
            ]):
                # numeriska default
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol.startswith("MCap Datum"):
                df[kol] = ""  # datumtext för historik
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"):
                df[kol] = ""
            else:
                df[kol] = ""
    # ta bort ev. dubletter i kolumnnamn
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
        "MCap (nu)", "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
        "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    str_cols = ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
                "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4","Finansiell valuta"]
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR,
    samt 'P/S (Yahoo)' (TTM) och 'MCap (nu)' om tillgängligt.
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

        # Pris
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        # Valuta & namn
        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        # Utdelning
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # CAGR approx
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

        # P/S (Yahoo) – vanligaste nyckeln i yfinance-info
        ps_y = info.get("priceToSalesTrailing12Months")
        if ps_y is None:
            ps_y = info.get("priceToSalesTTM") or info.get("priceToSalesRatioTTM")
        try:
            if ps_y is not None and float(ps_y) > 0:
                out["P/S (Yahoo)"] = float(ps_y)
        except Exception:
            pass

        # Market cap
        mcap = info.get("marketCap")
        try:
            if mcap is not None and float(mcap) > 0:
                out["MCap (nu)"] = float(mcap)
        except Exception:
            pass

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
    Om force_ts=True stämplas TS/källa även om värdena inte förändras.
    """
    changed_fields = []
    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f] if f in df.columns else None

        # bedömning om vi ska skriva nytt värde
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # skriv positiva (eller noll för vissa icke-P/S-fält)
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        # skriv om nytt och skiljer sig
        if write_ok and ((pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))):
            df.at[row_idx, f] = v
            changed_fields.append(f)
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)

        # även om värdet inte skrevs (samma som innan), stämpla TS om force_ts och f är spårat
        elif force_ts and (f in TS_FIELDS):
            _stamp_ts_for_field(df, row_idx, f)

    # stämpla auto-uppdatering/källa om något ändrats ELLER force_ts=True
    if changed_fields or force_ts:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return bool(changed_fields)

    return False

# app.py — Del 4/7
# --- Datakällor: FMP, SEC (US + IFRS/6-K/10-K/20-F), Yahoo fallback, Finnhub --

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
            try: out["_marketCap"] = float(q0["marketCap"])
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
    Hämtar upp till 'max_quarters' kvartalsintäkter (≈3 mån) från SEC companyfacts.
    Stöd för Q4 i 10-K (US GAAP) samt 20-F/40-F (IFRS). Returnerar (rows, unit)
    där rows = [(end_date, value), ...] sorterade nyast→äldst.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A", "10-K", "10-K/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "20-F", "20-F/A", "40-F", "40-F/A", "10-Q", "10-Q/A")}),
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

    def _collect_rows_for_unit(arr, forms_ok):
        tmp = []
        for it in arr:
            form = (it.get("form") or "").upper()
            if forms_ok and not any(f in form for f in forms_ok):
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
            # kvartalsfilter ≈ 3 månader
            if dur is None or dur < 70 or dur > 100:
                continue
            try:
                v = float(val)
                tmp.append((end, v))
            except Exception:
                pass
        if not tmp:
            return []
        # deduplicera per end-date
        ded = {}
        for end, v in tmp:
            ded[end] = v
        rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
        return rows

    facts_all = (facts.get("facts") or {})
    for taxo, cfg in taxos:
        gaap = facts_all.get(taxo, {})
        best_rows = []
        best_unit = None

        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})

            # 1) försök prefererade valutor
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list):
                    continue
                rows = _collect_rows_for_unit(arr, cfg["forms"])
                if rows:
                    return rows, unit_code

            # 2) fallback: vilken valuta som helst som ger flest 3m-kvartal
            candidate_best = []
            candidate_unit = None
            for unit_code, arr in units.items():
                if not isinstance(arr, list):
                    continue
                ukey = str(unit_code).upper()
                if not (2 <= len(ukey) <= 6):
                    continue
                rows = _collect_rows_for_unit(arr, cfg["forms"])
                if rows and (len(rows) > len(candidate_best)):
                    candidate_best = rows
                    candidate_unit = unit_code
            if candidate_best:
                if len(candidate_best) > len(best_rows):
                    best_rows = candidate_best
                    best_unit = candidate_unit

        if best_rows:
            return best_rows, best_unit

    return [], None

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

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q/10-K eller IFRS 6-K/20-F),
    pris/valuta/namn från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik. Sparar även MCap-historik.
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "MCap (nu)", "P/S (Yahoo)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()

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
        out["Utestående aktier"] = shares_used / 1e6  # i miljoner

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

    # P/S (TTM) nu via SEC (om möjligt)
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik + MCap Q1–Q4 (baserat på samma shares_used)
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            # mcap historisk via close * shares_used
            px = px_map.get(d_end, None)
            if px and px > 0:
                mcap_hist = shares_used * float(px)
                out[f"MCap Q{idx}"] = float(mcap_hist)
                out[f"MCap Datum Q{idx}"] = d_end.strftime("%Y-%m-%d")
                if ttm_rev_px and ttm_rev_px > 0:
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    return out

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik + MCap Q1–Q4.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price + Yahoo P/S & MCap
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","P/S (Yahoo)","MCap (nu)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()
    mcap = float(out.get("MCap (nu)", 0.0) or 0.0)

    # Implied shares → fallback sharesOutstanding
    shares = 0.0
    if mcap > 0 and px > 0:
        shares = mcap / px
        out["_debug_shares_source"] = "Yahoo implied (mcap/price)"
    else:
        info = _yfi_info_dict(t)
        so = info.get("sharesOutstanding")
        try:
            so = float(so or 0.0)
        except Exception:
            so = 0.0
        if so > 0 and px > 0:
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
    info2 = _yfi_info_dict(t)
    fin_ccy = str(info2.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        conv = _fx_rate_cached(fin_ccy, px_ccy) or 1.0
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # Market cap (nu)
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px
        out["MCap (nu)"] = mcap

    # P/S (TTM) nu
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historisk) + MCap Q1–Q4
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            p = px_map.get(d_end)
            if p and p > 0:
                mcap_hist = shares * p
                out[f"MCap Q{idx}"] = float(mcap_hist)
                out[f"MCap Datum Q{idx}"] = d_end.strftime("%Y-%m-%d")
                if ttm_rev_px and ttm_rev_px > 0:
                    out[f"P/S Q{idx}"] = (mcap_hist / ttm_rev_px)

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

# app.py — Del 5/7
# --- Snapshots, auto-uppdatering, batchvågor, debug & kontroll ----------------

# ========== Snapshot till separat flik =======================================

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

# ========== TS-inspektörer för kontrollvy ====================================

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

# ========== Hjälpare: hitta radindex via ticker ==============================

def _find_row_index_by_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    try:
        mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
        idxs = list(df[mask].index)
        return idxs[0] if idxs else None
    except Exception:
        return None

# ========== Auto-fetch pipeline (enskild ticker) =============================

def auto_fetch_for_ticker(ticker: str) -> tuple[dict, dict]:
    """
    Pipeline:
      1) SEC + Yahoo (inkl. fallback för Q4 via 10-K/20-F/40-F) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S) om saknas
      4) Yahoo nyckeltal (D/E, marginaler, kassa)
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
            "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4","_debug_shares_source"
        ]}
        for k in [
            "Bolagsnamn","Valuta","Aktuell kurs",
            "Utestående aktier","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
            "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"
        ]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) Finnhub estimat om saknas
    try:
        need_rev_today = ("Omsättning idag" not in vals) or (float(vals.get("Omsättning idag", 0.0)) <= 0.0)
        need_rev_next  = ("Omsättning nästa år" not in vals) or (float(vals.get("Omsättning nästa år", 0.0)) <= 0.0)
        if need_rev_today or need_rev_next:
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
        if ("P/S" not in vals) or (float(vals.get("P/S", 0.0)) <= 0.0):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier")}
            if fmpl.get("P/S") not in (None, "", 0, 0.0):
                vals["P/S"] = fmpl["P/S"]
            # shares fallback om vi saknar
            if ("Utestående aktier" not in vals) and (fmpl.get("Utestående aktier") not in (None, "", 0, 0.0)):
                vals["Utestående aktier"] = fmpl["Utestående aktier"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # 4) Yahoo nyckeltal (D/E, marginaler, kassa, fin-valuta)
    try:
        nyck = hamta_yahoo_nyckeltal(ticker)
        debug["yahoo_keys"] = nyck
        for k in ["Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa","Finansiell valuta"]:
            v = nyck.get(k, None)
            if v is not None:
                vals[k] = v
    except Exception as e:
        debug["yahoo_keys_err"] = str(e)

    return vals, debug

# ========== Enskilt pris / enskild full auto =================================

def update_price_for_ticker(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, bool, str]:
    """Uppdaterar endast 'Aktuell kurs' och stämplar auto/källa."""
    i = _find_row_index_by_ticker(df, ticker)
    if i is None:
        return df, False, f"{ticker} hittades inte i tabellen."
    y = hamta_yahoo_fält(ticker)
    px = float(y.get("Aktuell kurs", 0.0) or 0.0)
    if px > 0:
        old = float(df.at[i, "Aktuell kurs"] or 0.0)
        changed = (abs(px - old) > 1e-9)
        df.at[i, "Aktuell kurs"] = px
        _note_auto_update(df, i, source="Pris-only (Yahoo)")
        return df, changed, "OK"
    else:
        _note_auto_update(df, i, source="Pris-only (Yahoo miss)")
        return df, False, "Kunde inte hämta pris"

def update_full_for_ticker(df: pd.DataFrame, ticker: str, user_rates: dict, *, force_ts: bool = True):
    """
    Full auto för en ticker: SEC/Yahoo → Finnhub → FMP → Yahoo nyckeltal.
    Returnerar (df, changed: bool, debug: dict).
    """
    i = _find_row_index_by_ticker(df, ticker)
    if i is None:
        st.warning(f"{ticker} hittades inte i tabellen.")
        return df, False, {}

    new_vals, debug = auto_fetch_for_ticker(ticker)
    changed = apply_auto_updates_to_row(
        df, i, new_vals, source="Auto (SEC/Yahoo→Finnhub→FMP→Yahoo Keys)",
        changes_map=st.session_state.setdefault("auto_changes_map", {}),
        force_ts=force_ts
    )
    # räkna om
    df = uppdatera_berakningar(df, user_rates)
    return df, changed, debug

# ========== Auto-uppdatera ALLA ==============================================

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, *, force_ts: bool = True):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla nya värden.
    Stämplar TS_ per fält, samt 'Senast auto-uppdaterad' + källa.
    """
    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()

    total = len(df)
    any_changed = False

    for idx_vis, (_, row) in enumerate(df.reset_index().iterrows(), start=1):
        i = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress(idx_vis / max(total, 1))
            continue

        status.write(f"Uppdaterar {idx_vis}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(
                df, i, new_vals, source="Auto (SEC/Yahoo→Finnhub→FMP→Yahoo Keys)",
                changes_map=log["changed"], force_ts=force_ts
            )
            if not changed:
                log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
            any_changed = any_changed or changed
            if idx_vis <= 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress(idx_vis / max(total, 1))

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    if any_changed or force_ts:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Data sparad (TS/källa uppdaterad).")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# ========== Pris-only ALLA ====================================================

def update_prices_all(df: pd.DataFrame):
    """
    Uppdaterar endast 'Aktuell kurs' för alla tickers. Returnerar logg.
    """
    log = {"priced": [], "miss": []}
    progress = st.sidebar.progress(0)
    total = len(df)
    for idx_vis, (_, row) in enumerate(df.reset_index().iterrows(), start=1):
        i = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress(idx_vis / max(total, 1))
            continue
        df, changed, msg = update_price_for_ticker(df, tkr)
        if changed:
            log["priced"].append(tkr)
        else:
            log["miss"].append(f"{tkr}: {msg}")
        progress.progress(idx_vis / max(total, 1))
    return df, log

# ========== VÅG: kö-ordning & batchkörning ===================================

def _wave_build_queue(df: pd.DataFrame, mode: str = "oldest") -> list[str]:
    """
    Skapar en kö av tickers att processa.
    - 'oldest': sortera på äldst TS (alla spårade) först
    - 'alphabetic': sortera A–Ö på Bolagsnamn
    """
    work = add_oldest_ts_col(df.copy())
    if mode == "alphabetic":
        work = work.sort_values(by=["Bolagsnamn","Ticker"])
    else:
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    tickers = [str(t).upper() for t in work["Ticker"].astype(str).tolist() if str(t).strip()]
    return tickers

def start_wave(df: pd.DataFrame, mode: str = "oldest"):
    st.session_state["wave_queue"] = _wave_build_queue(df, mode=mode)
    st.session_state["wave_done"] = []
    st.session_state["wave_changed"] = []
    st.session_state["wave_miss"] = []
    st.session_state["wave_mode"] = mode
    st.session_state["wave_started_at"] = now_stamp()

def run_wave_step(df: pd.DataFrame, user_rates: dict, batch_size: int = 10, make_snapshot: bool = True, *, force_ts: bool = True):
    q = st.session_state.get("wave_queue", []) or []
    done = st.session_state.get("wave_done", []) or []
    chgd = st.session_state.get("wave_changed", []) or []
    miss = st.session_state.get("wave_miss", []) or []

    if not q:
        return df, {"processed": 0, "remaining": 0}

    take = q[:batch_size]
    rest = q[batch_size:]
    for tkr in take:
        df, changed, _dbg = update_full_for_ticker(df, tkr, user_rates, force_ts=force_ts)
        done.append(tkr)
        if changed:
            chgd.append(tkr)
        else:
            miss.append(tkr)

    # spara efter batch
    spara_data(df, do_snapshot=make_snapshot)

    st.session_state["wave_queue"] = rest
    st.session_state["wave_done"] = done
    st.session_state["wave_changed"] = chgd
    st.session_state["wave_miss"] = miss

    return df, {"processed": len(take), "remaining": len(rest)}

# ========== Debug & kontroll ==================================================

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

# app.py — Del 6/7
# --- Analys, Portfölj, Investeringsförslag & Lägg till/uppdatera -------------

# ===== Hjälpfunktioner för visningar =========================================

def _pretty_mcap(val: float, ccy: str = "") -> str:
    try:
        v = float(val or 0.0)
    except Exception:
        v = 0.0
    if v >= 1e12:
        s = f"{v/1e12:.2f} biljoner"
    elif v >= 1e9:
        s = f"{v/1e9:.2f} miljarder"
    elif v >= 1e6:
        s = f"{v/1e6:.2f} miljoner"
    elif v >= 1e3:
        s = f"{v/1e3:.2f} tusen"
    else:
        s = f"{v:.0f}"
    return f"{s} {ccy}".strip()

def _risk_label_from_mcap(mcap: float) -> str:
    """
    Enkel buckets (USD-ekvivalenter om möjligt).
    """
    try:
        v = float(mcap or 0.0)
    except Exception:
        v = 0.0
    if v < 300e6:
        return "Microcap"
    elif v < 2e9:
        return "Smallcap"
    elif v < 10e9:
        return "Midcap"
    elif v < 200e9:
        return "Largecap"
    else:
        return "Megacap"

def _ps_hist_avg(row: pd.Series) -> float:
    vals = []
    for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
        try:
            v = float(row.get(k, 0.0) or 0.0)
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    return round(float(np.mean(vals)) if vals else 0.0, 2)

def _row_mcap_now(row: pd.Series) -> float:
    v = float(row.get("MCap (nu)", 0.0) or 0.0)
    if v > 0:
        return v
    # fallback: pris * shares (shares i miljoner)
    try:
        px = float(row.get("Aktuell kurs", 0.0) or 0.0)
        sh_m = float(row.get("Utestående aktier", 0.0) or 0.0)
        if px > 0 and sh_m > 0:
            return px * sh_m * 1e6
    except Exception:
        pass
    return 0.0

def _update_stamp_badge(row: pd.Series) -> str:
    auto = str(row.get("Senast auto-uppdaterad","")).strip()
    manu = str(row.get("Senast manuellt uppdaterad","")).strip()
    src  = str(row.get("Senast uppdaterad källa","")).strip()
    parts = []
    if auto:
        parts.append(f"🛠️ Auto: {auto}{(' • ' + src) if src else ''}")
    if manu:
        parts.append(f"✍️ Manuell: {manu}")
    return " | ".join(parts) if parts else "—"

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
    st.caption(_update_stamp_badge(r))

    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S (Yahoo)",
        "P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
        "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
        "Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa","Finansiell valuta",
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

# ===== Investeringsförslag ====================================================

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

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Beräkna mcap & risklabel & filter
    base["MCap_calc"] = base.apply(_row_mcap_now, axis=1)
    base["Risklabel"] = base["MCap_calc"].apply(_risk_label_from_mcap)
    caps = ["Microcap","Smallcap","Midcap","Largecap","Megacap"]
    valda_caps = st.multiselect("Filtrera på börsvärdesklass", caps, default=caps)
    base = base[base["Risklabel"].isin(valda_caps)].copy()
    if base.empty:
        st.info("Inga bolag efter filter.")
        return

    # Potential
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

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

    # Portföljandelar
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    # Köp-simulering
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

    # P/S-snitt (senaste 4)
    ps_hist_snitt = _ps_hist_avg(rad)

    # Visa
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']}) — {rad['Risklabel']}")
    st.caption(_update_stamp_badge(rad))

    # Bygg detaljer
    det_lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Nuvarande P/S (beräknat):** {round(float(rad.get('P/S',0.0) or 0.0), 2)}",
        f"- **P/S (Yahoo):** {round(float(rad.get('P/S (Yahoo)',0.0) or 0.0), 2)}",
        f"- **P/S 4Q-snitt:** {ps_hist_snitt}",
        f"- **Börsvärde (nu):** {_pretty_mcap(rad.get('MCap_calc', 0.0), rad.get('Valuta',''))}",
    ]
    # MCap historik Q1–Q4
    for i in range(1,5):
        mk = float(rad.get(f"MCap Q{i}", 0.0) or 0.0)
        dt = str(rad.get(f"MCap Datum Q{i}", "") or "")
        if mk > 0 and dt:
            det_lines.append(f"- **MCap Q{i} ({dt}):** {_pretty_mcap(mk, rad.get('Valuta',''))}")
    # Valfria nyckeltal
    de = float(rad.get("Debt/Equity", 0.0) or 0.0)
    gm = float(rad.get("Bruttomarginal (%)", 0.0) or 0.0)
    nm = float(rad.get("Nettomarginal (%)", 0.0) or 0.0)
    cash = float(rad.get("Kassa", 0.0) or 0.0)
    fccy = str(rad.get("Finansiell valuta","") or "")
    if de > 0: det_lines.append(f"- **Debt/Equity:** {round(de,2)}")
    if gm > 0: det_lines.append(f"- **Bruttomarginal:** {round(gm,2)} %")
    if nm > 0: det_lines.append(f"- **Nettomarginal:** {round(nm,2)} %")
    if cash > 0: det_lines.append(f"- **Kassa:** {_pretty_mcap(cash, fccy)}")

    # Köputfall
    det_lines += [
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
    ]
    st.markdown("\n".join(det_lines))

# ===== Lägg till / uppdatera bolag ===========================================

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

    # Status-badge
    if not bef.empty:
        st.caption(_update_stamp_badge(bef))

    # Snabbknappar för vald ticker
    if not bef.empty:
        tkr = bef.get("Ticker","")
        colb1, colb2, colb3 = st.columns(3)
        with colb1:
            if st.button("💹 Uppdatera endast pris (denna)"):
                df, changed, msg = update_price_for_ticker(df, tkr)
                spara_data(df, do_snapshot=False)
                st.success("Pris uppdaterat och sparat." if changed else f"Ingen prisförändring (sparat TS/källa).")
                st.experimental_rerun()
        with colb2:
            if st.button("🔄 Full auto (endast denna)"):
                df, changed, _dbg = update_full_for_ticker(df, tkr, user_rates, force_ts=True)
                spara_data(df, do_snapshot=False)
                st.success("Full auto uppdaterad & sparad." if changed else "Inga förändringar hittades (TS/källa uppdaterad).")
                st.experimental_rerun()
        with colb3:
            pass  # plats för framtida knappar

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
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%), P/S (Yahoo) och MCap (nu) via Yahoo")
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
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Sätt manuell TS + TS_ per fält
        ridx = df.index[df["Ticker"]==ticker][0]
        if datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo (inkl P/S Yahoo & MCap nu)
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if "P/S (Yahoo)" in data and data.get("P/S (Yahoo)") is not None:         df.loc[ridx, "P/S (Yahoo)"]     = float(data.get("P/S (Yahoo)") or 0.0)
        if "MCap (nu)" in data and data.get("MCap (nu)") is not None:             df.loc[ridx, "MCap (nu)"]       = float(data.get("MCap (nu)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df, do_snapshot=False)
        st.success("Sparat.")
        st.experimental_rerun()

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
# --- Yahoo-nyckeltal (Debt/Equity, marginaler, kassa) + MAIN -----------------

def _yfi_pick_df(*dfs):
    for d in dfs:
        try:
            if isinstance(d, pd.DataFrame) and not d.empty:
                return d
        except Exception:
            pass
    return None

def _yfi_latest(df: pd.DataFrame, names: list[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    try:
        idx = [str(x).strip() for x in df.index]
        for nm in names:
            if nm in idx:
                row = df.loc[nm].dropna()
                if not row.empty:
                    # ta senaste kolumnen (högst datum/position)
                    val = row.iloc[0] if hasattr(row, "iloc") else list(row.values)[0]
                    return float(val)
    except Exception:
        pass
    return None

def hamta_yahoo_nyckeltal(ticker: str) -> dict:
    """
    Försöker hämta Debt/Equity, bruttomarginal, nettomarginal, kassa och finansiell valuta via yfinance.
    Robust mot varierande indexnamn.
    """
    out = {
        "Debt/Equity": 0.0,
        "Bruttomarginal (%)": 0.0,
        "Nettomarginal (%)": 0.0,
        "Kassa": 0.0,
        "Finansiell valuta": "",
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        out["Finansiell valuta"] = str(info.get("financialCurrency") or info.get("currency") or "").upper()

        # === Balansräkning (annual/quarterly) ===
        bs = _yfi_pick_df(getattr(t, "balance_sheet", None), getattr(t, "quarterly_balance_sheet", None))
        # Total debt: försök direkt eller summera long+short
        total_debt = _yfi_latest(bs, [
            "Total Debt","Total debt","Short Long Term Debt Total","Short/Long Term Debt",
            "TotalBorrowings","Debt","InterestBearingLiabilities"
        ])
        if total_debt is None:
            lt_debt = _yfi_latest(bs, ["Long Term Debt","LongTermDebt","Long-term debt"])
            st_debt = _yfi_latest(bs, ["Short Long Term Debt","ShortTermDebt","Short-term debt","Current Debt"])
            try:
                total_debt = float(lt_debt or 0.0) + float(st_debt or 0.0)
            except Exception:
                total_debt = None

        total_equity = _yfi_latest(bs, [
            "Total Stockholder Equity","Stockholders Equity","Total Equity Gross Minority Interest",
            "TotalShareholdersEquity","Total shareholders equity"
        ])

        cash_val = _yfi_latest(bs, [
            "Cash And Cash Equivalents","CashAndCashEquivalents","Cash And Cash Equivalents, And Short Term Investments",
            "CashCashEquivalentsAndShortTermInvestments","Cash"
        ])
        if cash_val is not None and cash_val >= 0:
            out["Kassa"] = float(cash_val)

        # D/E
        try:
            if total_debt is not None and total_equity and float(total_equity) != 0:
                out["Debt/Equity"] = float(total_debt) / float(total_equity)
        except Exception:
            pass

        # === Resultaträkning (annual/quarterly) ===
        # För bruttomarginal & nettomarginal. Prova både "financials" och moderna "income_stmt".
        is_annual = _yfi_pick_df(getattr(t, "financials", None), getattr(t, "income_stmt", None))
        is_quart  = _yfi_pick_df(getattr(t, "quarterly_financials", None))
        inc_df = is_quart if is_quart is not None else is_annual

        gross_profit = _yfi_latest(inc_df, ["Gross Profit","GrossProfit","Gross profit"])
        revenue      = _yfi_latest(inc_df, ["Total Revenue","TotalRevenue","Revenues","Revenue","Sales"])
        net_income   = _yfi_latest(inc_df, ["Net Income","NetIncome","Net income"])

        # Bruttomarginal
        if gross_profit is not None and revenue and float(revenue) > 0:
            out["Bruttomarginal (%)"] = round((float(gross_profit) / float(revenue)) * 100.0, 2)
        else:
            # fallback via info.ratio
            gm = info.get("grossMargins", None)
            try:
                if gm is not None and float(gm) > 0:
                    out["Bruttomarginal (%)"] = round(float(gm) * 100.0, 2)
            except Exception:
                pass

        # Nettomarginal
        if net_income is not None and revenue and float(revenue) > 0:
            out["Nettomarginal (%)"] = round((float(net_income) / float(revenue)) * 100.0, 2)
        else:
            pm = info.get("profitMargins", None)
            try:
                if pm is not None and float(pm) != 0:
                    out["Nettomarginal (%)"] = round(float(pm) * 100.0, 2)
            except Exception:
                pass

    except Exception:
        pass

    return out

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
