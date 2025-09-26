# app.py — Del 1/? — Importer, tidszon, Sheets-koppling, valutakurser, kolumnschema

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

# Slutlig kolumnlista i databasen (inkl. några extra som används i vyerna)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    # Extra fält som används i appen
    "Direktavkastning (%)", "Sektor", "_marketCap_raw",
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
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","yield","marketcap","_marketcap"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sektor"):
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
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "Direktavkastning (%)", "_marketCap_raw"
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

# app.py — Del 2/? — Yahoo-hjälpare & beräkningar & kvartalsdata (fallback)

# --- Yahoo-hjälpare ----------------------------------------------------------

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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Årlig utdelning, Direktavkastning, Sektor, CAGR.
    Lägger även in market cap rått som _marketCap_raw.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "Direktavkastning (%)": 0.0,
        "Sektor": "",
        "CAGR 5 år (%)": 0.0,
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

        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        sector = info.get("sector") or ""
        if sector:
            out["Sektor"] = str(sector)

        # Dividend & yield
        div_rate = info.get("dividendRate", None)
        if div_rate is None:
            div_rate = info.get("trailingAnnualDividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # Direktavkastning i %
        if out["Årlig utdelning"] and out["Aktuell kurs"]:
            try:
                out["Direktavkastning (%)"] = round((out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0, 2)
            except Exception:
                pass
        else:
            # fallback till Yahoo's trailingAnnualDividendYield om finns
            dy = info.get("trailingAnnualDividendYield")
            try:
                if dy is not None:
                    out["Direktavkastning (%)"] = round(float(dy) * 100.0, 2)
            except Exception:
                pass

        # Market cap
        mcap = info.get("marketCap", None)
        try:
            if mcap is not None:
                out["_marketCap_raw"] = float(mcap)
        except Exception:
            pass

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# --- Beräkningar -------------------------------------------------------------

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier
    Obs: 'Utestående aktier' lagras i miljoner.
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        try:
            ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        except Exception:
            ps_clean = []
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
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
            aktier_ut_mil = float(rad.get("Utestående aktier", 0.0) or 0.0)  # i miljoner
        except Exception:
            aktier_ut_mil = 0.0
        aktier_ut = aktier_ut_mil * 1e6

        if aktier_ut > 0 and ps_snitt > 0:
            try:
                oms_idag  = float(rad.get("Omsättning idag", 0.0) or 0.0) * 1e6
                oms_1y    = float(rad.get("Omsättning nästa år", 0.0) or 0.0) * 1e6
                oms_2y    = float(df.at[i, "Omsättning om 2 år"] or 0.0) * 1e6
                oms_3y    = float(df.at[i, "Omsättning om 3 år"] or 0.0) * 1e6
            except Exception:
                oms_idag = oms_1y = oms_2y = oms_3y = 0.0

            df.at[i, "Riktkurs idag"]    = round((oms_idag * ps_snitt) / aktier_ut, 2) if oms_idag  > 0 else 0.0
            df.at[i, "Riktkurs om 1 år"] = round((oms_1y   * ps_snitt) / aktier_ut, 2) if oms_1y    > 0 else 0.0
            df.at[i, "Riktkurs om 2 år"] = round((oms_2y   * ps_snitt) / aktier_ut, 2) if oms_2y    > 0 else 0.0
            df.at[i, "Riktkurs om 3 år"] = round((oms_3y   * ps_snitt) / aktier_ut, 2) if oms_3y    > 0 else 0.0
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# --- Kvartalsintäkter via Yahoo (global fallback för icke-SEC) --------------

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

# --- Prislinje för specifika datum (t.ex. P/S historik) ----------------------

from datetime import datetime as _dt, timedelta as _td

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

# app.py — Del 3/? — SEC + FMP + Finnhub + Auto-merge

# ---------------- SEC (US + FPI/IFRS) ---------------------------------------

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

# ---------- IFRS/GAAP kvartalsintäkter + valuta ------------------------------

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
    return 0.0

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    Inkluderar Dec/Jan-fix: håller endast en rapport per FY-ordning men föredrar kalender-kvartalsintervall ~90 dagar.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A")}),
    ]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "Revenues", "Revenue",
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
                    # Varaktighet 70–100 dagar ≈ kvartal
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
                # Deduplicera per end-datum och sortera nyast→äldst
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)

                # Dec/Jan-fix: inget särskilt tas bort, vi låter TTM-fönster välja de 4 senaste oavsett om FY bryter i Jan.
                rows = rows[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters)
    return rows

def _sec_quarterly_revenues(facts: dict):
    rows = _sec_quarterly_revenues_dated(facts, max_quarters=4)
    return [v for (_, v) in rows]

# ---------- Implied shares via Yahoo + pris vid datum ------------------------

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

# ---------- Yahoo global fallback (icke-SEC) ---------------------------------

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def _yfi_quarterly_revenues(t: yf.Ticker) -> list:
    # (redan definierad i Del 2)

    return []

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/sector/yield etc.
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","Direktavkastning (%)","Sektor","_marketCap_raw"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else float(out.get("_marketCap_raw", 0.0) or 0.0)
    except Exception:
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
        out["Utestående aktier"] = shares / 1e6  # i miljoner

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

    # P/S (TTM) nu
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historisk)
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    out[f"P/S Q{idx}"] = (shares * p) / ttm_rev_px

    return out

# ---------- SEC + Yahoo combo -----------------------------------------------

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn/sector/dividend från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "Direktavkastning (%)", "Sektor", "_marketCap_raw"):
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
        out["Utestående aktier"] = shares_used / 1e6  # i miljoner

    # Market cap (nu)
    mcap_now = float(out.get("_marketCap_raw", 0.0) or 0.0)
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

    # P/S Q1–Q4 historik — säkra att vi verkligen tar de 4 senaste perioderna (inkl. ev Dec/Jan)
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        # exakta 4 senaste TTM-slut:
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px_hist = px_map.get(d_end)
                if px_hist and px_hist > 0:
                    mcap_hist = shares_used * float(px_hist)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    return out

# ---------------- FMP (light/full) ------------------------------------------

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

@st.cache_data(show_spinner=False, ttl=1800)
def hamta_fmp_falt(yahoo_ticker: str) -> dict:
    out = {"_debug": {}}
    sym = _fmp_pick_symbol(yahoo_ticker)
    out["_symbol"] = sym

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
            try: market_cap = float(q0["marketCap"])
            except: pass

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

    if "P/S" not in out and market_cap > 0:
        isttm, sc_isttm = _fmp_get(f"api/v3/income-statement-ttm/{sym}", stable=False)
        out["_debug"]["income_ttm_sc"] = sc_isttm
        revenue_ttm = 0.0
        if isinstance(isttm, list) and isttm:
            cand = isttm[0]
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

# ---------------- Finnhub (estimat) -----------------------------------------

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

# ---------------- Merge-hjälpare --------------------------------------------

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, changes_map: dict, always_stamp: bool = False) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde.
    Uppdaterar TS_ för spårade fält, sätter 'Senast auto-uppdaterad' + källa.
    always_stamp=True stämplar TS även om värdet råkar vara oförändrat (enligt din önskan).
    Returnerar True om något fält faktiskt ändrades (värdemässigt).
    """
    changed_fields = []
    any_written = False

    for f, v in new_vals.items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f] if f in df.columns else None

        # skrivpolicy
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            write_ok = v is not None

        if not write_ok and not always_stamp:
            continue

        # skriv värde om det skiljer sig
        if write_ok and ((pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))):
            df.at[row_idx, f] = v
            changed_fields.append(f)
            any_written = True

        # TS-stämpel om spårat fält
        if f in TS_FIELDS and (always_stamp or (write_ok and ((pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))))):
            _stamp_ts_for_field(df, row_idx, f)

    # notera auto-update + källa, även om inget ändrades värdemässigt (då syns att vi försökte idag)
    _note_auto_update(df, row_idx, source)
    if changed_fields:
        changes_map.setdefault(str(df.at[row_idx, "Ticker"]), []).extend(changed_fields)
        return True
    return False

# ---------------- Huvud-pipeline för en ticker ------------------------------

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S) om saknas
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","Direktavkastning (%)","Sektor","_marketCap_raw","_debug_shares_source"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Direktavkastning (%)","Sektor","_marketCap_raw",
                  "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
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
            if ("_marketCap_raw" not in vals) and (fmpl.get("_marketCap_raw") not in (None, "", 0, 0.0)):
                vals["_marketCap_raw"] = fmpl["_marketCap_raw"]
            if ("Aktuell kurs" not in vals) and (fmpl.get("Aktuell kurs") not in (None, "", 0, 0.0)):
                vals["Aktuell kurs"] = fmpl["Aktuell kurs"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    return vals, debug

# app.py — Del 4/? — Snapshots, batchjobb, kontrollvy & hjälp

# ---------------- Snapshots --------------------------------------------------

def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df. Kräver endast Sheets (inte Drive).
    """
    try:
        ss = get_spreadsheet()
        snap_name = f"Snapshot-{_ts_str()}"
        ss.add_worksheet(title=snap_name, rows=max(1000, len(df)+10), cols=max(50, len(df.columns)+2))
        ws = ss.worksheet(snap_name)
        _with_backoff(ws.clear)
        _with_backoff(ws.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())
        st.success(f"Snapshot skapad: {snap_name}")
    except Exception as e:
        st.warning(f"Misslyckades skapa snapshot-flik: {e}")

# ---------------- TS-inspektion & kontroll-listor ---------------------------

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

# ---------------- Batch-kö (med 1/X) ----------------------------------------

BATCH_SESSION_KEY = "batch_queue"      # lista av tickers i den ordning de ska köras
BATCH_LAST_RUN_KEY = "batch_last_run"  # {ticker: "YYYY-MM-DD"} senaste kördatum (för att inte köra exakt samma 20 om du vill)

def _ensure_batch_state():
    st.session_state.setdefault(BATCH_SESSION_KEY, [])
    st.session_state.setdefault(BATCH_LAST_RUN_KEY, {})

def reset_batch_queue():
    _ensure_batch_state()
    st.session_state[BATCH_SESSION_KEY] = []

def fill_batch_queue(df: pd.DataFrame, mode: str, n: int = 20):
    """
    mode: "Äldst först" eller "A–Ö"
    Fyller batch-kön med n tickers (som inte redan ligger i kön) i vald ordning.
    """
    _ensure_batch_state()
    existing = set(st.session_state[BATCH_SESSION_KEY])

    if mode == "Äldst först":
        work = add_oldest_ts_col(df.copy())
        ordered = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        ordered = df.sort_values(by=["Bolagsnamn","Ticker"])

    queue = []
    for _, r in ordered.iterrows():
        tkr = str(r.get("Ticker","")).strip().upper()
        if not tkr or tkr in existing:
            continue
        queue.append(tkr)
        if len(queue) >= n:
            break

    st.session_state[BATCH_SESSION_KEY].extend(queue)
    return queue

def run_batch_queue(df: pd.DataFrame, user_rates: dict, count: int = 10, do_snapshot: bool = False, always_stamp: bool = True):
    """
    Kör 'count' tickers från kön (FIFO). Visar progress 1/X och textstatus.
    Sparar endast om något ändrats eller always_stamp=True stämplar datum.
    """
    _ensure_batch_state()
    queue = st.session_state[BATCH_SESSION_KEY]
    if not queue:
        st.info("Batch-kön är tom. Fyll den först.")
        return df, {"changed": {}, "misses": {}, "debug": []}

    to_run = min(count, len(queue))
    progress = st.progress(0.0)
    status = st.empty()
    log = {"changed": {}, "misses": {}, "debug": []}
    any_changed = False

    for i in range(to_run):
        tkr = queue.pop(0)  # FIFO
        status.write(f"Uppdaterar {i+1}/{to_run}: **{tkr}**")
        try:
            vals, debug = auto_fetch_for_ticker(tkr)
            idxs = list(df.index[df["Ticker"].astype(str).str.upper() == tkr])
            if not idxs:
                log["misses"][tkr] = ["Ticker saknas i tabellen"]
            else:
                idx = idxs[0]
                changed = apply_auto_updates_to_row(df, idx, vals, source="Batch auto", changes_map=log["changed"], always_stamp=always_stamp)
                any_changed = any_changed or changed
                # uppdatera beräkningar på just raden
                df = uppdatera_berakningar(df, user_rates)
                st.session_state[BATCH_LAST_RUN_KEY][tkr] = now_stamp()
                log["debug"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/to_run)
        status.write(f"Klar {i+1}/{to_run}: **{tkr}**")

    # Spara en gång efter batch-delkörningen
    if any_changed or always_stamp:
        spara_data(df, do_snapshot=do_snapshot)
        st.success("Batch-körning sparad.")

    return df, log

# ---------------- Full auto-uppdatering (för alla) ---------------------------

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, always_stamp: bool = True):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla nya värden,
    men TS stämplas alltid om always_stamp=True. Visar progress 1/X.
    """
    log = {"changed": {}, "misses": {}, "debug_first_30": []}
    total = len(df)
    progress = st.progress(0.0)
    status = st.empty()

    any_changed = False

    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1))
            continue
        status.write(f"Uppdaterar {i+1}/{total}: **{tkr}**")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(df, idx, new_vals, source="Auto (SEC/Yahoo→Finnhub→FMP)", changes_map=log["changed"], always_stamp=always_stamp)
            any_changed = any_changed or changed
            if i < 30:
                log["debug_first_30"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]
        progress.progress((i+1)/max(total,1))

    df = uppdatera_berakningar(df, user_rates)

    if any_changed or always_stamp:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Ändringar/TS sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring – ingen skrivning/snapshot gjordes.")

    return df, log

# ---------------- Kontrollvy -------------------------------------------------

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

    # 3) Senaste körlogg (om du nyss körde Auto/Batch)
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log") or st.session_state.get("last_batch_log")
    if not log:
        st.info("Ingen auto- eller batchkörning i denna session ännu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ändringar** (ticker → fält)")
            if log.get("changed"):
                st.json(log["changed"])
            else:
                st.write("–")
        with col2:
            st.markdown("**Missar** (ticker → anteckning)")
            if log.get("misses"):
                st.json(log["misses"])
            else:
                st.write("–")
        if log.get("debug_first_30"):
            st.markdown("**Debug (första 30)**")
            st.json(log.get("debug_first_30"))
        if log.get("debug"):
            st.markdown("**Debug (batch)**")
            st.json(log.get("debug"))

# app.py — Del 5/? — Enskild uppdatering, hjälpfunktioner, Analys-vy

# ---------------- Enskild uppdatering (pris / full) -------------------------

def uppdatera_endast_kurs(df: pd.DataFrame, ticker: str, user_rates: dict) -> pd.DataFrame:
    tkr = str(ticker).strip().upper()
    idxs = list(df.index[df["Ticker"].astype(str).str.upper() == tkr])
    if not idxs:
        st.warning(f"Kurs = Ingen förändring: {tkr} hittades inte i tabellen.")
        return df
    idx = idxs[0]
    # hämta endast kurs/valuta/namn/utdelning/yield/sector/mcap
    yvals = hamta_yahoo_fält(tkr)
    write = {}
    for k in ("Aktuell kurs","Valuta","Bolagsnamn","Årlig utdelning","Direktavkastning (%)","Sektor","_marketCap_raw"):
        if yvals.get(k) not in (None, "", 0, 0.0):
            write[k] = yvals[k]
    changed = apply_auto_updates_to_row(df, idx, write, source="Prisuppdatering (Yahoo)", changes_map={}, always_stamp=True)
    df = uppdatera_berakningar(df, user_rates)
    if changed:
        st.success(f"Kurs uppdaterad för {tkr}.")
    else:
        st.info(f"Kurs = Ingen faktisk ändring för {tkr} (TS stämplad).")
    return df

def uppdatera_fullt_bolag(df: pd.DataFrame, ticker: str, user_rates: dict) -> pd.DataFrame:
    tkr = str(ticker).strip().upper()
    idxs = list(df.index[df["Ticker"].astype(str).str.upper() == tkr])
    if not idxs:
        st.warning(f"Full auto = {tkr} hittades inte i tabellen.")
        return df
    idx = idxs[0]
    vals, debug = auto_fetch_for_ticker(tkr)
    changed = apply_auto_updates_to_row(df, idx, vals, source="Enskild auto (SEC/Yahoo→Finnhub→FMP)", changes_map={}, always_stamp=True)
    df = uppdatera_berakningar(df, user_rates)
    if changed:
        st.success(f"Full auto: uppdaterade {tkr}.")
    else:
        st.info("Inga ändringar hittades vid auto-uppdatering (TS stämplad).")
    return df

# ---------------- Hjälpfunktioner (format & etiketter) ----------------------

def format_mcap_short(x: float, valuta: str = "") -> str:
    """
    Formatera market cap i kort text: T, B, M (engelsk skalning), behåll ev valuta-prefix/suffix inte här.
    """
    try:
        v = float(x)
    except Exception:
        return "-"
    sign = "-" if v < 0 else ""
    v = abs(v)
    unit = ""
    if v >= 1e12:
        v = v/1e12; unit = "T"
    elif v >= 1e9:
        v = v/1e9; unit = "B"
    elif v >= 1e6:
        v = v/1e6; unit = "M"
    s = f"{sign}{v:,.2f}{unit}"
    return s

def ps_label(ps: float) -> str:
    try:
        v = float(ps)
    except Exception:
        return "–"
    if v <= 2: return "Låg"
    if v <= 6: return "Normal"
    if v <= 15: return "Hög"
    return "Extrem"

def cap_bucket(mcap: float) -> str:
    try:
        v = float(mcap)
    except Exception:
        return "Okänd"
    if v >= 2e11: return "Mega"
    if v >= 1e10: return "Large"
    if v >= 2e9:  return "Mid"
    if v >= 3e8:  return "Small"
    if v >= 5e7:  return "Micro"
    return "Nano"

# ---------------- Analys-vy --------------------------------------------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if not len(etiketter):
        st.info("Inga bolag i databasen ännu.")
        return

    # robust index i session
    key_idx = "analys_idx"
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0
    st.session_state[key_idx] = max(0, min(st.session_state[key_idx], len(etiketter)-1))

    left, mid, right = st.columns([1,3,1])
    with left:
        if st.button("⬅️ Föregående", use_container_width=True):
            st.session_state[key_idx] = max(0, st.session_state[key_idx]-1)
    with mid:
        st.selectbox("Välj bolag", etiketter, index=st.session_state[key_idx], key="analys_selectbox")
        # håll index i synk om selectbox ändras
        try:
            st.session_state[key_idx] = etiketter.index(st.session_state["analys_selectbox"])
        except Exception:
            pass
    with right:
        if st.button("➡️ Nästa", use_container_width=True):
            st.session_state[key_idx] = min(len(etiketter)-1, st.session_state[key_idx]+1)

    st.write(f"Post {st.session_state[key_idx]+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state[key_idx]]
    tkr = str(r["Ticker"]).upper()
    px  = float(r.get("Aktuell kurs", 0.0) or 0.0)
    mcap = float(r.get("_marketCap_raw", 0.0) or 0.0)
    psn  = float(r.get("P/S-snitt", 0.0) or 0.0)
    valuta = str(r.get("Valuta",""))

    cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","_marketCap_raw","Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Direktavkastning (%)",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]

    # visa tabell
    vis = pd.DataFrame([r[cols].to_dict()])
    if "_marketCap_raw" in vis.columns:
        vis["_marketCap_raw"] = vis["_marketCap_raw"].apply(lambda v: f"{format_mcap_short(v)} ({valuta})" if float(v or 0)>0 else "-")
        vis = vis.rename(columns={"_marketCap_raw": "Market cap (kort)"})
    st.dataframe(vis, use_container_width=True, hide_index=True)

    # snabb etikett
    left2, right2 = st.columns(2)
    with left2:
        st.metric("Market cap", format_mcap_short(mcap, valuta), help="Råvärde i '_marketCap_raw'")
    with right2:
        st.metric("P/S-snitt", f"{psn:.2f}", help=f"Klassning: {ps_label(psn)}")

    # snabbknappar för enskild uppdatering
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Uppdatera endast kurs"):
            st.session_state["_analys_force_save"] = True
            df2 = uppdatera_endast_kurs(df, tkr, user_rates)
            st.session_state["last_df"] = df2
            st.experimental_rerun()
    with c2:
        if st.button("⚡ Full uppdatering för detta bolag"):
            st.session_state["_analys_force_save"] = True
            df2 = uppdatera_fullt_bolag(df, tkr, user_rates)
            st.session_state["last_df"] = df2
            st.experimental_rerun()

# app.py — Del 6/? — Lägg till/uppdatera bolag, Portfölj & Investeringsförslag

# ---------------- Lägg till / uppdatera bolag -------------------------------

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsval för listan
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # Navigering / bläddring
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = max(0, min(st.session_state.edit_index, len(val_lista)-1))

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=st.session_state.edit_index, key="edit_selectbox")
    try:
        st.session_state.edit_index = val_lista.index(st.session_state.get("edit_selectbox", ""))
    except Exception:
        pass

    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {max(0, st.session_state.edit_index)}/{max(0, len(val_lista)-1)}")
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
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, Direktavkastning, Sektor via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

        c3, c4 = st.columns(2)
        with c3:
            do_price = st.form_submit_button("🔄 Uppdatera endast kurs")
        with c4:
            do_full = st.form_submit_button("⚡ Full uppdatering för detta bolag")
        spar = st.form_submit_button("💾 Spara")

    if do_price and ticker:
        df2 = uppdatera_endast_kurs(df, ticker, user_rates)
        spara_data(df2)
        st.experimental_rerun()

    if do_full and ticker:
        df2 = uppdatera_fullt_bolag(df, ticker, user_rates)
        spara_data(df2)
        st.experimental_rerun()

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

        # Hämta basfält från Yahoo (namn/valuta/kurs/utdelning/yield/sector/mcap)
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "Direktavkastning (%)" in data and data.get("Direktavkastning (%)") is not None: df.loc[ridx, "Direktavkastning (%)"] = float(data.get("Direktavkastning (%)") or 0.0)
        if "Sektor" in data: df.loc[ridx, "Sektor"] = str(data.get("Sektor") or "")
        if "_marketCap_raw" in data and data.get("_marketCap_raw") is not None: df.loc[ridx, "_marketCap_raw"] = float(data.get("_marketCap_raw") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")
        st.experimental_rerun()

    # Äldst uppdaterade (alla spårade fält) — topp 10
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

# ---------------- Portfölj-vy ------------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, (port["Värde (SEK)"] / total_värde * 100.0).round(2), 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    # Sektorvikt
    st.subheader("Sektorvikt")
    sec = port.groupby("Sektor", dropna=False)["Värde (SEK)"].sum().sort_values(ascending=False)
    if not sec.empty:
        st.bar_chart(sec)

# ---------------- Investeringsförslag ---------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Filter och inställningar
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )
    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)

    # Risklabel/cap-bucket-filter
    cap_filter = st.selectbox("Filtrera på cap-bucket", ["Alla","Mega","Large","Mid","Small","Micro","Nano"], index=0)

    # Sektorfilter
    sectors = sorted([s for s in df.get("Sektor", pd.Series([])).dropna().unique() if str(s).strip()])
    chosen_sectors = st.multiselect("Filtrera på sektor (valfritt)", sectors, default=[])

    # Basdata
    base = df.copy() if subset == "Alla bolag" else df[df["Antal aktier"] > 0].copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()

    # Cap bucket beräkning
    base["_cap_bucket"] = base["_marketCap_raw"].apply(cap_bucket)
    if cap_filter != "Alla":
        base = base[base["_cap_bucket"] == cap_filter]

    # Sektorfilter
    if chosen_sectors:
        base = base[base["Sektor"].isin(chosen_sectors)]

    if base.empty:
        st.info("Inga bolag matchar dina filter just nu.")
        return

    # Beräkna potential & diff
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    # Sortera: Störst potential som standard
    base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    # Bläddringsindex
    key_idx = "forslags_index"
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0
    st.session_state[key_idx] = max(0, min(st.session_state[key_idx], len(base)-1))

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state[key_idx] = max(0, st.session_state[key_idx] - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state[key_idx]+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state[key_idx] = min(len(base)-1, st.session_state[key_idx] + 1)

    rad = base.iloc[st.session_state[key_idx]]

    # Portföljvärde för andelsberäkning
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
        r2 = port[port["Ticker"] == rad["Ticker"]]
        if not r2.empty:
            nuv_innehav = float(r2["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Nuvarande market cap och P/S-info
    mcap = float(rad.get("_marketCap_raw", 0.0) or 0.0)
    psn = float(rad.get("P/S-snitt", 0.0) or 0.0)
    ps_list = [float(rad.get(f"P/S Q{i}", 0.0) or 0.0) for i in range(1,5)]
    ps_hist_clean = [x for x in ps_list if x > 0]
    ps_hist_avg = round(float(np.mean(ps_hist_clean)), 2) if ps_hist_clean else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Utestående aktier:** {round(float(rad.get('Utestående aktier',0.0) or 0.0), 2)} M",
        f"- **Market cap:** {format_mcap_short(mcap)} ({rad['Valuta']}) — bucket: **{cap_bucket(mcap)}**",
        f"- **P/S (nu):** {round(float(rad.get('P/S',0.0) or 0.0),2)} | **P/S-snitt:** {psn:.2f} | **P/S 4Q-snitt:** {ps_hist_avg:.2f} ({ps_label(psn)})",
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

    # Expander med mer detaljer
    with st.expander("Visa detaljer"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**P/S-historik (4 senaste TTM-punkter)**")
            for i in range(1,5):
                st.write(f"- P/S Q{i}: {round(float(rad.get(f'P/S Q{i}',0.0) or 0.0),2)}")
            st.write(f"**Snitt (4Q):** {ps_hist_avg:.2f}")
        with c2:
            st.write("**Market cap råvärde**")
            st.write(f"{int(mcap):,} ({rad['Valuta']})")
            st.write("**Direktavkastning:** " + f"{round(float(rad.get('Direktavkastning (%)',0.0) or 0.0),2)} %")

# app.py — Del 7/? — Sidebar (valutor), Batch-panel, Main & routing

def _init_rate_state():
    # Säkra grundvärden i session_state för växelkurser
    if "rate_usd" not in st.session_state: st.session_state.rate_usd = float(STANDARD_VALUTAKURSER["USD"])
    if "rate_nok" not in st.session_state: st.session_state.rate_nok = float(STANDARD_VALUTAKURSER["NOK"])
    if "rate_cad" not in st.session_state: st.session_state.rate_cad = float(STANDARD_VALUTAKURSER["CAD"])
    if "rate_eur" not in st.session_state: st.session_state.rate_eur = float(STANDARD_VALUTAKURSER["EUR"])
    # toggles
    st.session_state.setdefault("rates_reload", 0)

def _sidebar_rates():
    st.sidebar.header("💱 Valutakurser → SEK")
    _init_rate_state()

    # 1) Läs sparade och initiera session_state (en gång per "Läs sparade kurser" eller vid app-start)
    if st.sidebar.button("↻ Läs sparade kurser"):
        st.cache_data.clear()
        saved = las_sparade_valutakurser()
        st.session_state.rate_usd = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
        st.session_state.rate_nok = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        st.session_state.rate_cad = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        st.session_state.rate_eur = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
        st.success("Kurser lästa från arkivet.")
        st.experimental_rerun()

    # 2) Auto-hämtning — gör före widget-instansiering och rerun
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        st.sidebar.success(f"Valutakurser uppdaterade (källa: {provider}).")
        st.experimental_rerun()

    # 3) Visa och redigera
    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("Återställ standard"):
            st.session_state.rate_usd = float(STANDARD_VALUTAKURSER["USD"])
            st.session_state.rate_nok = float(STANDARD_VALUTAKURSER["NOK"])
            st.session_state.rate_cad = float(STANDARD_VALUTAKURSER["CAD"])
            st.session_state.rate_eur = float(STANDARD_VALUTAKURSER["EUR"])
            st.experimental_rerun()

    return {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

# ---------------- Batch-panel (fyll & kör delmängder) -----------------------

def batch_panel(df: pd.DataFrame, user_rates: dict):
    st.header("🧵 Batch-körning")
    _ensure_batch_state()

    st.markdown("Fyll kön och kör delmängder. Du får **1/X**-progress och kan köra nästa omgång senare.")
    mode = st.selectbox("Ordning att fylla kön", ["Äldst först","A–Ö"], index=0)
    n_to_fill = st.number_input("Antal att lägga i kön nu", min_value=1, max_value=200, value=20, step=1)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("➕ Fyll kön"):
            added = fill_batch_queue(df, mode=mode, n=int(n_to_fill))
            if added:
                st.success(f"Lade till {len(added)} tickers i kön.")
            else:
                st.info("Inget att lägga till (kanske alla redan i kön?).")
    with c2:
        if st.button("🧹 Töm kön"):
            reset_batch_queue()
            st.warning("Kön tömd.")
    with c3:
        st.metric("I kön nu", len(st.session_state[BATCH_SESSION_KEY]))

    st.divider()
    run_n = st.number_input("Kör nästa N tickers", min_value=1, max_value=100, value=10, step=1)
    make_snapshot = st.checkbox("Skapa snapshot före skrivning", value=False)
    always_stamp = st.checkbox("Stämpla datum även vid oförändrade värden", value=True)

    if st.button("▶️ Kör batch"):
        df2, log = run_batch_queue(df, user_rates, count=int(run_n), do_snapshot=make_snapshot, always_stamp=always_stamp)
        st.session_state["last_batch_log"] = log
        # spara df2 i session och visa enklare summering
        st.session_state["last_df"] = df2
        st.success("Delbatch körd.")
        # Visa logg-nycklar kort
        if log.get("changed"):
            st.write("**Ändringar (sammanfattning):**")
            st.json(log["changed"])
        if log.get("misses"):
            st.write("**Missar:**")
            st.json(log["misses"])

# ---------------- MAIN -------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidebar: valutor och utility
    user_rates = _sidebar_rates()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.experimental_rerun()

    # Läs data
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa data från Google Sheets: {e}")
        # skapa tom struktur så appen kan starta
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    # Säkerställ schema, migrera och typer
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Sidebar: Auto-uppdatering för ALLA
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Auto-uppdatering (alla)")
    make_snapshot = st.sidebar.checkbox("Snapshot före skrivning", value=True, key="auto_snap_all")
    always_stamp = st.sidebar.checkbox("Stämpla datum även om oförändrat", value=True, key="auto_stamp_all")
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo→Finnhub→FMP)"):
        df2, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot, always_stamp=always_stamp)
        st.session_state["last_auto_log"] = log
        st.session_state["last_df"] = df2
        st.experimental_rerun()

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj","Batch"], index=0)

    # Routing
    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # om något sparades: uppdatera vår lokala referens så nästa vy ser ny data
        try:
            st.session_state["last_df"] = df2
        except Exception:
            pass
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)
    elif meny == "Batch":
        batch_panel(df, user_rates)

if __name__ == "__main__":
    main()
