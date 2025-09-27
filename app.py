# -*- coding: utf-8 -*-

# =========================================
# app.py — Del 1/N
# Basimporter, tidszon, Sheets, valutor, schema, TS-hjälpare
# =========================================

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

# =========================================
# Google Sheets-koppling
# =========================================
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
    try:
        sheet = skapa_koppling()
        data = _with_backoff(sheet.get_all_records)
        df = pd.DataFrame(data)
    except Exception:
        df = pd.DataFrame()
    return df

def spara_data(df: pd.DataFrame, do_snapshot: bool = False):
    """Skriv hela DataFrame till huvudbladet. Optionellt: skapa snapshot-flik först."""
    if do_snapshot:
        try:
            backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)  # definieras senare
        except Exception as e:
            st.warning(f"Kunde inte skapa snapshot före skrivning: {e}")
    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    _with_backoff(sheet.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# --- Utility ---
def _ts_str():
    return now_dt().strftime("%Y%m%d-%H%M%S")

def _ts_datetime():
    return now_dt()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def human_money(x: float) -> str:
    """Formatera stora tal prydligt (t, Md, M, k)."""
    try:
        x = float(x)
    except Exception:
        return "–"
    neg = x < 0
    x = abs(x)
    if x >= 1e12:
        s = f"{x/1e12:.2f} tn"
    elif x >= 1e9:
        s = f"{x/1e9:.2f} Md"
    elif x >= 1e6:
        s = f"{x/1e6:.2f} M"
    elif x >= 1e3:
        s = f"{x/1e3:.2f} k"
    else:
        s = f"{x:.0f}"
    return f"-{s}" if neg else s

# =========================================
# Valutakurser (lagring + auto-hämtning)
# =========================================

# Standard valutakurser till SEK (fallback/startvärden)
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
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))

# Automatisk valutahämtning (FMP -> Frankfurter -> exchangerate.host)
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

# =========================================
# Kolumnschema & TS-hantering
# =========================================

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

# Slutlig kolumnlista i databasen (inkl GAV (SEK))
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "GAV (SEK)",
    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
    # Extra fält som kan komma i senare delar (risk/sektor osv) skapas dynamiskt vid behov
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    for kol in FINAL_COLS:
        if kol not in df.columns:
            lower = kol.lower()
            if any(x in lower for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","mcap","ebitda","ev","fcf","yield","gav"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
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
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "GAV (SEK)"
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

# =========================================
# Del 2/N — Yahoo-hjälpare, beräkningar, merge/write-helpers
# =========================================

def _yfi_get(tkr: yf.Ticker, *keys):
    """Säker hämtning ur yfinance.info med fallback."""
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR, ev. sektor/industri.
    Notera: vi sätter INTE 'Omsättning idag/nästa år' här (du matar dessa manuellt).
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        # meta
        "Sektor": "",
        "Industri": "",
        "MCAP nu": 0.0,
        "_y_info_ok": False,
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

        mcap = info.get("marketCap")
        try:
            if mcap is not None:
                out["MCAP nu"] = float(mcap)
        except Exception:
            pass

        out["Sektor"] = str(info.get("sector", "") or "")
        out["Industri"] = str(info.get("industry", "") or "")

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
        out["_y_info_ok"] = True
    except Exception:
        pass
    return out

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp (50% max, 2% min vid negativ)
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [
            _safe_float(rad.get("P/S Q1", 0)),
            _safe_float(rad.get("P/S Q2", 0)),
            _safe_float(rad.get("P/S Q3", 0)),
            _safe_float(rad.get("P/S Q4", 0)),
        ]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: >100% → 50%, <0% → 2%
        cagr = _safe_float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år"
        oms_next = _safe_float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev tidigare ifyllda
            df.at[i, "Omsättning om 2 år"] = _safe_float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = _safe_float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        aktier_ut_milj = _safe_float(rad.get("Utestående aktier", 0.0))
        aktier_ut = aktier_ut_milj * 1e6
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((_safe_float(rad.get("Omsättning idag", 0.0))      * 1e6 * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((_safe_float(rad.get("Omsättning nästa år", 0.0))  * 1e6 * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((_safe_float(df.at[i, "Omsättning om 2 år"])       * 1e6 * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((_safe_float(df.at[i, "Omsättning om 3 år"])       * 1e6 * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    stamp_even_if_same: bool = True
) -> bool:
    """
    Skriver fält som har meningsfulla värden.
    - Om stamp_even_if_same=True stämplas TS_ även när värdet är oförändrat (ditt önskemål).
    - Sätter 'Senast auto-uppdaterad' + källa.
    Returnerar True om något fält faktiskt ändrades (värdet i df blev nytt).
    """
    changed_fields = []
    stamped_fields = []

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            # skapa kolumn on-the-fly om det ser ut som numeriskt/mått
            if isinstance(v, (int, float, np.floating)):
                df[f] = 0.0
            else:
                df[f] = ""
        old = df.at[row_idx, f] if f in df.columns else None

        # giltigt att skriva?
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        if not write_ok:
            # även om vi inte skriver värdet vill vi ev. stämpla TS om fältet är spårat och vi fick ett “fresh pull”
            if stamp_even_if_same and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                stamped_fields.append(f)
            continue

        # skriv och notera ändring
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[row_idx, f] = v
            changed_fields.append(f)

        # stämpla TS per fält (även om oförändrat)
        if f in TS_FIELDS and stamp_even_if_same:
            _stamp_ts_for_field(df, row_idx, f)
            stamped_fields.append(f)

    did_change = bool(changed_fields)
    if did_change or stamped_fields:
        _note_auto_update(df, row_idx, source)
        if did_change:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
    return did_change

# =========================================
# Del 2/N — Yahoo-hjälpare, beräkningar, merge/write-helpers
# =========================================

def _yfi_get(tkr: yf.Ticker, *keys):
    """Säker hämtning ur yfinance.info med fallback."""
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR, ev. sektor/industri.
    Notera: vi sätter INTE 'Omsättning idag/nästa år' här (du matar dessa manuellt).
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        # meta
        "Sektor": "",
        "Industri": "",
        "MCAP nu": 0.0,
        "_y_info_ok": False,
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

        mcap = info.get("marketCap")
        try:
            if mcap is not None:
                out["MCAP nu"] = float(mcap)
        except Exception:
            pass

        out["Sektor"] = str(info.get("sector", "") or "")
        out["Industri"] = str(info.get("industry", "") or "")

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
        out["_y_info_ok"] = True
    except Exception:
        pass
    return out

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp (50% max, 2% min vid negativ)
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [
            _safe_float(rad.get("P/S Q1", 0)),
            _safe_float(rad.get("P/S Q2", 0)),
            _safe_float(rad.get("P/S Q3", 0)),
            _safe_float(rad.get("P/S Q4", 0)),
        ]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: >100% → 50%, <0% → 2%
        cagr = _safe_float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år"
        oms_next = _safe_float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev tidigare ifyllda
            df.at[i, "Omsättning om 2 år"] = _safe_float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = _safe_float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        aktier_ut_milj = _safe_float(rad.get("Utestående aktier", 0.0))
        aktier_ut = aktier_ut_milj * 1e6
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((_safe_float(rad.get("Omsättning idag", 0.0))      * 1e6 * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((_safe_float(rad.get("Omsättning nästa år", 0.0))  * 1e6 * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((_safe_float(df.at[i, "Omsättning om 2 år"])       * 1e6 * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((_safe_float(df.at[i, "Omsättning om 3 år"])       * 1e6 * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    stamp_even_if_same: bool = True
) -> bool:
    """
    Skriver fält som har meningsfulla värden.
    - Om stamp_even_if_same=True stämplas TS_ även när värdet är oförändrat (ditt önskemål).
    - Sätter 'Senast auto-uppdaterad' + källa.
    Returnerar True om något fält faktiskt ändrades (värdet i df blev nytt).
    """
    changed_fields = []
    stamped_fields = []

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            # skapa kolumn on-the-fly om det ser ut som numeriskt/mått
            if isinstance(v, (int, float, np.floating)):
                df[f] = 0.0
            else:
                df[f] = ""
        old = df.at[row_idx, f] if f in df.columns else None

        # giltigt att skriva?
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        if not write_ok:
            # även om vi inte skriver värdet vill vi ev. stämpla TS om fältet är spårat och vi fick ett “fresh pull”
            if stamp_even_if_same and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                stamped_fields.append(f)
            continue

        # skriv och notera ändring
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[row_idx, f] = v
            changed_fields.append(f)

        # stämpla TS per fält (även om oförändrat)
        if f in TS_FIELDS and stamp_even_if_same:
            _stamp_ts_for_field(df, row_idx, f)
            stamped_fields.append(f)

    did_change = bool(changed_fields)
    if did_change or stamped_fields:
        _note_auto_update(df, row_idx, source)
        if did_change:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
    return did_change

# =========================================
# Del 3/N — Datakällor: SEC (US+IFRS), Yahoo-fallback, FMP (light)
# =========================================

# ---------- SEC grund ----------
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
    # ~9–10MB JSON → cachea 24h
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

# ---------- datum & helpers ----------
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

# ---------- shares (robust, multi-class) ----------
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
    Summerar multi-class per senaste datum ('end'). Returnerar aktier (styck), 0 om ej hittat.
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

# ---------- FX utils ----------
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

# ---------- SEC kvartalsintäkter (robust med Jan/Dec-hantering) ----------
def _fiscal_q_key(end_date):
    """
    Skapa en 'fiscal key' av datum → (year, q). Jan/Feb/Mar → Q1 (år = datum.year),
    Apr/May/Jun → Q2, Jul/Aug/Sep → Q3, Oct/Nov/Dec → Q4.
    OBS: Om månaden är Jan (1) **och** datumet är 1–31 Jan (typ NVDA),
    så kan detta vara Q4 i ett förskjutet räkenskapsår. Vi låter nyckeln bli (year, Q1)
    men vi deduplicerar på (year,quarter) och sorterar på riktiga datum—målet är att
    inte tappa Jan/Dec-rapporten.
    """
    m = end_date.month
    if m in (1,2,3):
        q = 1; y = end_date.year
    elif m in (4,5,6):
        q = 2; y = end_date.year
    elif m in (7,8,9):
        q = 3; y = end_date.year
    else:
        q = 4; y = end_date.year
    return (y, q)

def _is_quarter_like(start_date, end_date) -> bool:
    try:
        dur = (end_date - start_date).days
    except Exception:
        return False
    # 3-månadersperiod ~ 70–100 dagar
    return (dur >= 70 and dur <= 100)

def _dedupe_latest_quarters(rows, max_quarters: int = 20):
    """
    rows = [(end_date, value, form)]
    - Deduplicera på 'fiscal key' (year, q), behåll senaste end_date för varje key.
    - Returnera nyast → äldst, begränsa till max_quarters.
    """
    seen = {}
    for (end_date, val, form) in rows:
        key = _fiscal_q_key(end_date)
        if key not in seen or end_date > seen[key][0]:
            seen[key] = (end_date, val, form)
    uniq = list(seen.values())
    uniq.sort(key=lambda x: x[0], reverse=True)
    return uniq[:max_quarters]

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    Robust deduplicering så att Jan/Dec-kombination inte tappas.
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
                    if not _is_quarter_like(start, end):
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v, form))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # dedupe per 'fiscal key' för att få med Jan/Dec korrekt
                uniq = _dedupe_latest_quarters(tmp, max_quarters=max_quarters)
                rows_only = [(d, v) for (d, v, _) in uniq]
                if rows_only:
                    return rows_only, unit_code
    return [], None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters)
    return rows

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

# ---------- Yahoo pris & implied shares ----------
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

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

# ---------- Yahoo quarterly fallback (global) ----------
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

# ---------- SEC + Yahoo combo ----------
def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
    NOTERA: 'Omsättning idag'/'Omsättning nästa år' lämnas orörda (manuella).
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "MCAP nu", "Sektor", "Industri"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()
    last_px = float(out.get("Aktuell kurs") or 0.0)

    # Shares: implied → fallback SEC robust
    implied = _implied_shares_from_yahoo(ticker, price=last_px, mcap=out.get("MCAP nu"))
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
    mcap_now = float(out.get("MCAP nu") or 0.0)
    if mcap_now <= 0 and last_px > 0 and shares_used > 0:
        mcap_now = last_px * shares_used
        out["MCAP nu"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering till prisvaluta
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
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

    # P/S Q1–Q4 historik (använder shares_used * historiskt pris)
    if shares_used > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    return out

# ---------- Global Yahoo fallback ----------
def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (t.ex. .TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik.
    OBS: 'Omsättning idag/nästa år' fylls inte här (manuellt fält).
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/MCAP/Sektor/Industri
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","MCAP nu","Sektor","Industri"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()
    info = _yfi_info_dict(t)

    mcap = float(out.get("MCAP nu") or 0.0)
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

    # Valutakonvertering om financialCurrency != prisvaluta
    fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        conv = _fx_rate_cached(fin_ccy, px_ccy) or 1.0
    q_rows_px = [(d, v * conv) for (d, v) in q_rows]
    ttm_list = _ttm_windows(q_rows_px, need=4)

    # Market cap (nu)
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px
        out["MCAP nu"] = mcap

    # P/S (TTM) nu
    if mcap > 0 and ttm_list:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historisk)
    if shares > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    out[f"P/S Q{idx}"] = (shares * p) / ttm_rev_px

    return out

# ---------- FMP (light) för P/S fallback ----------
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

# =========================================
# Del 4/N — Snapshots, TS/utility, kurs/auto/batch-uppdatering
# =========================================

# ---- Säkra hjälpare ----
def _safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        if isinstance(x, str):
            x = x.replace(",", ".").strip()
        return float(x)
    except Exception:
        return default

def format_mcap_short(v: float) -> str:
    """Formatera market cap till kort svenskt format."""
    try:
        v = float(v)
    except Exception:
        return "-"
    absv = abs(v)
    sign = "-" if v < 0 else ""
    if absv >= 1e12:
        return f"{sign}{absv/1e12:.2f} T"
    if absv >= 1e9:
        return f"{sign}{absv/1e9:.2f} Md"
    if absv >= 1e6:
        return f"{sign}{absv/1e6:.2f} M"
    return f"{v:.0f}"

# ---- Snapshot till Google Sheet ----
def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df. Kräver endast Sheets (inte Drive).
    """
    if df is None or df.empty:
        st.warning("Hoppar över snapshot (tom DataFrame).")
        return
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

# ---- Äldsta TS per rad ----
def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """Returnerar äldsta tidsstämpeln bland alla TS_-kolumner i TS_FIELDS (eller None)."""
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
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    # Använd 2099 som “saknas”-fyllnad så att sortering ÄLDST→NYAST funkar
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Bolag som sannolikt kräver manuell hantering (prognosfält):
    - saknar någon av kärnfälten eller TS,
    - och/eller äldsta TS är äldre än 'older_than_days'.
    OBS: Prognosfält (manuella): 'Omsättning idag', 'Omsättning nästa år'.
    """
    need_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS[c] for c in TS_FIELDS if c in need_cols]

    out_rows = []
    cutoff = now_dt() - timedelta(days=int(older_than_days))

    for _, r in df.iterrows():
        # saknas någon kärnkolumn?
        missing_val = any((_safe_float(r.get(c, 0.0)) <= 0.0) for c in need_cols)
        # saknas TS på spårade?
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

# ---- Kursuppdatering (alla tickers) ----
def update_prices_all(df: pd.DataFrame, stamp_even_if_same: bool = True):
    """
    Uppdaterar endast Aktuell kurs, Valuta och MCAP nu för alla rader via Yahoo.
    Sätter 'Senast auto-uppdaterad' + källa och stämplar inga TS_-fält (pris fält har ingen TS).
    """
    if df is None or df.empty:
        st.warning("Ingen data att uppdatera.")
        return df

    progress = st.sidebar.progress(0, text="Uppdaterar kurser…")
    status = st.sidebar.empty()
    total = len(df)
    changed_any = False
    changes = {}

    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row.get("Ticker","")).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1), text=f"{i+1}/{total} – (tom ticker)")
            continue

        status.write(f"**{i+1}/{total}** – {tkr}")
        y = hamta_yahoo_fält(tkr)
        new_vals = {}
        if _safe_float(y.get("Aktuell kurs", 0)) > 0:
            new_vals["Aktuell kurs"] = _safe_float(y["Aktuell kurs"])
        if y.get("Valuta"):
            new_vals["Valuta"] = str(y["Valuta"]).upper()
        if _safe_float(y.get("MCAP nu", 0)) > 0:
            new_vals["MCAP nu"] = _safe_float(y["MCAP nu"])
        if y.get("Sektor"):
            new_vals["Sektor"] = y["Sektor"]
        if y.get("Industri"):
            new_vals["Industri"] = y["Industri"]

        changed = apply_auto_updates_to_row(
            df, idx, new_vals, source="Auto (Yahoo kurs/valuta/mcap)", changes_map=changes,
            stamp_even_if_same=stamp_even_if_same
        )
        changed_any = changed_any or changed
        progress.progress((i+1)/max(total,1), text=f"{i+1}/{total} – {tkr}")

    df = uppdatera_berakningar(df, user_rates={})  # rates ej relevanta för kurs
    if changed_any:
        spara_data(df, do_snapshot=False)
        st.sidebar.success("Kurser uppdaterade.")
    else:
        st.sidebar.info("Inga faktiska ändringar i kurser.")

    return df

# ---- Auto-fetch för EN ticker (utan prognosfält) ----
def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline (utan prognosfält — dessa är manuella):
      1) SEC + Yahoo (implied shares, P/S TTM, P/S Q1–Q4, MCAP, valuta, namn, sektor, industri)
      2) FMP light (P/S/shares) som fallback
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","MCAP nu","Sektor","Industri",
            "_debug_shares_source"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","MCAP nu","Sektor","Industri"]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) FMP light P/S/shares om saknas
    try:
        if ("P/S" not in vals) or ("Utestående aktier" not in vals):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier")}
            if "P/S" not in vals and _safe_float(fmpl.get("P/S", 0)) > 0:
                vals["P/S"] = _safe_float(fmpl["P/S"])
            if "Utestående aktier" not in vals and _safe_float(fmpl.get("Utestående aktier", 0)) > 0:
                vals["Utestående aktier"] = _safe_float(fmpl["Utestående aktier"])
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # OBS: Vi fyller INTE 'Omsättning idag' / 'Omsättning nästa år' här (manuella)
    return vals, debug

# ---- Auto-uppdatering: enskild ticker ----
def auto_update_single(df: pd.DataFrame, user_rates: dict, ticker: str, make_snapshot: bool = False):
    """
    Kör auto-uppdatering för EN ticker.
    Skriv endast meningsfulla fält, stämpla TS där relevant, och uppdatera beräkningar.
    """
    tkr = str(ticker).strip().upper()
    if not tkr:
        st.warning("Ingen ticker angiven.")
        return df, {}

    if "Ticker" not in df.columns:
        st.error("Databasen saknar kolumnen 'Ticker'.")
        return df, {}

    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        st.warning(f"{tkr} hittades inte i tabellen.")
        return df, {}

    idx = df.index[mask][0]
    vals, debug = auto_fetch_for_ticker(tkr)
    log_changes = {}
    changed = apply_auto_updates_to_row(
        df, idx, vals, source="Auto (SEC/Yahoo→FMP)", changes_map=log_changes, stamp_even_if_same=True
    )
    df = uppdatera_berakningar(df, user_rates)

    if changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.success(f"{tkr}: auto-uppdaterad.")
    else:
        st.info("Inga ändringar hittades vid auto-uppdatering.")

    return df, {"changed": log_changes, "debug": debug}

# ---- Auto-uppdatering: alla (tung) + 1/X-status ----
def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False):
    """
    Kör auto-uppdatering för ALLA rader.
    Skriver endast meningsfulla nya värden.
    Visar progressbar + 'i/X'.
    """
    if df is None or df.empty:
        st.warning("Ingen data att uppdatera.")
        return df, {}

    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    progress = st.sidebar.progress(0, text="Auto-uppdaterar alla…")
    status = st.sidebar.empty()

    total = len(df)
    any_changed = False

    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1), text=f"{i+1}/{total} – (tom ticker)")
            continue

        status.write(f"**{i+1}/{total}** – {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(
                df, idx, new_vals, source="Auto (SEC/Yahoo→FMP)", changes_map=log["changed"], stamp_even_if_same=True
            )
            if not changed:
                log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
            any_changed = any_changed or changed
            if i < 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(total,1), text=f"{i+1}/{total} – {tkr}")

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    if any_changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Ändringar sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# ---- Batch-kö (väljer N tickers, kör alla i en session) ----
def _build_batch_queue(df: pd.DataFrame, strategy: str, batch_size: int) -> list:
    """
    Bygg en lista av tickers att köra i batch:
      strategy ∈ {"Äldst uppdaterade först", "A–Ö (bolagsnamn)"}
    """
    df2 = df.copy()
    # Säkerställ äldst-kolumn
    df2 = add_oldest_ts_col(df2)

    if strategy.startswith("Äldst"):
        # sortera på äldst TS (minst datum först), sedan namn
        df2 = df2.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn", "Ticker"], ascending=[True, True, True])
    else:
        # A–Ö på bolagsnamn, sedan ticker
        df2 = df2.sort_values(by=["Bolagsnamn", "Ticker"], ascending=[True, True])

    out = []
    for _, r in df2.iterrows():
        t = str(r.get("Ticker","")).strip().upper()
        if t:
            out.append(t)
        if len(out) >= int(batch_size):
            break
    return out

def run_batch_update(df: pd.DataFrame, user_rates: dict, tickers: list, make_snapshot: bool = False):
    """
    Kör en batchlista av tickers och visar progress 1/X.
    """
    if not tickers:
        st.info("Batch-listan är tom.")
        return df, {"changed": {}, "misses": {}}

    progress = st.sidebar.progress(0, text="Batch-uppdatering…")
    status = st.sidebar.empty()

    total = len(tickers)
    log = {"changed": {}, "misses": {}}
    any_changed = False

    # Kör
    for i, tkr in enumerate(tickers):
        status.write(f"**{i+1}/{total}** – {tkr}")
        try:
            mask = (df["Ticker"].astype(str).str.upper() == str(tkr).upper())
            if not mask.any():
                log["misses"][tkr] = ["ticker saknas i df"]
                progress.progress((i+1)/total, text=f"{i+1}/{total} – {tkr} (saknas)")
                continue
            idx = df.index[mask][0]
            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(
                df, idx, new_vals, source="Auto (Batch SEC/Yahoo→FMP)", changes_map=log["changed"], stamp_even_if_same=True
            )
            if not changed:
                log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
            any_changed = any_changed or changed
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/total, text=f"{i+1}/{total} – {tkr}")

    # Räkna om & spara
    df = uppdatera_berakningar(df, user_rates)
    if any_changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Batch klar – ändringar sparade.")
    else:
        st.sidebar.info("Batch klar – inga faktiska ändringar.")

    return df, log

# =========================================
# Del 5/N — Kontroll-vy & Lägg till / uppdatera bolag
# =========================================

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

    # 2) Kräver manuell hantering (översikt)
    st.subheader("🛠️ Kräver manuell hantering (översikt)")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30, key="kontroll_older_days")
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga tydliga kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Senaste körlogg (om nyss kört Auto/Batch)
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log")
    if not log:
        st.info("Ingen auto/batch-körning i denna session ännu.")
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
        if "debug_first_20" in log:
            st.markdown("**Debug (första 20)**")
            st.json(log.get("debug_first_20", []))


def _ensure_cols(df: pd.DataFrame, cols_defaults: dict) -> pd.DataFrame:
    """Säkerställ att kolumner finns, annars skapa dem med default."""
    for c, d in cols_defaults.items():
        if c not in df.columns:
            df[c] = d
    return df

def _manual_prognoslista(df: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    """Lista över manuella prognosfält (Omsättning idag/nästa år) sorterat på äldsta TS."""
    df2 = df.copy()
    # säkerställ TS-kolumnerna finns så vi inte kraschar
    for c in ["TS_Omsättning idag","TS_Omsättning nästa år"]:
        if c not in df2.columns:
            df2[c] = ""
    # beräkna 'äldsta av just dessa två'
    def _min_two_ts(row):
        cand = []
        for c in ("TS_Omsättning idag","TS_Omsättning nästa år"):
            s = str(row.get(c,"")).strip()
            if s:
                d = pd.to_datetime(s, errors="coerce")
                if pd.notna(d): cand.append(d)
        return min(cand) if cand else pd.NaT
    df2["_oldest_two"] = df2.apply(_min_two_ts, axis=1)
    df2["_oldest_two_fill"] = df2["_oldest_two"].fillna(pd.Timestamp("2099-12-31"))
    cols = ["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år","_oldest_two"]
    df2 = df2.sort_values(by=["_oldest_two_fill","Bolagsnamn","Ticker"]).loc[:, cols].head(limit)
    return df2

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Säkerställ extra kolumner vi använder i denna vy
    df = _ensure_cols(df, {
        "Sektor": "",
        "Industri": "",
        "MCAP nu": 0.0,
        "GAV (SEK)": 0.0,
    })

    # Sorteringsval för listan att redigera
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], key="edit_sort_choice")
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # Etikettlista
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + etiketter

    # Robust bläddring
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(val_lista)-1))

    col_nav1, col_nav2, col_nav3 = st.columns([1,2,1])
    with col_nav1:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_nav2:
        st.write(f"Post {max(0, st.session_state.edit_index)}/{max(1, len(val_lista)-1)}")
    with col_nav3:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=st.session_state.edit_index, key="edit_selectbox")
    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    # TS-etiketter för markerade fält (visas ovanför formuläret)
    if not bef.empty:
        st.caption(
            "TS-manual/auto per fält • "
            f"Utestående aktier: {bef.get('TS_Utestående aktier','–')}  |  "
            f"P/S: {bef.get('TS_P/S','–')}  |  "
            f"P/S Q1: {bef.get('TS_P/S Q1','–')}  |  "
            f"P/S Q2: {bef.get('TS_P/S Q2','–')}  |  "
            f"P/S Q3: {bef.get('TS_P/S Q3','–')}  |  "
            f"P/S Q4: {bef.get('TS_P/S Q4','–')}  |  "
            f"Omsättning idag: {bef.get('TS_Omsättning idag','–')}  |  "
            f"Omsättning nästa år: {bef.get('TS_Omsättning nästa år','–')}"
        )
        st.caption(
            f"Senast manuellt uppdaterad: **{bef.get('Senast manuellt uppdaterad','')}**  •  "
            f"Senast auto-uppdaterad: **{bef.get('Senast auto-uppdaterad','')}**  •  "
            f"Källa: **{bef.get('Senast uppdaterad källa','')}**"
        )

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=_safe_float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=_safe_float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gavsek = st.number_input("GAV (SEK)", value=_safe_float(bef.get("GAV (SEK)",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=_safe_float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=_safe_float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=_safe_float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=_safe_float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=_safe_float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=_safe_float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=_safe_float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Vid spara uppdateras automatiskt (utan att röra prognosfältens manuell/0):**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, MCAP nu, Årlig utdelning, CAGR 5 år (%), Sektor, Industri via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

        spar = st.form_submit_button("💾 Spara")

    if spar and ticker:
        # se till att kolumner finns
        needed_cols = {
            "Ticker": "",
            "Bolagsnamn": "",
            "Valuta": "",
            "Senast manuellt uppdaterad": "",
            "Senast auto-uppdaterad": "",
            "Senast uppdaterad källa": "",
            "Utestående aktier": 0.0,
            "Antal aktier": 0.0,
            "GAV (SEK)": 0.0,
            "P/S": 0.0, "P/S Q1": 0.0, "P/S Q2": 0.0, "P/S Q3": 0.0, "P/S Q4": 0.0,
            "Omsättning idag": 0.0, "Omsättning nästa år": 0.0,
            "MCAP nu": 0.0,
            "Årlig utdelning": 0.0,
            "CAGR 5 år (%)": 0.0,
            "Sektor": "", "Industri": "",
        }
        df = _ensure_cols(df, needed_cols)

        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV (SEK)": gavsek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Bestäm om manuell-ts ska sättas + vilka TS-fält som ska stämplas
        MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
        datum_sätt = False
        changed_manual_fields = []
        if not bef.empty:
            before = {f: _safe_float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: _safe_float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            for k in MANUELL_FALT_FOR_DATUM:
                if before[k] != after[k]:
                    datum_sätt = True
                    changed_manual_fields.append(k)
        else:
            if any(_safe_float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
                changed_manual_fields = [f for f in MANUELL_FALT_FOR_DATUM if _safe_float(ny.get(f,0.0)) != 0.0]

        # Skriv in nya fält
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sektor","Industri"] and not str(c).startswith("TS_") else "") for c in df.columns}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Sätt manuell TS + TS_ per fält
        if datum_sätt:
            ridx = df.index[df["Ticker"]==ticker][0]
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo (rör ej prognosfält)
        data = hamta_yahoo_fält(ticker)
        ridx = df.index[df["Ticker"]==ticker][0]
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if _safe_float(data.get("Aktuell kurs",0))>0: df.loc[ridx, "Aktuell kurs"] = _safe_float(data["Aktuell kurs"])
        if _safe_float(data.get("MCAP nu",0))>0:      df.loc[ridx, "MCAP nu"]      = _safe_float(data["MCAP nu"])
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None:
            df.loc[ridx, "Årlig utdelning"] = _safe_float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:
            df.loc[ridx, "CAGR 5 år (%)"]   = _safe_float(data.get("CAGR 5 år (%)") or 0.0)
        if data.get("Sektor"):   df.loc[ridx, "Sektor"]   = str(data["Sektor"])
        if data.get("Industri"): df.loc[ridx, "Industri"] = str(data["Industri"])

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # --- Snabb-åtgärder för valt bolag ---
    st.markdown("### ⚡ Snabb-åtgärder (valt bolag)")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📈 Uppdatera kurs (endast)", key="single_price_update") and valt_label and valt_label in namn_map:
            tkr = namn_map[valt_label]
            mask = (df["Ticker"].astype(str).str.upper() == tkr.upper())
            if mask.any():
                ridx = df.index[mask][0]
                y = hamta_yahoo_fält(tkr)
                new_vals = {}
                if _safe_float(y.get("Aktuell kurs",0))>0: new_vals["Aktuell kurs"] = _safe_float(y["Aktuell kurs"])
                if y.get("Valuta"): new_vals["Valuta"] = str(y["Valuta"]).upper()
                if _safe_float(y.get("MCAP nu",0))>0: new_vals["MCAP nu"] = _safe_float(y["MCAP nu"])
                changes = {}
                apply_auto_updates_to_row(df, ridx, new_vals, source="Auto (Yahoo kurs/valuta/mcap)", changes_map=changes, stamp_even_if_same=True)
                df = uppdatera_berakningar(df, user_rates)
                spara_data(df, do_snapshot=False)
                st.success(f"{tkr}: kurs/valuta/mcap uppdaterade.")
            else:
                st.warning(f"{tkr} hittades inte i tabellen.")
    with col_b:
        if st.button("🤖 Full auto för detta bolag (ej prognosfält)", key="single_full_auto") and valt_label and valt_label in namn_map:
            df2, log = auto_update_single(df, user_rates, namn_map[valt_label], make_snapshot=False)
            st.session_state["last_auto_log"] = log
            st.success("Klar.")
            df = df2

    # Visa lite fakta om valt bolag
    if valt_label and valt_label in namn_map:
        tkr = namn_map[valt_label]
        r = df[df["Ticker"] == tkr]
        if not r.empty:
            r0 = r.iloc[0]
            st.subheader(f"{r0.get('Bolagsnamn','')} ({tkr})")
            mcap_fmt = format_mcap_short(_safe_float(r0.get("MCAP nu",0.0)))
            st.markdown(
                f"- **Aktuell kurs:** {round(_safe_float(r0.get('Aktuell kurs',0.0)),2)} {r0.get('Valuta','')}\n"
                f"- **MCAP nu:** {mcap_fmt}\n"
                f"- **Sektor/Industri:** {r0.get('Sektor','')} / {r0.get('Industri','')}\n"
                f"- **GAV (SEK):** {round(_safe_float(r0.get('GAV (SEK)',0.0)),2)}\n"
            )

    st.divider()
    # --- Manuell prognoslista (flyttad hit enligt önskemål) ---
    st.subheader("📝 Manuell prognoslista (äldst TS: Omsättning idag / nästa år)")
    limit = st.slider("Visa topp N (äldst)", min_value=5, max_value=100, value=30, step=5, key="man_progn_limit")
    mp = _manual_prognoslista(df, limit=int(limit))
    if mp.empty:
        st.info("Alla prognosfält verkar nyligen uppdaterade.")
    else:
        st.dataframe(mp, use_container_width=True, hide_index=True)

    return df

# =========================================
# Del 6/N — Analys, Portfölj & Investeringsförslag
# =========================================

# ---- Risketikett från MCAP ----
def risk_label_from_mcap(mcap_val: float) -> str:
    v = _safe_float(mcap_val, 0.0)
    # Grova trösklar (oavsett valuta) – används som heuristik
    if v >= 2e11:  # >= 200B
        return "Mega"
    if v >= 1e10:  # 10–200B
        return "Large"
    if v >= 2e9:   # 2–10B
        return "Mid"
    if v >= 3e8:   # 0.3–2B
        return "Small"
    return "Micro"

def _ensure_metrics_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Säkerställ kolumner som används i scoring/visning finns."""
    defaults = {
        "Aktuell kurs": 0.0, "Valuta": "", "MCAP nu": 0.0, "Utestående aktier": 0.0,
        "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0,
        "P/S": 0.0, "P/S Q1": 0.0, "P/S Q2": 0.0, "P/S Q3": 0.0, "P/S Q4": 0.0, "P/S-snitt": 0.0,
        "Omsättning idag": 0.0, "Omsättning nästa år": 0.0,
        "Riktkurs idag": 0.0, "Riktkurs om 1 år": 0.0, "Riktkurs om 2 år": 0.0, "Riktkurs om 3 år": 0.0,
        "Sektor": "", "Industri": "",
        # Valfria kvalitetsmått – fylls om vi börjar hämta dem:
        "Debt/Equity": 0.0, "Bruttomarginal (%)": 0.0, "Nettomarginal (%)": 0.0,
        "EV/EBITDA": 0.0, "ROIC (%)": 0.0, "ROE (%)": 0.0,
        "R&D / Sales (%)": 0.0, "Capex / Sales (%)": 0.0,
        "Kassa": 0.0, "Skulder": 0.0, "FCF TTM": 0.0, "OpCF TTM": 0.0,
        "Payout FCF (%)": 0.0, "GAV (SEK)": 0.0,
        # Historik (om vi senare fyller):
        "MCAP Q1": 0.0, "MCAP Q2": 0.0, "MCAP Q3": 0.0, "MCAP Q4": 0.0,
    }
    for c, d in defaults.items():
        if c not in df.columns:
            df[c] = d
    return df

# ---- Scoring ----
def _norm01(x, xmin, xmax):
    try:
        x = float(x)
        return max(0.0, min(1.0, (x - xmin) / (xmax - xmin))) if xmax > xmin else 0.0
    except Exception:
        return 0.0

def _score_growth(row: pd.Series, riktkurs_val: str) -> float:
    price = _safe_float(row.get("Aktuell kurs", 0.0))
    target = _safe_float(row.get(riktkurs_val, 0.0))
    ps_now = _safe_float(row.get("P/S", 0.0))
    ps_avg = _safe_float(row.get("P/S-snitt", 0.0))
    cagr = max(0.0, _safe_float(row.get("CAGR 5 år (%)", 0.0)))  # clamp lower bound

    # 1) Uppsida
    upside = 0.0
    if price > 0 and target > 0:
        upside = (target - price) / price  # kan vara negativ
    s_up = _norm01(upside, -0.3, 1.0)   # -30%..+100%

    # 2) Tillväxt (CAGR 0..50%+)
    s_cagr = _norm01(min(cagr, 50.0), 0.0, 50.0)

    # 3) Värdering relativ P/S-snitt (<= snitt är bäst)
    if ps_now > 0 and ps_avg > 0:
        rel = ps_now / ps_avg
        s_val = 1.0 - _norm01(min(rel, 2.0), 1.0, 2.0)  # 1.0 (vid =snitt) -> 0 vid 2x snitt
        s_val = max(0.0, s_val)
    else:
        s_val = 0.5

    # 4) Kvalitet (marginaler/ROIC om finns)
    gm = _safe_float(row.get("Bruttomarginal (%)", 0.0))
    nm = _safe_float(row.get("Nettomarginal (%)", 0.0))
    roic = _safe_float(row.get("ROIC (%)", 0.0))
    qual_raw = 0.4*_norm01(gm, 0, 60) + 0.4*_norm01(nm, -20, 30) + 0.2*_norm01(roic, 0, 30)
    s_qual = max(0.0, min(1.0, qual_raw))

    # 5) Risk (större bolag = något högre)
    risk = risk_label_from_mcap(row.get("MCAP nu", 0.0))
    s_risk = {"Micro":0.2,"Small":0.4,"Mid":0.7,"Large":0.9,"Mega":1.0}.get(risk,0.5)

    # Viktning
    score = 0.40*s_up + 0.30*s_cagr + 0.15*s_val + 0.10*s_qual + 0.05*s_risk
    return round(score*100.0, 1)

def _score_dividend(row: pd.Series) -> float:
    price = _safe_float(row.get("Aktuell kurs", 0.0))
    div_per_sh = _safe_float(row.get("Årlig utdelning", 0.0))
    ps_now = _safe_float(row.get("P/S", 0.0))
    ps_avg = _safe_float(row.get("P/S-snitt", 0.0))
    fcf = _safe_float(row.get("FCF TTM", 0.0))
    payout_fcf = _safe_float(row.get("Payout FCF (%)", 0.0))
    cash = _safe_float(row.get("Kassa", 0.0))

    dy = (div_per_sh/price)*100.0 if price > 0 and div_per_sh > 0 else 0.0
    s_yield = _norm01(min(dy, 12.0), 0.0, 12.0)  # prefer 6–12%

    # Payout på FCF (lägre bättre, 0–80% föredras)
    if payout_fcf > 0:
        s_payout = 1.0 - _norm01(min(payout_fcf, 120.0), 40.0, 120.0)  # bäst ~40–60%
        s_payout = max(0.0, s_payout)
    else:
        # Om vi saknar payout men FCF positiv, ge neutral-poäng
        s_payout = 0.5 if fcf >= 0 else 0.1

    # Valuation via P/S mot snitt
    if ps_now > 0 and ps_avg > 0:
        rel = ps_now / ps_avg
        s_val = 1.0 - _norm01(min(rel, 2.0), 1.0, 2.0)
        s_val = max(0.0, s_val)
    else:
        s_val = 0.5

    # Kassastyrka (proxy): Cash relativt MCAP
    mcap = _safe_float(row.get("MCAP nu",0.0))
    s_cash = _norm01(cash/max(mcap,1e-9), 0.0, 0.2) if mcap>0 else 0.5  # 0..20% av MCAP

    # Risk
    risk = risk_label_from_mcap(row.get("MCAP nu", 0.0))
    s_risk = {"Micro":0.2,"Small":0.4,"Mid":0.7,"Large":0.9,"Mega":1.0}.get(risk,0.5)

    score = 0.40*s_yield + 0.30*s_payout + 0.15*s_val + 0.10*s_cash + 0.05*s_risk
    return round(score*100.0, 1)

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    df = _ensure_metrics_cols(df)
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
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
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","MCAP nu",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","GAV (SEK)",
        "Sektor","Industri",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    show = pd.DataFrame([r[cols].to_dict()])
    if "MCAP nu" in show.columns:
        show["MCAP nu"] = show["MCAP nu"].apply(format_mcap_short)
    st.dataframe(show, use_container_width=True, hide_index=True)

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    df = _ensure_metrics_cols(df)
    port = df[_safe_float_series(df.get("Antal aktier", pd.Series([0]*len(df)))) > 0].copy() if "Antal aktier" in df else pd.DataFrame()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs + SEK-värden
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)
    port["Total årlig utdelning (SEK)"] = (port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]).astype(float)
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    port["Risklabel"] = port["MCAP nu"].apply(risk_label_from_mcap)

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    # Tabell
    port["MCAP (kort)"] = port["MCAP nu"].apply(format_mcap_short)
    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)","Risklabel","MCAP (kort)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    # Sektorfördelning
    if "Sektor" in port.columns:
        sect = port.groupby("Sektor", dropna=False)["Värde (SEK)"].sum().sort_values(ascending=False)
        if not sect.empty:
            st.markdown("**Sektorfördelning (SEK)**")
            st.dataframe(sect.reset_index().rename(columns={"Värde (SEK)":"Summa (SEK)"}), use_container_width=True, hide_index=True)

def _safe_float_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    except Exception:
        return pd.Series([0.0]*len(s))

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    df = _ensure_metrics_cols(df)

    # Val: tillväxt eller utdelning
    mode = st.radio("Fokus", ["Tillväxt","Utdelning"], horizontal=True, key="if_mode")

    # Riktkurs-val för tillväxtläge
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas? (Tillväxt-läget)",
        ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=0, key="if_riktkurs"
    )

    # Filter: sektor och risklabel
    sektorer = sorted(list({str(x) for x in df.get("Sektor", pd.Series([])) if str(x).strip()}))
    risklabels = ["Micro","Small","Mid","Large","Mega"]
    f_sek = st.multiselect("Filtrera sektor(er)", sektorer, default=sektorer if sektorer else [], key="if_sectors")
    f_risk = st.multiselect("Filtrera risklabel", risklabels, default=risklabels, key="if_risks")

    # Baspopulation: måste ha pris & namn
    base = df[(df["Aktuell kurs"] > 0) & (df["Bolagsnamn"].astype(str).str.len() > 0)].copy()
    if f_sek:
        base = base[base["Sektor"].astype(str).isin(f_sek)]
    if f_risk:
        base["Risklabel"] = base["MCAP nu"].apply(risk_label_from_mcap)
        base = base[base["Risklabel"].isin(f_risk)]

    if mode == "Tillväxt":
        # Måste ha riktkurs + ps-data
        base = base[(base[riktkurs_val] > 0) & ((base["P/S"] > 0) | (base["P/S-snitt"] > 0))].copy()
        base["Score"] = base.apply(lambda r: _score_growth(r, riktkurs_val), axis=1)
        base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    else:
        # Utdelning: måste ha utdelning och pris
        base = base[(base["Årlig utdelning"] > 0)].copy()
        base["Score"] = base.apply(_score_dividend, axis=1)
        base["Direktavkastning (%)"] = np.where(base["Aktuell kurs"]>0, (base["Årlig utdelning"]/base["Aktuell kurs"])*100.0, 0.0)

    if base.empty:
        st.info("Inga bolag matchar dina filter just nu.")
        return

    # Sortera på score
    base = base.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # Robust bläddring
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag", key="if_prev"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}  •  Visar **{mode.lower()}**-läge")
    with col_next:
        if st.button("➡️ Nästa förslag", key="if_next"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Växelkurs och SEK-kurs
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx

    # Nuvarande portföljandel
    port = df[df.get("Antal aktier",0) > 0].copy()
    nuv_innehav = 0.0; port_värde = 0.0; nuv_andel = 0.0
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
        rmatch = port[port["Ticker"].astype(str).str.upper() == str(rad["Ticker"]).upper()]
        if not rmatch.empty:
            nuv_innehav = float(rmatch["Värde (SEK)"].sum())
            nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Utskrift
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    # P/S-data
    ps_now = _safe_float(rad.get("P/S",0.0))
    ps_avg = _safe_float(rad.get("P/S-snitt",0.0))

    # MCAP nu i snyggt format
    mcap_fmt = format_mcap_short(_safe_float(rad.get("MCAP nu",0.0)))

    lines = [
        f"- **Aktuell kurs:** {round(_safe_float(rad['Aktuell kurs']),2)} {rad['Valuta']}  (~{round(kurs_sek,2)} SEK)",
        f"- **Utestående aktier:** {round(_safe_float(rad.get('Utestående aktier',0.0)),2)} M",
        f"- **MCAP nu:** {mcap_fmt}",
        f"- **P/S (nu):** {round(ps_now,2)}",
        f"- **P/S-snitt (Q1–Q4):** {round(ps_avg,2)}",
    ]

    if mode == "Tillväxt":
        pot = (rad[riktkurs_val] - rad["Aktuell kurs"]) / rad["Aktuell kurs"] * 100.0 if rad["Aktuell kurs"]>0 and rad[riktkurs_val]>0 else 0.0
        lines += [
            f"- **Vald riktkurs ({riktkurs_val}):** {round(_safe_float(rad[riktkurs_val]),2)} {rad['Valuta']}",
            f"- **Uppsida (valda riktkursen):** {round(pot,2)} %",
            f"- **CAGR 5 år:** {round(_safe_float(rad.get('CAGR 5 år (%)',0.0)),2)} %",
            f"- **Score (tillväxt):** {round(_safe_float(rad.get('Score',0.0)),1)}",
        ]
    else:
        dy = (rad["Årlig utdelning"]/rad["Aktuell kurs"]*100.0) if rad["Aktuell kurs"]>0 and rad["Årlig utdelning"]>0 else 0.0
        lines += [
            f"- **Årlig utdelning:** {round(_safe_float(rad.get('Årlig utdelning',0.0)),2)} {rad['Valuta']}  (DY ~ {round(dy,2)} %)",
            f"- **Score (utdelning):** {round(_safe_float(rad.get('Score',0.0)),1)}",
        ]

    lines += [
        f"- **Nuvarande andel i portfölj:** {nuv_andel} %",
        f"- **Sektor/Industri:** {rad.get('Sektor','')} / {rad.get('Industri','')}",
    ]
    st.markdown("\n".join(lines))

    # Expander för mer nyckeltal
    with st.expander("Visa mer nyckeltal"):
        more = []
        # Historiska P/S & MCAP om finns (>0)
        for q in [1,2,3,4]:
            v = _safe_float(rad.get(f"P/S Q{q}", 0.0))
            if v > 0:
                more.append(f"- **P/S Q{q}:** {round(v,2)}")
        for q in [1,2,3,4]:
            v = _safe_float(rad.get(f"MCAP Q{q}", 0.0))
            if v > 0:
                more.append(f"- **MCAP Q{q}:** {format_mcap_short(v)}")

        # Kvalitetsnyckeltal (om finns)
        kv = [
            ("Debt/Equity","Debt/Equity"),
            ("Bruttomarginal (%)","%"), ("Nettomarginal (%)","%"),
            ("EV/EBITDA","x"),
            ("ROIC (%)","%"), ("ROE (%)","%"),
            ("R&D / Sales (%)","%"), ("Capex / Sales (%)","%"),
        ]
        for k, suf in kv:
            vv = rad.get(k, None)
            if vv is not None and _safe_float(vv) != 0.0:
                more.append(f"- **{k}:** {round(_safe_float(vv),2)}{suf if suf!='x' else 'x'}")

        # Kassa/FCF & runway
        cash = _safe_float(rad.get("Kassa",0.0))
        fcf = _safe_float(rad.get("FCF TTM",0.0))
        if cash != 0.0:
            more.append(f"- **Kassa:** {format_mcap_short(cash)}")
        if fcf != 0.0:
            more.append(f"- **FCF TTM:** {format_mcap_short(fcf)}")
        runway = None
        if fcf < 0:
            quarterly_burn = abs(fcf)/4.0
            runway = cash/quarterly_burn if quarterly_burn > 0 else None
        if runway is not None:
            more.append(f"- **Runway (kvartal):** {round(runway,1)}")

        if more:
            st.markdown("\n".join(more))
        else:
            st.write("–")

# =========================================
# Del 7/N — Sidopanel, Batch, Main
# =========================================

# --------- Sidopanel: Valutakurser (utan widget-krockar) ----------
def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # Läs sparade / standard
    saved_rates = las_sparade_valutakurser()
    def _def(k, fallback):
        try:
            return float(saved_rates.get(k, fallback))
        except Exception:
            return float(fallback)

    # Init “state” (icke-widget) en gång
    if "rate_usd" not in st.session_state: st.session_state.rate_usd = _def("USD", STANDARD_VALUTAKURSER["USD"])
    if "rate_nok" not in st.session_state: st.session_state.rate_nok = _def("NOK", STANDARD_VALUTAKURSER["NOK"])
    if "rate_cad" not in st.session_state: st.session_state.rate_cad = _def("CAD", STANDARD_VALUTAKURSER["CAD"])
    if "rate_eur" not in st.session_state: st.session_state.rate_eur = _def("EUR", STANDARD_VALUTAKURSER["EUR"])

    # Separata widget-keys för inputs (så vi kan uppdatera dem programatiskt)
    if "rate_usd_input" not in st.session_state: st.session_state.rate_usd_input = st.session_state.rate_usd
    if "rate_nok_input" not in st.session_state: st.session_state.rate_nok_input = st.session_state.rate_nok
    if "rate_cad_input" not in st.session_state: st.session_state.rate_cad_input = st.session_state.rate_cad
    if "rate_eur_input" not in st.session_state: st.session_state.rate_eur_input = st.session_state.rate_eur

    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd_input), step=0.01, format="%.4f", key="rate_usd_input")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok_input), step=0.01, format="%.4f", key="rate_nok_input")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad_input), step=0.01, format="%.4f", key="rate_cad_input")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur_input), step=0.01, format="%.4f", key="rate_eur_input")

    # Auto-hämtning
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        # Uppdatera INPUT-fälten (inte de underliggande rate_*)
        st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
        st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
        st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
        st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
        st.sidebar.success(f"Valutakurser uppdaterade (källa: {provider}).")
        if misses:
            st.sidebar.warning("Kunde inte hämta:\n- " + "\n- ".join(misses))

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            # Spara från INPUTS till state och ark
            st.session_state.rate_usd = float(st.session_state.rate_usd_input)
            st.session_state.rate_nok = float(st.session_state.rate_nok_input)
            st.session_state.rate_cad = float(st.session_state.rate_cad_input)
            st.session_state.rate_eur = float(st.session_state.rate_eur_input)

            to_save = {
                "USD": st.session_state.rate_usd,
                "NOK": st.session_state.rate_nok,
                "CAD": st.session_state.rate_cad,
                "EUR": st.session_state.rate_eur,
                "SEK": 1.0,
            }
            spara_valutakurser(to_save)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            sr = las_sparade_valutakurser()
            # Uppdatera INPUT-fälten
            st.session_state.rate_usd_input = float(sr.get("USD", st.session_state.rate_usd_input))
            st.session_state.rate_nok_input = float(sr.get("NOK", st.session_state.rate_nok_input))
            st.session_state.rate_cad_input = float(sr.get("CAD", st.session_state.rate_cad_input))
            st.session_state.rate_eur_input = float(sr.get("EUR", st.session_state.rate_eur_input))
            st.sidebar.info("Läste sparade kurser.")

    # Bygg user_rates att använda i appen (utifrån INPUTS)
    user_rates = {
        "USD": float(st.session_state.rate_usd_input),
        "NOK": float(st.session_state.rate_nok_input),
        "CAD": float(st.session_state.rate_cad_input),
        "EUR": float(st.session_state.rate_eur_input),
        "SEK": 1.0,
    }
    return user_rates


# --------- Sidopanel: Batch-kö, Auto & Kursknapp ----------
def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: dict):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Uppdatering")

    # Kurs-knapp (alla)
    if st.sidebar.button("💹 Uppdatera endast kurser (alla)"):
        df2 = update_prices_all(df, stamp_even_if_same=True)
        st.session_state["_df_ref"] = df2

    # Auto-uppdatera ALLA
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=False, key="chk_snapshot_all")
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → FMP)"):
        df2, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot)
        st.session_state["_df_ref"] = df2
        st.session_state["last_auto_log"] = log

    # Batch-kö
    st.sidebar.subheader("🗂️ Batch-kö")
    strategy = st.sidebar.selectbox("Urval", ["Äldst uppdaterade först","A–Ö (bolagsnamn)"], key="batch_strategy")
    batch_size = st.sidebar.number_input("Bygg batch-lista (N)", min_value=1, max_value=200, value=20, step=1, key="batch_size")

    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []

    cols_b = st.sidebar.columns(3)
    with cols_b[0]:
        if st.button("Bygg kö"):
            st.session_state.batch_queue = _build_batch_queue(df, strategy=strategy, batch_size=int(batch_size))
            st.sidebar.success(f"Kö byggd med {len(st.session_state.batch_queue)} tickers.")
    with cols_b[1]:
        if st.button("Rensa kö"):
            st.session_state.batch_queue = []
            st.sidebar.info("Kö rensad.")
    with cols_b[2]:
        chunk = st.sidebar.number_input("Kör nästa M", min_value=1, max_value=200, value=min(10, max(1, len(st.session_state.batch_queue))), step=1, key="batch_chunk")

    if st.sidebar.button("▶️ Kör batch (nästa M)"):
        if not st.session_state.batch_queue:
            st.sidebar.warning("Kö är tom.")
        else:
            to_run = st.session_state.batch_queue[: int(chunk)]
            remaining = st.session_state.batch_queue[int(chunk):]
            df2, log = run_batch_update(df, user_rates, to_run, make_snapshot=False)
            st.session_state["_df_ref"] = df2
            st.session_state["last_auto_log"] = log
            st.session_state.batch_queue = remaining
            if remaining:
                st.sidebar.info(f"Kvar i kö: {len(remaining)}")
            else:
                st.sidebar.success("Kö färdig.")


# --------- MAIN ----------
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Sidopanel: valutor
    user_rates = _sidebar_rates()

    # 2) Läs data (robust — skriv aldrig tomt schema automatiskt)
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        df = pd.DataFrame(columns=FINAL_COLS)

    if df is None or df.empty:
        st.warning("Tabellen verkar tom (eller kunde inte läsas). Jag skapar ett **lokalt** tomt df så att du kan gå igenom vyerna. Inget skrivs till Sheets förrän du sparar.")
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
    else:
        # Säkerställ schema, migrera och konvertera
        df = säkerställ_kolumner(df)
        df = migrera_gamla_riktkurskolumner(df)
        df = konvertera_typer(df)

    # Håll df i session så batch/åtgärder kan uppdatera
    if "_df_ref" not in st.session_state:
        st.session_state["_df_ref"] = df
    else:
        # Om schema ändrats, aligna
        st.session_state["_df_ref"] = säkerställ_kolumner(st.session_state["_df_ref"])

    # 3) Sidopanel: batch & actions
    _sidebar_batch_and_actions(st.session_state["_df_ref"], user_rates)

    # 4) Välj vy
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], key="main_view")

    # 5) Kör vy
    if meny == "Kontroll":
        kontrollvy(st.session_state["_df_ref"])
    elif meny == "Analys":
        analysvy(st.session_state["_df_ref"], user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates)
        st.session_state["_df_ref"] = df2
    elif meny == "Investeringsförslag":
        df_calc = uppdatera_berakningar(st.session_state["_df_ref"].copy(), user_rates)
        visa_investeringsforslag(df_calc, user_rates)
    elif meny == "Portfölj":
        df_calc = uppdatera_berakningar(st.session_state["_df_ref"].copy(), user_rates)
        visa_portfolj(df_calc, user_rates)

if __name__ == "__main__":
    main()

# =========================================
# Del 8/N — Helpers, Batch, Yahoo-basics
# =========================================

# --- Småhjälpare -------------------------------------------------------------

def _safe_float(x, default: float = 0.0):
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def format_mcap_short(x: float) -> str:
    """Snygg kortform (svenska): tn, mdr, mn, tkr."""
    try:
        n = float(x)
    except Exception:
        return "-"
    absv = abs(n)
    sign = "-" if n < 0 else ""
    if absv >= 1e12:
        return f"{sign}{absv/1e12:.2f} tn"
    if absv >= 1e9:
        return f"{sign}{absv/1e9:.2f} mdr"
    if absv >= 1e6:
        return f"{sign}{absv/1e6:.2f} mn"
    if absv >= 1e3:
        return f"{sign}{absv/1e3:.0f} tkr"
    return f"{n:.0f}"

# --- APPLY uppdateringar (med val: stämpla även om samma värde) --------------

def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    stamp_even_if_same: bool = False
) -> bool:
    """
    Skriver endast fält som får nytt (meningsfullt) värde – eller stämplar även om lika
    om 'stamp_even_if_same=True'. Uppdaterar TS_ för spårade fält, samt auto-TS/källa.
    Returnerar True om något fält ändrades (skrevs om).
    """
    changed_fields = []
    wrote_anything = False

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f] if row_idx in df.index else None

        # Bestäm om v är "skrivbart"
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            #För klassiska P/S-fält kräver vi >0, för övriga numeric ≥0
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]:
                write_ok = (float(v) > 0)
            else:
                write_ok = (float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        if not write_ok and not stamp_even_if_same:
            continue

        # Om lika och inte tvingad stämpling → hoppa
        same = (str(old) == str(v))
        if same and not stamp_even_if_same:
            continue

        # Skriv värdet om vi har ett vettigt värde eller om vi tvingar stämpling
        if write_ok or stamp_even_if_same:
            df.at[row_idx, f] = v
            wrote_anything = True
            if not same:
                changed_fields.append(f)
            # TS för spårade fält
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)

    # Auto-TS och källa – även om värden blev lika men vi stampade
    if wrote_anything or stamp_even_if_same:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            tkr = df.at[row_idx, "Ticker"] if "Ticker" in df.columns else f"row{row_idx}"
            changes_map.setdefault(str(tkr), []).extend(changed_fields)
        return bool(changed_fields)
    return False

# --- Förbättrad Yahoo-basics (lägger till MCAP/Sektor/Industri) --------------

def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Basfält från Yahoo:
      - Bolagsnamn, Kurs, Valuta, Årlig utdelning, CAGR 5 år (%)
      - + MCAP nu, Sektor, Industri (om tillgängligt)
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "MCAP nu": 0.0,
        "Sektor": "",
        "Industri": "",
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

        # Valuta
        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()

        # Namn
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

        # MCAP
        mcap = info.get("marketCap", None)
        try:
            if mcap is not None:
                out["MCAP nu"] = float(mcap)
        except Exception:
            pass

        # Sektor/Industri
        if info.get("sector"):  out["Sektor"] = str(info["sector"])
        if info.get("industry"): out["Industri"] = str(info["industry"])

        # CAGR (enkel approx på Total Revenue)
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# --- PRIS-uppdatering för alla -----------------------------------------------

def update_prices_all(df: pd.DataFrame, stamp_even_if_same: bool = True) -> pd.DataFrame:
    """
    Hämtar Aktuell kurs, Valuta, MCAP nu för alla rader (utan att röra prognosfält).
    Stämplar 'Senast auto-uppdaterad' även om priset råkar vara oförändrat.
    """
    total = len(df)
    if total == 0:
        st.sidebar.info("Inget att uppdatera.")
        return df

    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()
    changes = {}
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        tkr = str(row.get("Ticker","")).strip().upper()
        status.write(f"Prisuppdatering {i}/{total}: {tkr}")
        if not tkr:
            progress.progress(i/total)
            continue
        try:
            y = hamta_yahoo_fält(tkr)
            new_vals = {}
            if _safe_float(y.get("Aktuell kurs",0)) >= 0: new_vals["Aktuell kurs"] = _safe_float(y["Aktuell kurs"])
            if y.get("Valuta"): new_vals["Valuta"] = str(y["Valuta"]).upper()
            if _safe_float(y.get("MCAP nu",0)) >= 0: new_vals["MCAP nu"] = _safe_float(y["MCAP nu"])
            ridx = df.index[df["Ticker"].astype(str).str.upper()==tkr]
            if len(ridx)>0:
                apply_auto_updates_to_row(df, ridx[0], new_vals, source="Auto (Yahoo: kurs/valuta/mcap)", changes_map=changes, stamp_even_if_same=stamp_even_if_same)
        except Exception:
            pass
        progress.progress(i/total)

    df = uppdatera_berakningar(df, user_rates={"USD":1,"NOK":1,"CAD":1,"EUR":1,"SEK":1})  # beräkningar ej beroende av FX här
    spara_data(df, do_snapshot=False)
    st.sidebar.success("Klar med kursuppdatering.")
    return df

# --- AUTO-uppdatering för EN ticker ------------------------------------------

def auto_update_single(df: pd.DataFrame, user_rates: dict, ticker: str, make_snapshot: bool = False):
    """
    Kör auto_fetch_for_ticker för EN ticker och commit:ar om något skrivs.
    Returnerar (df, log)
    """
    log = {"changed": {}, "misses": {}}
    tkr = str(ticker).upper().strip()
    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        st.warning(f"{tkr} hittades inte i tabellen.")
        return df, log
    ridx = df.index[mask][0]
    try:
        new_vals, debug = auto_fetch_for_ticker(tkr)
        changed = apply_auto_updates_to_row(df, ridx, new_vals, source="Auto (SEC/Yahoo→FMP)", changes_map=log["changed"])
        if not changed:
            log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df, do_snapshot=make_snapshot)
        st.success(f"Auto-uppdatering klar för {tkr}.")
    except Exception as e:
        log["misses"][tkr] = [f"error: {e}"]
        st.warning(f"Kunde inte auto-uppdatera {tkr}: {e}")
    return df, log

# --- Batch-kö & körning ------------------------------------------------------

def _candidate_order(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Returnerar DataFrame sorterad enligt strategi."""
    if strategy.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
        return work
    else:
        return df.sort_values(by=["Bolagsnamn","Ticker"])

def _build_batch_queue(df: pd.DataFrame, strategy: str, batch_size: int) -> list:
    """
    Bygger en kö av tickers enligt vald strategi, hoppar över tickers som redan
    körts i nuvarande session (batch_seen). Återställ via 'Rensa kö'.
    """
    if "batch_seen" not in st.session_state:
        st.session_state.batch_seen = set()

    ordered = _candidate_order(df, strategy)
    tickers = []
    for _, r in ordered.iterrows():
        tkr = str(r.get("Ticker","")).strip().upper()
        if not tkr or tkr in st.session_state.batch_seen:
            continue
        tickers.append(tkr)
        if len(tickers) >= batch_size:
            break
    return tickers

def run_batch_update(df: pd.DataFrame, user_rates: dict, tickers: list, make_snapshot: bool = False):
    """
    Kör auto_fetch_for_ticker på en lista tickers (del av kö). Visar progress 1/X.
    Sparar efter varje lyckad uppdatering för att inte riskera dataförlust.
    """
    total = len(tickers)
    log = {"changed": {}, "misses": {}}
    if total == 0:
        st.sidebar.info("Inget i batchlistan.")
        return df, log

    if "batch_seen" not in st.session_state:
        st.session_state.batch_seen = set()

    prog = st.sidebar.progress(0.0)
    status = st.sidebar.empty()
    for i, tkr in enumerate(tickers, start=1):
        status.write(f"Kör {i}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            # hitta rad
            mask = (df["Ticker"].astype(str).str.upper() == tkr.upper())
            if not mask.any():
                log["misses"][tkr] = ["(ticker saknas i tabellen)"]
            else:
                ridx = df.index[mask][0]
                changed = apply_auto_updates_to_row(df, ridx, new_vals, source="Auto (SEC/Yahoo→FMP)", changes_map=log["changed"])
                if not changed:
                    log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
                # räkna om och spara efter varje ticker för säkerhet
                df = uppdatera_berakningar(df, user_rates)
                spara_data(df, do_snapshot=False)
            st.session_state.batch_seen.add(tkr)
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]
        prog.progress(i/total)

    st.sidebar.success("Batch-körning klar.")
    return df, log
