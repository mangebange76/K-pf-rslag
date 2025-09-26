# -*- coding: utf-8 -*-
# app.py — Del 1/?  — Grund, Google Sheets, valutor, schema & TS-helpers

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

# ===================== Google Sheets-koppling ================================

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

# ========================= Tids-/utility =====================================

def _ts_str():
    return now_dt().strftime("%Y%m%d-%H%M%S")

def _ts_datetime():
    return now_dt()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# =================== Standard valutakurser till SEK ==========================

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

# =================== Kolumnschema & tidsstämplar =============================

# Spårade fält → respektive TS-kolumn (uppdateras när fältet ändras eller stämplas)
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

# Slutlig kolumnlista i databasen (bas)
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
    # TS-kolumner
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# Extra kolumner för utökad analys/visning
EXTRA_DEFAULT_COLS = [
    "Sektor","Industri","MarketCap (SEK)","Risklabel",
    "Momentum 6m (%)","Momentum 12m (%)",
    "EV/S (TTM)","EV/EBITDA (TTM)","P/E (TTM)",
    "FCF (TTM)","FCF-yield (%)",
    "Direktavkastning (%)","Payout FCF","Dividend-säkerhet",
    "_marketCap_raw", "_debug_shares_source"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    for kol in FINAL_COLS + EXTRA_DEFAULT_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in [
                "kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs",
                "aktier","snitt","yield","payout","momentum","ev/","p/e","fcf",
                "marketcap"
            ]):
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
        "Momentum 6m (%)","Momentum 12m (%)","EV/S (TTM)","EV/EBITDA (TTM)","P/E (TTM)",
        "FCF (TTM)","FCF-yield (%)","MarketCap (SEK)","_marketCap_raw"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Sektor","Industri",
              "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
              "Dividend-säkerhet","Risklabel","_debug_shares_source"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)
    return df

# ======================= Tidsstämpelshjälpare ================================

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

# app.py — Del 2/? — Yahoo-hjälpare, beräkningar & TS/diagnostik

# ===================== Yahoo-hjälpare & basfält ==============================

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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Årlig utdelning, CAGR, Sektor, Industri.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
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

        pris = info.get("regularMarketPrice")
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

        div_rate = info.get("dividendRate")
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # Sektor/industri om tillgängligt
        if info.get("sector"):
            out["Sektor"] = str(info.get("sector"))
        if info.get("industry"):
            out["Industri"] = str(info.get("industry"))

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ===================== Beräkningar (riktkurser m.m.) =========================

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar per rad:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier (miljoner)
    Antaganden:
      * 'Utestående aktier' lagras i MILJONER st.
      * 'Omsättning …' lagras i MILJONER i bolagsvaluta.
      => Riktkurs = (Omsättning (mn) * P/S) / (Aktier (mn))  [valuta/aktie]
    """
    if df.empty:
        return df
    df = df.copy()

    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if pd.notna(x) and float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: >100% → 50%, <0% → 2%
        try:
            cagr = float(rad.get("CAGR 5 år (%)", 0.0) or 0.0)
        except Exception:
            cagr = 0.0
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år"
        try:
            oms_next = float(rad.get("Omsättning nästa år", 0.0) or 0.0)
        except Exception:
            oms_next = 0.0
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # lämna befintliga värden om de finns
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0) or 0.0)
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0) or 0.0)

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        try:
            aktier_ut_mn = float(rad.get("Utestående aktier", 0.0) or 0.0)  # miljONER
        except Exception:
            aktier_ut_mn = 0.0

        if aktier_ut_mn > 0 and ps_snitt > 0:
            try:
                oms_idag_mn  = float(rad.get("Omsättning idag", 0.0) or 0.0)
                oms_1y_mn    = float(rad.get("Omsättning nästa år", 0.0) or 0.0)
                oms_2y_mn    = float(df.at[i, "Omsättning om 2 år"] or 0.0)
                oms_3y_mn    = float(df.at[i, "Omsättning om 3 år"] or 0.0)
            except Exception:
                oms_idag_mn = oms_1y_mn = oms_2y_mn = oms_3y_mn = 0.0

            def _rk(oms_mn: float) -> float:
                if oms_mn <= 0:
                    return 0.0
                return round((oms_mn * ps_snitt) / aktier_ut_mn, 2)

            df.at[i, "Riktkurs idag"]    = _rk(oms_idag_mn)
            df.at[i, "Riktkurs om 1 år"] = _rk(oms_1y_mn)
            df.at[i, "Riktkurs om 2 år"] = _rk(oms_2y_mn)
            df.at[i, "Riktkurs om 3 år"] = _rk(oms_3y_mn)
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# ===================== TS/ändringsskrivning ==================================

def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    always_stamp: bool = False
) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde,
    och stämplar TS_ för spårade fält. Om always_stamp=True stämplas TS även
    om värdet är samma (bra för “jag har kollat idag”).
    Returnerar True om något fält faktiskt ändrades.
    """
    changed_fields = []
    stamped_any = False

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue

        # Bestäm om det är skrivbart
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # P/S & aktier kräver >0, övriga numeriska >=0
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]:
                write_ok = (float(v) > 0)
            else:
                write_ok = (float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            write_ok = v is not None

        if not write_ok:
            # även om vi inte skriver nytt värde, kan vi TS-stämpla om always_stamp
            if always_stamp and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                stamped_any = True
            continue

        old = df.at[row_idx, f]
        should_write = (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))

        if should_write:
            df.at[row_idx, f] = v
            changed_fields.append(f)
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                stamped_any = True
        else:
            # ingen faktisk ändring – men TS-stämpla om always_stamp
            if always_stamp and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                stamped_any = True

    if changed_fields or stamped_any:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return bool(changed_fields)
    return False

# ===================== Diagnostik: äldsta TS =================================

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
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(oldest_any_ts, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

# ===================== Kontroll-lista (manuell hantering) ====================

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Bolag som sannolikt kräver manuell hantering:
    - saknar någon av kärnfälten eller TS,
    - och äldsta TS är äldre än 'older_than_days'.
    """
    need_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS[c] for c in TS_FIELDS if c in need_cols]

    out_rows = []
    try:
        days = int(older_than_days or 365)
    except Exception:
        days = 365
    cutoff = now_dt() - timedelta(days=days)

    for _, r in df.iterrows():
        missing_val = any((float(r.get(c, 0.0) or 0.0) <= 0.0) for c in need_cols)
        missing_ts  = any((not str(r.get(ts, "") or "").strip()) for ts in ts_cols if ts in r)
        oldest = oldest_any_ts(r)
        oldest_dt = pd.to_datetime(oldest, errors="coerce")
        too_old = bool(pd.notna(oldest_dt) and oldest_dt.to_pydatetime() < cutoff)

        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Äldsta TS": oldest_dt.strftime("%Y-%m-%d") if pd.notna(oldest_dt) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })

    return pd.DataFrame(out_rows)

# app.py — Del 3/? — Datakällor: SEC + Yahoo + FMP + Finnhub och extra nyckeltal

# ======================= SEC (US + FPI/IFRS) =================================

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

# ---------- datumhjälp & robusta "instant" aktier (multi-class) --------------

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

# ---------- IFRS/GAAP kvartalsintäkter + valuta (med Dec/Jan-fix) ------------

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
    Dec/Jan-fix: deduplicera på kvartal (YYYY-Q) & fyll luckor där SEC lagt Q4 i jan istället för dec.
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

    def quarter_key(d):
        # definiera kvartal efter månad → Q1(2-4), Q2(5-7), Q3(8-10), Q4(11-1)
        # hantera att vissa Q4 slutar i jan.
        m = d.month
        if m in (2,3,4):  q = 1; y = d.year
        elif m in (5,6,7): q = 2; y = d.year
        elif m in (8,9,10): q = 3; y = d.year
        else:  # (11,12,1)
            q = 4
            y = d.year if m in (11,12) else d.year-1  # jan hör till föregående räkenskapsår
        return f"{y}-Q{q}"

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
                # Deduplicera per kvartal-nyckel, välj den senast rapporterade posten per kvartal
                ded = {}
                for end, v in tmp:
                    k = quarter_key(end)
                    if (k not in ded) or (end > ded[k][0]):
                        ded[k] = (end, v)
                rows = sorted(ded.values(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
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
    Innehåller deduplicering per kvartal (Dec/Jan-fix).
    """
    def quarter_key(d):
        m = d.month
        if m in (2,3,4):  q = 1; y = d.year
        elif m in (5,6,7): q = 2; y = d.year
        elif m in (8,9,10): q = 3; y = d.year
        else:
            q = 4
            y = d.year if m in (11,12) else d.year-1
        return f"{y}-Q{q}"

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
                    tmp = []
                    for c, v in row.items():
                        try:
                            d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                            tmp.append((d, float(v)))
                        except Exception:
                            pass
                    # dedup per kvartal
                    ded = {}
                    for d, v in tmp:
                        k = quarter_key(d)
                        if (k not in ded) or (d > ded[k][0]):
                            ded[k] = (d, v)
                    out = sorted(ded.values(), key=lambda x: x[0], reverse=True)
                    return out
    except Exception:
        pass

    # 2) income_stmt quarterly via v1-api (kan vara tomt)
    try:
        df_is = getattr(t, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
            tmp = []
            for c, v in ser.items():
                try:
                    d = c.date() if hasattr(c, "date") else pd.to_datetime(c).date()
                    tmp.append((d, float(v)))
                except Exception:
                    pass
            ded = {}
            for d, v in tmp:
                k = quarter_key(d)
                if (k not in ded) or (d > ded[k][0]):
                    ded[k] = (d, v)
            out = sorted(ded.values(), key=lambda x: x[0], reverse=True)
            return out
    except Exception:
        pass

    return []

# =================== FMP (lätt & full) =======================================

FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.5))
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20))

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
    Namn/valuta fylls via Yahoo senare om saknas.
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

# =================== Extra nyckeltal via Yahoo ================================

def _calc_ttm_from_q(df_q: pd.DataFrame, row_name: str) -> Optional[float]:
    """Summera senaste fyra kvartal för en rad i Yahoo quarterly_* DataFrame."""
    try:
        if not isinstance(df_q, pd.DataFrame) or df_q.empty or row_name not in df_q.index:
            return None
        ser = df_q.loc[row_name].dropna().astype(float)
        if ser.empty:
            return None
        vals = list(ser.values)[:4]  # senaste → äldst
        if len(vals) < 1:
            return None
        return float(sum(vals))
    except Exception:
        return None

def enrich_extra_metrics_from_yahoo(ticker: str, base_currency: str) -> dict:
    """
    Försöker hämta EV/EBITDA, P/E, FCF(TTM), FCF-yield, direktavkastning, total skuld/kassa, marketcap m.m.
    """
    out = {}
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info_dict(t)

        # Market cap rå
        mcap = info.get("marketCap")
        if mcap is not None:
            try:
                out["_marketCap_raw"] = float(mcap)
            except Exception:
                pass

        # EV och EBITDA
        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        pe = info.get("trailingPE") or info.get("forwardPE")

        if ev and ebitda and float(ebitda) != 0:
            try: out["EV/EBITDA (TTM)"] = float(ev) / float(ebitda)
            except Exception: pass
        if pe:
            try: out["P/E (TTM)"] = float(pe)
            except Exception: pass

        # FCF (TTM) ≈ OperatingCF - CapEx (från quarterly_cashflow)
        qcf = getattr(t, "quarterly_cashflow", None)
        ocf_ttm = _calc_ttm_from_q(qcf, "Total Cash From Operating Activities") or \
                  _calc_ttm_from_q(qcf, "Operating Cash Flow")
        capex_ttm = _calc_ttm_from_q(qcf, "Capital Expenditures")
        if ocf_ttm is not None and capex_ttm is not None:
            try:
                fcf_ttm = float(ocf_ttm) - float(capex_ttm)
                out["FCF (TTM)"] = fcf_ttm
            except Exception:
                pass

        # FCF-yield = FCF / MarketCap
        if out.get("FCF (TTM)") and out.get("_marketCap_raw"):
            try:
                out["FCF-yield (%)"] = (float(out["FCF (TTM)"]) / float(out["_marketCap_raw"])) * 100.0
            except Exception:
                pass

        # Dividend yield (om pris & dividendRate kända)
        price = info.get("regularMarketPrice")
        div_rate = info.get("dividendRate")
        if price and div_rate and float(price) > 0:
            try:
                out["Direktavkastning (%)"] = (float(div_rate) / float(price)) * 100.0
            except Exception:
                pass

        # Skuld/kassa (kan nyttjas i andra vyer)
        td = info.get("totalDebt")
        tc = info.get("totalCash")
        if td is not None:
            try: out["Total Debt"] = float(td)
            except: pass
        if tc is not None:
            try: out["Total Cash"] = float(tc)
            except: pass

    except Exception:
        pass
    return out

# =================== SEC + Yahoo combo & Yahoo fallback ======================

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K) med Dec/Jan-fix,
    pris/valuta/namn + extra nyckeltal från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
    Om CIK saknas → hamta_yahoo_global_combo.
    """
    out = {}
    cik = _sec_cik_for(ticker)
    if not cik:
        # fallback
        yout = hamta_yahoo_global_combo(ticker)
        out.update(yout)
        return out

    facts, sc = _sec_companyfacts(cik)
    if sc != 200 or not isinstance(facts, dict):
        yout = hamta_yahoo_global_combo(ticker)
        out.update(yout)
        return out

    # Yahoo-basics + extra
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "Sektor", "Industri", "Årlig utdelning", "CAGR 5 år (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    extra = enrich_extra_metrics_from_yahoo(ticker, base_currency=out.get("Valuta","USD"))
    out.update(extra)

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
        out["Utestående aktier"] = shares_used / 1e6  # milj

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
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik. Hämtar extra nyckeltal.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","Sektor","Industri","Årlig utdelning","CAGR 5 år (%)"):
        if y.get(k): out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else 0.0
    except Exception:
        mcap = 0.0
    if mcap > 0:
        out["_marketCap_raw"] = mcap

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

    # Kvartalsintäkter → TTM (med Dec/Jan-fix)
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows or len(q_rows) < 4:
        # komplettera extra metrics ändå
        out.update(enrich_extra_metrics_from_yahoo(ticker, base_currency=px_ccy))
        return out
    ttm_list = _ttm_windows(q_rows, need=4)

    # Valutakonvertering om financialCurrency != prisvaluta
    fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        conv = _fx_rate_cached(fin_ccy, px_ccy) or 1.0
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # Market cap (nu)
    if (mcap <= 0) and shares > 0 and px > 0:
        mcap = shares * px
        out["_marketCap_raw"] = mcap

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

    # Extra nyckeltal
    out.update(enrich_extra_metrics_from_yahoo(ticker, base_currency=px_ccy))
    return out

# =================== Finnhub (valfritt, estimat) =============================

FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets. Hämtar annual revenue estimates:
    current FY + next FY (om finns). Return i miljoner (bolagsvaluta).
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

# =================== Auto-fetch pipeline (singel & batch) ====================

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares & P/S + extra metrics) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S) om saknas
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","Sektor","Industri","_debug_shares_source",
            "_marketCap_raw","EV/EBITDA (TTM)","P/E (TTM)","FCF (TTM)","FCF-yield (%)","Direktavkastning (%)","Total Debt","Total Cash"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                  "_marketCap_raw","EV/EBITDA (TTM)","P/E (TTM)","FCF (TTM)","FCF-yield (%)","Direktavkastning (%)","Sektor","Industri","Total Debt","Total Cash"]:
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
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    return vals, debug

# app.py — Del 4/? — Snapshots, formattering, pris-uppdatering, singel- & batch-auto

# ======================= Snapshot ===========================================

def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df.
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

# ======================= Hjälpfunktioner (visning & etiketter) ==============

def format_large(n: float) -> str:
    """Formatera stora tal (market cap etc) med enheter."""
    try:
        x = float(n)
    except Exception:
        return "-"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000_000:
        return f"{sign}{x/1_000_000_000_000:.2f} T"
    if x >= 1_000_000_000:
        return f"{sign}{x/1_000_000_000:.2f} B"
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.2f} M"
    if x >= 1_000:
        return f"{sign}{x/1_000:.2f} k"
    return f"{sign}{x:.0f}"

def risklabel_from_mcap(usd_mcap: float) -> str:
    """Grovt risklabel baserat på market cap i USD."""
    try:
        m = float(usd_mcap or 0.0)
    except Exception:
        m = 0.0
    if m >= 200_000_000_000:  # >= $200B
        return "Mega"
    if m >= 10_000_000_000:
        return "Large"
    if m >= 2_000_000_000:
        return "Mid"
    if m >= 300_000_000:
        return "Small"
    if m > 0:
        return "Micro"
    return ""

def safe_row_index(df: pd.DataFrame, ticker: str) -> Optional[int]:
    """Returnera första radindex för given ticker, annars None."""
    try:
        hits = df.index[df["Ticker"].astype(str).str.upper() == str(ticker).upper()]
        return int(hits[0]) if len(hits) else None
    except Exception:
        return None

# ======================= Pris-uppdatering (alla el. enskild) =================

def _yahoo_price_only(ticker: str) -> Optional[float]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        p = info.get("regularMarketPrice")
        if p is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                p = float(h["Close"].iloc[-1])
        return float(p) if p is not None else None
    except Exception:
        return None

def update_prices_for_all(df: pd.DataFrame) -> pd.DataFrame:
    """Uppdatera endast 'Aktuell kurs' för samtliga tickers (snabbare & lättare)."""
    if df.empty:
        st.info("Ingen data att uppdatera.")
        return df
    df = df.copy()
    tickers = list(df["Ticker"].astype(str))
    total = len(tickers)
    pbar = st.progress(0.0, text="Uppdaterar kurser …")
    status = st.empty()
    changed_map = {}

    for i, tkr in enumerate(tickers, start=1):
        status.write(f"Pris {i} av {total}: {tkr}")
        px = _yahoo_price_only(tkr)
        ridx = safe_row_index(df, tkr)
        if ridx is None:
            continue
        if px is not None and px > 0:
            before = float(df.at[ridx, "Aktuell kurs"] or 0.0)
            df.at[ridx, "Aktuell kurs"] = float(px)
            # Stämpla auto/TS även om samma pris – vi vill markera “kollad”
            _stamp_ts_for_field(df, ridx, "P/S")  # “något” TS; men bättre: separat TS_kurs? ej i schema → hoppa
            _note_auto_update(df, ridx, source="Pris (Yahoo)")
            if before != px:
                changed_map.setdefault(tkr, []).append("Aktuell kurs")
        pbar.progress(i/total)

    st.success("Kurser uppdaterade.")
    # Spara direkt för att minimera risk för dataförlust
    try:
        spara_data(df)
    except Exception as e:
        st.warning(f"Kunde inte spara efter prisuppdatering: {e}")
    return df

def update_price_for_single(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Uppdatera endast 'Aktuell kurs' för en ticker."""
    df = df.copy()
    ridx = safe_row_index(df, ticker)
    if ridx is None:
        st.warning(f"{ticker} hittades inte i tabellen.")
        return df
    px = _yahoo_price_only(ticker)
    if px is None:
        st.info("Kunde inte hämta pris.")
        return df
    df.at[ridx, "Aktuell kurs"] = float(px)
    _note_auto_update(df, ridx, source="Pris (Yahoo)")
    try:
        spara_data(df)
        st.success(f"Pris uppdaterat för {ticker}.")
    except Exception as e:
        st.warning(f"Kunde inte spara efter prisuppdatering: {e}")
    return df

# ======================= Full auto – singelrad ===============================

def full_auto_for_single(df: pd.DataFrame, user_rates: dict, ticker: str, always_stamp: bool = True) -> pd.DataFrame:
    """
    Kör auto_fetch_for_ticker för en ticker och skriv in till df, inkl. beräkningar.
    Stämplar TS även om samma värden (always_stamp=True).
    """
    df = df.copy()
    ridx = safe_row_index(df, ticker)
    if ridx is None:
        st.warning(f"{ticker} hittades inte i tabellen.")
        return df

    try:
        new_vals, debug = auto_fetch_for_ticker(ticker)
        changed = apply_auto_updates_to_row(
            df, ridx, new_vals, source="Auto (SEC/Yahoo→Finnhub→FMP)", changes_map={},
            always_stamp=always_stamp
        )
        # Risklabel + MarketCap (SEK) om möjligt
        if "_marketCap_raw" in new_vals:
            usd_mcap = float(new_vals["_marketCap_raw"])
            df.at[ridx, "Risklabel"] = risklabel_from_mcap(usd_mcap)
        # Uppdatera beräkningar
        df = uppdatera_berakningar(df, user_rates)
        # Spara direkt
        spara_data(df)
        if changed:
            st.success(f"Full auto-uppdatering klar för {ticker}.")
        else:
            st.info("Inga faktiska fält ändrades (TS stämplades ändå).")
    except Exception as e:
        st.error(f"Fel vid auto-uppdatering för {ticker}: {e}")
    return df

# ======================= Batch-motor (vald ordning & storlek) ================

BATCH_KEY_Q = "batch_queue"
BATCH_KEY_POS = "batch_pos"
BATCH_KEY_DONE = "batch_done"
BATCH_KEY_SKIPPED = "batch_skipped"

def _build_batch_queue(df: pd.DataFrame, mode: str, size: int) -> list:
    """
    Bygg en lista av tickers enligt valt läge:
      - "A–Ö (bolagsnamn)" → sortera alfabetiskt på Bolagsnamn
      - "Äldst TS först" → använd add_oldest_ts_col och sortera stigande
    Returnerar första 'size' tickers (eller alla om size=0).
    """
    if df.empty:
        return []
    mode = str(mode or "")
    if mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"], ascending=[True, True])
    else:
        work = df.sort_values(by=["Bolagsnamn","Ticker"])
    tickers = list(work["Ticker"].astype(str))
    if size and size > 0:
        tickers = tickers[:size]
    return tickers

def init_batch(df: pd.DataFrame, mode: str, size: int):
    st.session_state[BATCH_KEY_Q] = _build_batch_queue(df, mode, size)
    st.session_state[BATCH_KEY_POS] = 0
    st.session_state[BATCH_KEY_DONE] = []
    st.session_state[BATCH_KEY_SKIPPED] = []

def _next_in_batch() -> Optional[str]:
    q = st.session_state.get(BATCH_KEY_Q, [])
    pos = int(st.session_state.get(BATCH_KEY_POS, 0))
    if pos >= len(q):
        return None
    return q[pos]

def _advance_batch(step: int = 1):
    st.session_state[BATCH_KEY_POS] = int(st.session_state.get(BATCH_KEY_POS, 0)) + int(step)

def run_batch_step(df: pd.DataFrame, user_rates: dict, always_stamp: bool = True) -> pd.DataFrame:
    """
    Kör exakt EN ticker i kön. Återvänd uppdaterad df.
    """
    df = df.copy()
    tkr = _next_in_batch()
    q = st.session_state.get(BATCH_KEY_Q, [])
    if not tkr:
        st.info("Kön är slut.")
        return df

    i = int(st.session_state.get(BATCH_KEY_POS, 0)) + 1
    total = len(q)
    st.info(f"Kör {i} av {total}: {tkr}")

    ridx = safe_row_index(df, tkr)
    if ridx is None:
        st.warning(f"{tkr} hittades inte i tabellen – hoppar.")
        st.session_state[BATCH_KEY_SKIPPED].append(tkr)
        _advance_batch(1)
        return df

    new_vals, debug = auto_fetch_for_ticker(tkr)
    apply_auto_updates_to_row(
        df, ridx, new_vals, source="Batch auto (SEC/Yahoo→Finnhub→FMP)",
        changes_map={}, always_stamp=always_stamp
    )
    # Risklabel
    if "_marketCap_raw" in new_vals:
        usd_m = float(new_vals["_marketCap_raw"])
        df.at[ridx, "Risklabel"] = risklabel_from_mcap(usd_m)

    df = uppdatera_berakningar(df, user_rates)
    # Spara var 5:e samt när vi är klara
    try:
        if i % 5 == 0 or i == total:
            spara_data(df)
    except Exception as e:
        st.warning(f"Kunde inte spara efter batch-steg: {e}")

    st.session_state[BATCH_KEY_DONE].append(tkr)
    _advance_batch(1)
    st.success(f"Klar: {tkr} ({i}/{total})")
    return df

def batch_progress_ui():
    q = st.session_state.get(BATCH_KEY_Q, [])
    pos = int(st.session_state.get(BATCH_KEY_POS, 0))
    done = st.session_state.get(BATCH_KEY_DONE, [])
    skipped = st.session_state.get(BATCH_KEY_SKIPPED, [])
    total = len(q)
    cur = min(pos, total)
    st.progress(total and (cur/total), text=f"Batch-status: {cur} av {total}")
    st.write(f"✅ Klara: {len(done)} | ⏭️ Skippade: {len(skipped)}")
    if q:
        st.caption("Kö (första 25): " + ", ".join(q[:25]))

# app.py — Del 5/? — Kontrollvy & Analysvy (inkl. TS-etiketter och singeluppdatering)

# ======================= TS-etiketter (per fält) =============================

def _ts_badge_for_field(row: pd.Series, field: str) -> str:
    """
    Returnerar en kort etikett för ett spårat fält, t.ex.:
      "2025-09-24 • AUTO" / "2025-09-21 • MANUELL" / "–"
    Heuristik: om TS-datum == 'Senast auto-uppdaterad' -> AUTO,
               om TS-datum == 'Senast manuellt uppdaterad' -> MANUELL,
               annars bara datum.
    """
    ts_col = TS_FIELDS.get(field)
    if not ts_col or ts_col not in row:
        return "–"
    ts_val = str(row.get(ts_col) or "").strip()
    if not ts_val:
        return "–"
    auto_d = str(row.get("Senast auto-uppdaterad") or "").strip()
    man_d  = str(row.get("Senast manuellt uppdaterad") or "").strip()
    tag = ""
    if ts_val and auto_d and ts_val == auto_d:
        tag = " • AUTO"
    elif ts_val and man_d and ts_val == man_d:
        tag = " • MANUELL"
    return f"{ts_val}{tag}"

def _render_ts_summary(row: pd.Series):
    """Visa en kompakt lista med TS per spårat fält."""
    fields = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    lines = []
    for f in fields:
        if TS_FIELDS.get(f) in row.index:
            lines.append(f"- **{f}:** { _ts_badge_for_field(row, f) }")
    st.markdown("\n".join(lines) if lines else "_Inga spårade TS-fält hittades._")

# ======================= Kontroll-vy =========================================

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
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30, key="kontroll_older_days")
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Prognosbehov (Omsättning idag / nästa år)
    st.subheader("📅 Prognosbehov (Omsättning idag / nästa år)")
    rows = []
    for _, r in df.iterrows():
        ts_i = str(r.get("TS_Omsättning idag") or "")
        ts_n = str(r.get("TS_Omsättning nästa år") or "")
        miss_i = (float(r.get("Omsättning idag", 0.0) or 0.0) <= 0.0) or (not ts_i)
        miss_n = (float(r.get("Omsättning nästa år", 0.0) or 0.0) <= 0.0) or (not ts_n)
        if miss_i or miss_n:
            rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "TS Oms. idag": ts_i,
                "TS Oms. nästa år": ts_n,
            })
    prog = pd.DataFrame(rows)
    if prog.empty:
        st.info("Alla prognosfält verkar aktuella.")
    else:
        st.warning("Följande bolag saknar aktuell 'Omsättning idag' och/eller 'Omsättning nästa år':")
        st.dataframe(prog, use_container_width=True, hide_index=True)

    st.divider()

    # 4) Batch-status (om en kö finns)
    st.subheader("📦 Batch-status")
    batch_progress_ui()

# ======================= Analys-vy ===========================================

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    # Håll robust state
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(max(0, st.session_state.analys_idx), len(etiketter)-1)

    # Välj bolag
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #",
        min_value=0, max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx, step=1, key="analys_idx_num"
    )
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    col_nav1, col_nav2, col_nav3 = st.columns([1,2,1])
    with col_nav1:
        if st.button("⬅️ Föregående", key="analys_prev_btn"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_nav3:
        if st.button("➡️ Nästa", key="analys_next_btn"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    st.caption(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    # Snabbåtgärder (singel)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("💲 Uppdatera kurs", key="analys_update_price"):
            df2 = update_price_for_single(df, r["Ticker"])
            st.experimental_set_query_params(refresh=str(time.time()))
    with c2:
        if st.button("🤖 Full auto (endast denna)", key="analys_full_auto"):
            df2 = full_auto_for_single(df, user_rates, r["Ticker"], always_stamp=True)
            st.experimental_set_query_params(refresh=str(time.time()))

    # Tabell med huvuddata
    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Årlig utdelning",
        "_marketCap_raw","Sektor","Industri",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    ]
    cols = [c for c in cols if c in df.columns]
    show_row = pd.DataFrame([r[cols].to_dict()])
    # format MCAP
    if "_marketCap_raw" in show_row.columns:
        show_row["_marketCap_raw"] = show_row["_marketCap_raw"].apply(format_large)
        show_row = show_row.rename(columns={"_marketCap_raw":"MarketCap (USD, approx)"})
    st.dataframe(show_row, use_container_width=True, hide_index=True)

    with st.expander("🕒 Uppdateringsspårning (TS per fält)"):
        _render_ts_summary(r)

# app.py — Del 6/? — Investeringsförslag (med filter, scorer & extra nyckeltal)

# ======================= Scoring-hjälp =======================================

def _nz(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return float(d)

def _normalize_pos(x, lo, hi):
    """Skala där högre = bättre. Klipp inom [0,1]."""
    x = _nz(x)
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))

def _normalize_neg(x, lo, hi):
    """Skala där lägre = bättre (t.ex. EV/EBITDA)."""
    x = _nz(x)
    if hi <= lo:
        return 0.0
    # invertera: mapping x in [lo,hi] -> 1..0
    v = 1.0 - _normalize_pos(x, lo, hi)
    return float(max(0.0, min(1.0, v)))

def compute_growth_score(row: pd.Series) -> float:
    """
    Enkel tillväxt-score (0..100) baserat på:
      + Potential (%) mot vald riktkurs (vi fyller nedanför i runtime)
      + CAGR 5 år
      + EV/EBITDA (lägre bättre)
      + FCF-yield (högre bättre)
      + Risklabel (storleks-premium: mega/large > micro)
    """
    pot = _nz(row.get("Potential (%)"), 0.0)               # -100..+200 typiskt
    cagr = _nz(row.get("CAGR 5 år (%)"), 0.0)              # -inf..+200+ (klipps i beräkningar tidigare)
    ev_ebitda = _nz(row.get("EV/EBITDA (TTM)"), 0.0)       # 0..50+ (extremer)
    fcfy = _nz(row.get("FCF-yield (%)"), 0.0)              # -50..+30 typiskt

    # Normalisering / klipp
    pot_n = _normalize_pos(pot, -20.0, 100.0)
    cagr_n = _normalize_pos(cagr, 0.0, 40.0)
    ev_n   = _normalize_neg(ev_ebitda, 5.0, 25.0)          # <5 premium → bra; >25 dyrt → dåligt
    fcfy_n = _normalize_pos(fcfy, 0.0, 10.0)

    # Storleks-premium
    rl = str(row.get("Risklabel","") or "")
    size_boost = {
        "Mega": 1.00, "Large": 0.95, "Mid": 0.90, "Small": 0.85, "Micro": 0.80
    }.get(rl, 0.90)

    score = (0.35*pot_n + 0.25*cagr_n + 0.20*ev_n + 0.20*fcfy_n) * 100.0 * size_boost
    return round(float(score), 1)

def compute_dividend_score(row: pd.Series) -> float:
    """
    Utdelnings-score (0..100) för inkomstfokus:
      + Direktavkastning
      + FCF-yield (betalningsförmåga)
      + Skuldsättning (Total Debt / (Total Cash + 1)) lägre bättre
      + Storleks-premium
    """
    dy   = _nz(row.get("Direktavkastning (%)"), 0.0)
    fcfy = _nz(row.get("FCF-yield (%)"), 0.0)
    debt = _nz(row.get("Total Debt"), 0.0)
    cash = _nz(row.get("Total Cash"), 0.0)
    debt_cov = debt / (cash + 1.0)  # enkel proxy: lägre bättre

    dy_n   = _normalize_pos(dy, 2.0, 10.0)       # 2%..10%+
    fcfy_n = _normalize_pos(fcfy, 0.0, 10.0)
    debt_n = _normalize_neg(debt_cov, 0.2, 3.0)  # <0.2 väldigt bra; >3 dåligt

    rl = str(row.get("Risklabel","") or "")
    size_boost = {
        "Mega": 1.00, "Large": 0.95, "Mid": 0.90, "Small": 0.85, "Micro": 0.80
    }.get(rl, 0.90)

    score = (0.45*dy_n + 0.30*fcfy_n + 0.25*debt_n) * 100.0 * size_boost
    return round(float(score), 1)

# ======================= Investeringsförslag =================================

def _cap_bucket_from_label(label: str) -> tuple:
    """Översätt risklabel till (min,max) mcap USD för filter (inkl kanter)."""
    if label == "Mega":
        return (200_000_000_000, float("inf"))
    if label == "Large":
        return (10_000_000_000, 200_000_000_000)
    if label == "Mid":
        return (2_000_000_000, 10_000_000_000)
    if label == "Small":
        return (300_000_000, 2_000_000_000)
    if label == "Micro":
        return (0, 300_000_000)
    return (0, float("inf"))

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Parametrar & filter
    colf1, colf2, colf3 = st.columns([1,1,1])
    with colf1:
        kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=500.0, min_value=0.0, key="inv_capital")
    with colf2:
        segment = st.selectbox("Fokus", ["Tillväxt", "Utdelning"], index=0, key="inv_segment")
    with colf3:
        riktkurs_val = st.selectbox(
            "Riktkurs att jämföra mot",
            ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
            index=1, key="inv_riktkurs_sel"
        )

    colff1, colff2 = st.columns([1,1])
    with colff1:
        # Filtrera på storleksklass (risklabel)
        avail_labels = ["Mega","Large","Mid","Small","Micro"]
        chosen_labels = st.multiselect("Storleksklass", avail_labels, default=avail_labels, key="inv_cap_filter")
    with colff2:
        # Filtrera på sektor
        sectors = sorted([s for s in df.get("Sektor", pd.Series(dtype=str)).dropna().unique() if str(s).strip()])
        chosen_sectors = st.multiselect("Sektor", sectors, default=sectors if sectors else [], key="inv_sector_filter")

    subset = st.radio("Vilka bolag ska beaktas?", ["Alla bolag","Endast portfölj"], horizontal=True, key="inv_subset")
    sort_mode = st.radio("Sortering", ["Högst score","Störst potential","Närmast riktkurs"], horizontal=True, key="inv_sort")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Basurval
    base = df.copy()
    if subset == "Endast portfölj" and "Antal aktier" in base.columns:
        base = base[base["Antal aktier"] > 0].copy()

    # Kräver pris & riktkurs för potential
    needed_cols = [riktkurs_val, "Aktuell kurs", "Valuta"]
    for c in needed_cols:
        if c not in base.columns:
            base[c] = 0.0 if c != "Valuta" else ""

    # Market cap-krav: använd _marketCap_raw om finns för Risklabel & visning
    if "_marketCap_raw" not in base.columns:
        base["_marketCap_raw"] = 0.0

    # Beräkna potential
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar (saknar riktkurser eller pris).")
        return

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    # P/S-snitt – säkerställ kolumn
    if "P/S-snitt" not in base.columns:
        base["P/S-snitt"] = 0.0

    # Risklabel auto om saknas
    if "Risklabel" not in base.columns:
        base["Risklabel"] = base["_marketCap_raw"].apply(risklabel_from_mcap)

    # Filtrera på storlek
    if chosen_labels:
        base = base[base["Risklabel"].isin(chosen_labels)].copy()

    # Filtrera på sektor
    if chosen_sectors and "Sektor" in base.columns:
        base = base[base["Sektor"].isin(chosen_sectors)].copy()

    if base.empty:
        st.info("Inget bolag kvar efter filter.")
        return

    # Beräkna score enligt segment
    # (Vi lägger in default-kolumner om de saknas)
    for col in ["EV/EBITDA (TTM)","P/E (TTM)","FCF-yield (%)","Direktavkastning (%)","Total Debt","Total Cash","CAGR 5 år (%)"]:
        if col not in base.columns:
            base[col] = 0.0

    if segment == "Tillväxt":
        base["Score"] = base.apply(compute_growth_score, axis=1)
    else:
        base["Score"] = base.apply(compute_dividend_score, axis=1)

    # Sortering
    if sort_mode == "Högst score":
        base = base.sort_values(by=["Score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)
    elif sort_mode == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = (base["Aktuell kurs"] - base[riktkurs_val]).abs() / base[riktkurs_val].replace(0, np.nan)
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Navigering robust
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

    # Växelkurs & köpstorlek
    vx = hamta_valutakurs(rad.get("Valuta","USD"), user_rates)
    kurs_sek = _nz(rad.get("Aktuell kurs"), 0.0) * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    # Portföljandel före/efter (om portfölj finns)
    port = df[df.get("Antal aktier", 0) > 0].copy() if "Antal aktier" in df.columns else pd.DataFrame()
    nuv_innehav = 0.0; port_värde = 0.0
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())

    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Visning
    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad.get('Ticker','')})")
    # huvudrad
    lines = [
        f"- **Aktuell kurs:** {round(_nz(rad.get('Aktuell kurs')),2)} {rad.get('Valuta','')}",
        f"- **Utestående aktier (mn):** {round(_nz(rad.get('Utestående aktier')),2)}",
        f"- **MarketCap (USD):** {format_large(_nz(rad.get('_marketCap_raw', 0.0)))}",
        f"- **P/S (nu):** {round(_nz(rad.get('P/S')),2)}",
        f"- **P/S-snitt (Q1–Q4):** {round(_nz(rad.get('P/S-snitt')),2)}",
        f"- **Riktkurs (vald):** {round(_nz(rad.get(riktkurs_val)),2)} {rad.get('Valuta','')}",
        f"- **Uppsida (vald riktkurs):** {round(_nz(rad.get('Potential (%)')),2)} %",
        f"- **Score ({segment}):** {round(_nz(rad.get('Score')),1)} / 100",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
        f"- **Sektor:** {rad.get('Sektor','') or '-'} • **Storlek:** {rad.get('Risklabel','') or '-'}",
    ]
    st.markdown("\n".join(lines))

    with st.expander("📊 Nyckeltal & historik"):
        krows = []
        def addkv(k, v):
            krows.append([k, v])
        addkv("EV/EBITDA (TTM)", round(_nz(rad.get("EV/EBITDA (TTM)")),2))
        addkv("P/E (TTM)", round(_nz(rad.get("P/E (TTM)")),2))
        addkv("FCF (TTM)", format_large(_nz(rad.get("FCF (TTM)"))))
        addkv("FCF-yield (%)", round(_nz(rad.get("FCF-yield (%)")),2))
        addkv("Direktavkastning (%)", round(_nz(rad.get("Direktavkastning (%)")),2))
        addkv("Total Debt", format_large(_nz(rad.get("Total Debt"))))
        addkv("Total Cash", format_large(_nz(rad.get("Total Cash"))))
        addkv("CAGR 5 år (%)", round(_nz(rad.get("CAGR 5 år (%)")),2))
        addkv("P/S Q1", round(_nz(rad.get("P/S Q1")),2))
        addkv("P/S Q2", round(_nz(rad.get("P/S Q2")),2))
        addkv("P/S Q3", round(_nz(rad.get("P/S Q3")),2))
        addkv("P/S Q4", round(_nz(rad.get("P/S Q4")),2))
        # visa tabell
        kv_df = pd.DataFrame(krows, columns=["Nyckeltal","Värde"])
        st.dataframe(kv_df, use_container_width=True, hide_index=True)

    # Minimera krascher från state
    st.caption("Tip: Använd pilarna för att bläddra utan att scrolla mellan expanderare.")

# app.py — Del 7/? — Portfölj, Säljvakt, Lägg till/uppdatera, Batch-UI & MAIN

# ======================= Portfölj & Säljvakt =================================

def _pick_best_target_col(df: pd.DataFrame) -> str:
    for c in ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"]:
        if c in df.columns:
            return c
    return "Riktkurs om 1 år"

def _overvaluation_flags(row: pd.Series, target_col: str) -> list:
    """Returnera en lista med orsaker till ev. övervärdering."""
    reasons = []
    ps = _nz(row.get("P/S"))
    psn = _nz(row.get("P/S-snitt"))
    if ps > 0 and psn > 0 and ps > 1.25 * psn:
        reasons.append(f"P/S {ps:.1f} över {1.25*psn:.1f}")
    ev = _nz(row.get("EV/EBITDA (TTM)"))
    if ev > 25:
        reasons.append(f"EV/EBITDA {ev:.1f} (>25)")
    tgt = _nz(row.get(target_col))
    px = _nz(row.get("Aktuell kurs"))
    if tgt > 0 and px > 1.10 * tgt:
        reasons.append(f"Kurs {px:.2f} > 110% av {target_col} ({tgt:.2f})")
    return reasons

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    if df.empty or "Antal aktier" not in df.columns:
        st.info("Du äger inga aktier.")
        return

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs & värde
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())

    # Andel
    port["Andel (%)"] = np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)

    # Utdelning
    if "Årlig utdelning" not in port.columns:
        port["Årlig utdelning"] = 0.0
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    # Visning topp
    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK "
                f"(**≈/mån:** {round(tot_utd/12.0,2)} SEK)")

    # Visa fler nycklar om de finns
    extra_cols = []
    for c in ["Risklabel","Sektor","P/S","P/S-snitt","EV/EBITDA (TTM)","Direktavkastning (%)","_marketCap_raw"]:
        if c in port.columns:
            extra_cols.append(c)
    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
                 "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"] + extra_cols

    vis = port[show_cols].copy()
    if "_marketCap_raw" in vis.columns:
        vis["_marketCap_raw"] = vis["_marketCap_raw"].apply(format_large)
        vis = vis.rename(columns={"_marketCap_raw":"MarketCap (USD, approx)"})

    st.dataframe(vis.sort_values(by="Värde (SEK)", ascending=False),
                 use_container_width=True, hide_index=True)

    # Sektor-exponering
    if "Sektor" in port.columns:
        by_sec = (port.groupby("Sektor")["Värde (SEK)"].sum() / total_värde * 100.0).round(2)
        with st.expander("🏷️ Sektorexponering"):
            st.dataframe(by_sec.reset_index().rename(columns={"Värde (SEK)":"Andel (%)"}),
                         use_container_width=True, hide_index=True)

    # Säljvakt (beta)
    with st.expander("⚠️ Säljvakt (beta) — överväg trimning"):
        target_col = st.selectbox("Jämför mot riktkurs", 
                                  ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"],
                                  index=0, key="sv_target")
        trim_pct = st.slider("Föreslagen trim (%)", min_value=5, max_value=30, value=10, step=5, key="sv_trim")
        top_n = st.slider("Visa topp N att trimma", min_value=3, max_value=15, value=7, step=1, key="sv_topn")

        # Flagga övervärderade
        cand_rows = []
        for _, r in port.iterrows():
            reasons = _overvaluation_flags(r, target_col)
            if not reasons:
                continue
            # beräkna övervärde-signal
            ps, psn = _nz(r.get("P/S")), _nz(r.get("P/S-snitt"))
            ratio_ps = ps/psn if (ps>0 and psn>0) else 1.0
            tgt = _nz(r.get(target_col)); px = _nz(r.get("Aktuell kurs"))
            ratio_tgt = (px/tgt) if (tgt>0) else 1.0
            score_over = max(ratio_ps, ratio_tgt, _nz(r.get("EV/EBITDA (TTM)"))/25.0)
            # GAV (SEK) stöd om finns
            gav_sek = _nz(r.get("GAV (SEK)"), None) if "GAV (SEK)" in r else None
            gain_pct = None
            if gav_sek and gav_sek > 0:
                cur_value_per_share = r["Aktuell kurs"] * r["Växelkurs"]
                gain_pct = (cur_value_per_share / gav_sek - 1.0) * 100.0
            # föreslagen trim
            trim_shares = int(np.ceil(_nz(r.get("Antal aktier")) * (trim_pct/100.0)))
            cand_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Andel (%)": r.get("Andel (%)",0.0),
                "Övervärde-signal": round(score_over,2),
                "Skäl": "; ".join(reasons),
                "Antal": int(_nz(r.get("Antal aktier"),0)),
                "Föreslagen trim (antal)": trim_shares,
                "Gain % vs GAV": round(gain_pct,2) if gain_pct is not None else "-"
            })
        out = pd.DataFrame(cand_rows).sort_values(by=["Övervärde-signal","Andel (%)"], ascending=[False, False]).head(top_n)
        if out.empty:
            st.info("Inga tydliga övervärderade innehav enligt kriterierna.")
        else:
            st.warning("Följande innehav ser övervärderade ut enligt enkla kriterier:")
            st.dataframe(out, use_container_width=True, hide_index=True)

# ======================= Lägg till / uppdatera bolag =========================

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsordning
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst TS först"], index=0, key="edit_sort")
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    # Robust state
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(val_lista)-1))

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1), key="edit_select")
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    bef = df[df["Ticker"] == namn_map.get(valt_label, "__NA__")]
    bef = bef.iloc[0] if not bef.empty else pd.Series({}, dtype=object)

    # Form
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav = st.number_input("GAV (SEK) per aktie", value=float(bef.get("GAV (SEK)",0.0)) if "GAV (SEK)" in df.columns and not bef.empty else 0.0, key="edit_gav")
            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Vid spara uppdateras också automatiskt:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

        colb1, colb2, colb3, colb4 = st.columns([1,1,1,1])
        with colb1:
            spar = st.form_submit_button("💾 Spara")
        with colb2:
            upd_px = st.form_submit_button("💲 Uppdatera kurs")
        with colb3:
            upd_full = st.form_submit_button("🤖 Full auto (denna)")
        with colb4:
            new_empty = st.form_submit_button("🆕 Lägg till ny (tom)")

    # Hantera formaktioner
    if new_empty:
        if ticker:
            if ticker.upper() in df["Ticker"].astype(str).str.upper().unique():
                st.warning("Ticker finns redan.")
            else:
                tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta",
                                            "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]
                           and not str(c).startswith("TS_")
                           else "") for c in FINAL_COLS}
                tom.update({"Ticker": ticker})
                df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
                try:
                    spara_data(df)
                except Exception as e:
                    st.warning(f"Kunde inte spara: {e}")
                st.success(f"La till {ticker}.")
        else:
            st.info("Ange en ticker för att skapa ny rad.")

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }
        if "GAV (SEK)" in df.columns:
            ny["GAV (SEK)"] = float(gav or 0.0)

        # Skriv in rad
        if not bef.empty:
            ridx = safe_row_index(df, ticker)
            for k,v in ny.items():
                df.at[ridx, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta",
                                        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]
                       and not str(c).startswith("TS_")
                       else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Stämpla manuell uppdatering + TS för ändrade manuella fält
        ridx = safe_row_index(df, ticker)
        _note_manual_update(df, ridx)
        for f in MANUELL_FALT_FOR_DATUM:
            _stamp_ts_for_field(df, ridx, f)

        # Basfält från Yahoo
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if _nz(data.get("Aktuell kurs"),0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None:
            df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:
            df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        try:
            spara_data(df)
            st.success("Sparat.")
        except Exception as e:
            st.warning(f"Kunde inte spara: {e}")

    if upd_px and ticker:
        df = update_price_for_single(df, ticker)

    if upd_full and ticker:
        df = full_auto_for_single(df, user_rates, ticker, always_stamp=True)

    # Snabb batch-kontroller
    st.markdown("---")
    st.subheader("📦 Batch-uppdatering")
    colq1, colq2, colq3 = st.columns([2,1,1])
    with colq1:
        mode = st.selectbox("Urval för kö", ["A–Ö (bolagsnamn)","Äldst TS först"], key="batch_mode")
        size = st.number_input("Antal att lägga i kö (0 = alla)", min_value=0, max_value=5000, value=10, step=5, key="batch_size")
    with colq2:
        if st.button("Bygg kö"):
            init_batch(df, mode, int(size))
            st.success(f"Kö skapad med {len(st.session_state.get(BATCH_KEY_Q,[]))} tickers.")
    with colq3:
        if st.button("Töm kö"):
            for k in [BATCH_KEY_Q,BATCH_KEY_POS,BATCH_KEY_DONE,BATCH_KEY_SKIPPED]:
                st.session_state.pop(k, None)
            st.info("Kö rensad.")

    colr1, colr2, colr3 = st.columns([1,1,1])
    with colr1:
        if st.button("▶️ Kör nästa"):
            df = run_batch_step(df, user_rates, always_stamp=True)
    with colr2:
        if st.button("⏩ Kör 5 steg"):
            for _ in range(5):
                df = run_batch_step(df, user_rates, always_stamp=True)
    with colr3:
        if st.button("⏭️ Kör 10 steg"):
            for _ in range(10):
                df = run_batch_step(df, user_rates, always_stamp=True)

    batch_progress_ui()
    return df

# ======================= MAIN =================================================

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --------- Initiera session_state (valutakurser) före widgets -------------
    # Läs sparade kurser (utan att skriva widgets ännu)
    saved_rates = las_sparade_valutakurser()
    if "rate_usd" not in st.session_state:
        st.session_state.rate_usd = float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"]))
    if "rate_nok" not in st.session_state:
        st.session_state.rate_nok = float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
    if "rate_cad" not in st.session_state:
        st.session_state.rate_cad = float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
    if "rate_eur" not in st.session_state:
        st.session_state.rate_eur = float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

    # Om en tidigare knapptryckning bad om auto-rates, applicera dem här och rensa flaggan
    if st.session_state.get("pending_auto_rates"):
        sr = st.session_state.pop("pending_auto_rates")
        st.session_state.rate_usd = float(sr.get("USD", st.session_state.rate_usd))
        st.session_state.rate_nok = float(sr.get("NOK", st.session_state.rate_nok))
        st.session_state.rate_cad = float(sr.get("CAD", st.session_state.rate_cad))
        st.session_state.rate_eur = float(sr.get("EUR", st.session_state.rate_eur))

    # --------- Sidopanel: valutakurser ---------------------------------------
    st.sidebar.header("💱 Valutakurser → SEK")

    colr1, colr2 = st.sidebar.columns(2)
    with colr1:
        auto_rates_btn = st.button("🌐 Hämta automatiskt")
    with colr2:
        save_rates_btn = st.button("💾 Spara kurser")

    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd), step=0.01, format="%.4f", key="rate_usd")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok), step=0.01, format="%.4f", key="rate_nok")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad), step=0.01, format="%.4f", key="rate_cad")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur), step=0.01, format="%.4f", key="rate_eur")

    if auto_rates_btn:
        auto_rates, misses, provider = hamta_valutakurser_auto()
        # Lagra i state och kör om, så slipper vi streamlit-key-krock
        st.session_state["pending_auto_rates"] = auto_rates
        st.experimental_rerun()

    if save_rates_btn:
        user_rates_to_save = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}
        try:
            spara_valutakurser(user_rates_to_save)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
        except Exception as e:
            st.sidebar.warning(f"Kunde inte spara kurser: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.experimental_rerun()

    # --------- Läs data -------------------------------------------------------
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    # Säkerställ schema/typer men spara **inte** automatiskt om tomt
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # User rates (för beräkningar)
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    # --------- Auto-uppdatering (snabba kontroller) ---------------------------
    st.sidebar.subheader("🛠️ Uppdatering")
    if st.sidebar.button("💲 Uppdatera **priser** (alla)"):
        df = update_prices_for_all(df)

    st.sidebar.subheader("📦 Batch")
    with st.sidebar.expander("Kö-kontroller"):
        mode = st.selectbox("Urval", ["A–Ö (bolagsnamn)","Äldst TS först"], key="sb_batch_mode")
        size = st.number_input("Antal i kö (0 = alla)", min_value=0, max_value=5000, value=20, step=5, key="sb_batch_size")
        if st.button("Bygg kö", key="sb_build"):
            init_batch(df, mode, int(size))
            st.success(f"Kö skapad ({len(st.session_state.get(BATCH_KEY_Q,[]))} st).")
        colsb1, colsb2, colsb3 = st.columns(3)
        with colsb1:
            if st.button("▶️ Nästa", key="sb_next"):
                df = run_batch_step(df, user_rates, always_stamp=True)
        with colsb2:
            if st.button("⏩ +5", key="sb_next5"):
                for _ in range(5):
                    df = run_batch_step(df, user_rates, always_stamp=True)
        with colsb3:
            if st.button("⏭️ +10", key="sb_next10"):
                for _ in range(10):
                    df = run_batch_step(df, user_rates, always_stamp=True)
        batch_progress_ui()

    # --------- Meny -----------------------------------------------------------
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
