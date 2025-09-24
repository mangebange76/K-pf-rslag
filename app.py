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
    Säkerhet: blockerar TOM skrivning om inte 'destructive_ok' är aktiv i session_state.
    """
    if df is None or df.empty:
        if not st.session_state.get("destructive_ok", False):
            st.error("⛔ Blockerat: Försök att skriva en TOM tabell till Google Sheets. Aktivera 'Tillåt tom skrivning' om du verkligen vill rensa bladet.")
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

# Slutlig kolumnlista i databasen (inkl. nya nycklar & MCap-historik)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    # Market cap
    "MCap (nu)", "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
    "MCap Datum Q1", "MCap Datum Q2", "MCap Datum Q3", "MCap Datum Q4",
    # Nyckeltal (valfria)
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
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","mcap","debt","marginal","kassa"]):
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
        "Utestående aktier", "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "MCap (nu)", "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
        "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Finansiell valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
              "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]:
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

# app.py — Del 2/7
# --- Yahoo-hjälpare, beräkningar & auto-write med TS -------------------------

def _yfi_get(tkr: yf.Ticker, *keys):
    """Säker hämtning ur yfinance.info (fallbacks stöds genom att ange flera keys)."""
    try:
        info = tkr.info or {}
        for k in keys:
            if k in info and info[k] is not None:
                return info[k]
    except Exception:
        pass
    return None

def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """CAGR baserat på 'Total Revenue' (årsbasis), enkel approx från yfinance."""
    try:
        # försök income_stmt (nyare) → financials (äldre)
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR, P/S (Yahoo) & MCap (nu).
    P/S (Yahoo) kommer från YF:s TTM-nyckeltal om tillgängligt.
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
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
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

        # Utdelning årlig (rate)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # CAGR 5 år (approx)
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

        # Market cap & Yahoo P/S (TTM)
        mcap = info.get("marketCap", None)
        try:
            if mcap is not None:
                out["MCap (nu)"] = float(mcap)
        except Exception:
            pass

        # YF varierar i key-namn över tid; prova flera
        for key in ("priceToSalesTrailing12Months", "priceToSalesTTM", "priceToSalesRatioTTM"):
            v = info.get(key, None)
            try:
                if v is not None and float(v) > 0:
                    out["P/S (Yahoo)"] = float(v)
                    break
            except Exception:
                continue

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
        aktier_ut_m = float(rad.get("Utestående aktier", 0.0))  # i miljoner
        if aktier_ut_m > 0 and ps_snitt > 0:
            aktier_ut = aktier_ut_m * 1e6
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt * 1e6) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt * 1e6) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt * 1e6) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt * 1e6) / aktier_ut, 2)
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
    Uppdaterar TS_ för spårade fält (alltid om force_ts=True, annars vid ändring),
    sätter 'Senast auto-uppdaterad' + källa.
    Returnerar True om något fält faktiskt ändrades.
    """
    changed_fields = []
    any_seen_spared_field = False

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue

        # Bestäm om vi "tillåter" skrivning av nollor för vissa fält
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            # Kritiskt fält kräver > 0
            critical = f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]
            write_ok = (fv > 0) or (not critical and fv >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            write_ok = v is not None

        if not write_ok and not force_ts:
            continue  # varken skriv eller TS-stämpla

        old = df.at[row_idx, f] if f in df.columns else None
        is_changed = (str(old) != str(v)) if write_ok else False

        if write_ok and is_changed:
            df.at[row_idx, f] = v
            changed_fields.append(f)

        # TS-hantering för spårade fält
        if f in TS_FIELDS:
            any_seen_spared_field = True
            if is_changed or force_ts:
                _stamp_ts_for_field(df, row_idx, f)

    # Auto-stämpel & källa om något ändrades ELLER vi tvingar TS (t.ex. batchvågor/pris-only)
    if changed_fields or (force_ts and (any_seen_spared_field or new_vals)):
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)

    return len(changed_fields) > 0

# app.py — Del 3/7
# --- SEC (US/IFRS) + Yahoo fallback: shares, kvartalsintäkter, P/S Q1–Q4 -----

# =============== SEC (US + FPI/IFRS) =========================================

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
                        continue  # inte ett 3-månadersperiod
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # dedupe per end-datum
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

def _sec_annual_revenue_dated_with_unit(facts: dict, max_years: int = 6):
    """
    Hämta årsintäkter (10-K/20-F/40-F) för ev. Q4-fyllnad.
    Returnerar (rows, unit) med rows=[(end_date, annual_value), ...] nyast→äldst.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-K", "10-K/A")}),
        ("ifrs-full", {"forms": ("20-F", "40-F", "20-F/A", "40-F/A", "10-K", "10-K/A")}),
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
                    if dur is None or dur < 350 or dur > 380:
                        continue  # ungefär ett år
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
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_years]
                if rows:
                    return rows, unit_code
    return [], None

def _fill_q4_if_missing(q_quarters: list, annuals: list) -> list:
    """
    Om vi bara får 3 senaste kvartal från SEC (pga 10-K/20-F-övergång), försök räkna Q4:
    Q4 = Annual(FY) - sum(Q1..Q3) där FY_end matchar kvartalens FY.
    q_quarters: [(end_date, q_rev), ...] nyast→äldst
    annuals:    [(fy_end, fy_rev), ...]   nyast→äldst
    Returnerar ny lista (kan vara oförändrad) nyast→äldst med ev. Q4 tillagd.
    """
    if len(q_quarters) >= 4 or len(q_quarters) < 2 or not annuals:
        return q_quarters

    # Ta ut de senaste 3 kvartalen och se om vi hittar ett FY som slutar på samma senaste end-året
    q = sorted(q_quarters, key=lambda t: t[0], reverse=True)
    latest_end = q[0][0]
    fy_candidates = [a for a in annuals if a[0].year == latest_end.year or a[0] >= latest_end]
    if not fy_candidates:
        # ta nyaste annual ändå
        fy_candidates = [annuals[0]]

    fy_end, fy_rev = fy_candidates[0]
    # summera tre äldsta av de 3 senaste? (dvs alla tre)
    # antag att q representerar Q1..Q3 i ett FY (ordningen nyast→äldst)
    three_sum = sum(v for (_, v) in q[:3])
    q4_val = float(fy_rev) - float(three_sum)
    # sanity
    if q4_val > 0 and q4_val < fy_rev * 0.9:  # grov sanity
        # placera Q4 på FY_end
        extended = q.copy()
        extended.insert(3, (fy_end, q4_val))  # nu finns 4 poster
        return extended
    return q_quarters

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
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    fyll Q4 via 10-K/20-F/40-F om nödvändigt, pris/valuta/namn från Yahoo.
    P/S (TTM) nu + P/S Q1–Q4 historik, samt MCap-historik Q1–Q4.
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "MCap (nu)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or y.get("Valuta") or "USD").upper()

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

    # SEC kvartalsintäkter + unit → ev. Q4-fyllnad → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    if not q_rows or not rev_unit:
        return out

    # Q4-fyllare om SEC missar ett kvartal vid FY
    annual_rows, annual_unit = _sec_annual_revenue_dated_with_unit(facts, max_years=6)
    if annual_rows and annual_unit == rev_unit:
        q_rows = _fill_q4_if_missing(q_rows, annual_rows)

    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    q_rows_px = [(d, v * conv) for (d, v) in q_rows]

    # Bygg TTM-listor (upp till 4 fönster)
    ttm_list = _ttm_windows(q_rows_px, need=4)  # [(end, ttm), ...] nyast→äldst

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik + MCap Q1–Q4
    if shares_used > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCap Q{idx}"] = float(mcap_hist)
                    out[f"MCap Datum Q{idx}"] = d_end.strftime("%Y-%m-%d")

    return out

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik & MCap Q1–Q4.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/mcap
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","MCap (nu)"):
        if y.get(k): out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()
    mcap = float(out.get("MCap (nu)", 0.0) or 0.0)

    info = _yfi_info_dict(t)

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
        out["MCap (nu)"] = float(mcap)

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
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares * p
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCap Q{idx}"] = float(mcap_hist)
                    out[f"MCap Datum Q{idx}"] = d_end.strftime("%Y-%m-%d")

    return out

# app.py — Del 4/7
# --- FMP, Finnhub, auto-pipeline, pris-only, vågkörning & debug --------------

# =============== FMP (Financial Modeling Prep) ===============================

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

    delays = [0.0, 1.0, 2.0]
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

# =============== Finnhub (valfritt, estimat) =================================
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets. Hämtar annual revenue estimates:
    current FY + next FY (om finns). Returnerar värden i miljoner.
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
      1) SEC + Yahoo (implied shares + P/S/Q-historik + MCap-historik) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S/price/MCap) om saknas
      4) Yahoo-nyckeltal (Debt/Equity, marginaler, kassa)
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","MCap (nu)",
            "MCap Q1","MCap Q2","MCap Q3","MCap Q4",
            "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
            "_debug_shares_source"
        ]}
        for k in [
            "Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
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
        if ("Omsättning idag" not in vals) or ("Omsättning nästa år" not in vals):
            fh = hamta_finnhub_revenue_estimates(ticker)
            debug["finnhub"] = fh
            for k in ["Omsättning idag","Omsättning nästa år"]:
                v = fh.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        debug["finnhub_err"] = str(e)

    # 3) FMP light P/S/price/MCap om saknas
    try:
        need_any = ("P/S" not in vals) or ("Aktuell kurs" not in vals) or ("MCap (nu)" not in vals)
        if need_any:
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier"), "MCap (nu)": fmpl.get("MCap (nu)")}
            for k in ["P/S","Aktuell kurs","MCap (nu)","Utestående aktier"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals.setdefault(k, v)
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # 4) Yahoo-nyckeltal + Yahoo P/S
    try:
        yb = hamta_yahoo_fält(ticker)  # tar även med "P/S (Yahoo)" och "MCap (nu)"
        for k in ["P/S (Yahoo)","MCap (nu)","Aktuell kurs","Valuta","Bolagsnamn","Årlig utdelning","CAGR 5 år (%)"]:
            v = yb.get(k)
            if v not in (None, "", 0, 0.0):
                vals.setdefault(k, v)
        nyck = hamta_yahoo_nyckeltal(ticker)
        for k in ["Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa","Finansiell valuta"]:
            v = nyck.get(k)
            if v not in (None, "", 0, 0.0):
                vals.setdefault(k, v)
        debug["yf_basics"] = yb
        debug["yf_metrics"] = nyck
    except Exception as e:
        debug["yf_err"] = str(e)

    return vals, debug

# =============== Pris-only & Full update helpers =============================

def update_price_for_ticker(df: pd.DataFrame, ticker: str):
    """
    Uppdatera endast pris/valuta/MCap (nu) för en ticker.
    Force TS (auto) även om inga TS-fält ändras, så du får stämpeln.
    """
    t = yf.Ticker(ticker)
    out = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    px = info.get("regularMarketPrice")
    if px is None:
        try:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        except Exception:
            px = None
    if px is not None:
        out["Aktuell kurs"] = float(px)
    ccy = info.get("currency")
    if ccy:
        out["Valuta"] = str(ccy).upper()
    mcap = info.get("marketCap")
    try:
        if mcap is not None and float(mcap) > 0:
            out["MCap (nu)"] = float(mcap)
    except Exception:
        pass

    idxs = np.where(df["Ticker"].astype(str).str.upper() == ticker.upper())[0]
    if len(idxs) == 0:
        return df, False, "Ticker hittades inte i tabellen."
    ridx = int(idxs[0])

    changed = apply_auto_updates_to_row(df, ridx, out, source="Pris-only (Yahoo)", changes_map={}, force_ts=True)
    return df, changed, "OK"

def update_full_for_ticker(df: pd.DataFrame, ticker: str, user_rates: dict, *, force_ts: bool = False):
    """
    Kör full auto-pipeline för 1 ticker och skriv endast meningsfulla fält.
    """
    idxs = np.where(df["Ticker"].astype(str).str.upper() == ticker.upper())[0]
    if len(idxs) == 0:
        return df, False, {"err": "Ticker hittades inte i tabellen."}
    ridx = int(idxs[0])

    vals, debug = auto_fetch_for_ticker(ticker)
    changed = apply_auto_updates_to_row(df, ridx, vals, source="Auto (SEC/Yahoo→Finnhub→FMP→Yahoo nyckeltal)", changes_map={}, force_ts=force_ts)

    # Räkna om beräkningar och lämna ev. oförändrat om inget ändrades (men vi kan ändå vilja spara).
    df = uppdatera_berakningar(df, user_rates)
    return df, changed, debug

# =============== Batch: priser & full auto (alla) ============================

def update_prices_all(df: pd.DataFrame):
    """
    Pris-only uppdatering för alla tickers i DF.
    """
    changed_tickers = []
    misses = []
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()

    total = len(df)
    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1))
            continue
        status.write(f"Uppdaterar pris {i+1}/{total}: {tkr}")
        try:
            df, changed, msg = update_price_for_ticker(df, tkr)
            if changed:
                changed_tickers.append(tkr)
            if "OK" not in msg:
                misses.append(tkr)
        except Exception as e:
            misses.append(tkr)
        progress.progress((i+1)/max(total,1))

    st.sidebar.success("Prisuppdatering klar.")
    return df, {"priced": changed_tickers, "miss": misses}

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, *, force_ts: bool = False):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla nya värden.
    Stämplar TS_ per fält där värde uppdateras (eller alltid om force_ts=True),
    samt 'Senast auto-uppdaterad' + källa.
    """
    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    progress = st.sidebar.progress(0)
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
            new_df, changed, debug = update_full_for_ticker(df, tkr, user_rates, force_ts=force_ts)
            df = new_df  # ersätt referens
            if changed:
                log["changed"].setdefault(tkr, []).append("auto")
            else:
                log["misses"][tkr] = ["(inga nya fält)"]
            any_changed = any_changed or changed
            if i < 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(total,1))

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    if any_changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Ändringar sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# =============== Vågvis uppdatering (kösystem) ===============================

def _queue_build(df: pd.DataFrame, mode: str = "oldest") -> list[str]:
    tickers = []
    work = df.copy()
    if mode == "alphabetic":
        work = work.sort_values(by=["Bolagsnamn","Ticker"])
    else:
        # oldest: använd äldsta TS bland spårade fält
        work = add_oldest_ts_col(work)
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    for _, r in work.iterrows():
        t = str(r.get("Ticker","")).strip().upper()
        if t:
            tickers.append(t)
    return tickers

def start_wave(df: pd.DataFrame, mode: str = "oldest"):
    st.session_state["wave_queue"] = _queue_build(df, mode=mode)
    st.session_state["wave_done"] = []
    st.session_state["wave_changed"] = []
    st.session_state["wave_miss"] = []
    st.session_state["wave_started_at"] = _ts_str()
    st.session_state["wave_mode"] = mode

def run_wave_step(df: pd.DataFrame, user_rates: dict, *, batch_size: int = 10, make_snapshot: bool = False, force_ts: bool = True):
    q = st.session_state.get("wave_queue", []) or []
    done = st.session_state.get("wave_done", []) or []
    changed_list = st.session_state.get("wave_changed", []) or []
    miss_list = st.session_state.get("wave_miss", []) or []

    if not q:
        return df, {"processed": 0, "remaining": 0}

    take = q[:batch_size]
    rest = q[batch_size:]

    for tkr in take:
        try:
            df, changed, _dbg = update_full_for_ticker(df, tkr, user_rates, force_ts=force_ts)
            done.append(tkr)
            if changed:
                changed_list.append(tkr)
            else:
                miss_list.append(tkr)
        except Exception:
            done.append(tkr)
            miss_list.append(tkr)

    # skriv efter varje batch
    spara_data(df, do_snapshot=make_snapshot)

    # uppdatera kö-state
    st.session_state["wave_queue"] = rest
    st.session_state["wave_done"] = done
    st.session_state["wave_changed"] = changed_list
    st.session_state["wave_miss"] = miss_list

    return df, {"processed": len(take), "remaining": len(rest)}

# =============== Debug ========================================================

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

# app.py — Del 5/7
# --- Snapshots, hjälpfunktioner (mcap-format, risklabel), kontrollvy ----------

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

# --------- formatterare & riskklass ------------------------------------------

def _pretty_mcap(v: float, ccy: str = "") -> str:
    """
    Snygg formattering av market cap (t.ex. 1.23 tn, 456.7 mdr, 12.3 mkr).
    Enheten följer "prisvalutan" (ccy).
    """
    try:
        x = float(v or 0.0)
    except Exception:
        x = 0.0
    sign = "-" if x < 0 else ""
    x = abs(x)

    # Skala
    if x >= 1e12:
        num, unit = x/1e12, "tn"
    elif x >= 1e9:
        num, unit = x/1e9, "mdr"
    elif x >= 1e6:
        num, unit = x/1e6, "m"
    elif x >= 1e3:
        num, unit = x/1e3, "t"
    else:
        num, unit = x, ""

    if unit:
        return f"{sign}{num:,.2f} {unit} {ccy}".replace(",", " ")
    return f"{sign}{num:,.0f} {ccy}".strip()

def _fmt_pct(v: float) -> str:
    try:
        return f"{float(v):.2f} %"
    except Exception:
        return "0.00 %"

def _fmt_num(v: float) -> str:
    try:
        return f"{float(v):,.2f}".replace(",", " ")
    except Exception:
        return "0.00"

def risk_label_for_mcap(mcap: float) -> str:
    """
    Grov risklabel baserat på market cap (i samma valuta som priset):
      < 300M  → Microcap
      <   2B  → Small
      <  10B  → Mid
      < 200B  → Large
      ≥ 200B  → Mega
    """
    try:
        x = float(mcap or 0.0)
    except Exception:
        x = 0.0
    if x < 3e8:      return "Microcap"
    if x < 2e9:      return "Smallcap"
    if x < 1e10:     return "Midcap"
    if x < 2e11:     return "Largecap"
    return "Megacap"

def compute_current_mcap(row: pd.Series) -> float:
    """
    Beräkna nuvarande MCap utifrån 'MCap (nu)' om tillgänglig,
    annars pris * utestående (i SEK? nej – i prisvaluta: price * shares_styck).
    """
    m = float(row.get("MCap (nu)", 0.0) or 0.0)
    if m > 0:
        return m
    price = float(row.get("Aktuell kurs", 0.0) or 0.0)
    shares_m = float(row.get("Utestående aktier", 0.0) or 0.0)  # i miljoner
    if price > 0 and shares_m > 0:
        return price * shares_m * 1e6
    return 0.0

# --------- kontrollvyer ------------------------------------------------------

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

    # 3) Senaste körloggar
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log")
    if not log:
        st.info("Ingen full auto-körning loggad i denna session.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ändringar** (ticker → 'auto')")
            if log.get("changed"):
                st.json(log["changed"])
            else:
                st.write("–")
        with col2:
            st.markdown("**Missar** (ticker → orsak)")
            if log.get("misses"):
                st.json(log["misses"])
            else:
                st.write("–")
        st.markdown("**Debug (första 20)**")
        st.json(log.get("debug_first_20", []))

    # 4) Våg-köstatus
    st.divider()
    st.subheader("🌊 Våg-köstatus")
    q = st.session_state.get("wave_queue", []) or []
    d = st.session_state.get("wave_done", []) or []
    ch = st.session_state.get("wave_changed", []) or []
    mi = st.session_state.get("wave_miss", []) or []
    if q or d or ch or mi:
        st.markdown(f"- **Kvar i kö:** {len(q)}  \n- **Klara:** {len(d)}  \n- **Ändrade:** {len(ch)}  \n- **Miss:** {len(mi)}")
        if d:
            st.write("Senaste klara tickers:")
            st.code(", ".join(d[-20:]), language="text")
    else:
        st.info("Ingen aktiv våg just nu.")

# app.py — Del 6/7
# --- Analys, Portfölj & Investeringsförslag + Lägg till/uppdatera ------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    st.session_state.analys_idx = st.number_input(
        "Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
        value=int(st.session_state.analys_idx), step=1
    )
    st.selectbox("Eller välj i lista", etiketter, index=int(st.session_state.analys_idx if etiketter else 0), key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, int(st.session_state.analys_idx)-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, int(st.session_state.analys_idx)+1)
    st.write(f"Post {int(st.session_state.analys_idx)+1}/{len(etiketter)}")

    r = vis_df.iloc[int(st.session_state.analys_idx)]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    # Snabb statuschips (manuell/auto + källa)
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.markdown(f"**Senast manuellt:** {r.get('Senast manuellt uppdaterad','') or '—'}")
    with c2:
        st.markdown(f"**Senast auto:** {r.get('Senast auto-uppdaterad','') or '—'}")
    with c3:
        src = r.get("Senast uppdaterad källa","") or "—"
        st.markdown(f"**Källa:** {src}")

    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
        "P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa","Finansiell valuta",
        "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
        "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
        "TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

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

    # Risk/kap-filter
    cap_filter = st.selectbox(
        "Filtrera på risk/kapitalisering",
        ["Alla","Microcap","Smallcap","Midcap","Largecap","Megacap"],
        index=0
    )

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Beräkna nuvarande MCap & risklabel
    base["MCap_calc"] = base.apply(compute_current_mcap, axis=1)
    base["Risklabel"] = base["MCap_calc"].apply(risk_label_for_mcap)

    # Filtrera på cap
    if cap_filter != "Alla":
        base = base[base["Risklabel"] == cap_filter].copy()
        if base.empty:
            st.warning("Filtret gav inga träffar.")
            return

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(int(st.session_state.forslags_index), len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, int(st.session_state.forslags_index) - 1)
    with col_mid:
        st.write(f"Förslag {int(st.session_state.forslags_index)+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base)-1, int(st.session_state.forslags_index) + 1)

    rad = base.iloc[int(st.session_state.forslags_index)]

    # Portföljandelar & köp
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
        rmatch = port[port["Ticker"] == rad["Ticker"]]
        if not rmatch.empty:
            nuv_innehav = float(rmatch["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    # Snabb sammanfattning
    lines = [
        f"- **Riskklass:** {rad.get('Risklabel','')}",
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

    # Detaljer för expander
    det_lines = []
    # Lägg nuvarande marketcap överst
    det_lines.append(f"- **Nuvarande börsvärde:** {_pretty_mcap(rad.get('MCap_calc', 0.0), rad.get('Valuta',''))}")

    # P/S info
    det_lines.append(f"- **P/S (nu, beräknat):** {_fmt_num(rad.get('P/S', 0.0))}")
    det_lines.append(f"- **P/S (Yahoo):** {_fmt_num(rad.get('P/S (Yahoo)', 0.0))}")
    det_lines.append(f"- **P/S-snitt (4 kvartal):** {_fmt_num(rad.get('P/S-snitt', 0.0))}")

    # MCap-historik Q1–Q4
    for q in (1,2,3,4):
        mc = rad.get(f"MCap Q{q}", 0.0)
        md = rad.get(f"MCap Datum Q{q}", "")
        if mc:
            det_lines.append(f"- **MCap Q{q} ({md}):** {_pretty_mcap(mc, rad.get('Valuta',''))}")

    # Nyckeltal
    det_lines.append(f"- **Debt/Equity:** {_fmt_num(rad.get('Debt/Equity', 0.0))}")
    det_lines.append(f"- **Bruttomarginal:** {_fmt_pct(rad.get('Bruttomarginal (%)', 0.0))}")
    det_lines.append(f"- **Nettomarginal:** {_fmt_pct(rad.get('Nettomarginal (%)', 0.0))}")
    # Kassa + valuta
    cash_ccy = rad.get("Finansiell valuta") or rad.get("Valuta","")
    det_lines.append(f"- **Kassa:** {_pretty_mcap(rad.get('Kassa', 0.0), cash_ccy)}")

    with st.expander("📊 Visa detaljer (P/S, MCap, nyckeltal)", expanded=False):
        st.markdown("\n".join(det_lines))

def _ts_badge_grid(row: pd.Series) -> None:
    """Liten ruta som visar TS-datum per spårat fält (för 'Lägg till/uppdatera')."""
    st.markdown("**📅 Senaste uppdateringar per fält**")
    grid_cols = st.columns(4)
    pairs = [
        ("Utestående aktier", "TS_Utestående aktier"),
        ("P/S", "TS_P/S"),
        ("P/S Q1", "TS_P/S Q1"),
        ("P/S Q2", "TS_P/S Q2"),
        ("P/S Q3", "TS_P/S Q3"),
        ("P/S Q4", "TS_P/S Q4"),
        ("Omsättning idag", "TS_Omsättning idag"),
        ("Omsättning nästa år", "TS_Omsättning nästa år"),
    ]
    for i, (label, ts_key) in enumerate(pairs):
        col = grid_cols[i % 4]
        val = str(row.get(ts_key,"") or "—")
        with col:
            st.caption(f"{label}")
            st.code(val, language="text")

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
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(int(st.session_state.edit_index), len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, int(st.session_state.edit_index) - 1)
    with col_pos:
        st.write(f"Post {int(st.session_state.edit_index)}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, int(st.session_state.edit_index) + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    # Snabba actions för vald ticker
    if not bef.empty:
        st.markdown("### ⚡ Snabbåtgärder")
        act1, act2 = st.columns(2)
        with act1:
            if st.button("💹 Uppdatera endast pris (denna)"):
                df, changed, msg = update_price_for_ticker(df, str(bef.get("Ticker","")))
                if msg != "OK":
                    st.warning(msg)
                spara_data(df)
                st.success("Pris uppdaterat.")
                st.rerun()
        with act2:
            if st.button("🔄 Full auto (endast denna)"):
                df, changed, dbg = update_full_for_ticker(df, str(bef.get("Ticker","")), user_rates, force_ts=True)
                spara_data(df)
                if changed:
                    st.success("Full auto klar – ändringar sparade.")
                else:
                    st.info("Full auto klar – inga ändringar hittades.")
                st.rerun()

        # Statuschips & TS per fält
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            st.markdown(f"**Senast manuellt:** {bef.get('Senast manuellt uppdaterad','') or '—'}")
        with c2:
            st.markdown(f"**Senast auto:** {bef.get('Senast auto-uppdaterad','') or '—'}")
        with c3:
            st.markdown(f"**Källa:** {bef.get('Senast uppdaterad källa','') or '—'}")
        _ts_badge_grid(bef)

    # Formulär
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
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%), P/S (Yahoo), MCap (nu) via Yahoo")
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

        # Hämta basfält från Yahoo (inkl P/S (Yahoo), MCap (nu), utdelning, CAGR)
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if "P/S (Yahoo)" in data and data.get("P/S (Yahoo)") is not None:         df.loc[ridx, "P/S (Yahoo)"]     = float(data.get("P/S (Yahoo)") or 0.0)
        if "MCap (nu)" in data and data.get("MCap (nu)") is not None:             df.loc[ridx, "MCap (nu)"]       = float(data.get("MCap (nu)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")
        st.rerun()

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
# --- EXTRA helper: Yahoo-nyckeltal (Debt/Equity, marginaler, kassa) ---------

def hamta_yahoo_nyckeltal(ticker: str) -> dict:
    """
    Försöker hämta:
      - Debt/Equity (ratio)
      - Bruttomarginal (%)
      - Nettomarginal (%)
      - Kassa (absolut, i financialCurrency)
      - Finansiell valuta (financialCurrency)
    via yfinance.info (bäst-effort; nycklar kan variera över tid).
    """
    out = {"Debt/Equity": 0.0, "Bruttomarginal (%)": 0.0, "Nettomarginal (%)": 0.0, "Kassa": 0.0, "Finansiell valuta": ""}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}

    try:
        de = info.get("debtToEquity")
        if de is not None:
            out["Debt/Equity"] = float(de)
    except Exception:
        pass

    try:
        gm = info.get("grossMargins")
        if gm is not None:
            out["Bruttomarginal (%)"] = float(gm) * 100.0
    except Exception:
        pass

    try:
        pm = info.get("profitMargins")
        if pm is not None:
            out["Nettomarginal (%)"] = float(pm) * 100.0
    except Exception:
        pass

    try:
        cash = info.get("totalCash")
        if cash is not None:
            out["Kassa"] = float(cash)
    except Exception:
        pass

    try:
        fc = info.get("financialCurrency")
        if fc:
            out["Finansiell valuta"] = str(fc).upper()
    except Exception:
        pass

    return out

# --- MAIN --------------------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # ========== Sidopanel: Valutakurser ==========
    st.sidebar.header("💱 Valutakurser → SEK")

    saved_rates = las_sparade_valutakurser()

    # Initiera session-keys för inputs så att vi kan program-mässigt uppdatera dem
    for k, defv in [("rate_usd", saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])),
                    ("rate_nok", saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
                    ("rate_cad", saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
                    ("rate_eur", saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"]))]:
        if k not in st.session_state:
            try:
                st.session_state[k] = float(defv)
            except Exception:
                st.session_state[k] = float(STANDARD_VALUTAKURSER["USD"] if k=="rate_usd" else 1.0)

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")

    # Auto-hämtning av kurser
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.sidebar.success(f"Valutakurser (källa: {provider}) hämtade.")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        # Uppdatera inputs via session_state så de faktiskt ändras visuellt
        st.session_state.rate_usd = float(auto_rates.get("USD", usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", eur))
        st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
        st.rerun()

    user_rates = {"USD": float(st.session_state.rate_usd), "NOK": float(st.session_state.rate_nok),
                  "CAD": float(st.session_state.rate_cad), "EUR": float(st.session_state.rate_eur), "SEK": 1.0}

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear()
            # Läs om och injicera i session_state
            reloaded = las_sparade_valutakurser()
            st.session_state.rate_usd = float(reloaded.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(reloaded.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(reloaded.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(reloaded.get("EUR", st.session_state.rate_eur))
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # Safety: tillåt tom skrivning (avsiktlig rensning) – standard OFF
    st.sidebar.markdown("---")
    st.sidebar.checkbox("🛡️ Tillåt tom skrivning (avancerat)", key="destructive_ok", value=False,
                        help="Kryssa i endast om du medvetet vill rensa bladet. Skyddar mot oavsiktlig tom-skrivning.")

    # ========== Läs data ==========
    df = hamta_data()
    if df.empty:
        # Skapa tom struktur (endast rubriker) och skriv – kräver att vi temporärt tillåter tom skrivning
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        _prev_flag = st.session_state.get("destructive_ok", False)
        st.session_state["destructive_ok"] = True
        spara_data(df)  # skriver rubriker
        st.session_state["destructive_ok"] = _prev_flag
        st.success("Startade ett nytt (tomt) ark med rätt rubriker. Lägg till bolag under 'Lägg till / uppdatera bolag'.")

    # Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # ========== Sidopanel: uppdateringsknappar ==========
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Uppdatering (alla)")

    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True)
    force_ts_all = st.sidebar.checkbox("TS-stämpla även oförändrat", value=True,
                                       help="Stämplar datum även om värdet inte ändrats (för spårade fält).")

    if st.sidebar.button("💹 Uppdatera endast kurser (alla)"):
        df, price_log = update_prices_all(df)
        spara_data(df, do_snapshot=make_snapshot)
        st.session_state["last_price_log"] = price_log
        st.sidebar.success("Klart – kurser sparade.")

    if st.sidebar.button("🔄 Full auto (alla)"):
        df, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot, force_ts=force_ts_all)
        st.session_state["last_auto_log"] = log

    # ========== Sidopanel: Vågkörning ==========
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌊 Vågkörning (partiell auto)")
    mode = st.sidebar.selectbox("Kö-ordning", ["Äldst först", "A–Ö"], index=0)
    mode_key = "oldest" if mode.startswith("Äldst") else "alphabetic"
    batch_size = st.sidebar.number_input("Batchstorlek", min_value=1, max_value=100, value=10, step=1)

    cols_w1, cols_w2, cols_w3 = st.sidebar.columns(3)
    with cols_w1:
        if st.button("Starta våg"):
            start_wave(df, mode=mode_key)
            st.sidebar.info(f"Våg startad ({mode}).")
    with cols_w2:
        if st.button("Fortsätt våg"):
            df, wave_info = run_wave_step(df, user_rates, batch_size=int(batch_size), make_snapshot=make_snapshot, force_ts=True)
            st.sidebar.success(f"Körde {wave_info['processed']} – kvar i kö: {wave_info['remaining']}.")
    with cols_w3:
        if st.button("Återställ vågkö"):
            for k in ["wave_queue","wave_done","wave_changed","wave_miss","wave_started_at","wave_mode"]:
                st.session_state.pop(k, None)
            st.sidebar.info("Vågkö återställd.")

    # ========== Huvudmeny ==========
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

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
