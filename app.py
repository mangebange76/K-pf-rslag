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

# Slutlig kolumnlista i databasen (utökad)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Finansiell valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt", "MCap (nu)",
    # MCap-historik (för 4 senaste TTM-fönstren)
    "MCap Q1","MCap Q2","MCap Q3","MCap Q4",
    "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
    # Nyckeltal
    "Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa",
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
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Valuta","Finansiell valuta","Bolagsnamn","Ticker"):
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
        "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
        "Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa"
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Finansiell valuta, Utdelning, CAGR, P/S (Yahoo), MCap.
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

        fin_ccy = info.get("financialCurrency") or ""
        if fin_ccy:
            out["Finansiell valuta"] = str(fin_ccy).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        ps_y = info.get("priceToSalesTrailing12Months") or info.get("priceToSalesTTM")
        try:
            if ps_y and float(ps_y) > 0:
                out["P/S (Yahoo)"] = float(ps_y)
        except Exception:
            pass

        mcap = info.get("marketCap")
        try:
            if mcap and float(mcap) > 0:
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
    Om force_ts=True stämplas TS + auto-källa även om värdet inte ändras.
    Returnerar True om något fält faktiskt ändrades (värdemässigt).
    """
    changed_fields = []
    any_ts_touched = False

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f]
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # skriv positiva värden; tillåt 0 för vissa icke-P/S-fält
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]:
                write_ok = (float(v) > 0)
            else:
                write_ok = (float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            write_ok = v is not None

        if not write_ok:
            # även om vi inte skriver, kan vi vilja stämpla TS om force_ts och det är ett spårat fält
            if force_ts and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                any_ts_touched = True
            continue

        # ändringsdetektion
        changed = (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))
        if changed:
            df.at[row_idx, f] = v
            changed_fields.append(f)
        # stämpla TS om fältet är spårat, antingen vid ändring eller om force_ts begärs
        if f in TS_FIELDS and (changed or force_ts):
            _stamp_ts_for_field(df, row_idx, f)
            any_ts_touched = True

    if changed_fields or (force_ts and any_ts_touched):
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return bool(changed_fields)
    return False

# app.py — Del 3/7
# --- SEC (US/IFRS), TTM/P/S, MCap-historik & Yahoo-nyckeltal -----------------

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

# ---------- datum/parse-hjälp ------------------------------------------------
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

# ---------- Aktier (instant; multi-class) ------------------------------------
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

# ---------- FX & Yahoo-stöd för TTM/Mcap -------------------------------------
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

# ---------- SEC kvartalsintäkter (Q1–Q4) + HOTFIX för Q4 via 10-K ------------
def _sec_collect_quarters_for_unit(facts: dict, taxo: str, unit_code: str, forms=("10-Q","10-Q/A")) -> dict:
    """Kvartalsintäkter (≈90 dagar) för given taxonomi och enhet."""
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet","Revenues","Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    out = {}
    gaap = (facts.get("facts") or {}).get(taxo, {})
    for name in rev_keys:
        fact = gaap.get(name)
        if not fact:
            continue
        units = (fact.get("units") or {})
        arr = units.get(unit_code)
        if not isinstance(arr, list):
            continue
        for it in arr:
            form = (it.get("form") or "").upper()
            if forms and not any(f in form for f in forms):
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
            except Exception:
                continue
            out[end] = v if end not in out else out[end]
    return out

def _sec_collect_annual_for_unit(facts: dict, taxo: str, unit_code: str, forms=("10-K","10-K/A")) -> dict:
    """Helårsintäkter (≈365 dagar) för given taxonomi och enhet."""
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet","Revenues","Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    out = {}
    gaap = (facts.get("facts") or {}).get(taxo, {})
    for name in rev_keys:
        fact = gaap.get(name)
        if not fact:
            continue
        units = (fact.get("units") or {})
        arr = units.get(unit_code)
        if not isinstance(arr, list):
            continue
        for it in arr:
            form = (it.get("form") or "").upper()
            if forms and not any(f in form for f in forms):
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
            if dur is None or dur < 330 or dur > 380:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            out[end] = v if end not in out else out[end]
    return out

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Slå ihop flera revenue-koncept, välj enhet med flest kvartal och
    SYNTA ett Q4 om bara 10-K finns: Q4 = 10-K – (Q1+Q2+Q3).
    Returnerar (rows, unit) nyast→äldst.
    """
    taxos = [
        ("us-gaap",  {"qforms": ("10-Q", "10-Q/A"), "aforms": ("10-K","10-K/A")}),
        ("ifrs-full", {"qforms": ("6-K", "6-K/A", "10-Q", "10-Q/A"), "aforms": ("6-K","6-K/A")}),
    ]
    prefer_units = ("USD","CAD","EUR","GBP","SEK","NOK","DKK","CHF","JPY")

    best_rows = []
    best_unit = None

    for taxo, cfg in taxos:
        for unit_code in prefer_units:
            qmap = _sec_collect_quarters_for_unit(facts, taxo, unit_code, cfg["qforms"])
            if not qmap:
                continue

            # försök syntetisera Q4 från 10-K om den saknas
            amap = _sec_collect_annual_for_unit(facts, taxo, unit_code, cfg["aforms"])
            if amap:
                a_end = max(amap.keys())  # senaste helår
                if a_end not in qmap:
                    q_before = [d for d in qmap.keys() if d < a_end]
                    q_before.sort(reverse=True)
                    if len(q_before) >= 3:
                        qsum3 = qmap[q_before[0]] + qmap[q_before[1]] + qmap[q_before[2]]
                        q4_est = float(amap[a_end]) - float(qsum3)
                        if q4_est > 0:
                            qmap[a_end] = q4_est

            rows = sorted(qmap.items(), key=lambda t: t[0], reverse=True)
            if len(rows) > len(best_rows):
                best_rows = rows
                best_unit = unit_code

    if not best_rows:
        return [], None

    return best_rows[:max_quarters], best_unit

def _sec_quarterly_revenues(facts: dict):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=4)
    return [v for (_, v) in rows]

# ---------- Global Yahoo fallback (icke-SEC) ---------------------------------
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

# ---------- Yahoo nyckeltal (Debt/Equity, marginaler, kassa) -----------------
def hamta_yahoo_nyckeltal(ticker: str) -> dict:
    out = {"Debt/Equity": 0.0, "Bruttomarginal (%)": 0.0, "Nettomarginal (%)": 0.0, "Kassa": 0.0, "Finansiell valuta": ""}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # finansiell valuta (om finns här)
        fccy = info.get("financialCurrency")
        if fccy:
            out["Finansiell valuta"] = str(fccy).upper()

        # balansräkning & IS
        bs = getattr(t, "balance_sheet", None)
        is_ = getattr(t, "income_stmt", None)
        qbs = getattr(t, "quarterly_balance_sheet", None)
        qis = getattr(t, "quarterly_income_stmt", None)

        def _last_val(df: pd.DataFrame, keys: list[str]) -> float:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return 0.0
            idx = [str(x).strip() for x in df.index]
            for k in keys:
                if k in idx:
                    ser = df.loc[k].dropna()
                    if not ser.empty:
                        try:
                            return float(ser.iloc[0])
                        except Exception:
                            return 0.0
            return 0.0

        # total debt / equity
        total_debt = _last_val(qbs or bs, ["Total Debt","TotalDebt","Total debt"])
        total_equity = _last_val(qbs or bs, ["Total Stockholder Equity","Total stockholders' equity","TotalEquityGrossMinorityInterest","Total Equity"])
        if total_equity and total_equity != 0:
            out["Debt/Equity"] = float(total_debt) / float(total_equity)

        # gross/net marginals
        total_rev = _last_val(qis or is_, ["Total Revenue","TotalRevenue","Revenues","Revenue"])
        gross_profit = _last_val(qis or is_, ["Gross Profit","GrossProfit"])
        net_income = _last_val(qis or is_, ["Net Income","NetIncome","Net income applicable to common shares"])
        if total_rev and total_rev != 0:
            if gross_profit:
                out["Bruttomarginal (%)"] = float(gross_profit) / float(total_rev) * 100.0
            if net_income:
                out["Nettomarginal (%)"] = float(net_income) / float(total_rev) * 100.0

        # cash
        cash = _last_val(qbs or bs, ["Cash And Cash Equivalents","CashAndCashEquivalents","Cash"])
        out["Kassa"] = float(cash or 0.0)

    except Exception:
        pass
    return out

# ---------- SEC + Yahoo combo, med P/S & MCap-historik -----------------------
def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q/10-K eller IFRS 6-K),
    pris/valuta/namn från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
    Sparar även MCap (nu) och historiska MCap (Q1–Q4) + datum.
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
    for k in ("Bolagsnamn", "Valuta", "Finansiell valuta", "Aktuell kurs", "P/S (Yahoo)", "MCap (nu)"):
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
        out["Utestående aktier"] = shares_used / 1e6

    # Market cap (nu)
    mcap_now = float(out.get("MCap (nu)", 0.0) or 0.0)
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used
        out["MCap (nu)"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    if not q_rows:
        return out

    conv = 1.0
    fin_unit = (rev_unit or out.get("Finansiell valuta") or px_ccy or "USD").upper()
    if fin_unit != px_ccy:
        conv = _fx_rate_cached(fin_unit, px_ccy) or 1.0

    # bygg TTM-lista (max 4)
    ttm_list = _ttm_windows(q_rows, need=4)
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # Historik: P/S Q1–Q4 + MCap Q1–Q4 (via historiskt pris * konstanta shares_used)
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

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik + MCap Q1–Q4.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/pris/P/S(Yahoo)/MCap
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","P/S (Yahoo)","MCap (nu)"):
        if y.get(k): out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(out.get("MCap (nu)", 0.0) or 0.0)
    try:
        if mcap <= 0 and info.get("marketCap"):
            mcap = float(info.get("marketCap"))
            out["MCap (nu)"] = mcap
    except Exception:
        pass

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
    fin_ccy = str(info.get("financialCurrency") or out.get("Finansiell valuta") or px_ccy).upper()
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

    # P/S Q1–Q4 + MCap Q1–Q4
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
                    out[f"MCap Datum Q{idx}"] = str(d_end)

    return out

# app.py — Del 4/7
# --- FMP, Finnhub, snapshots, per-ticker & batch-uppdatering -----------------

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

    # pris + shares + mcap
    q, sc_q = _fmp_get(f"api/v3/quote/{out['_symbol']}", stable=False)
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
    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{out['_symbol']}", stable=False)
    out["_debug"]["ratios_ttm_sc"] = sc_rttm
    if isinstance(rttm, list) and rttm:
        v = rttm[0].get("priceToSalesTTM") or rttm[0].get("priceToSalesRatioTTM")
        try:
            if v and float(v) > 0:
                out["P/S"] = float(v)
        except Exception:
            pass

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

# =============== Snapshots & hjälpare =======================================

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
    """Returnerar äldsta TS bland alla TS_-kolumner (eller None)."""
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

# =============== Per-ticker uppdaterare ======================================

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares + TTM) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S/MCap/shares) om saknas
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","Finansiell valuta","P/S (Yahoo)","MCap (nu)",
            "MCap Q1","MCap Q2","MCap Q3","MCap Q4","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"
        ]}
        for k in ["Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","Utestående aktier","P/S","P/S (Yahoo)",
                  "P/S Q1","P/S Q2","P/S Q3","P/S Q4","MCap (nu)",
                  "MCap Q1","MCap Q2","MCap Q3","MCap Q4","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]:
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

    # 3) FMP light P/S eller MCap/shares om saknas
    try:
        need_fmp = ("P/S" not in vals) or ("MCap (nu)" not in vals) or ("Utestående aktier" not in vals)
        if need_fmp:
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier"), "MCap (nu)": fmpl.get("MCap (nu)")}
            for k in ["P/S","Utestående aktier","MCap (nu)","Aktuell kurs"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    return vals, debug

def _row_index_for_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    idx = df.index[df["Ticker"].astype(str).str.upper() == str(ticker).upper()]
    return int(idx[0]) if len(idx) else None

def update_price_for_ticker(df: pd.DataFrame, ticker: str):
    """Uppdatera ENDAST pris + MCap (nu) för en ticker."""
    ridx = _row_index_for_ticker(df, ticker)
    if ridx is None:
        return df, False, f"{ticker} hittades inte i tabellen."

    y = hamta_yahoo_fält(ticker)
    new_vals = {}
    if y.get("Aktuell kurs", 0) > 0:
        new_vals["Aktuell kurs"] = y["Aktuell kurs"]
    if y.get("MCap (nu)", 0) > 0:
        new_vals["MCap (nu)"] = y["MCap (nu)"]

    changed = apply_auto_updates_to_row(df, ridx, new_vals, source="Pris (Yahoo)", changes_map={}, force_ts=True)
    # beräkna om riktkurser m.m.
    return uppdatera_berakningar(df, {}), changed, None

def update_full_for_ticker(df: pd.DataFrame, ticker: str, user_rates: dict, *, force_ts: bool = True):
    """Full auto för EN ticker (SEC/Yahoo→Finnhub→FMP)."""
    ridx = _row_index_for_ticker(df, ticker)
    if ridx is None:
        return df, False, {"error": "ticker not found"}
    new_vals, debug = auto_fetch_for_ticker(ticker)

    # Skriv värden
    changed = apply_auto_updates_to_row(
        df, ridx, new_vals,
        source="Auto (SEC/Yahoo→Finnhub→FMP)",
        changes_map={},
        force_ts=force_ts
    )

    # Nyckeltal (bäst-effort)
    nyck = hamta_yahoo_nyckeltal(ticker)
    if nyck:
        for k, v in nyck.items():
            if k in df.columns and v not in (None, ""):
                try:
                    df.at[ridx, k] = float(v) if isinstance(v, (int,float,np.floating)) else v
                except Exception:
                    df.at[ridx, k] = v

    df = uppdatera_berakningar(df, user_rates)
    return df, changed, debug

# =============== Batch: pris & auto för alla =================================

def update_prices_all(df: pd.DataFrame):
    """Uppdatera pris + MCap för alla rader."""
    log = {"updated": [], "skipped": []}
    progress = st.sidebar.progress(0)
    total = len(df)
    for i, row in df.iterrows():
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1))
            continue
        y = hamta_yahoo_fält(tkr)
        new_vals = {}
        if y.get("Aktuell kurs", 0) > 0:
            new_vals["Aktuell kurs"] = y["Aktuell kurs"]
        if y.get("MCap (nu)", 0) > 0:
            new_vals["MCap (nu)"] = y["MCap (nu)"]
        changed = apply_auto_updates_to_row(df, i, new_vals, source="Pris (Yahoo)", changes_map={}, force_ts=True)
        if changed:
            log["updated"].append(tkr)
        else:
            log["skipped"].append(tkr)
        progress.progress((i+1)/max(total,1))
    df = uppdatera_berakningar(df, {})
    return df, log

def auto_update_all(df: pd.DataFrame, user_rates: dict, *, make_snapshot: bool = False, force_ts: bool = True):
    """
    Kör auto-uppdatering för alla rader (tunga anrop). Skriver endast meningsfulla värden,
    men stämplar TS/källa om force_ts=True även vid oförändrat värde.
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
            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(
                df, idx, new_vals,
                source="Auto (SEC/Yahoo→Finnhub→FMP)",
                changes_map=log["changed"],
                force_ts=force_ts
            )
            if not changed and not new_vals:
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
        st.sidebar.info("Ingen faktisk ändring upptäcktes – endast datum/källa kan ha stämplats.")

    return df, log

# =============== Vågkörning (batch i omgångar) ===============================

def _wave_build_queue(df: pd.DataFrame, mode: str = "oldest") -> list[str]:
    """Bygg kö med tickers enligt vald prioritet."""
    work = add_oldest_ts_col(df.copy())
    if mode == "oldest":
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        work = work.sort_values(by=["Bolagsnamn","Ticker"])
    tickers = [str(t).upper() for t in work["Ticker"].tolist() if str(t).strip()]
    return tickers

def start_wave(df: pd.DataFrame, mode: str = "oldest"):
    st.session_state["wave_queue"] = _wave_build_queue(df, mode)
    st.session_state["wave_pos"] = 0

def run_wave_step(df: pd.DataFrame, user_rates: dict, *, batch_size: int = 10, make_snapshot: bool = False, force_ts: bool = True):
    """
    Kör nästa batch i kön (session_state["wave_queue"]). Returnerar (df, stat).
    stat: {"processed": N, "remaining": M}
    """
    queue = st.session_state.get("wave_queue") or []
    pos = int(st.session_state.get("wave_pos", 0))
    if not queue or pos >= len(queue):
        return df, {"processed": 0, "remaining": 0}

    end = min(len(queue), pos + int(batch_size))
    to_run = queue[pos:end]
    for tkr in to_run:
        df, _, _ = update_full_for_ticker(df, tkr, user_rates, force_ts=force_ts)
    spara_data(df, do_snapshot=make_snapshot)
    st.session_state["wave_pos"] = end
    return df, {"processed": len(to_run), "remaining": max(0, len(queue) - end)}

# app.py — Del 5/7
# --- Hjälplistor, formattering & Kontroll-vy ---------------------------------

def _fmt_money_short(v: float) -> str:
    """Kortformat för stora tal (K/M/B/T)."""
    try:
        x = float(v or 0.0)
    except Exception:
        return "0"
    neg = x < 0
    x = abs(x)
    if x >= 1_000_000_000_000:
        s = f"{x/1_000_000_000_000:.2f}T"
    elif x >= 1_000_000_000:
        s = f"{x/1_000_000_000:.2f}B"
    elif x >= 1_000_000:
        s = f"{x/1_000_000:.2f}M"
    elif x >= 1_000:
        s = f"{x/1_000:.2f}K"
    else:
        s = f"{x:.0f}"
    return f"-{s}" if neg else s

def _risk_label_from_mcap(mcap_now: float) -> str:
    """
    Enkel cap-klassning (USD-nivåer ungefärliga, men vi klassar på beloppet oavsett valuta):
    Megacap ≥ 200B, Largecap ≥ 10B, Midcap ≥ 2B, Smallcap ≥ 300M, annars Microcap.
    """
    try:
        x = float(mcap_now or 0.0)
    except Exception:
        x = 0.0
    if x >= 200_000_000_000:
        return "Megacap"
    if x >= 10_000_000_000:
        return "Largecap"
    if x >= 2_000_000_000:
        return "Midcap"
    if x >= 300_000_000:
        return "Smallcap"
    return "Microcap"

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Bolag som sannolikt kräver manuell hantering:
    - saknar någon av kärnfälten eller TS,
    - och/eller äldsta TS är äldre än 'older_than_days'.
    """
    need_cols = [
        "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Aktuell kurs"
    ]
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
    st.subheader("📒 Senaste körloggar")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Auto-uppdatering (senaste)**")
        log = st.session_state.get("last_auto_log")
        if not log:
            st.info("Ingen auto-körning i denna session ännu.")
        else:
            if log.get("changed"):
                st.markdown("**Ändrade fält** (ticker → fält)")
                st.json(log["changed"])
            if log.get("misses"):
                st.markdown("**Missar** (ticker → orsak)")
                st.json(log["misses"])
            if log.get("debug_first_20"):
                st.markdown("**Debug (första 20)**")
                st.json(log["debug_first_20"])
    with col2:
        st.markdown("**Prisuppdatering (senaste)**")
        plog = st.session_state.get("last_price_log")
        if not plog:
            st.info("Ingen prisuppdatering i denna session ännu.")
        else:
            st.write(f"Uppdaterade: {len(plog.get('updated', []))} | Skippade: {len(plog.get('skipped', []))}")
            if plog.get("updated"):
                st.markdown("**Uppdaterade tickers**")
                st.write(", ".join(plog["updated"]))
            if plog.get("skipped"):
                st.markdown("**Skippade tickers**")
                st.write(", ".join(plog["skipped"]))

    st.divider()

    # 4) Snabbtest: visa inhämtning för en ticker
    st.subheader("🧪 Snabbtest av inhämtning")
    tkr = st.text_input("Ticker att testa (valfritt)", "")
    if tkr:
        colL, colR = st.columns(2)
        with colL:
            st.write("**SEC/Yahoo combo (inkl. Q4-hotfix & MCap-historik)**")
            try:
                v = hamta_sec_yahoo_combo(tkr.upper().strip())
                st.json({k: v.get(k) for k in ["Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]})
            except Exception as e:
                st.error(f"SEC/Yahoo fel: {e}")
        with colR:
            st.write("**Yahoo-nyckeltal**")
            try:
                yk = hamta_yahoo_nyckeltal(tkr.upper().strip())
                st.json(yk)
            except Exception as e:
                st.error(f"Nyckeltalsfel: {e}")

# app.py — Del 6/7
# --- Analys, Portfölj & Investeringsförslag ----------------------------------

def _badge_dates(row: pd.Series) -> str:
    manu = str(row.get("Senast manuellt uppdaterad","")).strip()
    auto = str(row.get("Senast auto-uppdaterad","")).strip()
    src  = str(row.get("Senast uppdaterad källa","")).strip()
    parts = []
    if manu:
        parts.append(f"✍️ Manuell: {manu}")
    if auto:
        parts.append(f"⚙️ Auto: {auto}")
    if src:
        parts.append(f"🔗 Källa: {src}")
    return " | ".join(parts) if parts else "—"

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        # snabbknappar: uppdatera pris / full auto för just denna
        tkr = vis_df.iloc[st.session_state.analys_idx]["Ticker"]
        if st.button("💹 Uppdatera pris", key="analys_price"):
            df2, changed, _ = update_price_for_ticker(df, tkr)
            if changed:
                spara_data(df2, do_snapshot=False)
                st.success(f"Pris uppdaterat för {tkr}.")
            else:
                st.info("Ingen prisförändring (TS/källa stämplad).")
            st.experimental_rerun()
    with col_c:
        tkr = vis_df.iloc[st.session_state.analys_idx]["Ticker"]
        if st.button("🤖 Full auto (detta bolag)", key="analys_fullauto"):
            df2, _, _ = update_full_for_ticker(df, tkr, user_rates, force_ts=True)
            spara_data(df2, do_snapshot=False)
            st.success(f"Auto-uppdatering klar för {tkr}.")
            st.experimental_rerun()

    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    st.caption(_badge_dates(r))

    cols = [
        "Ticker","Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","Utestående aktier",
        "P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
        "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

    with st.expander("📜 Historik & nyckeltal"):
        left, right = st.columns(2)
        with left:
            st.markdown("**P/S-historik**")
            ps_hist = {k: r.get(k, 0.0) for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]}
            st.json(ps_hist)
            st.markdown("**MCap-historik**")
            mcap_hist = {
                "Q1": f"{_fmt_money_short(r.get('MCap Q1',0))} ({r.get('MCap Datum Q1','')})",
                "Q2": f"{_fmt_money_short(r.get('MCap Q2',0))} ({r.get('MCap Datum Q2','')})",
                "Q3": f"{_fmt_money_short(r.get('MCap Q3',0))} ({r.get('MCap Datum Q3','')})",
                "Q4": f"{_fmt_money_short(r.get('MCap Q4',0))} ({r.get('MCap Datum Q4','')})",
            }
            st.json(mcap_hist)
        with right:
            st.markdown("**Nyckeltal (senaste)**")
            nyck = {
                "Debt/Equity": round(float(r.get("Debt/Equity",0.0)), 3),
                "Bruttomarginal (%)": round(float(r.get("Bruttomarginal (%)",0.0)), 2),
                "Nettomarginal (%)": round(float(r.get("Nettomarginal (%)",0.0)), 2),
                "Kassa": _fmt_money_short(r.get("Kassa",0.0)),
            }
            st.json(nyck)
            st.markdown(f"**Risklabel:** {_risk_label_from_mcap(r.get('MCap (nu)',0.0))}")

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

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]
    if "MCap (nu)" in port.columns:
        port["Risklabel"] = port["MCap (nu)"].apply(_risk_label_from_mcap)
        show_cols.insert(2, "Risklabel")
    st.dataframe(port[show_cols], use_container_width=True, hide_index=True)

def _avg_ps_last4(row: pd.Series) -> float:
    vals = [float(row.get(k,0.0)) for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"] if float(row.get(k,0.0)) > 0]
    return round(float(np.mean(vals)), 2) if vals else 0.0

def _current_ps(row: pd.Series) -> float:
    ps = float(row.get("P/S", 0.0))
    if ps <= 0:
        ps = float(row.get("P/S (Yahoo)", 0.0))
    return round(ps, 2) if ps > 0 else 0.0

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    cap_filter = st.selectbox("Filtrera på börsvärde", ["Alla","Microcap","Smallcap","Midcap","Largecap","Megacap"], index=0)

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # risklabel / cap filter
    base["MCap (nu)"] = pd.to_numeric(base.get("MCap (nu)"), errors="coerce").fillna(0.0)
    base["Risklabel"] = base["MCap (nu)"].apply(_risk_label_from_mcap)
    if cap_filter != "Alla":
        base = base[base["Risklabel"] == cap_filter].copy()
        if base.empty:
            st.info("Inga bolag i denna cap-klass.")
            return

    # nycklar
    base["P/S (nu)"] = base.apply(_current_ps, axis=1)
    base["P/S-snitt 4Q"] = base.apply(_avg_ps_last4, axis=1)
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

    # portföljandelar
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
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    ps_nu = rad.get("P/S (nu)", 0.0)
    ps_avg = rad.get("P/S-snitt 4Q", 0.0)
    mcap_now_fmt = _fmt_money_short(rad.get("MCap (nu)", 0.0))

    # huvudinfo
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Nuvarande börsvärde:** {mcap_now_fmt}",
        f"- **P/S (nu):** {ps_nu}",
        f"- **P/S-snitt (senaste 4):** {ps_avg}",
        f"- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
        f"- **Risklabel:** {rad.get('Risklabel','-')}",
    ]
    st.markdown("\n".join(lines))

    # expander med extra detaljer
    with st.expander("🔎 Detaljer (marginaler, skulder, MCAP-historik)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Nyckeltal**")
            st.write(f"- Debt/Equity: {round(float(rad.get('Debt/Equity',0.0)),3)}")
            st.write(f"- Bruttomarginal: {round(float(rad.get('Bruttomarginal (%)',0.0)),2)} %")
            st.write(f"- Nettomarginal: {round(float(rad.get('Nettomarginal (%)',0.0)),2)} %")
            st.write(f"- Kassa: {_fmt_money_short(rad.get('Kassa',0.0))}")
        with col2:
            st.markdown("**MCAP-historik (senaste 4 TTM-fönster)**")
            hist_rows = []
            for q in ["Q1","Q2","Q3","Q4"]:
                val = rad.get(f"MCap {q}", 0.0)
                dat = rad.get(f"MCap Datum {q}", "")
                hist_rows.append(f"- {q}: {_fmt_money_short(val)} ({dat})")
            st.markdown("\n".join(hist_rows))

# app.py — Del 7/7
# --- Lägg till/uppdatera + MAIN ---------------------------------------------

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

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
            st.experimental_rerun()
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)
            st.experimental_rerun()

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
        st.subheader(f"{bef.get('Bolagsnamn','')} ({bef.get('Ticker','')})")
        st.caption(_badge_dates(bef))
        # Snabbknappar: pris & full auto för just detta bolag
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💹 Uppdatera kurs (detta bolag)"):
                df2, changed, err = update_price_for_ticker(df, bef["Ticker"])
                if err:
                    st.error(err)
                else:
                    spara_data(df2, do_snapshot=False)
                    st.success("Kurs uppdaterad (TS/källa stämplades).")
                    st.experimental_rerun()
        with c2:
            if st.button("🤖 Full auto (detta bolag)"):
                df2, _, _ = update_full_for_ticker(df, bef["Ticker"], user_rates, force_ts=True)
                spara_data(df2, do_snapshot=False)
                st.success("Auto-uppdatering klar.")
                st.experimental_rerun()
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",    value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",        value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)",    value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Vid spara uppdateras också (utan att skriva över manuella 0-värden):**")
            st.write("- Bolagsnamn, Valuta, Finansiell valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%), P/S (Yahoo), MCap (nu) via Yahoo")
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
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Finansiell valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
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
        if data.get("Finansiell valuta"): df.loc[ridx, "Finansiell valuta"] = data["Finansiell valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if data.get("MCap (nu)",0)>0:    df.loc[ridx, "MCap (nu)"] = data["MCap (nu)"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None:
            df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:
            df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if data.get("P/S (Yahoo)",0)>0:
            df.loc[ridx, "P/S (Yahoo)"] = data["P/S (Yahoo)"]

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

# --- MAIN --------------------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: valutakurser
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, format="%.4f")

    # Auto-hämtning av kurser (uppdatera lokala variabler – undvik session_state-fel)
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.sidebar.success(f"Valutakurser (källa: {provider}) hämtade.")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        usd = float(auto_rates.get("USD", usd))
        nok = float(auto_rates.get("NOK", nok))
        cad = float(auto_rates.get("CAD", cad))
        eur = float(auto_rates.get("EUR", eur))

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

    # --- Sidebar: uppdateringar ------------------------------------------------
    st.sidebar.subheader("🛠️ Uppdatering")
    # Pris för ALLA
    if st.sidebar.button("💹 Uppdatera kurser (alla)"):
        df2, plog = update_prices_all(df.copy())
        spara_data(df2, do_snapshot=False)
        st.session_state["last_price_log"] = plog
        st.sidebar.success("Priser uppdaterade för alla.")
        st.experimental_rerun()

    # Full auto för ALLA
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=False)
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → Finnhub → FMP)"):
        df2, log = auto_update_all(df.copy(), user_rates, make_snapshot=make_snapshot, force_ts=True)
        st.session_state["last_auto_log"] = log
        st.experimental_rerun()

    # Vågkörning
    st.sidebar.subheader("🌊 Vågkörning (batch i omgångar)")
    mode = st.sidebar.selectbox("Kö-ordning", ["Äldst först","A–Ö"], index=0)
    batch_size = st.sidebar.slider("Batch-storlek per körning", min_value=5, max_value=50, value=10, step=5)

    cA, cB, cC = st.sidebar.columns(3)
    with cA:
        if st.button("Starta våg"):
            start_wave(df, mode="oldest" if mode.startswith("Äldst") else "alpha")
            st.sidebar.success("Kö skapad.")
    with cB:
        if st.button("Kör nästa batch"):
            if "wave_queue" not in st.session_state or not st.session_state["wave_queue"]:
                st.sidebar.warning("Ingen kö – klicka 'Starta våg' först.")
            else:
                df2, stat = run_wave_step(df.copy(), user_rates, batch_size=batch_size, make_snapshot=False, force_ts=True)
                spara_data(df2, do_snapshot=False)
                st.sidebar.success(f"Körde {stat['processed']} tickers. Kvar: {stat['remaining']}.")
                st.experimental_rerun()
    with cC:
        if st.button("Återställ kö"):
            st.session_state.pop("wave_queue", None)
            st.session_state.pop("wave_pos", None)
            st.sidebar.info("Kö återställd.")

    # Köstatus
    queue = st.session_state.get("wave_queue") or []
    pos = int(st.session_state.get("wave_pos", 0))
    if queue:
        st.sidebar.caption(f"Våg-kö: {pos}/{len(queue)} bearbetade.")

    # --- Meny -----------------------------------------------------------------
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # om df uppdaterades, skriv tillbaka lokalt så efterföljande vyer ser ändring utan full reload
        df = df2
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
