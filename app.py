# app.py — Del 1/? — Importer, tid, konstanter, Google Sheets & valuta
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
    "Direktavkastning (%)", "Sektor", "_marketCap_raw",
    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
    # TS-kolumner
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","avkastning","marketcap","cap"]):
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
    """Sätter TS-kolumnen för ett spårat fält om den finns (stämpla alltid vid ändring/körning)."""
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

# app.py — Del 2/? — Yahoo-hjälpare, beräkningar & rad-uppdatering

# ---------- trygga hämtningar ur yfinance.info ------------------------------

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
    Basfält från Yahoo: Namn, Kurs, Valuta, Årlig utdelning, Direktavkastning, Sektor, MarketCap (rå),
    samt CAGR 5 år (%) (approx via finansiella).
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "Direktavkastning (%)": 0.0,
        "Sektor": "",
        "_marketCap_raw": 0.0,
        "CAGR 5 år (%)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris (regularMarketPrice -> fallback hist)
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

        # Sektor
        if info.get("sector"):
            out["Sektor"] = str(info.get("sector"))

        # Market cap rå
        mc = info.get("marketCap")
        try:
            if mc is not None:
                out["_marketCap_raw"] = float(mc)
        except Exception:
            pass

        # Utdelning & yield
        # dividendYield är fraktion (t.ex. 0.021 → 2.1%)
        dy = info.get("dividendYield", None)
        if dy is not None:
            try:
                out["Direktavkastning (%)"] = float(dy) * 100.0
            except Exception:
                pass
        # dividendRate (belopp/år)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
                # Om yield saknas men vi har kurs
                if out.get("Direktavkastning (%)", 0.0) == 0.0 and out.get("Aktuell kurs", 0.0) > 0:
                    out["Direktavkastning (%)"] = round((out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0, 4)
            except Exception:
                pass

        # CAGR (5 år) approx
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---------------- Beräkningar & sammanställning ------------------------------

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier
    Obs: Omsättningskolumnerna är i miljoner (bolagets valuta).
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if pd.notna(x) and float(x) > 0]
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp för prognos ( >100% → 50%, <0% → 2% )
        cagr = float(rad.get("CAGR 5 år (%)", 0.0) or 0.0)
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år" (miljoner)
        oms_next = float(rad.get("Omsättning nästa år", 0.0) or 0.0)
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev manuella värden
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0) or 0.0)
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0) or 0.0)

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        # Notera: omsättningskolumner = miljoner; Utestående aktier = miljoner → matchar och ger pris i bolagets valuta.
        aktier_ut_m = float(rad.get("Utestående aktier", 0.0) or 0.0)  # miljoner
        if aktier_ut_m > 0 and ps_snitt > 0:
            oms_idag_m = float(rad.get("Omsättning idag", 0.0) or 0.0)
            df.at[i, "Riktkurs idag"]    = round((oms_idag_m               * ps_snitt) / aktier_ut_m, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(oms_next)          * ps_snitt) / aktier_ut_m, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"]) * ps_snitt) / aktier_ut_m, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"]) * ps_snitt) / aktier_ut_m, 2)
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
    always_stamp: bool = False
) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde.
    Uppdaterar TS_ för spårade fält.
    Sätter 'Senast auto-uppdaterad' + källa.
    Om always_stamp=True stämplas TS_ även om värdet är oförändrat (och auto-datum/källa sätts).
    Returnerar True om något fält faktiskt ändrades.
    """
    changed_fields = []
    any_written_or_stamped = False

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f] if f in df.columns else None

        # Skrivpolicy: skriv numeriskt om >0 (eller >=0 om fältet inte är P/S/aktier), text om ej tomt
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # P/S & aktier måste vara >0; övriga får vara >=0
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]:
                write_ok = (float(v) > 0)
            else:
                write_ok = (float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        # Om vi inte ska skriva, men always_stamp=True och fältet är TS-spårat → stämpla ändå
        if not write_ok:
            if always_stamp and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)   # stämpla trots oförändrat
                any_written_or_stamped = True
            continue

        # Jämför och skriv om det faktiskt skiljer sig
        do_write = (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))
        if do_write:
            df.at[row_idx, f] = v
            changed_fields.append(f)
            any_written_or_stamped = True
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
        else:
            # Värdet oförändrat – men om always_stamp och f spåras → stämpla
            if always_stamp and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                any_written_or_stamped = True

    # Sätt auto-datum/källa om något hände (skrivning eller stämpling)
    if any_written_or_stamped:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            tkr = str(df.at[row_idx, "Ticker"]).strip().upper()
            if tkr:
                changes_map.setdefault(tkr, []).extend(changed_fields)
        return bool(changed_fields)  # True om något fält ändrades
    return False

# app.py — Del 3/? — FMP/SEC/Finnhub, robust kvartal/TTM & auto-fetch-pipeline

# ==================== FMP (Financial Modeling Prep) ==========================

FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.0))
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20))

def _fmp_get(path: str, params=None):
    """
    Throttlad GET med enkel backoff + 'circuit breaker' vid 429.
    Returnerar (json, statuscode).
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
    js, sc = _fmp_get(f"api/v3/quote-short/{sym}")
    if isinstance(js, list) and js:
        return sym
    js, sc = _fmp_get("api/v3/search", {"query": yahoo_ticker, "limit": 1})
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
    q, sc_q = _fmp_get(f"api/v3/quote/{sym}")
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
    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}")
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
    Fullare variant: försöker hämta P/S (TTM), P/S Q1–Q4 via ratios quarterly.
    """
    out = {"_debug": {}}
    sym = _fmp_pick_symbol(yahoo_ticker)
    out["_symbol"] = sym

    # Quote (pris + marketCap)
    qfull, sc_qfull = _fmp_get(f"api/v3/quote/{sym}")
    out["_debug"]["quote_sc"] = sc_qfull
    market_cap = 0.0
    if isinstance(qfull, list) and qfull:
        q0 = qfull[0]
        if "price" in q0:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
        if q0.get("marketCap") is not None:
            try: market_cap = float(q0["marketCap"]); out["_marketCap_raw"] = market_cap
            except: pass

    # P/S via ratios-ttm
    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}")
    out["_debug"]["ratios_ttm_sc"] = sc_rttm
    if isinstance(rttm, list) and rttm:
        try:
            v = rttm[0].get("priceToSalesTTM") or rttm[0].get("priceToSalesRatioTTM")
            if v is not None and float(v) > 0:
                out["P/S"] = float(v)
                out["_debug"]["ps_source"] = "ratios-ttm"
        except Exception:
            pass

    # P/S Q1–Q4 (FMP ratios-quarterly om tillgängligt)
    rq, sc_rq = _fmp_get(f"api/v3/ratios/{sym}", {"period": "quarter", "limit": 8})
    out["_debug"]["ratios_quarter_sc"] = sc_rq
    if isinstance(rq, list) and rq:
        # ta de 4 senaste unika perioderna (FMP sorteras oftast nyast först)
        ps_quarters = []
        seen_periods = set()
        for row in rq:
            ps = row.get("priceToSalesRatio")
            period = row.get("date") or row.get("period")
            if period and period not in seen_periods:
                seen_periods.add(period)
                if ps is not None:
                    try:
                        ps_quarters.append(float(ps))
                    except Exception:
                        ps_quarters.append(0.0)
            if len(ps_quarters) >= 4:
                break
        for i, val in enumerate(ps_quarters, start=1):
            out[f"P/S Q{i}"] = float(val or 0.0)

    return out

# =============================== SEC =========================================

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

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 24):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (≈3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    Inkluderar “year-end” kvartal (Dec/Jan) genom tolerans 70–110 dagar.
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

    all_rows = []
    all_unit = None

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
                    # tolerera 70–110 för att inte tappa Dec/Jan quarter
                    if dur is None or dur < 70 or dur > 110:
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
                    # deduplicera per end-date
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)
                all_rows.extend(rows)
                all_unit = all_unit or unit_code

    # deduplicera igen om både us-gaap och ifrs-full gav något
    if all_rows:
        seen = {}
        for d, v in all_rows:
            seen[d] = v
        rows = sorted(seen.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
        return rows, all_unit
    return [], None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 24):
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

# ---------- Yahoo pris & implied shares & pris-historik ----------------------

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
    Inkluderar December/January genom båda källorna och robust datumparser.
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

# ==================== SEC + Yahoo = komboflöde ===============================

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn/sector/mcap från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
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
        if y.get(k): out[k] = y[k]
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
        out["Utestående aktier"] = shares_used / 1e6  # M-styck

    # Market cap (nu)
    mcap_now = float(out.get("_marketCap_raw", 0.0) or 0.0)
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used

    # SEC kvartalsintäkter + unit → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=24)
    if not q_rows or not rev_unit:
        return out
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    # Bygg minst 5 TTM-fönster (så vi inte tappar Dec/Jan när fyra senaste sträcker sig över årsskiftet)
    ttm_list = _ttm_windows(q_rows, need=6)
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik (exakt 4 senaste TTM-slutpunkter)
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        ps_vals = []
        for (d_end, ttm_rev_px) in ttm_list_px[:6]:  # räkna fler och ta 4 senaste icke-noll
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    ps_vals.append(float(mcap_hist / ttm_rev_px))
        # ta de 4 senaste icke-nollvärdena
        ps_vals = [v for v in ps_vals if v > 0][:4]
        for idx, val in enumerate(ps_vals, start=1):
            out[f"P/S Q{idx}"] = float(val)

    return out

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/sector/mcap
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","Direktavkastning (%)","Sektor","_marketCap_raw"):
        if y.get(k): out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = out.get("_marketCap_raw", 0.0) or info.get("marketCap") or 0.0
    try:
        mcap = float(mcap)
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
    # ta fler fönster för att inte tappa Dec/Jan
    ttm_list = _ttm_windows(q_rows, need=6)

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

    # P/S Q1–Q4 (historisk, 4 senaste icke-noll)
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        ps_vals = []
        for (d_end, ttm_rev_px) in ttm_list_px[:6]:
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    ps_vals.append((shares * p) / ttm_rev_px)
        ps_vals = [v for v in ps_vals if v > 0][:4]
        for idx, val in enumerate(ps_vals, start=1):
            out[f"P/S Q{idx}"] = float(val)

    return out

# ==================== Finnhub (valfritt, estimat) ============================

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

# ==================== Auto-fetch pipeline (per ticker) =======================

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

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
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
        need_any = ("Omsättning idag" not in vals) or ("Omsättning nästa år" not in vals)
        if need_any:
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
        # marketcap fallback
        if ("_marketCap_raw" not in vals):
            fmpl2 = hamta_fmp_falt_light(ticker)
            mcfb = fmpl2.get("_marketCap_raw")
            try:
                if mcfb and float(mcfb) > 0:
                    vals["_marketCap_raw"] = float(mcfb)
            except Exception:
                pass
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    return vals, debug

# app.py — Del 4/? — Batch-kö, progress (1/X), kurs-uppdatering, enkel-ticker auto & Kontroll-vy

# -------------------- Utils: marketcap-format --------------------------------

def format_mcap(num: float) -> str:
    try:
        v = float(num or 0.0)
    except Exception:
        v = 0.0
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v/1e12:.2f} T"
    if abs_v >= 1e9:
        return f"{v/1e9:.2f} B"
    if abs_v >= 1e6:
        return f"{v/1e6:.2f} M"
    if abs_v >= 1e3:
        return f"{v/1e3:.2f} K"
    return f"{v:.0f}"

# -------------------- Enbart kurs-uppdatering per ticker ---------------------

def update_price_only(df: pd.DataFrame, ticker: str, changes_map: dict) -> bool:
    """
    Uppdaterar endast 'Aktuell kurs' (och '_marketCap_raw' om möjligt) från Yahoo/FMP.
    TS_P/S etc rörs inte; endast 'Senast auto-uppdaterad' + källa sätts om något skrivs.
    """
    ticker = str(ticker).strip().upper()
    if not ticker:
        return False
    idxs = df.index[df["Ticker"].str.upper() == ticker]
    if len(idxs) == 0:
        return False
    ridx = int(idxs[0])

    vals = {}
    # Yahoo pris
    y = hamta_yahoo_fält(ticker)
    if y.get("Aktuell kurs", 0) > 0:
        vals["Aktuell kurs"] = float(y["Aktuell kurs"])
    if y.get("_marketCap_raw", 0) > 0:
        vals["_marketCap_raw"] = float(y["_marketCap_raw"])

    # Fallback FMP på pris/mcap om Yahoo gav 0
    if ("Aktuell kurs" not in vals) or ("_marketCap_raw" not in vals):
        fmpl = hamta_fmp_falt_light(ticker)
        if fmpl.get("Aktuell kurs", 0) > 0 and "Aktuell kurs" not in vals:
            vals["Aktuell kurs"] = float(fmpl["Aktuell kurs"])
        if fmpl.get("_marketCap_raw", 0) > 0 and "_marketCap_raw" not in vals:
            vals["_marketCap_raw"] = float(fmpl["_marketCap_raw"])

    if not vals:
        return False

    changed = apply_auto_updates_to_row(
        df, ridx, vals, source="Auto (pris)", changes_map=changes_map, always_stamp=True
    )
    return changed

# -------------------- Full auto per ticker -----------------------------------

def update_full_for_single_ticker(df: pd.DataFrame, ticker: str, changes_map: dict) -> tuple[bool, dict]:
    """
    Hämtar fulla fält via auto_fetch_for_ticker och skriver in.
    Stämplar TS_ även om värdet är oförändrat (always_stamp=True).
    """
    ticker = str(ticker).strip().upper()
    if not ticker:
        return False, {"err": "tom ticker"}
    idxs = df.index[df["Ticker"].str.upper() == ticker]
    if len(idxs) == 0:
        return False, {"err": f"{ticker} hittades inte i tabellen."}
    ridx = int(idxs[0])

    vals, debug = auto_fetch_for_ticker(ticker)
    if not vals:
        # Stämpla ändå auto-källa/datum för att visa körning skedd
        _note_auto_update(df, ridx, "Auto (full) – inga nya fält")
        return False, debug

    changed = apply_auto_updates_to_row(
        df, ridx, vals, source="Auto (full)", changes_map=changes_map, always_stamp=True
    )
    # Räkna om derivat (P/S-snitt, riktkurser etc)
    df.loc[[ridx]] = uppdatera_berakningar(df.loc[[ridx]].copy(), user_rates={})
    return changed, debug

# -------------------- Batch-kö: urval + progress (1/X) -----------------------

def _build_batch_candidates(df: pd.DataFrame, mode: str, size: int) -> list[str]:
    """
    mode:
      - 'A–Ö'  : sortera alfabetiskt på Bolagsnamn, ta första `size` tickers från position batch_cursor_AZ
      - 'Äldst': sortera på äldsta TS (bland spårade fält), ta `size` tickers från toppen
    Återanvänder st.session_state för att komma vidare i A–Ö utan att repetera samma grupp.
    """
    work = df.copy()
    # säkerställ kolumnen för äldsta TS
    work = add_oldest_ts_col(work)
    if mode == "A–Ö":
        work = work.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
        cur = int(st.session_state.get("batch_cursor_AZ", 0))
        end = min(cur + size, len(work))
        subs = work.iloc[cur:end]
        # uppdatera cursor om vi faktiskt får något
        if not subs.empty:
            st.session_state["batch_cursor_AZ"] = end if end < len(work) else 0  # wrap-around
        return [str(t).upper() for t in subs["Ticker"].astype(str).tolist()]
    else:
        # Äldst uppdaterade först
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
        subs = work.head(size)
        return [str(t).upper() for t in subs["Ticker"].astype(str).tolist()]

def run_batch_update(df: pd.DataFrame, mode: str, size: int, variant: str = "full") -> tuple[pd.DataFrame, dict]:
    """
    Kör batchuppdatering för `size` tickers enligt `mode`:
      mode in {"A–Ö", "Äldst"}
      variant in {"full", "pris"}
    Visar progressbar + text "i/X".
    Sparar ändringar endast om något faktiskt ändrats.
    """
    tickers = _build_batch_candidates(df, mode=mode, size=size)
    total = len(tickers)
    log = {"changed": {}, "debug": {}, "misses": []}

    if total == 0:
        st.info("Inga kandidater för batch just nu.")
        return df, log

    prog = st.progress(0.0, text=f"0/{total}")
    stat = st.empty()

    any_changed = False
    for i, tkr in enumerate(tickers, start=1):
        stat.write(f"Kör {i}/{total}: {tkr} ({variant})")
        try:
            if variant == "pris":
                changed = update_price_only(df, tkr, log["changed"])
                log["debug"][tkr] = {"variant": "pris", "changed": changed}
            else:
                chg, dbg = update_full_for_single_ticker(df, tkr, log["changed"])
                changed = chg
                log["debug"][tkr] = dbg
            any_changed = any_changed or changed
        except Exception as e:
            log["misses"].append(f"{tkr}: {e}")
        prog.progress(i/total, text=f"{i}/{total}")

    # Efter loop — räkna om hela df bara om något ändrats mycket; annars minimal
    if any_changed:
        df = uppdatera_berakningar(df, user_rates={})
        spara_data(df, do_snapshot=False)
        st.success("Batch klar – ändringar sparade.")
    else:
        st.info("Batch klar – inga faktiska ändringar upptäcktes.")
    return df, log

# -------------------- Kontroll-vy (fixar Too-Old & listor) -------------------

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
        missing_val = any((float(r.get(c, 0.0) or 0.0) <= 0.0) for c in need_cols)
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

    # 3) Snabb batch-körning (pris eller full) – med progress
    st.subheader("⚙️ Snabb batch-körning")
    colm = st.columns([1,1,1,1])
    with colm[0]:
        mode = st.selectbox("Urval", ["Äldst","A–Ö"], index=0)
    with colm[1]:
        size = st.number_input("Antal i batch", min_value=1, max_value=200, value=10, step=1)
    with colm[2]:
        variant = st.selectbox("Variant", ["full","pris"], index=0)
    with colm[3]:
        do_run = st.button("🚀 Kör batch nu")

    if do_run:
        df2, blog = run_batch_update(df.copy(), mode=mode, size=int(size), variant=variant)
        st.session_state["last_auto_log"] = blog
        st.success("Batch körd.")
        # Visa enkel summering
        with st.expander("Visa batchlogg"):
            st.write("**Ändringar (ticker → fält):**")
            st.json(blog.get("changed", {}))
            st.write("**Missar:**")
            st.json(blog.get("misses", []))
        # Uppdatera visuellt df (utan att tvinga rerun)
        for c in df.columns:
            df[c] = df2[c]

    st.divider()

    # 4) Senaste körlogg (om du nyss körde Auto/Batch)
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log")
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
            st.markdown("**Missar** (ticker → fel/miss)")
            if log.get("misses"):
                st.json(log["misses"])
            else:
                st.write("–")
        st.markdown("**Debug (urval)**")
        st.json(log.get("debug", {}))

# app.py — Del 5/? — Lägg till / uppdatera bolag (singel-uppdatering + bläddring)

def _etikett_ts(r: pd.Series) -> str:
    """Bygger en kompakt etikett om senast auto/manuell + källa."""
    auto = str(r.get("Senast auto-uppdaterad","") or "").strip()
    manu = str(r.get("Senast manuellt uppdaterad","") or "").strip()
    src  = str(r.get("Senast uppdaterad källa","") or "").strip()
    bits = []
    if auto: bits.append(f"Auto: {auto}")
    if manu: bits.append(f"Manuellt: {manu}")
    if src:  bits.append(f"Källa: {src}")
    return " | ".join(bits) if bits else "—"

def _ts_grid(r: pd.Series) -> pd.DataFrame:
    """Returnerar en liten DF med TS-kolumner för visning under formuläret."""
    rows = []
    for f, ts_col in TS_FIELDS.items():
        ts = str(r.get(ts_col,"") or "")
        rows.append({"Fält": f, "Uppdaterad (TS)": ts})
    return pd.DataFrame(rows)

def _ensure_nav_state(keys_count: int):
    if "edit_sort_mode" not in st.session_state:
        st.session_state.edit_sort_mode = "A–Ö (bolagsnamn)"
    if "edit_pos" not in st.session_state:
        st.session_state.edit_pos = 0
    # clamp index
    st.session_state.edit_pos = max(0, min(st.session_state.edit_pos, max(0, keys_count-1)))

def _sorted_keys_for_edit(df: pd.DataFrame, mode: str) -> list[str]:
    work = df.copy()
    work = add_oldest_ts_col(work)
    if mode.startswith("Äldst"):
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        work = work.sort_values(by=["Bolagsnamn","Ticker"])
    return [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in work.iterrows()]

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsläge & navigering
    sort_mode = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"],
                             index=0, key="edit_sort_mode")
    labels = _sorted_keys_for_edit(df, sort_mode)
    _ensure_nav_state(len(labels))

    # Synlig position + dropdown
    cols_top = st.columns([1,3,1,1,1])
    with cols_top[0]:
        st.write(f"Post {st.session_state.edit_pos+1}/{max(1,len(labels))}")
    with cols_top[1]:
        cur_label = labels[st.session_state.edit_pos] if labels else ""
        chosen = st.selectbox("Välj bolag", labels if labels else [""], index=st.session_state.edit_pos if labels else 0)
        if chosen != cur_label:
            st.session_state.edit_pos = labels.index(chosen) if chosen in labels else 0
    with cols_top[2]:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_pos = max(0, st.session_state.edit_pos - 1)
    with cols_top[3]:
        if st.button("➡️ Nästa"):
            st.session_state.edit_pos = min(max(0,len(labels)-1), st.session_state.edit_pos + 1)
    with cols_top[4]:
        # snabbknappar (pris/auto) kör mot vald rad
        pass  # knapparna kommer inne i formuläret (behöver fälten)

    # Hämta befintlig rad (om någon finns)
    bef = pd.Series({}, dtype=object)
    if labels:
        lab = labels[st.session_state.edit_pos]
        # labbformat: "Name (TICKER)"
        tkr = lab.split("(")[-1].rstrip(")") if "(" in lab else ""
        row = df[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
        if not row.empty:
            bef = row.iloc[0]

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, format="%.6f")
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, format="%.6f")
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, format="%.6f")
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, format="%.6f")
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, format="%.6f")
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            utd_arl   = st.number_input("Årlig utdelning (per aktie, i noteringsvaluta)", value=float(bef.get("Årlig utdelning",0.0)) if not bef.empty else 0.0)
            cagr5     = st.number_input("CAGR 5 år (%)", value=float(bef.get("CAGR 5 år (%)",0.0)) if not bef.empty else 0.0, step=0.1)

            st.caption("Vid spara uppdateras även basfält via Yahoo (namn/valuta/kurs/dividend/CAGR) samt omräkningar.")

        # Action-knappar
        col_btns = st.columns([1,1,1,2])
        with col_btns[0]:
            do_save = st.form_submit_button("💾 Spara")
        with col_btns[1]:
            do_price = st.form_submit_button("🔁 Uppdatera kurs")
        with col_btns[2]:
            do_full = st.form_submit_button("🤖 Full auto (endast detta bolag)")

    # Spara/uppdatera manuellt ifyllt
    if do_save and ticker:
        # bygg rad
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
            "Årlig utdelning": utd_arl, "CAGR 5 år (%)": cagr5
        }

        # sätt manuell TS om ngt av MANUELL_FALT_FOR_DATUM ändrats
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

        # skriv in i df (ny rad om ej finns)
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
            ridx = df.index[df["Ticker"]==ticker][0]
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            ridx = df.index[df["Ticker"]==ticker][0]

        # TS för manuell
        if datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Berika med Yahoo basfält (namn/valuta/kurs/div/CAGR)
        base = hamta_yahoo_fält(ticker)
        if base.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = base["Bolagsnamn"]
        if base.get("Valuta"):     df.loc[ridx, "Valuta"] = base["Valuta"]
        if base.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = base["Aktuell kurs"]
        if "Årlig utdelning" in base and base.get("Årlig utdelning") is not None:
            df.loc[ridx, "Årlig utdelning"] = float(base.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in base and base.get("CAGR 5 år (%)") is not None:
            df.loc[ridx, "CAGR 5 år (%)"] = float(base.get("CAGR 5 år (%)") or 0.0)

        # Omräkningar + spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # Singel-uppdatering: enbart kurs
    if do_price:
        if not ticker:
            st.error("Ange ticker först.")
        else:
            changes = {}
            ok = update_price_only(df, ticker, changes)
            if ok:
                ridx = df.index[df["Ticker"].str.upper()==ticker.upper()]
                if len(ridx):
                    _note_auto_update(df, int(ridx[0]), "Auto (pris)")
                spara_data(df, do_snapshot=False)
                st.success(f"Kurs uppdaterad för {ticker}.")
            else:
                st.info(f"Ingen förändring: {ticker} hittades inte eller inget pris kunde hämtas.")

    # Singel-uppdatering: full auto
    if do_full:
        if not ticker:
            st.error("Ange ticker först.")
        else:
            changes = {}
            changed, dbg = update_full_for_single_ticker(df, ticker, changes)
            if changed:
                spara_data(df, do_snapshot=False)
                st.success(f"Auto-uppdaterad: {ticker}.")
            else:
                st.info("Inga ändringar hittades vid auto-uppdatering.")
            with st.expander("Debug (senaste körningen)"):
                st.json(dbg)

    # Miniöversikt + etiketter
    st.divider()
    if not bef.empty:
        ridx = df.index[df["Ticker"].astype(str).str.upper()==str(bef.get("Ticker","")).upper()]
        r = df.iloc[int(ridx[0])] if len(ridx) else bef
        st.markdown(f"**{r.get('Bolagsnamn','')} ({r.get('Ticker','')})** — {_etikett_ts(r)}")
        # liten info-rad
        vvx = hamta_valutakurs(r.get("Valuta",""), user_rates)
        mk  = float(r.get("_marketCap_raw", 0.0) or 0.0)
        psn = float(r.get("P/S-snitt", 0.0) or 0.0)
        st.caption(f"Valuta: {r.get('Valuta','—')} | Kurs: {round(float(r.get('Aktuell kurs',0.0) or 0.0),2)} "
                   f"| Växelkurs→SEK: {round(vvx,4)} | Mcap: {format_mcap(mk)} | P/S-snitt: {round(psn,2)}")

        with st.expander("Tidsstämplar per fält"):
            st.dataframe(_ts_grid(r), use_container_width=True, hide_index=True)

    # Topplista: äldst uppdaterade (topp 10) som snabb kö
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

# app.py — Del 6/? — Scoring + Investeringsförslag

# -------------------- Risklabel & hjälpare -----------------------------------

def risklabel_from_mcap(mcap_raw: float) -> str:
    try:
        v = float(mcap_raw or 0.0)
    except Exception:
        v = 0.0
    if v >= 200e9:   return "Mega"
    if v >= 10e9:    return "Large"
    if v >= 2e9:     return "Mid"
    if v >= 300e6:   return "Small"
    return "Micro"

def _ps_avg_last4(row: pd.Series) -> float:
    vals = []
    for k in ("P/S Q1","P/S Q2","P/S Q3","P/S Q4"):
        try:
            v = float(row.get(k, 0.0) or 0.0)
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    return round(float(np.mean(vals)) if vals else 0.0, 2)

def _dividend_yield_pct(row: pd.Series) -> float:
    # baserat på årlig utdelning / aktuell kurs
    try:
        div = float(row.get("Årlig utdelning", 0.0) or 0.0)
        px  = float(row.get("Aktuell kurs", 0.0) or 0.0)
        if px > 0 and div >= 0:
            return round(div / px * 100.0, 2)
    except Exception:
        pass
    return 0.0

def _is_dividend_stock(row: pd.Series) -> bool:
    # betraktas som utdelningsbolag om utdelningsyield ≥ 1% eller Årlig utdelning > 0
    y = _dividend_yield_pct(row)
    return (y >= 1.0) or (float(row.get("Årlig utdelning", 0.0) or 0.0) > 0)

def _potential_pct(row: pd.Series, target_col: str = "Riktkurs om 1 år") -> float:
    try:
        tgt = float(row.get(target_col, 0.0) or 0.0)
        px  = float(row.get("Aktuell kurs", 0.0) or 0.0)
        if tgt > 0 and px > 0:
            return (tgt - px) / px * 100.0
    except Exception:
        pass
    return 0.0

def _ensure_nav_state_proposal(max_len: int):
    # separat state för investeringsförslag
    if "prop_idx" not in st.session_state:
        st.session_state.prop_idx = 0
    st.session_state.prop_idx = max(0, min(st.session_state.prop_idx, max(0, max_len-1)))

# -------------------- Scoring (enkel första version) -------------------------

def score_growth(row: pd.Series) -> float:
    """
    Tillväxtscore – väger potential mot värdering + momentum i P/S.
    Skala ungefär 0–100 (kan gå negativt om extremt dåligt).
    """
    pot = _potential_pct(row, "Riktkurs om 1 år")
    ps_now = float(row.get("P/S", 0.0) or 0.0)
    ps_avg = _ps_avg_last4(row)
    cagr = float(row.get("CAGR 5 år (%)", 0.0) or 0.0)

    # hög potential & CAGR upp, men straffa extrem PS
    base = 0.4*pot + 0.3*cagr
    if ps_now > 0 and ps_avg > 0:
        prem = (ps_avg - ps_now) / ps_avg * 100.0  # positivt om ps_now < ps_avg
        base += 0.2 * prem
    # clamp lite
    return round(max(-50.0, min(150.0, base)), 2)

def score_dividend(row: pd.Series) -> float:
    """
    Utdelningsscore – väger yield & hållbarhet mot värdering.
    Kräver att du senare fyller på fler fält (utdelningsandel FCF m.m.) – nu enkel.
    """
    y = _dividend_yield_pct(row)               # % yield
    ps_now = float(row.get("P/S", 0.0) or 0.0)
    ps_avg = _ps_avg_last4(row)

    base = 0.7 * y
    if ps_now > 0 and ps_avg > 0:
        prem = (ps_avg - ps_now) / ps_avg * 100.0
        base += 0.2 * prem
    # liten bonus om bolaget klassas som Large/Mega (lägre risk)
    mc = float(row.get("_marketCap_raw", 0.0) or 0.0)
    rk = risklabel_from_mcap(mc)
    base += {"Mega": 6, "Large": 4, "Mid": 2, "Small": 0, "Micro": -2}.get(rk, 0)

    return round(max(-30.0, min(120.0, base)), 2)

# -------------------- Investeringsförslag-vy ---------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Mode: Tillväxt/utdelning
    mode = st.radio("Fokus", ["Tillväxt", "Utdelning"], horizontal=True, index=0)

    # Riktkurs att mäta potential mot
    target_col = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=0
    )

    # Cap-filter & sektorfilter
    cap_choice = st.selectbox("Storleksklass", ["Alla","Micro","Small","Mid","Large","Mega"], index=0)
    all_sectors = sorted([s for s in df.get("Sektor", pd.Series([])).dropna().astype(str).unique() if s.strip()])
    sectors = st.multiselect("Filter sektor (valfritt)", options=all_sectors, default=[])

    # Basurval
    base = df.copy()
    # måste ha pris + riktkurskolumn
    base = base[(base["Aktuell kurs"] > 0) & (base[target_col] > 0)]
    if base.empty:
        st.info("Inga kandidater just nu.")
        return

    # cap label + filter
    base["_marketCap_raw"] = base.get("_marketCap_raw", 0.0).fillna(0.0).astype(float)
    base["Cap"] = base["_marketCap_raw"].apply(risklabel_from_mcap)
    if cap_choice != "Alla":
        base = base[base["Cap"] == cap_choice]

    # sektorfilter
    if sectors:
        base = base[base.get("Sektor","").isin(sectors)]

    # Dividend/growth-urval
    if mode == "Utdelning":
        base = base[base.apply(_is_dividend_stock, axis=1)]
    # beräkna nycklar
    base["P/S-snitt"] = base.apply(_ps_avg_last4, axis=1)
    base["Potential (%)"] = base.apply(lambda r: _potential_pct(r, target_col), axis=1)
    base["Direktavkastning (%)"] = base.apply(_dividend_yield_pct, axis=1)

    # score
    if mode == "Tillväxt":
        base["_score"] = base.apply(score_growth, axis=1)
    else:
        base["_score"] = base.apply(score_dividend, axis=1)

    # sortera efter score desc
    base = base.sort_values(by=["_score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # robust nav
    _ensure_nav_state_proposal(len(base))
    col_nav = st.columns([1,2,1,1])
    with col_nav[0]:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.prop_idx = max(0, st.session_state.prop_idx - 1)
    with col_nav[1]:
        st.write(f"Förslag {st.session_state.prop_idx+1}/{len(base)}")
    with col_nav[2]:
        if st.button("➡️ Nästa förslag"):
            st.session_state.prop_idx = min(len(base)-1, st.session_state.prop_idx + 1)
    with col_nav[3]:
        # inget särskilt här nu, plats för framtida knapp
        pass

    # aktuell rad
    try:
        rad = base.iloc[st.session_state.prop_idx]
    except Exception:
        st.session_state.prop_idx = 0
        rad = base.iloc[0]

    # valutakonvertering för visning
    vx = hamta_valutakurs(rad.get("Valuta",""), user_rates)
    kurs = float(rad.get("Aktuell kurs", 0.0) or 0.0)
    rikt = float(rad.get(target_col, 0.0) or 0.0)
    pot  = float(rad.get("Potential (%)", 0.0) or 0.0)
    mcap = float(rad.get("_marketCap_raw", 0.0) or 0.0)
    psn  = float(rad.get("P/S-snitt", 0.0) or 0.0)
    ps0  = float(rad.get("P/S", 0.0) or 0.0)
    shares_m = float(rad.get("Utestående aktier", 0.0) or 0.0)  # miljoner
    dy   = float(rad.get("Direktavkastning (%)", 0.0) or 0.0)

    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad.get('Ticker','')})")
    left, right = st.columns([1,1])
    with left:
        st.markdown(
            f"- **Aktuell kurs:** {round(kurs,2)} {rad.get('Valuta','')}"
            f"  (≈ {round(kurs*vx,2)} SEK)"
        )
        st.markdown(f"- **Riktkurs ({target_col}):** {round(rikt,2)} {rad.get('Valuta','')}")
        st.markdown(f"- **Uppsida:** {round(pot,2)} %")
        st.markdown(f"- **P/S (nu):** {round(ps0,2)}  |  **P/S-snitt (4):** {round(psn,2)}")
        st.markdown(f"- **Utestående aktier:** {round(shares_m,2)} M")
    with right:
        st.markdown(f"- **Market cap:** {format_mcap(mcap)}")
        st.markdown(f"- **Storleksklass:** {rad.get('Cap','—')}")
        st.markdown(f"- **Direktavkastning:** {round(dy,2)} %")
        st.markdown(f"- **Sektor:** {rad.get('Sektor','—')}")
        st.markdown(f"- **Score ({mode}):** **{round(float(rad.get('_score',0.0)),2)}**")

    # kvartalsvisa P/S och Mcap om de finns
    with st.expander("Historik: P/S (Q1–Q4) och MCap (Q1–Q4)"):
        cols = st.columns(2)
        with cols[0]:
            ps_rows = []
            for i in range(1,5):
                ps_rows.append([f"P/S Q{i}", round(float(rad.get(f"P/S Q{i}", 0.0) or 0.0), 3)])
            st.table(pd.DataFrame(ps_rows, columns=["Nyckel","Värde"]))
        with cols[1]:
            # Mcap historik om kolumner finns (vi använder samma Q-logik som tidigare kodbase)
            mcap_cols = [f"MCAP Q{i}" for i in range(1,5)]
            mcap_rows = []
            for c in mcap_cols:
                if c in base.columns or c in df.columns:
                    mcap_rows.append([c, format_mcap(float(rad.get(c, 0.0) or 0.0))])
            if mcap_rows:
                st.table(pd.DataFrame(mcap_rows, columns=["Nyckel","Värde"]))
            else:
                st.caption("Ingen sparad Mcap-historik.")

    # Slutlista – topp 10 som snabb översikt
    st.divider()
    st.markdown("### 🏁 Topp 10 enligt score")
    show_cols = ["Ticker","Bolagsnamn","Sektor","Cap","Aktuell kurs","P/S","P/S-snitt","Direktavkastning (%)","_score","_marketCap_raw",target_col,"Potential (%)"]
    show_cols = [c for c in show_cols if c in base.columns]
    tmp = base[show_cols].copy()
    if "_marketCap_raw" in tmp.columns:
        tmp["Market cap"] = tmp["_marketCap_raw"].apply(format_mcap)
        tmp = tmp.drop(columns=["_marketCap_raw"])
    st.dataframe(tmp.head(10), use_container_width=True, hide_index=True)

# app.py — Del 7/? — Sidopanel (valutakurser), Portfölj, Analys & main()

# -------------------- Portföljvy ---------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df.copy()
    port = port[pd.to_numeric(port.get("Antal aktier", 0.0), errors="coerce").fillna(0.0) > 0.0]
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Valuta"] = port.get("Valuta", "").astype(str)
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Aktuell kurs"] = pd.to_numeric(port.get("Aktuell kurs", 0.0), errors="coerce").fillna(0.0)
    port["Antal aktier"] = pd.to_numeric(port.get("Antal aktier", 0.0), errors="coerce").fillna(0.0)
    port["_marketCap_raw"] = pd.to_numeric(port.get("_marketCap_raw", 0.0), errors="coerce").fillna(0.0)

    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    if total_värde <= 0:
        st.info("Kunde inte beräkna portföljvärde.")
        return

    port["Andel (%)"] = (port["Värde (SEK)"] / total_värde * 100.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * pd.to_numeric(port.get("Årlig utdelning", 0.0), errors="coerce").fillna(0.0) * port["Växelkurs"]

    # summering
    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    # sektor & cap
    port["Sektor"] = port.get("Sektor", "").astype(str)
    port["_marketCap_raw"] = pd.to_numeric(port["_marketCap_raw"], errors="coerce").fillna(0.0)
    port["Cap"] = port["_marketCap_raw"].apply(risklabel_from_mcap)

    show_cols = ["Ticker","Bolagsnamn","Sektor","Cap","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    tmp = port[show_cols].copy()
    st.dataframe(tmp.sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    # sektorallokering
    with st.expander("Sektorfördelning"):
        grp = port.groupby("Sektor", dropna=False)["Värde (SEK)"].sum().reset_index()
        grp["Andel (%)"] = (grp["Värde (SEK)"] / grp["Värde (SEK)"].sum() * 100.0).round(2)
        st.dataframe(grp.sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

# -------------------- Analys-vy (enkel inspektör) ----------------------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if not etiketter:
        st.info("Inga bolag i databasen ännu.")
        return
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=len(etiketter)-1, value=st.session_state.analys_idx, step=1)
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
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","_marketCap_raw","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Direktavkastning (%)","Sektor",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    row_df = pd.DataFrame([r[cols].to_dict()])
    if "_marketCap_raw" in row_df.columns:
        row_df["Market cap (fmt)"] = row_df["_marketCap_raw"].apply(format_mcap)
    st.dataframe(row_df, use_container_width=True, hide_index=True)

# -------------------- Sidopanel: valutakurser (utan widget-krockar) ---------

def _init_rate_state():
    if "rate_usd" not in st.session_state: st.session_state.rate_usd = float(STANDARD_VALUTAKURSER.get("USD", 10.0))
    if "rate_nok" not in st.session_state: st.session_state.rate_nok = float(STANDARD_VALUTAKURSER.get("NOK", 1.0))
    if "rate_cad" not in st.session_state: st.session_state.rate_cad = float(STANDARD_VALUTAKURSER.get("CAD", 7.0))
    if "rate_eur" not in st.session_state: st.session_state.rate_eur = float(STANDARD_VALUTAKURSER.get("EUR", 11.0))

def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")

    # init state innan widgets skapas
    _init_rate_state()

    # visa inputs bundna till state
    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd), step=0.01, format="%.4f", key="rate_usd")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok), step=0.01, format="%.4f", key="rate_nok")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad), step=0.01, format="%.4f", key="rate_cad")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur), step=0.01, format="%.4f", key="rate_eur")

    col_rates1, col_rates2 = st.sidebar.columns(2)

    with col_rates1:
        if st.button("💾 Spara kurser", use_container_width=True):
            spara_valutakurser({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")

    with col_rates2:
        if st.button("↻ Läs sparade kurser", use_container_width=True):
            st.cache_data.clear()
            # ladda om cache till state utan att röra widgetkeys (värdena uppdateras ändå eftersom widgets är bundna till state)
            sr = las_sparade_valutakurser()
            st.session_state.rate_usd = float(sr.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(sr.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(sr.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(sr.get("EUR", st.session_state.rate_eur))
            st.sidebar.info("Sparade kurser inlästa.")

    # Auto-hämtning
    if st.sidebar.button("🌐 Hämta kurser automatiskt", use_container_width=True):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.sidebar.success(f"Valutakurser (källa: {provider}) hämtade.")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        # uppdatera state (widgets reflekterar detta eftersom de är bundna)
        st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets", use_container_width=True):
        st.cache_data.clear()
        st.session_state["force_reload_sheet"] = True  # plockas upp i main()

    # returnera rates från state
    return {"USD": float(st.session_state.rate_usd), "NOK": float(st.session_state.rate_nok),
            "CAD": float(st.session_state.rate_cad), "EUR": float(st.session_state.rate_eur), "SEK": 1.0}

# -------------------- MAIN ---------------------------------------------------

def main():
    st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: valutakurser (inkl. auto/läs-sparade utan widget-krock)
    user_rates = _sidebar_rates()

    # Läs data (möjlig tvingad omläsning)
    if st.session_state.get("force_reload_sheet"):
        st.session_state["force_reload_sheet"] = False
        st.cache_data.clear()
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    # Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Auto-uppdatering i sidopanel – batch
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Auto-uppdatering")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=False)
    # snabb batch-körning flyttad till Kontroll-vy (med progress 1/X)
    # men lämna knappar här om man vill köra ALLA (tungt) – valfritt, låter bli för att undvika “tung körning”-problemet.

    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # om df2 har fler/samma kolumner – synka i minnet
        if set(df2.columns) == set(df.columns) and len(df2) == len(df):
            for c in df.columns:
                df[c] = df2[c]
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
