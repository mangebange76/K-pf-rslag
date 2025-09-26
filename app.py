# app.py — Del 1/? — Imports, Sheets, valutakurser, schema & TS-hjälp
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

# --- Streamlit: säker omstart (för att slippa experimental_rerun-problem) ---
def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            st.warning("Kunde inte trigga omstart automatiskt. Ladda om sidan manuellt.")

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

# Spårade fält → respektive TS-kolumn (uppdateras när fältet ändras auto/manuellt)
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

# Slutlig kolumnlista i databasen (inkl. extra nyckeltal & metadata)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Extra nyckeltal / metadata
    "_marketCap_raw", "Risklabel",
    "EV/EBITDA (TTM)", "P/E (TTM)", "FCF (TTM)", "FCF-yield (%)",
    "Direktavkastning (%)", "Total Debt", "Total Cash",
    "Sektor", "Industri", "GAV (SEK)",

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
            if any(x in kol.lower() for x in
                   ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","yield","debt","cash","ebitda","p/e","mcap","gav","_marketcap"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sektor","Industri","Risklabel","Valuta","Bolagsnamn","Ticker"):
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
        "_marketCap_raw", "EV/EBITDA (TTM)", "P/E (TTM)", "FCF (TTM)", "FCF-yield (%)",
        "Direktavkastning (%)", "Total Debt", "Total Cash", "GAV (SEK)"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Sektor","Industri","Risklabel",
              "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]:
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

# app.py — Del 2/? — Yahoo-hjälpare, beräkningar & uppdateringsskrivning

# --- Yahoo-hjälpare ----------------------------------------------------------

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
    """Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR."""
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "_marketCap_raw": 0.0,
        "Sektor": "",
        "Industri": "",
        "Direktavkastning (%)": 0.0,
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

        dy = info.get("dividendYield")
        try:
            if dy is not None:
                out["Direktavkastning (%)"] = float(dy) * 100.0 if dy < 1.0 else float(dy)
        except Exception:
            pass

        mc = info.get("marketCap")
        try:
            if mc is not None:
                out["_marketCap_raw"] = float(mc)
        except Exception:
            pass

        sector = info.get("sector")
        if sector: out["Sektor"] = str(sector)

        industry = info.get("industry")
        if industry: out["Industri"] = str(industry)

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# --- Visningshjälp -----------------------------------------------------------

def format_large(x: float) -> str:
    """Formatera stora tal i USD (enhet)."""
    try:
        v = float(x)
    except Exception:
        return "-"
    if v >= 1_000_000_000_000:
        return f"{v/1_000_000_000_000:.2f} T"
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f} B"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f} M"
    return f"{v:,.0f}"

def risklabel_from_mcap(mcap: float) -> str:
    try:
        v = float(mcap or 0.0)
    except Exception:
        v = 0.0
    if v >= 200_000_000_000: return "Mega"
    if v >= 10_000_000_000:  return "Large"
    if v >= 2_000_000_000:   return "Mid"
    if v >= 300_000_000:     return "Small"
    return "Micro"

# --- Beräkningar -------------------------------------------------------------

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

        # Risklabel auto om mcap finns
        if "_marketCap_raw" in df.columns and float(rad.get("_marketCap_raw", 0.0)) > 0:
            df.at[i, "Risklabel"] = risklabel_from_mcap(float(rad.get("_marketCap_raw", 0.0)))

    return df

# --- Uppdateringsskrivning (med always_stamp för batch-rotation) -------------

def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    always_stamp: bool = False
) -> bool:
    """
    Skriver endast fält som får ett nytt (meningsfullt) värde.
    Stämplar TS_ för fält som ändras. Om always_stamp=True stämplas även TS + 'Senast auto-uppdaterad'
    även om inget värde ändras (så att batch 'Äldst först' roterar korrekt).
    Returnerar True om något faktiskt fält ändrades (värdebyte) — TS-only räknas inte som ändring.
    """
    changed_fields = []
    wrote_any_value = False

    for f, v in (new_vals or {}).items():
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
            wrote_any_value = True
            changed_fields.append(f)
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)

    # Alltid stämpla vid always_stamp, även om inga faktiska fält byttes
    if wrote_any_value or always_stamp:
        # Bas-källa + datum
        _note_auto_update(df, row_idx, source)
        if always_stamp:
            # Stämpla TS för centrala fält om de finns i new_vals eller är kärnfält
            core_fields = ["P/S","Utestående aktier","Omsättning idag","Omsättning nästa år","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]
            for f in core_fields:
                if f in TS_FIELDS and (f in (new_vals or {}) or f in ["P/S","Utestående aktier","Omsättning idag","Omsättning nästa år"]):
                    _stamp_ts_for_field(df, row_idx, f)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)

    return wrote_any_value

# app.py — Del 3/? — Datakällor: SEC+Yahoo, Yahoo global, FMP, Finnhub
# Inkl. fix för Q4 (dec/jan) via sammanslagning SEC + Yahoo kvartal

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

    # Nyckeltal (om finns)
    if isinstance(rttm, list) and rttm:
        try:
            ev_e = rttm[0].get("enterpriseToEbitdaTTM")
            if ev_e and float(ev_e) > 0:
                out["EV/EBITDA (TTM)"] = float(ev_e)
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

# ---------- ISO & instant helpers -------------------------------------------
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
    Hämtar alla 'instant' aktieposter från dei/us-gaap/ifrs-full.
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
    Summerar multi-class per senaste 'end' (instant). Om flera 'end' finns,
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
                        # kvartal ska vara ~90 dagar → filtrera bort helår etc.
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # Deduplicera på end-date
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

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

def _ttm_windows(values: list, need: int = 6) -> list:
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
    try:
        if mcap is None:
            mcap = _yfi_get(t, "market_cap", "marketCap")
        if price is None:
            price = _yfi_get(t, "last_price", "regularMarketPrice")
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

    # 2) fallback: income_stmt quarterly (ibland samma sak)
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

def _merge_quarter_rows(sec_rows: list, y_rows: list, prefer="sec") -> list:
    """
    Sammanfoga SEC- och Yahoo-kvartal, dedupe per end-date (SEC prioritet).
    Returnerar nyast→äldst.
    """
    dd = {}
    if prefer == "sec":
        for d,v in sec_rows:
            dd[d] = ("sec", float(v))
        for d,v in y_rows:
            dd.setdefault(d, ("y", float(v)))
    else:
        for d,v in y_rows:
            dd[d] = ("y", float(v))
        for d,v in sec_rows:
            dd.setdefault(d, ("sec", float(v)))
    out = sorted([(d, val) for d,(_,val) in dd.items()], key=lambda t: t[0], reverse=True)
    return out

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
    Fyller luckor med Yahoo-kvartal (fix för dec/jan-Q4).
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "_marketCap_raw", "Sektor", "Industri", "Direktavkastning (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()

    # EV/EBITDA, P/E, FCF, Debt/Cash via Yahoo info
    try:
        t = yf.Ticker(ticker)
        info = _yfi_info_dict(t)
        if "enterpriseToEbitda" in info and info["enterpriseToEbitda"]:
            out["EV/EBITDA (TTM)"] = float(info["enterpriseToEbitda"])
        if "trailingPE" in info and info["trailingPE"]:
            out["P/E (TTM)"] = float(info["trailingPE"])
        if "freeCashflow" in info and info["freeCashflow"]:
            out["FCF (TTM)"] = float(info["freeCashflow"])
        if "totalDebt" in info and info["totalDebt"]:
            out["Total Debt"] = float(info["totalDebt"])
        if "totalCash" in info and info["totalCash"]:
            out["Total Cash"] = float(info["totalCash"])
        # FCF-yield (mot market cap)
        mc = float(out.get("_marketCap_raw", 0.0))
        fcf = float(out.get("FCF (TTM)", 0.0))
        if mc > 0 and fcf != 0:
            out["FCF-yield (%)"] = round((fcf / mc) * 100.0, 2)
    except Exception:
        pass

    # Shares: implied → fallback SEC robust
    implied = _implied_shares_from_yahoo(ticker, price=out.get("Aktuell kurs"), mcap=None)
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
    mcap_now = _yfi_get(yf.Ticker(ticker), "market_cap", "marketCap")
    try:
        mcap_now = float(mcap_now or 0.0)
    except Exception:
        mcap_now = 0.0
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used
    if mcap_now > 0:
        out["_marketCap_raw"] = mcap_now

    # SEC kvartalsintäkter + unit → komplettera med Yahoo kvartal för att få Q4 (dec/jan)
    sec_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    y_rows = _yfi_quarterly_revenues(yf.Ticker(ticker))
    # valuta-konvertering
    conv_sec = 1.0
    if sec_rows and rev_unit and rev_unit.upper() != px_ccy:
        conv_sec = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    # Yahoo financialCurrency
    fin_ccy = ( _yfi_info_dict(yf.Ticker(ticker)).get("financialCurrency") or px_ccy ).upper()
    conv_y = _fx_rate_cached(fin_ccy, px_ccy) if fin_ccy != px_ccy else 1.0

    sec_rows_px = [(d, v * conv_sec) for (d, v) in sec_rows]
    y_rows_px = [(d, v * conv_y) for (d, v) in y_rows]
    merged = _merge_quarter_rows(sec_rows_px, y_rows_px, prefer="sec")

    # TTM-fönster (ta minst 4, gärna 6)
    ttm_list = _ttm_windows(merged, need=6)

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list:
        ltm_now = ttm_list[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik (ttm-baserat) med historiska priser
    if shares_used > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)

    return out

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik. Hämtar nyckeltal via Yahoo.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/sector/industry/mcap/divyield
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","_marketCap_raw","Sektor","Industri","Direktavkastning (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(info.get("marketCap") or out.get("_marketCap_raw") or 0.0)

    # EV/EBITDA, P/E, FCF, Debt/Cash
    try:
        if "enterpriseToEbitda" in info and info["enterpriseToEbitda"]:
            out["EV/EBITDA (TTM)"] = float(info["enterpriseToEbitda"])
        if "trailingPE" in info and info["trailingPE"]:
            out["P/E (TTM)"] = float(info["trailingPE"])
        if "freeCashflow" in info and info["freeCashflow"]:
            out["FCF (TTM)"] = float(info["freeCashflow"])
        if "totalDebt" in info and info["totalDebt"]:
            out["Total Debt"] = float(info["totalDebt"])
        if "totalCash" in info and info["totalCash"]:
            out["Total Cash"] = float(info["totalCash"])
        mc = float(out.get("_marketCap_raw", mcap) or 0.0)
        fcf = float(out.get("FCF (TTM)", 0.0))
        if mc > 0 and fcf != 0:
            out["FCF-yield (%)"] = round((fcf / mc) * 100.0, 2)
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

    # Kvartalsintäkter → TTM (Yahoo) och P/S
    q_rows = _yfi_quarterly_revenues(t)
    if q_rows and len(q_rows) >= 4:
        # Valutakonvertering om financialCurrency != prisvaluta
        fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
        conv = _fx_rate_cached(fin_ccy, px_ccy) if fin_ccy != px_ccy else 1.0
        q_rows_px = [(d, v*conv) for (d,v) in q_rows]
        ttm_list = _ttm_windows(q_rows_px, need=6)

        # Market cap (nu)
        if mcap <= 0 and shares > 0 and px > 0:
            mcap = shares * px
            out["_marketCap_raw"] = mcap

        # P/S (TTM) nu
        if mcap > 0 and ttm_list:
            ltm_now = ttm_list[0][1]
            if ltm_now > 0:
                out["P/S"] = mcap / ltm_now

        # P/S Q1–Q4 (historisk, TTM)
        if shares > 0 and ttm_list:
            q_dates = [d for (d, _) in ttm_list[:4]]
            px_map = _yahoo_prices_for_dates(ticker, q_dates)
            for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
                if ttm_rev_px and ttm_rev_px > 0:
                    p = px_map.get(d_end)
                    if p and p > 0:
                        out[f"P/S Q{idx}"] = (shares * p) / ttm_rev_px

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

# =============== Auto-pipeline (single) ======================================

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares + kvartal merge SEC/Yahoo) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S/EV/EBITDA) om saknas
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","_debug_shares_source",
            "_marketCap_raw","EV/EBITDA (TTM)","P/E (TTM)","FCF (TTM)","FCF-yield (%)",
            "Total Debt","Total Cash","Sektor","Industri","Direktavkastning (%)"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","_marketCap_raw","Sektor","Industri","Direktavkastning (%)",
                  "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                  "EV/EBITDA (TTM)","P/E (TTM)","FCF (TTM)","FCF-yield (%)","Total Debt","Total Cash"]:
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
        if ("P/S" not in vals) or ("EV/EBITDA (TTM)" not in vals):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier"),
                                  "_marketCap_raw": fmpl.get("_marketCap_raw"), "EV/EBITDA (TTM)": fmpl.get("EV/EBITDA (TTM)")}
            for k in ["P/S","EV/EBITDA (TTM)","_marketCap_raw"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
            if ("Utestående aktier" not in vals) and (fmpl.get("Utestående aktier") not in (None, "", 0, 0.0)):
                vals["Utestående aktier"] = fmpl["Utestående aktier"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    return vals, debug

# app.py — Del 4/? — Snapshots, batch-kö, auto-uppdatering & Kontroll-vy

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

# --- TS-analys ---------------------------------------------------------------

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

# --- Behovslistor ------------------------------------------------------------

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

def build_forecast_needs_df(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Lista bolag där prognosfälten (Omsättning idag/nästa år) saknar TS eller är äldst.
    Sorterar på äldsta av de två TS-kolumnerna.
    """
    df2 = df.copy()
    df2["TS_oms_idag_dt"] = pd.to_datetime(df2.get("TS_Omsättning idag", ""), errors="coerce")
    df2["TS_oms_next_dt"] = pd.to_datetime(df2.get("TS_Omsättning nästa år", ""), errors="coerce")
    df2["TS_oms_both_oldest"] = df2[["TS_oms_idag_dt","TS_oms_next_dt"]].min(axis=1)
    need_mask = df2["TS_oms_both_oldest"].isna()
    # Ta även med de som har datum men är äldst
    df2 = df2.sort_values(by=["TS_oms_both_oldest"], ascending=True)
    vis = df2[need_mask | True].head(top_n)  # visar top_n äldsta oavsett
    out = vis[["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år"]].copy()
    out = out.rename(columns={
        "TS_Omsättning idag":"TS Oms. idag",
        "TS_Omsättning nästa år":"TS Oms. nästa år"
    })
    return out

# --- Batch-kö ---------------------------------------------------------------

def _build_batch_queue(df: pd.DataFrame, mode: str, size: int) -> list:
    """
    Bygger en lista av tickers att köra, med läge:
    - "Äldst uppdaterade först (alla fält)"
    - "A–Ö (bolagsnamn)"
    Skipper rader vars äldsta TS är 'idag' för att undvika att samma 20 väljs igen samma dag.
    """
    if df.empty:
        return []
    mode = str(mode or "")
    if mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        today = now_stamp()
        # Filtrera bort allt som redan stämplats idag (vilken som helst TS-kolumn)
        # Vi använder _oldest_any_ts för “gammalhets”-mått; om det är idag, hoppa över.
        if "_oldest_any_ts" not in work.columns:
            work = add_oldest_ts_col(work)
        # keep rows där oldest TS inte är idag (eller NaT = 2099 i fill)
        mask = ~work["_oldest_any_ts"].astype(str).str.startswith(today)
        work = work[mask]
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"], ascending=[True, True])
    else:
        work = df.sort_values(by=["Bolagsnamn","Ticker"])

    tickers = list(work["Ticker"].astype(str))
    if size and size > 0:
        tickers = tickers[:size]
    return tickers

# --- Auto-uppdatering (batch) -----------------------------------------------

def auto_update_batch(df: pd.DataFrame, user_rates: dict, tickers: list, make_snapshot: bool = False, always_stamp: bool = True):
    """
    Kör auto-uppdatering för en lista av tickers. Skriver endast fält med meningsfulla nya värden,
    men om always_stamp=True stämplas TS + 'Senast auto-uppdaterad' även om inga fält ändras.
    """
    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    total = len(tickers)
    if total == 0:
        st.sidebar.info("Ingen ticker i kön.")
        return df, log

    # Sidopanel: progress
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()

    any_changed = False
    for i, tkr in enumerate(tickers):
        status.write(f"Uppdaterar {i+1}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            # hitta radindex
            mask = (df["Ticker"].astype(str).str.upper() == str(tkr).upper())
            if not mask.any():
                log["misses"][tkr] = ["hittades inte i tabellen"]
            else:
                ridx = df.index[mask][0]
                changed = apply_auto_updates_to_row(
                    df, ridx, new_vals,
                    source="Auto (SEC/Yahoo→Yahoo→Finnhub→FMP)",
                    changes_map=log["changed"],
                    always_stamp=always_stamp
                )
                any_changed = any_changed or changed
                if not changed and not new_vals:
                    log["misses"][tkr] = ["(inga nya fält)"]
            if i < 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(total,1))

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    if any_changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success(f"Klart! Ändringar sparade. ({total} körda)")
    else:
        # även om inga värden byttes kan TS ha stämplats – men vi undviker onödig skrivning
        st.sidebar.info(f"Klar. Inga faktiska värdeändringar upptäcktes. ({total} körda)")

    return df, log

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False):
    """
    Legacy: uppdatera alla tickers (A–Ö). Använd hellre batch-funktionen i UI.
    """
    tickers = list(df["Ticker"].astype(str))
    return auto_update_batch(df, user_rates, tickers, make_snapshot=make_snapshot, always_stamp=True)

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

    # 3) Prognosfält att se över (Omsättning idag/nästa år)
    st.subheader("📅 Prognosfält att se över (Omsättning idag / nästa år)")
    fore = build_forecast_needs_df(df, top_n=30)
    st.dataframe(fore, use_container_width=True, hide_index=True)

    st.divider()

    # 4) Senaste körlogg (om du nyss körde Auto)
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

# app.py — Del 5/? — Scoring, etiketter & Investeringsförslag

# --- Småhjälpare -------------------------------------------------------------

def _nz(v, dflt=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return dflt
        return float(v)
    except Exception:
        return dflt

def _row_mcap(row: pd.Series) -> float:
    """Market cap i basvaluta (vanligen USD)."""
    mc = _nz(row.get("_marketCap_raw"), 0.0)
    if mc > 0:
        return mc
    px = _nz(row.get("Aktuell kurs"), 0.0)
    sh_m = _nz(row.get("Utestående aktier"), 0.0)  # i miljoner
    if px > 0 and sh_m > 0:
        return px * (sh_m * 1_000_000.0)
    return 0.0

def _dividend_yield_pct(row: pd.Series) -> float:
    dy = _nz(row.get("Direktavkastning (%)"), 0.0)
    if dy > 0:
        return dy
    # fallback från årslön / pris
    utd = _nz(row.get("Årlig utdelning"), 0.0)
    px  = _nz(row.get("Aktuell kurs"), 0.0)
    if utd > 0 and px > 0:
        return (utd / px) * 100.0
    return 0.0

# --- Övervärderingsflaggor ---------------------------------------------------

def _overvaluation_flags(row: pd.Series, target_col: str) -> list:
    flags = []
    pot = _nz(row.get("Potential (%)"), 0.0)
    if pot < -10:
        flags.append("Negativ potential >10%")
    ps = _nz(row.get("P/S"), 0.0); psn = _nz(row.get("P/S-snitt"), 0.0)
    if psn > 0 and ps/psn >= 1.6:
        flags.append("P/S >> historiskt snitt")
    ev = _nz(row.get("EV/EBITDA (TTM)"), 0.0)
    if ev >= 25:
        flags.append("Hög EV/EBITDA")
    fcfy = _nz(row.get("FCF-yield (%)"), 0.0)
    if fcfy < 0:
        flags.append("Negativ FCF-yield")
    # Utdelningsrisk
    fcf = _nz(row.get("FCF (TTM)"), 0.0)
    utd = _nz(row.get("Årlig utdelning"), 0.0)
    if fcf != 0 and utd > 0:
        shares = _nz(row.get("Utestående aktier"), 0.0) * 1_000_000.0
        if shares > 0:
            payout_fcf = (utd * shares) / fcf
            if payout_fcf > 0.9:
                flags.append("Payout >90% av FCF")
            if payout_fcf > 1.2:
                flags.append("Utd. över FCF")
    return flags

# --- Score-etiketter & värderingssignal -------------------------------------

def score_to_label(score: float, segment: str) -> str:
    s = float(score or 0.0)
    if s >= 85: return "🟢 Mycket bra – köp"
    if s >= 70: return "✅ Bra"
    if s >= 55: return "⚪ Fair – behåll"
    if s >= 40: return "🟠 Något övervärderad – trimma"
    if s >= 25: return "🟡 Övervärderad – trimma/sälj"
    return "🔴 Akta dig – sälj"

def valuation_label_and_flags(row: pd.Series, target_col: str) -> tuple[str, list]:
    flags = _overvaluation_flags(row, target_col)
    pot   = _nz(row.get("Potential (%)"), 0.0)
    ps    = _nz(row.get("P/S"), 0.0)
    psn   = _nz(row.get("P/S-snitt"), 0.0)
    ev    = _nz(row.get("EV/EBITDA (TTM)"), 0.0)
    fcfy  = _nz(row.get("FCF-yield (%)"), 0.0)

    score = 0
    if pot >= 20: score += 2
    elif pot >= 5: score += 1
    elif pot <= -15: score -= 2
    elif pot < -5: score -= 1

    if psn > 0 and ps > 0:
        ratio = ps/psn
        if ratio <= 0.90: score += 1
        elif ratio >= 1.60: score -= 2
        elif ratio >= 1.25: score -= 1

    if ev > 0:
        if ev < 12: score += 1
        elif ev > 25: score -= 2
        elif ev > 20: score -= 1

    if fcfy >= 6: score += 1
    elif fcfy < 0: score -= 1

    if score >= 3:   label = "🟢 Billig / attraktiv"
    elif score >= 1: label = "✅ OK / rimlig"
    elif score >= 0: label = "⚪ Fair – behåll"
    elif score >= -2:label = "🟠 Något övervärderad – trimma"
    elif score >= -4:label = "🟡 Övervärderad – trimma/sälj"
    else:            label = "🔴 Akta dig – sälj"

    return label, flags

# --- Growth-score & Dividend-score ------------------------------------------

def compute_growth_score(row: pd.Series, target_col: str) -> float:
    s = 50.0  # bas
    pot = _nz(row.get("Potential (%)"), 0.0)
    cagr = _nz(row.get("CAGR 5 år (%)"), 0.0)
    ps = _nz(row.get("P/S"), 0.0); psn = _nz(row.get("P/S-snitt"), 0.0)
    ev = _nz(row.get("EV/EBITDA (TTM)"), 0.0)
    fcfy = _nz(row.get("FCF-yield (%)"), 0.0)
    risk = (row.get("Risklabel") or "").lower()

    # Potential
    s += np.clip(pot/2.0, -20, 20)  # +10 vid +20% potential, -10 vid -20%

    # Tillväxt
    s += np.clip(cagr/2.0, -10, 20) # 40% cagr -> +20p

    # Värdering mot historik
    if psn > 0 and ps > 0:
        ratio = ps/psn
        if ratio <= 0.8:  s += 6
        elif ratio <= 1.0:s += 3
        elif ratio >= 1.8:s -= 10
        elif ratio >= 1.3:s -= 5

    # EV/EBITDA
    if ev > 0:
        if ev < 12: s += 6
        elif ev < 18: s += 2
        elif ev > 30: s -= 8
        elif ev > 22: s -= 4

    # FCF-yield
    if fcfy > 0: s += min(fcfy/2.0, 8)   # t.ex. 10% -> +5
    else:        s -= 6

    # Risk (storlek)
    if "micro" in risk: s -= 8
    elif "small" in risk: s -= 3
    elif "mega" in risk: s += 1

    return float(np.clip(s, 0, 100))

def compute_dividend_score(row: pd.Series) -> float:
    s = 50.0
    dy = _dividend_yield_pct(row)
    fcf = _nz(row.get("FCF (TTM)"), 0.0)
    mc  = _row_mcap(row)
    debt = _nz(row.get("Total Debt"), 0.0)
    cash = _nz(row.get("Total Cash"), 0.0)
    risk = (row.get("Risklabel") or "").lower()

    # Yield – stark drivare
    if dy >= 8: s += 20
    elif dy >= 6: s += 14
    elif dy >= 4: s += 8
    elif dy >= 3: s += 4
    elif dy >= 2: s += 2
    elif dy < 1: s -= 6

    # Payout mot FCF
    utd = _nz(row.get("Årlig utdelning"), 0.0)
    shares = _nz(row.get("Utestående aktier"), 0.0) * 1_000_000.0
    payout_fcf = None
    if fcf != 0 and utd > 0 and shares > 0:
        payout_fcf = (utd * shares) / fcf
        if payout_fcf < 0.5:  s += 10
        elif payout_fcf <= 0.8: s += 5
        elif payout_fcf <= 1.0: s -= 4
        elif payout_fcf <= 1.3: s -= 10
        else: s -= 16
    elif fcf <= 0 and utd > 0:
        s -= 14  # betalar utdelning utan FCF

    # Finansiell buffert
    if cash > debt: s += 6
    elif debt > 0 and cash/debt < 0.3: s -= 8

    # Storleksrisk
    if "micro" in risk: s -= 10
    elif "small" in risk: s -= 4
    elif "mega" in risk: s += 2

    # Liten kvalitetsbonus vid positiv FCF-yield
    fcfy = _nz(row.get("FCF-yield (%)"), 0.0)
    if fcfy > 4: s += 4
    elif fcfy < 0: s -= 6

    return float(np.clip(s, 0, 100))

# --- Investeringsförslag -----------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Valt kapital (SEK) och segment (Tillväxt/Utdelning)
    col0, col1, col2 = st.columns([1,1,1])
    with col0:
        kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    with col1:
        segment = st.selectbox("Segment", ["Tillväxt","Utdelning"], index=0)
    with col2:
        riktkurs_val = st.selectbox(
            "Vilken riktkurs ska användas?",
            ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
            index=1
        )

    # Filter
    colf1, colf2, colf3 = st.columns([1,1,2])
    with colf1:
        risk_filter = st.selectbox("Kap-segment", ["Alla","Micro","Small","Mid","Large","Mega"], index=0)
    with colf2:
        sort_mode = st.selectbox("Sortering", ["Högst score","Störst potential","Närmast riktkurs"], index=0)
    with colf3:
        sectors = sorted([s for s in df.get("Sektor", pd.Series(dtype=str)).astype(str).unique() if s])
        sector_pick = st.multiselect("Sektorfilter (valfritt)", sectors, default=[])

    # Basurval
    base = df.copy()
    # risklabel
    if risk_filter != "Alla":
        base = base[base["Risklabel"].astype(str).str.lower() == risk_filter.lower()]
    # sektor
    if sector_pick:
        base = base[base["Sektor"].isin(sector_pick)]
    # målfält >0 + pris >0
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar filter just nu.")
        return

    # Potential
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0

    # Risklabel (om saknas) från mcap
    if "_marketCap_raw" in base.columns:
        base["Risklabel"] = base.apply(lambda r: r.get("Risklabel") if str(r.get("Risklabel","")).strip()
                                       else risklabel_from_mcap(_row_mcap(r)), axis=1)

    # Scores
    if segment == "Tillväxt":
        base["Score"] = base.apply(lambda r: compute_growth_score(r, riktkurs_val), axis=1)
    else:
        base["Score"] = base.apply(lambda r: compute_dividend_score(r), axis=1)

    # Etiketter
    base["Score-etikett"] = base.apply(lambda r: score_to_label(_nz(r.get("Score")), segment), axis=1)
    _val_pairs = base.apply(lambda r: valuation_label_and_flags(r, riktkurs_val), axis=1)
    base["Värdering"] = [p[0] for p in _val_pairs]
    base["_val_flags"] = [p[1] for p in _val_pairs]

    # Sortering
    if sort_mode == "Högst score":
        base = base.sort_values(by=["Score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)
    elif sort_mode == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = (base["Aktuell kurs"] - base[riktkurs_val]).abs() / base[riktkurs_val].replace(0, np.nan)
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Topplista vid högst score
    if sort_mode == "Högst score":
        with st.expander("🏆 Topplista (Högst score)"):
            cols_show = ["Ticker","Bolagsnamn","Score","Score-etikett","Värdering",
                         "Potential (%)","P/S","P/S-snitt","EV/EBITDA (TTM)",
                         "FCF-yield (%)","Direktavkastning (%)","Risklabel","Sektor"]
            cols_exist = [c for c in cols_show if c in base.columns]
            st.dataframe(base[cols_exist].head(25), use_container_width=True, hide_index=True)

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

    # Rad att visa
    rad = base.iloc[st.session_state.forslags_index]

    # Växelkurs & köpberäkning
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    # Portföljandel före/efter (om du har portfölj redan)
    port = df[df["Antal aktier"] > 0].copy()
    port_värde = 0.0
    nuv_innehav = 0.0
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
        r2 = port[port["Ticker"] == rad["Ticker"]]
        if not r2.empty:
            nuv_innehav = float(r2["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Visning
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    score_label = score_to_label(_nz(rad.get("Score")), segment)
    val_label, val_flags = valuation_label_and_flags(rad, riktkurs_val)
    mcap_now = _row_mcap(rad)

    # huvudrad-info
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Utestående aktier:** {round(_nz(rad.get('Utestående aktier')),2)} M",
        f"- **Nuvarande market cap:** {format_large(mcap_now)}",
        f"- **P/S (nu):** {round(_nz(rad.get('P/S')),2)}  |  **P/S-snitt (Q1–Q4):** {round(_nz(rad.get('P/S-snitt')),2)}",
        f"- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Score:** {round(_nz(rad.get('Score')),1)}  |  **Score-etikett:** {score_label}",
        f"- **Värdering:** {val_label}",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %  →  **Efter köp:** {ny_andel} %",
    ]
    if segment == "Utdelning":
        lines.insert(1, f"- **Direktavkastning (nu):** {round(_dividend_yield_pct(rad),2)} %")

    st.markdown("\n".join(lines))

    # Expanders
    with st.expander("📊 Nyckeltal & historik"):
        colA, colB, colC = st.columns(3)
        with colA:
            st.write(f"**EV/EBITDA (TTM):** {round(_nz(rad.get('EV/EBITDA (TTM)')),2)}")
            st.write(f"**P/E (TTM):** {round(_nz(rad.get('P/E (TTM)')),2)}")
        with colB:
            st.write(f"**FCF (TTM):** {format_large(_nz(rad.get('FCF (TTM)')))}")
            st.write(f"**FCF-yield:** {round(_nz(rad.get('FCF-yield (%)')),2)} %")
        with colC:
            st.write(f"**Skuld:** {format_large(_nz(rad.get('Total Debt')))}")
            st.write(f"**Kassa:** {format_large(_nz(rad.get('Total Cash')))}")
        st.write(f"**Sektor:** {rad.get('Sektor','-')}  |  **Risklabel:** {rad.get('Risklabel','-')}")

    if val_flags:
        with st.expander("⚠️ Kommentarer & flaggor"):
            st.markdown("- " + "\n- ".join(val_flags))

# app.py — Del 6/? — Portfölj, Säljvakt, Enskild uppdatering & Batch-UI

# --- Små format-hjälpare -----------------------------------------------------

def _format_pct(x: float) -> str:
    try:
        return f"{float(x):.2f} %"
    except Exception:
        return "-"

# --- Endast kursuppdatering --------------------------------------------------

def update_only_prices(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Uppdaterar bara Aktuell kurs + _marketCap_raw (+ ev. namn/valuta/direktavkastning) via Yahoo för samtliga rader.
    Sparar inte per automatik – låt anroparen spara.
    Returnerar (df, log).
    """
    log = {"ok": [], "miss": []}
    total = len(df)
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()
    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        status.write(f"Kursuppdatering {i+1}/{total}: {tkr}")
        try:
            y = hamta_yahoo_fält(tkr)
            # skriv fält om de är meningsfulla
            if y.get("Aktuell kurs", 0) > 0:
                df.at[idx, "Aktuell kurs"] = float(y["Aktuell kurs"])
            if y.get("_marketCap_raw", 0) > 0:
                if "_marketCap_raw" not in df.columns:
                    df["_marketCap_raw"] = 0.0
                df.at[idx, "_marketCap_raw"] = float(y["_marketCap_raw"])
            if y.get("Bolagsnamn"):
                df.at[idx, "Bolagsnamn"] = y["Bolagsnamn"]
            if y.get("Valuta"):
                df.at[idx, "Valuta"] = y["Valuta"]
            if "Direktavkastning (%)" in y and y["Direktavkastning (%)"] is not None:
                if "Direktavkastning (%)" not in df.columns:
                    df["Direktavkastning (%)"] = 0.0
                df.at[idx, "Direktavkastning (%)"] = float(y["Direktavkastning (%)"] or 0.0)

            # stämpla auto
            _note_auto_update(df, idx, source="Kursuppdatering (Yahoo)")
            log["ok"].append(tkr)
        except Exception as e:
            log["miss"].append(f"{tkr}: {e}")
        progress.progress((i+1)/max(total,1))
    return df, log

# --- Säljvakt för innehav ----------------------------------------------------

def _sell_signal_for_position(row: pd.Series, target_col: str) -> tuple[str, float, list]:
    """
    Returnerar (label, score, flags) för ett innehav.
    Score <0 = sälj/trim, >0 = ok/behåll. Använder valuation_label_and_flags + Potential.
    """
    pot = _nz(row.get("Potential (%)"), 0.0)
    label, flags = valuation_label_and_flags(row, target_col)
    score = 0.0
    # enkel poäng: potential negativ -> minuspoäng, starkt negativ -> stark signal
    if pot <= -20: score -= 3
    elif pot <= -10: score -= 2
    elif pot <= -5: score -= 1
    elif pot >= 15: score += 1
    # extra minus om flera flags
    score -= min(len(flags), 3) * 0.5
    return label, score, flags

def build_sell_watchlist(df: pd.DataFrame, user_rates: dict, target_col: str = "Riktkurs om 1 år") -> pd.DataFrame:
    """
    Skapar en lista över dina innehav med potentiella trim/sälj-förslag.
    Beaktar Potential (%) mot target_col samt övervärderingsflaggor.
    """
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        return pd.DataFrame(columns=["Ticker","Bolagsnamn","Potential (%)","Värde (SEK)","Signal","Kommentarer"])

    # SEK-värden
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]

    # Potential
    mask = (port[target_col] > 0) & (port["Aktuell kurs"] > 0)
    port.loc[mask, "Potential (%)"] = (port.loc[mask, target_col] - port.loc[mask, "Aktuell kurs"]) / port.loc[mask, "Aktuell kurs"] * 100.0
    port.loc[~mask, "Potential (%)"] = np.nan

    # Risklabel auto (om saknas)
    if "_marketCap_raw" in port.columns:
        port["Risklabel"] = port.apply(lambda r: r.get("Risklabel") if str(r.get("Risklabel","")).strip()
                                       else risklabel_from_mcap(_row_mcap(r)), axis=1)

    # Signal
    out_rows = []
    for _, r in port.iterrows():
        lab, sc, fl = _sell_signal_for_position(r, target_col)
        out_rows.append({
            "Ticker": r.get("Ticker",""),
            "Bolagsnamn": r.get("Bolagsnamn",""),
            "Potential (%)": round(_nz(r.get("Potential (%)")),2),
            "Värde (SEK)": float(r.get("Värde (SEK)",0.0)),
            "Risklabel": r.get("Risklabel",""),
            "Signal": lab,
            "_score": sc,
            "Kommentarer": "; ".join(fl) if fl else ""
        })
    out = pd.DataFrame(out_rows)
    out = out.sort_values(by=["_score","Värde (SEK)"], ascending=[True, False]).drop(columns=["_score"])
    return out

# --- Portföljvy --------------------------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    månadsutd = tot_utd / 12.0

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total årlig utdelning:** {round(tot_utd,2)} SEK  →  **≈ per månad:** {round(månadsutd,2)} SEK")

    # visa tabell
    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    # Säljvakt
    with st.expander("⚠️ Sälj/trim-vakt (baserat på ‘Riktkurs om 1 år’ + värderingsflaggor)"):
        target_col = "Riktkurs om 1 år" if "Riktkurs om 1 år" in df.columns else "Riktkurs idag"
        watch = build_sell_watchlist(df, user_rates, target_col=target_col)
        if watch.empty:
            st.write("Inga varningar just nu.")
        else:
            st.dataframe(watch, use_container_width=True, hide_index=True)

# --- Enskild ticker: kurs & full auto ---------------------------------------

def _update_course_for_ticker(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, str]:
    """Uppdaterar enbart kurs & relaterat för 1 ticker."""
    tkr = str(ticker).upper().strip()
    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        return df, f"{tkr} hittades inte i tabellen."
    ridx = df.index[mask][0]
    try:
        y = hamta_yahoo_fält(tkr)
        if y.get("Aktuell kurs", 0) > 0:
            df.at[ridx, "Aktuell kurs"] = float(y["Aktuell kurs"])
        if y.get("_marketCap_raw", 0) > 0:
            if "_marketCap_raw" not in df.columns: df["_marketCap_raw"] = 0.0
            df.at[ridx, "_marketCap_raw"] = float(y["_marketCap_raw"])
        if y.get("Bolagsnamn"): df.at[ridx, "Bolagsnamn"] = y["Bolagsnamn"]
        if y.get("Valuta"): df.at[ridx, "Valuta"] = y["Valuta"]
        if "Direktavkastning (%)" in y and y["Direktavkastning (%)"] is not None:
            if "Direktavkastning (%)" not in df.columns: df["Direktavkastning (%)"] = 0.0
            df.at[ridx, "Direktavkastning (%)"] = float(y["Direktavkastning (%)"] or 0.0)
        _note_auto_update(df, ridx, source="Kursuppdatering (Yahoo)")
        return df, "Kurs uppdaterad."
    except Exception as e:
        return df, f"Fel vid kursuppdatering: {e}"

def _full_auto_for_ticker(df: pd.DataFrame, user_rates: dict, ticker: str, always_stamp: bool = True) -> tuple[pd.DataFrame, str]:
    """Kör full auto-pipeline för en ticker och uppdaterar beräkningar."""
    tkr = str(ticker).upper().strip()
    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        return df, f"{tkr} hittades inte i tabellen."
    ridx = df.index[mask][0]
    try:
        new_vals, debug = auto_fetch_for_ticker(tkr)
        changed = apply_auto_updates_to_row(
            df, ridx, new_vals,
            source="Auto (SEC/Yahoo→Yahoo→Finnhub→FMP)",
            changes_map={},
            always_stamp=always_stamp
        )
        df = uppdatera_berakningar(df, user_rates)
        return df, ("Full auto: ändringar gjordes." if changed else "Full auto: inga nya fält, men TS stämplades.")
    except Exception as e:
        return df, f"Fel vid full auto: {e}"

# --- Lägg till / uppdatera vy -----------------------------------------------

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsläge i listan
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, len(val_lista)-1)

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=st.session_state.edit_index if val_lista else 0, key="edit_select_lbl")

    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {max(1, st.session_state.edit_index)}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    # Statusetiketter (senast uppdaterad + källa)
    if not bef.empty:
        st.markdown(
            f"**Senast manuellt:** {bef.get('Senast manuellt uppdaterad','') or '-'}  |  "
            f"**Senast auto:** {bef.get('Senast auto-uppdaterad','') or '-'}  |  "
            f"**Källa:** {bef.get('Senast uppdaterad källa','') or '-'}"
        )

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
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, Direktavkastning, Sektor/Industri via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

        spar = st.form_submit_button("💾 Spara")

    # Snabbknappar för enskild uppdatering
    colu1, colu2, colu3 = st.columns([1,1,2])
    with colu1:
        do_price = st.button("💱 Uppdatera kurs")
    with colu2:
        do_full = st.button("🚀 Full auto för valda")

    # Spara manuellt
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
        if datum_sätt:
            ridx = df.index[df["Ticker"]==ticker][0]
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo
        data = hamta_yahoo_fält(ticker)
        ridx = df.index[df["Ticker"]==ticker][0]
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if "Direktavkastning (%)" in data and data.get("Direktavkastning (%)") is not None:
            if "Direktavkastning (%)" not in df.columns: df["Direktavkastning (%)"] = 0.0
            df.loc[ridx, "Direktavkastning (%)"] = float(data.get("Direktavkastning (%)") or 0.0)
        if data.get("_marketCap_raw",0)>0:
            if "_marketCap_raw" not in df.columns: df["_marketCap_raw"] = 0.0
            df.loc[ridx, "_marketCap_raw"] = float(data["_marketCap_raw"])
        if data.get("Sektor"): df.loc[ridx, "Sektor"] = data["Sektor"]
        if data.get("Industri"): df.loc[ridx, "Industri"] = data["Industri"]

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # Enskild kursuppdatering
    if do_price and (valt_label in namn_map):
        tkr = namn_map[valt_label]
        df, msg = _update_course_for_ticker(df, tkr)
        spara_data(df)
        st.success(msg)

    # Enskild full auto
    if do_full and (valt_label in namn_map):
        tkr = namn_map[valt_label]
        df, msg = _full_auto_for_ticker(df, user_rates, tkr, always_stamp=True)
        spara_data(df)
        st.success(msg)

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

# --- Batchpanel (kör X st nu) -----------------------------------------------

def visa_batchpanel(df: pd.DataFrame, user_rates: dict):
    st.subheader("🧰 Batch-körning (smidigare än 'alla')")
    left, right = st.columns([1,2])
    with left:
        mode = st.selectbox("Urval", ["Äldst uppdaterade först (alla fält)","A–Ö (bolagsnamn)"])
        size = st.number_input("Antal att köra nu", min_value=1, max_value=max(1,len(df)), value=min(20, max(1,len(df))), step=1)
    with right:
        tickers = _build_batch_queue(df, mode=mode, size=int(size))
        st.write(f"Kö ({len(tickers)}): " + (", ".join(tickers[:15]) + (" ..." if len(tickers)>15 else "")))
        colb1, colb2 = st.columns([1,1])
        with colb1:
            run = st.button("▶️ Kör batch nu")
        with colb2:
            just_prices = st.button("💱 Endast kurser (alla)")

    if run and tickers:
        df2, log = auto_update_batch(df.copy(), user_rates, tickers, make_snapshot=True, always_stamp=True)
        st.session_state["last_auto_log"] = log
        st.success("Batch körd.")
        # skriv tillbaka
        return df2

    if just_prices:
        df2, log = update_only_prices(df.copy())
        spara_data(df2, do_snapshot=True)
        st.success("Kurser uppdaterade för alla.")
        return df2

    return df

# app.py — Del 7/? — Hjälpare (format/risk), förbättrad Yahoo-hämtning, apply_auto_updates_to_row & MAIN

# --- Format & risk -----------------------------------------------------------

def format_large(n: float) -> str:
    try:
        x = float(n)
    except Exception:
        return "-"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12:
        return f"{sign}{x/1e12:.2f} tn"
    if x >= 1e9:
        return f"{sign}{x/1e9:.2f} mdr"
    if x >= 1e6:
        return f"{sign}{x/1e6:.2f} M"
    if x >= 1e3:
        return f"{sign}{x:,.0f}".replace(",", " ")
    return f"{sign}{x:.0f}"

def risklabel_from_mcap(mcap_raw: float) -> str:
    """
    Standard-trösklar (USD):
      Micro < 300M
      Small 300M–2B
      Mid   2B–10B
      Large 10B–200B
      Mega  >200B
    """
    try:
        mc = float(mcap_raw or 0.0)
    except Exception:
        mc = 0.0
    if mc <= 0: return "Unknown"
    if mc < 3e8:   return "Micro"
    if mc < 2e9:   return "Small"
    if mc < 1e10:  return "Mid"
    if mc < 2e11:  return "Large"
    return "Mega"

# --- Förbättrad Yahoo-hämtning (överskuggar tidigare definition) ------------

def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Utökad: försöker hämta även:
      - _marketCap_raw
      - Direktavkastning (%)
      - Sektor / Industri
    Behåller tidigare fält.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "_marketCap_raw": 0.0,
        "Direktavkastning (%)": 0.0,
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

        # direktavkastning
        dy = info.get("dividendYield", None)
        if dy is not None:
            try:
                out["Direktavkastning (%)"] = float(dy) * 100.0
            except Exception:
                pass
        elif out["Årlig utdelning"] and out["Aktuell kurs"]:
            out["Direktavkastning (%)"] = (out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0

        # market cap
        mc = info.get("marketCap")
        try:
            mc = float(mc or 0.0)
        except Exception:
            mc = 0.0
        out["_marketCap_raw"] = mc

        # sektor/industri
        if info.get("sector"):   out["Sektor"]   = str(info["sector"])
        if info.get("industry"): out["Industri"] = str(info["industry"])

        # CAGR
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# --- Uppdaterad apply_auto_updates_to_row: always_stamp-stöd -----------------

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
    Uppdaterar TS_ för spårade fält, sätter 'Senast auto-uppdaterad' + källa.
    Om always_stamp=True stämplas alltid 'Senast auto-uppdaterad' (även om inget ändras).
    Returnerar True om något fält faktiskt ändrades.
    """
    changed_fields = []
    for f, v in (new_vals or {}).items():
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

    if changed_fields or always_stamp:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return bool(changed_fields)
    return False

# --- MAIN --------------------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Initiera session_state för valutor innan widgets skapas
    if "rate_usd" not in st.session_state or "rate_nok" not in st.session_state \
       or "rate_cad" not in st.session_state or "rate_eur" not in st.session_state:
        sv = las_sparade_valutakurser()
        st.session_state.rate_usd = float(sv.get("USD", STANDARD_VALUTAKURSER["USD"]))
        st.session_state.rate_nok = float(sv.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        st.session_state.rate_cad = float(sv.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        st.session_state.rate_eur = float(sv.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

    # Sidopanel: valutakurser
    st.sidebar.header("💱 Valutakurser → SEK")
    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")

    # Auto-hämtning av kurser
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
        st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
        st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
        st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        st.sidebar.success(f"Valutakurser uppdaterade (källa: {provider}).")
        st.rerun()

    user_rates = {"USD": st.session_state.rate_usd, "NOK": st.session_state.rate_nok,
                  "CAD": st.session_state.rate_cad, "EUR": st.session_state.rate_eur, "SEK": 1.0}

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear()
            sv = las_sparade_valutakurser()
            st.session_state.rate_usd = float(sv.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(sv.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(sv.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(sv.get("EUR", st.session_state.rate_eur))
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # Läs data
    df = hamta_data()
    # Skriv ALDRIG tillbaka automatiskt om df är tom – undvik att rensa bladet
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)

    # Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Auto-uppdatering i sidopanel (legacy "alla"), men föredra batch-panelen i huvudytan
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Auto-uppdatering")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True)
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → Yahoo → Finnhub → FMP)"):
        tickers = list(df["Ticker"].astype(str))
        df2, log = auto_update_batch(df.copy(), user_rates, tickers, make_snapshot=make_snapshot, always_stamp=True)
        st.session_state["last_auto_log"] = log
        # Spara redan gjort i funktionen om ändringar skedde
        st.rerun()

    # Meny
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Kontroll","Batch","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"],
        index=0
    )

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Batch":
        df = visa_batchpanel(df, user_rates)
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
