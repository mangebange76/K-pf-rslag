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

# Slutlig kolumnlista i databasen (inkl. nya fält för mcap/nyckeltal)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Finansiell valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    # Market cap (nu + historik)
    "MCap (nu)", "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
    "MCap Datum Q1", "MCap Datum Q2", "MCap Datum Q3", "MCap Datum Q4",
    # Nyckeltal
    "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa",
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
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","mcap","kassa","marginal","debt"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
                         "Valuta","Finansiell valuta","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4","Bolagsnamn","Ticker"):
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
    str_cols = ["Ticker","Bolagsnamn","Valuta","Finansiell valuta",
                "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
                "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Finansiell valuta, Utdelning, CAGR,
    P/S (Yahoo), MCap (nu).
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

        # Pris
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        # Valutor
        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()
        fccy = info.get("financialCurrency", None)
        if fccy:
            out["Finansiell valuta"] = str(fccy).upper()

        # Namn
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        # Utdelning (årstakt)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # CAGR (från resultaträkningen)
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

        # P/S (Yahoo-fält)
        ps_y = info.get("priceToSalesTrailing12Months") or info.get("priceToSalesTrailing12Months")
        try:
            if ps_y and float(ps_y) > 0:
                out["P/S (Yahoo)"] = float(ps_y)
        except Exception:
            pass

        # Market cap
        mcap = info.get("marketCap")
        try:
            mcap = float(mcap) if mcap is not None else 0.0
        except Exception:
            mcap = 0.0
        if mcap <= 0:
            so = info.get("sharesOutstanding")
            try:
                so = float(so or 0.0)
            except Exception:
                so = 0.0
            if so > 0 and out["Aktuell kurs"] > 0:
                mcap = so * out["Aktuell kurs"]
        if mcap > 0:
            out["MCap (nu)"] = float(mcap)

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
            # behåll ev. befintliga
            try:
                df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
                df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))
            except Exception:
                df.at[i, "Omsättning om 2 år"] = 0.0
                df.at[i, "Omsättning om 3 år"] = 0.0

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        try:
            aktier_ut = float(rad.get("Utestående aktier", 0.0))
        except Exception:
            aktier_ut = 0.0
        if aktier_ut > 0 and ps_snitt > 0:
            try:
                o_idag = float(rad.get("Omsättning idag", 0.0))
                df.at[i, "Riktkurs idag"]    = round((o_idag      * ps_snitt) / aktier_ut, 2)
            except Exception:
                df.at[i, "Riktkurs idag"] = 0.0
            try:
                o_next = float(rad.get("Omsättning nästa år", 0.0))
                df.at[i, "Riktkurs om 1 år"] = round((o_next       * ps_snitt) / aktier_ut, 2)
            except Exception:
                df.at[i, "Riktkurs om 1 år"] = 0.0
            try:
                o2 = float(df.at[i, "Omsättning om 2 år"])
                o3 = float(df.at[i, "Omsättning om 3 år"])
                df.at[i, "Riktkurs om 2 år"] = round((o2 * ps_snitt) / aktier_ut, 2)
                df.at[i, "Riktkurs om 3 år"] = round((o3 * ps_snitt) / aktier_ut, 2)
            except Exception:
                df.at[i, "Riktkurs om 2 år"] = 0.0
                df.at[i, "Riktkurs om 3 år"] = 0.0
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
    Skriver fält med nya (meningsfulla) värden.
    - Om force_ts=True stämplas TS_ för spårade fält och 'Senast auto-uppdaterad' + källa,
      även om värdet är oförändrat (används för att markera “kontrollerad idag”).
    Returnerar True om något fält faktiskt ÄNDRADES (värdebyte).
    """
    changed_fields = []
    any_processed_field = False

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue

        # avgör om v är skrivbart
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # för P/S & aktier kräver vi > 0, annars ≥ 0
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]:
                write_ok = float(v) > 0
            else:
                write_ok = float(v) >= 0
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        if not write_ok and not force_ts:
            continue  # hoppa om v saknar mening och vi inte tvingar TS

        any_processed_field = True

        # skriv om det verkligen ändras
        old = df.at[row_idx, f] if f in df.columns else None
        will_change = (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))

        if will_change and write_ok:
            df.at[row_idx, f] = v
            changed_fields.append(f)

        # TS-stämpla för spårade fält när vi skriver/för TS-forcering
        if f in TS_FIELDS:
            _stamp_ts_for_field(df, row_idx, f)

    # stämpla auto-uppdatering/källa om något processats
    if any_processed_field or changed_fields:
        _note_auto_update(df, row_idx, source)

    if changed_fields:
        tkr = df.at[row_idx, "Ticker"] if "Ticker" in df.columns else ""
        changes_map.setdefault(str(tkr), []).extend(changed_fields)
        return True
    return False

# app.py — Del 3/7
# --- SEC (US/IFRS), Q4-hotfix, Yahoo-fallback, TTM/MCAP & nyckeltal ----------

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

# ---------- datum & aktier (robusta "instant") -------------------------------
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

# ---------- IFRS/GAAP kvartals- & årsintäkter + valuta -----------------------
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

def _pick_best_unit(units: dict, prefer=("USD","CAD","EUR","GBP")):
    for u in prefer:
        if isinstance(units.get(u), list) and units.get(u):
            return u
    # fallback till första
    for k, v in units.items():
        if isinstance(v, list) and v:
            return k
    return None

def _collect_duration_rows(fact_obj: dict, forms_ok: tuple, want_quarter: bool) -> list[tuple]:
    """
    Returnera [(end_date, value, unit, form, start_date)] filtrerat på forms_ok.
    want_quarter=True: 70–100 dagar (10-Q/6-K). False: 330–400 dagar (10-K/20-F/40-F).
    """
    out = []
    units = (fact_obj.get("units") or {})
    unit_key = _pick_best_unit(units)
    if not unit_key:
        return out
    arr = units.get(unit_key) or []
    for it in arr:
        form = (it.get("form") or "").upper()
        if forms_ok and form not in forms_ok:
            continue
        end = _parse_iso(str(it.get("end", "")))
        start = _parse_iso(str(it.get("start", "")))
        val = it.get("val", None)
        if not (end and start and val is not None):
            continue
        try:
            dur = (end - start).days
        except Exception:
            continue
        if want_quarter and not (70 <= dur <= 100):
            continue
        if (not want_quarter) and not (330 <= dur <= 400):
            continue
        try:
            v = float(val)
            out.append((end, v, unit_key, form, start))
        except Exception:
            pass
    # dedupe per end (nyaste vinner)
    ded = {}
    for end, v, u, f, s in out:
        ded[end] = (end, v, u, f, s)
    res = list(ded.values())
    res.sort(key=lambda t: t[0], reverse=True)
    return res

def _sec_quarterly_and_annual_revenues(facts: dict, max_quarters: int = 24):
    """
    Hämtar kvartalsintäkter (10-Q/6-K) och årsintäkter (10-K/20-F/40-F).
    Returnerar:
      q_rows: [(end, value, unit)], a_rows: [(end, value, unit)]
    """
    taxos = [("us-gaap", ("10-Q","10-Q/A"), ("10-K","10-K/A")),
             ("ifrs-full", ("6-K","6-K/A","10-Q","10-Q/A"), ("20-F","40-F","20-F/A","40-F/A","10-K","10-K/A"))]
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    for taxo, q_forms, a_forms in taxos:
        gaap = (facts.get("facts") or {}).get(taxo, {})
        # prova i ordning
        for name in rev_keys:
            fact = gaap.get(name)
            if not fact:
                continue
            q_full = _collect_duration_rows(fact, q_forms, want_quarter=True)
            a_full = _collect_duration_rows(fact, a_forms, want_quarter=False)
            if q_full or a_full:
                q_rows = [(e, v, u) for (e, v, u, _f, _s) in q_full][:max_quarters]
                a_rows = [(e, v, u) for (e, v, u, _f, _s) in a_full][:12]
                return q_rows, a_rows
    return [], []

def _inject_missing_q4_with_annual(q_rows: list, a_rows: list):
    """
    Om årsrapport finns men Q4 (sista kvartalet) saknas:
      synth_Q4 = Annual - sum(Q1..Q3) för samma räkenskapsår (≈12 mån före annual.end).
    Lägg till synth_Q4 om den är positiv och rimlig.
    """
    if not a_rows or not q_rows:
        return q_rows
    # Gör index över kvartal per år (styrt av annual_end.year)
    q = list(q_rows)
    for a_end, a_val, a_unit in a_rows:
        # leta tre kvartal inom ~11 månader före a_end
        three = []
        for (e, v, u) in q_rows:
            if u != a_unit:
                continue
            if e <= a_end and (a_end - e).days <= 370:
                three.append((e, v))
        # plocka de tre senaste före a_end
        three = sorted([t for t in three if t[0] < a_end], key=lambda t: t[0], reverse=True)[:3]
        if len(three) == 3:
            s3 = sum(v for (_, v) in three)
            synth = float(a_val) - float(s3)
            # rimlighetskoll
            if synth > 0 and synth < (a_val * 1.5):
                # finns redan kvartal exakt på a_end?
                if not any(e == a_end for (e, _, _) in q_rows):
                    q.append((a_end, synth, a_unit))
    # dedupe + sort
    ded = {}
    for e, v, u in q:
        ded[e] = (e, v, u)
    res = list(ded.values())
    res.sort(key=lambda t: t[0], reverse=True)
    return res

def _sec_quarterly_revenues_dated_with_unit_fixed(facts: dict, need: int = 20):
    """
    Kvartalsintäkter med Q4-hotfix (syntetiskt Q4 från årsrapport om saknas).
    Returnerar ([(end, value)], unit).
    """
    q_rows, a_rows = _sec_quarterly_and_annual_revenues(facts, max_quarters=need)
    if not q_rows and not a_rows:
        return [], None
    if a_rows:
        q_rows = _inject_missing_q4_with_annual(q_rows, a_rows)
    # välj en gemensam unit (ta från q_rows om finns, annars annual)
    unit = q_rows[0][2] if q_rows else a_rows[0][2]
    rows = [(e, v) for (e, v, u) in q_rows if u == unit]
    rows.sort(key=lambda t: t[0], reverse=True)
    return rows[:need], unit

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
    vals = list(values)
    vals.sort(key=lambda t: t[0], reverse=True)
    if len(vals) < 4:
        return out
    for i in range(0, min(need, len(vals) - 3)):
        end_i = vals[i][0]
        ttm_i = sum(v for (_, v) in vals[i:i+4])
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

# ---------- Nyckeltal från Yahoo (Debt/Equity, marginaler, kassa) ------------
def hamta_yahoo_nyckeltal(ticker: str) -> dict:
    """
    Best-effort: räknar Debt/Equity, bruttomarginal, nettomarginal samt kassa
    från Yahoo-balanser och resultaträkningar (senaste period).
    """
    out = {"Debt/Equity": 0.0, "Bruttomarginal (%)": 0.0, "Nettomarginal (%)": 0.0, "Kassa": 0.0}
    try:
        t = yf.Ticker(ticker)

        # Balance sheet
        bs = getattr(t, "balance_sheet", None)
        total_debt = None
        equity = None
        cash_like = None
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            col = bs.columns[0]  # senaste
            # fält kan variera
            for k in ["Total Debt", "TotalDebt", "Short Long Term Debt Total", "ShortLongTermDebtTotal"]:
                if k in bs.index and not pd.isna(bs.loc[k, col]):
                    try: total_debt = float(bs.loc[k, col]); break
                    except: pass
            for k in ["Total Stockholder Equity", "Total Stockholders Equity", "Stockholders Equity", "TotalEquityGrossMinorityInterest"]:
                if k in bs.index and not pd.isna(bs.loc[k, col]):
                    try: equity = float(bs.loc[k, col]); break
                    except: pass
            for k in ["Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents", "Cash And Short Term Investments", "CashAndShortTermInvestments"]:
                if k in bs.index and not pd.isna(bs.loc[k, col]):
                    try: cash_like = float(bs.loc[k, col]); break
                    except: pass

        # Income statement (TTM/quarter)
        isq = getattr(t, "quarterly_financials", None)
        gross = rev = net = None
        if isinstance(isq, pd.DataFrame) and not isq.empty:
            col = isq.columns[0]
            if "Total Revenue" in isq.index and not pd.isna(isq.loc["Total Revenue", col]):
                try: rev = float(isq.loc["Total Revenue", col])
                except: pass
            elif "TotalRevenue" in isq.index and not pd.isna(isq.loc["TotalRevenue", col]):
                try: rev = float(isq.loc["TotalRevenue", col])
                except: pass
            for k in ["Gross Profit","GrossProfit"]:
                if k in isq.index and not pd.isna(isq.loc[k, col]):
                    try: gross = float(isq.loc[k, col]); break
                    except: pass
            for k in ["Net Income","NetIncome"]:
                if k in isq.index and not pd.isna(isq.loc[k, col]):
                    try: net = float(isq.loc[k, col]); break
                    except: pass

        if total_debt is not None and equity not in (None, 0):
            out["Debt/Equity"] = float(total_debt) / float(equity)
        if gross is not None and rev not in (None, 0):
            out["Bruttomarginal (%)"] = (gross / rev) * 100.0
        if net is not None and rev not in (None, 0):
            out["Nettomarginal (%)"] = (net / rev) * 100.0
        if cash_like is not None:
            out["Kassa"] = float(cash_like)
    except Exception:
        pass
    return out

# ---------- SEC + Yahoo combo -------------------------------------------------
def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (inkl. Q4-hotfix),
    pris/valuta/namn/P/S(Yahoo)/MCap(nu) från Yahoo.
    P/S (TTM) nu + P/S Q1–Q4 historik, MCAP Q1–Q4 + datum.
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
    for k in ["Bolagsnamn", "Valuta", "Finansiell valuta", "Aktuell kurs", "P/S (Yahoo)", "MCap (nu)"]:
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

    # SEC kvartalsintäkter + unit (med Q4-hotfix) → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit_fixed(facts, need=20)
    if not q_rows or not rev_unit:
        return out
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    ttm_list = _ttm_windows(q_rows, need=4)
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # Market cap (nu)
    mcap_now = float(out.get("MCap (nu)", 0.0) or 0.0)
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used
        out["MCap (nu)"] = mcap_now

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik & MCAP-historik vid respektive TTM-end
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCap Q{idx}"] = float(mcap_hist)
                    out[f"MCap Datum Q{idx}"] = d_end.strftime("%Y-%m-%d")

    return out

# ---------- Yahoo global fallback --------------------------------------------
def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, P/S Q1–Q4 & MCAP-historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/ps_y/mcap
    y = hamta_yahoo_fält(ticker)
    for k in ["Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","P/S (Yahoo)","MCap (nu)"]:
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(out.get("MCap (nu)", 0.0) or 0.0)
    if mcap <= 0 and px > 0:
        so = info.get("sharesOutstanding")
        try:
            so = float(so or 0.0)
        except Exception:
            so = 0.0
        if so > 0:
            mcap = so * px
            out["MCap (nu)"] = mcap

    # Implied shares
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

    # P/S nu
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S & MCAP historik
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px[:4]]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares * p
                    out[f"P/S Q{idx}"] = (mcap_hist / ttm_rev_px)
                    out[f"MCap Q{idx}"] = float(mcap_hist)
                    out[f"MCap Datum Q{idx}"] = d_end.strftime("%Y-%m-%d")

    return out

# app.py — Del 4/7
# --- FMP, Finnhub, snapshots, pris/full uppd., vågkörning & auto-all ---------

# =============== FMP (lätt) ==================================================
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.0))      # skonsam default
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20)) # paus efter 429

def _fmp_get(path: str, params=None):
    """
    Throttlad GET med enkel backoff + 'circuit breaker' vid 429.
    Returnerar (json, statuscode). path t.ex. 'api/v3/quote/AAPL'.
    """
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

            if sc in (429, 403, 502, 503, 504):
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
    """
    out = {"_symbol": _fmp_pick_symbol(yahoo_ticker)}
    sym = out["_symbol"]

    # pris, marketCap, sharesOutstanding
    q, sc_q = _fmp_get(f"api/v3/quote/{sym}")
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
    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}")
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
        data.sort(key=lambda d: d.get("period",""))
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

# =============== Snapshot till flik ==========================================
def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df.
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

# =============== Pris & full uppdatering (per ticker / alla) =================
def update_price_for_ticker(df: pd.DataFrame, ticker: str):
    """
    Uppdaterar endast pris/valutor/P-S(Yahoo)/MCap(nu) för ett specifikt bolag.
    Stämplar auto-källa/TS (force_ts) även om värdet inte ändras.
    """
    tkr = str(ticker).strip().upper()
    if "Ticker" not in df.columns or tkr not in set(df["Ticker"].astype(str).str.upper()):
        return df, False, f"{tkr} hittades inte i tabellen."

    ridx = df.index[df["Ticker"].astype(str).str.upper() == tkr][0]
    y = hamta_yahoo_fält(tkr)
    if not y:
        _note_auto_update(df, ridx, "Pris (Yahoo – inget svar)")
        return df, False, None

    new_vals = {}
    for k in ["Aktuell kurs","Valuta","Finansiell valuta","P/S (Yahoo)","MCap (nu)"]:
        if y.get(k) not in (None, "", 0, 0.0):
            new_vals[k] = y[k]

    changed = apply_auto_updates_to_row(
        df, ridx, new_vals, source="Pris (Yahoo)", changes_map={}, force_ts=True
    )
    return df, changed, None

def update_prices_all(df: pd.DataFrame):
    """
    Uppdaterar pris för alla tickers (Yahoo).
    """
    updated, skipped = [], []
    total = len(df)
    prog = st.sidebar.progress(0.0)
    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row.get("Ticker","")).upper().strip()
        if not tkr:
            prog.progress((i+1)/max(1,total)); continue
        df, changed, _ = update_price_for_ticker(df, tkr)
        if changed: updated.append(tkr)
        else: skipped.append(tkr)
        prog.progress((i+1)/max(1,total))
    log = {"updated": updated, "skipped": skipped}
    return df, log

def update_full_for_ticker(df: pd.DataFrame, ticker: str, user_rates: dict, *, force_ts: bool = True):
    """
    Kör full uppdatering för ETT bolag:
      1) SEC+Yahoo combo (inkl. Q4-hotfix, MCAP-historik)
      2) Finnhub estimat (om saknas)
      3) FMP light P/S (om saknas)
      4) Nyckeltal (Debt/Equity, marginaler, Kassa) från Yahoo
    """
    tkr = str(ticker).strip().upper()
    if "Ticker" not in df.columns or tkr not in set(df["Ticker"].astype(str).str.upper()):
        return df, False, f"{tkr} hittades inte i tabellen."

    ridx = df.index[df["Ticker"].astype(str).str.upper() == tkr][0]
    vals = {}
    # 1) SEC / Yahoo combo
    try:
        base = hamta_sec_yahoo_combo(tkr)
        for k in [
            "Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs",
            "Utestående aktier","P/S","P/S (Yahoo)",
            "P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
            "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"
        ]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        st.warning(f"SEC/Yahoo-fel för {tkr}: {e}")

    # 2) Finnhub estimat (om saknas)
    try:
        if ("Omsättning idag" not in vals) or ("Omsättning nästa år" not in vals):
            fh = hamta_finnhub_revenue_estimates(tkr)
            for k in ["Omsättning idag","Omsättning nästa år"]:
                v = fh.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        st.warning(f"Finnhub-fel för {tkr}: {e}")

    # 3) FMP light P/S (om saknas)
    try:
        if ("P/S" not in vals):
            fmpl = hamta_fmp_falt_light(tkr)
            v = fmpl.get("P/S")
            if v not in (None, "", 0, 0.0):
                vals["P/S"] = v
            # ibland får vi bättre shares/mcap från FMP
            for k in ["Utestående aktier","MCap (nu)","Aktuell kurs"]:
                v2 = fmpl.get(k)
                if v2 not in (None, "", 0, 0.0) and k not in vals:
                    vals[k] = v2
    except Exception as e:
        st.warning(f"FMP-fel för {tkr}: {e}")

    # 4) Nyckeltal
    try:
        yk = hamta_yahoo_nyckeltal(tkr)
        for k in ["Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa"]:
            v = yk.get(k)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        st.warning(f"Nyckeltalsfel för {tkr}: {e}")

    changed = apply_auto_updates_to_row(
        df, ridx, vals, source="Auto (SEC/Yahoo→Finnhub→FMP)", changes_map={}, force_ts=force_ts
    )

    # beräkningar (riktkurser etc.)
    df = uppdatera_berakningar(df, user_rates)
    return df, changed, None

# =============== Auto-uppdatera alla =========================================
def auto_update_all(df: pd.DataFrame, user_rates: dict, *, make_snapshot: bool = False, force_ts: bool = True):
    """
    Kör auto-uppdatering för alla rader. Skriver endast meningsfulla värden.
    Stämplar 'Senast auto-uppdaterad' + källa. TS forceras om force_ts=True.
    """
    log = {"changed": {}, "misses": {}}
    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()

    total = len(df)
    any_changed = False

    for i, row in df.reset_index().iterrows():
        idx = row["index"]
        tkr = str(row["Ticker"]).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(1,total)); continue

        status.write(f"Uppdaterar {i+1}/{total}: {tkr}")
        try:
            df, changed, err = update_full_for_ticker(df, tkr, user_rates, force_ts=force_ts)
            if err:
                log["misses"][tkr] = [str(err)]
            elif not changed:
                log["misses"][tkr] = ["(inga värdeändringar – TS/källa uppdaterades)"]
            else:
                # markera ändringar (per-rad ändringar loggas redan i apply_auto_updates_to_row om man vill)
                log["changed"].setdefault(tkr, ["värden uppdaterade"])
            any_changed = any_changed or changed
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(1,total))

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    if any_changed or force_ts:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Uppdatering genomförd.")
    else:
        st.sidebar.info("Ingen faktisk värdeändring – ingen skrivning/snapshot gjordes.")

    return df, log

# =============== Vågkörning (batch i omgångar) ===============================
def start_wave(df: pd.DataFrame, mode: str = "oldest"):
    """
    Bygger en kö av tickers som ska köras i vågor.
    mode: 'oldest' (äldst TS först) eller 'alpha' (A–Ö).
    """
    tickers = list(df["Ticker"].astype(str).str.upper())
    if mode == "alpha":
        order = sorted(tickers)
    else:
        # kräver add_oldest_ts_col (definieras i Del 5/7)
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
        order = list(work["Ticker"].astype(str).str.upper())
    st.session_state["wave_queue"] = order
    st.session_state["wave_pos"] = 0

def run_wave_step(df: pd.DataFrame, user_rates: dict, *, batch_size: int = 10, make_snapshot: bool = False, force_ts: bool = True):
    """
    Kör nästa batch i vågen. Returnerar (df, stat) där stat innehåller processed/remaining.
    """
    queue = st.session_state.get("wave_queue") or []
    pos = int(st.session_state.get("wave_pos", 0))
    if not queue or pos >= len(queue):
        return df, {"processed": 0, "remaining": 0}

    end = min(len(queue), pos + int(batch_size))
    chunk = queue[pos:end]
    processed = 0
    for tkr in chunk:
        try:
            df, _, _ = update_full_for_ticker(df, tkr, user_rates, force_ts=force_ts)
            processed += 1
        except Exception as e:
            st.warning(f"Vågfel {tkr}: {e}")

    st.session_state["wave_pos"] = end
    remaining = max(0, len(queue) - end)

    # spara efter varje batch
    spara_data(df, do_snapshot=make_snapshot)

    return df, {"processed": processed, "remaining": remaining}

# app.py — Del 5/7
# --- Hjälpfunktioner (TS/badges/format) + Kontroll-vy ------------------------

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
    - och/eller äldsta TS är äldre än 'older_than_days'.
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

def _badge_dates(row: pd.Series) -> str:
    """
    Liten etikett som visar när posten uppdaterats och hur.
    """
    manu = str(row.get("Senast manuellt uppdaterad","")).strip()
    auto = str(row.get("Senast auto-uppdaterad","")).strip()
    src  = str(row.get("Senast uppdaterad källa","")).strip()

    parts = []
    if manu:
        parts.append(f"📝 Manuell: {manu}")
    if auto:
        parts.append(f"⚙️ Auto: {auto}")
    if src:
        parts.append(f"🔗 Källa: {src}")
    if not parts:
        return "Ingen uppdateringshistorik ännu."
    return " • ".join(parts)

# ---- Enkla formatterare ------------------------------------------------------

def _format_large_sv(n: float) -> str:
    """
    Formatera stora tal till svenskt kortformat:
    - tn = triljoner, md = miljarder, mn = miljoner
    """
    try:
        v = float(n)
    except Exception:
        return str(n)
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12:
        return f"{sign}{v/1e12:.2f} tn"
    if v >= 1e9:
        return f"{sign}{v/1e9:.2f} md"
    if v >= 1e6:
        return f"{sign}{v/1e6:.2f} mn"
    if v >= 1e3:
        return f"{sign}{v/1e3:.0f} t"
    return f"{sign}{v:.2f}"

def _risk_label(mcap: float) -> str:
    """
    Grov indelning: Micro (<300m USD), Small (0.3–2b), Mid (2–10b), Large (10–200b), Mega (>200b).
    OBS: mcap antas i *prisvalutans* enhet (oftast USD).
    """
    try:
        v = float(mcap)
    except Exception:
        return ""
    if v <= 0: return ""
    if v < 3e8: return "Microcap"
    if v < 2e9: return "Smallcap"
    if v < 1e10: return "Midcap"
    if v < 2e11: return "Largecap"
    return "Megacap"

# ---- Kontroll-vy -------------------------------------------------------------

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
        st.markdown("**Auto (alla)**")
        log = st.session_state.get("last_auto_log")
        if log:
            st.json(log)
        else:
            st.write("–")
    with col2:
        st.markdown("**Prisuppdatering (alla)**")
        plog = st.session_state.get("last_price_log")
        if plog:
            st.json(plog)
        else:
            st.write("–")

    # 4) Vågstatus
    st.divider()
    st.subheader("🌊 Vågstatus")
    queue = st.session_state.get("wave_queue") or []
    pos = int(st.session_state.get("wave_pos", 0))
    if queue:
        st.info(f"Kö-längd: {len(queue)} | Bearbetade: {pos} | Kvar: {max(0,len(queue)-pos)}")
        if st.button("Töm kö här också"):
            st.session_state.pop("wave_queue", None)
            st.session_state.pop("wave_pos", None)
            st.success("Kö återställd.")
    else:
        st.write("Ingen aktiv kö.")

# app.py — Del 6/7
# --- Investeringsförslag & Portfölj ------------------------------------------

def _ps_display(row: pd.Series) -> tuple[float, float]:
    """
    Returnera (P/S nu, P/S (Yahoo)) där P/S nu prioriterar vårt TTM-beräknade P/S.
    """
    try:
        ps_now = float(row.get("P/S", 0.0))
    except Exception:
        ps_now = 0.0
    try:
        ps_yh = float(row.get("P/S (Yahoo)", 0.0))
    except Exception:
        ps_yh = 0.0
    return ps_now, ps_yh

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # ---- Filter & val --------------------------------------------------------
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)

    riskfilter = st.selectbox(
        "Filtrera efter kap-segment (risklabel)",
        ["Alla","Microcap","Smallcap","Midcap","Largecap","Megacap"],
        index=0,
        help="Baserat på Market Cap (nu): Micro(<0.3 md USD), Small(0.3–2), Mid(2–10), Large(10–200), Mega(>200)."
    )

    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    # ---- Urval ---------------------------------------------------------------
    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    # Kräver riktkurs & pris
    base = base[(base.get(riktkurs_val, 0) > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Risklabel (från MCap (nu))
    base["Risklabel"] = base["MCap (nu)"].apply(_risk_label)
    if riskfilter != "Alla":
        base = base[base["Risklabel"] == riskfilter].copy()
        if base.empty:
            st.warning("Inga bolag kvar efter riskfilter.")
            return

    # Potentialer
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    # Sortering
    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Paginering bland förslag
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

    # ---- Portföljvärden för andelsberäkning ---------------------------------
    port = df[df["Antal aktier"] > 0].copy()
    port_värde = 0.0
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())

    # Beräkna köpantal i Valuta→SEK
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

    # ---- Visning -------------------------------------------------------------
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.caption(_badge_dates(rad))

    ps_now, ps_yh = _ps_display(rad)
    ps_snitt = float(rad.get("P/S-snitt", 0.0) or 0.0)

    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Rekommenderat antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande portföljandel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
        f"- **Risklabel:** {rad.get('Risklabel','') or '—'}",
    ]
    st.markdown("\n".join(lines))

    # ---- Expander: detaljer & historik --------------------------------------
    with st.expander("📊 Detaljer & historik (MCap, P/S, nyckeltal)", expanded=False):
        # Market cap nu, P/S nu + Yahoo, snitt
        mcap_now = float(rad.get("MCap (nu)", 0.0) or 0.0)
        mcap_str = _format_large_sv(mcap_now)
        st.markdown(
            f"- **Market Cap (nu):** {mcap_str} ({rad.get('Valuta','')})\n"
            f"- **P/S (TTM – modell):** {round(ps_now, 3) if ps_now else '—'}\n"
            f"- **P/S (Yahoo):** {round(ps_yh, 3) if ps_yh else '—'}\n"
            f"- **P/S-snitt (Q1..Q4):** {round(ps_snitt, 3) if ps_snitt else '—'}"
        )

        # P/S Q1..Q4 + MCap Q1..Q4 tabell
        def _row_val(c): 
            try: return float(rad.get(c, 0.0) or 0.0)
            except: return 0.0
        ps_cols = [("Q1","P/S Q1"),("Q2","P/S Q2"),("Q3","P/S Q3"),("Q4","P/S Q4")]
        mc_cols = [("Q1","MCap Q1","MCap Datum Q1"),("Q2","MCap Q2","MCap Datum Q2"),
                   ("Q3","MCap Q3","MCap Datum Q3"),("Q4","MCap Q4","MCap Datum Q4")]

        data_rows = []
        for (lab, ps_c) in ps_cols:
            ps_val = _row_val(ps_c)
            mc_val = 0.0; d_str = ""
            for (lab2, mc_c, d_c) in mc_cols:
                if lab2 == lab:
                    mc_val = _row_val(mc_c)
                    d_str = str(rad.get(d_c,"") or "")
                    break
            data_rows.append({
                "Period": lab,
                "P/S": round(ps_val, 3) if ps_val else None,
                "MCap": _format_large_sv(mc_val) if mc_val else None,
                "Datum (TTM-slut)": d_str or None
            })
        hist_df = pd.DataFrame(data_rows)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Nyckeltal
        d_e = float(rad.get("Debt/Equity", 0.0) or 0.0)
        gm = float(rad.get("Bruttomarginal (%)", 0.0) or 0.0)
        nm = float(rad.get("Nettomarginal (%)", 0.0) or 0.0)
        cash = float(rad.get("Kassa", 0.0) or 0.0)

        nyckel_lines = [
            f"- **Debt/Equity:** {round(d_e, 2) if d_e else '—'}",
            f"- **Bruttomarginal:** {round(gm, 2)} %" if gm else "- **Bruttomarginal:** —",
            f"- **Nettomarginal:** {round(nm, 2)} %" if nm else "- **Nettomarginal:** —",
            f"- **Kassa:** {_format_large_sv(cash)}" if cash else "- **Kassa:** —",
        ]
        st.markdown("\n".join(nyckel_lines))

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

# app.py — Del 7/7
# --- Analys, Lägg till/uppdatera & MAIN --------------------------------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    if vis_df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #",
        min_value=0,
        max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx,
        step=1
    )
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
            st.rerun()
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
            st.rerun()
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    st.caption(_badge_dates(r))

    cols = [
        "Ticker","Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","Utestående aktier","MCap (nu)",
        "P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Debt/Equity","Bruttomarginal (%)","Nettomarginal (%)","Kassa",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år",
        "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

    # Snabbåtgärder för vald ticker
    tkr = r["Ticker"]
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💹 Uppdatera enbart kurs (detta bolag)"):
            df2, ch, err = update_price_for_ticker(df.copy(), tkr)
            if err:
                st.warning(err)
            else:
                spara_data(uppdatera_berakningar(df2, {}), do_snapshot=False)
                st.success(f"Pris uppdaterat för {tkr}.")
                st.rerun()
    with c2:
        if st.button("🤖 Full auto (detta bolag)"):
            df2, ch, err = update_full_for_ticker(df.copy(), tkr, user_rates, force_ts=True)
            if err:
                st.warning(err)
            else:
                spara_data(df2, do_snapshot=False)
                st.success(f"Auto-uppdatering klar för {tkr}.")
                st.rerun()

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
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
            st.rerun()
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)
            st.rerun()

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
        st.caption(_badge_dates(bef))
    else:
        bef = pd.Series({}, dtype=object)

    # Snabbknappar för valt bolag (om finns)
    if not bef.empty:
        tkr = bef.get("Ticker","")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💹 Uppdatera kurs (detta bolag)"):
                df2, ch, err = update_price_for_ticker(df.copy(), tkr)
                if err:
                    st.warning(err)
                else:
                    spara_data(uppdatera_berakningar(df2, user_rates), do_snapshot=False)
                    st.success("Kurs uppdaterad (TS/källa stämplades).")
                    st.rerun()
        with c2:
            if st.button("🤖 Full auto (detta bolag)"):
                df2, ch, err = update_full_for_ticker(df.copy(), tkr, user_rates, force_ts=True)
                if err:
                    st.warning(err)
                else:
                    spara_data(df2, do_snapshot=False)
                    st.success("Auto-uppdatering klar.")
                    st.rerun()

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

            st.markdown("**Vid spara hämtas även basfält via Yahoo och riktkurser räknas om.**")
            st.write("- Bolagsnamn, Valuta, Finansiell valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%), P/S (Yahoo), MCap (nu)")

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
        if data.get("Bolagsnamn"):          df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):              df.loc[ridx, "Valuta"]     = data["Valuta"]
        if data.get("Finansiell valuta"):   df.loc[ridx, "Finansiell valuta"] = data["Finansiell valuta"]
        if data.get("Aktuell kurs",0)>0:    df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data:       df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:         df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if "P/S (Yahoo)" in data:           df.loc[ridx, "P/S (Yahoo)"]     = float(data.get("P/S (Yahoo)") or 0.0)
        if "MCap (nu)" in data:             df.loc[ridx, "MCap (nu)"]       = float(data.get("MCap (nu)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df, do_snapshot=False)
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

# ------------------------------ MAIN -----------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: valutakurser
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, format="%.4f")

    # Auto-hämtning av kurser
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
        spara_data(df, do_snapshot=False)

    # Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Sidopanel: snabblock för uppdateringar ------------------------------
    st.sidebar.subheader("⚡ Snabbåtgärder")
    if st.sidebar.button("💹 Uppdatera kurs för alla tickers"):
        df2, plog = update_prices_all(df.copy())
        spara_data(df2, do_snapshot=False)
        st.session_state["last_price_log"] = plog
        st.sidebar.success("Priser uppdaterade för alla.")
        st.rerun()

    st.sidebar.subheader("🛠️ Auto-uppdatering (alla)")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True)
    force_ts = st.sidebar.checkbox("Stämpla datum även om värdet ej ändras", value=True)
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → Finnhub → FMP)"):
        df2, log = auto_update_all(df.copy(), user_rates, make_snapshot=make_snapshot, force_ts=force_ts)
        st.session_state["last_auto_log"] = log
        st.rerun()

    st.sidebar.subheader("🌊 Vågkörning (batch)")
    mode = st.sidebar.selectbox("Kö-ordning", ["Äldst först","A–Ö"], index=0)
    bsize = st.sidebar.number_input("Batchstorlek", min_value=1, max_value=100, value=10, step=1)
    colw1, colw2, colw3 = st.sidebar.columns(3)
    with colw1:
        if st.button("Starta kö"):
            start_wave(df, mode="oldest" if mode.startswith("Äldst") else "alpha")
            st.success("Kö skapad.")
    with colw2:
        if st.button("Kör nästa batch"):
            df2, stat = run_wave_step(df.copy(), user_rates, batch_size=int(bsize), make_snapshot=False, force_ts=force_ts)
            spara_data(df2, do_snapshot=False)
            st.sidebar.success(f"Körde {stat['processed']} tickers. Kvar: {stat['remaining']}.")
            st.rerun()
    with colw3:
        if st.button("Töm kö"):
            st.session_state.pop("wave_queue", None)
            st.session_state.pop("wave_pos", None)
            st.success("Kö återställd.")

    # --- Huvudmeny -----------------------------------------------------------
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
