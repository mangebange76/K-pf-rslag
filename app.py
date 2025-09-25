# app.py — Del 1/8
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

# === FAILSAFE: spara_data som inte kan tömma databasen =======================
def spara_data(df: pd.DataFrame, do_snapshot: bool = False, forbid_empty: bool = True, soft_overwrite: bool = True):
    """
    Skriv hela DataFrame till huvudbladet – säkert.
    - Skriver INTE om df är tom eller saknar tickers (forbid_empty=True).
    - Soft overwrite: vi kallar inte clear() först; vi uppdaterar header + data.
      (Hellre lämna gamla resterande rader än att råka tömma bladet.)
    """
    # 0) Grundkoll
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("Sparning avbruten: ogiltig DataFrame.")
        return
    if forbid_empty:
        if len(df) == 0:
            st.warning("Sparning avbruten: tom DataFrame (inget skrivs).")
            return
        has_any_ticker = ("Ticker" in df.columns) and df["Ticker"].astype(str).str.strip().ne("").any()
        if not has_any_ticker:
            st.warning("Sparning avbruten: inga tickers i data (inget skrivs).")
            return

    # 1) Snapshot först om valt
    if do_snapshot:
        try:
            backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)
        except Exception as e:
            st.warning(f"Kunde inte skapa snapshot före skrivning: {e}")

    # 2) Skriv (soft overwrite = ingen clear först)
    sheet = skapa_koppling()
    values = [df.columns.values.tolist()] + df.astype(str).values.tolist()
    try:
        if soft_overwrite:
            _with_backoff(sheet.update, values)  # skriv utan clear
        else:
            _with_backoff(sheet.clear)
            _with_backoff(sheet.update, values)
    except Exception as e:
        st.error(f"Skrivning misslyckades: {e}")

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

# app.py — Del 2/8
# --- Kolumnschema & tidsstämplar --------------------------------------------

# Spårade fält → respektive TS-kolumn (uppdateras när fältet ändras)
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

# Slutlig kolumnlista i databasen (lägg till nya fält här)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S-snitt",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "Beta", "Sektor",

    # Market cap (nu & historik)
    "MCap (nu)",
    "MCap Q1","MCap Q2","MCap Q3","MCap Q4",
    "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",

    # Balans/kassaflöde (kvartals/TTM där så anges)
    "Kassa", "Total skuld",
    "Operativt kassaflöde (Q)", "CapEx (Q)", "Fritt kassaflöde (Q)",
    "Burn rate (Q)", "Runway (kvartal)",
    "Operating Expense (Q)", "FoU (Q)", "SG&A (Q)",
    "EBITDA (TTM)", "Räntekostnad (TTM)",
    "Current assets", "Current liabilities",

    # Tidsstämplar & källor
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    numeric_like = {
        "Utestående aktier","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","Beta",
        "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
        "Kassa","Total skuld","Operativt kassaflöde (Q)","CapEx (Q)","Fritt kassaflöde (Q)",
        "Burn rate (Q)","Runway (kvartal)","Operating Expense (Q)","FoU (Q)","SG&A (Q)",
        "EBITDA (TTM)","Räntekostnad (TTM)","Current assets","Current liabilities"
    }
    text_like = {
        "Ticker","Bolagsnamn","Valuta","Sektor",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"
    }
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if kol in numeric_like:
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in text_like:
                df[kol] = ""
            else:
                # default text för allt övrigt (t.ex. nya textfält)
                df[kol] = ""
    # ta bort ev dubbletter
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
        "Utestående aktier","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","Beta",
        "MCap (nu)","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
        "Kassa","Total skuld","Operativt kassaflöde (Q)","CapEx (Q)","Fritt kassaflöde (Q)",
        "Burn rate (Q)","Runway (kvartal)","Operating Expense (Q)","FoU (Q)","SG&A (Q)",
        "EBITDA (TTM)","Räntekostnad (TTM)","Current assets","Current liabilities"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    str_cols = ["Ticker","Bolagsnamn","Valuta","Sektor","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
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

# app.py — Del 3/8
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR, Beta, P/S (Yahoo), MCap (nu), Sektor.
    P/S (Yahoo) = priceToSalesTrailing12Months (om tillgänglig).
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "Beta": 0.0,
        "P/S (Yahoo)": 0.0,
        "MCap (nu)": 0.0,
        "Sektor": ""
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

        # Utdelning (årstakt)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # Beta
        beta = info.get("beta") or info.get("beta3Year") or info.get("beta5Year")
        try:
            if beta is not None:
                out["Beta"] = float(beta)
        except Exception:
            pass

        # P/S (Yahoo)
        ps_y = info.get("priceToSalesTrailing12Months") or info.get("priceToSalesRatio") or info.get("priceToSales")
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

        # Sektor
        sector = info.get("sector") or info.get("industry")
        if sector:
            out["Sektor"] = str(sector)

        # CAGR (5 år) approx
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier (milj)
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if pd.notna(x) and float(x) > 0]
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
            # behåll ev tidigare
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        aktier_ut_milj = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut_milj > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut_milj, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut_milj, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut_milj, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut_milj, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    force_ts: bool = False
) -> bool:
    """
    Skriver endast fält som får ett nytt (meningsfullt) värde.
    - Om force_ts=True: stämpla TS för spårade fält även om värdet inte ändras.
    - Sätter 'Senast auto-uppdaterad' + källa om något fält skrevs/stämlades.
    Returnerar True om något fält faktiskt ÄNDRADES (värdet).
    """
    changed_fields = []
    ts_touched = False
    tkr = str(df.at[row_idx, "Ticker"]) if "Ticker" in df.columns else ""

    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue

        # Bestäm om värdet är "skrivbart"
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # för numeriska: tillåt >= 0 (P/S etc kan vara 0 i edgefall)
            try:
                write_ok = (float(v) >= 0.0)
            except Exception:
                write_ok = False
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            # andra typer ignoreras
            write_ok = False

        if not write_ok and not force_ts:
            continue

        old = df.at[row_idx, f] if f in df.columns else None
        is_different = (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))

        # Skriv nytt värde om det skiljer sig eller om det saknas
        if write_ok and is_different:
            df.at[row_idx, f] = v
            changed_fields.append(f)
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                ts_touched = True
        else:
            # ingen faktisk ändring; men om force_ts och f är spårat fält, stämpla TS ändå
            if force_ts and f in TS_FIELDS:
                _stamp_ts_for_field(df, row_idx, f)
                ts_touched = True

    # Om något ändrats eller TS stämplats → uppdatera auto-meta
    if changed_fields or ts_touched:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(tkr, []).extend(changed_fields)
        return bool(changed_fields)

    return False

# app.py — Del 4/8
# --- Datakällor: FMP, SEC (US + IFRS/6-K), Yahoo global fallback -------------

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

@st.cache_data(show_spinner=False, ttl=1800)
def hamta_fmp_falt(yahoo_ticker: str) -> dict:
    """
    Fullare variant: försöker hämta namn/valuta/pris/shares, P/S (TTM),
    P/S Q1–Q4 (ratios quarterly) samt analytikerestimat (om plan tillåter).
    """
    out = {"_debug": {}}
    sym = _fmp_pick_symbol(yahoo_ticker)
    out["_symbol"] = sym

    # Profile (namn/valuta/price/shares)
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

    # Quote (pris + marketCap)
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

    # Shares fallback
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

    # P/S via ratios-ttm
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

    # P/S Q1–Q4 (FMP ratios-quarterly om tillgängligt)
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

# ---------- IFRS/GAAP kvartalsintäkter + valuta med Q4-syntes ----------------
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

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20, synth_q4: bool = True):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    För US-GAAP försöker vi även SYNTA Q4 från 10-K: Q4 = FY - (Q1+Q2+Q3) om Q4 saknas.
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    """
    # gemensamma parametrar
    rev_keys = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    prefer_units = ("USD","CAD","EUR","GBP")

    def _collect_entries(taxo: str):
        """Samlar duration-poster för givna taxo/koncept; returnerar dict per unit."""
        results_per_unit = {u: [] for u in prefer_units}
        facts_all = (facts.get("facts") or {}).get(taxo, {})
        for name in rev_keys:
            fact = facts_all.get(name)
            if not fact:
                continue
            units = (fact.get("units") or {})
            for unit_code in prefer_units:
                arr = units.get(unit_code)
                if not isinstance(arr, list):
                    continue
                for it in arr:
                    end = _parse_iso(str(it.get("end", "")))
                    start = _parse_iso(str(it.get("start", "")))
                    val = it.get("val", None)
                    form = (it.get("form") or "").upper()
                    if not (end and start and val is not None):
                        continue
                    try:
                        dur = (end - start).days
                        v = float(val)
                    except Exception:
                        continue
                    results_per_unit[unit_code].append({
                        "end": end, "start": start, "dur": dur, "val": v, "form": form, "concept": name
                    })
        return results_per_unit

    def _dedupe_by_end(entries: list):
        """Behåll en post per end-datum (tar den med högst val)."""
        by_end = {}
        for e in entries:
            k = e["end"]
            if k not in by_end or (e["val"] > by_end[k]["val"]):
                by_end[k] = e
        out = list(by_end.values())
        out.sort(key=lambda x: x["end"], reverse=True)
        return out

    # ---------- US-GAAP med syntetisk Q4 via 10-K ----------
    gaap = _collect_entries("us-gaap")
    for unit_code in prefer_units:
        arr = gaap.get(unit_code, [])
        if not arr:
            continue
        # Kvartal ≈ 70–110 dagar
        q = [e for e in arr if 70 <= (e["dur"] or 0) <= 110 and ("10-Q" in e["form"] or "10-Q/A" in e["form"])]
        q = _dedupe_by_end(q)

        # År ≈ 330–400 dagar (10-K)
        fy = [e for e in arr if 330 <= (e["dur"] or 0) <= 400 and ("10-K" in e["form"] or "10-K/A" in e["form"])]
        fy = _dedupe_by_end(fy)

        # Försök synta senaste Q4 om saknas en kvartalspost för FY-änden
        if synth_q4 and fy:
            last_fy = fy[0]  # senaste FY
            fy_end = last_fy["end"]
            fy_val = last_fy["val"]

            # Har vi redan en kvartalspost med exakt samma end? (då finns Q4)
            has_q4 = any(e["end"] == fy_end for e in q)

            if not has_q4 and fy_val and fy_val > 0:
                # Plocka upp till tre kvartal i samma FY
                three_q = []
                for e in q:
                    if e["end"] <= fy_end and (fy_end - e["start"]).days <= 400:
                        three_q.append(e)
                    if len(three_q) >= 3:
                        break
                if len(three_q) >= 3:
                    s = sum(x["val"] for x in three_q[:3])
                    q4_val = fy_val - s
                    try:
                        med = np.median([x["val"] for x in three_q[:3]])
                    except Exception:
                        med = None
                    ok = (q4_val > 0)
                    if med and med > 0:
                        ok = ok and (q4_val < med * 6.0)  # outlier guard
                    if ok:
                        q4 = {"end": fy_end, "start": three_q[0]["end"], "dur": 90, "val": float(q4_val), "form": "SYNTH_Q4", "concept": "us-gaap:Q4Synth"}
                        q.insert(0, q4)  # gör Q4 till senaste

        if q:
            rows = [(e["end"], float(e["val"])) for e in q][:max_quarters]
            return rows, unit_code

    # ---------- IFRS (6-K/10-Q) ----------
    ifrs = _collect_entries("ifrs-full")
    for unit_code in prefer_units:
        arr = ifrs.get(unit_code, [])
        if not arr:
            continue
        q = [e for e in arr if 70 <= (e["dur"] or 0) <= 110 and any(f in e["form"] for f in ("6-K","6-K/A","10-Q","10-Q/A"))]
        q = _dedupe_by_end(q)
        if q:
            rows = [(e["end"], float(e["val"])) for e in q][:max_quarters]
            return rows, unit_code

    # Inga träffar
    return [], None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters, synth_q4=True)
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
    US/FPIs: Shares + kvartalsintäkter från SEC (US-GAAP 10-Q eller IFRS 6-K),
    pris/valuta/namn från Yahoo. P/S (TTM) nu + P/S Q1–Q4 historik.
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "MCap (nu)", "Sektor", "Beta", "P/S (Yahoo)"):
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
        out["Utestående aktier"] = shares_used / 1e6  # vi lagrar i miljoner

    # Market cap (nu)
    mcap_now = float(out.get("MCap (nu)", 0.0) or 0.0)
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares_used > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares_used
        out["MCap (nu)"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20, synth_q4=True)
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
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/mcap/ps_yahoo/sector/beta
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","MCap (nu)","Sektor","Beta","P/S (Yahoo)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(out.get("MCap (nu)", 0.0) or 0.0)

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
        out["MCap (nu)"] = mcap

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

# app.py — Del 5/8
# --- Snapshots, auto-uppdatering, batch & kassaflöde/balans ------------------

# =============== Finnhub (valfritt, estimat) =================================
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets. Hämtar annual revenue estimates:
    current FY + next FY (om finns). Returnerar MSEK.
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
                if v and float(v) != 0:
                    out["Omsättning idag"] = float(v) / 1e6
            except Exception:
                pass
        if len(last_two) == 2:
            v = last_two[1].get("revenueAvg") or last_two[1].get("revenueMean") or last_two[1].get("revenue")
            try:
                if v and float(v) != 0:
                    out["Omsättning nästa år"] = float(v) / 1e6
            except Exception:
                pass
        return out
    except Exception:
        return {}

# =============== Snapshot & "äldsta TS" ======================================
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

# =============== Yahoo CF/BS-hämtning (kvartal) ==============================
def _yahoo_quarterly_cf_bs(ticker: str) -> dict:
    """
    Hämtar senaste kvartalens kassaflöde & balans via yfinance:
      - Operativt kassaflöde (Q), CapEx (Q), Fritt kassaflöde (Q)
      - Operating Expense (Q), FoU (Q), SG&A (Q)
      - EBITDA (TTM, om tillgängligt), Räntekostnad (TTM approx)
      - Kassa, Total skuld, Current assets, Current liabilities
      - Burn rate (Q) & Runway (kvartal) = Kassa / max(1, -FCF_Q) om FCF_Q<0, annars None
    Alla belopp i basvalutan från Yahoo (ingen SEK-konvertering här).
    """
    out = {}
    try:
        t = yf.Ticker(ticker)

        # --- Cashflow (kvartal)
        try:
            qcf = t.quarterly_cashflow
            if isinstance(qcf, pd.DataFrame) and not qcf.empty:
                col = qcf.columns[0]  # senaste kvartal
                def get_cf(*names):
                    for n in names:
                        if n in qcf.index:
                            try:
                                return float(qcf.loc[n, col])
                            except Exception:
                                pass
                    return 0.0
                cfo = get_cf("Total Cash From Operating Activities", "Operating Cash Flow", "NetCashProvidedByUsedInOperatingActivities")
                capex = get_cf("Capital Expenditures", "InvestmentsInPropertyPlantAndEquipment")
                # CapEx är oftast negativt; FCF = CFO - CapEx
                fcf = cfo - capex if (cfo is not None and capex is not None) else 0.0

                out["Operativt kassaflöde (Q)"] = float(cfo or 0.0)
                out["CapEx (Q)"] = float(capex or 0.0)
                out["Fritt kassaflöde (Q)"] = float(fcf or 0.0)
        except Exception:
            pass

        # --- Financials (kvartal) för Opex / R&D / SG&A & Interest expense
        qfin = None
        try:
            qfin = t.quarterly_financials
        except Exception:
            qfin = None
        if isinstance(qfin, pd.DataFrame) and not qfin.empty:
            col = qfin.columns[0]
            def get_fin(*names):
                for n in names:
                    if n in qfin.index:
                        try:
                            return float(qfin.loc[n, col])
                        except Exception:
                            pass
                return 0.0
            opex = get_fin("Operating Expense","OperatingExpenses","Total Operating Expenses")
            rnd  = get_fin("Research Development","Research And Development")
            sga  = get_fin("Selling General Administrative","Selling General And Administrative")
            out["Operating Expense (Q)"] = float(opex or 0.0)
            out["FoU (Q)"] = float(rnd or 0.0)
            out["SG&A (Q)"] = float(sga or 0.0)

            # Räntekostnad (TTM approx) – summera 4 kvartal om möjligt
            try:
                interest_row_names = ["Interest Expense", "InterestExpense"]
                interest_ttm = 0.0
                for c in qfin.columns[:4]:
                    val = 0.0
                    for nm in interest_row_names:
                        if nm in qfin.index and pd.notna(qfin.loc[nm, c]):
                            try: val = float(qfin.loc[nm, c]); break
                            except: pass
                    interest_ttm += float(val or 0.0)
                out["Räntekostnad (TTM)"] = float(interest_ttm)
            except Exception:
                pass

        # --- EBITDA (TTM) från info om tillgängligt
        try:
            info = t.info or {}
            ebitda = info.get("ebitda")
            if ebitda is not None:
                out["EBITDA (TTM)"] = float(ebitda)
        except Exception:
            pass

        # --- Balans (kvartal)
        try:
            qbs = t.quarterly_balance_sheet
            if isinstance(qbs, pd.DataFrame) and not qbs.empty:
                col = qbs.columns[0]
                def get_bs(*names):
                    for n in names:
                        if n in qbs.index and pd.notna(qbs.loc[n, col]):
                            try: return float(qbs.loc[n, col])
                            except Exception: pass
                    return 0.0
                cash = get_bs("Cash And Cash Equivalents","Cash","Cash And Short Term Investments")
                debt_total = get_bs("Total Debt","Short Long Term Debt") + get_bs("Long Term Debt")
                cur_assets = get_bs("Total Current Assets","Current Assets")
                cur_liab   = get_bs("Total Current Liabilities","Current Liabilities")

                out["Kassa"] = float(cash or 0.0)
                out["Total skuld"] = float(debt_total or 0.0)
                out["Current assets"] = float(cur_assets or 0.0)
                out["Current liabilities"] = float(cur_liab or 0.0)
        except Exception:
            pass

        # --- Burn & Runway
        try:
            fcf_q = float(out.get("Fritt kassaflöde (Q)", 0.0))
            cash  = float(out.get("Kassa", 0.0))
            if fcf_q < 0:
                out["Burn rate (Q)"] = abs(fcf_q)
                out["Runway (kvartal)"] = round(cash / max(1.0, abs(fcf_q)), 2) if cash > 0 else 0.0
            else:
                out["Burn rate (Q)"] = 0.0
                out["Runway (kvartal)"] = 0.0
        except Exception:
            pass

    except Exception:
        pass
    return out

# =============== Hjälpare ====================================================
def _find_row_idx_by_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    if "Ticker" not in df.columns:
        return None
    t = str(ticker).upper().strip()
    hits = df.index[df["Ticker"].astype(str).str.upper().str.strip() == t]
    if len(hits) == 0:
        return None
    return int(hits[0])

def _format_big_number(n: float) -> str:
    """Snygg formattering av market cap o.dyl."""
    try:
        x = float(n)
    except Exception:
        return "0"
    absx = abs(x)
    if absx >= 1e12:
        return f"{x/1e12:.2f} tn"
    if absx >= 1e9:
        return f"{x/1e9:.2f} md"
    if absx >= 1e6:
        return f"{x/1e6:.2f} mn"
    if absx >= 1e3:
        return f"{x/1e3:.2f} k"
    return f"{x:.0f}"

# =============== Auto-fetch pipeline =========================================
def auto_fetch_for_ticker(ticker: str, want_mcap_hist: bool = True):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares & Q4-syntes) ELLER Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S) om saknas
      4) Yahoo CF/BS (kassa/skuld/CF/FCF/OpEx/R&D/SG&A/EBITDA/Interest/CA/CL; burn & runway)
      5) (Valfritt) MCap Q1–Q4 från P/S Qn * TTM_rev_n ELLER shares*pris_hist

    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. Q4-syntes) eller Yahoo fallback global
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","_debug_shares_source",
            "MCap (nu)","Sektor","Beta","P/S (Yahoo)"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                  "MCap (nu)","Sektor","Beta","P/S (Yahoo)"]:
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

    # 4) Yahoo CF/BS
    try:
        ycfbs = _yahoo_quarterly_cf_bs(ticker)
        debug["yahoo_cf_bs"] = ycfbs
        for k in ["Kassa","Total skuld","Operativt kassaflöde (Q)","CapEx (Q)","Fritt kassaflöde (Q)",
                  "Operating Expense (Q)","FoU (Q)","SG&A (Q)","EBITDA (TTM)","Räntekostnad (TTM)",
                  "Current assets","Current liabilities","Burn rate (Q)","Runway (kvartal)"]:
            v = ycfbs.get(k, None)
            if v not in (None, ""):
                # 0-värden får skrivas – de är informativa
                vals[k] = float(v)
    except Exception as e:
        debug["yahoo_cf_bs_err"] = str(e)

    # 5) MCap-historik Q1–Q4 (för investeringsförslag)
    try:
        if want_mcap_hist:
            # Försök SEC-baserat först (TTM fönster + historiska priser)
            cik = _sec_cik_for(ticker)
            shares_used = 0.0
            try:
                # implied shares från Yahoo (mark cap / price) om finns
                if "MCap (nu)" in vals and "Aktuell kurs" in vals and vals.get("Aktuell kurs", 0) > 0:
                    shares_used = float(vals["MCap (nu)"]) / float(vals["Aktuell kurs"])
                elif "Utestående aktier" in vals and vals.get("Utestående aktier", 0) > 0:
                    shares_used = float(vals["Utestående aktier"]) * 1e6
            except Exception:
                shares_used = 0.0

            dates_for_labels = []
            mcap_hist = []

            if cik:
                facts, sc = _sec_companyfacts(cik)
                if sc == 200 and isinstance(facts, dict):
                    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=8, synth_q4=True)
                    if q_rows and rev_unit:
                        px_ccy = (vals.get("Valuta") or "USD").upper()
                        conv = 1.0 if rev_unit.upper() == px_ccy else (_fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0)
                        ttm_list = _ttm_windows(q_rows, need=4)
                        ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]
                        # Om P/S Qn finns → MCap = PS * TTM_rev
                        if ttm_list_px:
                            for idx in range(4):
                                d_end, ttm_rev = ttm_list_px[idx] if idx < len(ttm_list_px) else (None, None)
                                psn = vals.get(f"P/S Q{idx+1}")
                                if d_end and ttm_rev and psn and float(psn) > 0:
                                    mcap_hist.append(float(psn) * float(ttm_rev))
                                    dates_for_labels.append(d_end.strftime("%Y-%m-%d"))
            # Om ovan inte gav något, försök shares*pris (historiskt)
            if (not mcap_hist) and shares_used > 0:
                # använd Yahoo-priser kring samma Q-datum som P/S Qn (om vi kan hitta dem)
                # fallback: använd de 4 senaste kvartalsperioderna från Yahoo quarterly_financials
                t = yf.Ticker(ticker)
                q_rows = _yfi_quarterly_revenues(t)
                if len(q_rows) >= 4:
                    ends = [d for (d, _) in q_rows[:4]]
                    px_map = _yahoo_prices_for_dates(ticker, ends)
                    for d in ends:
                        p = px_map.get(d)
                        if p and p > 0:
                            mcap_hist.append(float(p) * float(shares_used))
                            dates_for_labels.append(d.strftime("%Y-%m-%d"))

            # Skriv in (senaste först som Q1)
            if mcap_hist:
                for i, v in enumerate(mcap_hist[:4], start=1):
                    vals[f"MCap Q{i}"] = float(v)
                    if i-1 < len(dates_for_labels):
                        vals[f"MCap Datum Q{i}"] = dates_for_labels[i-1]
    except Exception as e:
        debug["mcap_hist_err"] = str(e)

    return vals, debug

# =============== Auto-uppdatera ALLA (försiktig skrivning) ===================
def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, force_ts: bool = False):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla nya värden.
    Stämplar TS_ per fält (om ändrat eller force_ts), samt 'Senast auto-uppdaterad' + källa.
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
            changed = apply_auto_updates_to_row(df, idx, new_vals, source="Auto (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)", changes_map=log["changed"], force_ts=force_ts)
            if not changed and not force_ts:
                log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
            any_changed = any_changed or changed
            if i < 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(total,1))

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    # Säker skrivning
    if any_changed or force_ts:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Ändringar sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring – ingen skrivning.")

    return df, log

# =============== Batch-körning (robust) ======================================
def batchvy(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Kör delmängder i batch, baserat på TICKER-listor (stabilt över skrivningar)."""
    st.header("🧵 Batch-körning")

    sort_mode = st.selectbox("Ordning", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], index=0, key="batch_sort_mode")
    batch_size = st.number_input("Hur många per körning?", min_value=1, max_value=200, value=10, step=1, key="batch_size")
    force_ts = st.checkbox("Tidsstämpla även oförändrade fält", value=True, key="batch_force_ts")
    snapshot_before = st.checkbox("Skapa snapshot före skrivning", value=False, key="batch_snapshot")

    # Bygg TICKER-ordning (inte index)
    if sort_mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"]).reset_index(drop=True)
    else:
        work = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)

    ordered_tickers = [str(t).upper().strip() for t in work["Ticker"].astype(str).tolist() if str(t).strip()]
    key_prefix = f"batch2_{'oldest' if sort_mode.startswith('Äldst') else 'az'}"

    # Init / resync
    if f"{key_prefix}_order" not in st.session_state:
        st.session_state[f"{key_prefix}_order"] = ordered_tickers
        st.session_state[f"{key_prefix}_pos"] = 0
    else:
        old_order = st.session_state[f"{key_prefix}_order"]
        if old_order != ordered_tickers:
            pos = int(st.session_state.get(f"{key_prefix}_pos", 0))
            processed = set(old_order[:pos])
            new_pos = sum(1 for t in ordered_tickers if t in processed)
            st.session_state[f"{key_prefix}_order"] = ordered_tickers
            st.session_state[f"{key_prefix}_pos"] = new_pos

    pos = int(st.session_state.get(f"{key_prefix}_pos", 0))
    start = pos
    stop = min(start + int(batch_size), len(st.session_state[f"{key_prefix}_order"]))
    window_tickers = st.session_state[f"{key_prefix}_order"][start:stop]

    st.caption(f"Förhandsvisar {start+1}–{stop} av {len(st.session_state[f'{key_prefix}_order'])}")

    # Lista fönsterinnehåll
    if window_tickers:
        rows = []
        for t in window_tickers:
            idx = _find_row_idx_by_ticker(df, t)
            if idx is not None:
                rows.append(df.loc[idx, ["Ticker","Bolagsnamn","Senast auto-uppdaterad","Senast manuellt uppdaterad"]])
            else:
                rows.append(pd.Series({"Ticker": t, "Bolagsnamn": "(saknas)", "Senast auto-uppdaterad": "", "Senast manuellt uppdaterad": ""}))
        vis = pd.DataFrame(rows)
        st.dataframe(vis, use_container_width=True, hide_index=True)
    else:
        st.info("Inga fler poster i denna batch-sekvens.")

    col_run, col_next, col_reset = st.columns([1,1,1])
    run_clicked   = col_run.button("🚀 Kör denna batch")
    next_clicked  = col_next.button("➡️ Nästa fönster")
    reset_clicked = col_reset.button("🔁 Återställ position")

    if reset_clicked:
        st.session_state[f"{key_prefix}_pos"] = 0
        st.info("Batch-position återställd till början.")

    changed_any = False
    log_local = {"changed": {}, "misses": {}}

    if run_clicked and window_tickers:
        progress = st.progress(0.0)
        status = st.empty()
        for j, tkr in enumerate(window_tickers, start=1):
            status.write(f"Uppdaterar: {tkr} ({j}/{len(window_tickers)})")
            idx = _find_row_idx_by_ticker(df, tkr)
            try:
                if idx is None:
                    log_local["misses"][tkr] = ["(ticker saknas i DF)"]
                else:
                    new_vals, debug = auto_fetch_for_ticker(tkr)
                    changed = apply_auto_updates_to_row(
                        df, idx, new_vals,
                        source="Batch (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)",
                        changes_map=log_local["changed"],
                        force_ts=force_ts
                    )
                    if not changed and not force_ts:
                        log_local["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
                    changed_any = changed_any or changed
            except Exception as e:
                log_local["misses"][tkr] = [f"error: {e}"]
            progress.progress(j/len(window_tickers))

        df = uppdatera_berakningar(df, user_rates)

        if changed_any or force_ts:
            spara_data(df, do_snapshot=snapshot_before)
            st.success("Batch sparad till Google Sheets.")
        else:
            st.info("Inga ändringar – ingen skrivning.")

        st.session_state[f"{key_prefix}_pos"] = stop
        with st.expander("Visa batch-körlogg"):
            st.json(log_local)

    if next_clicked:
        st.session_state[f"{key_prefix}_pos"] = stop
        (getattr(st, "rerun", None) or st.experimental_rerun)()

    return df

# app.py — Del 6/8
# --- Kontroll, Analys, Portfölj & Investeringsförslag ------------------------

def _risk_label_from_mcap(mcap: float) -> str:
    try:
        x = float(mcap or 0.0)
    except Exception:
        x = 0.0
    if x >= 200e9: return "Mega"
    if x >= 10e9:  return "Large"
    if x >= 2e9:   return "Mid"
    if x >= 300e6: return "Small"
    if x >= 50e6:  return "Micro"
    if x > 0:      return "Nano"
    return "Okänd"

def _ps_avg4(row: pd.Series) -> float:
    vals = [row.get("P/S Q1",0), row.get("P/S Q2",0), row.get("P/S Q3",0), row.get("P/S Q4",0)]
    clean = [float(x) for x in vals if pd.notna(x) and float(x) > 0]
    return round(float(np.mean(clean)) if clean else 0.0, 2)

def _build_requires_manual_prognos_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Rapport för fält där prognoser ofta saknas:
    - 'Omsättning idag' och 'Omsättning nästa år'
    Tar fram de bolag där TS för dessa fält är äldst, eller saknas.
    """
    targets = ["Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS[t] for t in targets]
    rows = []
    cut = now_dt() - timedelta(days=older_than_days)

    for _, r in df.iterrows():
        oldest = None
        any_missing_ts = False
        any_zero_val = False

        for t, ts in zip(targets, ts_cols):
            val = float(r.get(t, 0.0))
            if val <= 0.0:
                any_zero_val = True
            ts_str = str(r.get(ts, "")).strip()
            if ts_str:
                ts_dt = pd.to_datetime(ts_str, errors="coerce")
                if pd.notna(ts_dt):
                    if (oldest is None) or (ts_dt < oldest):
                        oldest = ts_dt
            else:
                any_missing_ts = True

        too_old = (oldest is not None and oldest.to_pydatetime() < cut)
        if any_zero_val or any_missing_ts or too_old:
            rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "TS_Omsättning idag": str(r.get("TS_Omsättning idag","")),
                "TS_Omsättning nästa år": str(r.get("TS_Omsättning nästa år","")),
                "Kommentar": "Saknar värde/TS" if (any_zero_val or any_missing_ts) else "Gammal TS",
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["TS_Omsättning idag","TS_Omsättning nästa år","Bolagsnamn"])
    return out

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
    need = _build_requires_manual_prognos_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell prognosuppdatering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell uppdatering (prognoser).")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Senaste körlogg (om du nyss körde Auto/Batch)
    st.subheader("📒 Senaste körlogg")
    log = st.session_state.get("last_auto_log") or st.session_state.get("last_batch_log")
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
        if log.get("debug_first_20"):
            st.markdown("**Debug (första 20)**")
            st.json(log.get("debug_first_20", []))

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1, key="analys_num")
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
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","MCap (nu)","Sektor","Beta",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    show = pd.DataFrame([r[cols].to_dict()])
    if "MCap (nu)" in show.columns:
        show["MCap (nu)"] = show["MCap (nu)"].apply(_format_big_number)
    st.dataframe(show, use_container_width=True, hide_index=True)

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)
    port["Andel (%)"] = port["Andel (%)"].round(2)
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

    # --- Filter/rattar
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0, key="if_kapital")

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1, key="if_rikt"
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True, key="if_subset")
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True, key="if_sort")

    # Sektorfilter
    sektor_list = sorted([s for s in df.get("Sektor", pd.Series(dtype=str)).dropna().unique().tolist() if str(s).strip()])
    valda_sektorer = st.multiselect("Filtrera på sektor", options=sektor_list, default=sektor_list, key="if_sektor")

    # Cap-filter (risklabel)
    cap_options = ["Mega","Large","Mid","Small","Micro","Nano","Okänd"]
    valda_caps = st.multiselect("Filtrera på cap-klass (risklabel)", options=cap_options, default=cap_options, key="if_caps")

    # --- Urval
    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Risklabel och sektorfilter
    base["MCap (nu)"] = pd.to_numeric(base.get("MCap (nu)", 0.0), errors="coerce").fillna(0.0)
    base["Risklabel"] = base["MCap (nu)"].apply(_risk_label_from_mcap)
    if valda_sektorer:
        base = base[base["Sektor"].isin(valda_sektorer)]
    if valda_caps:
        base = base[base["Risklabel"].isin(valda_caps)]
    if base.empty:
        st.info("Inga bolag kvar efter filtrering.")
        return

    # --- Nyckeltal
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0
    base["P/S 4Q-snitt"] = base.apply(_ps_avg4, axis=1)

    # sortera
    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # robust bläddring
    n = len(base)
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = int(np.clip(st.session_state.forslags_index, 0, max(0, n-1)))

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag", key="if_prev"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{n}")
    with col_next:
        if st.button("➡️ Nästa förslag", key="if_next"):
            st.session_state.forslags_index = min(n-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Valuta → SEK
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    # portföljandelar
    port = df[df["Antal aktier"] > 0].copy()
    port_värde = 0.0
    nuv_innehav = 0.0
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
        rsel = port[port["Ticker"] == rad["Ticker"]]
        if not rsel.empty:
            nuv_innehav = float(rsel["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    # övergripande info
    ps_now = float(rad.get("P/S", 0.0))
    ps_avg = float(rad.get("P/S 4Q-snitt", 0.0))
    mcap_now_fmt = _format_big_number(rad.get("MCap (nu)", 0.0))

    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Risklabel:** {rad.get('Risklabel','Okänd')}  ·  **Sektor:** {rad.get('Sektor','')}",
        f"- **Market cap (nu):** {mcap_now_fmt}",
        f"- **P/S (nu):** {round(ps_now,2)}  ·  **P/S 4Q-snitt:** {round(ps_avg,2)}",
        f"- **Riktkurs (vald):** {round(rad[riktkurs_val],2)} {rad['Valuta']}",
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %  ·  **Andel efter köp:** {ny_andel} %",
    ]
    st.markdown("\n".join(lines))

    with st.expander("Visa finansiella detaljer (senaste Q/TTM)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Balans/Kassa**")
            st.write(f"Kassa: {_format_big_number(rad.get('Kassa',0.0))}")
            st.write(f"Total skuld: {_format_big_number(rad.get('Total skuld',0.0))}")
            st.write(f"Current assets: {_format_big_number(rad.get('Current assets',0.0))}")
            st.write(f"Current liabilities: {_format_big_number(rad.get('Current liabilities',0.0))}")
        with c2:
            st.markdown("**Kassaflöde**")
            st.write(f"Operativt kassaflöde (Q): {_format_big_number(rad.get('Operativt kassaflöde (Q)',0.0))}")
            st.write(f"CapEx (Q): {_format_big_number(rad.get('CapEx (Q)',0.0))}")
            st.write(f"Fritt kassaflöde (Q): {_format_big_number(rad.get('Fritt kassaflöde (Q)',0.0))}")
            st.write(f"Burn rate (Q): {_format_big_number(rad.get('Burn rate (Q)',0.0))}")
            runway = rad.get("Runway (kvartal)", 0.0)
            st.write(f"Runway (kvartal): {runway if runway else 0}")
        with c3:
            st.markdown("**Resultat/TTM**")
            st.write(f"EBITDA (TTM): {_format_big_number(rad.get('EBITDA (TTM)',0.0))}")
            st.write(f"Räntekostnad (TTM): {_format_big_number(rad.get('Räntekostnad (TTM)',0.0))}")
            st.write(f"Beta: {round(float(rad.get('Beta',0.0)),2)}")

    with st.expander("Visa MCap-historik (Q1–Q4)"):
        mrows = []
        for i in range(1,5):
            mcap_i = rad.get(f"MCap Q{i}", 0.0)
            mcap_dt = rad.get(f"MCap Datum Q{i}", "")
            if mcap_i:
                mrows.append((f"Q{i}", mcap_dt, _format_big_number(mcap_i)))
        if mrows:
            md = pd.DataFrame(mrows, columns=["Q","Periodslut","Market cap (beräknad)"])
            st.dataframe(md, use_container_width=True, hide_index=True)
        else:
            st.info("Ingen MCap-historik beräknad ännu.")

# app.py — Del 7/8
# --- Lägg till/uppdatera + enskild uppdatering & TS-etiketter ----------------

def _ts_badge(ts_str: str, label: str) -> str:
    """Returnerar en enkel 'badge'-text för ett TS-fält."""
    ts = str(ts_str or "").strip()
    if not ts:
        return f"• {label}: saknas"
    today = now_stamp()
    if ts == today:
        return f"• {label}: {ts} ✅"
    return f"• {label}: {ts}"

def _render_ts_chips_for_row(r: pd.Series):
    chips = []
    mapping = [
        ("TS_Utestående aktier","Utestående aktier"),
        ("TS_P/S","P/S"),
        ("TS_P/S Q1","P/S Q1"),
        ("TS_P/S Q2","P/S Q2"),
        ("TS_P/S Q3","P/S Q3"),
        ("TS_P/S Q4","P/S Q4"),
        ("TS_Omsättning idag","Omsättning idag"),
        ("TS_Omsättning nästa år","Omsättning nästa år")
    ]
    for ts_col, label in mapping:
        if ts_col in r.index:
            chips.append(_ts_badge(r.get(ts_col,""), label))
    if chips:
        st.markdown("**Uppdateringsetiketter (fält):**\n\n" + "\n".join(chips))

def _update_only_price(df: pd.DataFrame, tkr: str) -> bool:
    """Uppdatera endast kurs/namn/valuta för given ticker. Returnerar True om värdet ändrades."""
    idx = _find_row_idx_by_ticker(df, tkr)
    if idx is None:
        st.warning(f"{tkr} hittades inte i tabellen.")
        return False
    y = hamta_yahoo_fält(tkr)
    changed = False
    # Kurs
    if y.get("Aktuell kurs", 0) > 0:
        old = float(df.at[idx, "Aktuell kurs"] or 0.0)
        new = float(y["Aktuell kurs"])
        if old != new:
            df.at[idx, "Aktuell kurs"] = new
            changed = True
    # Namn/valuta (om tomma)
    if y.get("Bolagsnamn"):
        if str(df.at[idx, "Bolagsnamn"] or "").strip() == "":
            df.at[idx, "Bolagsnamn"] = y["Bolagsnamn"]
    if y.get("Valuta"):
        if str(df.at[idx, "Valuta"] or "").strip() == "":
            df.at[idx, "Valuta"] = y["Valuta"]

    # Notera auto-uppdatering
    _note_auto_update(df, idx, source="Yahoo (pris)")
    return changed

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsval + datakälla för listan
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], key="edit_sort")
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    # Bläddringsindex robust
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = int(np.clip(st.session_state.edit_index, 0, max(0, len(val_lista)-1)))

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

    # Hämta befintlig rad (om vald)
    if valt_label and valt_label in namn_map:
        bef_mask = (df["Ticker"].astype(str).str.upper().str.strip() == namn_map[valt_label].upper())
        if not bef_mask.any():
            bef = pd.Series({}, dtype=object)
        else:
            bef = df[bef_mask].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    # Visuell etikett för när senaste auto/manuell skedde
    if not bef.empty:
        st.markdown(
            f"Senast **manuellt**: {str(bef.get('Senast manuellt uppdaterad','') or '–')} &nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Senast **auto**: {str(bef.get('Senast auto-uppdaterad','') or '–')} "
            f"(_källa:_ {str(bef.get('Senast uppdaterad källa','') or '–')})"
        )
        _render_ts_chips_for_row(bef)

    # Formulär
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0, step=0.01)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0, step=1.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0, step=1.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0, step=1.0)

            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över manuella 0-värden):**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

        spar = st.form_submit_button("💾 Spara formulär")

    # Hantera spara (manuell uppdatering)
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
        ridx = _find_row_idx_by_ticker(df, ticker)
        if ridx is not None:
            if datum_sätt:
                _note_manual_update(df, ridx)
                for f in changed_manual_fields:
                    _stamp_ts_for_field(df, ridx, f)

            # Hämta basfält från Yahoo (namn/valuta/kurs/utdelning/cagr) – skriv även om samma värde, men TS sätts inte här
            data = hamta_yahoo_fält(ticker)
            if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
            if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
            if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = float(data["Aktuell kurs"])
            if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # Knappar för enskild uppdatering (kurs & full auto)
    col_kurs, col_auto = st.columns([1,1])
    if col_kurs.button("📈 Uppdatera KURS för denna", key="edit_update_price"):
        if not bef.empty:
            tkr = str(bef.get("Ticker","")).upper().strip()
        else:
            tkr = ""
        if not tkr:
            st.warning("Välj ett befintligt bolag först.")
        else:
            changed = _update_only_price(df, tkr)
            df = uppdatera_berakningar(df, user_rates)
            spara_data(df)
            if changed:
                st.success(f"Kurs uppdaterad för {tkr}.")
            else:
                st.info(f"Ingen kursförändring för {tkr} (tidsstämpel och källa uppdaterade).")

    if col_auto.button("⚡ Full auto för denna (med tidsstämpling)", key="edit_full_auto"):
        if not bef.empty:
            tkr = str(bef.get("Ticker","")).upper().strip()
        else:
            tkr = ""
        if not tkr:
            st.warning("Välj ett befintligt bolag först.")
        else:
            ridx = _find_row_idx_by_ticker(df, tkr)
            if ridx is None:
                st.warning(f"{tkr} hittades inte i tabellen.")
            else:
                new_vals, debug = auto_fetch_for_ticker(tkr)
                changed = apply_auto_updates_to_row(
                    df, ridx, new_vals,
                    source="Enskild auto (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)",
                    changes_map={},
                    force_ts=True  # stämpla även om värdet råkar bli detsamma
                )
                df = uppdatera_berakningar(df, user_rates)
                spara_data(df)
                if changed:
                    st.success(f"Auto-uppdatering klar för {tkr}.")
                else:
                    st.info(f"Inga värdeändringar för {tkr} – men TS stämplades.")

    # Visa topp 10 äldsta TS för överblick i denna vy
    st.markdown("### ⏱️ Äldst uppdaterade (alla spårade fält, topp 10)")
    work2 = add_oldest_ts_col(df.copy())
    topp = work2.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"], ascending=[True, True]).head(10)

    visa_kol = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
              "TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in df.columns:
            visa_kol.append(k)
    visa_kol.append("_oldest_any_ts")

    st.dataframe(topp[visa_kol], use_container_width=True, hide_index=True)

    return df

# app.py — Del 8/8
# --- MAIN --------------------------------------------------------------------

def _ensure_rate_state_from_saved():
    """Initiera sidopanelens växelkurser i session_state från sparade värden en gång."""
    saved = las_sparade_valutakurser()
    if "rate_usd" not in st.session_state:
        st.session_state.rate_usd = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
    if "rate_nok" not in st.session_state:
        st.session_state.rate_nok = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
    if "rate_cad" not in st.session_state:
        st.session_state.rate_cad = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
    if "rate_eur" not in st.session_state:
        st.session_state.rate_eur = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

def _apply_pending_auto_rates_if_any():
    """
    Om vi tidigare klickat 'Hämta kurser automatiskt' sparas värden i pending_auto_rates
    och vi kör en rerun. Vid nästa run sätter vi state innan widgetarna skapas.
    """
    if "pending_auto_rates" in st.session_state:
        ar = st.session_state.pop("pending_auto_rates", {})
        st.session_state.rate_usd = float(ar.get("USD", st.session_state.get("rate_usd", STANDARD_VALUTAKURSER["USD"])))
        st.session_state.rate_nok = float(ar.get("NOK", st.session_state.get("rate_nok", STANDARD_VALUTAKURSER["NOK"])))
        st.session_state.rate_cad = float(ar.get("CAD", st.session_state.get("rate_cad", STANDARD_VALUTAKURSER["CAD"])))
        st.session_state.rate_eur = float(ar.get("EUR", st.session_state.get("rate_eur", STANDARD_VALUTAKURSER["EUR"])))
        # Lägg en “just applied”-flagga så vi kan visa meddelande denna run
        provider = st.session_state.pop("pending_auto_provider", "okänd")
        misses = st.session_state.pop("pending_auto_misses", [])
        st.session_state.last_rates_applied = {"provider": provider, "misses": misses}

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Initiera valutakurser i state (innan widgets)
    _ensure_rate_state_from_saved()
    _apply_pending_auto_rates_if_any()

    # Sidopanel: valutakurser
    st.sidebar.header("💱 Valutakurser → SEK")

    # Visa ev. “applicerat” info från förra klicket
    if "last_rates_applied" in st.session_state:
        info = st.session_state.pop("last_rates_applied")
        st.sidebar.success(f"Valutakurser (källa: {info.get('provider','okänd')}) uppdaterade.")
        if info.get("misses"):
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(info["misses"]))

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")

    # Auto-hämtning av kurser
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        # spara i pending och kör rerun så vi sätter state innan widgets nästa gång
        st.session_state.pending_auto_rates = auto_rates
        st.session_state.pending_auto_provider = provider
        st.session_state.pending_auto_misses = misses
        (getattr(st, "rerun", None) or st.experimental_rerun)()

    # Användar-rates (läs från state)
    user_rates = {
        "USD": st.session_state.rate_usd,
        "NOK": st.session_state.rate_nok,
        "CAD": st.session_state.rate_cad,
        "EUR": st.session_state.rate_eur,
        "SEK": 1.0
    }

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            try:
                spara_valutakurser({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
                # bump cache nonce
                st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
                st.sidebar.success("Valutakurser sparade.")
            except Exception as e:
                st.sidebar.warning(f"Kunde inte spara kurser: {e}")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            # Läs sparat och lägg in i state före widgets nästa run
            saved = las_sparade_valutakurser()
            st.session_state.pending_auto_rates = {
                "USD": float(saved.get("USD", STANDARD_VALUTAKURSER["USD"])),
                "NOK": float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
                "CAD": float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
                "EUR": float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
            }
            st.session_state.pending_auto_provider = "Sparade"
            st.session_state.pending_auto_misses = []
            (getattr(st, "rerun", None) or st.experimental_rerun)()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        (getattr(st, "rerun", None) or st.experimental_rerun)()

    # Läs data (observera: skriv ALDRIG tillbaka en tom df automatiskt)
    try:
        df = hamta_data()
    except Exception:
        df = pd.DataFrame(columns=FINAL_COLS)

    # Säkerställ schema, migrera och typer (utan att auto-spara)
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Auto-uppdatera alla, med val
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Auto-uppdatering")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=False, key="auto_snap")
    force_ts_all = st.sidebar.checkbox("Tidsstämpla även oförändrat", value=False, key="auto_force_ts")
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → Finnhub → FMP → CF/BS)"):
        df2, log = auto_update_all(df.copy(), user_rates, make_snapshot=make_snapshot, force_ts=force_ts_all)
        st.session_state["last_auto_log"] = log
        # df2 är uppdaterad kopia; använd den i sessionen
        df = df2

    # --- Meny
    meny = st.sidebar.radio("📌 Välj vy", [
        "Kontroll",
        "Analys",
        "Lägg till / uppdatera bolag",
        "Investeringsförslag",
        "Portfölj",
        "Batch"
    ])

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # uppdatera df i minnet om funktionen returnerade förändrad
        if not df2.equals(df):
            df = df2
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)
    elif meny == "Batch":
        df2 = batchvy(df, user_rates)
        if not df2.equals(df):
            df = df2

if __name__ == "__main__":
    main()
