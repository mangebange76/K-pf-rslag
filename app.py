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
from typing import Optional, Tuple, Dict, List

# --- Lokal Stockholm-tid (om pytz finns) ---
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
    """Skriv hela DataFrame till huvudbladet. Optionellt snapshot i separat flik."""
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
    rows = _with_backoff(ws.get_all_records)
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

    # 1) FMP om API-nyckel finns
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

    # Fyll luckor med sparade/standard
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

# Slutlig kolumnlista i databasen (inkl. nya fält: sektor, nyckeltal, kassaflöde, mcap-historia)
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Finansiell valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Extra yfinance/analysfält
    "P/S (Yahoo)", "MCap (nu)", "Sektor",
    "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa",
    "Total skuld", "EBITDA (TTM)", "Räntekostnad (TTM)", "Beta",
    "Current assets", "Current liabilities",

    # Kassaflöde/kostnader (kvartal) + runway
    "Operativt kassaflöde (Q)", "CapEx (Q)", "Fritt kassaflöde (Q)",
    "Operating Expense (Q)", "FoU (Q)", "SG&A (Q)",
    "Burn rate (Q)", "Runway (kvartal)",

    # MCap-historik (Q1–Q4) + datum
    "MCap Datum Q1","MCap Q1","MCap Datum Q2","MCap Q2","MCap Datum Q3","MCap Q3","MCap Datum Q4","MCap Q4",

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
                "kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt",
                "mcap", "debt", "beta", "kassa", "ebitda", "räntekostnad", "assets", "liabilities",
                "kassaflöde", "capex", "opex", "runway", "burn", "marginal"
            ]):
                df[kol] = 0.0
            elif kol.startswith("TS_") or kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"):
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
        "P/S (Yahoo)", "MCap (nu)",
        "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "Kassa",
        "Total skuld", "EBITDA (TTM)", "Räntekostnad (TTM)", "Beta",
        "Current assets", "Current liabilities",
        "Operativt kassaflöde (Q)", "CapEx (Q)", "Fritt kassaflöde (Q)",
        "Operating Expense (Q)", "FoU (Q)", "SG&A (Q)",
        "Burn rate (Q)", "Runway (kvartal)",
        "MCap Q1","MCap Q2","MCap Q3","MCap Q4"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Finansiell valuta","Sektor","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
              "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in df.columns:
        if str(c).startswith("TS_"):
            df[c] = df[c].astype(str)
    return df

# --- Tidsstämpelshjälpare ----------------------------------------------------

def _stamp_ts_for_field(df: pd.DataFrame, row_idx: int, field: str, when: Optional[str] = None):
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    date_str = when if when else now_stamp()
    try:
        df.at[row_idx, ts_col] = date_str
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _note_manual_update(df: pd.DataFrame, row_idx: int):
    try:
        df.at[row_idx, "Senast manuellt uppdaterad"] = now_stamp()
    except Exception:
        pass

# Fält som triggar "Senast manuellt uppdaterad" i formuläret
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# app.py — Del 2/7
# --- Yahoo-hjälpare & beräkningar -------------------------------------------

def _yfi_get(tkr: yf.Ticker, *keys):
    """Säker hämtning ur yfinance.info/värden."""
    try:
        info = tkr.info or {}
        for k in keys:
            if k in info and info[k] is not None:
                return info[k]
    except Exception:
        pass
    return None

def _yfi_info_dict(t: yf.Ticker) -> dict:
    try:
        return t.info or {}
    except Exception:
        return {}

def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """CAGR baserat på 'Total Revenue' (årsbasis), enkel approx (procent)."""
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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Finansiell valuta, Utdelning,
    CAGR 5 år (%), P/S (Yahoo), MCap (nu), Sektor.
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
        "Sektor": "",
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
        if info.get("currency"):
            out["Valuta"] = str(info.get("currency")).upper()
        if info.get("financialCurrency"):
            out["Finansiell valuta"] = str(info.get("financialCurrency")).upper()

        # Namn & sektor
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)
        if info.get("sector"):
            out["Sektor"] = str(info.get("sector"))

        # Utdelning (årstakt)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        # CAGR approx
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

        # P/S (Yahoo direkt från info om finns)
        ps_y = info.get("priceToSalesTrailing12Months") or info.get("priceToSalesTrailing12MonthsRaw")
        try:
            if ps_y is not None and float(ps_y) > 0:
                out["P/S (Yahoo)"] = float(ps_y)
        except Exception:
            pass

        # Market Cap
        mc = info.get("marketCap")
        try:
            if mc is not None and float(mc) > 0:
                out["MCap (nu)"] = float(mc)
        except Exception:
            pass
    except Exception:
        pass
    return out

def _yfi_quarterly_revenues(t: yf.Ticker) -> list[tuple[datetime.date, float]]:
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
        ps_snitt = round(np.mean(ps_clean), 3) if ps_clean else 0.0
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
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        try:
            aktier_ut = float(rad.get("Utestående aktier", 0.0))
        except Exception:
            aktier_ut = 0.0
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, changes_map: dict, force_ts: bool = False) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde.
    Uppdaterar TS_ för spårade fält, sätter 'Senast auto-uppdaterad' + källa.
    Om force_ts=True: stämpla 'Senast auto-uppdaterad' även om inget ändras.
    Returnerar True om något fält faktiskt ändrades.
    """
    changed_fields = []
    for f, v in new_vals.items():
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

    if changed_fields or force_ts:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return bool(changed_fields)
    return False

# app.py — Del 3/7
# --- SEC (US/FPI) + robust kvartal & Yahoo-fallback --------------------------

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

# ---------- datum/parsers & shares (instant) ---------------------------------
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
    Summerar multi-class per senaste 'end' (instant). Returnerar aktier (styck).
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

# ---------- FX & kvartalsintäkter (SEC) --------------------------------------
@st.cache_data(show_spinner=False, ttl=21600)
def _fx_rate_cached(base: str, quote: str) -> float:
    """Enkel FX (dagens) via Frankfurter → exchangerate.host fallback."""
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            v = float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
            if v > 0:
                return v
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            v = float((r.json() or {}).get("rates", {}).get(quote, 0.0) or 0.0)
            if v > 0:
                return v
    except Exception:
        pass
    return 0.0

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (≈3 mån) från SEC XBRL.
    Tillåter 10-Q/6-K samt 10-K/20-F/40-F/8-K för Q4-fönster.
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    """
    taxos = [
        ("us-gaap",  {"forms": ("10-Q", "10-Q/A", "10-K", "10-K/A", "8-K", "8-K/A")}),
        ("ifrs-full", {"forms": ("6-K", "6-K/A", "10-Q", "10-Q/A", "20-F", "40-F")}),
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
                    # 3-mån fönster ~ 70..100 dagar
                    if dur is None or dur < 70 or dur > 100:
                        continue
                    try:
                        v = float(val)
                        tmp.append((end, v))
                    except Exception:
                        pass
                if not tmp:
                    continue
                # deduplicera per end-datum och sortera
                ded = {}
                for end, v in tmp:
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

# ---------- Yahoo-priser & TTM-fönster ---------------------------------------
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

# ---------- SEC + Yahoo kombinerat -------------------------------------------
def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (10-Q/6-K samt 10-K/20-F/8-K för Q4),
    komplettera saknade kvartal från Yahoo. Pris/valuta/namn från Yahoo.
    P/S (TTM) nu + P/S Q1–Q4 historik + MCAP Q1–Q4.
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
    for k in ("Bolagsnamn", "Valuta", "Finansiell valuta", "Aktuell kurs", "P/S (Yahoo)", "MCap (nu)", "Sektor"):
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

    # SEC kvartalsintäkter + unit
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)

    # 🔁 Fallback: fyll saknade kvartal med Yahoo om SEC ger för få rader
    if (not q_rows) or (len(q_rows) < 4):
        try:
            t = yf.Ticker(ticker)
            y_rows = _yfi_quarterly_revenues(t)  # [(date, value)] nyast→äldst
            if y_rows:
                fin_ccy = str(_yfi_info_dict(t).get("financialCurrency") or (rev_unit or "USD")).upper()
                comb = {d: v for (d, v) in q_rows}
                for d, v in y_rows:
                    comb.setdefault(d, v)
                q_rows = sorted(comb.items(), key=lambda t: t[0], reverse=True)[:20]
                if rev_unit is None:
                    rev_unit = fin_ccy
        except Exception:
            pass

    if not q_rows or not rev_unit:
        return out

    # Konvertering till prisvaluta
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0

    # Bygg TTM-fönster (gärna 6 för robusthet)
    ttm_list = _ttm_windows(q_rows, need=6)
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]
    if not ttm_list_px:
        return out

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 + MCap Q1–Q4 historik
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

# ---------- Global Yahoo fallback (icke-SEC: .TO/.V/.CN + EU/Norden) ---------
def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC. Räknar implied shares, P/S (TTM) nu,
    samt P/S Q1–Q4 + MCAP Q1–Q4 historik från Yahoo quarterly + pris.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/finansiell valuta/ps_yahoo/mcap/sector
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","P/S (Yahoo)","MCap (nu)","Sektor"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(y.get("MCap (nu)") or 0.0)
    try:
        if mcap <= 0:
            mcap = float(info.get("marketCap") or 0.0)
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
    if mcap > 0:
        out["MCap (nu)"] = float(mcap)

    # P/S (TTM) nu
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

    # P/S Q1–Q4 (historisk) + MCAP Q1–Q4
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
# --- Datakällor: FMP, Finnhub, Yahoo kassaflöden & balans/IS, Auto-pipeline --

# =============== FMP =========================================================
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.2))      # skonsam default
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 15)) # paus efter 429

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
                out["_debug"]["ps_source"] = "FMP ratios-ttm"
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

# =============== Yahoo kassaflöden & kostnader + balans/IS (TTM) =============

def hamta_yahoo_cashflow_and_costs(ticker: str) -> dict:
    """
    Hämtar senaste kvartalets CFO, CapEx, FCF samt Opex/FoU/SG&A från Yahoo.
    Alla värden är i bolagets financialCurrency (samma som övriga Yahoo-rapporter).
    """
    out = {}
    t = yf.Ticker(ticker)

    # --- Cashflow (kvartalsvis) ---
    try:
        qcf = t.quarterly_cashflow
        if isinstance(qcf, pd.DataFrame) and not qcf.empty:
            def pick_cf(names):
                for nm in names:
                    if nm in qcf.index:
                        s = qcf.loc[nm].dropna()
                        if not s.empty:
                            return float(s.iloc[0])
                return None

            cfo   = pick_cf(["Total Cash From Operating Activities","OperatingCashFlow","NetCashProvidedByUsedInOperatingActivities","Net Cash Provided By Operating Activities","Operating Cash Flow"])
            capex = pick_cf(["Capital Expenditures","CapitalExpenditures"])
            if cfo is not None:   out["Operativt kassaflöde (Q)"] = cfo
            if capex is not None: out["CapEx (Q)"]                 = capex
            if (cfo is not None) and (capex is not None):
                # Obs: på Yahoo är CapEx vanligtvis negativt; FCF = CFO + CapEx
                out["Fritt kassaflöde (Q)"] = float(cfo) + float(capex)
    except Exception:
        pass

    # --- Kostnader (kvartalsvis) ---
    try:
        qfin = t.quarterly_financials
        if isinstance(qfin, pd.DataFrame) and not qfin.empty:
            def pick_fin(names):
                for nm in names:
                    if nm in qfin.index:
                        s = qfin.loc[nm].dropna()
                        if not s.empty:
                            return float(s.iloc[0])
                return None

            opex = pick_fin(["Operating Expense","OperatingExpenses","Total Operating Expenses","TotalOperatingExpenses"])
            rnd  = pick_fin(["Research Development","ResearchAndDevelopment"])
            sga  = pick_fin(["Selling General Administrative","SellingGeneralAndAdministrative"])

            if opex is not None: out["Operating Expense (Q)"] = opex
            if rnd  is not None: out["FoU (Q)"]               = rnd
            if sga  is not None: out["SG&A (Q)"]              = sga
    except Exception:
        pass

    return out

def hamta_yahoo_balans_och_is_ttm(ticker: str) -> dict:
    """
    Hämtar Sektor (info) samt Kassa, Total skuld, EBITDA (TTM), Räntekostnad (TTM),
    Current assets/liabilities och Beta (info). Allt Yahoo.
    """
    out = {}
    t = yf.Ticker(ticker)
    info = _yfi_info_dict(t)

    # Sektor & Beta från info
    if info.get("sector"):
        out["Sektor"] = str(info.get("sector"))
    b = info.get("beta") or info.get("beta3Year") or info.get("beta5Year")
    try:
        if b is not None:
            out["Beta"] = float(b)
    except Exception:
        pass

    # Balansräkning (kvartal) – hämta kassa, skuld, current A/L
    try:
        qbs = getattr(t, "quarterly_balance_sheet", None)
        if isinstance(qbs, pd.DataFrame) and not qbs.empty:
            def last_val(idx_names):
                for nm in idx_names:
                    if nm in qbs.index:
                        s = qbs.loc[nm].dropna()
                        if not s.empty:
                            return float(s.iloc[0])
                return None
            # Kassa/cash & STD-inv
            cash = last_val([
                "Cash And Cash Equivalents",
                "CashAndCashEquivalents",
                "Cash Cash Equivalents And Short Term Investments",
                "CashAndCashEquivalentsAndShortTermInvestments",
            ])
            tot_debt = last_val(["Total Debt","TotalDebt"])
            cur_assets = last_val(["Total Current Assets","TotalCurrentAssets"])
            cur_liab   = last_val(["Total Current Liabilities","TotalCurrentLiabilities"])
            if cash is not None: out["Kassa"] = cash
            if tot_debt is not None: out["Total skuld"] = tot_debt
            if cur_assets is not None: out["Current assets"] = cur_assets
            if cur_liab   is not None: out["Current liabilities"] = cur_liab
    except Exception:
        pass

    # Resultaträkning (TTM) – EBITDA & Räntekostnad
    try:
        qis = getattr(t, "quarterly_income_stmt", None) or getattr(t, "quarterly_income_statement", None)
        if qis is None:
            qis = getattr(t, "quarterly_financials", None)
        def _y_ttm_from_quarterly(df: pd.DataFrame, names: list[str]) -> float | None:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None
            for nm in names:
                if nm in df.index:
                    ser = df.loc[nm].dropna()
                    if ser.empty:
                        continue
                    try:
                        ser = ser.sort_index()
                    except Exception:
                        pass
                    vals = [float(v) for v in list(ser.values)[-4:]]
                    if vals:
                        return float(sum(vals))
            return None
        eb_ttm = _y_ttm_from_quarterly(qis, ["EBITDA","Ebitda","Earnings Before Interest Taxes Depreciation Amortization"])
        int_ttm = _y_ttm_from_quarterly(qis, ["Interest Expense","InterestExpense","InterestExpenseNonOperating"])
        if eb_ttm is not None: out["EBITDA (TTM)"] = eb_ttm
        if int_ttm is not None: out["Räntekostnad (TTM)"] = float(abs(int_ttm))  # gör positiv för täckningsgrad
    except Exception:
        pass

    return out

# =============== Auto-pipeline (ett bolag / alla) ============================

def auto_fetch_for_ticker(ticker: str) -> Tuple[dict, dict]:
    """
    Pipeline (per ticker):
      1) SEC + Yahoo (implied shares) + Yahoo-fallback för saknade kvartal.
      2) Finnhub (estimat) om saknas.
      3) FMP light (P/S) om saknas.
      4) Yahoo kassaflöden/kostnader Q + balans/IS TTM (inkl. Sektor, Beta, Kassa, Skuld).
    Returnerar (vals, debug).
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","Finansiell valuta","P/S (Yahoo)","MCap (nu)","Sektor",
            "MCap Datum Q1","MCap Q1","MCap Datum Q2","MCap Q2","MCap Datum Q3","MCap Q3","MCap Datum Q4","MCap Q4"
        ]}
        for k in ["Bolagsnamn","Valuta","Finansiell valuta","Aktuell kurs","Utestående aktier","P/S","P/S (Yahoo)",
                  "P/S Q1","P/S Q2","P/S Q3","P/S Q4","MCap (nu)","Sektor",
                  "MCap Datum Q1","MCap Q1","MCap Datum Q2","MCap Q2","MCap Datum Q3","MCap Q3","MCap Datum Q4","MCap Q4"]:
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

    # 3) FMP light P/S + shares/mcap om saknas
    try:
        need_ps = ("P/S" not in vals) or (float(vals.get("P/S", 0.0)) <= 0)
        if need_ps or ("Utestående aktier" not in vals) or ("MCap (nu)" not in vals):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier"), "MCap (nu)": fmpl.get("MCap (nu)")}
            for k in ["P/S","Utestående aktier","MCap (nu)","Aktuell kurs"]:
                v = fmpl.get(k)
                if v not in (None, "", 0, 0.0):
                    vals[k] = v
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # 4a) Yahoo cashflow & costs (Q)
    try:
        cf = hamta_yahoo_cashflow_and_costs(ticker)
        debug["yahoo_cashflow"] = cf
        for k in ["Operativt kassaflöde (Q)","CapEx (Q)","Fritt kassaflöde (Q)","Operating Expense (Q)","FoU (Q)","SG&A (Q)"]:
            v = cf.get(k)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["yahoo_cashflow_err"] = str(e)

    # 4b) Yahoo balans + IS (TTM) + sektor/beta/kassa/skuld
    try:
        ybi = hamta_yahoo_balans_och_is_ttm(ticker)
        debug["yahoo_balans_is"] = ybi
        for k in ["Sektor","Kassa","Total skuld","EBITDA (TTM)","Räntekostnad (TTM)","Current assets","Current liabilities","Beta"]:
            v = ybi.get(k)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["yahoo_balans_is_err"] = str(e)

    # 4c) Runway (kvartal) om Kassa & negativt FCF (Q)
    try:
        cash = vals.get("Kassa", None)
        fcfq = vals.get("Fritt kassaflöde (Q)", None)
        if (cash is not None) and (fcfq is not None):
            burn = abs(float(fcfq)) if float(fcfq) < 0 else 0.0
            vals["Burn rate (Q)"] = burn
            if burn > 0:
                vals["Runway (kvartal)"] = float(cash) / burn
    except Exception:
        pass

    return vals, debug

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, force_ts: bool = False):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla nya värden.
    Stämplar TS_ per fält, samt 'Senast auto-uppdaterad' + källa.
    Om force_ts=True: stämpla datum även om inget ändras (för "hämtad idag").
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
            if not changed:
                log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
            any_changed = any_changed or changed
            if i < 20:
                log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress((i+1)/max(total,1))

    # Efter loop — räkna om & spara
    df = uppdatera_berakningar(df, user_rates)

    if any_changed or force_ts:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Ändringar sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# app.py — Del 5/7
# --- Snapshots, Kontroll-vy, formatterare & bedömningar ----------------------

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

# ---------- Formatterings-hjälpare & bedömningar -----------------------------

def _mcap_label(v: float) -> str:
    """Formatera stora tal till tn/md/mn med 2 decimaler."""
    try:
        x = float(v)
    except Exception:
        return "—"
    sign = "-" if x < 0 else ""
    ax = abs(x)
    if ax >= 1e12:
        return f"{sign}{ax/1e12:.2f} tn"
    if ax >= 1e9:
        return f"{sign}{ax/1e9:.2f} md"
    if ax >= 1e6:
        return f"{sign}{ax/1e6:.2f} mn"
    if ax >= 1e3:
        return f"{sign}{ax/1e3:.2f} k"
    return f"{x:.2f}"

def _fmt_price(px: float, ccy: str) -> str:
    try:
        p = float(px)
    except Exception:
        return "—"
    return f"{p:.2f} {ccy or ''}".strip()

def _score_band(v: float, good: float, ok: float, reverse: bool = False):
    """
    Returnerar (emoji, label) enligt trösklar. reverse=True betyder 'lägre är bättre'.
    """
    try:
        x = float(v)
    except Exception:
        return "▪️", "–"
    if reverse:
        if x <= good:  return "✅", "bra"
        if x <= ok:    return "🟡", "ok"
        return "❌", "dåligt"
    else:
        if x >= good:  return "✅", "bra"
        if x >= ok:    return "🟡", "ok"
        return "❌", "dåligt"

def bedomningar(rad: pd.Series) -> list[tuple[str,str,str]]:
    """
    Bygger upp [(namn, värde_fmt, bedömning)] för nyckeltal.
    Vi använder MCap, P/S, Kassa, Total skuld, EBITDA (TTM), Räntekostnad (TTM) m.fl.
    """
    out = []

    mcap = float(rad.get("MCap (nu)", 0.0))
    ps_mod = float(rad.get("P/S", 0.0)) or float(rad.get("P/S (Yahoo)", 0.0))
    rev_ttm = (mcap / ps_mod) if (mcap > 0 and ps_mod > 0) else None  # modell-rev TTM om exakt saknas
    cash = float(rad.get("Kassa", 0.0))
    debt = float(rad.get("Total skuld", 0.0))
    ev = (mcap + debt - cash) if (mcap > 0) else None

    ebitda = float(rad.get("EBITDA (TTM)", 0.0) or 0.0)
    int_ttm = float(rad.get("Räntekostnad (TTM)", 0.0) or 0.0)
    cur_a = float(rad.get("Current assets", 0.0) or 0.0)
    cur_l = float(rad.get("Current liabilities", 0.0) or 0.0)
    current_ratio = (cur_a / cur_l) if (cur_a > 0 and cur_l > 0) else None

    fcf_ttm = None
    # Approx via Q * 4 om TTM saknas
    fcf_q = float(rad.get("Fritt kassaflöde (Q)", 0.0) or 0.0)
    if fcf_q != 0:
        fcf_ttm = fcf_q * 4.0

    # EV/S
    if ev is not None and rev_ttm:
        evs = ev / rev_ttm if rev_ttm != 0 else None
        if evs is not None:
            emoji, lab = _score_band(evs, good=3.0, ok=8.0, reverse=True)  # lägre bättre
            out.append(("EV/S (TTM)", f"{evs:.2f}", f"{emoji} {lab}"))

    # EV/EBITDA
    if ev is not None and ebitda and ebitda != 0:
        eve = ev / ebitda
        emoji, lab = _score_band(eve, good=10.0, ok=20.0, reverse=True)
        out.append(("EV/EBITDA (TTM)", f"{eve:.1f}", f"{emoji} {lab}"))

    # FCF-yield
    if mcap > 0 and fcf_ttm is not None:
        fcf_y = fcf_ttm / mcap
        emoji, lab = _score_band(fcf_y * 100.0, good=5.0, ok=0.0, reverse=False)  # >5% bra, 0–5 ok, <0 dåligt
        out.append(("FCF-yield (TTM)", f"{fcf_y*100:.1f} %", f"{emoji} {lab}"))

    # FCF-marginal (om rev_ttm finns)
    if fcf_ttm is not None and rev_ttm:
        fcf_m = fcf_ttm / rev_ttm
        emoji, lab = _score_band(fcf_m * 100.0, good=10.0, ok=0.0, reverse=False)  # >10% bra
        out.append(("FCF-marg (TTM)", f"{fcf_m*100:.1f} %", f"{emoji} {lab}"))

    # Current ratio
    if current_ratio is not None:
        emoji, lab = _score_band(current_ratio, good=1.5, ok=1.0, reverse=False)
        out.append(("Current ratio", f"{current_ratio:.2f}", f"{emoji} {lab}"))

    # Nettoskuld / Omsättning
    if rev_ttm:
        net_debt = max(0.0, debt - cash)
        nd_rev = net_debt / rev_ttm
        emoji, lab = _score_band(nd_rev, good=0.5, ok=1.5, reverse=True)  # lägre bättre
        out.append(("Nettoskuld / TTM-intäkt", f"{nd_rev:.2f}", f"{emoji} {lab}"))

    # Ränte­täckningsgrad = EBITDA / räntekostnad (approx)
    if ebitda and int_ttm:
        ic = ebitda / int_ttm if int_ttm != 0 else None
        if ic is not None:
            emoji, lab = _score_band(ic, good=5.0, ok=1.0, reverse=False)
            out.append(("Räntetäckningsgrad", f"{ic:.1f}x", f"{emoji} {lab}"))

    # Burn runway
    burn = float(rad.get("Burn rate (Q)", 0.0) or 0.0)
    runway_q = float(rad.get("Runway (kvartal)", 0.0) or 0.0)
    if burn > 0:
        emoji, lab = _score_band(runway_q, good=8.0, ok=4.0, reverse=False)
        out.append(("Runway", f"{runway_q:.1f} kv", f"{emoji} {lab}"))
    else:
        out.append(("Runway", "–", "✅ positivt FCF"))

    # Beta (volatilitet) – lägre “bättre” för defensivt
    beta = float(rad.get("Beta", 0.0) or 0.0)
    if beta > 0:
        emoji, lab = _score_band(beta, good=0.9, ok=1.3, reverse=True)
        out.append(("Beta", f"{beta:.2f}", f"{emoji} {lab}"))

    return out

# app.py — Del 6/7
# --- Analys, Portfölj & Investeringsförslag + Lägg till/uppdatera ------------

def _risk_label_sek(mcap_val: float, ccy: str, user_rates: dict) -> str:
    """
    Klassar risklabel baserat på Market Cap i SEK.
    Trösklar (SEK): Micro < 3bn, Small 3–30bn, Mid 30–100bn, Large 100–300bn, Mega > 300bn.
    """
    try:
        x = float(mcap_val or 0.0)
    except Exception:
        x = 0.0
    fx = hamta_valutakurs(ccy or "SEK", user_rates)
    sek = x * fx
    if sek < 3e9:
        return "Micro"
    if sek < 30e9:
        return "Small"
    if sek < 100e9:
        return "Mid"
    if sek < 300e9:
        return "Large"
    return "Mega"

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

    cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta","Finansiell valuta","Aktuell kurs","Utestående aktier",
        "MCap (nu)","P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Kassa","Total skuld","EBITDA (TTM)","Räntekostnad (TTM)","Current assets","Current liabilities","Beta",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år",
        "MCap Datum Q1","MCap Q1","MCap Datum Q2","MCap Q2","MCap Datum Q3","MCap Q3","MCap Datum Q4","MCap Q4"
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
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)
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

    # --- Filtersektion ---
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )
    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()

    # Marketcap i SEK + Risklabel (beräknas nu för filtrering)
    base["MCap_SEK"] = base.apply(lambda r: float(r.get("MCap (nu)",0.0)) * hamta_valutakurs(r.get("Valuta","SEK"), user_rates), axis=1)
    base["Risklabel"] = base.apply(lambda r: _risk_label_sek(r.get("MCap (nu)",0.0), r.get("Valuta","SEK"), user_rates), axis=1)

    # Sektorfiler
    sektorer = sorted([s for s in base.get("Sektor", pd.Series(dtype=str)).dropna().unique().tolist() if str(s).strip() != ""])
    sektor_choice = st.selectbox("Filter: Sektor", ["Alla"] + sektorer, index=0)
    if sektor_choice != "Alla":
        base = base[base["Sektor"] == sektor_choice].copy()

    # Riskfilter
    risk_opts = ["Alla","Micro","Small","Mid","Large","Mega"]
    risk_choice = st.selectbox("Filter: Risklabel (MCap)", risk_opts, index=0)
    if risk_choice != "Alla":
        base = base[base["Risklabel"] == risk_choice].copy()

    # Basurval för förslagsräkning
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Potential & sort
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)
    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Navigator
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

    # Portföljandel före/efter köp
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
        rsel = port[port["Ticker"] == rad["Ticker"]]
        if not rsel.empty:
            nuv_innehav = float(rsel["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Nuvarande MCAP/P-S
    mcap_now = float(rad.get("MCap (nu)", 0.0))
    ps_now = float(rad.get("P/S", 0.0) or 0.0)
    ps_yh = float(rad.get("P/S (Yahoo)", 0.0) or 0.0)
    ps_now_show = ps_now if ps_now > 0 else ps_yh
    ps_snitt = float(rad.get("P/S-snitt", 0.0) or 0.0)

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    # Huvudrad-data
    lines = [
        f"- **Sektor:** {rad.get('Sektor','—')}",
        f"- **Aktuell kurs:** {_fmt_price(rad['Aktuell kurs'], rad['Valuta'])}",
        f"- **Nuvarande Market Cap:** {_mcap_label(mcap_now)}",
        f"- **Nuvarande P/S:** {ps_now_show:.2f}" if ps_now_show > 0 else "- **Nuvarande P/S:** —",
        f"- **P/S-snitt (4 senaste):** {ps_snitt:.2f}" if ps_snitt > 0 else "- **P/S-snitt (4 senaste):** —",
        f"- **Riktkurs (vald):** {round(rad[riktkurs_val],2)} {rad['Valuta']}",
        f"- **Uppsida (vald riktkurs):** {round(rad['Potential (%)'],2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
    ]
    st.markdown("\n".join(lines))

    # Expanderdetaljer
    with st.expander("📚 Detaljer: MCAP-historik, kassaflöde, kostnader & bedömning", expanded=False):
        # MCAP Q1–Q4
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**MCAP (Q1–Q4)**")
            mc_rows = []
            for i in range(1,5):
                d = rad.get(f"MCap Datum Q{i}", "")
                v = rad.get(f"MCap Q{i}", 0.0)
                if d or v:
                    mc_rows.append((d, _mcap_label(v)))
            if mc_rows:
                st.table(pd.DataFrame(mc_rows, columns=["Periodslut","MCap"]))
            else:
                st.caption("Saknar historisk MCAP.")

        with col2:
            st.markdown("**P/S (Q1–Q4)**")
            ps_rows = []
            for i in range(1,5):
                val = float(rad.get(f"P/S Q{i}", 0.0) or 0.0)
                if val > 0:
                    ps_rows.append((f"Q{i}", f"{val:.2f}"))
            if ps_rows:
                st.table(pd.DataFrame(ps_rows, columns=["Fönster","P/S"]))
            else:
                st.caption("Saknar historisk P/S.")

        st.markdown("---")
        # Kassaflöde & kostnader (senaste Q)
        st.markdown("**Kassaflöde & kostnader (senaste kvartal)**")
        cf_rows = [
            ("Operativt kassaflöde (Q)", rad.get("Operativt kassaflöde (Q)", 0.0)),
            ("CapEx (Q)", rad.get("CapEx (Q)", 0.0)),
            ("Fritt kassaflöde (Q)", rad.get("Fritt kassaflöde (Q)", 0.0)),
            ("Operating Expense (Q)", rad.get("Operating Expense (Q)", 0.0)),
            ("FoU (Q)", rad.get("FoU (Q)", 0.0)),
            ("SG&A (Q)", rad.get("SG&A (Q)", 0.0)),
            ("Kassa", rad.get("Kassa", 0.0)),
            ("Total skuld", rad.get("Total skuld", 0.0)),
            ("EBITDA (TTM)", rad.get("EBITDA (TTM)", 0.0)),
            ("Räntekostnad (TTM)", rad.get("Räntekostnad (TTM)", 0.0)),
            ("Current assets", rad.get("Current assets", 0.0)),
            ("Current liabilities", rad.get("Current liabilities", 0.0)),
            ("Runway (kvartal)", rad.get("Runway (kvartal)", 0.0)),
        ]
        st.table(pd.DataFrame([(k, _mcap_label(v) if "MCap" in k else f"{v:,.0f}") for (k,v) in cf_rows], columns=["Nyckel","Värde"]))

        st.markdown("**Bedömning (nyckeltal)**")
        for namn, vf, dom in bedomningar(rad):
            st.write(f"{namn}: **{vf}** — {dom}")

def _write_row_from_form(df: pd.DataFrame, ticker: str, ny: dict, bef: pd.Series):
    """Gemensam skrivare för formuläret (utan auto-hämtning)."""
    if not bef.empty:
        for k,v in ny.items():
            df.loc[df["Ticker"]==ticker, k] = v
    else:
        tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Finansiell valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
        tom.update(ny)
        df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
    return df

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

    # Status-etiketter (senast auto/manuellt)
    if not bef.empty:
        st.info(
            f"**Senast auto-uppdaterad:** {bef.get('Senast auto-uppdaterad','')}  |  "
            f"**Källa:** {bef.get('Senast uppdaterad källa','')}  |  "
            f"**Senast manuellt uppdaterad:** {bef.get('Senast manuellt uppdaterad','')}"
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

            # TS-notiser vid sidan av fälten
            if not bef.empty:
                st.caption(f"TS P/S: {bef.get('TS_P/S','')} | Q1: {bef.get('TS_P/S Q1','')} | Q2: {bef.get('TS_P/S Q2','')}")
                st.caption(f"TS Q3: {bef.get('TS_P/S Q3','')} | Q4: {bef.get('TS_P/S Q4','')} | TS Utest: {bef.get('TS_Utestående aktier','')}")

        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Vid spara uppdateras också (utan att skriva över manuella 0-värden):**")
            st.write("- Bolagsnamn, Valuta, Finansiell valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

            if not bef.empty:
                st.caption(f"TS Oms. idag: {bef.get('TS_Omsättning idag','')} | TS Oms. nästa år: {bef.get('TS_Omsättning nästa år','')}")

        colf1, colf2, colf3 = st.columns([1,1,1])
        spar = colf1.form_submit_button("💾 Spara")
        # Singel-ticker knappar
        btn_price = colf2.form_submit_button("💹 Uppdatera kurs (denna ticker)")
        btn_auto  = colf3.form_submit_button("🤖 Full auto (denna ticker)")

    # Spara-formulär
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
        df = _write_row_from_form(df, ticker, ny, bef)

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
        if data.get("Finansiell valuta"): df.loc[ridx, "Finansiell valuta"] = data["Finansiell valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)
        if "P/S (Yahoo)" in data and data.get("P/S (Yahoo)") is not None:         df.loc[ridx, "P/S (Yahoo)"]     = float(data.get("P/S (Yahoo)") or 0.0)
        if "MCap (nu)" in data and data.get("MCap (nu)") is not None:             df.loc[ridx, "MCap (nu)"]       = float(data.get("MCap (nu)") or 0.0)
        if "Sektor" in data and data.get("Sektor"):                                df.loc[ridx, "Sektor"]          = str(data.get("Sektor") or "")

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")
        st.rerun()

    # Endast kurs (denna ticker)
    if btn_price:
        if not bef.empty:
            tkr = bef.get("Ticker", "").strip().upper()
            if not tkr:
                st.error("Ange ticker först.")
            else:
                data = hamta_yahoo_fält(tkr)
                ridx = df.index[df["Ticker"]==tkr][0]
                changed_any = False
                for k in ["Aktuell kurs","P/S (Yahoo)","MCap (nu)","Valuta","Finansiell valuta","Bolagsnamn","Sektor"]:
                    if data.get(k) not in (None, "", 0, 0.0):
                        df.loc[ridx, k] = data[k]
                        changed_any = True
                if changed_any:
                    _note_auto_update(df, ridx, source="Pris (Yahoo)")
                    df = uppdatera_berakningar(df, user_rates)
                    spara_data(df)
                    st.success(f"Kurs/MCAP uppdaterad för {tkr}.")
                    st.rerun()
                else:
                    st.info("Ingen förändring: hittade inga fält att uppdatera.")
        else:
            st.warning("Välj ett befintligt bolag först.")

    # Full auto (denna ticker)
    if btn_auto:
        if not bef.empty:
            tkr = bef.get("Ticker", "").strip().upper()
            ridx = df.index[df["Ticker"]==tkr][0]
            new_vals, debug = auto_fetch_for_ticker(tkr)
            changed = apply_auto_updates_to_row(df, ridx, new_vals, source="Auto (singel)", changes_map={}, force_ts=True)
            df = uppdatera_berakningar(df, user_rates)
            spara_data(df)
            if changed:
                st.success(f"Auto-uppdaterat {tkr}.")
            else:
                st.info(f"Inga ändringar hittades vid auto-uppdatering ({tkr}), men datum stämplat.")
            st.rerun()
        else:
            st.warning("Välj ett befintligt bolag först.")

    # --- Snabblista: äldst uppdaterade ---
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
# --- MAIN + batch “pris bara” ------------------------------------------------

def uppdatera_endast_priser_alla(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lätt batch: uppdatera endast Aktuell kurs, P/S (Yahoo), MCap (nu), Valuta,
    Finansiell valuta, Bolagsnamn, Sektor för alla tickers via Yahoo.
    Sätter 'Senast auto-uppdaterad' men rör inte övriga fält.
    """
    if df.empty:
        st.sidebar.info("Ingen data i tabellen.")
        return df

    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()
    n = len(df)
    any_changed = False

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        tkr = str(row.get("Ticker","")).strip().upper()
        if not tkr:
            progress.progress(i/max(n,1))
            continue

        status.write(f"Uppdaterar kurs {i}/{n}: {tkr}")
        try:
            data = hamta_yahoo_fält(tkr)
            changed = False
            for k in ["Aktuell kurs","P/S (Yahoo)","MCap (nu)","Valuta","Finansiell valuta","Bolagsnamn","Sektor"]:
                v = data.get(k)
                if v not in (None, "", 0, 0.0):
                    if (pd.isna(df.at[idx, k]) and not pd.isna(v)) or (str(df.at[idx, k]) != str(v)):
                        df.at[idx, k] = v
                        changed = True
            if changed:
                _note_auto_update(df, idx, source="Pris (Yahoo, batch)")
                any_changed = True
        except Exception:
            # hoppa vidare
            pass
        progress.progress(i/max(n,1))

    if any_changed:
        spara_data(df, do_snapshot=False)
        st.sidebar.success("Kurser uppdaterade.")
    else:
        st.sidebar.info("Inga kursförändringar hittades.")

    return df

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Sidopanel: valutakurser (läs sparat & initiera state) ---
    st.sidebar.header("💱 Valutakurser → SEK")

    saved_rates = las_sparade_valutakurser()
    # Initiera session_state NYCKLAR före widgets:
    if "rate_usd" not in st.session_state:
        st.session_state.rate_usd = float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"]))
    if "rate_nok" not in st.session_state:
        st.session_state.rate_nok = float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
    if "rate_cad" not in st.session_state:
        st.session_state.rate_cad = float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
    if "rate_eur" not in st.session_state:
        st.session_state.rate_eur = float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
    if "rates_reload" not in st.session_state:
        st.session_state.rates_reload = 0

    # Auto-hämtning av FX innan widgets skapas (så att widgets visar nya värden direkt)
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        auto_rates, misses, provider = hamta_valutakurser_auto()
        # Skriv in i session_state (widgets plockar upp dessa värden nedan)
        try:
            st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
        except Exception:
            pass
        st.sidebar.success(f"Valutakurser hämtade (källa: {provider}).")
        if misses:
            st.sidebar.warning("Vissa par kunde inte hämtas:\n- " + "\n- ".join(misses))
        # Rerun så att number_input visar nya värden omedelbart
        st.rerun()

    # Widgets bundna till session_state
    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")

    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            st.session_state.rates_reload += 1
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear()
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # --- Läs data från Google Sheets ---
    df = hamta_data()
    if df.empty:
        # Skapa tom mall om bladet var tomt
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    # Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Auto-uppdateringssektion i sidopanelen ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Auto-uppdatering")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True)
    force_ts_all = st.sidebar.checkbox("Stämpla datum även om inget ändras", value=True)

    # Lätt batch: endast kurser
    if st.sidebar.button("💹 Uppdatera endast kurser (alla)"):
        df = uppdatera_endast_priser_alla(df)
        st.session_state["last_auto_log"] = {"changed": "(pris batch) – se blad", "misses": {}, "debug_first_20": []}
        st.rerun()

    # Full auto för alla
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → Finnhub → FMP → Yahoo CF/BS)"):
        df, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot, force_ts=force_ts_all)
        st.session_state["last_auto_log"] = log
        st.rerun()

    # --- Meny & vyer ---
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # Om funktionen returnerar ny df (t.ex. efter spara), uppdatera referensen
        df = df2
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
