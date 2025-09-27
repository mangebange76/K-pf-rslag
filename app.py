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

# === NYTT: Lås manuella prognosfält (aldrig auto-skrivning) ==================

# Dessa fält ska alltid matas in manuellt (miljoner i bolagets valuta)
MANUAL_ONLY_FIELDS = ["Omsättning idag", "Omsättning nästa år"]

def strip_manual_only_fields(d: dict) -> dict:
    """
    Tar bort manuella prognosfält från en uppsättning auto-hämtade värden
    så de aldrig kan skrivas av misstag.
    """
    if not isinstance(d, dict):
        return d
    for k in MANUAL_ONLY_FIELDS:
        if k in d:
            d.pop(k, None)
    return d

# --- Kolumnschema & tidsstämplar --------------------------------------------

# Spårade fält → respektive TS-kolumn (uppdateras när fältet ändras manuellt/auto)
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

# Slutlig kolumnlista i databasen
FINAL_COLS = [
    # Grund
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",

    # Extra nyckeltal / värdering
    "Sektor",
    "MCAP nu", "MCAP Q1", "MCAP Q2", "MCAP Q3", "MCAP Q4",
    "EV", "EBITDA (TTM)", "EV/EBITDA",
    "Bruttomarginal (%)", "Nettomarginal (%)",
    "Skuldsättning D/E", "Current ratio", "Quick ratio",
    "Totalt kassa", "OCF (TTM)", "CapEx (TTM)", "FCF (TTM)",
    "Kassaförbrukning/kvartal", "Runway (kvartal)",
    "Direktavkastning (%)", "EPS (TTM)", "P/E (TTM)",

    # Portfölj / GAV
    "GAV (SEK)",

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
            # numeriska fält (vanliga)
            if any(x in kol.lower() for x in [
                "kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt",
                "mcap","ev","ebitda","bruttomarginal","nettomarginal","skuldsättning","ratio",
                "kassa","ocf","capex","fcf","kassaförbrukning","runway","direktavkastning","eps","p/e","gav"
            ]):
                df[kol] = 0.0
            # tidsstämplar
            elif kol.startswith("TS_") or kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"):
                df[kol] = ""
            # övrigt (t.ex. textfält som Sektor)
            else:
                df[kol] = ""
    # ta bort ev. dubletter i kolumnnamn
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
        "MCAP nu", "MCAP Q1", "MCAP Q2", "MCAP Q3", "MCAP Q4",
        "EV", "EBITDA (TTM)", "EV/EBITDA",
        "Bruttomarginal (%)", "Nettomarginal (%)",
        "Skuldsättning D/E", "Current ratio", "Quick ratio",
        "Totalt kassa", "OCF (TTM)", "CapEx (TTM)", "FCF (TTM)",
        "Kassaförbrukning/kvartal", "Runway (kvartal)",
        "Direktavkastning (%)", "EPS (TTM)", "P/E (TTM)",
        "GAV (SEK)",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    str_cols = ["Ticker","Bolagsnamn","Valuta","Sektor","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]
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
# (Omsättningsfälten är manuella-only; vi TS-stämplar dem här när du ändrar dem.)
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# --- Yahoo-hjälpare & beräkningar & SEC/FMP-integration ----------------------

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
    """Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR."""
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "Sektor": "",
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

        sector = info.get("sector") or ""
        if sector:
            out["Sektor"] = str(sector)

        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---------- SEC helper (USA & FPIs/IFRS) -------------------------------------

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
                    ded[end] = v
                rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)[:max_quarters]
                if rows:
                    return rows, unit_code
    return [], None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters)
    return rows

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
    OBS: Skriver INTE 'Omsättning idag'/'Omsättning nästa år' – dessa är manuella.
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs", "Sektor"):
        if y.get(k):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()

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
        out["MCAP nu"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    if not q_rows or not rev_unit:
        return out
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    ttm_list = _ttm_windows(q_rows, need=8)  # ta ut tillräckligt många
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik + MCAP-historik via pris & shares_used
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares_used * float(px)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCAP Q{idx}"] = float(mcap_hist)

    return out

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC (.TO/.V/.CN + EU/Norden m.fl.).
    Räknar implied shares, P/S (TTM) nu, samt P/S Q1–Q4 historik.
    OBS: Skriver INTE 'Omsättning idag'/'Omsättning nästa år' – dessa är manuella.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/sektor
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","Sektor"):
        if y.get(k): out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = info.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else 0.0
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
        out["MCAP nu"] = shares * px if (shares>0 and px>0) else 0.0

    # Kvartalsintäkter → TTM
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows or len(q_rows) < 4:
        return out
    ttm_list = _ttm_windows(q_rows, need=8)

    # Valutakonvertering om financialCurrency != prisvaluta
    fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        conv = _fx_rate_cached(fin_ccy, px_ccy) or 1.0
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if shares > 0 and px > 0 and ttm_list_px:
        mcap_now = shares * px
        if mcap_now > 0:
            out["MCAP nu"] = mcap_now
            ltm_now = ttm_list_px[0][1]
            if ltm_now > 0:
                out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 (historisk)
    if shares > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list_px[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                p = px_map.get(d_end)
                if p and p > 0:
                    mcap_hist = shares * p
                    out[f"P/S Q{idx}"] = (mcap_hist / ttm_rev_px)
                    out[f"MCAP Q{idx}"] = float(mcap_hist)

    return out

# =============== FMP =========================================================
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

@st.cache_data(show_spinner=False, ttl=1800)
def hamta_fmp_falt(yahoo_ticker: str) -> dict:
    """
    Fullare variant: försöker hämta namn/valuta/pris/shares, P/S (TTM, key-metrics, beräkning),
    P/S Q1–Q4 (ratios quarterly).
    OBS: Skriver INTE 'Omsättning idag'/'Omsättning nästa år' – dessa är manuella.
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
    if market_cap > 0:
        out["MCAP nu"] = market_cap

    # Shares fallback (float API)
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

    # key-metrics-ttm
    if "P/S" not in out:
        kttm, sc_kttm = _fmp_get(f"api/v3/key-metrics-ttm/{sym}", stable=False)
        out["_debug"]["key_metrics_ttm_sc"] = sc_kttm
        if isinstance(kttm, list) and kttm:
            try:
                v = kttm[0].get("priceToSalesRatioTTM") or kttm[0].get("priceToSalesTTM")
                if v and float(v) > 0:
                    out["P/S"] = float(v)
                    out["_debug"]["ps_source"] = "key-metrics-ttm"
            except Exception:
                pass

    # P/S = marketCap / revenueTTM (om båda finns)
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

    # P/S Q1–Q4 (ratios quarterly)
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

    # OBS: Inga analytikerestimat skrivs – manual only för omsättningsfält
    out["_est_status"] = None
    return out

# --- Formatering & beräkningar ----------------------------------------------

def human_money(n: float) -> str:
    """Snygg formattering för stora belopp: 1.23 T / 456.7 B / 12.3 M / 12 345."""
    try:
        n = float(n)
    except Exception:
        return "-"
    absn = abs(n)
    if absn >= 1e12:
        return f"{n/1e12:.2f} T"
    if absn >= 1e9:
        return f"{n/1e9:.2f} B"
    if absn >= 1e6:
        return f"{n/1e6:.2f} M"
    return f"{n:,.0f}".replace(",", " ")

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def _ps_snitt_from_quarters(row: pd.Series) -> float:
    vals = [
        _safe_float(row.get("P/S Q1",0.0)),
        _safe_float(row.get("P/S Q2",0.0)),
        _safe_float(row.get("P/S Q3",0.0)),
        _safe_float(row.get("P/S Q4",0.0)),
    ]
    pos = [v for v in vals if v > 0]
    return round(float(np.mean(pos)), 2) if pos else 0.0

def _clamped_cagr(cagr_pct: float) -> float:
    """CAGR clamp: >100% → 50%, <0% → 2% (konservativt)."""
    if cagr_pct > 100.0:
        return 50.0
    if cagr_pct < 0.0:
        return 2.0
    return cagr_pct

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning om 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser (idag/1/2/3) = (Omsättning * P/S-snitt) / Utestående aktier
      - EV/EBITDA om data finns
      - MCAP nu om kurs * shares kan räknas
      - P/E (TTM) om EPS (TTM) och kurs finns
    OBS: Omsättningsfälten är manuella och lämnas orörda här.
    """
    # säkerställ kolumner existerar även om de inte fanns i bladet
    for col in ["P/S-snitt","Omsättning om 2 år","Omsättning om 3 år","Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år","EV/EBITDA","MCAP nu","P/E (TTM)"]:
        if col not in df.columns:
            df[col] = 0.0

    for i, rad in df.iterrows():
        # P/S-snitt
        ps_snitt = _ps_snitt_from_quarters(rad)
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp
        cagr = _safe_float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = _clamped_cagr(cagr)
        g = just_cagr / 100.0

        # Omsättning 2 & 3 år (från manuella 'Omsättning nästa år')
        oms_next = _safe_float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # lämna befintliga värden om du fyllt dem manuellt
            df.at[i, "Omsättning om 2 år"] = _safe_float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = _safe_float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        aktier_ut_milj = _safe_float(rad.get("Utestående aktier", 0.0))  # i miljoner
        if aktier_ut_milj > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((_safe_float(rad.get("Omsättning idag", 0.0))     * ps_snitt) / aktier_ut_milj, 2)
            df.at[i, "Riktkurs om 1 år"] = round((_safe_float(rad.get("Omsättning nästa år", 0.0)) * ps_snitt) / aktier_ut_milj, 2)
            df.at[i, "Riktkurs om 2 år"] = round((_safe_float(df.at[i, "Omsättning om 2 år"])      * ps_snitt) / aktier_ut_milj, 2)
            df.at[i, "Riktkurs om 3 år"] = round((_safe_float(df.at[i, "Omsättning om 3 år"])      * ps_snitt) / aktier_ut_milj, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0

        # EV/EBITDA om det går
        ev = _safe_float(rad.get("EV", 0.0))
        ebitda = _safe_float(rad.get("EBITDA (TTM)", 0.0))
        if ev > 0 and ebitda > 0:
            df.at[i, "EV/EBITDA"] = round(ev / ebitda, 2)
        else:
            df.at[i, "EV/EBITDA"] = _safe_float(rad.get("EV/EBITDA", 0.0))

        # P/E (TTM) om EPS finns
        eps = _safe_float(rad.get("EPS (TTM)", 0.0))
        pris = _safe_float(rad.get("Aktuell kurs", 0.0))
        if eps > 0 and pris > 0:
            df.at[i, "P/E (TTM)"] = round(pris / eps, 2)
        else:
            df.at[i, "P/E (TTM)"] = _safe_float(rad.get("P/E (TTM)", 0.0))

        # Uppskatta MCAP nu om möjligt (kurs * shares)
        shares_st = _safe_float(rad.get("Utestående aktier", 0.0)) * 1e6  # tillbaka till styck
        if pris > 0 and shares_st > 0:
            mcap_calc = pris * shares_st
            # skriv endast om fältet saknas eller 0 → undvik onödiga överlagringar
            if _safe_float(rad.get("MCAP nu", 0.0)) <= 0:
                df.at[i, "MCAP nu"] = float(mcap_calc)

    return df

# --- Auto-merge / skrivregler -----------------------------------------------

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, changes_map: dict, stamp_even_if_same: bool = True) -> bool:
    """
    Skriver endast fält som får ett nytt (positivt/meningsfullt) värde – men
    respekterar att 'Omsättning idag' och 'Omsättning nästa år' ALDRIG skrivs automatiskt.
    Uppdaterar TS_ för spårade fält. Om 'stamp_even_if_same' = True, tidsstämplas
    även om värdet råkar vara oförändrat (för att markera att källan körts).
    Returnerar True om något fält faktiskt ändrades, men tidsstämplar kan ske ändå.
    """
    # plocka bort manuella-only fält helt
    new_vals = strip_manual_only_fields(new_vals.copy() if isinstance(new_vals, dict) else {})

    changed_fields = []
    any_valid_candidate = False

    for f, v in new_vals.items():
        if f not in df.columns:
            continue

        # skrivregler (positiva / meningsfulla)
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            write_ok = v is not None

        if not write_ok:
            continue

        any_valid_candidate = True
        old = df.at[row_idx, f]

        # Skriv endast om värdet verkligen ändras
        value_changed = (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))
        if value_changed:
            df.at[row_idx, f] = v
            changed_fields.append(f)

        # TS-stämpel även om värdet inte ändrats, om fältet är spårat
        if f in TS_FIELDS and (value_changed or stamp_even_if_same):
            _stamp_ts_for_field(df, row_idx, f)

    # Om något fält ändrades → sätt auto-uppdaterad/källa
    # Om inget ändrades men vi åtminstone hade en giltig kandidat och stämplat TS → sätt ändå.
    if changed_fields or (any_valid_candidate and stamp_even_if_same):
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            tkr = df.at[row_idx, "Ticker"] if "Ticker" in df.columns else f"row{row_idx}"
            changes_map.setdefault(str(tkr), []).extend(changed_fields)
        return bool(changed_fields)

    return False

# --- Snapshots, ålder & hjälpare -------------------------------------------

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
    (Fokus på de spårade fälten, *inte* på manuella prognoser – de listas separat)
    """
    need_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]
    ts_cols = [TS_FIELDS[c] for c in TS_FIELDS if c in need_cols]

    out_rows = []
    cutoff = now_dt() - timedelta(days=older_than_days)

    for _, r in df.iterrows():
        missing_val = any((_safe_float(r.get(c, 0.0)) <= 0.0) for c in need_cols)
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

def build_manual_forecast_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lista över bolag sorterade på äldst TS för *manuella* fälten
    'Omsättning idag' och 'Omsättning nästa år'.
    Visas i Lägg till/uppdatera-vyn.
    """
    rows = []
    for _, r in df.iterrows():
        ts_idag = str(r.get("TS_Omsättning idag", "") or "").strip()
        ts_next = str(r.get("TS_Omsättning nästa år", "") or "").strip()
        d_idag = pd.to_datetime(ts_idag, errors="coerce")
        d_next = pd.to_datetime(ts_next, errors="coerce")
        oldest = min([d for d in [d_idag, d_next] if pd.notna(d)], default=pd.NaT)
        rows.append({
            "Ticker": r.get("Ticker",""),
            "Bolagsnamn": r.get("Bolagsnamn",""),
            "TS_Omsättning idag": ts_idag,
            "TS_Omsättning nästa år": ts_next,
            "Äldst (manuell)": oldest
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Äldst_sort"] = pd.to_datetime(out["Äldst (manuell)"], errors="coerce").fillna(pd.Timestamp("1990-01-01"))
    out = out.sort_values(by=["Äldst_sort","Bolagsnamn"], ascending=[True, True]).drop(columns=["Äldst_sort"])
    return out

# --- Auto-hämtning pipeline --------------------------------------------------

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares) eller Yahoo global fallback
      2) FMP light (P/S + pris/shares/mcap) som komplettering
    Returnerar (vals, debug)
    OBS: MANUAL_ONLY_FIELDS tas alltid bort innan skrivning.
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","Sektor","MCAP nu","_debug_shares_source"
        ]}
        for k in ["Bolagsnamn","Valuta","Sektor","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","MCAP nu","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4"]:
            v = base.get(k, None)
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) FMP light som komplettering om något saknas
    try:
        fmpl = hamta_fmp_falt_light(ticker)
        debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier"), "Aktuell kurs": fmpl.get("Aktuell kurs")}
        for k in ["Aktuell kurs","Utestående aktier","P/S"]:
            v = fmpl.get(k)
            if v not in (None, "", 0, 0.0) and k not in vals:
                vals[k] = v
        if ("MCAP nu" not in vals) and fmpl.get("_marketCap"):
            try:
                mc = float(fmpl.get("_marketCap"))
                if mc > 0: vals["MCAP nu"] = mc
            except Exception:
                pass
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # Sanera från manuella-only fält
    vals = strip_manual_only_fields(vals)
    return vals, debug

def _list_tickers_ordered(df: pd.DataFrame, order: str = "Äldst först") -> list:
    """
    Returnerar lista med tickers i vald ordning.
    - 'Äldst först' → sortera på äldsta TS bland spårade fält
    - 'A–Ö' → Bolagsnamn, Ticker
    """
    order = (order or "").lower()
    if order.startswith("ä") or "old" in order:
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"], ascending=[True, True, True])
        return [str(t).upper() for t in work["Ticker"].tolist()]
    # default A–Ö
    work = df.copy().sort_values(by=["Bolagsnamn","Ticker"])
    return [str(t).upper() for t in work["Ticker"].tolist()]

def prepare_batch_queue(df: pd.DataFrame, order: str = "Äldst först", size: int = 20):
    """
    Förbereder batch-kö i session_state:
      - st.session_state.batch_order
      - st.session_state.batch_size
      - st.session_state.batch_queue (lista av tickers)
      - st.session_state.batch_done (set)
    """
    tickers = _list_tickers_ordered(df, order=order)
    st.session_state.batch_order = order
    st.session_state.batch_size = int(max(1, size))
    st.session_state.batch_queue = tickers
    st.session_state.batch_done = set()
    st.success(f"Batchkö förberedd: {len(tickers)} tickers ({order}).")

def _select_next_batch(n: int) -> list:
    """
    Tar nästa n tickers från batch_queue som inte finns i batch_done.
    """
    q = st.session_state.get("batch_queue", []) or []
    done = st.session_state.get("batch_done", set())
    out = []
    for t in q:
        if t not in done:
            out.append(t)
            if len(out) >= n:
                break
    return out

def run_batch_step(df: pd.DataFrame, user_rates: dict, snapshot_before_write: bool = True):
    """
    Kör en *stegvis* batch utifrån session_state (size/order).
    Hämtar nästa N tickers och uppdaterar dem. Sparar endast om ändringar skett.
    Visar progressbar + 'i/X'-text.
    """
    if "batch_queue" not in st.session_state:
        st.warning("Ingen batchkö är förberedd. Förbered först.")
        return df, {"changed": {}, "misses": {}, "debug_first_20": []}

    size = int(st.session_state.get("batch_size", 20))
    targets = _select_next_batch(size)
    if not targets:
        st.info("Batchkön är redan klar (inga kvar).")
        return df, st.session_state.get("last_auto_log", {"changed": {}, "misses": {}, "debug_first_20": []})

    total = len(targets)
    progress = st.sidebar.progress(0, text=f"0/{total}")
    status = st.sidebar.empty()

    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    any_changed = False

    for i, tkr in enumerate(targets, start=1):
        status.write(f"Uppdaterar {i}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            # hitta radindex
            idx_list = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()].tolist()
            if not idx_list:
                log["misses"][tkr] = ["hittades ej i tabellen"]
            else:
                ridx = idx_list[0]
                changed = apply_auto_updates_to_row(df, ridx, new_vals, source="Auto (SEC/Yahoo→FMP light)", changes_map=log["changed"], stamp_even_if_same=True)
                if not changed:
                    # även om inget ändrades, markera "miss" för transparens
                    log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
                any_changed = any_changed or changed
                if i <= 20:
                    log["debug_first_20"].append({tkr: debug})

            # markera klar i batch_done
            st.session_state.batch_done.add(tkr)

        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress(i/total, text=f"{i}/{total}")

    # Räkna om & ev. spara
    df = uppdatera_berakningar(df, user_rates)
    if any_changed:
        spara_data(df, do_snapshot=snapshot_before_write)
        st.sidebar.success("Batchsteg klart! Ändringar sparade.")
    else:
        st.sidebar.info("Batchsteg klart – inga faktiska ändringar, ingen skrivning.")

    st.session_state["last_auto_log"] = log
    return df, log

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False):
    """
    Full auto-uppdatering (alla tickers) – tung. Behålls för “nån gång ibland”.
    Använder progressbar + i/X-text. Respekterar manuella fält.
    """
    tickers = _list_tickers_ordered(df, order="A–Ö")
    total = len(tickers)
    progress = st.sidebar.progress(0, text=f"0/{total}")
    status = st.sidebar.empty()
    log = {"changed": {}, "misses": {}, "debug_first_20": []}
    any_changed = False

    for i, tkr in enumerate(tickers, start=1):
        status.write(f"Uppdaterar {i}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            idx_list = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()].tolist()
            if not idx_list:
                log["misses"][tkr] = ["hittades ej i tabellen"]
            else:
                ridx = idx_list[0]
                changed = apply_auto_updates_to_row(df, ridx, new_vals, source="Auto (SEC/Yahoo→FMP light)", changes_map=log["changed"], stamp_even_if_same=True)
                if not changed:
                    log["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
                any_changed = any_changed or changed
                if i <= 20:
                    log["debug_first_20"].append({tkr: debug})
        except Exception as e:
            log["misses"][tkr] = [f"error: {e}"]

        progress.progress(i/total, text=f"{i}/{total}")

    df = uppdatera_berakningar(df, user_rates)

    if any_changed:
        spara_data(df, do_snapshot=make_snapshot)
        st.sidebar.success("Klart! Ändringar sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    st.session_state["last_auto_log"] = log
    return df, log

# --- Snabb uppdatering: endast kurs för EN ticker ---------------------------

def update_price_only_for_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Hämtar endast aktuell kurs (och valuta om den kommer med) från Yahoo för en enskild ticker.
    Påverkar inte några spårade fält / TS (eftersom 'Aktuell kurs' ej är TS-fält).
    """
    if not ticker:
        return df
    t = str(ticker).upper().strip()
    idx_list = df.index[df["Ticker"].astype(str).str.upper() == t].tolist()
    if not idx_list:
        st.warning(f"{t} hittades inte i tabellen.")
        return df
    ridx = idx_list[0]
    y = hamta_yahoo_fält(t)
    if y.get("Aktuell kurs", 0) > 0:
        df.at[ridx, "Aktuell kurs"] = float(y["Aktuell kurs"])
    if y.get("Valuta"):
        df.at[ridx, "Valuta"] = str(y["Valuta"]).upper()
    return df

# --- Debug: testa datakällor för en ticker ----------------------------------

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

# --- KONTROLL & LÄGG TILL/UPPDATERA -----------------------------------------

def _ts_badge(date_str: str, label: str) -> str:
    if not str(date_str).strip():
        return f"🕓 {label}: –"
    try:
        d = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(d):
            return f"🕓 {label}: –"
        days = (now_dt().date() - d.date()).days
        if days <= 7:
            return f"🟢 {label}: {d.date().isoformat()}"
        if days <= 30:
            return f"🟡 {label}: {d.date().isoformat()}"
        return f"🟠 {label}: {d.date().isoformat()}"
    except Exception:
        return f"🕓 {label}: –"

def _render_update_badges(row: pd.Series):
    c1, c2 = st.columns(2)
    with c1:
        st.caption(_ts_badge(row.get("Senast manuellt uppdaterad",""), "Senast manuellt"))
    with c2:
        src = str(row.get("Senast uppdaterad källa","") or "")
        txt = _ts_badge(row.get("Senast auto-uppdaterad",""), "Senast auto")
        if src:
            txt += f"  •  Källa: *{src}*"
        st.caption(txt)

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns: cols_show.append(k)
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Kräver manuell hantering? (generell datakvalitet)
    st.subheader("🛠️ Kräver manuell hantering (datakvalitet)")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30)
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Batchkörning – UI
    st.subheader("🚚 Batchkörning (stegvis)")
    colL, colR = st.columns([2,1])
    with colL:
        order = st.selectbox("Ordning", ["Äldst först","A–Ö"], index=0, key="batch_order_ui")
        size = st.number_input("Antal per körning", min_value=1, max_value=200, value=20, step=1, key="batch_size_ui")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📦 Förbered kö"):
                prepare_batch_queue(df, order=order, size=int(size))
        with c2:
            if st.button("▶️ Kör nästa"):
                df2, log = run_batch_step(df, user_rates={"USD":1,"NOK":1,"CAD":1,"EUR":1,"SEK":1}, snapshot_before_write=True)
                # uppdatera referens i session_state om sådant används i main
                st.session_state["_df_ref"] = df2
        with c3:
            left = 0
            if "batch_queue" in st.session_state and "batch_done" in st.session_state:
                left = max(0, len(st.session_state.batch_queue) - len(st.session_state.batch_done))
            st.metric("Återstår i kö", left)
    with colR:
        st.write("**Status**")
        if "last_auto_log" in st.session_state and st.session_state["last_auto_log"]:
            ch = st.session_state["last_auto_log"].get("changed", {})
            ms = st.session_state["last_auto_log"].get("misses", {})
            st.caption(f"Senaste körning – ändrade: {len(ch)} • missar: {len(ms)}")
        else:
            st.caption("Ingen körning i denna session ännu.")

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sortering för redigering
    sort_val = st.selectbox("Sortera lista", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (spårade fält)"], index=0)
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: 
        st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(val_lista)-1))

    # Select + bläddra
    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=st.session_state.edit_index)
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

    # Huvudform – basdata (manuell inmatning av P/S & prognoser sker separat)
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            valuta = st.text_input("Valuta (t.ex. USD/EUR/SEK)", value=bef.get("Valuta","USD") if not bef.empty else "USD").upper()
        with c2:
            bol_namn = st.text_input("Bolagsnamn", value=bef.get("Bolagsnamn","") if not bef.empty else "")
            kurs = st.number_input("Aktuell kurs", value=float(bef.get("Aktuell kurs",0.0)) if not bef.empty else 0.0)
            utd = st.number_input("Årlig utdelning (per aktie)", value=float(bef.get("Årlig utdelning",0.0)) if not bef.empty else 0.0)
            cagr5 = st.number_input("CAGR 5 år (%)", value=float(bef.get("CAGR 5 år (%)",0.0)) if not bef.empty else 0.0)
        spara_bas = st.form_submit_button("💾 Spara basdata")

    if spara_bas and ticker:
        # skriv in (utan att röra manuella prognosfält här)
        if (df["Ticker"] == ticker).any():
            ridx = df.index[df["Ticker"]==ticker][0]
            df.loc[ridx, "Utestående aktier"] = utest
            df.loc[ridx, "Antal aktier"] = antal
            df.loc[ridx, "Valuta"] = valuta
            df.loc[ridx, "Bolagsnamn"] = bol_namn
            df.loc[ridx, "Aktuell kurs"] = kurs
            df.loc[ridx, "Årlig utdelning"] = utd
            df.loc[ridx, "CAGR 5 år (%)"] = cagr5
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update({
                "Ticker": ticker, "Bolagsnamn": bol_namn, "Valuta": valuta,
                "Utestående aktier": utest, "Antal aktier": antal,
                "Aktuell kurs": kurs, "Årlig utdelning": utd, "CAGR 5 år (%)": cagr5
            })
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Basdata sparad.")

    # Visa valt bolag + badges
    if valt_label and valt_label in namn_map and (df["Ticker"]==namn_map[valt_label]).any():
        ridx = df.index[df["Ticker"]==namn_map[valt_label]][0]
        row = df.loc[ridx]
        st.subheader(f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})")
        _render_update_badges(row)

        show_cols = [
            "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "CAGR 5 år (%)","Antal aktier","Årlig utdelning","MCAP nu",
            "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
            "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(pd.DataFrame([row[show_cols].to_dict()]), use_container_width=True, hide_index=True)

        # Snabbknappar
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            if st.button("💹 Uppdatera kurs (Yahoo)"):
                df = update_price_only_for_ticker(df, row["Ticker"])
                df = uppdatera_berakningar(df, user_rates)
                spara_data(df)
                st.success("Kurs uppdaterad.")
        with cc2:
            if st.button("🤖 Full auto för detta bolag"):
                new_vals, debug = auto_fetch_for_ticker(row["Ticker"])
                changed = apply_auto_updates_to_row(df, ridx, new_vals, source="Auto (SEC/Yahoo→FMP light)", changes_map={}, stamp_even_if_same=True)
                df = uppdatera_berakningar(df, user_rates)
                if changed:
                    spara_data(df)
                    st.success("Auto-uppdatering klar och sparad.")
                else:
                    st.info("Inga ändringar hittades vid auto-uppdatering.")
        with cc3:
            if st.button("🔎 Debug datakällor"):
                debug_test_single_ticker(row["Ticker"])

        # Manuell prognos – redigera endast dessa 2 fält
        with st.expander("✍️ Manuell prognos (Omsättning i miljoner – *sparas direkt* med TS)"):
            f1, f2 = st.columns(2)
            with f1:
                oms_idag_in = st.number_input("Omsättning idag (miljoner)", value=float(row.get("Omsättning idag",0.0)))
            with f2:
                oms_next_in = st.number_input("Omsättning nästa år (miljoner)", value=float(row.get("Omsättning nästa år",0.0)))
            if st.button("💾 Spara prognos (manuell)"):
                df.at[ridx, "Omsättning idag"] = float(oms_idag_in)
                df.at[ridx, "Omsättning nästa år"] = float(oms_next_in)
                # stämpla manuellt + per-fält-TS
                _note_manual_update(df, ridx)
                _stamp_ts_for_field(df, ridx, "Omsättning idag")
                _stamp_ts_for_field(df, ridx, "Omsättning nästa år")
                # räkna om och spara
                df = uppdatera_berakningar(df, user_rates)
                spara_data(df)
                st.success("Manuell prognos sparad och tidsstämplad.")

    st.divider()

    # --- Manuell prognoslista (äldre först) ---
    st.subheader("🗒️ Manuell prognoslista (äldst först)")
    mp = build_manual_forecast_watchlist(df)
    if mp.empty:
        st.info("Inga poster i prognoslistan ännu.")
    else:
        st.dataframe(mp[["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år","Äldst (manuell)"]], use_container_width=True, hide_index=True)

    return df

# --- Hjälpare för investeringslogik -----------------------------------------

def _fx(v: str, user_rates: dict) -> float:
    if not v:
        return 1.0
    return float(user_rates.get(str(v).upper(), 1.0))

def _mcap_from_row(row: pd.Series) -> float:
    """MCAP nu (kr) i *bolagsvaluta*. Preferera 'MCAP nu'; annars kurs*shares."""
    mcap = _safe_float(row.get("MCAP nu", 0.0))
    if mcap > 0:
        return mcap
    px = _safe_float(row.get("Aktuell kurs", 0.0))
    sh_mil = _safe_float(row.get("Utestående aktier", 0.0))
    if px > 0 and sh_mil > 0:
        return px * sh_mil * 1e6
    return 0.0

def risk_label_from_mcap(mcap: float) -> str:
    """Klassning efter market cap (USD-gränser typiskt – vi använder *bolagsvaluta* här som proxy)."""
    # Skärningar: Micro<300M, Small<2B, Mid<10B, Large<200B, annars Mega
    if mcap >= 2e11: return "Mega"
    if mcap >= 1e10: return "Large"
    if mcap >= 2e9:  return "Mid"
    if mcap >= 3e8:  return "Small"
    return "Micro"

def _norm(value: float, lo: float, hi: float, clamp: bool = True) -> float:
    """Normalisera till 0..1. Om clamp=False → kan bli <0/>1."""
    try:
        x = (float(value) - lo) / (hi - lo) if hi != lo else 0.5
    except Exception:
        x = 0.5
    if clamp:
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
    return x

def _inv(x: float) -> float:
    """Invert – t.ex. skuldsättning där lägre är bättre. x antas redan vara 0..∞."""
    try:
        return 1.0 / (1.0 + float(x))  # 0=>1, 1=>0.5, 4=>0.2
    except Exception:
        return 0.5

def _safe_pct(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def _current_ps(row: pd.Series) -> float:
    ps_now = _safe_float(row.get("P/S", 0.0))
    if ps_now > 0:
        return ps_now
    # fallback om vi har P/S-snitt och marknadskap/omsättning idag
    mcap = _mcap_from_row(row)
    rev = _safe_float(row.get("Omsättning idag", 0.0)) * 1e6  # miljoner → enheter
    if mcap > 0 and rev > 0:
        return mcap / rev
    return 0.0

def _ps_avg4(row: pd.Series) -> float:
    vals = [
        _safe_float(row.get("P/S Q1",0.0)),
        _safe_float(row.get("P/S Q2",0.0)),
        _safe_float(row.get("P/S Q3",0.0)),
        _safe_float(row.get("P/S Q4",0.0)),
    ]
    pos = [v for v in vals if v > 0]
    return round(float(np.mean(pos)), 2) if pos else 0.0

def _score_growth(row: pd.Series, potential_pct: float) -> float:
    """Poäng för tillväxtläge – 0..1."""
    # inputs (valfria)
    cagr = _safe_pct(row.get("CAGR 5 år (%)", 0.0))
    gm = _safe_pct(row.get("Bruttomarginal (%)", row.get("Gross margin (%)", 0.0)))
    nm = _safe_pct(row.get("Nettomarginal (%)", row.get("Net margin (%)", 0.0)))
    de = _safe_float(row.get("Debt/Equity", row.get("Skuldsättningsgrad", 0.0)))
    ev_ebitda = _safe_float(row.get("EV/EBITDA", 0.0))
    ps_now = _current_ps(row)
    ps_avg = _ps_avg4(row)

    # komponenter (alla 0..1)
    k_up   = _norm(potential_pct, -30, 60)       # -30%..+60%
    k_cagr = _norm(cagr, 0, 40)                  # 0..40%+
    k_gm   = _norm(gm, 20, 80)
    k_nm   = _norm(nm, 0, 30)
    k_de   = _inv(de)                            # lägre skuld bättre
    k_ev   = _inv(ev_ebitda if ev_ebitda>0 else 10)  # 10 ~ neutral
    k_relp = _norm((ps_avg/ps_now) if (ps_now>0 and ps_avg>0) else 1.0, 0.5, 1.5)  # <1 bättre

    # vikter
    w = {
        "up":0.40, "cagr":0.20, "gm":0.10, "nm":0.10, "de":0.10, "ev":0.05, "relp":0.05
    }
    score = (k_up*w["up"] + k_cagr*w["cagr"] + k_gm*w["gm"] + k_nm*w["nm"] + k_de*w["de"] + k_ev*w["ev"] + k_relp*w["relp"])
    return float(round(score, 4))

def _score_dividend(row: pd.Series, potential_pct: float) -> float:
    """Poäng för utdelningsläge – 0..1."""
    price = _safe_float(row.get("Aktuell kurs", 0.0))
    div_pa = _safe_float(row.get("Årlig utdelning", 0.0))
    dy = (div_pa / price * 100.0) if (price > 0 and div_pa > 0) else 0.0
    fcf = _safe_float(row.get("FCF (TTM)", 0.0))
    div_ttm = _safe_float(row.get("Utdelningar (TTM)", 0.0))
    cov = (fcf / div_ttm) if (fcf > 0 and div_ttm > 0) else 1.0
    dgr = _safe_pct(row.get("Utdelningstillväxt 5y (%)", 0.0))
    de  = _safe_float(row.get("Debt/Equity", 0.0))
    ps_now = _current_ps(row)
    ps_avg = _ps_avg4(row)

    # komponenter
    k_y   = _norm(dy, 2, 10)                 # 2..10%+
    k_cov = _norm(cov, 1, 3)                 # 1..3+
    k_dgr = _norm(dgr, 0, 12)                # 0..12%+
    k_de  = _inv(de)
    k_up  = _norm(potential_pct, -20, 40)    # -20..+40
    k_relp= _norm((ps_avg/ps_now) if (ps_now>0 and ps_avg>0) else 1.0, 0.6, 1.4)

    w = {"y":0.40, "cov":0.20, "dgr":0.10, "de":0.10, "up":0.10, "relp":0.10}
    score = k_y*w["y"] + k_cov*w["cov"] + k_dgr*w["dgr"] + k_de*w["de"] + k_up*w["up"] + k_relp*w["relp"]
    return float(round(score, 4))

def _score_label(score: float) -> str:
    if score >= 0.75: return "🌟 Utmärkt"
    if score >= 0.60: return "✅ Bra"
    if score >= 0.45: return "🟨 Ok"
    return "⚠️ Svag"

def _valuation_label(potential_pct: float, score: float) -> str:
    # enkel policy som kombinerar uppsida & score
    if potential_pct >= 30 and score >= 0.65: return "KÖP"
    if potential_pct >= 10 and score >= 0.55: return "Öka"
    if -10 <= potential_pct <= 10: return "Behåll"
    if potential_pct <= -20 and score < 0.40: return "Sälj/Övervärderad"
    if potential_pct < -10 and score < 0.45: return "Trimma"
    return "Behåll"

# --- Portfölj ----------------------------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: _fx(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {human_money(total_värde)} SEK")
    st.markdown(f"**Total kommande utdelning:** {human_money(tot_utd)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {human_money(tot_utd/12.0)} SEK")

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

# --- Investeringsförslag -----------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Läge: Tillväxt / Utdelning
    mode = st.radio("Läge", ["Tillväxt","Utdelning"], horizontal=True, index=0)

    # Filtrering
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)
    subset = st.radio("Urval", ["Alla bolag","Endast portfölj"], horizontal=True, index=0)

    # Sektor & Risklabel filter
    df_work = df.copy()
    df_work["Sektor"] = df_work["Sektor"].fillna("").replace("", "Okänd")
    sektorer = sorted(df_work["Sektor"].unique().tolist())
    choose_sekt = st.multiselect("Filtrera sektor(er)", options=sektorer, default=sektorer)
    df_work = df_work[df_work["Sektor"].isin(choose_sekt)]

    # Risklabel
    df_work["MCAP now calc"] = df_work.apply(_mcap_from_row, axis=1)
    df_work["Risklabel"] = df_work["MCAP now calc"].apply(risk_label_from_mcap)
    risk_opts = ["Micro","Small","Mid","Large","Mega"]
    choose_risk = st.multiselect("Filtrera risklabel", options=risk_opts, default=risk_opts)
    df_work = df_work[df_work["Risklabel"].isin(choose_risk)]

    # Subsetval
    base = df_work[df_work["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df_work.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar nuvarande filter.")
        return

    # Beräkningar för förslag
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["P/S (nu)"] = base.apply(_current_ps, axis=1)
    base["P/S (snitt 4q)"] = base.apply(_ps_avg4, axis=1)
    base["MCAP nu (fmt)"] = base.apply(lambda r: human_money(_mcap_from_row(r)), axis=1)

    # Score
    if mode == "Tillväxt":
        base["Score"] = base.apply(lambda r: _score_growth(r, _safe_float(r.get("Potential (%)",0.0))), axis=1)
    else:
        base["Score"] = base.apply(lambda r: _score_dividend(r, _safe_float(r.get("Potential (%)",0.0))), axis=1)

    base["Score label"] = base["Score"].apply(_score_label)
    base["Rekommendation"] = base.apply(lambda r: _valuation_label(_safe_float(r.get("Potential (%)",0.0)), _safe_float(r.get("Score",0.0))), axis=1)

    # Sortering: högst score överst
    base = base.sort_values(by=["Score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # Bläddrings-UI
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

    # Sammanfattning
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.caption(f"Sektor: **{rad.get('Sektor','Okänd')}** • Risklabel: **{rad.get('Risklabel','?')}**")
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Score:** {rad['Score']:.2f}  •  **{rad['Score label']}**  •  **{rad['Rekommendation']}**",
    ]
    # Utestående aktier synligt (du efterfrågade det)
    lines.append(f"- **Utestående aktier:** {human_money(_safe_float(rad.get('Utestående aktier',0.0))*1e6)} st")
    # Marketcap synligt
    lines.append(f"- **Market cap (nu):** {rad['MCAP nu (fmt)']} ({rad['Valuta']})")
    # P/S detaljer
    lines.append(f"- **P/S (nu):** {(_safe_float(rad.get('P/S (nu)', rad.get('P/S',0.0)))):.2f} • **P/S snitt (4q):** {rad['P/S (snitt 4q)']:.2f}")

    st.markdown("\n".join(lines))

    # Expanders
    with st.expander("📊 Värderingsdetaljer"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**P/S per kvartal (senaste→äldsta)**")
            st.write(f"Q1: {rad.get('P/S Q1',0):.2f}  •  Q2: {rad.get('P/S Q2',0):.2f}")
            st.write(f"Q3: {rad.get('P/S Q3',0):.2f}  •  Q4: {rad.get('P/S Q4',0):.2f}")
        with c2:
            st.write("**Market cap historik (om finns)**")
            mcaps = []
            for q in ["MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4"]:
                v = _safe_float(rad.get(q, 0.0))
                if v > 0:
                    mcaps.append(f"{q}: {human_money(v)}")
            if mcaps:
                st.write(" • ".join(mcaps))
            else:
                st.write("–")

    with st.expander("💵 Kassaflöde & lönsamhet (om tillgängligt)"):
        gm = rad.get("Bruttomarginal (%)", rad.get("Gross margin (%)", None))
        nm = rad.get("Nettomarginal (%)", rad.get("Net margin (%)", None))
        ebitda = rad.get("EBITDA (TTM)", None)
        ev = rad.get("EV", None)
        ev_e = rad.get("EV/EBITDA", None)
        fcf = rad.get("FCF (TTM)", None)
        rowL, rowR = st.columns(2)
        with rowL:
            st.write(f"**Bruttomarginal:** {gm if gm not in (None,'') else '–'} %")
            st.write(f"**Nettomarginal:** {nm if nm not in (None,'') else '–'} %")
            st.write(f"**EBITDA (TTM):** {human_money(_safe_float(ebitda)) if ebitda not in (None,'') else '–'}")
        with rowR:
            st.write(f"**EV:** {human_money(_safe_float(ev)) if ev not in (None,'') else '–'}")
            st.write(f"**EV/EBITDA:** {ev_e if ev_e not in (None,'') else '–'}")
            st.write(f"**FCF (TTM):** {human_money(_safe_float(fcf)) if fcf not in (None,'') else '–'}")

    with st.expander("🏦 Skuld & risk (om tillgängligt)"):
        de = rad.get("Debt/Equity", None)
        st.write(f"**Debt/Equity:** {de if de not in (None,'') else '–'}")
        st.write(f"**Risklabel:** {rad.get('Risklabel','–')}")

    # Tabellvy under (10 topp)
    st.markdown("### Topp 10 enligt score")
    show = base[[
        "Ticker","Bolagsnamn","Sektor","Risklabel",
        "Aktuell kurs", riktkurs_val, "Potential (%)",
        "P/S (nu)","P/S (snitt 4q)","MCAP nu (fmt)","Score","Score label","Rekommendation"
    ]].head(10)
    st.dataframe(show, use_container_width=True, hide_index=True)

# --- Sidopanel: valutakurser & snabbuttag -----------------------------------

def _ensure_rate_state_defaults():
    # Läs sparade som baseline (en gång per session)
    saved = las_sparade_valutakurser()
    if "rate_usd" not in st.session_state: st.session_state.rate_usd = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
    if "rate_nok" not in st.session_state: st.session_state.rate_nok = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
    if "rate_cad" not in st.session_state: st.session_state.rate_cad = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
    if "rate_eur" not in st.session_state: st.session_state.rate_eur = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
    if "rate_sek" not in st.session_state: st.session_state.rate_sek = 1.0
    # UI keys (separata för att undvika "key already set"-fel)
    if "rate_usd_input" not in st.session_state: st.session_state.rate_usd_input = st.session_state.rate_usd
    if "rate_nok_input" not in st.session_state: st.session_state.rate_nok_input = st.session_state.rate_nok
    if "rate_cad_input" not in st.session_state: st.session_state.rate_cad_input = st.session_state.rate_cad
    if "rate_eur_input" not in st.session_state: st.session_state.rate_eur_input = st.session_state.rate_eur

def _sidebar_rates() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")
    _ensure_rate_state_defaults()

    # Inputs (egna keys)
    usd = st.sidebar.number_input("USD → SEK", key="rate_usd_input", value=float(st.session_state.rate_usd_input), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok_input", value=float(st.session_state.rate_nok_input), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad_input", value=float(st.session_state.rate_cad_input), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur_input", value=float(st.session_state.rate_eur_input), step=0.01, format="%.4f")

    col_rates1, col_rates2, col_rates3 = st.sidebar.columns(3)
    with col_rates1:
        if st.button("🌐 Hämta automatiskt"):
            auto_rates, misses, provider = hamta_valutakurser_auto()
            # uppdatera både "sanning" och ui-keys
            st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
            st.session_state.rate_usd_input = st.session_state.rate_usd
            st.session_state.rate_nok_input = st.session_state.rate_nok
            st.session_state.rate_cad_input = st.session_state.rate_cad
            st.session_state.rate_eur_input = st.session_state.rate_eur
            st.sidebar.success(f"Valutakurser hämtade (källa: {provider}).")
            if misses:
                st.sidebar.warning("Vissa par misslyckades:\n- " + "\n- ".join(misses))
            st.rerun()
    with col_rates2:
        if st.button("💾 Spara kurser"):
            to_save = {
                "USD": float(st.session_state.rate_usd_input),
                "NOK": float(st.session_state.rate_nok_input),
                "CAD": float(st.session_state.rate_cad_input),
                "EUR": float(st.session_state.rate_eur_input),
                "SEK": 1.0,
            }
            spara_valutakurser(to_save)
            # uppdatera “sanningen”
            st.session_state.rate_usd = to_save["USD"]
            st.session_state.rate_nok = to_save["NOK"]
            st.session_state.rate_cad = to_save["CAD"]
            st.session_state.rate_eur = to_save["EUR"]
            st.sidebar.success("Valutakurser sparade.")
    with col_rates3:
        if st.button("↻ Läs sparade kurser"):
            sr = las_sparade_valutakurser()
            st.session_state.rate_usd = float(sr.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(sr.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(sr.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(sr.get("EUR", st.session_state.rate_eur))
            st.session_state.rate_usd_input = st.session_state.rate_usd
            st.session_state.rate_nok_input = st.session_state.rate_nok
            st.session_state.rate_cad_input = st.session_state.rate_cad
            st.session_state.rate_eur_input = st.session_state.rate_eur
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # Retur: använd de UI-justerade värdena för beräkningar
    return {
        "USD": float(st.session_state.rate_usd_input),
        "NOK": float(st.session_state.rate_nok_input),
        "CAD": float(st.session_state.rate_cad_input),
        "EUR": float(st.session_state.rate_eur_input),
        "SEK": 1.0,
    }

def update_all_prices_once(df: pd.DataFrame) -> pd.DataFrame:
    """Snabbkörning: uppdatera ENDAST kurs & valuta för alla tickers (Yahoo)."""
    tickers = [str(t).upper() for t in df["Ticker"].fillna("").astype(str).tolist() if str(t).strip()]
    total = len(tickers)
    if total == 0:
        st.sidebar.info("Inga tickers att uppdatera.")
        return df
    progress = st.sidebar.progress(0, text=f"0/{total}")
    status = st.sidebar.empty()
    for i, t in enumerate(tickers, start=1):
        status.write(f"Uppdaterar kurs {i}/{total}: {t}")
        try:
            y = hamta_yahoo_fält(t)
            idx = df.index[df["Ticker"].astype(str).str.upper()==t].tolist()
            if idx:
                ridx = idx[0]
                if y.get("Aktuell kurs", 0) > 0:
                    df.at[ridx, "Aktuell kurs"] = float(y["Aktuell kurs"])
                if y.get("Valuta"):
                    df.at[ridx, "Valuta"] = str(y["Valuta"]).upper()
        except Exception:
            pass
        progress.progress(i/total, text=f"{i}/{total}")
    st.sidebar.success("Kursuppdatering klar (ingen TS påverkas).")
    return df

# --- Enkel analysvy (bibehåller din gamla känsla) ----------------------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    # robust index
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(st.session_state.analys_idx, len(etiketter)-1)

    st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1, key="analys_idx")
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
    _render_update_badges(r)

    cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","MCAP nu",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# --- MAIN --------------------------------------------------------------------

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: valutor
    user_rates = _sidebar_rates()

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Snabbkörningar")
    if st.sidebar.button("💹 Uppdatera endast kurser (alla tickers)"):
        df0 = hamta_data()
        df0 = säkerställ_kolumner(df0)
        df0 = migrera_gamla_riktkurskolumner(df0)
        df0 = konvertera_typer(df0)
        df0 = update_all_prices_once(df0)
        df0 = uppdatera_berakningar(df0, user_rates)
        spara_data(df0)
        st.sidebar.success("Klar – kurser uppdaterade och sparade.")
        st.rerun()

    st.sidebar.subheader("🛠️ Auto-uppdatering (tung)")
    make_snapshot = st.sidebar.checkbox("Skapa snapshot före skrivning", value=True)
    if st.sidebar.button("🔄 Auto-uppdatera alla (SEC/Yahoo → FMP light)"):
        df0 = hamta_data()
        df0 = säkerställ_kolumner(df0)
        df0 = migrera_gamla_riktkurskolumner(df0)
        df0 = konvertera_typer(df0)
        df0, log = auto_update_all(df0, user_rates, make_snapshot=make_snapshot)
        st.session_state["last_auto_log"] = log
        st.rerun()

    # Läs data (efter ev. sidopanelåtgärder)
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    # Säkerställ schema, migrera och typer + räkna
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)
    df = uppdatera_berakningar(df, user_rates)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        if not df2.equals(df):
            # redan sparat inuti funktionen, men uppdatera lokalt
            df = df2
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
