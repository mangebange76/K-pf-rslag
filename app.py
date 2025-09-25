# -*- coding: utf-8 -*-
# app.py — Del 1/8

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
from typing import Optional, Tuple, List, Dict

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

# app.py — Del 2/8
# --- Kolumnschema, TS-fält & typ-hantering -----------------------------------

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

# Slutlig kolumnlista i databasen (inkl. nya nyckeltal/CF/BS/IS & mcap-historik)
FINAL_COLS = [
    # Identitet & bas
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)", "Beta", "Sektor",
    "Utestående aktier", "MCap (nu)",

    # P/S & historik
    "P/S", "P/S (Yahoo)", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",

    # Omsättningar & riktkurser
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Portfölj
    "Antal aktier",

    # MCAP historik (matchar TTM-punkter som P/S Q1..Q4)
    "MCap Q1", "MCap Q2", "MCap Q3", "MCap Q4",
    "MCap Datum Q1", "MCap Datum Q2", "MCap Datum Q3", "MCap Datum Q4",

    # Kassaflöden / Balans / Resultat (kompakt uppsättning per diskussion)
    "Kassa", "Total skuld",
    "Operativt kassaflöde (Q)", "CapEx (Q)", "Fritt kassaflöde (Q)",
    "Burn rate (Q)", "Runway (kvartal)",
    "Operating Expense (Q)", "FoU (Q)", "SG&A (Q)",
    "EBITDA (TTM)", "Räntekostnad (TTM)",
    "Current assets", "Current liabilities",

    # Metadata om uppdateringar
    "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",

    # TS-kolumner (en per spårat fält)
    TS_FIELDS["Utestående aktier"],
    TS_FIELDS["P/S"], TS_FIELDS["P/S Q1"], TS_FIELDS["P/S Q2"], TS_FIELDS["P/S Q3"], TS_FIELDS["P/S Q4"],
    TS_FIELDS["Omsättning idag"], TS_FIELDS["Omsättning nästa år"],
]

# Vilka kolumner som är numeriska
NUMERIC_COLS = {
    "Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","Beta","Utestående aktier","MCap (nu)",
    "P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier",
    "MCap Q1","MCap Q2","MCap Q3","MCap Q4",
    "Kassa","Total skuld",
    "Operativt kassaflöde (Q)","CapEx (Q)","Fritt kassaflöde (Q)",
    "Burn rate (Q)","Runway (kvartal)",
    "Operating Expense (Q)","FoU (Q)","SG&A (Q)",
    "EBITDA (TTM)","Räntekostnad (TTM)",
    "Current assets","Current liabilities",
}

TEXT_COLS = {
    "Ticker","Bolagsnamn","Valuta","Sektor",
    "MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
    "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
}

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade kolumner och sätt rimliga defaultvärden."""
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if kol in NUMERIC_COLS:
                df[kol] = 0.0
            elif kol in TEXT_COLS or kol.startswith("MCap Datum"):
                df[kol] = ""
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            else:
                df[kol] = ""
    # Ta bort ev. dubblerade kolumner
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
    # Numeriska
    for c in df.columns:
        if c in NUMERIC_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Text
    for c in df.columns:
        if (c in TEXT_COLS) or c.startswith("TS_"):
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

# Robust sök efter rad-index via ticker
def _find_row_idx_by_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    if "Ticker" not in df.columns:
        return None
    t = (ticker or "").upper().strip()
    if not t:
        return None
    try:
        mask = df["Ticker"].astype(str).str.upper().str.strip() == t
        idxs = df.index[mask]
        if len(idxs) == 0:
            return None
        return int(idxs[0])
    except Exception:
        return None

# app.py — Del 3/8
# --- Yahoo-hjälpare, finansiella snapshots & beräkningar ---------------------

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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, Beta, P/S (Yahoo), MCap (nu), Sektor, CAGR.
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

        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            try:
                out["Årlig utdelning"] = float(div_rate)
            except Exception:
                pass

        beta = info.get("beta") or info.get("beta3Year") or info.get("beta5Year")
        try:
            if beta is not None:
                out["Beta"] = float(beta)
        except Exception:
            pass

        ps_y = info.get("priceToSalesTrailing12Months") or info.get("priceToSalesTrailing12Months")  # dubblett by design
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

        sektor = info.get("sector") or ""
        if sektor:
            out["Sektor"] = str(sektor)

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# --------- Yahoo: kompakt CF/BS/IS snapshot (senaste kvartal/TTM) ------------
def _pick_row_value(df: pd.DataFrame, names: List[str], agg: str = "latest"):
    """
    För DataFrame med konton i index och datum som kolumner.
    agg='latest' => ta värdet från senaste kolumnen.
    agg='ttm'    => summera senaste 4 kolumnerna.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    # hitta första matchande rad
    row = None
    for nm in names:
        if nm in df.index:
            row = df.loc[nm]
            break
    if row is None:
        # prova case-insensitive
        idx_norm = {str(i).lower(): i for i in df.index}
        for nm in names:
            key = str(nm).lower()
            if key in idx_norm:
                row = df.loc[idx_norm[key]]
                break
    if row is None:
        return 0.0

    # ta bara numeriska, senaste först
    s = pd.to_numeric(row, errors="coerce").dropna()
    if s.empty:
        return 0.0
    s = s.sort_index(ascending=False)
    if agg == "ttm":
        return float(s.head(4).sum())
    return float(s.iloc[0])

def yahoo_financial_snapshots(ticker: str) -> dict:
    """
    Försöker plocka ut nycklar:
    - Kassa, Total skuld (BS, senaste kvartal)
    - Operativt kassaflöde (Q), CapEx (Q), Fritt kassaflöde (Q) (CF, senaste kvartal)
    - Burn rate (Q) = max(0, -FCF Q) ; Runway (kvartal) = Kassa / Burn
    - Operating Expense (Q), FoU (Q), SG&A (Q) (IS, senaste kvartal)
    - EBITDA (TTM), Räntekostnad (TTM) (IS TTM)
    - Current assets, Current liabilities (BS, senaste kvartal)
    """
    out = {}
    try:
        t = yf.Ticker(ticker)

        # Cashflow quarterly
        qcf = getattr(t, "quarterly_cashflow", None)
        ocf_q = _pick_row_value(qcf, ["Operating Cash Flow","Total Cash From Operating Activities","OperatingCashFlow"], agg="latest")
        capex_q = _pick_row_value(qcf, ["Capital Expenditures","CapitalExpenditures","Investments"], agg="latest")
        fcf_q = ocf_q - capex_q

        # Balance sheet quarterly
        qbs = getattr(t, "quarterly_balance_sheet", None)
        cash = _pick_row_value(qbs, ["Cash And Cash Equivalents","Cash","CashAndCashEquivalents"], agg="latest")
        total_debt = _pick_row_value(qbs, ["Total Debt","Long Term Debt","Short Long Term Debt","LongTermDebt","ShortLongTermDebt"], agg="latest")
        current_assets = _pick_row_value(qbs, ["Total Current Assets","Current Assets","CurrentAssets"], agg="latest")
        current_liab   = _pick_row_value(qbs, ["Total Current Liabilities","Current Liabilities","CurrentLiabilities"], agg="latest")

        # Income statement quarterly/ttm
        qis = getattr(t, "quarterly_income_stmt", None)
        opex_q = _pick_row_value(qis, ["Operating Expense","OperatingExpenses","Total Operating Expenses","TotalOperatingExpenses"], agg="latest")
        rnd_q  = _pick_row_value(qis, ["Research Development","ResearchAndDevelopment","R&D"], agg="latest")
        sga_q  = _pick_row_value(qis, ["Selling General Administrative","SellingGeneralAndAdministrative","SG&A"], agg="latest")

        # TTM från annual/quarterly summer
        is_annual = getattr(t, "income_stmt", None)
        ebitda_ttm = 0.0
        int_ttm    = 0.0
        # försök TTM via quarterly
        if isinstance(qis, pd.DataFrame) and not qis.empty:
            ebitda_ttm = _pick_row_value(qis, ["Ebitda","EBITDA"], agg="ttm")
            int_ttm    = _pick_row_value(qis, ["Interest Expense","InterestExpense"], agg="ttm")
        # fallback: annual single (inte ttm men bättre än 0)
        if ebitda_ttm == 0.0 and isinstance(is_annual, pd.DataFrame) and not is_annual.empty:
            ebitda_ttm = _pick_row_value(is_annual, ["Ebitda","EBITDA"], agg="latest")
        if int_ttm == 0.0 and isinstance(is_annual, pd.DataFrame) and not is_annual.empty:
            int_ttm    = _pick_row_value(is_annual, ["Interest Expense","InterestExpense"], agg="latest")

        burn_q = max(0.0, -fcf_q)
        runway_q = (cash / burn_q) if burn_q > 0 else 0.0

        out.update({
            "Kassa": cash,
            "Total skuld": total_debt,
            "Operativt kassaflöde (Q)": ocf_q,
            "CapEx (Q)": capex_q,
            "Fritt kassaflöde (Q)": fcf_q,
            "Burn rate (Q)": burn_q,
            "Runway (kvartal)": runway_q,
            "Operating Expense (Q)": opex_q,
            "FoU (Q)": rnd_q,
            "SG&A (Q)": sga_q,
            "EBITDA (TTM)": ebitda_ttm,
            "Räntekostnad (TTM)": int_ttm,
            "Current assets": current_assets,
            "Current liabilities": current_liab,
        })
    except Exception:
        pass
    return out

# --------- Beräkningar --------------------------------------------------------
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp (>100%→50%, <0%→2%)
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

# --------- Auto-apply & auto-fetch -------------------------------------------
def apply_auto_updates_to_row(
    df: pd.DataFrame,
    row_idx: int,
    new_vals: dict,
    source: str,
    changes_map: dict,
    force_ts: bool = False
) -> bool:
    """
    Skriver fält som får ett nytt (positivt/meningsfullt) värde.
    Uppdaterar TS_ för spårade fält, sätter 'Senast auto-uppdaterad' + källa.
    force_ts=True: stämplar datum även om värdet är oförändrat.
    Returnerar True om något fält faktiskt ändrades.
    """
    changed_fields = []
    for f, v in new_vals.items():
        if f not in df.columns:
            continue
        old = df.at[row_idx, f] if f in df.columns else None

        # ska vi skriva?
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            write_ok = (float(v) > 0) or (f not in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"] and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            # andra typer: skriv inte
            write_ok = False

        if not write_ok and not force_ts:
            continue

        # skriv om värdet skiljer sig OCH write_ok
        if write_ok and ((pd.isna(old) and not pd.isna(v)) or (str(old) != str(v))):
            df.at[row_idx, f] = v
            changed_fields.append(f)

        # stampa TS även om värdet inte ändrats, om force_ts=True
        if f in TS_FIELDS and (write_ok or force_ts):
            _stamp_ts_for_field(df, row_idx, f)

    # allmän auto-stämpel alltid om force_ts, annars bara vid ändring
    if changed_fields or force_ts:
        _note_auto_update(df, row_idx, source)
        if changed_fields:
            changes_map.setdefault(df.at[row_idx, "Ticker"], []).extend(changed_fields)
        return bool(changed_fields)
    return False

# ====== Huvud-hämtare för EN ticker (kedja SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS) ======
def auto_fetch_for_ticker(ticker: str) -> Tuple[dict, dict]:
    """
    Pipeline:
      1) SEC + Yahoo (implied shares) eller Yahoo global fallback (P/S nu + Q1–Q4, MCap-historik, namn/valuta/price)
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S) om saknas
      4) Yahoo financial snapshots (CF/BS/IS)
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback) – definieras i Del 4/8
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","MCap (nu)","Sektor","_debug_shares_source",
            "MCap Q1","MCap Q2","MCap Q3","MCap Q4","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
            "P/S (Yahoo)"
        ]}
        # merga in allt men hoppa över tomma
        for k, v in base.items():
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["sec_yahoo_err"] = str(e)

    # 2) Finnhub estimat om saknas – definieras i Del 4/8
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

    # 3) FMP light P/S om saknas – definieras i Del 4/8
    try:
        if ("P/S" not in vals):
            fmpl = hamta_fmp_falt_light(ticker)
            debug["fmp_light"] = {"P/S": fmpl.get("P/S"), "Utestående aktier": fmpl.get("Utestående aktier")}
            v = fmpl.get("P/S")
            if v not in (None, "", 0, 0.0):
                vals["P/S"] = v
            if ("Utestående aktier" not in vals) and (fmpl.get("Utestående aktier") not in (None, "", 0, 0.0)):
                vals["Utestående aktier"] = fmpl["Utestående aktier"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # 4) Yahoo CF/BS/IS snapshot
    try:
        snap = yahoo_financial_snapshots(ticker)
        debug["yahoo_cf_bs"] = snap
        for k, v in snap.items():
            if v not in (None, "", 0, 0.0):
                vals[k] = v
    except Exception as e:
        debug["yahoo_cf_bs_err"] = str(e)

    # komplettera med Yahoo basfält för säkerhets skull (namn, sektor, beta, MCap, P/S Yahoo)
    try:
        yb = hamta_yahoo_fält(ticker)
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","Beta","P/S (Yahoo)","MCap (nu)","Sektor"]:
            v = yb.get(k)
            if v not in (None, "", 0, 0.0):
                vals.setdefault(k, v)
    except Exception as e:
        debug["yahoo_basic_err"] = str(e)

    return vals, debug

# app.py — Del 4/8
# --- Datakällor: SEC (US/FPI), Yahoo fallback, Finnhub & FMP -----------------

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
    # ~9–10MB JSON → cache 24h
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

# ---------- helpers: datum & shares ------------------------------------------
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
            return (d2 - d1).days <= 2
        except Exception:
            return False
    return False

def _collect_share_entries(facts: dict) -> list:
    """
    Hämtar alla 'instant' aktieposter från dei/us-gaap/ifrs-full.
    Returnerar [{"end": date, "val": float, "frame": "..."}].
    """
    entries = []
    facts_all = (facts.get("facts") or {})
    sources = [
        ("dei", ["EntityCommonStockSharesOutstanding", "EntityCommonSharesOutstanding"]),
        ("us-gaap", ["CommonStockSharesOutstanding", "ShareIssued"]),
        ("ifrs-full", ["NumberOfSharesIssued", "IssuedCapitalNumberOfShares", "OrdinarySharesNumber"]),
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
                            entries.append({"end": end, "val": v, "frame": it.get("frame") or "", "taxo": taxo})
                        except Exception:
                            pass
    return entries

def _sec_latest_shares_robust(facts: dict) -> float:
    """Summerar multi-class per senaste 'end' (instant)."""
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

# ---------- FX helper (för rev-unit till prisvaluta) --------------------------
@st.cache_data(show_spinner=False, ttl=21600)
def _fx_rate_cached(base: str, quote: str) -> float:
    base = (base or "").upper(); quote = (quote or "").upper()
    if not base or not quote or base == quote:
        return 1.0
    try:
        r = requests.get("https://api.frankfurter.app/latest", params={"from": base, "to": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v: return float(v)
    except Exception:
        pass
    try:
        r = requests.get("https://api.exchangerate.host/latest", params={"base": base, "symbols": quote}, timeout=12)
        if r.status_code == 200:
            v = (r.json() or {}).get("rates", {}).get(quote)
            if v: return float(v)
    except Exception:
        pass
    return 0.0

# ---------- SEC intäkter (kvartal) + Syntetisk Q4 -----------------------------
def _collect_duration_entries(fact: dict, unit_code: str) -> list[tuple]:
    """Plockar (start,end,val,form) för given unit_code."""
    arr = (fact.get("units") or {}).get(unit_code)
    out = []
    if not isinstance(arr, list):
        return out
    for it in arr:
        end = _parse_iso(str(it.get("end", "")))
        start = _parse_iso(str(it.get("start", "")))
        form = (it.get("form") or "").upper()
        val = it.get("val", None)
        if end and start and val is not None:
            try:
                out.append((start, end, float(val), form))
            except Exception:
                pass
    return out

def _sec_quarterly_revenues_dated_with_unit(facts: dict, max_quarters: int = 20):
    """
    Hämtar upp till 'max_quarters' kvartalsintäkter (3-mån) för US-GAAP (10-Q) och IFRS (6-K).
    Lagar 'syntetisk Q4' från 10-K/årsrapport när Q4 saknas.
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    """
    facts_all = (facts.get("facts") or {})
    taxo_keys = [("us-gaap", ("10-Q","10-Q/A","10-K","10-K/A")),
                 ("ifrs-full", ("6-K","6-K/A","10-Q","10-Q/A","20-F","40-F"))]

    rev_names = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "Revenues", "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]
    prefer_units = ("USD","CAD","EUR","GBP")

    # försök i ordning: us-gaap, ifrs-full
    for taxo, allowed_forms in taxo_keys:
        sect = facts_all.get(taxo, {})
        # hitta första revenue-concept som har någon av prefer_units
        for name in rev_names:
            fact = sect.get(name)
            if not fact:
                continue

            # plocka första unit som finns
            unit_code = None
            for u in prefer_units:
                if (fact.get("units") or {}).get(u):
                    unit_code = u; break
            if not unit_code:
                continue

            # duration-poster för denna unit
            dur = _collect_duration_entries(fact, unit_code)
            if not dur:
                continue

            # separera kvartal (≈70–100 dagar) vs annual (≈330–400 dagar)
            q_rows = []
            a_rows = []
            for (start, end, val, form) in dur:
                days = (end - start).days if (start and end) else None
                if not days:
                    continue
                if 70 <= days <= 100 and form in allowed_forms:
                    q_rows.append((end, val))
                elif 330 <= days <= 400:
                    a_rows.append((end, val))  # årsintäkt

            # deduplicera på end, behåll senaste
            def _dedup(rows):
                d = {}
                for end, v in rows:
                    d[end] = float(v)
                return [(k, d[k]) for k in d.keys()]

            q_rows = _dedup(q_rows)
            a_rows = _dedup(a_rows)

            # Bygg syntetisk Q4: om för ett års “end” saknas en kvartalspost med samma end
            # och det finns tre tidigare kvartal inom samma FY, så Q4 = Annual - sum(Q1..Q3).
            # För att hitta de tre: ta kvartals-poster med end <= annual_end och > annual_end - 365.
            if a_rows:
                # sortera asc för att lättare summera
                q_rows_sorted = sorted(q_rows, key=lambda x: x[0])
                a_rows_sorted = sorted(a_rows, key=lambda x: x[0])
                q_set = {d for (d, _) in q_rows}
                for a_end, a_val in a_rows_sorted:
                    # om redan finns kvartal med detta end → hoppa
                    if a_end in q_set:
                        continue
                    # samla tre föregående kvartal inom 365 dagar
                    three = []
                    for (qe, qv) in reversed(q_rows_sorted):
                        if qe < a_end and (a_end - qe).days <= 365:
                            three.append((qe, qv))
                            if len(three) == 3:
                                break
                    if len(three) == 3:
                        sum3 = sum(v for (_, v) in three)
                        q4 = a_val - sum3
                        # ibland kan avrundning ge negativt – ignorera orimliga värden
                        if q4 > 0:
                            q_rows.append((a_end, float(q4)))
                            q_set.add(a_end)

            if not q_rows:
                continue

            # sortera nyast→äldst, trimma
            q_rows = sorted(q_rows, key=lambda x: x[0], reverse=True)[:max_quarters]
            return q_rows, unit_code

    return [], None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20):
    rows, _ = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=max_quarters)
    return rows

# ---------- Yahoo pris & implied shares & TTM-fönster ------------------------
def _yahoo_prices_for_dates(ticker: str, dates: list) -> dict:
    """Hämtar dagliga priser och returnerar Close på eller närmast FÖRE respektive datum."""
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

def _implied_shares_from_yahoo(tkr: yf.Ticker, price: float = None, mcap: float = None) -> float:
    info = {}
    try:
        info = tkr.info or {}
    except Exception:
        info = {}
    if mcap is None:
        mcap = info.get("marketCap")
    if price is None:
        price = info.get("regularMarketPrice")
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

def hamta_yahoo_global_combo(ticker: str) -> dict:
    """
    Global fallback för tickers utan SEC.
    Räknar implied shares, P/S (TTM) nu, P/S Q1–Q4 historik + MCAP-historik.
    """
    out = {}
    t = yf.Ticker(ticker)

    # Bas: namn/valuta/price/mcap/ps(yahoo)/sektor/beta
    y = hamta_yahoo_fält(ticker)
    for k in ["Bolagsnamn","Valuta","Aktuell kurs","MCap (nu)","P/S (Yahoo)","Sektor","Beta"]:
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px = float(out.get("Aktuell kurs") or 0.0)
    px_ccy = (out.get("Valuta") or "USD").upper()

    info = _yfi_info_dict(t)
    mcap = float(info.get("marketCap") or out.get("MCap (nu)", 0.0) or 0.0)

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

    # Kvartalsintäkter → TTM (+ datum)
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows or len(q_rows) < 4:
        return out
    ttm_list = _ttm_windows(q_rows, need=4)

    # Financial currency vs prisvaluta
    fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        conv = _fx_rate_cached(fin_ccy, px_ccy) or 1.0
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # Market cap (nu)
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px
    if mcap > 0:
        out["MCap (nu)"] = mcap

    # P/S (TTM) nu + P/S Q1–Q4 + MCAP-historik
    if mcap > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap / ltm_now

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

# ---------- SEC + Yahoo kombination ------------------------------------------
def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US/FPIs: Shares + kvartalsintäkter från SEC (10-Q, 6-K + syntetisk Q4 via 10-K).
    Pris/valuta/namn/MCap/Sektor/Beta från Yahoo.
    P/S (TTM) nu + P/S Q1–Q4 historik och MCAP-historik.
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
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","MCap (nu)","Sektor","Beta","P/S (Yahoo)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
    px_ccy = (out.get("Valuta") or "USD").upper()
    px_now = float(out.get("Aktuell kurs") or 0.0)

    # Shares: implied → fallback SEC robust
    tkr_obj = yf.Ticker(ticker)
    implied = _implied_shares_from_yahoo(tkr_obj, price=px_now, mcap=out.get("MCap (nu)"))
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
    mcap_now = float(out.get("MCap (nu)", 0.0))
    if mcap_now <= 0 and px_now > 0 and shares_used > 0:
        mcap_now = shares_used * px_now
        out["MCap (nu)"] = mcap_now

    # SEC kvartalsintäkter + unit → TTM & konvertering (med syntetisk Q4)
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

    # P/S Q1–Q4 historik + MCAP Q1–Q4
    if shares_used > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev_px) in enumerate(ttm_list[:4], start=1):
            if ttm_rev_px and ttm_rev_px > 0:
                px_hist = px_map.get(d_end, None)
                if px_hist and px_hist > 0:
                    mcap_hist = shares_used * float(px_hist)
                    out[f"P/S Q{idx}"] = float(mcap_hist / ttm_rev_px)
                    out[f"MCap Q{idx}"] = float(mcap_hist)
                    out[f"MCap Datum Q{idx}"] = d_end.strftime("%Y-%m-%d")

    return out

# =============== Finnhub (valfritt, estimat) =================================
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets. Hämtar annual revenue estimates:
    current FY + next FY (om finns). Returnerar i "miljoner".
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

# =============== FMP (kompakt) ===============================================
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.0))
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20))

def _fmp_get(path: str, params=None):
    params = (params or {}).copy()
    if FMP_KEY:
        params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}/{path}"
    try:
        if FMP_CALL_DELAY > 0:
            time.sleep(FMP_CALL_DELAY)
        r = requests.get(url, params=params, timeout=20)
        if 200 <= r.status_code < 300:
            try:
                return r.json(), r.status_code
            except Exception:
                return None, r.status_code
        return None, r.status_code
    except Exception:
        return None, 0

def _fmp_pick_symbol(yahoo_ticker: str) -> str:
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

    # P/S TTM
    rttm, sc_rttm = _fmp_get(f"api/v3/ratios-ttm/{sym}")
    if isinstance(rttm, list) and rttm:
        v = rttm[0].get("priceToSalesTTM") or rttm[0].get("priceToSalesRatioTTM")
        try:
            if v and float(v) > 0:
                out["P/S"] = float(v)
        except Exception:
            pass

    # shares from quote
    q, sc_q = _fmp_get(f"api/v3/quote/{sym}")
    if isinstance(q, list) and q:
        q0 = q[0]
        if q0.get("sharesOutstanding") is not None:
            try: out["Utestående aktier"] = float(q0["sharesOutstanding"]) / 1e6
            except: pass
        if q0.get("price") is not None:
            try: out["Aktuell kurs"] = float(q0["price"])
            except: pass
    return out

# app.py — Del 5/8
# --- Snapshots, kontroll-listor, auto-uppdatering & Kontroll-vy --------------

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

# ---- TS-ålder & kontrolltabeller --------------------------------------------

def oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    """Returnerar äldsta TS bland alla TS_-kolumner för en rad (eller None)."""
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

# ---- Prognosvakt (Omsättning idag / nästa år) -------------------------------

def _min_forecast_ts(row: pd.Series):
    """Minsta (äldsta) TS av de två prognosfälten för en rad."""
    vals = []
    for col in ("TS_Omsättning idag", "TS_Omsättning nästa år"):
        if col in row and str(row[col]).strip():
            d = pd.to_datetime(str(row[col]).strip(), errors="coerce")
            if pd.notna(d):
                vals.append(d)
    if not vals:
        return pd.NaT
    return min(vals)

def build_forecast_needs_df(df: pd.DataFrame, older_than_days: int = 180) -> pd.DataFrame:
    """
    Listar bolag där prognosfälten (Omsättning idag / nästa år) saknar värden/TS
    eller har äldsta prognos-TS äldre än older_than_days. Sorterar äldst först.
    """
    out = []
    cutoff = now_dt() - timedelta(days=older_than_days)

    for _, r in df.iterrows():
        val_idag  = float(r.get("Omsättning idag", 0.0) or 0.0)
        val_next  = float(r.get("Omsättning nästa år", 0.0) or 0.0)

        ts_idag_s = str(r.get("TS_Omsättning idag", "") or "").strip()
        ts_next_s = str(r.get("TS_Omsättning nästa år", "") or "").strip()

        ts_idag = pd.to_datetime(ts_idag_s, errors="coerce") if ts_idag_s else pd.NaT
        ts_next = pd.to_datetime(ts_next_s, errors="coerce") if ts_next_s else pd.NaT

        oldest_ts = _min_forecast_ts(r)
        too_old = (pd.notna(oldest_ts) and oldest_ts.to_pydatetime() < cutoff)
        missing_val = (val_idag <= 0.0 or val_next <= 0.0)
        missing_ts = (not ts_idag_s or not ts_next_s)

        if missing_val or missing_ts or too_old:
            days_ago = (now_dt() - oldest_ts.to_pydatetime()).days if pd.notna(oldest_ts) else ""
            out.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Omsättning idag (M)": val_idag,
                "Omsättning nästa år (M)": val_next,
                "TS_Omsättning idag": ts_idag_s,
                "TS_Omsättning nästa år": ts_next_s,
                "Äldsta prognos-TS": oldest_ts.strftime("%Y-%m-%d") if pd.notna(oldest_ts) else "",
                "Dagar sedan prognos": days_ago,
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })

    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["_sort"] = pd.to_datetime(res["Äldsta prognos-TS"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    res = res.sort_values(by=["_sort","Bolagsnamn","Ticker"]).drop(columns=["_sort"])
    return res

# ---- Auto-uppdatering för hela tabellen -------------------------------------

def auto_update_all(df: pd.DataFrame, user_rates: dict, make_snapshot: bool = False, force_ts: bool = False):
    """
    Kör auto-uppdatering för alla rader. Skriver endast fält med meningsfulla värden.
    Stämplar TS_ per fält, samt 'Senast auto-uppdaterad' + källa.
    force_ts=True: stämplar TS även om värdet inte ändras (enligt önskemål).
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
                source="Auto (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)",
                changes_map=log["changed"],
                force_ts=force_ts
            )
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
        st.sidebar.success("Klart! Ändringar/TS sparade.")
    else:
        st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

    return df, log

# ---- Debug: en ticker --------------------------------------------------------

def debug_test_single_ticker(ticker: str):
    """Visar vad källorna levererar för en ticker, för felsökning."""
    st.markdown(f"### Testa datakällor för: **{ticker}**")
    cols = st.columns(2)

    with cols[0]:
        st.write("**SEC/Yahoo combo (inkl. Yahoo fallback + syntetisk Q4)**")
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

# ---- Kontroll-vy ------------------------------------------------------------

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
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30, key="ctrl_old_days")
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Prognosvakt (Omsättning idag / nästa år)
    st.subheader("🔮 Prognosvakt: Omsättning idag / nästa år")
    forecast_days = st.number_input(
        "Flagga om prognos-TS är äldre än (dagar)", min_value=30, max_value=2000, value=180, step=30, key="forecast_days"
    )
    fore = build_forecast_needs_df(df, older_than_days=int(forecast_days))
    if fore.empty:
        st.success("Alla prognosfält verkar nyligen uppdaterade 🤝")
    else:
        st.warning(f"{len(fore)} bolag behöver prognos-koll (saknar värden/TS eller äldre än {int(forecast_days)} dagar):")
        st.dataframe(
            fore[[
                "Ticker","Bolagsnamn",
                "Omsättning idag (M)","Omsättning nästa år (M)",
                "TS_Omsättning idag","TS_Omsättning nästa år",
                "Äldsta prognos-TS","Dagar sedan prognos",
                "Saknar värde?","Saknar TS?"
            ]],
            use_container_width=True, hide_index=True
        )
        csv_bytes = fore.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Ladda ner lista (CSV)", data=csv_bytes, file_name="prognosvakt.csv", mime="text/csv")

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

# app.py — Del 6/8
# --- Analys, Portfölj & Investeringsförslag ----------------------------------

# Hjälpare: formatera stora tal på svenska (bilj/mdr/milj)
def format_mcap_sv(v: float, ccy: str = "") -> str:
    try:
        x = float(v)
    except Exception:
        return "-"
    unit = ""
    val = x
    if x >= 1_000_000_000_000:      # 10^12
        val = x / 1_000_000_000_000
        unit = " bilj"
    elif x >= 1_000_000_000:        # 10^9
        val = x / 1_000_000_000
        unit = " mdr"
    elif x >= 1_000_000:            # 10^6
        val = x / 1_000_000
        unit = " milj"
    else:
        val = x
        unit = ""
    if ccy:
        return f"{val:,.2f}{unit} {ccy}".replace(",", " ").replace(".", ",")
    return f"{val:,.2f}{unit}".replace(",", " ").replace(".", ",")

# Hjälpare: risklabel baserat på mcap (USD-ekv., heuristik)
def _risklabel_from_mcap(mcap_native: float, ccy: str, user_rates: dict) -> str:
    try:
        m = float(mcap_native)
    except Exception:
        return "Okänd"
    # konvertera till USD-ekv via SEK som brygga om möjligt
    # m_native -> SEK -> USD
    rate_ccy = user_rates.get((ccy or "USD").upper(), None)
    rate_usd = user_rates.get("USD", None)
    if rate_ccy and rate_usd and rate_ccy > 0 and rate_usd > 0:
        m_usd = (m * rate_ccy) / rate_usd
    else:
        # fallback: anta redan USD
        m_usd = m

    if m_usd < 300_000_000:
        return "Micro"
    if m_usd < 2_000_000_000:
        return "Small"
    if m_usd < 10_000_000_000:
        return "Mid"
    if m_usd < 200_000_000_000:
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

    # välj bolag
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx, step=1
    )
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

    # etikett om senaste uppdateringstyp
    latest_auto = str(r.get("Senast auto-uppdaterad","")).strip()
    latest_man  = str(r.get("Senast manuellt uppdaterad","")).strip()
    if latest_auto and latest_man:
        st.caption(f"Senast auto: **{latest_auto}** • Senast manuell: **{latest_man}**")
    elif latest_auto:
        st.caption(f"Senast auto: **{latest_auto}**")
    elif latest_man:
        st.caption(f"Senast manuell: **{latest_man}**")

    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","MCap (nu)",
        "P/S","P/S (Yahoo)","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Sektor",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år",
        "MCap Q1","MCap Q2","MCap Q3","MCap Q4","MCap Datum Q1","MCap Datum Q2","MCap Datum Q3","MCap Datum Q4",
        "Kassa","Total skuld","Operativt kassaflöde (Q)","CapEx (Q)","Fritt kassaflöde (Q)","Burn rate (Q)","Runway (kvartal)",
        "Operating Expense (Q)","FoU (Q)","SG&A (Q)","EBITDA (TTM)","Räntekostnad (TTM)","Current assets","Current liabilities"
    ]
    cols = [c for c in cols if c in df.columns]
    view_row = pd.DataFrame([r[cols].to_dict()])
    # snygga till market cap i tabellen (extra kolumn)
    if "MCap (nu)" in view_row.columns:
        view_row.insert(view_row.columns.get_loc("MCap (nu)")+1,
                        "MCap (nu) (fmt)",
                        view_row["MCap (nu)"].apply(lambda v: format_mcap_sv(v, r.get("Valuta",""))))
    st.dataframe(view_row, use_container_width=True, hide_index=True)

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    # säkrare procent
    port["Andel (%)"] = np.where(total_värde > 0.0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)
    port["Andel (%)"] = np.round(port["Andel (%)"], 2)
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

    # --- nya filter: risklabel + sektor
    # för-beräkna risklabel
    df = df.copy()
    if "Risklabel" not in df.columns:
        df["Risklabel"] = df.apply(
            lambda r: _risklabel_from_mcap(float(r.get("MCap (nu)", 0.0)), str(r.get("Valuta","") or "USD"), user_rates),
            axis=1
        )

    # sektorlista
    sektorer = sorted([s for s in df["Sektor"].dropna().astype(str).unique() if s.strip()]) if "Sektor" in df.columns else []
    colf1, colf2 = st.columns([1,1])
    with colf1:
        risk_filter = st.multiselect("Riskklass", ["Micro","Small","Mid","Large","Mega"], default=["Micro","Small","Mid","Large","Mega"])
    with colf2:
        sektor_filter = st.multiselect("Sektor", sektorer, default=sektorer)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    # applicera filter
    if risk_filter:
        base = base[base["Risklabel"].isin(risk_filter)]
    if sektor_filter and "Sektor" in base.columns:
        base = base[base["Sektor"].isin(sektor_filter)]

    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # beräkna potential & diff
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # robust bläddring
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(max(0, st.session_state.forslags_index), len(base)-1)

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

    # portföljvärde för andelar
    port = df[df["Antal aktier"] > 0].copy()
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
    else:
        port_värde = 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        rsel = port[port["Ticker"].astype(str).str.upper().str.strip() == str(rad["Ticker"]).upper().strip()]
        if not rsel.empty:
            nuv_innehav = float(rsel["Värde (SEK)"].sum())

    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")

    # Huvudlista
    lines = [
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
        f"- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""),
        f"- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""),
        f"- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""),
        f"- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}" + (" **⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""),
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
        f"- **Riskklass:** {rad.get('Risklabel','Okänd')}",
        f"- **Sektor:** {rad.get('Sektor','')}",
    ]
    st.markdown("\n".join(lines))

    # Expander med fler nyckeltal
    with st.expander("🔎 Detaljer & nyckeltal"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Värdering**")
            st.write("• Nuvarande P/S:", round(float(rad.get("P/S",0.0)), 2))
            st.write("• P/S (Yahoo):", round(float(rad.get("P/S (Yahoo)",0.0)), 2))
            st.write("• P/S-snitt (Q1–Q4):", round(float(rad.get("P/S-snitt",0.0)), 2))
            mcap_fmt = format_mcap_sv(float(rad.get("MCap (nu)",0.0)), str(rad.get("Valuta","")))
            st.write("• Nuvarande mcap:", mcap_fmt)

        with c2:
            st.markdown("**Kassa, skuld & kassaflöde**")
            st.write("• Kassa:", format_mcap_sv(float(rad.get("Kassa",0.0)), str(rad.get("Valuta",""))))
            st.write("• Total skuld:", format_mcap_sv(float(rad.get("Total skuld",0.0)), str(rad.get("Valuta",""))))
            st.write("• OCF (Q):", format_mcap_sv(float(rad.get("Operativt kassaflöde (Q)",0.0)), str(rad.get("Valuta",""))))
            st.write("• CapEx (Q):", format_mcap_sv(float(rad.get("CapEx (Q)",0.0)), str(rad.get("Valuta",""))))
            st.write("• FCF (Q):", format_mcap_sv(float(rad.get("Fritt kassaflöde (Q)",0.0)), str(rad.get("Valuta",""))))
            st.write("• Burn (Q):", format_mcap_sv(float(rad.get("Burn rate (Q)",0.0)), str(rad.get("Valuta",""))))
            st.write("• Runway (kvartal):", round(float(rad.get("Runway (kvartal)",0.0)), 1))

        with c3:
            st.markdown("**Kostnader & TTM**")
            st.write("• Opex (Q):", format_mcap_sv(float(rad.get("Operating Expense (Q)",0.0)), str(rad.get("Valuta",""))))
            st.write("• FoU (Q):", format_mcap_sv(float(rad.get("FoU (Q)",0.0)), str(rad.get("Valuta",""))))
            st.write("• SG&A (Q):", format_mcap_sv(float(rad.get("SG&A (Q)",0.0)), str(rad.get("Valuta",""))))
            st.write("• EBITDA (TTM):", format_mcap_sv(float(rad.get("EBITDA (TTM)",0.0)), str(rad.get("Valuta",""))))
            st.write("• Räntekostnad (TTM):", format_mcap_sv(float(rad.get("Räntekostnad (TTM)",0.0)), str(rad.get("Valuta",""))))

        st.markdown("---")
        st.markdown("**MCAP-historik (senaste 4 TTM-punkter)**")
        mcols = st.columns(4)
        for i in range(1,5):
            with mcols[i-1]:
                d = str(rad.get(f"MCap Datum Q{i}","") or "")
                v = float(rad.get(f"MCap Q{i}", 0.0) or 0.0)
                st.write(f"Q{i} ({d})")
                st.write(format_mcap_sv(v, str(rad.get("Valuta",""))))

# app.py — Del 7/8
# --- Batch-vy & Lägg till / uppdatera bolag ----------------------------------

def _price_only_for_ticker(ticker: str) -> dict:
    """Hämtar bara aktuell kurs (+ev valuta/namn/mcap/beta/sektor) från Yahoo."""
    out = {}
    try:
        y = hamta_yahoo_fält(ticker)
        for k in ["Aktuell kurs","Valuta","Bolagsnamn","MCap (nu)","Beta","Sektor","P/S (Yahoo)"]:
            v = y.get(k)
            if v not in (None, "", 0, 0.0):
                out[k] = v
    except Exception:
        pass
    return out

def batchvy(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Kör delmängder av tabellen i batch, med valbar sortering och batch-storlek."""
    st.header("🧵 Batch-körning")

    # välj sortering
    sort_mode = st.selectbox("Ordning", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], index=0, key="batch_sort_mode")
    batch_size = st.number_input("Hur många per körning?", min_value=1, max_value=200, value=10, step=1, key="batch_size")
    force_ts = st.checkbox("Tidsstämpla även oförändrade fält", value=True, key="batch_force_ts")
    snapshot_before = st.checkbox("Skapa snapshot före skrivning", value=False, key="batch_snapshot")

    # Bygg ordnad lista
    if sort_mode.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"]).reset_index(drop=False)
        order = list(work["index"].values)
    else:
        work = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=False)
        order = list(work["index"].values)

    # init session-state för batch
    key_prefix = f"batch_{'oldest' if sort_mode.startswith('Äldst') else 'az'}"
    if f"{key_prefix}_pos" not in st.session_state:
        st.session_state[f"{key_prefix}_pos"] = 0

    # Visa nästa fönster av tickers
    start = st.session_state[f"{key_prefix}_pos"]
    stop = min(start + int(batch_size), len(order))
    window_idxs = order[start:stop]
    win_df = df.loc[window_idxs].copy()
    st.caption(f"Förhandsvisar {start+1}–{stop} av {len(order)}")

    if not win_df.empty:
        st.dataframe(win_df[["Ticker","Bolagsnamn","Senast auto-uppdaterad","Senast manuellt uppdaterad"]], use_container_width=True, hide_index=True)
    else:
        st.info("Inga fler poster i denna batch-sekvens.")

    col_run, col_skip, col_reset = st.columns([1,1,1])
    run_clicked = col_run.button("🚀 Kör nästa batch")
    skip_clicked = col_skip.button("⏭️ Hoppa över denna batch")
    reset_clicked = col_reset.button("🔁 Återställ position")

    if reset_clicked:
        st.session_state[f"{key_prefix}_pos"] = 0
        st.info("Batch-position återställd till början.")

    changed_any = False
    log_local = {"changed": {}, "misses": {}}

    if run_clicked and window_idxs:
        progress = st.progress(0.0)
        status = st.empty()
        for j, ridx in enumerate(window_idxs, start=1):
            tkr = str(df.at[ridx, "Ticker"]).strip().upper()
            status.write(f"Uppdaterar: {tkr} ({j}/{len(window_idxs)})")
            try:
                new_vals, debug = auto_fetch_for_ticker(tkr)
                changed = apply_auto_updates_to_row(
                    df, ridx, new_vals,
                    source="Batch (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)",
                    changes_map=log_local["changed"],
                    force_ts=force_ts
                )
                if not changed:
                    log_local["misses"][tkr] = list(new_vals.keys()) if new_vals else ["(inga nya fält)"]
                changed_any = changed_any or changed
            except Exception as e:
                log_local["misses"][tkr] = [f"error: {e}"]
            progress.progress(j/len(window_idxs))

        df = uppdatera_berakningar(df, user_rates)

        # skriv om något ändrats eller force_ts
        if changed_any or force_ts:
            spara_data(df, do_snapshot=snapshot_before)
            st.success("Batch sparad till Google Sheets.")
        else:
            st.info("Inga ändringar – ingen skrivning.")

        # bumpa position
        st.session_state[f"{key_prefix}_pos"] = stop

        # visa lätt logg
        with st.expander("Visa batch-körlogg"):
            st.json(log_local)

    if skip_clicked:
        st.session_state[f"{key_prefix}_pos"] = stop
        st.info(f"Hoppade fram till post {stop+1}.")

    return df

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # sort- och lista
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], key="edit_sort")
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    # bläddringsindex i sorted listan
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    # välj via selectbox
    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    # knappar för bläddring
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef_mask = df["Ticker"].astype(str).str.upper().str.strip() == str(namn_map[valt_label]).upper().strip()
        if bef_mask.any():
            bef = df[bef_mask].iloc[0]
            ridx_existing = df.index[bef_mask][0]
        else:
            bef = pd.Series({}, dtype=object)
            ridx_existing = None
    else:
        bef = pd.Series({}, dtype=object)
        ridx_existing = None

    # snabb etikett om senaste uppdateringar
    if not bef.empty:
        latest_auto = str(bef.get("Senast auto-uppdaterad","")).strip()
        latest_man  = str(bef.get("Senast manuellt uppdaterad","")).strip()
        if latest_auto or latest_man:
            st.caption("Senaste uppdateringar: " +
                       (f"Auto **{latest_auto}**" if latest_auto else "") +
                       (" • " if (latest_auto and latest_man) else "") +
                       (f"Manuell **{latest_man}**" if latest_man else ""))

    # formulär
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
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%), Beta, P/S (Yahoo), MCap (nu), Sektor (Yahoo)")
            st.write("- Riktkurser/beräkningar räknas om")

        spar = st.form_submit_button("💾 Spara")

    # knappar under formuläret för denna ticker
    colx1, colx2, colx3 = st.columns([1,1,1])
    do_price = colx1.button("💱 Uppdatera bara kurs", disabled=(not ticker))
    do_full  = colx2.button("🔄 Full auto för denna", disabled=(not ticker))
    do_test  = colx3.button("🧪 Visa källor (debug)", disabled=(not ticker))

    # Spara formulärvärden
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

        # Skriv in nya fält i DF
        if not bef.empty:
            ridx = ridx_existing
            for k,v in ny.items():
                df.at[ridx, k] = v
        else:
            tom = {c: (0.0 if c in NUMERIC_COLS else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            ridx = df.index[df["Ticker"].astype(str).str.upper().str.strip() == ticker.upper().strip()][0]

        # Sätt manuell TS + TS_ per fält
        if datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.at[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.at[ridx, "Valuta"] = data["Valuta"]
        for fld in ["Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","Beta","P/S (Yahoo)","MCap (nu)","Sektor"]:
            if fld in data and data.get(fld) is not None:
                df.at[ridx, fld] = float(data.get(fld)) if fld not in ("Bolagsnamn","Valuta","Sektor") else data.get(fld)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # Kurs-uppdatering endast
    if do_price and ticker:
        idx = _find_row_idx_by_ticker(df, ticker)
        if idx is None:
            st.warning(f"Kunde inte hitta **{ticker}** i tabellen.")
        else:
            vals = _price_only_for_ticker(ticker)
            if vals:
                changed = apply_auto_updates_to_row(
                    df, idx, vals, source="Kurs (Yahoo)", changes_map={}, force_ts=True
                )
                df = uppdatera_berakningar(df, user_rates)
                spara_data(df)
                st.success("Kurs uppdaterad.")
            else:
                st.info("Kunde inte hämta kurs.")

    # Full auto för en ticker
    if do_full and ticker:
        idx = _find_row_idx_by_ticker(df, ticker)
        if idx is None:
            st.warning(f"**{ticker}** hittades inte i tabellen.")
        else:
            vals, debug = auto_fetch_for_ticker(ticker)
            changed = apply_auto_updates_to_row(
                df, idx, vals, source="Manuell full auto (SEC/Yahoo→Finnhub→FMP→Yahoo CF/BS)",
                changes_map={}, force_ts=True
            )
            df = uppdatera_berakningar(df, user_rates)
            spara_data(df)
            st.success("Full auto körd och sparad.")

    # Debug-källor
    if do_test and ticker:
        debug_test_single_ticker(ticker)

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

# app.py — Del 8/8
# --- Hjälpare + MAIN ----------------------------------------------------------

# Kolumner som vi betraktar som numeriska (för tom-rad/init)
NUMERIC_COLS = set([
    "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt",
    "MCap (nu)","P/S (Yahoo)","Beta","MCap Q1","MCap Q2","MCap Q3","MCap Q4",
    "Kassa","Total skuld","Operativt kassaflöde (Q)","CapEx (Q)","Fritt kassaflöde (Q)","Burn rate (Q)","Runway (kvartal)",
    "Operating Expense (Q)","FoU (Q)","SG&A (Q)","EBITDA (TTM)","Räntekostnad (TTM)","Current assets","Current liabilities"
])

def _find_row_idx_by_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    """Returnerar DF-index för ticker (case/trim-säkrad), annars None."""
    if "Ticker" not in df.columns:
        return None
    mask = df["Ticker"].astype(str).str.upper().str.strip() == str(ticker).upper().strip()
    if not mask.any():
        return None
    return int(df.index[mask][0])

def update_all_prices_fast(df: pd.DataFrame, user_rates: dict, snapshot_before: bool = False):
    """
    Snabb uppdatering: endast 'Aktuell kurs' (+ ev Valuta/Bolagsnamn/MCap/Beta/Sektor från Yahoo) för alla tickers.
    Stämplar TS även om värde oförändrat (force_ts=True).
    """
    st.sidebar.info("Startar snabb kursuppdatering…")
    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()

    total = len(df)
    any_changed = False
    misses = {}

    for i, r in df.reset_index().iterrows():
        idx = r["index"]
        tkr = str(r["Ticker"]).strip().upper()
        if not tkr:
            progress.progress((i+1)/max(total,1)); continue
        status.write(f"Uppdaterar kurs {i+1}/{total}: {tkr}")
        try:
            vals = _price_only_for_ticker(tkr)
            if vals:
                ch = apply_auto_updates_to_row(
                    df, idx, vals, source="Snabb kurs (Yahoo)", changes_map={}, force_ts=True
                )
                any_changed = any_changed or ch
            else:
                misses[tkr] = ["(ingen kurs hittades)"]
        except Exception as e:
            misses[tkr] = [f"error: {e}"]
        progress.progress((i+1)/max(total,1))

    df = uppdatera_berakningar(df, user_rates)
    if any_changed:
        spara_data(df, do_snapshot=snapshot_before)
        st.sidebar.success("Klart! Kurser uppdaterade och sparade.")
    else:
        st.sidebar.info("Inga kursändringar (TS ändå stämplade).")
    if misses:
        with st.sidebar.expander("Misslyckade tickers"):
            st.json(misses)
    return df

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- SIDOPANEL: Valutakurser till SEK ------------------------------------
    st.sidebar.header("💱 Valutakurser → SEK")

    # init session_state för växelkurser
    if "rate_usd" not in st.session_state or not isinstance(st.session_state.get("rate_usd"), (int,float)):
        saved_rates_boot = las_sparade_valutakurser()
        st.session_state.rate_usd = float(saved_rates_boot.get("USD", STANDARD_VALUTAKURSER["USD"]))
        st.session_state.rate_nok = float(saved_rates_boot.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        st.session_state.rate_cad = float(saved_rates_boot.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        st.session_state.rate_eur = float(saved_rates_boot.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

    usd = st.sidebar.number_input("USD → SEK", value=float(st.session_state.rate_usd), step=0.01, format="%.4f", key="rate_usd")
    nok = st.sidebar.number_input("NOK → SEK", value=float(st.session_state.rate_nok), step=0.01, format="%.4f", key="rate_nok")
    cad = st.sidebar.number_input("CAD → SEK", value=float(st.session_state.rate_cad), step=0.01, format="%.4f", key="rate_cad")
    eur = st.sidebar.number_input("EUR → SEK", value=float(st.session_state.rate_eur), step=0.01, format="%.4f", key="rate_eur")

    col_rates_btn1, col_rates_btn2 = st.sidebar.columns(2)
    with col_rates_btn1:
        if st.button("🌐 Hämta kurser automatiskt"):
            auto_rates, misses, provider = hamta_valutakurser_auto()
            try:
                st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
                st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
                st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
                st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
                st.sidebar.success(f"Valutakurser hämtade (källa: {provider}).")
                if misses:
                    st.sidebar.warning("Missar:\n- " + "\n- ".join(misses))
                st.rerun()  # visa direkt i inputs
            except Exception as e:
                st.sidebar.error(f"Kunde inte uppdatera sidopanelens fält: {e}")
    with col_rates_btn2:
        if st.button("💾 Spara kurser"):
            to_save = {
                "USD": float(st.session_state.rate_usd),
                "NOK": float(st.session_state.rate_nok),
                "CAD": float(st.session_state.rate_cad),
                "EUR": float(st.session_state.rate_eur),
                "SEK": 1.0
            }
            spara_valutakurser(to_save)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")

    if st.sidebar.button("↻ Läs sparade kurser"):
        st.cache_data.clear()
        try:
            saved = las_sparade_valutakurser()
            st.session_state.rate_usd = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
            st.session_state.rate_nok = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
            st.session_state.rate_cad = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
            st.session_state.rate_eur = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
        except Exception as e:
            st.sidebar.error(f"Fel vid läsning: {e}")
        st.rerun()

    user_rates = {
        "USD": float(st.session_state.rate_usd),
        "NOK": float(st.session_state.rate_nok),
        "CAD": float(st.session_state.rate_cad),
        "EUR": float(st.session_state.rate_eur),
        "SEK": 1.0
    }

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # --- LÄS DATA -------------------------------------------------------------
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = säkerställ_kolumner(df)
        spara_data(df)

    # Säkerställ schema, migrera och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Snabba uppdateringsknappar i sidopanelen -----------------------------
    st.sidebar.subheader("⚙️ Uppdateringar")
    make_snapshot = st.sidebar.checkbox("Snapshot före skrivning", value=False, key="global_snapshot")
    force_ts_all  = st.sidebar.checkbox("Tidsstämpla även oförändrade fält", value=True, key="global_force_ts")

    if st.sidebar.button("💱 Uppdatera alla kurser (snabb)"):
        df = update_all_prices_fast(df, user_rates, snapshot_before=make_snapshot)

    if st.sidebar.button("🔄 Auto-uppdatera alla (full, tung)"):
        df, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot, force_ts=force_ts_all)
        st.session_state["last_auto_log"] = log

    # --- Meny -----------------------------------------------------------------
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj","Batch"],
        index=0
    )

    if meny == "Kontroll":
        kontrollvy(df)
    elif meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # om df uppdaterats, ersätt och visa success i sidopanel
        if not df2.equals(df):
            df = df2
            st.sidebar.success("Ändringar sparade.")
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
            st.sidebar.success("Batchändringar sparade.")

if __name__ == "__main__":
    main()
