# app.py — Del 1/?
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
from typing import Optional, Tuple

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

# =========================
#   Google Sheets-koppling
# =========================
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

# ===============================
#   Valutakurser (manuell/auto)
# ===============================
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

def hamta_valutakurser_auto():
    """Automatisk FX: FMP -> Frankfurter -> exchangerate.host; fyller luckor från sparat/standard."""
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

# =========================
#   Kolumnschema & TS-fält
# =========================

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

    # Nya fält för analyser/rankning/inkomst
    "Momentum 6m (%)", "Momentum 12m (%)",
    "ADV (SEK)", "MarketCap (SEK)",
    "FCF (TTM)",                 # om du redan fyller detta – annars lämnas 0/NaN
    "FCF payout (%)",
    "Dividend safety (x)",
    "Dividend-säkerhet",
    "Add-score (Growth)", "Add-score (Dividend)",
    "EV/S (TTM)", "EV/EBITDA (TTM)", "P/E (TTM)", "FCF-yield (%)",
    "Rule of 40", "Quality-score", "Köp-score",
    "Sektor", "Industri",

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
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","momentum","adv","marketcap","ev/","p/e","yield","rule","score","fcf"]):
                df[kol] = 0.0
            elif kol.startswith("TS_"):
                df[kol] = ""  # tidsstämplar
            elif kol in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sektor","Industri","Dividend-säkerhet"):
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
        "Momentum 6m (%)","Momentum 12m (%)","ADV (SEK)","MarketCap (SEK)",
        "FCF (TTM)","FCF payout (%)","Dividend safety (x)",
        "EV/S (TTM)","EV/EBITDA (TTM)","P/E (TTM)","FCF-yield (%)","Rule of 40","Quality-score","Köp-score",
        "Add-score (Growth)","Add-score (Dividend)"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Sektor","Industri",
              "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Dividend-säkerhet"]:
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
    """Sätter auto-uppdaterad-tidsstämpel och källa (stämplas alltid, även om värde är samma)."""
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

# === (Fortsättning följer i nästa del) ===

# app.py — Del 2/?
# --- Hjälpare & beräkningar (P/S, riktkurser, KPI, utdelningslogik) ---------

# ---------- Skala/format-hjälpare -------------------------------------------

def _scale_0_100(x: float, lo: float, hi: float, invert: bool = False) -> float:
    """Skalar x till 0..100 mellan [lo, hi]. Clampas utanför intervallet. invert=True → 100 bra vid lågt värde."""
    try:
        xv = float(x)
        if hi == lo:
            return 50.0
        t = (xv - lo) / (hi - lo)
        t = 0.0 if t < 0 else (1.0 if t > 1 else t)
        s = (1.0 - t) if invert else t
        return round(100.0 * s, 1)
    except Exception:
        return 50.0

def _fmt_large(n: float) -> str:
    """Kompakt numeriskt format (ingen valuta), T/B/M/K."""
    try:
        v = float(n)
    except Exception:
        return str(n)
    absv = abs(v)
    if absv >= 1_000_000_000_000:
        return f"{v/1_000_000_000_000:.2f}T"
    if absv >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if absv >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if absv >= 1_000:
        return f"{v/1_000:.2f}K"
    return f"{v:.0f}"

def _fmt_sek(x: float) -> str:
    try:
        return f"{x:,.0f} SEK".replace(",", " ")
    except Exception:
        return f"{x} SEK"

def _fmt_pct(x: float) -> str:
    try:
        return f"{x:.2f}%"
    except Exception:
        return f"{x}%"

# ---------- ADV (SEK) 20d ----------------------------------------------------

def _adv_sek_20d(ticker: str, fx_rate: float) -> float:
    """
    Grov ADV i SEK: genomsnitt av (Close*Volume) senaste ~20 handelsdagar (hämtar ~30 dagar och tar min(20, antal)).
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="45d", interval="1d")  # 45d för säkerhetsmarginal
        if hist is None or hist.empty:
            return 0.0
        hist = hist.dropna(subset=["Close","Volume"])
        if hist.empty:
            return 0.0
        # Ta de senaste 20 trading-dagarna
        tail = hist.tail(20)
        sek_val = (tail["Close"] * tail["Volume"] * float(fx_rate)).mean()
        return float(sek_val) if pd.notna(sek_val) else 0.0
    except Exception:
        return 0.0

# ---------- Yahoo-hjälpare ----------------------------------------------------

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
    Basfält från Yahoo: Bolagsnamn, Kurs, Valuta, Utdelning, CAGR, Sektor, Industri.
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

        # sektor / industri
        if info.get("sector"):
            out["Sektor"] = str(info.get("sector"))
        if info.get("industry"):
            out["Industri"] = str(info.get("industry"))

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---------- Dividend helpers --------------------------------------------------

def _dividend_yield_pct(row: pd.Series) -> float:
    """Dividend yield (%) = Årlig utdelning / Aktuell kurs * 100."""
    try:
        div_ps = float(row.get("Årlig utdelning", 0.0) or 0.0)
        px = float(row.get("Aktuell kurs", 0.0) or 0.0)
        if div_ps > 0 and px > 0:
            return (div_ps / px) * 100.0
    except Exception:
        pass
    return float("nan")

def _position_underweight_boost(cur_pct: float, target_pct: float) -> float:
    """
    Underweight-boost 0..100: större boost ju mer under target.
    0 → target eller över. 100 → extremt under target (>= target*3 under)
    """
    try:
        cur = float(cur_pct); tgt = float(target_pct)
        if tgt <= 0:
            return 0.0
        gap = max(0.0, tgt - cur)  # hur mycket under
        # Skala gap 0..(3*tgt) → 0..100
        return _scale_0_100(gap, 0.0, (3.0 * tgt), invert=False)
    except Exception:
        return 0.0

def _estimate_dividends_paid(row: pd.Series) -> float:
    """
    Grov uppskattning av utdelningar kontant i bolagets valuta:
    Årlig utdelning per aktie * Utestående aktier * 1e6.
    """
    try:
        div_ps = float(row.get("Årlig utdelning", 0.0) or 0.0)
        sh_mn  = float(row.get("Utestående aktier", 0.0) or 0.0)  # milj.
        if div_ps > 0 and sh_mn > 0:
            return div_ps * sh_mn * 1_000_000.0
    except Exception:
        pass
    return 0.0

def _sector_mode_for_dividend(row: pd.Series) -> str:
    """
    Klassar logik för payout-bedömning: 'REIT', 'BANK', eller 'DEFAULT'.
    Baseras på Sektor/Industri-text.
    """
    s = (row.get("Sektor") or "") .lower()
    ind = (row.get("Industri") or "") .lower()
    txt = f"{s} {ind}"
    if "reit" in txt or "real estate" in txt or "property" in txt:
        return "REIT"
    if "bank" in txt or ("financial" in txt and ("bank" in txt or "lender" in txt)):
        return "BANK"
    return "DEFAULT"

def _fcf_payout_pct(row: pd.Series) -> float:
    """
    FCF payout (%) = utdelningar / FCF(TTM) * 100. Faller tillbaka till NaN om FCF ≤ 0.
    """
    try:
        fcf = float(row.get("FCF (TTM)", 0.0) or 0.0)  # i bolagets valuta
        if fcf <= 0:
            return float("nan")
        div_paid = _estimate_dividends_paid(row)
        if div_paid <= 0:
            return float("nan")
        return (div_paid / fcf) * 100.0
    except Exception:
        return float("nan")

def _dividend_safety_x(row: pd.Series) -> float:
    """
    Dividend safety (x) = FCF(TTM) / utdelningar (≈ täckningsgrad).
    """
    try:
        fcf = float(row.get("FCF (TTM)", 0.0) or 0.0)
        div_paid = _estimate_dividends_paid(row)
        if fcf > 0 and div_paid > 0:
            return fcf / div_paid
    except Exception:
        pass
    return float("nan")

def _dividend_badge(row: pd.Series, cfg: dict) -> tuple[str, str, str]:
    """
    Returnerar (badge_text, färg, motiv) enligt sektor-mode och trösklar i cfg.
    cfg nycklar:
      default_max_payout, default_min_cover,
      reit_max_payout,    reit_min_cover,
      bank_max_payout,    bank_min_cover
    """
    mode = _sector_mode_for_dividend(row)
    payout = _fcf_payout_pct(row)
    cover  = _dividend_safety_x(row)

    # Välj trösklar
    if mode == "REIT":
        max_p = cfg.get("reit_max_payout", 85.0)
        min_c = cfg.get("reit_min_cover", 1.10)
    elif mode == "BANK":
        max_p = cfg.get("bank_max_payout", 60.0)
        min_c = cfg.get("bank_min_cover", 1.30)
    else:
        max_p = cfg.get("default_max_payout", 80.0)
        min_c = cfg.get("default_min_cover", 1.20)

    # Bedömning
    badge, color, why = "OK", "orange", ""
    if pd.notna(payout) and pd.notna(cover):
        if cover >= (min_c + 0.3) and payout <= (max_p - 10):
            badge, color = "Stark", "green"
        elif cover >= min_c and payout <= max_p:
            badge, color = "OK", "orange"
        else:
            badge, color = "Risk", "red"
        why = f"payout {payout:.0f}% vs max {max_p:.0f}%, täckning {cover:.2f}× vs min {min_c:.2f}×"
    elif pd.notna(cover):
        if cover >= (min_c + 0.3):
            badge, color = "Stark", "green"
        elif cover >= min_c:
            badge, color = "OK", "orange"
        else:
            badge, color = "Risk", "red"
        why = f"täckning {cover:.2f}× vs min {min_c:.2f}× (payout saknas)"
    else:
        badge, color = "Okänt", "gray"
        why = "saknar FCF/utdelningsdata"

    return badge, color, why

def _classify_style(row: pd.Series, yield_min: float = 2.0, sector_override: bool = True) -> str:
    """
    'Dividend' om Årlig utdelning > 0 och yield >= yield_min.
    Med sector_override=True räknas REIT/Utilities/Telecom som Dividend om de har utdelning > 0,
    även vid lägre yield.
    """
    try:
        div_ps = float(row.get("Årlig utdelning", 0.0) or 0.0)
        px     = float(row.get("Aktuell kurs", 0.0) or 0.0)
        if div_ps <= 0 or px <= 0:
            return "Growth"
        y = (div_ps / px) * 100.0
        if y >= yield_min:
            return "Dividend"
        if sector_override:
            s = (row.get("Sektor") or "").lower()
            ind = (row.get("Industri") or "").lower()
            txt = f"{s} {ind}"
            if ("reit" in txt) or ("utilities" in txt) or ("utility" in txt) or ("telecom" in txt) or ("telecommunications" in txt):
                return "Dividend"
    except Exception:
        pass
    return "Growth"

def _valuation_score(row: pd.Series) -> float:
    """
    Sammanvägd värderingsscore 0..100 (högre = billigare/bättre) från EV/S, EV/EBITDA, P/E.
    Skalar varje nyckeltal till 0..100 och inverterar (lägre multipel = bättre).
    """
    parts = []
    try:
        evs = row.get("EV/S (TTM)")
        if evs is not None and not pd.isna(evs) and float(evs) > 0:
            parts.append(_scale_0_100(float(evs), 1.0, 20.0, invert=True))
    except Exception: pass
    try:
        eve = row.get("EV/EBITDA (TTM)")
        if eve is not None and not pd.isna(eve) and float(eve) > 0:
            parts.append(_scale_0_100(float(eve), 5.0, 40.0, invert=True))
    except Exception: pass
    try:
        pe = row.get("P/E (TTM)")
        if pe is not None and not pd.isna(pe) and float(pe) > 0:
            parts.append(_scale_0_100(float(pe), 8.0, 45.0, invert=True))
    except Exception: pass

    if parts:
        return round(float(np.mean(parts)), 1)
    return float("nan")

def _ensure_adv_and_mcap_sek(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Fyll 'ADV (SEK)' (20d) och 'MarketCap (SEK)' om saknas."""
    if df.empty: return df
    df = df.copy()
    if "ADV (SEK)" not in df.columns or df["ADV (SEK)"].isna().all():
        adv_vals = []
        for _, r in df.iterrows():
            fx = hamta_valutakurs(r.get("Valuta","SEK"), user_rates)
            adv_vals.append(_adv_sek_20d(str(r.get("Ticker","")), fx))
        df["ADV (SEK)"] = adv_vals
    if "MarketCap (SEK)" not in df.columns or df["MarketCap (SEK)"].isna().all():
        mcs = []
        for _, r in df.iterrows():
            try:
                px = float(r.get("Aktuell kurs", 0.0) or 0.0)
                sh_mn = float(r.get("Utestående aktier", 0.0) or 0.0)  # miljoner
                fx = hamta_valutakurs(r.get("Valuta","SEK"), user_rates)
                mc = px * sh_mn * 1_000_000.0 * fx
                mcs.append(round(mc, 2))
            except Exception:
                mcs.append(None)
        df["MarketCap (SEK)"] = mcs
    return df

# ---------- CAGR/omsättning/riktkurs-beräkningar -----------------------------

def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Beräknar:
      - P/S-snitt som snitt av positiva Q1–Q4
      - Omsättning 2 & 3 år från 'Omsättning nästa år' med CAGR clamp
      - Riktkurser idag/1/2/3 beroende på P/S-snitt och Utestående aktier
    """
    df = df.copy()
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
        aktier_ut_mn = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut_mn > 0 and ps_snitt > 0:
            shares = aktier_ut_mn * 1_000_000.0
            def _rk(rev_mn):
                try:
                    rev = float(rev_mn) * 1_000_000.0
                    return round((rev * ps_snitt) / shares, 2)
                except Exception:
                    return 0.0
            df.at[i, "Riktkurs idag"]    = _rk(rad.get("Omsättning idag", 0.0))
            df.at[i, "Riktkurs om 1 år"] = _rk(rad.get("Omsättning nästa år", 0.0))
            df.at[i, "Riktkurs om 2 år"] = _rk(df.at[i, "Omsättning om 2 år"])
            df.at[i, "Riktkurs om 3 år"] = _rk(df.at[i, "Omsättning om 3 år"])
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# ---------- Utdelnings-säkerhet (beräkna & spara kolumner) -------------------

def berakna_utdelningssakerhet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    # Standardtrösklar (kan UI skriva över i vyer)
    cfg_default = {
        "default_max_payout": 80.0, "default_min_cover": 1.20,
        "reit_max_payout":    85.0, "reit_min_cover":    1.10,
        "bank_max_payout":    60.0, "bank_min_cover":    1.30,
    }

    for i, r in df.iterrows():
        # FCF payout
        p = _fcf_payout_pct(r)
        if pd.notna(p):
            df.at[i, "FCF payout (%)"] = round(p, 1)

        # Dividend safety (x)
        c = _dividend_safety_x(r)
        if pd.notna(c):
            df.at[i, "Dividend safety (x)"] = round(c, 2)

        # Badge
        badge, _, _ = _dividend_badge(r, cfg_default)
        df.at[i, "Dividend-säkerhet"] = badge

    return df

# ---------- Portföljinkomst-KPI ----------------------------------------------

def _portfolio_income_kpis(df: pd.DataFrame, user_rates: dict) -> Tuple[float, float, float, float]:
    """
    Returnerar (total_värde_SEK, årlig_utdelning_SEK, månadsutdelning_SEK, inkomst_yield_%)
    för nuvarande portfölj (rader med Antal aktier > 0).
    """
    port = df[df.get("Antal aktier", 0) > 0].copy()
    if port.empty:
        return 0.0, 0.0, 0.0, 0.0
    port["FX"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = pd.to_numeric(port["Antal aktier"], errors="coerce").fillna(0) \
                        * pd.to_numeric(port["Aktuell kurs"], errors="coerce").fillna(0) \
                        * pd.to_numeric(port["FX"], errors="coerce").fillna(0)
    total_val = float(port["Värde (SEK)"].sum())

    utd_pa = pd.to_numeric(port.get("Årlig utdelning", 0.0), errors="coerce").fillna(0.0)
    shares = pd.to_numeric(port.get("Antal aktier", 0.0),  errors="coerce").fillna(0.0)
    fx     = pd.to_numeric(port["FX"], errors="coerce").fillna(0.0)
    annual_income = float((shares * utd_pa * fx).sum())

    income_yield = (annual_income / total_val * 100.0) if total_val > 0 else 0.0
    monthly_income = annual_income / 12.0
    return total_val, annual_income, monthly_income, income_yield

# === (Fortsättning följer i nästa del) ===

# app.py — Del 3/?
# --- Datakällor: FMP, SEC (US + IFRS/6-K), Yahoo global fallback, Finnhub ----

# =============== FMP =========================================================
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.0))      # skonsam default
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

# ---------- IFRS/GAAP kvartalsintäkter + valuta ------------------------------

def _pick_rev_candidates():
    return [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
        "RevenueFromContractsWithCustomers",
        "RevenueFromContractsWithCustomersExcludingSalesTaxes",
    ]

def _sec_collect_periods(facts: dict, taxo: str, form_whitelist: tuple, unit_whitelist: tuple):
    gaap = (facts.get("facts") or {}).get(taxo, {})
    out = {}
    for name in _pick_rev_candidates():
        fact = gaap.get(name)
        if not fact:
            continue
        units = (fact.get("units") or {})
        for unit_code in unit_whitelist:
            arr = units.get(unit_code)
            if not isinstance(arr, list):
                continue
            tmp = []
            for it in arr:
                form = (it.get("form") or "").upper()
                if form_whitelist and not any(f in form for f in form_whitelist):
                    continue
                end = _parse_iso(str(it.get("end", "")))
                start = _parse_iso(str(it.get("start", "")))
                val = it.get("val", None)
                if not (end and val is not None):
                    continue
                tmp.append((start, end, val, form))
            if tmp:
                out.setdefault((name, unit_code), []).extend(tmp)
    return out

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
    Försöker även fylla Q4 med årsrapport (10-K/20-F) om Q4 saknas.
    Returnerar (rows, unit) med rows=[(end_date, value), ...] nyast→äldst.
    """
    prefer_units = ("USD","CAD","EUR","GBP")

    # 1) Kvartal (10-Q/6-K)
    rows = []
    unit_found = None
    for taxo, forms in (("us-gaap", ("10-Q","10-Q/A")), ("ifrs-full", ("6-K","6-K/A","10-Q","10-Q/A"))):
        d = _sec_collect_periods(facts, taxo, forms, prefer_units)
        # välj första unit som har data
        for (name, unit_code), arr in d.items():
            # filtrera till duration ~ kvarter (70..100 dagar)
            tmp = []
            for start, end, val, form in arr:
                if not start:
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
            # de-dup per end date, latest wins
            ded = {}
            for end, v in tmp:
                ded[end] = v
            rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)
            unit_found = unit_code
            break
        if rows:
            break

    # 2) Fyll Q4 från årsrapport (10-K/20-F) om vi saknar en kvartal nära års-slut
    if unit_found:
        # Hämta årsdata
        annual_forms = ("10-K","20-F","10-K/A","20-F/A")
        ann = _sec_collect_periods(facts, "us-gaap", annual_forms, (unit_found,))
        if not ann:
            ann = _sec_collect_periods(facts, "ifrs-full", annual_forms, (unit_found,))
        # Map: end_date_annual -> annual_value
        annual_map = {}
        for (_, _u), arr in ann.items():
            for start, end, val, form in arr:
                try:
                    v = float(val)
                    annual_map[end] = max(v, annual_map.get(end, 0.0))
                except Exception:
                    pass

        # Bygg per-fiscal-year summer av kvartal vi redan har
        # och försök härleda en Q4 = År - (Q1+Q2+Q3) om datum matchar års-slut +/- 10 dagar
        q_by_year = {}
        for end, v in rows:
            fy = end.year  # approximation (fiscal vs calendar kan diffa, men bra nog)
            q_by_year.setdefault(fy, []).append((end, v))
        # kontrollera åren där årsrapport finns
        add_candidates = []
        for ann_end, ann_val in annual_map.items():
            fy = ann_end.year
            qs = sorted(q_by_year.get(fy, []), key=lambda t: t[0])  # äldst→nyast
            if len(qs) >= 3:
                s3 = sum(v for _, v in qs[-3:])
                q4 = ann_val - s3
                # rimlig Q4?
                if q4 > 0 and q4 < ann_val * 0.7:  # grov sanity
                    # inkludera om vi inte redan har ett kvartal nära årsslut (±15d)
                    have_near = any(abs((ann_end - q_end).days) <= 15 for q_end, _ in qs)
                    if not have_near:
                        add_candidates.append((ann_end, q4))
        if add_candidates:
            # slå ihop och sortera om
            all_rows = dict(rows)
            for end, v in add_candidates:
                all_rows[end] = v
            rows = sorted(all_rows.items(), key=lambda t: t[0], reverse=True)

    # returnera topp max_quarters
    rows = rows[:max_quarters]
    return rows, unit_found

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

def _ttm_windows(values: list, need: int = 5) -> list:
    """
    Tar [(end_date, kvartalsintäkt), ...] (nyast→äldst) och bygger upp till 'need' TTM-summor:
    [(end_date0, ttm0), (end_date1, ttm1), ...] där ttm0 = sum(q0..q3), ttm1 = sum(q1..q4), osv.
    Vi räknar 5 fönster för att öka chansen att få med Q4 (Dec/Jan) när dataset har luckor.
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
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs","Sektor","Industri","Årlig utdelning","CAGR 5 år (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
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

    # SEC kvartalsintäkter + unit → TTM & konvertering (inkl. Q4-fyll från årsrapport)
    q_rows, rev_unit = _sec_quarterly_revenues_dated_with_unit(facts, max_quarters=20)
    if not q_rows or not rev_unit:
        return out
    conv = 1.0
    if rev_unit.upper() != px_ccy:
        conv = _fx_rate_cached(rev_unit.upper(), px_ccy) or 1.0
    ttm_list = _ttm_windows(q_rows, need=5)  # 5 fönster för robusthet vid Dec/Jan
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list_px:
        ltm_now = ttm_list_px[0][1]
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 historik (4 senaste TTM-fönster)
    if shares_used > 0 and ttm_list_px:
        q_dates = [d for (d, _) in ttm_list_px]
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        # ta de 4 senaste distinkta fönstren
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

    # Bas: namn/valuta/price/sector/industry
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn","Valuta","Aktuell kurs","Sektor","Industri","Årlig utdelning","CAGR 5 år (%)"):
        if y.get(k) not in (None, "", 0, 0.0):
            out[k] = y[k]
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

    # Kvartalsintäkter → TTM
    q_rows = _yfi_quarterly_revenues(t)
    if not q_rows or len(q_rows) < 4:
        return out
    ttm_list = _ttm_windows(q_rows, need=5)

    # Valutakonvertering om financialCurrency != prisvaluta
    fin_ccy = str(info.get("financialCurrency") or px_ccy).upper()
    conv = 1.0
    if fin_ccy != px_ccy:
        conv = _fx_rate_cached(fin_ccy, px_ccy) or 1.0
    ttm_list_px = [(d, v * conv) for (d, v) in ttm_list]

    # Market cap (nu)
    if mcap <= 0 and shares > 0 and px > 0:
        mcap = shares * px

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

# app.py — Del 4/?
# --- Auto-enrichment: momentum, värderingsnyckeltal, FCF, m.m. ---------------

def _compute_momentum_pct(ticker: str, months: int = 6) -> float:
    """Enkel pris-momentum: (senaste close / close för ~months månader sedan - 1) * 100."""
    try:
        t = yf.Ticker(ticker)
        # Hämtar ~months*31 dagar bakåt för att få med hel månader
        days = int(months * 31)
        hist = t.history(period=f"{days+10}d", interval="1d")
        if hist is None or hist.empty or "Close" not in hist:
            return 0.0
        hist = hist.dropna(subset=["Close"])
        if hist.empty:
            return 0.0
        first = float(hist["Close"].iloc[0])
        last  = float(hist["Close"].iloc[-1])
        if first > 0:
            return round((last/first - 1.0) * 100.0, 2)
    except Exception:
        pass
    return 0.0

def _yahoo_enrich_valuation_and_fcf(tkr: yf.Ticker) -> dict:
    """
    Hämtar EV, multiplar, FCF m.m. från Yahoo (info + ev. statements).
    Returnerar värden i bolagets redovisningsvaluta (ej SEK).
    """
    out = {}
    info = {}
    try:
        info = tkr.info or {}
    except Exception:
        info = {}

    # Direkt från info om finns
    def _num(key):
        v = info.get(key, None)
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    ev         = _num("enterpriseValue")
    evebitda   = _num("enterpriseToEbitda")
    evsales    = _num("enterpriseToRevenue")
    pe_ttm     = _num("trailingPE")
    fcf_info   = _num("freeCashflow")
    ebitda_raw = _num("ebitda")
    marketCap  = _num("marketCap")

    if ev is not None and ev > 0:
        out["EV"] = ev
    if evebitda and evebitda > 0:
        out["EV/EBITDA (TTM)"] = evebitda
    if evsales and evsales > 0:
        out["EV/S (TTM)"] = evsales
    if pe_ttm and pe_ttm > 0:
        out["P/E (TTM)"] = pe_ttm
    if marketCap and marketCap > 0:
        out["_marketCap_raw"] = marketCap  # i bolagsvaluta

    # Försök räkna FCF(TTM) från statements om inte i info
    fcf_val = None
    if fcf_info is not None and fcf_info != 0:
        fcf_val = float(fcf_info)
    else:
        try:
            # Yahoo cashflow kan vara annual eller quarterly; försök TTM ≈ sum senaste 4 kvartal
            qcf = tkr.quarterly_cashflow  # rader: 'Total Cash From Operating Activities', 'Capital Expenditures', etc
            if isinstance(qcf, pd.DataFrame) and not qcf.empty:
                row_ops = None
                for cand in ["Total Cash From Operating Activities","Operating Cash Flow","NetCashProvidedByUsedInOperatingActivities"]:
                    if cand in qcf.index:
                        row_ops = qcf.loc[cand].dropna()
                        break
                row_capex = None
                for cand in ["Capital Expenditures","Investments","PurchaseOfPropertyPlantAndEquipment"]:
                    if cand in qcf.index:
                        row_capex = qcf.loc[cand].dropna()
                        break
                if row_ops is not None and not row_ops.empty and row_capex is not None and not row_capex.empty:
                    # Capex är oftast negativt; FCF = OCF + Capex
                    ocf_ttm = float(row_ops.iloc[:4].sum())
                    capex_ttm = float(row_capex.iloc[:4].sum())
                    fcf_val = ocf_ttm + capex_ttm
        except Exception:
            pass
    if fcf_val is not None:
        out["FCF (TTM)"] = float(fcf_val)

    # Härled ev multiplar om EV finns men inte kvoterna
    try:
        if "EV/S (TTM)" not in out or out["EV/S (TTM)"] <= 0:
            # approximera med: EV / (Revenue TTM)
            # använd income_stmt TTM om möjligt
            # (vi har redan TTM i SEC/Yahoo delar, men här fallback)
            inc = tkr.quarterly_financials
            if isinstance(inc, pd.DataFrame) and not inc.empty:
                rev_row = None
                for k in ["Total Revenue","TotalRevenue","Revenues","Revenue"]:
                    if k in inc.index:
                        rev_row = inc.loc[k].dropna()
                        break
                if rev_row is not None and not rev_row.empty and ev and ev > 0:
                    rev_ttm = float(rev_row.iloc[:4].sum())
                    if rev_ttm > 0:
                        out["EV/S (TTM)"] = float(ev / rev_ttm)
    except Exception:
        pass

    # FCF-yield = FCF / MarketCap
    try:
        mc = out.get("_marketCap_raw") or marketCap
        if mc and mc > 0 and ("FCF (TTM)" in out) and float(out["FCF (TTM)"]) != 0:
            out["FCF-yield (%)"] = float(out["FCF (TTM)"] / mc * 100.0)
    except Exception:
        pass

    return out

def apply_auto_updates_to_row(df: pd.DataFrame, row_idx: int, new_vals: dict, source: str, changes_map: dict, always_stamp: bool = True) -> bool:
    """
    Skriver endast fält som får ett nytt (meningsfullt) värde.
    Uppdaterar TS_ för spårade fält, sätter 'Senast auto-uppdaterad' + källa.
    Om always_stamp=True stämplas även om värdet råkar vara samma.
    Returnerar True om något fält faktiskt ändrades (värdebyte).
    """
    changed_fields = []
    wrote_anything = False
    for f, v in new_vals.items():
        if f not in df.columns:
            continue
        # skrivpolicy
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # Tillåt 0 för icke-nyckelfält; för P/S och aktier vill vi undvika 0
            if f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Utestående aktier"]:
                write_ok = (float(v) > 0)
            else:
                write_ok = (float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")
        else:
            continue
        if not write_ok:
            continue

        old = df.at[row_idx, f]
        same = (str(old) == str(v))
        # skriv alltid värdet (även om samma) för att kunna stämpla TS/källa
        df.at[row_idx, f] = v
        wrote_anything = True
        if (not same):
            changed_fields.append(f)
        if f in TS_FIELDS:
            _stamp_ts_for_field(df, row_idx, f)

    if wrote_anything or always_stamp:
        _note_auto_update(df, row_idx, source)
    if changed_fields:
        ticker = df.at[row_idx, "Ticker"]
        changes_map.setdefault(ticker, []).extend(changed_fields)
        return True
    return False

def auto_fetch_for_ticker(ticker: str):
    """
    Pipeline:
      1) SEC + Yahoo (implied shares) eller Yahoo global fallback
      2) Finnhub (estimat) om saknas
      3) FMP light (P/S) om saknas
      4) Momentum, multiplar, FCF m.m. (Yahoo info/statements)
    Returnerar (vals, debug)
    """
    debug = {"ticker": ticker}
    vals = {}

    # 1) SEC/Yahoo combo (inkl. global Yahoo fallback)
    try:
        base = hamta_sec_yahoo_combo(ticker)
        debug["sec_yahoo"] = {k: base.get(k) for k in [
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Aktuell kurs","Bolagsnamn","Valuta","Sektor","Industri","_debug_shares_source","Årlig utdelning","CAGR 5 år (%)"
        ]}
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Sektor","Industri","Årlig utdelning","CAGR 5 år (%)"]:
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
            if ("Aktuell kurs" not in vals) and (fmpl.get("Aktuell kurs") not in (None, "", 0, 0.0)):
                vals["Aktuell kurs"] = fmpl["Aktuell kurs"]
    except Exception as e:
        debug["fmp_light_err"] = str(e)

    # 4) Momentum + multiplar + FCF
    try:
        t = yf.Ticker(ticker)
        vals["Momentum 6m (%)"]  = _compute_momentum_pct(ticker, 6)
        vals["Momentum 12m (%)"] = _compute_momentum_pct(ticker, 12)
        enrich = _yahoo_enrich_valuation_and_fcf(t)
        debug["yahoo_enrich"] = enrich
        for k in ["EV/S (TTM)","EV/EBITDA (TTM)","P/E (TTM)","FCF (TTM)","FCF-yield (%)"]:
            v = enrich.get(k)
            if v is not None and v == v:  # not NaN
                vals[k] = float(v)
        # MarketCap rå (i bolagsvaluta) som stöd (SEK-varianten räknas senare)
        if enrich.get("_marketCap_raw"):
            vals["_marketCap_raw"] = float(enrich["_marketCap_raw"])
    except Exception as e:
        debug["yahoo_enrich_err"] = str(e)

    return vals, debug

# app.py — Del 5/?
# --- Snapshots, batch-körningar, kontroll & stödlistor -----------------------

# ===== Snapshots =============================================================

def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME):
    """
    Skapar en snapshot-flik i samma Google Sheet: 'Snapshot-YYYYMMDD-HHMMSS'
    och fyller den med hela df. Kräver endast Sheets (inte Drive).
    """
    if df is None or df.empty:
        st.warning("Hoppar över snapshot – inget innehåll.")
        return
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

# ===== Tidsstämpel-analys (äldst/kräver åtgärd) ==============================

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

def build_missing_estimates_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Lista över bolag där 'Omsättning idag' eller 'Omsättning nästa år' är saknad eller gammal.
    Används som snabb vy för manuell estimering.
    """
    out_rows = []
    cutoff = now_dt() - timedelta(days=older_than_days)
    for _, r in df.iterrows():
        o_idag = float(r.get("Omsättning idag", 0.0) or 0.0)
        o_next = float(r.get("Omsättning nästa år", 0.0) or 0.0)
        ts_idag = str(r.get("TS_Omsättning idag","")).strip()
        ts_next = str(r.get("TS_Omsättning nästa år","")).strip()
        ts_idag_dt = pd.to_datetime(ts_idag, errors="coerce")
        ts_next_dt = pd.to_datetime(ts_next, errors="coerce")
        need = (o_idag <= 0.0 or o_next <= 0.0 or
                (pd.notna(ts_idag_dt) and ts_idag_dt.to_pydatetime() < cutoff) or
                (pd.notna(ts_next_dt) and ts_next_dt.to_pydatetime() < cutoff))
        if need:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Omsättning idag": o_idag,
                "TS idag": ts_idag,
                "Omsättning nästa år": o_next,
                "TS nästa år": ts_next,
            })
    return pd.DataFrame(out_rows)

# ===== Batch-kö (A–Ö / Äldst först) ==========================================

def _sorted_ticker_list(df: pd.DataFrame, mode: str = "A–Ö (bolagsnamn)") -> list:
    """
    Returnerar en stabil lista med tickers enligt sorteringsläge:
      - "A–Ö (bolagsnamn)"
      - "Äldst uppdaterade först"
    """
    df2 = df.copy()
    df2 = add_oldest_ts_col(df2)
    if mode.startswith("Äldst"):
        df2 = df2.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"], ascending=[True, True, True])
    else:
        df2 = df2.sort_values(by=["Bolagsnamn","Ticker"], ascending=[True, True])
    return [str(t).upper() for t in df2["Ticker"].tolist() if str(t).strip()]

def init_batch_queue(df: pd.DataFrame, mode: str):
    """
    Bygger batch-kön och nollställer pekare.
    Lagrar i session_state: batch_queue, batch_cursor, batch_mode, batch_log
    """
    q = _sorted_ticker_list(df, mode=mode)
    st.session_state.batch_queue  = q
    st.session_state.batch_cursor = 0
    st.session_state.batch_mode   = mode
    st.session_state.batch_log    = {"changed": {}, "misses": {}}
    st.success(f"Batch-kö initierad ({len(q)} tickers), läge: {mode}")

def run_batch_step(df: pd.DataFrame, user_rates: dict, step_size: int = 10, make_snapshot: bool = False):
    """
    Kör nästa 'step_size' tickers från batch-kön. Sparar ändringar efter chunk.
    Returnerar uppdaterad df.
    """
    if "batch_queue" not in st.session_state or not st.session_state.batch_queue:
        st.warning("Ingen batch-kö initierad ännu.")
        return df
    cur = int(st.session_state.get("batch_cursor", 0))
    q = st.session_state.batch_queue
    if cur >= len(q):
        st.info("Batch-kön är redan färdigkörd.")
        return df

    upto = min(len(q), cur + max(1, int(step_size)))
    part = q[cur:upto]
    st.write(f"Kör {len(part)} tickers ({cur+1}–{upto} av {len(q)}): {', '.join(part[:5])}{'...' if len(part)>5 else ''}")

    changes_map = st.session_state.batch_log.get("changed", {})
    misses_map  = st.session_state.batch_log.get("misses", {})

    any_changed = False
    for tkr in part:
        try:
            vals, debug = auto_fetch_for_ticker(tkr)
            # hitta rad-index
            row_idx_list = df.index[df["Ticker"].str.upper() == tkr].tolist()
            if not row_idx_list:
                misses_map[tkr] = ["Ticker saknas i tabellen"]
                continue
            ridx = row_idx_list[0]
            changed = apply_auto_updates_to_row(df, ridx, vals, source=f"Batch ({st.session_state.batch_mode})", changes_map=changes_map, always_stamp=True)
            if not changed:
                # även om inga värdebyten → notera "inga nya fält" (men TS stämplad via always_stamp)
                misses_map[tkr] = list(vals.keys()) if vals else ["(inga nya fält)"]
            any_changed = any_changed or changed
        except Exception as e:
            misses_map[tkr] = [f"error: {e}"]

    # Räkna om beräkningar och spara efter chunk
    df = uppdatera_berakningar(df, user_rates)
    if any_changed or True:  # skriv även om inga ändringar, för TS-stämplingar
        spara_data(df, do_snapshot=make_snapshot)

    st.session_state.batch_log["changed"] = changes_map
    st.session_state.batch_log["misses"]  = misses_map
    st.session_state.batch_cursor = upto

    st.success(f"Klar: {cur+1}–{upto}. Återstår {len(q)-upto} st.")
    return df

# ===== Kontrollvy (översikt + manuella behov) ================================

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

    # 3) Estimat saknas/äldre (Omsättning idag / nästa år)
    st.subheader("📋 Estimat saknas eller gamla (Omsättning idag / nästa år)")
    older_days_est = st.number_input("Flagga om TS för estimat är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30, key="ctrl_old_est_days")
    est_need = build_missing_estimates_df(df, older_than_days=int(older_days_est))
    if est_need.empty:
        st.info("Alla estimat ser uppdaterade ut.")
    else:
        st.warning(f"{len(est_need)} bolag har saknade/gamla estimat:")
        st.dataframe(est_need, use_container_width=True, hide_index=True)

    st.divider()

    # 4) Senaste körlogg (om du nyss körde Auto eller Batch)
    st.subheader("📒 Senaste körlogg")
    log_auto  = st.session_state.get("last_auto_log")
    log_batch = st.session_state.get("batch_log")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Auto: Ändringar** (ticker → fält)")
        if log_auto and log_auto.get("changed"):
            st.json(log_auto["changed"])
        else:
            st.write("–")
    with col2:
        st.markdown("**Auto: Missar**")
        if log_auto and log_auto.get("misses"):
            st.json(log_auto["misses"])
        else:
            st.write("–")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Batch: Ändringar**")
        if log_batch and log_batch.get("changed"):
            st.json(log_batch["changed"])
        else:
            st.write("–")
    with col4:
        st.markdown("**Batch: Missar**")
        if log_batch and log_batch.get("misses"):
            st.json(log_batch["misses"])
        else:
            st.write("–")

# app.py — Del 6/?
# --- Risklabel, scoring & investeringsförslag (förbättrad) -------------------

# ---------- Cap bucket / risklabel -------------------------------------------

def _mcap_native(row: pd.Series) -> float:
    """Market cap i bolagets rapportvaluta (approx)."""
    try:
        if pd.notna(row.get("_marketCap_raw")) and float(row["_marketCap_raw"]) > 0:
            return float(row["_marketCap_raw"])
    except Exception:
        pass
    try:
        px = float(row.get("Aktuell kurs", 0.0) or 0.0)
        sh_mn = float(row.get("Utestående aktier", 0.0) or 0.0)  # milj.
        if px > 0 and sh_mn > 0:
            return px * sh_mn * 1_000_000.0
    except Exception:
        pass
    return 0.0

def _to_usd(amount_native: float, ccy: str, user_rates: dict) -> float:
    """
    Konvertera belopp i 'ccy' till USD via SEK-växelkurser i sidopanelen:
      x_ccy → SEK → USD.
    """
    try:
        if amount_native <= 0:
            return 0.0
        ccy = (ccy or "USD").upper()
        rate_ccy_to_SEK = float(hamta_valutakurs(ccy, user_rates))
        rate_USD_to_SEK = float(hamta_valutakurs("USD", user_rates))
        if rate_ccy_to_SEK > 0 and rate_USD_to_SEK > 0:
            sek = amount_native * rate_ccy_to_SEK
            usd = sek / rate_USD_to_SEK
            return float(usd)
    except Exception:
        pass
    return 0.0

def _cap_bucket_label_usd(mcap_usd: float) -> str:
    """Etikett baserad på USD-marketcap."""
    if mcap_usd >= 200_000_000_000:  # 200B
        return "Mega"
    if mcap_usd >= 10_000_000_000:
        return "Large"
    if mcap_usd >= 2_000_000_000:
        return "Mid"
    if mcap_usd >= 300_000_000:
        return "Small"
    if mcap_usd > 0:
        return "Micro"
    return "Okänd"

def _ensure_cap_bucket_cols(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Beräknar/uppdaterar MarketCap (SEK) och Risklabel."""
    if df.empty:
        return df
    df = df.copy()
    # MarketCap (SEK)
    if "MarketCap (SEK)" not in df.columns or df["MarketCap (SEK)"].isna().all():
        mcs = []
        for _, r in df.iterrows():
            try:
                mc_native = _mcap_native(r)
                fx = hamta_valutakurs(r.get("Valuta","SEK"), user_rates)
                mcs.append(round(mc_native * float(fx), 2))
            except Exception:
                mcs.append(None)
        df["MarketCap (SEK)"] = mcs
    # Risklabel (USD)
    labels = []
    for _, r in df.iterrows():
        mcu = _to_usd(_mcap_native(r), r.get("Valuta","USD"), user_rates)
        labels.append(_cap_bucket_label_usd(mcu))
    df["Risklabel"] = labels
    return df

# ---------- Scoring (Growth & Dividend) --------------------------------------

def _growth_score(row: pd.Series, riktkurs_col: str) -> float:
    """0..100. Viktar uppsida, värdering, momentum och FCF-yield."""
    parts = []

    # Uppsida mot vald riktkurs
    try:
        px = float(row.get("Aktuell kurs", 0.0) or 0.0)
        rk = float(row.get(riktkurs_col, 0.0) or 0.0)
        if px > 0 and rk > 0:
            pot = (rk - px) / px * 100.0
            pot_score = _scale_0_100(pot, 0.0, 150.0, invert=False)  # 0%→0, 150%→100
            parts.append(0.40 * pot_score)
    except Exception:
        pass

    # Värdering
    try:
        val_score = _valuation_score(row)
        if val_score == val_score:  # not NaN
            parts.append(0.35 * float(val_score))
    except Exception:
        pass

    # Momentum 12m
    try:
        mom = float(row.get("Momentum 12m (%)", 0.0) or 0.0)
        mom_score = _scale_0_100(mom, -50.0, 100.0, invert=False)
        parts.append(0.15 * mom_score)
    except Exception:
        pass

    # FCF-yield (högre bättre)
    try:
        fcfy = float(row.get("FCF-yield (%)", 0.0) or 0.0)
        fcf_score = _scale_0_100(fcfy, 0.0, 10.0, invert=False)
        parts.append(0.10 * fcf_score)
    except Exception:
        pass

    if not parts:
        return 0.0
    return round(sum(parts), 1)

def _dividend_score(row: pd.Series) -> float:
    """0..100. Viktar yield, säkerhet, värdering; sänker för Microcap."""
    parts = []

    # Yield
    try:
        y = _dividend_yield_pct(row)
        if y == y:
            # 3% → 0, 12% → 100 (hög yield bättre men risky >12% ger max)
            y_score = _scale_0_100(y, 3.0, 12.0, invert=False)
            parts.append(0.45 * y_score)
    except Exception:
        pass

    # Säkerhet (badge)
    try:
        badge = str(row.get("Dividend-säkerhet","Okänt"))
        safemap = {"Stark": 100.0, "OK": 65.0, "Risk": 25.0, "Okänt": 45.0}
        parts.append(0.35 * safemap.get(badge, 45.0))
    except Exception:
        pass

    # Värdering
    try:
        val = _valuation_score(row)
        if val == val:
            parts.append(0.20 * float(val))
    except Exception:
        pass

    score = sum(parts) if parts else 0.0

    # Microcap-penalti för utdelning
    try:
        rl = str(row.get("Risklabel","Okänd"))
        if rl == "Micro":
            score *= 0.85
    except Exception:
        pass

    return round(score, 1)

# ---------- Investeringsförslag (förbättrad vy) ------------------------------

def _format_mcap_pair(row: pd.Series, user_rates: dict) -> tuple[str, str]:
    """Returnerar (mcap_native_str, mcap_sek_str)."""
    mc_nat = _mcap_native(row)
    mc_sek = 0.0
    try:
        fx = hamta_valutakurs(row.get("Valuta","SEK"), user_rates)
        mc_sek = mc_nat * float(fx)
    except Exception:
        pass
    return f"{_fmt_large(mc_nat)} {row.get('Valuta','')}", _fmt_sek(mc_sek)

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Filters
    cap_filter = st.selectbox("Filter: Börsvärde", ["Alla","Mega","Large","Mid","Small","Micro"], index=0)
    sector_all = sorted({str(s) for s in df.get("Sektor", pd.Series(dtype=str)).fillna("").unique() if str(s).strip()})
    sectors = st.multiselect("Filter: Sektor(er)", sector_all, default=[])
    style = st.radio("Fokus", ["Growth","Dividend"], horizontal=True, index=0)

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0, min_value=0.0)
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)

    subset = st.radio("Urval", ["Alla bolag","Endast portfölj"], horizontal=True)
    sortläge = st.radio("Sortering", ["Högst score först","Störst uppsida"], horizontal=True)

    # Säkerställ nyckelkolumner
    df2 = uppdatera_berakningar(df.copy(), user_rates)
    df2 = berakna_utdelningssakerhet(df2)
    df2 = _ensure_cap_bucket_cols(df2, user_rates)

    base = df2[df2["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df2.copy()
    # Grundfilter: riktkurs och pris > 0
    base = base[(pd.to_numeric(base[riktkurs_val], errors="coerce") > 0) & (pd.to_numeric(base["Aktuell kurs"], errors="coerce") > 0)].copy()
    if cap_filter != "Alla":
        base = base[base["Risklabel"] == cap_filter].copy()
    if sectors:
        base = base[base["Sektor"].isin(sectors)].copy()

    if base.empty:
        st.info("Inga bolag matchar filtren just nu.")
        return

    # Potential %
    base["Potential (%)"] = (pd.to_numeric(base[riktkurs_val], errors="coerce") - pd.to_numeric(base["Aktuell kurs"], errors="coerce")) \
                            / pd.to_numeric(base["Aktuell kurs"], errors="coerce") * 100.0

    # Score
    if style == "Growth":
        base["Score"] = base.apply(lambda r: _growth_score(r, riktkurs_val), axis=1)
    else:
        base["Score"] = base.apply(_dividend_score, axis=1)

    # Sortering
    if sortläge.startswith("Högst score"):
        base = base.sort_values(by=["Score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)
    else:
        base = base.sort_values(by=["Potential (%)","Score"], ascending=[False, False]).reset_index(drop=True)

    # Robust bläddring
    key_idx = "forslags_index_v2"
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0
    st.session_state[key_idx] = min(st.session_state[key_idx], len(base)-1)

    # Navigering
    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", use_container_width=True):
            st.session_state[key_idx] = max(0, st.session_state[key_idx] - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state[key_idx]+1} / {len(base)}")
    with col_next:
        if st.button("➡️ Nästa", use_container_width=True):
            st.session_state[key_idx] = min(len(base)-1, st.session_state[key_idx] + 1)

    rad = base.iloc[st.session_state[key_idx]]

    # Beräkna köpstorlek
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = float(rad["Aktuell kurs"]) * float(vx)
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    # Portföljandel före/efter
    port = df2[df2["Antal aktier"] > 0].copy()
    port["FX"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = pd.to_numeric(port["Antal aktier"], errors="coerce").fillna(0) \
                        * pd.to_numeric(port["Aktuell kurs"], errors="coerce").fillna(0) \
                        * pd.to_numeric(port["FX"], errors="coerce").fillna(0)
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0
    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"].astype(str).str.upper() == str(rad["Ticker"]).upper()]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Rubrik
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']}) — {rad.get('Sektor','') or 'Okänd sektor'} • {rad.get('Risklabel','')} cap")

    # Snabbdata-rad
    mc_nat_str, mc_sek_str = _format_mcap_pair(rad, user_rates)
    ps_now = float(rad.get("P/S", 0.0) or 0.0)
    ps_snitt = float(rad.get("P/S-snitt", 0.0) or 0.0)
    sh_mn = float(rad.get("Utestående aktier", 0.0) or 0.0)

    cols_top = st.columns(4)
    with cols_top[0]:
        st.metric("Aktuell kurs", f"{rad['Aktuell kurs']:.2f} {rad['Valuta']}")
    with cols_top[1]:
        st.metric("MarketCap", mc_nat_str, help=f"I SEK: {mc_sek_str}")
    with cols_top[2]:
        st.metric("P/S (nu)", f"{ps_now:.2f}", help="TTM-baserat")
    with cols_top[3]:
        st.metric("P/S-snitt (Q1–Q4)", f"{ps_snitt:.2f}")

    # Huvudlista
    lines = [
        f"- **Vald riktkurs:** {riktkurs_val} → {round(rad[riktkurs_val],2)} {rad['Valuta']}",
        f"- **Uppsida:** {round(rad['Potential (%)'],2)} %",
        f"- **Utestående aktier:** {sh_mn:.2f} M",
        f"- **Score ({style}):** {rad['Score']:.1f}",
        f"- **Rekommenderat köp (≈):** {antal_köp} st  ⇒ investering ~ {_fmt_sek(investering)}",
        f"- **Portföljandel:** nu {nuv_andel}% → efter {ny_andel}%"
    ]
    st.markdown("\n".join(lines))

    # Detaljer
    with st.expander("🔎 Detaljer & nyckeltal"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.caption("Värdering")
            st.write(f"- EV/S (TTM): {rad.get('EV/S (TTM)','–')}")
            st.write(f"- EV/EBITDA (TTM): {rad.get('EV/EBITDA (TTM)','–')}")
            st.write(f"- P/E (TTM): {rad.get('P/E (TTM)','–')}")
            st.write(f"- Valuation score: {_valuation_score(rad)}")

        with c2:
            st.caption("Tillväxt & momentum")
            st.write(f"- CAGR 5 år: {_fmt_pct(rad.get('CAGR 5 år (%)', 0.0) or 0.0)}")
            st.write(f"- Momentum 6m: {_fmt_pct(rad.get('Momentum 6m (%)', 0.0) or 0.0)}")
            st.write(f"- Momentum 12m: {_fmt_pct(rad.get('Momentum 12m (%)', 0.0) or 0.0)}")

        with c3:
            st.caption("Kassa & utdelning")
            dy = _dividend_yield_pct(rad)
            st.write(f"- Årlig utdelning/aktie: {rad.get('Årlig utdelning', 0.0)} {rad['Valuta']}")
            st.write(f"- Direktavkastning: {_fmt_pct(dy) if dy==dy else '–'}")
            st.write(f"- FCF (TTM): {_fmt_large(rad.get('FCF (TTM)', 0.0) or 0.0)} {rad['Valuta']}")
            st.write(f"- FCF-yield: {_fmt_pct(rad.get('FCF-yield (%)', 0.0) or 0.0)}")
            st.write(f"- Dividend-säkerhet: {rad.get('Dividend-säkerhet','Okänt')}")

        st.divider()
        st.caption("Historik: P/S Q1–Q4")
        psq = [rad.get("P/S Q1", None), rad.get("P/S Q2", None), rad.get("P/S Q3", None), rad.get("P/S Q4", None)]
        st.write(", ".join([f"Q{i+1}: {psq[i]:.2f}" if psq[i] and psq[i]==psq[i] else f"Q{i+1}: –" for i in range(4)]))

# app.py — Del 7/?
# --- Hjälpfunktioner: formatering, scaling, värdering, utdelningssäkerhet ----

# ---------- Formatering ------------------------------------------------------

def _fmt_large(x: float) -> str:
    """Kompakt enhetsformatering i bolagets valuta: tn, mdr, mn, k."""
    try:
        x = float(x)
    except Exception:
        return "–"
    neg = x < 0
    x = abs(x)
    if x >= 1_000_000_000_000:   # 1e12
        v = x / 1_000_000_000_000.0
        s = f"{v:.2f} tn"
    elif x >= 1_000_000_000:     # 1e9
        v = x / 1_000_000_000.0
        s = f"{v:.2f} mdr"
    elif x >= 1_000_000:         # 1e6
        v = x / 1_000_000.0
        s = f"{v:.2f} mn"
    elif x >= 1_000:             # 1e3
        v = x / 1_000.0
        s = f"{v:.2f} k"
    else:
        s = f"{x:.2f}"
    return f"-{s}" if neg else s

def _fmt_sek(x: float) -> str:
    try:
        return f"{_fmt_large(float(x))} SEK"
    except Exception:
        return "–"

def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "–"

# ---------- Skala 0..100 -----------------------------------------------------

def _scale_0_100(value: float, low: float, high: float, invert: bool = False, clamp: bool = True) -> float:
    """
    Linjär skalning till 0..100. Om invert=True blir (low→100, high→0).
    """
    try:
        v = float(value)
        lo = float(low); hi = float(high)
        if hi == lo:
            return 0.0
        t = (v - lo) / (hi - lo)
        if clamp:
            t = max(0.0, min(1.0, t))
        if invert:
            t = 1.0 - t
        return round(t * 100.0, 1)
    except Exception:
        return 0.0

# ---------- Värderingsscore --------------------------------------------------

def _valuation_score(row: pd.Series) -> float:
    """
    0..100 (högre bättre/“billigare”).
    Använder flera signaler om de finns: P/S relativt PS-snitt, EV/S, P/E, FCF-yield.
    """
    parts = []

    # P/S relativt historiskt snitt (lägre än snitt → bättre)
    try:
        ps_now   = float(row.get("P/S", 0.0) or 0.0)
        ps_snitt = float(row.get("P/S-snitt", 0.0) or 0.0)
        if ps_now > 0 and ps_snitt > 0:
            ratio = ps_now / ps_snitt  # 0.5 = mkt billig, 1.0 = i linje, 2.0 = dyr
            # 0.5→100, 1.0→60, 1.5→30, 2.0→0 ungefär
            score = _scale_0_100(ratio, 2.0, 0.5, invert=False)  # högre ratio → sämre
            parts.append(0.35 * score)
    except Exception:
        pass

    # EV/S (lägre bättre)
    try:
        evs = float(row.get("EV/S (TTM)", 0.0) or 0.0)
        if evs > 0:
            score = _scale_0_100(evs, 15.0, 2.0, invert=True)  # 2→100, 15→0
            parts.append(0.25 * score)
    except Exception:
        pass

    # P/E (lägre bättre, men caps)
    try:
        pe = float(row.get("P/E (TTM)", 0.0) or 0.0)
        if pe > 0:
            score = _scale_0_100(pe, 40.0, 10.0, invert=True)  # 10→100, 40→0
            parts.append(0.20 * score)
    except Exception:
        pass

    # FCF-yield (högre bättre)
    try:
        fcfy = float(row.get("FCF-yield (%)", 0.0) or 0.0)
        score = _scale_0_100(fcfy, 0.0, 10.0, invert=False)  # 0→0, 10%→100
        parts.append(0.20 * score)
    except Exception:
        pass

    if not parts:
        return 0.0
    return round(sum(parts), 1)

# ---------- Utdelningsmått ---------------------------------------------------

def _dividend_yield_pct(row: pd.Series) -> float:
    """Direktavkastning i %, baserat på Årlig utdelning / Aktuell kurs."""
    try:
        dps = float(row.get("Årlig utdelning", 0.0) or 0.0)
        px  = float(row.get("Aktuell kurs", 0.0) or 0.0)
        if px > 0 and dps >= 0:
            return float(dps / px * 100.0)
    except Exception:
        pass
    return float("nan")

def _total_dividend_cash(row: pd.Series) -> float:
    """
    Total utdelningskostnad (i bolagsvalutan) ≈ DPS * antal aktier.
    Antal aktier lagras i miljoner.
    """
    try:
        dps = float(row.get("Årlig utdelning", 0.0) or 0.0)
        sh_mn = float(row.get("Utestående aktier", 0.0) or 0.0)
        if dps >= 0 and sh_mn > 0:
            return dps * sh_mn * 1_000_000.0
    except Exception:
        pass
    return 0.0

def berakna_utdelningssakerhet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar:
      - Direktavkastning (%)
      - Payout på FCF (Total utdelning / FCF TTM)
      - Badge 'Dividend-säkerhet': Stark / OK / Risk / Okänt
    """
    if df.empty:
        return df
    df = df.copy()
    yields = []
    payout_fcf = []
    safety = []

    for _, r in df.iterrows():
        yld = _dividend_yield_pct(r)
        yields.append(yld if yld == yld else float("nan"))  # bevara NaN

        fcf = r.get("FCF (TTM)", None)
        try:
            fcf = float(fcf) if fcf is not None else None
        except Exception:
            fcf = None

        tot_div = _total_dividend_cash(r)
        ratio = float("nan")
        if fcf is not None and fcf != 0:
            ratio = max(0.0, tot_div / fcf)  # om fcf<0 → NaN (osunt)
        payout_fcf.append(ratio)

        # Badge
        badge = "Okänt"
        try:
            if fcf is None or fcf == 0:
                badge = "Risk"
            elif fcf < 0:
                badge = "Risk"
            else:
                if ratio <= 0.6:
                    badge = "Stark"
                elif ratio <= 1.0:
                    badge = "OK"
                else:
                    badge = "Risk"
        except Exception:
            badge = "Okänt"
        safety.append(badge)

    df["Direktavkastning (%)"] = yields
    df["Payout FCF"] = payout_fcf
    df["Dividend-säkerhet"] = safety
    return df

# app.py — Del 8/?
# --- Lägg till/uppdatera bolag (enkel + per-ticker auto + pris-only) --------

# Fält som triggar "Senast manuellt uppdaterad" i formuläret
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

def _render_ts_badges(row: pd.Series):
    """Visar små etiketter för senaste auto/manuell + källa."""
    manu = str(row.get("Senast manuellt uppdaterad","")).strip()
    auto = str(row.get("Senast auto-uppdaterad","")).strip()
    src  = str(row.get("Senast uppdaterad källa","")).strip()
    parts = []
    if manu:
        parts.append(f"🖊️ **Manuellt:** {manu}")
    if auto:
        parts.append(f"⚙️ **Auto:** {auto}")
    if src:
        parts.append(f"**Källa:** {src}")
    if parts:
        st.info(" • ".join(parts))
    else:
        st.info("Inga uppdateringsstämplar ännu.")

def _update_price_only(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, bool]:
    """Hämtar endast aktuell kurs (Yahoo). Stämplar auto-källa."""
    tkr = str(ticker or "").strip().upper()
    if not tkr:
        st.warning("Ange ticker först.")
        return df, False
    ridx_list = df.index[df["Ticker"].astype(str).str.upper() == tkr].tolist()
    if not ridx_list:
        st.error(f"{tkr} hittades inte i tabellen.")
        return df, False
    ridx = ridx_list[0]
    y = hamta_yahoo_fält(tkr)
    changed = False
    if y.get("Aktuell kurs", 0) > 0:
        df.at[ridx, "Aktuell kurs"] = float(y["Aktuell kurs"])
        changed = True
    # komplettera namn/valuta om saknas
    if y.get("Bolagsnamn"):
        df.at[ridx, "Bolagsnamn"] = y["Bolagsnamn"]
    if y.get("Valuta"):
        df.at[ridx, "Valuta"] = y["Valuta"]
    _note_auto_update(df, ridx, source="Pris-only (Yahoo)")
    # inga TS_ för pris (ej spårat fält)
    return df, changed

def _update_full_single(df: pd.DataFrame, user_rates: dict, ticker: str) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline för EN ticker.
    Stämplar TS även om värdena råkar bli samma.
    """
    tkr = str(ticker or "").strip().upper()
    if not tkr:
        st.warning("Ange ticker först.")
        return df, {"err": "tom ticker"}
    ridx_list = df.index[df["Ticker"].astype(str).str.upper() == tkr].tolist()
    if not ridx_list:
        st.error(f"{tkr} hittades inte i tabellen.")
        return df, {"err": "saknas i tabellen"}
    ridx = ridx_list[0]

    vals, debug = auto_fetch_for_ticker(tkr)
    _ = apply_auto_updates_to_row(df, ridx, vals, source="Auto (singel)", changes_map={}, always_stamp=True)
    df = uppdatera_berakningar(df, user_rates)
    # skriv alltid (för TS-stämplar)
    spara_data(df, do_snapshot=False)
    return df, debug

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsläge för bläddring
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], key="edit_sort_mode")
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # Bygg etiketter och karta
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": str(r['Ticker']).upper() for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    # Robust index
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(val_lista)-1))

    # Välj rad
    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=st.session_state.edit_index, key="edit_selectbox")
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", use_container_width=True):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {max(1, st.session_state.edit_index)}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa", use_container_width=True):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        cur_ticker = namn_map[valt_label]
        bef_rows = df[df["Ticker"].astype(str).str.upper() == cur_ticker]
        bef = bef_rows.iloc[0] if not bef_rows.empty else pd.Series({}, dtype=object)
    else:
        cur_ticker = ""
        bef = pd.Series({}, dtype=object)

    # Visa TS-badges om valt bolag finns
    if cur_ticker:
        _render_ts_badges(bef)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0, step=0.01, format="%.4f")
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0, step=1.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, step=0.01)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, step=0.01)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, step=0.01)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, step=0.01)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, step=0.01)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0, step=1.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0, step=1.0)
            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över manuella 0-värden):**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) via Yahoo")
            st.write("- Riktkurser/beräkningar räknas om")

        spar = st.form_submit_button("💾 Spara")

    # Spara → uppdatera/infoga
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
                df.loc[df["Ticker"].astype(str).str.upper()==ticker.upper(), k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Sätt manuell TS + TS_ per fält
        ridx = df.index[df["Ticker"].astype(str).str.upper()==ticker.upper()][0]
        if datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Hämta basfält från Yahoo
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[ridx, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[ridx, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[ridx, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data and data.get("Årlig utdelning") is not None: df.loc[ridx, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data and data.get("CAGR 5 år (%)") is not None:     df.loc[ridx, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat.")

    # Snabb-åtgärder för valt bolag
    st.subheader("⚡ Snabbåtgärder (valt bolag)")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("🔄 Uppdatera kurs (Yahoo)"):
            if cur_ticker:
                df2, changed = _update_price_only(df.copy(), cur_ticker)
                if changed:
                    spara_data(df2)
                st.success("Kurs uppdaterad." if changed else "Ingen kursförändring.")
                df = df2
            else:
                st.warning("Välj ett bolag först.")
    with c2:
        if st.button("⚙️ Full auto (ett bolag)"):
            if cur_ticker:
                df2, dbg = _update_full_single(df.copy(), user_rates, cur_ticker)
                st.success("Auto-uppdatering klar för det valda bolaget.")
                st.json(dbg)
                df = df2
            else:
                st.warning("Välj ett bolag först.")
    with c3:
        st.caption("Tips: Använd pilarna ovan för att bläddra mellan bolag och kör 'Full auto' per bolag i tur och ordning.")

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

# app.py — Del 9/?
# --- Portföljvy, sektorfördelning & Säljvakt ---------------------------------

def _sector_table(port: pd.DataFrame) -> pd.DataFrame:
    """
    Summerar portföljvärde och utdelningar per sektor (SEK).
    Kräver kolumnerna: 'Sektor', 'Värde (SEK)', 'Total årlig utdelning (SEK)'.
    """
    if port.empty:
        return pd.DataFrame(columns=["Sektor","Värde (SEK)","Andel (%)","Årlig utdelning (SEK)","Yield (%)"])
    grp = port.groupby(port["Sektor"].fillna("Okänd"), dropna=False).agg({
        "Värde (SEK)": "sum",
        "Total årlig utdelning (SEK)": "sum"
    }).reset_index().rename(columns={"Sektor":"Sektor"})
    tot = float(grp["Värde (SEK)"].sum()) or 1.0
    grp["Andel (%)"] = (grp["Värde (SEK)"] / tot * 100.0).round(2)
    grp["Yield (%)"] = (grp["Årlig utdelning (SEK)"] / grp["Värde (SEK)"] * 100.0).replace([np.inf, -np.inf], np.nan).round(2)
    grp = grp.rename(columns={"Total årlig utdelning (SEK)":"Årlig utdelning (SEK)"})
    grp = grp.sort_values(by="Andel (%)", ascending=False)
    return grp

def _sell_guard_table(port: pd.DataFrame, riktkurs_col: str, overweight_cap: float) -> pd.DataFrame:
    """
    Bygger en tabell med 'trimningskandidater'.
    Heuristik:
      - Övervärderad om: Valuation score < 35 eller P/S > 1.3 * P/S-snitt, eller Dividend-säkerhet == Risk
      - Övervikt om portföljandel > overweight_cap (%)
      - Föreslagen trim (antal) ≈ sälj så att andelen når overweight_cap (avrundat nedåt till heltal)
    Kräver kolumner:
      'Aktuell kurs','Valuta','FX','Antal aktier','Värde (SEK)','P/S','P/S-snitt','Dividend-säkerhet','Score'/'valuation score'
    """
    if port.empty:
        return pd.DataFrame(columns=["Ticker","Bolagsnamn","Andel (%)","Övervärderad?","Skäl","Föreslagen trim (antal)"])

    rows = []
    tot_val = float(port["Värde (SEK)"].sum()) or 1.0

    for _, r in port.iterrows():
        try:
            andel = float(r.get("Andel (%)", 0.0) or 0.0)
            ps_now = float(r.get("P/S", 0.0) or 0.0)
            ps_avg = float(r.get("P/S-snitt", 0.0) or 0.0)
            val_score = _valuation_score(r)
            div_badge = str(r.get("Dividend-säkerhet","Okänt"))
            px = float(r.get("Aktuell kurs", 0.0) or 0.0)
            fx = float(r.get("FX", 1.0) or 1.0)
            qty = int(float(r.get("Antal aktier", 0.0) or 0.0))

            # Uppsida mot vald riktkurs (kan vara negativ om över värde)
            pot = float("nan")
            rk = float(r.get(riktkurs_col, 0.0) or 0.0)
            if px > 0 and rk > 0:
                pot = (rk - px) / px * 100.0

            overvalued = (val_score < 35.0) or (ps_now > 0 and ps_avg > 0 and ps_now > 1.3 * ps_avg) or (div_badge == "Risk")
            reasons = []
            if val_score < 35.0: reasons.append(f"Låg valuation-score ({val_score:.0f})")
            if ps_now > 0 and ps_avg > 0 and ps_now > 1.3 * ps_avg: reasons.append(f"P/S {ps_now:.1f} > 1.3×snitt {ps_avg:.1f}")
            if div_badge == "Risk": reasons.append("Utdelning: Risk")
            if pot == pot and pot < -10.0:  # pris långt över riktkurs
                overvalued = True
                reasons.append(f"Över riktkurs ({pot:.0f}%)")

            overweight = andel > float(overweight_cap)
            trim_qty = 0
            if overvalued and overweight and px > 0 and fx > 0 and qty > 0:
                # Sälj så att andel kommer ned till overweight_cap
                value_now = float(r["Värde (SEK)"])
                target_value = tot_val * (float(overweight_cap)/100.0)
                reduce_sek = max(0.0, value_now - target_value)
                trim_qty = int(np.floor(reduce_sek / (px * fx)))

            rows.append({
                "Ticker": r["Ticker"],
                "Bolagsnamn": r["Bolagsnamn"],
                "Andel (%)": round(andel,2),
                "Övervärderad?": "Ja" if overvalued else "Nej",
                "Skäl": "; ".join(reasons) if reasons else "—",
                "Föreslagen trim (antal)": int(trim_qty)
            })
        except Exception:
            pass

    out = pd.DataFrame(rows)
    out = out.sort_values(by=["Övervärderad?","Andel (%)"], ascending=[False, False])
    return out

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")

    # Filtrera innehav
    port = df[pd.to_numeric(df["Antal aktier"], errors="coerce").fillna(0) > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Räkna värden i SEK
    port["FX"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = pd.to_numeric(port["Antal aktier"], errors="coerce").fillna(0) \
                        * pd.to_numeric(port["Aktuell kurs"], errors="coerce").fillna(0) \
                        * pd.to_numeric(port["FX"], errors="coerce").fillna(0)
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = (np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)).round(2)
    port["Total årlig utdelning (SEK)"] = pd.to_numeric(port["Antal aktier"], errors="coerce").fillna(0) \
                                        * pd.to_numeric(port["Årlig utdelning"], errors="coerce").fillna(0) \
                                        * pd.to_numeric(port["FX"], errors="coerce").fillna(0)
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    # Topp-metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Totalt portföljvärde", _fmt_sek(total_värde))
    with c2:
        st.metric("Total årlig utdelning", _fmt_sek(tot_utd))
    with c3:
        st.metric("Ungefärlig månadsutdelning", _fmt_sek(tot_utd/12.0))

    # Vald riktkurs för beräkningar i säljvakt/indikatorer
    riktkurs_val = st.selectbox("Jämförelse mot riktkurs", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)

    # Visa tabell
    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)","Sektor","Risklabel"]
    for addc in ["P/S","P/S-snitt", riktkurs_val]:
        if addc not in show_cols and addc in port.columns:
            show_cols.append(addc)
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    st.divider()

    # Sektorfördelning
    st.subheader("📊 Sektorfördelning")
    sec_tbl = _sector_table(port)
    st.dataframe(sec_tbl, use_container_width=True, hide_index=True)

    st.divider()

    # Säljvakt
    st.subheader("🚨 Säljvakt")
    overweight_cap = st.slider("Max andel per enskild position (%)", min_value=5, max_value=30, value=15, step=1)
    trim_tbl = _sell_guard_table(port, riktkurs_col=riktkurs_val, overweight_cap=float(overweight_cap))

    # Visa TOP N förslag
    max_suggestions = st.slider("Visa upp till N trimningsförslag", min_value=3, max_value=20, value=8, step=1)
    vis_trim = trim_tbl.copy()
    # sortera: övervärderade först, därefter störst andel, sedan störst trimbehov
    vis_trim["SortKey"] = vis_trim.apply(lambda r: (0 if r["Övervärderad?"]=="Ja" else 1, -float(r["Andel (%)"]), -int(r["Föreslagen trim (antal)"])), axis=1)
    vis_trim = vis_trim.sort_values(by="SortKey").drop(columns=["SortKey"]).head(max_suggestions)

    if vis_trim.empty:
        st.info("Inga tydliga trimningskandidater hittades enligt nuvarande regler.")
    else:
        st.warning("Föreslagna trimningar (heuristik – manuellt omdöme krävs):")
        st.dataframe(vis_trim, use_container_width=True, hide_index=True)

    st.caption("Not: Säljvakt markerar positioner med låg valuation-score, klart högre P/S än historiskt snitt, riskabel utdelning, "
               "och/eller pris långt över vald riktkurs – samt där positionen är större än vald max-andel.")

# app.py — Del 10/?
# --- Pris-only (alla), MAIN, sidopanel, batch-UI -----------------------------

def _update_all_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lättviktskörning: uppdatera ENBART 'Aktuell kurs' för alla tickers via Yahoo.
    Stämplar 'Senast auto-uppdaterad' + källa men rör inga TS_-kolumner (pris ej spårat).
    """
    if df.empty:
        return df
    any_changed = False
    for ridx, row in df.reset_index().iterrows():
        i = row["index"]
        tkr = str(df.at[i, "Ticker"]).strip()
        if not tkr:
            continue
        try:
            y = hamta_yahoo_fält(tkr)
            px = float(y.get("Aktuell kurs", 0.0) or 0.0)
            if px > 0:
                df.at[i, "Aktuell kurs"] = px
                # komplettera namn/valuta om saknas
                if y.get("Bolagsnamn"):
                    df.at[i, "Bolagsnamn"] = y["Bolagsnamn"]
                if y.get("Valuta"):
                    df.at[i, "Valuta"] = y["Valuta"]
                _note_auto_update(df, i, source="Pris-only (Yahoo)")
                any_changed = True
            time.sleep(0.15)  # skonsam
        except Exception:
            pass
    if any_changed:
        spara_data(df, do_snapshot=False)
    return df

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # ---------- Sidopanel: valutakurser ----------
    st.sidebar.header("💱 Valutakurser → SEK")

    # Ladda sparade kurser (från Sheet) som baseline
    saved_rates = las_sparade_valutakurser()
    # Initiera session_state EN gång (innan widgets)
    if "rate_usd" not in st.session_state:
        st.session_state.rate_usd = float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"]))
    if "rate_nok" not in st.session_state:
        st.session_state.rate_nok = float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
    if "rate_cad" not in st.session_state:
        st.session_state.rate_cad = float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
    if "rate_eur" not in st.session_state:
        st.session_state.rate_eur = float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
    if "rates_provider" not in st.session_state:
        st.session_state.rates_provider = ""

    # Knapp för auto-hämtning FÖRE inputs (callback uppdaterar session_state och rerun)
    col_rates_btn, col_rates_info = st.sidebar.columns([1,1])
    with col_rates_btn:
        if st.button("🌐 Hämta automatiskt"):
            auto_rates, misses, provider = hamta_valutakurser_auto()
            st.session_state.rate_usd = float(auto_rates.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(auto_rates.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(auto_rates.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(auto_rates.get("EUR", st.session_state.rate_eur))
            st.session_state.rates_provider = provider
            # Visa info via session + rerun så att widgets visar nya värden
            st.session_state.rates_misses = misses
            st.rerun()
    with col_rates_info:
        if st.session_state.rates_provider:
            st.sidebar.success(f"Källa: {st.session_state.rates_provider}")
            if st.session_state.get("rates_misses"):
                st.sidebar.warning("Missar:\n- " + "\n- ".join(st.session_state["rates_misses"]))

    # Själva inputs (värden kommer från session_state)
    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.01, format="%.4f")

    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    col_rates1, col_rates2 = st.sidebar.columns(2)
    with col_rates1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            # uppdatera baseline för nästa session
            st.sidebar.success("Valutakurser sparade.")
    with col_rates2:
        if st.button("↻ Läs sparade kurser"):
            sr = las_sparade_valutakurser()
            st.session_state.rate_usd = float(sr.get("USD", st.session_state.rate_usd))
            st.session_state.rate_nok = float(sr.get("NOK", st.session_state.rate_nok))
            st.session_state.rate_cad = float(sr.get("CAD", st.session_state.rate_cad))
            st.session_state.rate_eur = float(sr.get("EUR", st.session_state.rate_eur))
            st.success("Läste sparade kurser.")
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # ---------- Läs data ----------
    df = hamta_data()
    # Säkerställ schema, migrera och typer
    if df is None or df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # ---------- Sidopanel: Auto/Batch/Pris ----------
    st.sidebar.subheader("🛠️ Uppdateringar")

    # Pris-only (lätt)
    if st.sidebar.button("🔁 Uppdatera endast kurser (alla)"):
        df = _update_all_prices(df)
        st.sidebar.success("Klar med prisuppdatering.")

    # Tung hel-uppdatering (finns kvar men använd batch helst)
    make_snapshot_all = st.sidebar.checkbox("Snapshot före 'Auto-uppdatera alla'", value=True, key="snap_all")
    if st.sidebar.button("⚙️ Auto-uppdatera alla (tung)"):
        df, log = auto_update_all(df, user_rates, make_snapshot=make_snapshot_all)
        st.session_state["last_auto_log"] = log

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧱 Batch-körning (rekommenderas)")

    # Välj sortläge & chunk-storlek
    batch_mode = st.sidebar.selectbox("Kö-läge", ["A–Ö (bolagsnamn)","Äldst uppdaterade först"], index=0, key="batch_mode_sel")
    chunk = st.sidebar.number_input("Stegstorlek (N bolag)", min_value=1, max_value=50, value=10, step=1, key="batch_chunk")
    make_snapshot = st.sidebar.checkbox("Snapshot före varje batch-steg", value=False, key="batch_snap")

    col_b1, col_b2, col_b3 = st.sidebar.columns(3)
    with col_b1:
        if st.button("Initiera kö"):
            init_batch_queue(df, mode=batch_mode)
    with col_b2:
        if st.button("Kör nästa N"):
            df = run_batch_step(df, user_rates, step_size=int(chunk), make_snapshot=make_snapshot)
    with col_b3:
        if st.button("Återställ kö"):
            for k in ["batch_queue","batch_cursor","batch_mode","batch_log"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.sidebar.info("Batch-kö återställd.")

    # ---------- Meny ----------
    st.sidebar.markdown("---")
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll","Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"], index=0)

    # Vyer
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
