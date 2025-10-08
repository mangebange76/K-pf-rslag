# app.py — DEL 1/5
from __future__ import annotations
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# yfinance (tolerera om saknas)
try:
    import yfinance as yf
    _YF_OK = True
except Exception:
    _YF_OK = False

# ---- Google Sheets (DIN setup: SHEET_URL + GOOGLE_CREDENTIALS) ----
import gspread
from google.oauth2.service_account import Credentials

# =========================
# KONFIG
# =========================
SHEET_URL = st.secrets["SHEET_URL"]
DATA_WS = "Blad1"            # din datatabell
RATES_WS = "Valutakurser"    # dina valutakurser

# SEC API – sätt gärna st.secrets["SEC_UA"] = "FirmName ContactPerson email@domain"
SEC_UA = st.secrets.get("SEC_UA", "StreamlitApp (email: test@example.com)")
SEC_BASE = "https://data.sec.gov"
SEC_HEADERS = {"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}

# Lokal fallback-filer (så appen inte dör om Sheets bråkar)
LOCAL_DATA = Path("local_data.csv")
LOCAL_RATES = Path("local_rates.json")

# Standard valutakurser
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

# =========================
# KOLUMN-SCHEMA (utökat för källor & tidsstämplar)
# =========================
FINAL_COLS = [
    # Bas
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning",
    "Utestående aktier", "Antal aktier",

    # P/S (TTM) och kvartal
    "P/S",
    "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/S Q1 datum", "P/S Q2 datum", "P/S Q3 datum", "P/S Q4 datum",
    "Källa P/S", "Källa P/S Q1", "Källa P/S Q2", "Källa P/S Q3", "Källa P/S Q4",

    # Omsättning & riktkurser
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Derivat/övrigt
    "CAGR 5 år (%)", "P/S-snitt",

    # Tidsstämplar & meta
    "Senast manuellt uppdaterad", "Senast auto uppdaterad",
    "TS P/S", "TS Utestående aktier", "TS Omsättning",
    "Källa Aktuell kurs", "Källa Utestående aktier",
]

# =========================
# HJÄLPARE
# =========================
def now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

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
    if last_err:
        raise last_err

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = [
        "Utestående aktier","Antal aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","P/S-snitt",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    str_cols = [
        "Ticker","Bolagsnamn","Valuta",
        "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
        "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
        "Senast manuellt uppdaterad","Senast auto uppdaterad",
        "TS P/S","TS Utestående aktier","TS Omsättning",
        "Källa Aktuell kurs","Källa Utestående aktier",
    ]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# =========================
# GOOGLE SHEETS (DIN gamla setup)
# =========================
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _open_sheet():
    try:
        return client.open_by_url(SHEET_URL)
    except Exception as e:
        return None

def _get_or_create_ws(sh, title: str, rows: int = 1000, cols: int = 40):
    try:
        return sh.worksheet(title)
    except Exception:
        try:
            return sh.add_worksheet(title=title, rows=str(rows), cols=str(cols))
        except Exception:
            return None

def _ws_data():
    sh = _open_sheet()
    return _get_or_create_ws(sh, DATA_WS) if sh else None

def _ws_rates():
    sh = _open_sheet()
    if not sh:
        return None
    try:
        return sh.worksheet(RATES_WS)
    except Exception:
        try:
            ws = sh.add_worksheet(title=RATES_WS, rows="20", cols="5")
            ws.update([["Valuta","Kurs"]])
            return ws
        except Exception:
            return None

def _read_ws_to_df(ws) -> pd.DataFrame:
    try:
        values = ws.get_all_values()
    except Exception:
        return pd.DataFrame()
    if not values:
        return pd.DataFrame()
    header, *rows = values
    if not header:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=header)

def _write_df_to_ws(ws, df: pd.DataFrame) -> bool:
    try:
        ws.clear()
        if df.empty:
            ws.update("A1", [[""]])
            return True
        values = [list(df.columns)]
        values += df.astype(object).where(pd.notnull(df), "").values.tolist()
        ws.update("A1", values)
        return True
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def hamta_data() -> pd.DataFrame:
    ws = _ws_data()
    if ws:
        df = _read_ws_to_df(ws)
        if not df.empty:
            return df
    # lokal fallback
    if LOCAL_DATA.exists():
        try:
            return pd.read_csv(LOCAL_DATA)
        except Exception:
            pass
    return pd.DataFrame({c: [] for c in FINAL_COLS})

def spara_data(df: pd.DataFrame) -> bool:
    ok = False
    ws = _ws_data()
    if ws:
        ok = _write_df_to_ws(ws, df)
    try:
        df.to_csv(LOCAL_DATA, index=False)
    except Exception:
        pass
    return ok

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser() -> Dict[str, float]:
    ws = _ws_rates()
    if ws:
        df = _read_ws_to_df(ws)
        if not df.empty and set(["Valuta","Kurs"]).issubset(df.columns):
            out = {}
            for _, r in df.iterrows():
                cur = str(r.get("Valuta","")).upper().strip()
                val = str(r.get("Kurs","")).replace(",", ".").strip()
                try:
                    out[cur] = float(val)
                except:
                    pass
            if out:
                out.setdefault("SEK", 1.0)
                return out
    # lokal fallback
    if LOCAL_RATES.exists():
        try:
            return json.loads(LOCAL_RATES.read_text())
        except Exception:
            pass
    return dict(STANDARD_VALUTAKURSER)

def spara_valutakurser(rates: Dict[str, float]) -> bool:
    ok = False
    ws = _ws_rates()
    df = pd.DataFrame([["Valuta","Kurs"]] + [[k, str(v)] for k, v in sorted(rates.items())])
    if ws:
        ok = _write_df_to_ws(ws, pd.DataFrame(df.values[1:], columns=df.values[0]))
    try:
        LOCAL_RATES.write_text(json.dumps(rates))
    except Exception:
        pass
    return ok

# =========================
# VALUTA (live via Yahoo) + hjälpare
# =========================
# Hämtar USD/NOK/CAD/EUR → SEK. Returnerar dict eller None om det misslyckas.
def _hamta_live_fx() -> Optional[Dict[str, float]]:
    if not _YF_OK:
        return None
    pairs = {"USD": "USDSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X", "EUR": "EURSEK=X"}
    try:
        # yfinance kan returnera multiindex om flera tickers
        tickers = " ".join(pairs.values())
        data = yf.download(
            tickers=tickers, period="5d", interval="1d",
            progress=False, group_by="ticker", threads=True,
        )
        out: Dict[str, float] = {}
        for k, ysym in pairs.items():
            try:
                if isinstance(data, pd.DataFrame) and ysym in data.columns.get_level_values(0):
                    px = data[ysym]["Close"].dropna()
                    if not px.empty:
                        out[k] = float(px.iloc[-1])
                else:
                    # fallback om bara en serie kom tillbaka
                    px = data["Close"].dropna()
                    if not px.empty:
                        out[k] = float(px.iloc[-1])
            except Exception:
                pass
        if out:
            out["SEK"] = 1.0
            return out
    except Exception:
        return None
    return None

def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    if not valuta:
        return 1.0
    return float(user_rates.get(valuta.upper(), 1.0))


# =========================
# ETIKETT-HJÄLPARE (för visning i formulär)
# =========================
def _lbl_with_ts(base: str, df_row: pd.Series, typ: str) -> str:
    """
    typ: 'man'  → lägg till [Senast manuellt uppdaterad]
         'auto' → lägg till [Senast auto uppdaterad]
    """
    if df_row is None or df_row.empty:
        return base
    if typ == "man":
        ts = df_row.get("Senast manuellt uppdaterad", "") or "—"
        return f"{base}  [{ts}]"
    if typ == "auto":
        ts = df_row.get("Senast auto uppdaterad", "") or "—"
        return f"{base}  [{ts}]"
    return base

def _psq_label(row: pd.Series, q: int) -> str:
    """Visuell etikett för P/S QX med datum och ev. källa."""
    d = str(row.get(f"P/S Q{q} datum", "") or "—")
    src = str(row.get(f"Källa P/S Q{q}", "") or "")
    src_part = f" ({src})" if src else ""
    return f"P/S Q{q} — {d}{src_part}"

def _ps_ttm_label(row: pd.Series) -> str:
    ts = str(row.get("TS P/S", "") or "")
    src = str(row.get("Källa P/S", "") or "Yahoo/ps_ttm")
    if ts:
        return f"P/S (TTM) [{ts}] ({src})"
    return f"P/S (TTM) ({src})"

def _shares_label(row: pd.Series) -> str:
    ts = str(row.get("TS Utestående aktier", "") or "")
    src = str(row.get("Källa Utestående aktier", "") or "Yahoo/info")
    base = "Utestående aktier (miljoner)"
    if ts and src: return f"{base} [{ts}] ({src})"
    if ts:         return f"{base} [{ts}]"
    if src:        return f"{base} ({src})"
    return base


# =========================
# BERÄKNINGSMOTOR
# =========================
def uppdatera_berakningar(df: pd.DataFrame, user_rates: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    - P/S-snitt: medel av positiva Q1–Q4
    - CAGR clamp: >100% → 50%, <0% → 2%
    - Omsättning om 2 & 3 år: från 'Omsättning nästa år' m.h.a. CAGR
    - Riktkurs per aktie: (Omsättning_x * P/S_use) / Utestående aktier
      * Omsättning_* antas vara i miljoner
      * Utestående aktier är i miljoner → enhetsmässigt blir resultatet i 'valutaper aktie'
    """
    df = df.copy()

    for i, rad in df.iterrows():
        # 1) P/S-snitt
        ps_vals = [
            float(rad.get("P/S Q1", 0.0)),
            float(rad.get("P/S Q2", 0.0)),
            float(rad.get("P/S Q3", 0.0)),
            float(rad.get("P/S Q4", 0.0)),
        ]
        ps_clean = [x for x in ps_vals if x > 0]
        ps_snitt = round(float(np.mean(ps_clean)) if ps_clean else 0.0, 2)
        df.at[i, "P/S-snitt"] = ps_snitt

        # 2) CAGR clamp
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # 3) Omsättning om 2 & 3 år (från "nästa år")
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev. existerande
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # 4) Riktkurser
        ps_use = ps_snitt if ps_snitt > 0 else float(rad.get("P/S", 0.0))
        aktier_ut_mn = float(rad.get("Utestående aktier", 0.0))

        if aktier_ut_mn > 0 and ps_use > 0:
            # alla omsättningsfält i miljoner → pris/aktie = (M * P/S) / (M aktier)
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_use) / aktier_ut_mn, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_use) / aktier_ut_mn, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_use) / aktier_ut_mn, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_use) / aktier_ut_mn, 2)
        else:
            for k in ("Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"):
                df.at[i, k] = 0.0

    return df

# =========================
# YAHOO (primär) – kvartalsintäkter, aktier & pris per kvartal
# =========================
def _yahoo_quarter_series(tkr: "yf.Ticker") -> Tuple[List[pd.Timestamp], List[float]]:
    """
    Returnerar (quarter_end_dates, quarterly_revenues) från yfinance.
    Använder t.quarterly_income_stmt['Total Revenue'].
    """
    try:
        qis = tkr.quarterly_income_stmt  # DataFrame: index = rows, columns = period-end timestamps
        if isinstance(qis, pd.DataFrame) and "Total Revenue" in qis.index and len(qis.columns) >= 1:
            ser = qis.loc["Total Revenue"].dropna()
            # yfinance returnerar ofta kolumner som Timestamps (period-end)
            # Sortera kronologiskt
            ser = ser.sort_index()
            dates = list(ser.index.to_pydatetime())
            vals = [float(v) for v in ser.values]
            return dates, vals
    except Exception:
        pass
    return [], []


def _yahoo_shares_series(tkr: "yf.Ticker") -> pd.DataFrame:
    """
    Hämtar historik över utestående aktier via yfinance (get_shares_full).
    Returnerar DataFrame med kolumner ['Date','Shares'] sorterad i tid.
    """
    try:
        s = tkr.get_shares_full(period="max")
        if isinstance(s, pd.Series) and not s.empty:
            df = s.reset_index()
            df.columns = ["Date", "Shares"]
            df = df.sort_values("Date").reset_index(drop=True)
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["Date","Shares"])


def _yahoo_price_on_or_after(tkr: "yf.Ticker", end_dt: datetime) -> Optional[float]:
    """
    Hämtar stängningskurs på eller strax efter kvartalsslut (nästa handelsdag).
    """
    try:
        start = (pd.Timestamp(end_dt) - pd.Timedelta(days=2)).to_pydatetime()
        stop  = (pd.Timestamp(end_dt) + pd.Timedelta(days=7)).to_pydatetime()
        h = tkr.history(start=start, end=stop, auto_adjust=False)
        if isinstance(h, pd.DataFrame) and not h.empty:
            # Ta första close på eller efter end_dt
            h = h.reset_index()
            h["Date"] = pd.to_datetime(h["Date"])
            cand = h[h["Date"] >= pd.Timestamp(end_dt)]
            if not cand.empty and "Close" in cand:
                return float(cand.iloc[0]["Close"])
            # annars ta sista innan
            if "Close" in h:
                return float(h.iloc[-1]["Close"])
    except Exception:
        pass
    return None


def _nearest_shares_before(sh_df: pd.DataFrame, when: datetime) -> Optional[float]:
    """Välj närmast datum <= when och returnera Shares (float)."""
    if sh_df is None or sh_df.empty:
        return None
    try:
        sh_df = sh_df.copy()
        sh_df["Date"] = pd.to_datetime(sh_df["Date"])
        cand = sh_df[sh_df["Date"] <= pd.Timestamp(when)]
        if not cand.empty:
            return float(cand.iloc[-1]["Shares"])
        # annars minsta efter
        cand2 = sh_df[sh_df["Date"] > pd.Timestamp(when)]
        if not cand2.empty:
            return float(cand2.iloc[0]["Shares"])
    except Exception:
        return None
    return None


def _rolling_ttm(quarter_vals: List[float], idx: int) -> Optional[float]:
    """
    TTM som summa av 4 senaste kvartalsvärden upp till och med index idx.
    Kräver idx >= 3.
    """
    if idx < 3:
        return None
    try:
        window = quarter_vals[idx-3:idx+1]
        if any(v is None for v in window):
            return None
        return float(sum(window))
    except Exception:
        return None


def yahoo_quarterly_ps(ticker: str) -> Dict[str, dict]:
    """
    Beräknar PS per kvartal med yfinance-data (primär källa).
    Returnerar en dict: { "items": [ {end, ps, ttm_rev, shares, price} ... ], "source": "YahooCalc" }
    """
    out = {"items": [], "source": "YahooCalc"}
    if not _YF_OK:
        return out

    try:
        t = yf.Ticker(ticker)
        q_dates, q_revs = _yahoo_quarter_series(t)
        if not q_dates or not q_revs:
            return out

        sh_df = _yahoo_shares_series(t)

        # Bygg lista av (end, ttm_rev, shares, price, ps)
        for ix in range(len(q_dates)):
            ttm_rev = _rolling_ttm(q_revs, ix)
            if ttm_rev is None or ttm_rev <= 0:
                continue
            end_dt = pd.Timestamp(q_dates[ix]).to_pydatetime()

            shares = _nearest_shares_before(sh_df, end_dt)
            if shares is None or shares <= 0:
                continue

            price = _yahoo_price_on_or_after(t, end_dt)
            if price is None or price <= 0:
                continue

            mcap = price * shares
            ps_val = float(mcap) / float(ttm_rev) if ttm_rev else None
            if ps_val is None or ps_val <= 0:
                continue

            out["items"].append({
                "end": end_dt.strftime("%Y-%m-%d"),
                "ttm_rev": float(ttm_rev),
                "shares": float(shares),
                "price": float(price),
                "ps": float(ps_val),
                "src": "YahooCalc",
            })

        # sortera senaste först
        out["items"].sort(key=lambda x: x["end"], reverse=True)
        return out
    except Exception:
        return out


# =========================
# SEC (fallback) – kvartalsintäkter & aktier (pris via Yahoo)
# =========================
def _http_get_json(url: str, params: Optional[dict] = None) -> Optional[dict]:
    try:
        r = requests.get(url, headers=SEC_HEADERS, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _sec_cik_lookup() -> Dict[str, str]:
    """
    Hämtar mapping TICKER -> CIK (str utan ledande nollor) från SEC.
    Cache: ja.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    js = _http_get_json(url)
    out = {}
    if isinstance(js, dict):
        # struktur: {"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."}, ...}
        try:
            for _, row in js.items():
                tkr = str(row.get("ticker","")).upper().strip()
                cik = str(row.get("cik_str","")).strip()
                if tkr and cik:
                    out[tkr] = cik
        except Exception:
            pass
    return out

def _sec_pad_cik(cik: str) -> str:
    s = str(cik).strip()
    return s.zfill(10)

def _sec_companyfacts(cik: str) -> Optional[dict]:
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{_sec_pad_cik(cik)}.json"
    return _http_get_json(url)

def _pick_best_revenue_tag(facts: dict) -> Optional[Tuple[str, List[dict]]]:
    """
    Välj bästa intäkts-tag i facts (units['USD']) i prioriteringsordning.
    Returnerar (tagname, series_list)
    """
    if not facts or "facts" not in facts:
        return None
    cand_tags = [
        "us-gaap/SalesRevenueNet",
        "us-gaap/RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap/Revenues",
    ]
    for tag in cand_tags:
        node = facts["facts"]
        for p in tag.split("/"):
            if p in node:
                node = node[p]
            else:
                node = None
                break
        if node and "units" in node:
            units = node["units"]
            # välj USD om möjligt
            if "USD" in units and isinstance(units["USD"], list) and len(units["USD"]) > 0:
                return tag, units["USD"]
            # ibland är det "USDm" etc; ta första nyckel
            for k, lst in units.items():
                if isinstance(lst, list) and len(lst) > 0:
                    return tag, lst
    return None

def _derive_quarter_values_from_sec(series: List[dict]) -> List[Tuple[datetime, float]]:
    """
    Försök härleda kvartalsintäkter från SEC-facts:
    - Om vi hittar per-kvartal värden (frame innehåller 'Q' utan 'YTD') → använd direkt
    - Annars differens av YTD (Q2YTD - Q1YTD etc).
    Returnerar lista [(end_dt, quarter_value)] kronologiskt.
    """
    rows = []
    for it in series:
        try:
            end = pd.to_datetime(it.get("end"))
            val = float(it.get("val"))
            fp = str(it.get("fp",""))
            form = str(it.get("form",""))
            frame = str(it.get("frame","") or "")
            if not np.isfinite(val) or val <= 0:
                continue
            if fp not in ("Q1","Q2","Q3","Q4"):
                continue
            if form not in ("10-Q","10-K","10-Q/A","10-K/A"):
                continue
            rows.append({
                "end": end,
                "val": val,
                "fp": fp,
                "form": form,
                "frame": frame,
                "is_ytd": ("YTD" in frame.upper()),
            })
        except Exception:
            continue
    if not rows:
        return []

    df = pd.DataFrame(rows).sort_values("end").reset_index(drop=True)

    # Försök hitta "direkta" kvartalsvärden (inte YTD)
    direct = df[~df["is_ytd"]].copy()
    if not direct.empty:
        out = [(r["end"].to_pydatetime(), float(r["val"])) for _, r in direct.iterrows()]
        return out

    # Annars differens av YTD för samma FY
    ytd = df[df["is_ytd"]].copy()
    if ytd.empty:
        return []
    # Grupp efter FY (härled från datumens år)
    ytd["year"] = ytd["end"].dt.year
    out_pairs: List[Tuple[datetime, float]] = []
    by_year = ytd.groupby("year")
    for _, g in by_year:
        g = g.sort_values("end")
        prev_val = None
        for _, r in g.iterrows():
            curr = float(r["val"])
            if prev_val is None:
                q_val = curr  # Q1 YTD antas ~ Q1
            else:
                q_val = max(0.0, curr - prev_val)
            out_pairs.append((r["end"].to_pydatetime(), float(q_val)))
            prev_val = curr
    out_pairs.sort(key=lambda x: x[0])
    return out_pairs


def _sec_quarterly_revenue_and_shares(ticker: str) -> Tuple[List[datetime], List[float], List[Tuple[datetime,float]]]:
    """
    Hämtar (quarter_end_dates, quarterly_revenues, shares_points) där shares_points är lista av (date, shares).
    shares hämtas från 'dei/EntityCommonStockSharesOutstanding' om möjligt, annars tom.
    """
    tkr = str(ticker).upper().strip()
    mapping = _sec_cik_lookup()
    cik = mapping.get(tkr)
    if not cik:
        return [], [], []

    facts = _sec_companyfacts(cik)
    if not facts:
        return [], [], []

    # Revenues
    tag = _pick_best_revenue_tag(facts)
    q_dates: List[datetime] = []
    q_vals: List[float] = []
    if tag:
        _, series = tag
        pairs = _derive_quarter_values_from_sec(series)
        if pairs:
            q_dates = [p[0] for p in pairs]
            q_vals  = [p[1] for p in pairs]

    # Shares (instantaneous)
    shares_pairs: List[Tuple[datetime, float]] = []
    try:
        dei = facts["facts"].get("dei", {})
        node = dei.get("EntityCommonStockSharesOutstanding", {})
        units = node.get("units", {})
        # välj 'shares' enhet
        candidates = None
        if "shares" in units:
            candidates = units["shares"]
        elif len(units) > 0:
            # ta första listan
            candidates = list(units.values())[0]
        if isinstance(candidates, list):
            rows = []
            for it in candidates:
                try:
                    end = pd.to_datetime(it.get("end"))
                    val = float(it.get("val"))
                    form = str(it.get("form",""))
                    if form not in ("10-Q","10-K","10-Q/A","10-K/A"):
                        continue
                    if val <= 0:
                        continue
                    rows.append((end.to_pydatetime(), float(val)))
                except Exception:
                    continue
            rows.sort(key=lambda x: x[0])
            shares_pairs = rows
    except Exception:
        pass

    return q_dates, q_vals, shares_pairs


def _nearest_sec_shares_before(sh_points: List[Tuple[datetime,float]], when: datetime) -> Optional[float]:
    if not sh_points:
        return None
    # välj senaste <= when
    prev = None
    for dt, val in sh_points:
        if dt <= when:
            prev = (dt, val)
        else:
            break
    if prev:
        return float(prev[1])
    # annars första efter
    return float(sh_points[0][1]) if sh_points else None


def sec_quarterly_ps(ticker: str) -> Dict[str, dict]:
    """
    Beräknar PS per kvartal med SEC-intäkter + SEC-aktier (pris via Yahoo).
    Returnerar {"items":[{end, ttm_rev, shares, price, ps, src="SECCalc"}], "source":"SECCalc"}.
    """
    out = {"items": [], "source": "SECCalc"}
    q_dates, q_revs, sh_pts = _sec_quarterly_revenue_and_shares(ticker)
    if not q_dates or not q_revs:
        return out
    # pris via yahoo
    if not _YF_OK:
        return out
    t = yf.Ticker(ticker)

    # rulla TTM över kvartal
    for ix in range(len(q_dates)):
        ttm_rev = _rolling_ttm(q_revs, ix)
        if ttm_rev is None or ttm_rev <= 0:
            continue
        end_dt = q_dates[ix]

        shares = _nearest_sec_shares_before(sh_pts, end_dt)
        if shares is None or shares <= 0:
            continue

        price = _yahoo_price_on_or_after(t, end_dt)
        if price is None or price <= 0:
            continue

        mcap = price * shares
        ps_val = float(mcap) / float(ttm_rev)
        if ps_val <= 0:
            continue

        out["items"].append({
            "end": pd.Timestamp(end_dt).strftime("%Y-%m-%d"),
            "ttm_rev": float(ttm_rev),
            "shares": float(shares),
            "price": float(price),
            "ps": float(ps_val),
            "src": "SECCalc",
        })
    out["items"].sort(key=lambda x: x["end"], reverse=True)
    return out


# =========================
# SAMMANSLAGNING & UTDATA TILL APPEN
# =========================
def hamta_ps_kvartal_auto(ticker: str) -> Dict[str, str | float]:
    """
    Kombinerar Yahoo (primär) + SEC (fallback/komplement) och fyller:
      P/S Q1..Q4, P/S Qx datum, Källa P/S Qx
    Q1 = senaste, Q2 = näst senaste, osv.
    Returnerar dict med fälten att skriva in i DF-raden.
    """
    res = {"meta_source": ""}

    y = yahoo_quarterly_ps(ticker)
    items = list(y.get("items", []))
    src_y = y.get("source", "YahooCalc")

    if len(items) < 4:
        s = sec_quarterly_ps(ticker)
        sec_items = list(s.get("items", []))
        # slå ihop per end-datum (Yahoo prioriteras)
        by_end = {}
        for it in sec_items:
            by_end[it["end"]] = it
        for it in items:
            by_end[it["end"]] = it  # overwrite with Yahoo
        merged = list(by_end.values())
        merged.sort(key=lambda x: x["end"], reverse=True)
        items = merged
        res["meta_source"] = f"{src_y}+SECCalc"
    else:
        res["meta_source"] = src_y

    # mappa till Q1..Q4
    out_fields: Dict[str, str | float] = {}
    for qi, it in enumerate(items[:4], start=1):
        out_fields[f"P/S Q{qi}"] = round(float(it["ps"]), 3)
        out_fields[f"P/S Q{qi} datum"] = str(it["end"])
        out_fields[f"Källa P/S Q{qi}"] = str(it.get("src","?"))

    return out_fields

# =========================
# YAHOO – basfält (namn, kurs, valuta, utdelning, aktier, P/S TTM, CAGR)
# =========================
def _cagr_from_financials(tkr: "yf.Ticker") -> float:
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
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Hämtar robusta basfält från Yahoo. Om yfinance saknas returneras tomma/förval.
    Fält: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, Utestående aktier (miljoner),
          P/S (TTM), CAGR 5 år (%), källor/tidsstämplar.
    """
    out = {
        "Bolagsnamn": "", "Valuta": "USD", "Aktuell kurs": 0.0,
        "Årlig utdelning": 0.0, "Utestående aktier": 0.0,
        "P/S": 0.0, "CAGR 5 år (%)": 0.0,
        "Källa Aktuell kurs": "Yahoo/info", "Källa Utestående aktier": "Yahoo/info",
        "Källa P/S": "Yahoo/ps_ttm",
        "TS Utestående aktier": now_stamp(), "TS P/S": now_stamp(),
    }
    if not _YF_OK:
        return out
    try:
        t = yf.Ticker(ticker)

        # info
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # pris
        pris = info.get("regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if isinstance(pris, (int,float)):
            out["Aktuell kurs"] = float(pris)

        # valuta & namn
        val = info.get("currency")
        if val: out["Valuta"] = str(val).upper()
        namn = info.get("shortName") or info.get("longName") or ""
        if namn: out["Bolagsnamn"] = str(namn)

        # utdelning
        div_rate = info.get("dividendRate")
        if isinstance(div_rate, (int,float)): out["Årlig utdelning"] = float(div_rate)

        # aktier (till miljoner)
        shares = info.get("sharesOutstanding")
        if isinstance(shares, (int,float)) and shares > 0:
            out["Utestående aktier"] = float(shares) / 1e6

        # P/S TTM
        ps_ttm = info.get("priceToSalesTrailing12Months")
        if isinstance(ps_ttm, (int,float)) and ps_ttm > 0:
            out["P/S"] = float(ps_ttm)

        # CAGR
        out["CAGR 5 år (%)"] = _cagr_from_financials(t)

    except Exception:
        pass
    return out


# =========================
# SIDOPANEL – valutakurser (sparning, läsning, live-knapp)
# =========================
def valutakurser_sidebar() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")
    saved = las_sparade_valutakurser()

    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, format="%.4f")
    rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("💾 Spara"):
            spara_valutakurser(rates)
            st.sidebar.success("Valutakurser sparade.")
            st.session_state["recalc_after_rates"] = True
    with c2:
        if st.button("↻ Läs"):
            st.cache_data.clear()
            st.rerun()
    with c3:
        if st.button("🌐 Live"):
            live = _hamta_live_fx()
            if live:
                spara_valutakurser(live)
                st.sidebar.success("Live-kurser hämtade & sparade.")
                st.session_state["recalc_after_rates"] = True
                st.rerun()
            else:
                st.sidebar.warning("Kunde inte hämta live-kurser just nu.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return rates


# =========================
# MASSUPPDATERA – Yahoo basfält + P/S Q1–Q4 (Yahoo+SEC)
# =========================
def massuppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla (Yahoo + SEC)"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        total = len(df)
        for i, row in df.iterrows():
            tkr = str(row.get("Ticker","")).strip()
            if not tkr:
                continue
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")

            # 1) Basfält från Yahoo
            base = hamta_yahoo_fält(tkr)
            for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","P/S","Källa P/S"]:
                if k in base: df.at[i, k] = base[k]
            if float(base.get("Utestående aktier",0) or 0) > 0:
                df.at[i, "Utestående aktier"] = float(base["Utestående aktier"])
            for k in ["TS Utestående aktier","TS P/S","Källa Aktuell kurs","Källa Utestående aktier"]:
                if k in base: df.at[i, k] = base[k]

            # 2) P/S per kvartal (Q1..Q4) via Yahoo+SEC
            q = hamta_ps_kvartal_auto(tkr)
            for qn in (1,2,3,4):
                df.at[i, f"P/S Q{qn}"] = float(q.get(f"P/S Q{qn}", 0.0) or 0.0)
                df.at[i, f"P/S Q{qn} datum"] = q.get(f"P/S Q{qn} datum", "")
                df.at[i, f"Källa P/S Q{qn}"] = q.get(f"Källa P/S Q{qn}", "")

            df.at[i, "Senast auto uppdaterad"] = now_stamp()

            time.sleep(0.2)
            bar.progress((i+1)/max(1,total))

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
    return df


# =========================
# LÄGG TILL / UPPDATERA (formulär)
# =========================
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum","Bolagsnamn"])
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

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input(_shares_label(bef), value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps_ttm = st.number_input(_ps_ttm_label(bef), value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, key="ps_ttm")
            ps1 = st.number_input(_psq_label(bef, 1), value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, key="ps_q1")
            ps2 = st.number_input(_psq_label(bef, 2), value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, key="ps_q2")
        with c2:
            ps3 = st.number_input(_psq_label(bef, 3), value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, key="ps_q3")
            ps4 = st.number_input(_psq_label(bef, 4), value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, key="ps_q4")

            oms_idag  = st.number_input(_lbl_with_ts("Omsättning idag (miljoner)", bef, "man"),
                                        value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input(_lbl_with_ts("Omsättning nästa år (miljoner)", bef, "man"),
                                        value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.caption("Senast manuellt uppdaterad: " + (bef.get("Senast manuellt uppdaterad","") or "—"))
            st.caption("Senast auto uppdaterad: " + (bef.get("Senast auto uppdaterad","") or "—"))

        spar = st.form_submit_button("💾 Spara & hämta (Yahoo + SEC)")

    if spar and ticker:
        # 1) Spara manuella fält
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps_ttm, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # 2) Hämta basfält via Yahoo
        base = hamta_yahoo_fält(ticker)
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","P/S","Källa P/S","Källa Aktuell kurs","Källa Utestående aktier","TS Utestående aktier","TS P/S"]:
            if k in base:
                df.loc[df["Ticker"]==ticker, k] = base[k]
        if float(base.get("Utestående aktier",0) or 0) > 0:
            df.loc[df["Ticker"]==ticker, "Utestående aktier"] = float(base["Utestående aktier"])

        # 3) Auto P/S Q1..Q4 (Yahoo+SEC)
        q = hamta_ps_kvartal_auto(ticker)
        for qn in (1,2,3,4):
            df.loc[df["Ticker"]==ticker, f"P/S Q{qn}"] = float(q.get(f"P/S Q{qn}", 0.0) or 0.0)
            df.loc[df["Ticker"]==ticker, f"P/S Q{qn} datum"] = q.get(f"P/S Q{qn} datum", "")
            df.loc[df["Ticker"]==ticker, f"Källa P/S Q{qn}"] = q.get(f"Källa P/S Q{qn}", "")

        df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = now_stamp()

        # 4) Beräkna & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat (Yahoo + SEC).")
        st.rerun()

    # Snabbknapp: hämta igen
    if not bef.empty and st.button("↻ Hämta igen denna ticker (Yahoo + SEC)"):
        tkr = bef.get("Ticker","").strip()
        if tkr:
            base = hamta_yahoo_fält(tkr)
            for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","P/S","Källa P/S","Källa Aktuell kurs","Källa Utestående aktier","TS Utestående aktier","TS P/S"]:
                if k in base:
                    df.loc[df["Ticker"]==tkr, k] = base[k]
            if float(base.get("Utestående aktier",0) or 0) > 0:
                df.loc[df["Ticker"]==tkr, "Utestående aktier"] = float(base["Utestående aktier"])

            q = hamta_ps_kvartal_auto(tkr)
            for qn in (1,2,3,4):
                df.loc[df["Ticker"]==tkr, f"P/S Q{qn}"] = float(q.get(f"P/S Q{qn}", 0.0) or 0.0)
                df.loc[df["Ticker"]==tkr, f"P/S Q{qn} datum"] = q.get(f"P/S Q{qn} datum", "")
                df.loc[df["Ticker"]==tkr, f"Källa P/S Q{qn}"] = q.get(f"Källa P/S Q{qn}", "")

            df.loc[df["Ticker"]==tkr, "Senast auto uppdaterad"] = now_stamp()
            df = uppdatera_berakningar(df, user_rates)
            spara_data(df)
            st.success("Hämtat igen.")
            st.rerun()

    # Lista: äldst manuellt uppdaterade (fokus på omsättning)
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )

    return df


# =========================
# VYER: Analys, Portfölj, Investeringsförslag
# =========================
def analysvy(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
                                                  value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")

    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
                "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                "Senast manuellt uppdaterad","Senast auto uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)


def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
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
        use_container_width=True
    )


def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
                                index=1)

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
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

    # Portföljandelar
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
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 3 år" else ""}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %
"""
    )

# =========================
# MAIN / APP-FLÖDE
# =========================
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Sidopanel – valutakurser (med live-knapp)
    user_rates = valutakurser_sidebar()

    # 2) Läs data från Google Sheets (med lokal fallback)
    df = hamta_data()
    if df.empty:
        # initiera tomt ark med rätt schema
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # 3) Säkerställ schema/typer (ifall arket saknar nya kolumner)
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # 4) Om valutakurser nyss ändrats/hämtats → räkna om & spara
    if st.session_state.get("recalc_after_rates", False):
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.session_state["recalc_after_rates"] = False

    # 5) Global massuppdatering (Yahoo basfält + P/S Q1..Q4 via Yahoo+SEC)
    df = massuppdatera(df, user_rates)

    # 6) Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Analys":
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
