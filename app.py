# app.py  —  ENDA filen
from __future__ import annotations

import os
import io
import json
import time
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ---------------------------- UI & STATE ----------------------------

st.set_page_config(page_title="Aktieanalys & P/S", layout="wide")

# Globalt offline-läge (blockar nätanrop snyggt)
if "offline_mode" not in st.session_state:
    st.session_state["offline_mode"] = False

# ---------------------------- Hjälpare ----------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

def _round2(x: float) -> float:
    try:
        return round(float(x), 2)
    except Exception:
        return 0.0

def _to_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return float(str(x).replace(",", "."))
    except Exception:
        return 0.0

def _headers_yahoo() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Connection": "close",
    }

def _headers_sec() -> Dict[str, str]:
    # Sätt gärna din mail i User-Agent om du har — SEC vill ha kontaktmöjlighet
    return {
        "User-Agent": "ps-analyzer/1.0 (contact: you@example.com)",
        "Accept": "application/json",
        "Connection": "close",
    }

@st.cache_data(show_spinner=False, ttl=60)
def _get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 12) -> Dict[str, Any]:
    if st.session_state.get("offline_mode", False):
        raise RuntimeError("Offline-läge är aktivt; blockar nätanrop.")
    r = requests.get(url, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _safe_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, Any], str]:
    try:
        js = _get_json(url, headers=headers)
        return True, js, ""
    except Exception as e:
        return False, {}, f"{type(e).__name__}: {e}"

# ---------------------------- Google Sheets + lokal fallback ----------------------------

@st.cache_data(show_spinner=False)
def _open_gs() -> Tuple[Any, str, str]:
    """Returnerar (worksheet, mode, info).
    mode: 'gsheets' eller 'csv'
    """
    # 1) Försök med service account i secrets
    try:
        import gspread
        sa_dict = st.secrets.get("gsheets", {}).get("service_account", None)
        sheet_url = st.secrets.get("gsheets", {}).get("spreadsheet_url", os.environ.get("SHEET_URL", ""))
        ws_name  = st.secrets.get("gsheets", {}).get("worksheet_name", os.environ.get("SHEET_NAME", "Aktier"))
        if sa_dict and sheet_url:
            gc = gspread.service_account_from_dict(sa_dict)
            sh = gc.open_by_url(sheet_url)
            ws = sh.worksheet(ws_name)
            return ws, "gsheets", f"{sheet_url}::{ws_name}"
    except Exception as e:
        # fortsätt till fallback
        pass

    # 2) CSV fallback
    return None, "csv", "data.csv"

def hamta_data() -> pd.DataFrame:
    ws, mode, info = _open_gs()
    if mode == "gsheets":
        try:
            recs = ws.get_all_records()
            df = pd.DataFrame(recs)
            return df
        except Exception as e:
            st.warning(f"⚠️ Kunde inte läsa Google Sheets ({e}). Använder lokal 'data.csv' om den finns.")
    # CSV fallback
    p = "data.csv"
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    # tomt schema
    return pd.DataFrame()

def spara_data(df: pd.DataFrame) -> None:
    ws, mode, info = _open_gs()
    if mode == "gsheets":
        try:
            # Säker: skriv rubriker + data
            body = [list(df.columns)] + df.astype(object).fillna("").values.tolist()
            ws.clear()
            ws.update("A1", body)
            return
        except Exception as e:
            st.error(f"❌ Kunde inte skriva till Google Sheets ({e}). Sparar till lokal 'data.csv'.")
    # CSV fallback
    df.to_csv("data.csv", index=False)

# ---------------------------- Valutakurser ----------------------------

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser() -> Dict[str, float]:
    p = "fx.json"
    if os.path.exists(p):
        try:
            return json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"USD": 10.0, "NOK": 1.0, "CAD": 7.5, "EUR": 11.0, "SEK": 1.0}

def spara_valutakurser(d: Dict[str, float]) -> None:
    json.dump(d, open("fx.json", "w", encoding="utf-8"))

def hamta_valutakurs(valuta: str, user_rates: Dict[str, float]) -> float:
    v = (valuta or "SEK").upper()
    return float(user_rates.get(v, 1.0))

@st.cache_data(show_spinner=False, ttl=3600)
def _fx_from_yahoo() -> Dict[str, float]:
    # Hämtar USDSEK, NOKSEK, CADSEK, EURSEK via Yahoo finance 'chart' pris
    pairs = {
        "USD": "SEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }
    out = {"SEK": 1.0}
    for code, tick in pairs.items():
        ok, js, err = _safe_get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{tick}?interval=1d&range=5d",
            headers=_headers_yahoo()
        )
        if not ok:
            continue
        try:
            close = js["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            vals = [x for x in close if isinstance(x, (int, float)) and x]
            if vals:
                out[code] = float(vals[-1])
        except Exception:
            pass
    # backfills
    base = las_sparade_valutakurser()
    for k in ["USD","NOK","CAD","EUR"]:
        out[k] = float(out.get(k, base.get(k, 1.0)))
    return out

def hamta_valutakurser_sidebar() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")
    off = st.sidebar.toggle(
        "🔌 Säkerhetsläge (offline)",
        value=st.session_state.get("offline_mode", False),
        help="Stoppar alla nätanrop (Yahoo/SEC/FX). Använd sparade värden."
    )
    st.session_state["offline_mode"] = bool(off)

    saved = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD", 10.0)), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK", 1.0)), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD", 7.5)), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR", 11.0)), step=0.01, format="%.4f")
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            st.sidebar.success("Valutakurser sparade.")
    with c2:
        if st.button("🌐 Live-kurser"):
            if st.session_state.get("offline_mode", False):
                st.sidebar.warning("Offline-läge är på. Stäng av för att hämta live-kurser.")
            else:
                live = _fx_from_yahoo()
                if live:
                    spara_valutakurser(live)
                    st.sidebar.success("Live-kurser hämtade & sparade.")
                    st.rerun()
                else:
                    st.sidebar.error("Kunde inte hämta live-kurser just nu. Behåller sparade värden.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Sheets/CSV"):
        st.cache_data.clear()
        st.rerun()

    return user_rates

# ---------------------------- Yahoo & SEC ----------------------------

def _unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

@st.cache_data(show_spinner=False, ttl=3600)
def yahoo_quote_summary(ticker: str) -> Dict[str, Any]:
    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=price,summaryDetail,defaultKeyStatistics,financialData,assetProfile,calendarEvents"
    ok, js, err = _safe_get_json(url, headers=_headers_yahoo())
    if not ok:
        raise RuntimeError(err)
    try:
        return js["quoteSummary"]["result"][0]
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=1800)
def yahoo_timeseries_revenue(ticker: str) -> Dict[str, Any]:
    # quarterly & trailing revenue
    url = f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}?type=quarterlyTotalRevenue,trailingTotalRevenue&merge=false"
    ok, js, err = _safe_get_json(url, headers=_headers_yahoo())
    if not ok:
        raise RuntimeError(err)
    return js

@st.cache_data(show_spinner=False, ttl=3600)
def yahoo_price_on(ticker: str, date: datetime) -> float:
    # hämtar stängning på (date) eller närmaste efter
    p1 = _unix(date - timedelta(days=2))
    p2 = _unix(date + timedelta(days=3))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={p1}&period2={p2}&interval=1d"
    ok, js, err = _safe_get_json(url, headers=_headers_yahoo())
    if not ok:
        raise RuntimeError(err)
    try:
        closes = js["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        ts = js["chart"]["result"][0]["timestamp"]
        pairs = [(datetime.fromtimestamp(t, tz=timezone.utc).date(), c) for t, c in zip(ts, closes)]
        pairs = [(d, c) if c is not None else (d, 0.0) for d, c in pairs]
        pairs = [p for p in pairs if p[1] and p[1] > 0]
        if not pairs:
            return 0.0
        # ta första >= date.date()
        for d, c in pairs:
            if d >= date.date():
                return float(c)
        return float(pairs[-1][1])
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False, ttl=86400)
def sec_company_tickers() -> Dict[str, Any]:
    ok, js, err = _safe_get_json("https://www.sec.gov/files/company_tickers.json", headers=_headers_sec())
    if not ok:
        raise RuntimeError(err)
    # indexerad dict {lower_ticker: CIK}
    out = {}
    try:
        for _, rec in js.items():
            out[str(rec["ticker"]).upper()] = int(rec["cik_str"])
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False, ttl=86400)
def sec_company_facts(cik: int) -> Dict[str, Any]:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    ok, js, err = _safe_get_json(url, headers=_headers_sec())
    if not ok:
        raise RuntimeError(err)
    return js

def sec_shares_series(cik: int) -> List[Tuple[datetime, float]]:
    """Returnerar lista (datum, antal aktier i miljoner)."""
    js = sec_company_facts(cik)
    out: List[Tuple[datetime, float]] = []
    try:
        facts = js["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]
        # Välj först bästa enhet (aktier) – förekommer "shares" / "SHRS"
        unit = facts.get("shares") or facts.get("SHRS") or list(facts.values())[0]
        for item in unit:
            d = datetime.fromisoformat(item["end"] + "T00:00:00")
            val = _to_float(item.get("val"))
            if val and val > 0:
                out.append((d, val / 1_000_000.0))  # -> miljoner
    except Exception:
        pass
    out.sort(key=lambda x: x[0])
    return out

def _nearest_value(series: List[Tuple[datetime, float]], target: datetime, max_days: int = 20) -> float:
    best = None
    bestdiff = 9999
    for d, v in series:
        diff = abs((d - target).days)
        if diff < bestdiff:
            best = v
            bestdiff = diff
    if best is None or bestdiff > max_days:
        return 0.0
    return float(best)

# ---------------------------- P/S-beräkningar ----------------------------

def compute_quarter_ps(ticker: str) -> Dict[str, Any]:
    """
    Hämtar:
      - TTM P/S från Yahoo (direkt/fallback)
      - Kvartalsomsättning från Yahoo (quarterlyTotalRevenue)
      - Antal aktier från SEC (närmast +1 dag efter kvartalsdatum)
      - Kurs från Yahoo på +1 dag efter kvartalsdatum
    Returnerar dict med fält för P/S, P/S Q1..Q4, datum/källor.
    """
    if st.session_state.get("offline_mode", False):
        raise RuntimeError("Offline-läge: inga nätanrop")

    out: Dict[str, Any] = {}
    # 1) Quote summary (valuta, namn, utdelning, ps ttm, current shares)
    qs = yahoo_quote_summary(ticker)

    # namn/valuta/kurs
    try:
        out["Bolagsnamn"] = qs.get("price", {}).get("shortName") or qs.get("price", {}).get("longName") or ""
        out["Valuta"] = qs.get("price", {}).get("currency", "USD")
        out["Aktuell kurs"] = _to_float(qs.get("price", {}).get("regularMarketPrice", 0))
        out["Årlig utdelning"] = _to_float(qs.get("summaryDetail", {}).get("dividendRate", 0))
    except Exception:
        pass

    # shares outstanding (nu) — i miljoner
    shares_now = 0.0
    try:
        sh = _to_float(qs.get("defaultKeyStatistics", {}).get("sharesOutstanding"))
        if sh <= 0:
            sh = _to_float(qs.get("price", {}).get("sharesOutstanding"))
        shares_now = sh / 1_000_000.0 if sh else 0.0
    except Exception:
        shares_now = 0.0
    if shares_now > 0:
        out["Utestående aktier"] = _round2(shares_now)
        out["Källa Utestående aktier"] = "Yahoo/info"
        out["TS Utestående aktier"] = _now_iso()

    # ps (ttm)
    ps_ttm = 0.0
    try:
        ps_ttm = _to_float(qs.get("summaryDetail", {}).get("priceToSalesTrailing12Months", 0))
        if ps_ttm <= 0:
            ps_ttm = _to_float(qs.get("defaultKeyStatistics", {}).get("priceToSalesTrailing12Months", 0))
    except Exception:
        pass
    if ps_ttm > 0:
        out["P/S"] = _round2(ps_ttm)
        out["Källa P/S"] = "Yahoo/ps_ttm"
        out["TS P/S"] = _now_iso()

    # 2) Yahoo quarterly revenue
    qjs = yahoo_timeseries_revenue(ticker)
    quarterly = []
    try:
        arr = qjs["timeseries"]["result"][0]["quarterlyTotalRevenue"]
        for it in arr:
            val = _to_float(it.get("reportedValue", {}).get("raw"))
            asof = it.get("asOfDate")
            if val > 0 and asof:
                d = datetime.fromisoformat(asof + "T00:00:00")
                quarterly.append((d, val / 1_000_000.0))  # -> miljoner
    except Exception:
        pass
    quarterly.sort(key=lambda x: x[0], reverse=True)
    quarterly = quarterly[:4]

    # 3) SEC shares serie
    cik_map = sec_company_tickers()
    cik = int(cik_map.get(ticker.upper(), 0))
    sec_series = sec_shares_series(cik) if cik else []

    # 4) Räkna per kvartal (pris@+1d, shares@+1d, ps = price*shares / revenue)
    for idx, (d, rev_mn) in enumerate(quarterly):
        d_plus = d + timedelta(days=1)
        try:
            px = yahoo_price_on(ticker, d_plus)
        except Exception:
            px = 0.0
        sh_mn = _nearest_value(sec_series, d_plus, max_days=30) if sec_series else shares_now
        ps = (px * sh_mn) / rev_mn if (px > 0 and sh_mn > 0 and rev_mn > 0) else 0.0
        qn = idx + 1  # Q1 = senaste
        out[f"P/S Q{qn}"] = _round2(ps)
        out[f"P/S Q{qn} datum"] = d.date().isoformat()
        src = "Computed/Yahoo-revenue+SEC-shares+1d-after" if sec_series else "Computed/Yahoo-revenue+Yahoo-shares+1d-after"
        out[f"Källa P/S Q{qn}"] = src

    return out

# ---------------------------- Data-schema ----------------------------

FINAL_COLS = [
    "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning",
    "Utestående aktier","Antal aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
    "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "CAGR 5 år (%)","P/S-snitt",
    "Senast manuellt uppdaterad","Senast auto uppdaterad",
    "TS P/S","TS Utestående aktier","TS Omsättning",
]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = 0.0 if any(k in c.lower() for k in ["kurs","oms","p/s","utdel","cagr","antal","rikt","snitt","aktier"]) else ""
    return df

def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","CAGR 5 år (%)",
                "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Utestående aktier","Aktuell kurs","Årlig utdelning","Antal aktier"]
    df = to_numeric(df, num_cols)

    for i, r in df.iterrows():
        # snitt av senast 4 kvartal
        vals = [r.get("P/S Q1",0), r.get("P/S Q2",0), r.get("P/S Q3",0), r.get("P/S Q4",0)]
        vals = [float(x) for x in vals if float(x) > 0]
        ps_snitt = _round2(np.mean(vals)) if vals else _round2(float(r.get("P/S",0)))
        df.at[i, "P/S-snitt"] = ps_snitt

        # Omsättning om 2/3 år via CAGR
        cagr = float(r.get("CAGR 5 år (%)", 0.0))
        if cagr > 100.0: cagr = 50.0
        if cagr < -50.0: cagr = -20.0
        g = cagr/100.0
        next_rev = float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            df.at[i, "Omsättning om 2 år"] = _round2(next_rev * (1+g))
            df.at[i, "Omsättning om 3 år"] = _round2(next_rev * ((1+g)**2))

        # Riktkurser (oms i miljoner, aktier i miljoner)
        a_mn = float(r.get("Utestående aktier", 0.0))
        if a_mn > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = _round2(float(r.get("Omsättning idag",0.0))    * ps_snitt / a_mn)
            df.at[i, "Riktkurs om 1 år"] = _round2(float(r.get("Omsättning nästa år",0.0))* ps_snitt / a_mn)
            df.at[i, "Riktkurs om 2 år"] = _round2(float(df.at[i, "Omsättning om 2 år"]) * ps_snitt / a_mn)
            df.at[i, "Riktkurs om 3 år"] = _round2(float(df.at[i, "Omsättning om 3 år"]) * ps_snitt / a_mn)
        else:
            for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i, k] = 0.0
    return df

# ---------------------------- UI: Lägg till / uppdatera ----------------------------

def _apply_row_updates(df: pd.DataFrame, ticker: str, data: Dict[str, Any]) -> pd.DataFrame:
    """Skriv in fälten från 'data' för 'ticker'."""
    if ticker not in df["Ticker"].astype(str).values:
        # skapa tom rad
        base = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Källa Aktuell kurs","Senast manuellt uppdaterad","Senast auto uppdaterad"] else "") for c in df.columns}
        base["Ticker"] = ticker
        df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)

    for k, v in data.items():
        if k in df.columns:
            df.loc[df["Ticker"] == ticker, k] = v
    return df

def _fetch_and_apply(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, bool, str]:
    """Hämtar Yahoo + SEC och applicerar i df. Returnerar (df, success, msg)."""
    try:
        with st.spinner(f"Hämtar från Yahoo/SEC för {ticker} …"):
            # rensa cache för färsk data
            st.cache_data.clear()
            data = compute_quarter_ps(ticker)

            if not data:
                return df, False, "Inga data från Yahoo/SEC."

            # Se till att källor på kurs presenteras
            if "Aktuell kurs" in data and data["Aktuell kurs"] > 0:
                data["Källa Aktuell kurs"] = "Yahoo/price"

            df = _apply_row_updates(df, ticker, data)
            df.loc[df["Ticker"] == ticker, "Senast auto uppdaterad"] = _now_iso()
            df = uppdatera_berakningar(df)
            spara_data(df)
        return df, True, "Klart! Data hämtade och sparade."
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        return df, False, f"Fel vid hämtning: {type(e).__name__}: {e}"

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    namn = [""] + [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    val = st.selectbox("Välj bolag (lämna tomt för nytt)", namn, index=0)
    bef = pd.Series({}, dtype=object)
    if val:
        tkr = val[val.rfind("(")+1:val.rfind(")")]
        r = df[df["Ticker"] == tkr]
        if not r.empty:
            bef = r.iloc[0]

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","")).upper()
            utest  = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)))
            antal  = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)))
            ps_ttm = st.number_input("P/S (TTM)", value=float(bef.get("P/S",0.0)))
            ps1    = st.number_input("P/S Q1 (senaste)", value=float(bef.get("P/S Q1",0.0)))
            ps2    = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)))
        with c2:
            ps3    = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)))
            ps4    = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)))
            oms_i  = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)))
            oms_n  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)))

            st.caption("Aktuell kurskälla: " + (bef.get("Källa Aktuell kurs","Yahoo/price") or "Yahoo/price"))
            st.caption("Senast manuellt uppdaterad: " + (bef.get("Senast manuellt uppdaterad","") or "—"))
            st.caption("Senast auto uppdaterad: " + (bef.get("Senast auto uppdaterad","") or "—"))

        spar = st.form_submit_button("💾 Spara & hämta (Yahoo + SEC)")

    if spar and ticker:
        # Skriv manuella fält först
        base = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps_ttm, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_i, "Omsättning nästa år": oms_n,
            "Senast manuellt uppdaterad": _now_iso(),
        }
        df = _apply_row_updates(df, ticker, base)
        spara_data(df)  # spara direkt
        df, ok, msg = _fetch_and_apply(df, ticker)
        (st.success if ok else st.error)(msg)
        st.rerun()

    if not bef.empty and st.button("↻ Hämta igen denna ticker (Yahoo + SEC)"):
        tkr = str(bef.get("Ticker","")).upper()
        df, ok, msg = _fetch_and_apply(df, tkr)
        (st.success if ok else st.error)(msg)
        st.rerun()

    # Lista – äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )
    return df

# ---------------------------- Analys & Portfölj ----------------------------

def analysvy(df: pd.DataFrame) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Tom databas.")
        return
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    r = vis_df.iloc[st.session_state.analys_idx]
    cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
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
    total = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = (port["Värde (SEK)"] / total * 100.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    st.markdown(f"**Totalt portföljvärde:** {_round2(total)} SEK")
    st.markdown(f"**Total kommande utdelning:** {_round2(float(port['Total årlig utdelning (SEK)'].sum()))} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {_round2(float(port['Total årlig utdelning (SEK)'].sum())/12.0)} SEK")
    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

# ---------------------------- MAIN ----------------------------

def main():
    st.title("📊 Aktieanalys & P/S (monolit)")

    user_rates = hamta_valutakurser_sidebar()

    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    df = ensure_schema(df)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Lägg till / uppdatera bolag","Analys","Portfölj"], index=0)

    if meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Analys":
        analysvy(df)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
