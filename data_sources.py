# data_sources.py
from __future__ import annotations
import math
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -------------------- Hjälp: tid/datum --------------------
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def ts_now():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
except Exception:
    def ts_now():
        return datetime.now().strftime("%Y-%m-%d %H:%M")

def _to_date(x) -> datetime:
    if isinstance(x, datetime):
        return x
    try:
        return pd.to_datetime(str(x)).to_pydatetime()
    except Exception:
        return None

# -------------------- Yahoo: pris, info, revenue --------------------
def _yahoo_price_on_or_after(ticker: str, d: datetime, window_days: int = 10) -> float | None:
    """Stängningskurs första handelsdag >= d (med tolerans upp till window_days)."""
    start = (d - timedelta(days=1)).strftime("%Y-%m-%d")
    end   = (d + timedelta(days=window_days)).strftime("%Y-%m-%d")
    try:
        h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if h.empty:
            return None
        # välj första rad på/efter d
        h = h.reset_index()
        h["Date"] = pd.to_datetime(h["Date"])
        row = h[h["Date"] >= pd.to_datetime(d)].head(1)
        if row.empty:
            return float(h["Close"].iloc[-1])
        return float(row["Close"].iloc[0])
    except Exception:
        return None

def _yahoo_fast_price(ticker: str) -> float | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        p = info.get("regularMarketPrice", None)
        if p is None:
            h = t.history(period="1d")
            if not h.empty:
                p = float(h["Close"].iloc[-1])
        return float(p) if p is not None else None
    except Exception:
        return None

def _yahoo_quarterly_revenue(ticker: str) -> pd.DataFrame:
    """
    Returnerar DF med kolumner: date (datetime), revenue (float, USD).
    Försöker Yahoo quarterly income/financials; annars tom DF.
    """
    t = yf.Ticker(ticker)
    out = []
    # yfinance har lite varierande API mellan versioner – testa flera vägar
    for attr in ["quarterly_income_stmt", "quarterly_financials"]:
        try:
            df = getattr(t, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "Total Revenue" in df.index:
                    ser = df.loc["Total Revenue"].dropna()
                    # kolumnnamn är perioder
                    for col, val in ser.items():
                        d = _to_date(col)
                        if d and pd.notna(val):
                            out.append({"date": d, "revenue": float(val)})
                break
        except Exception:
            pass

    # fallback – ibland finns income_stmt med kvartalskolumner
    if not out:
        try:
            df = getattr(t, "income_stmt", None)
            if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                ser = df.loc["Total Revenue"].dropna()
                # yfinance kan blanda årsdata – vi plockar senaste 8 datapunkter och hoppas de är kvartal
                for col, val in ser.items():
                    d = _to_date(col)
                    if d and pd.notna(val):
                        out.append({"date": d, "revenue": float(val)})
        except Exception:
            pass

    if not out:
        return pd.DataFrame(columns=["date","revenue"])

    df = pd.DataFrame(out).dropna().sort_values("date", ascending=False).reset_index(drop=True)
    # Rensa icke-positiva värden
    df = df[df["revenue"] > 0]
    return df.head(12)  # räcker för TTM runt 4 kvartal x 3 år

# -------------------- SEC: aktier & metadata (enkla endpoints) --------------------
def _sec_company_ticker_to_cik(ticker: str) -> str | None:
    """
    Kräver att du lagt en SEC-ticker-mapp i secrets under key 'SEC_TICKER_MAP_URL' – annars return None.
    För enkelhet använder vi cachad lista i secrets om du redan lagt in NVDA → CIK.
    """
    # Direkt från secrets om finns
    m = st.secrets.get("SEC_CIK_MAP", {})
    if m:
        v = m.get(ticker.upper())
        return str(v) if v else None
    return None

def _sec_recent_shares_points(cik: str) -> list[dict]:
    """
    Returnerar punkter med (date, shares) från senaste 10-Q/10-K headers 'Common shares outstanding…'
    Du kan fylla st.secrets["SEC_SHARES_POINTS"][CIK] med manuellt klippta punkter för hög precision.
    """
    # Stöd för manuellt seedade punkter via secrets
    pre = st.secrets.get("SEC_SHARES_POINTS", {})
    pts = pre.get(cik, [])
    out = []
    for p in pts:
        try:
            d = _to_date(p.get("date"))
            sh = float(p.get("shares"))
            if d and sh > 0:
                out.append({"date": d, "shares": sh})
        except Exception:
            pass
    # sortera nyast först
    out = sorted(out, key=lambda x: x["date"], reverse=True)
    return out[:12]

def _shares_on_or_after(cik: str, d: datetime, yahoo_fallback: float | None, pts: list[dict]) -> float | None:
    """
    Välj närmast datapunkt med datum <= d (eller första efter) från SEC; annars Yahoo fallback (impliedSharesOutstanding).
    """
    if pts:
        # välj punkt närmast d (före eller efter med liten tolerans)
        pts_sorted = sorted(pts, key=lambda x: abs((x["date"] - d).days))
        if pts_sorted:
            return float(pts_sorted[0]["shares"])
    return yahoo_fallback

# -------------------- P/S-beräkning: *TTM* per kvartal --------------------
def _compute_ps_ttm_series(ticker: str) -> tuple[dict, dict]:
    """
    Returnerar:
      ps_vals: { 'P/S': p_s_ttm_now (float) }
      ps_quarters: {
         1: {'value':..., 'date':..., 'source':'Computed/Yahoo-TTM-revenue+SEC-shares+1d-after' },
         2: {...}, 3: {...}, 4: {...}
      }
    """
    t = yf.Ticker(ticker)

    # 1) Revenue-kvartal från Yahoo → TTM-summor
    qrev = _yahoo_quarterly_revenue(ticker)  # nyast först
    ps_quarters = {}

    # 2) Yahoo pris nu
    price_now = _yahoo_fast_price(ticker)

    # 3) Yahoo implied shares (fallback)
    implied_shares = None
    try:
        finfo = t.fast_info or {}
        implied_shares = finfo.get("shares")
        if implied_shares is not None:
            implied_shares = float(implied_shares)
    except Exception:
        implied_shares = None

    # 4) SEC: försök få CIK + “shares points” seedade (från secrets)
    cik = _sec_company_ticker_to_cik(ticker) or ""
    sec_pts = _sec_recent_shares_points(cik) if cik else []

    # 5) P/S (TTM) just nu – om vi har TTM revenue (summa 4 senaste)
    p_s_now = None
    try:
        if len(qrev) >= 4 and price_now and (implied_shares or sec_pts):
            ttm_now = float(qrev.iloc[0:4]["revenue"].sum())
            shares_now = _shares_on_or_after(cik, _to_date(qrev.iloc[0]["date"]), implied_shares, sec_pts)
            if shares_now and shares_now > 0 and ttm_now > 0:
                p_s_now = float(price_now * shares_now / ttm_now)
    except Exception:
        p_s_now = None

    # 6) P/S för senaste 4 kvartal som *TTM vid kvartalet*
    #    Vi behöver: (a) TTM revenue = sum av kvartal i fönster [i..i+3]
    #                (b) pris = stängning första handelsdag efter kvartalsdatum
    #                (c) aktier ~ SEC-punkt närmast kvartalsdatum, annars Yahoo implied
    for qi in range(4):
        if len(qrev) >= qi + 4:
            qdate = _to_date(qrev.iloc[qi]["date"])
            ttm = float(qrev.iloc[qi:qi+4]["revenue"].sum())
            price_q = _yahoo_price_on_or_after(ticker, qdate + timedelta(days=1)) or price_now
            shares_q = _shares_on_or_after(cik, qdate, implied_shares, sec_pts)

            ps_val = None
            if price_q and shares_q and shares_q > 0 and ttm > 0:
                ps_val = float(price_q * shares_q / ttm)

            # litet sanity-clamp för TTM-PS (rimligt mellan 0 och 100)
            if ps_val is not None:
                if ps_val < 0 or ps_val > 1000:  # hård spärr ifall enheter spårar ur
                    ps_val = None

            ps_quarters[qi+1] = {
                "value": round(ps_val, 2) if ps_val is not None else 0.0,
                "date": qdate.strftime("%Y-%m-%d") if qdate else "–",
                "source": "Computed/Yahoo-TTM-revenue+SEC-shares+1d-after" if ps_val is not None else "n/a",
            }
        else:
            ps_quarters[qi+1] = {"value": 0.0, "date": "–", "source": "n/a"}

    ps_vals = {"P/S": round(p_s_now, 2) if p_s_now is not None else 0.0}
    # meta för logg
    meta = {
        "ps_source": "yahoo_ps_ttm",
        "q_cols": len(qrev),
        "price_hits": 0,  # (lämnas 0, vi loggar inte träffar per datum här)
        "sec_cik": cik,
        "sec_shares_pts": len(sec_pts),
        "cutoff_years": st.session_state.get("SEC_CUTOFF_YEARS", 6),
        "backfill_used": st.session_state.get("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF", False),
    }
    return {"ps_vals": ps_vals, "meta": meta, "ps_quarters": ps_quarters}, {"qrev": qrev}

# -------------------- Publika funktioner --------------------
def hamta_live_valutakurser() -> dict:
    # Hämtar USD, EUR, NOK, CAD mot SEK via yfinance (USDSEK=X etc).
    pairs = {"USD": "USDSEK=X", "EUR": "EURSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X"}
    out = {"SEK": 1.0}
    for k, s in pairs.items():
        try:
            h = yf.Ticker(s).history(period="5d")
            if not h.empty:
                out[k] = float(h["Close"].iloc[-1])
        except Exception:
            pass
    # rimlig fallback om någon misslyckas
    for k in ["USD","EUR","NOK","CAD"]:
        out.setdefault(k, {"USD":9.75,"EUR":11.18,"NOK":0.95,"CAD":7.05}[k])
    return out

def hamta_sec_filing_lankar(ticker: str) -> list[dict]:
    # Enkel visning från secrets-lista om du lagt in manuellt. Annars tomt.
    cik = _sec_company_ticker_to_cik(ticker)
    if not cik:
        return []
    links = []
    arr = st.secrets.get("SEC_LATEST_LINKS", {}).get(cik, [])
    for a in arr:
        links.append({
            "form": a.get("form",""),
            "date": a.get("date",""),
            "viewer": a.get("viewer",""),
            "url": a.get("url",""),
            "cik": cik
        })
    return links

def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Huvudhämtare för appen. Returnerar fält redo att skrivas in i DF.
    - Namn, Kurs, Valuta, Utdelning, CAGR (enkel), Utestående aktier (Yahoo/info)
    - P/S (TTM) och P/S Q1..Q4 = TTM-P/S vid respektive kvartal
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
        "Utestående aktier": 0.0,   # OBS: absolut antal / 1e6? Vi lämnar absolut, UI visar miljoner.
        "Källa Aktuell kurs": "",
        "Källa Utestående aktier": "",
        "Källa P/S": "",
        "Källa P/S Q1": "", "Källa P/S Q2": "", "Källa P/S Q3": "", "Källa P/S Q4": "",
        "P/S Q1 datum": "", "P/S Q2 datum": "", "P/S Q3 datum": "", "P/S Q4 datum": "",
        "TS P/S": ts_now(),
        "TS Utestående aktier": ts_now(),
    }

    # --- Yahoo basinfo ---
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # pris + valuta + namn + utdelning
    pris = info.get("regularMarketPrice", None)
    if pris is None:
        h = t.history(period="1d")
        if not h.empty and "Close" in h:
            pris = float(h["Close"].iloc[-1])
    if pris is not None:
        out["Aktuell kurs"] = float(pris)
        out["Källa Aktuell kurs"] = "Yahoo/info"

    valuta = info.get("currency", None)
    if valuta:
        out["Valuta"] = str(valuta).upper()

    namn = info.get("shortName") or info.get("longName") or ""
    if namn:
        out["Bolagsnamn"] = str(namn)

    div_rate = info.get("dividendRate", None)
    if div_rate is not None:
        out["Årlig utdelning"] = float(div_rate)

    # Utestående aktier – Yahoo implied, visas i UI som *miljoner*
    implied = None
    try:
        finfo = t.fast_info or {}
        implied = finfo.get("shares")
        if implied is not None:
            out["Utestående aktier"] = float(implied) / 1e6  # lagras i DF som miljoner för att matcha dina formler
            out["Källa Utestående aktier"] = "Yahoo/info"
    except Exception:
        pass

    # --- P/S (TTM) + kvartal (TTM vid kvartalet) ---
    ps_pack, _ = _compute_ps_ttm_series(ticker)
    out["P/S"] = float(ps_pack["ps_vals"].get("P/S", 0.0))
    out["Källa P/S"] = "Yahoo/ps_ttm"

    for qi in (1,2,3,4):
        item = ps_pack["ps_quarters"].get(qi, {})
        out[f"P/S Q{qi}"] = float(item.get("value", 0.0))
        out[f"P/S Q{qi} datum"] = item.get("date", "")
        out[f"Källa P/S Q{qi}"] = item.get("source", "")

    # --- “CAGR 5 år (%)” – försiktig uppskattning från annual revenue om möjligt ---
    out["CAGR 5 år (%)"] = _simple_revenue_cagr(t)

    # Loggning till session (för expander + ev. LOGS-blad)
    st.session_state.setdefault("fetch_logs", [])
    st.session_state["fetch_logs"].append({
        "ts": ts_now(),
        "ticker": ticker.upper(),
        "summary": f"Fetched Yahoo/SEC: price={out['Aktuell kurs']}, shares(mn)={out['Utestående aktier']:.2f}, ps_ttm_now={out['P/S']}",
        "ps": ps_pack.get("meta", {})
    })

    return out


# -------------------- Enkelt CAGR (annual revenue) --------------------
def _simple_revenue_cagr(tkr: yf.Ticker) -> float:
    try:
        df_fin = getattr(tkr, "financials", None)
        if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
            series = df_fin.loc["Total Revenue"].dropna().sort_index()
            if len(series) >= 2:
                start = float(series.iloc[0]); end = float(series.iloc[-1]); years = max(1, len(series)-1)
                if start > 0:
                    return round(((end / start) ** (1.0/years) - 1.0) * 100.0, 2)
    except Exception:
        pass
    return 0.0
