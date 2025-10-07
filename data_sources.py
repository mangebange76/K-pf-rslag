# data_sources.py
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd
import streamlit as st
import yfinance as yf

# -------------------- Tidshjälp --------------------
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def ts_now() -> str:
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
except Exception:
    def ts_now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M")


def _to_date(x) -> Optional[datetime]:
    if isinstance(x, datetime):
        return x
    try:
        return pd.to_datetime(str(x)).to_pydatetime()
    except Exception:
        return None


# -------------------- Yahoo: pris & revenue --------------------
def _yahoo_fast_price(ticker: str) -> Optional[float]:
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


def _yahoo_price_on_or_after(ticker: str, d: datetime, window_days: int = 30) -> Tuple[Optional[float], str]:
    """
    Stängningskurs första handelsdag >= d (upp till window_days). Returnerar (pris, prisdatum_str).
    """
    try:
        start = (d - timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (d + timedelta(days=window_days)).strftime("%Y-%m-%d")
        t = yf.Ticker(ticker)
        h = t.history(start=start, end=end, interval="1d", auto_adjust=False)
        if h.empty:
            return None, ""
        df = h.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        target = pd.to_datetime(d).tz_localize(None)
        row = df[df["Date"] >= target].head(1)
        if row.empty:
            row = df.sort_values("Date").tail(1)
        price = float(row["Close"].iloc[0])
        pdate = row["Date"].iloc[0].strftime("%Y-%m-%d")
        return price, pdate
    except Exception:
        return None, ""


def _yahoo_quarterly_revenue(ticker: str) -> pd.DataFrame:
    """
    Returnera DF ['date','revenue'] (nyast först). Samlar från flera Yahoo-källor
    och fyller upp till ~12 punkter. Har även valfri secrets-fallback (SEC_REVENUE_POINTS).
    """
    t = yf.Ticker(ticker)
    buckets: Dict[datetime.date, float] = {}

    # 1) Primärt: quarterly_income_stmt / quarterly_financials (Total Revenue)
    for attr in ["quarterly_income_stmt", "quarterly_financials"]:
        try:
            df = getattr(t, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                ser = df.loc["Total Revenue"].dropna()
                for col, val in ser.items():
                    d = _to_date(col)
                    if d and pd.notna(val) and float(val) > 0:
                        buckets.setdefault(pd.to_datetime(d).date(), float(val))
                break
        except Exception:
            pass

    # 2) Komplettera med quarterly_earnings (kolumn 'Revenue')
    try:
        qe = t.quarterly_earnings
        if isinstance(qe, pd.DataFrame) and not qe.empty:
            colname = None
            for c in qe.columns:
                if str(c).lower().strip() == "revenue":
                    colname = c
                    break
            if colname:
                for idx, row in qe.iterrows():
                    d = _to_date(idx)
                    rev = row.get(colname, None)
                    if d and rev is not None and float(rev) > 0:
                        buckets.setdefault(pd.to_datetime(d).date(), float(rev))
    except Exception:
        pass

    # 3) Valfri manuell SEC-fallback (secrets: SEC_REVENUE_POINTS)
    cik = _sec_company_ticker_to_cik(ticker)
    if cik:
        for p in _sec_quarterly_revenue_points(cik):
            buckets.setdefault(pd.to_datetime(p["date"]).date(), float(p["revenue"]))

    if not buckets:
        return pd.DataFrame(columns=["date", "revenue"])

    df = (
        pd.DataFrame(
            [{"date": pd.to_datetime(d), "revenue": float(v)} for d, v in buckets.items()]
        )
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    df = df[df["revenue"] > 0]
    return df.head(12)


# -------------------- SEC: CIK + manuella punkter (frivilligt via secrets) --------------------
def _sec_company_ticker_to_cik(ticker: str) -> Optional[str]:
    """
    Försök hitta CIK via secrets:
      - CIK_OVERRIDES: JSON/dict { "NVDA": "0001045810", ... }
      - SEC_CIK_MAP: dict { "NVDA": "0001045810", ... }
    """
    try:
        # Streamlit kan ha JSON-sträng eller dict
        ov = st.secrets.get("CIK_OVERRIDES", {})
        if isinstance(ov, str):
            import json
            ov = json.loads(ov)
        if isinstance(ov, dict):
            v = ov.get(ticker.upper())
            if v:
                return str(v)
    except Exception:
        pass
    m = st.secrets.get("SEC_CIK_MAP", {})
    if isinstance(m, dict):
        v = m.get(ticker.upper())
        if v:
            return str(v)
    return None


def _sec_quarterly_revenue_points(cik: str) -> List[Dict[str, Any]]:
    """
    Läser manuellt matade revenue-punkter från secrets:
      SEC_REVENUE_POINTS = { "0001045810": [ {"date":"2024-10-31","revenue":...}, ... ] }
    """
    arr = st.secrets.get("SEC_REVENUE_POINTS", {}).get(cik or "", [])
    out = []
    for p in arr:
        try:
            d = _to_date(p.get("date"))
            r = float(p.get("revenue"))
            if d and r > 0:
                out.append({"date": d, "revenue": r})
        except Exception:
            pass
    return sorted(out, key=lambda x: x["date"], reverse=True)[:12]


def _sec_recent_shares_points(cik: str) -> List[Dict[str, Any]]:
    """
    Läser manuellt matade shares-punkter från secrets:
      SEC_SHARES_POINTS = { "0001045810": [ {"date":"2025-08-25","shares": 24_000_000_000}, ... ] }
    """
    pts = st.secrets.get("SEC_SHARES_POINTS", {}).get(cik or "", [])
    out = []
    for p in pts:
        try:
            d = _to_date(p.get("date"))
            sh = float(p.get("shares"))
            if d and sh > 0:
                out.append({"date": d, "shares": sh})
        except Exception:
            pass
    return sorted(out, key=lambda x: x["date"], reverse=True)[:12]


def _shares_on_or_after(cik: str, d: datetime, yahoo_fallback: Optional[float], pts: List[Dict[str, Any]]) -> Optional[float]:
    """
    Välj närmaste datapunkt (före/efter) från SEC-punkter; annars Yahoo implied shares.
    """
    if pts:
        pts_sorted = sorted(pts, key=lambda x: abs((x["date"] - d).days))
        if pts_sorted:
            return float(pts_sorted[0]["shares"])
    return yahoo_fallback


# -------------------- FY-etiketter --------------------
def _parse_fiscal_year_end_month(info: dict, qrev: pd.DataFrame) -> int:
    """
    Läs 'fiscalYearEnd' ur Yahoo info (format MMDD) eller gissa.
    Returnerar månaden (1..12) då räkenskapsåret slutar.
    """
    try:
        if isinstance(info, dict):
            fye = info.get("fiscalYearEnd", None)
            if fye:
                s = str(fye).strip()
                if len(s) >= 2 and s[:2].isdigit():
                    m = int(s[:2])
                    if 1 <= m <= 12:
                        return m
    except Exception:
        pass

    if isinstance(qrev, pd.DataFrame) and not qrev.empty:
        months = qrev["date"].dt.month.tolist()
        patterns = {
            1:  [1,4,7,10],
            12: [12,3,6,9],
            3:  [3,6,9,12],
            7:  [7,10,1,4],
        }
        best_m = 1
        best_hits = -1
        for m_end, pats in patterns.items():
            hits = sum(1 for m in months if any(((m - p) % 12 == 0) for p in pats))
            if hits > best_hits:
                best_hits, best_m = hits, m_end
        return best_m

    return 1


def _fy_quarter_label(d: datetime, fy_end_month: int) -> Tuple[str, int]:
    """
    Beräkna ('FY25 Q2', 2) för ett kvartals slutdatum d och räkenskapsårsslutmånad.
    """
    if not isinstance(d, datetime):
        d = _to_date(d)
    if d is None:
        return "—", 0

    fy_year = d.year if d.month <= fy_end_month else d.year + 1
    start_m = 1 if fy_end_month == 12 else fy_end_month + 1
    offset = ((d.year * 12 + d.month) - (d.year * 12 + start_m)) % 12
    q = offset // 3 + 1
    return f"FY{str(fy_year)[-2:]} Q{q}", int(q)


# -------------------- P/S TTM nu + TTM per kvartal --------------------
def _compute_ps_ttm_series(ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returnerar:
      - pack: {"ps_vals": {"P/S": float}, "meta": {...}, "ps_quarters": {1:{...},2:{...},3:{...},4:{...}}}
      - extras: {"qrev": DataFrame}
    """
    t = yf.Ticker(ticker)

    # 1) Revenue-kvartal (nyast först)
    qrev = _yahoo_quarterly_revenue(ticker)
    if not qrev.empty:
        qrev = qrev.sort_values("date", ascending=False).reset_index(drop=True)

    # 2) Nuvarande pris + info (för FYE)
    price_now = _yahoo_fast_price(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # 3) Yahoo implied shares (fallback för SEC)
    implied_shares = None
    try:
        finfo = t.fast_info or {}
        implied_shares = finfo.get("shares")
        if implied_shares is not None:
            implied_shares = float(implied_shares)
    except Exception:
        implied_shares = None

    # 4) SEC-punkter (manuellt seedade i secrets)
    cik = _sec_company_ticker_to_cik(ticker) or ""
    sec_pts = _sec_recent_shares_points(cik) if cik else []

    # 5) P/S (TTM) nu
    p_s_now = None
    try:
        if len(qrev) >= 4 and price_now and (implied_shares or sec_pts):
            ttm_now = float(qrev.iloc[0:4]["revenue"].sum())
            shares_now = _shares_on_or_after(cik, _to_date(qrev.iloc[0]["date"]), implied_shares, sec_pts)
            if shares_now and shares_now > 0 and ttm_now > 0:
                p_s_now = float(price_now * shares_now / ttm_now)
    except Exception:
        p_s_now = None

    # 6) TTM-P/S för senaste fyra kvartal
    fy_end_month = _parse_fiscal_year_end_month(info, qrev)
    ps_quarters: Dict[int, Dict[str, Any]] = {}
    for qi in range(4):
        if len(qrev) >= qi + 4:
            qdate = _to_date(qrev.iloc[qi]["date"])
            label, _ = _fy_quarter_label(qdate, fy_end_month)
            ttm = float(qrev.iloc[qi:qi+4]["revenue"].sum())
            price_q, price_dt = _yahoo_price_on_or_after(ticker, qdate + timedelta(days=1))
            if price_q is None:
                price_q, price_dt = price_now, "now"
            shares_q = _shares_on_or_after(cik, qdate, implied_shares, sec_pts)

            ps_val = None
            if price_q and shares_q and shares_q > 0 and ttm > 0:
                ps_val = float(price_q * shares_q / ttm)
            if ps_val is not None and (ps_val < 0 or ps_val > 1000):
                ps_val = None

            ps_quarters[qi+1] = {
                "value": round(ps_val, 2) if ps_val is not None else 0.0,
                "date": qdate.strftime("%Y-%m-%d") if qdate else "–",
                "source": f"Computed/{label}/price@{price_dt}",
            }
        else:
            ps_quarters[qi+1] = {"value": 0.0, "date": "–", "source": "n/a"}

    ps_vals = {"P/S": round(p_s_now, 2) if p_s_now is not None else 0.0}
    meta = {
        "ps_source": "yahoo_ps_ttm",
        "q_cols": int(len(qrev)),
        "sec_cik": cik,
        "sec_shares_pts": int(len(sec_pts)),
    }
    return {"ps_vals": ps_vals, "meta": meta, "ps_quarters": ps_quarters}, {"qrev": qrev}


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


# -------------------- Publika funktioner till vyerna --------------------
def hamta_live_valutakurser() -> dict:
    pairs = {"USD": "USDSEK=X", "EUR": "EURSEK=X", "NOK": "NOKSEK=X", "CAD": "CADSEK=X"}
    out = {"SEK": 1.0}
    for k, s in pairs.items():
        try:
            h = yf.Ticker(s).history(period="5d")
            if not h.empty:
                out[k] = float(h["Close"].iloc[-1])
        except Exception:
            pass
    # rimliga defaults om något misslyckas
    defaults = {"USD":9.75,"EUR":11.18,"NOK":0.95,"CAD":7.05}
    for k,v in defaults.items():
        out.setdefault(k, v)
    return out


def hamta_sec_filing_lankar(ticker: str) -> List[Dict[str, str]]:
    """
    Enkel länkvisare: hämtar från secrets om du matat in:
      SEC_LATEST_LINKS = { "0001045810": [ {"form":"10-Q","date":"2025-08-28","viewer":"...","url":"..."} ] }
    """
    cik = _sec_company_ticker_to_cik(ticker)
    if not cik:
        return []
    arr = st.secrets.get("SEC_LATEST_LINKS", {}).get(cik, [])
    out = []
    for a in arr:
        out.append({
            "form": a.get("form",""),
            "date": a.get("date",""),
            "viewer": a.get("viewer",""),
            "url": a.get("url",""),
            "cik": cik
        })
    return out


def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Huvudhämtare:
      - Namn, Kurs, Valuta, Utdelning, CAGR
      - Utestående aktier (miljoner) från Yahoo implied shares
      - P/S (TTM) nu
      - P/S Q1..Q4 = TTM-P/S vid respektive kvartalsstopp (med FY-etikett i 'Källa P/S Qx')
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,

        # OBS: lagras i miljoner för att matcha dina formler/visning
        "Utestående aktier": 0.0,

        "Källa Aktuell kurs": "",
        "Källa Utestående aktier": "",
        "Källa P/S": "",

        "Källa P/S Q1": "", "Källa P/S Q2": "", "Källa P/S Q3": "", "Källa P/S Q4": "",
        "P/S Q1 datum": "", "P/S Q2 datum": "", "P/S Q3 datum": "", "P/S Q4 datum": "",

        "TS P/S": ts_now(),
        "TS Utestående aktier": ts_now(),
    }

    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    # Pris/valuta/namn/utdelning
    pris = info.get("regularMarketPrice", None)
    if pris is None:
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        except Exception:
            pris = None
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

    # Utestående aktier (miljoner) – Yahoo implied
    implied = None
    try:
        finfo = t.fast_info or {}
        implied = finfo.get("shares")
        if implied is not None:
            out["Utestående aktier"] = float(implied) / 1e6
            out["Källa Utestående aktier"] = "Yahoo/info"
    except Exception:
        pass

    # P/S (TTM) + kvartal (TTM vid kvartal)
    ps_pack, _extras = _compute_ps_ttm_series(ticker)
    out["P/S"] = float(ps_pack["ps_vals"].get("P/S", 0.0))
    out["Källa P/S"] = "Yahoo/ps_ttm"

    for qi in (1,2,3,4):
        item = ps_pack["ps_quarters"].get(qi, {})
        out[f"P/S Q{qi}"] = float(item.get("value", 0.0))
        out[f"P/S Q{qi} datum"] = item.get("date", "")
        out[f"Källa P/S Q{qi}"] = item.get("source", "")

    # CAGR (enkel)
    out["CAGR 5 år (%)"] = _simple_revenue_cagr(t)

    # Logg
    st.session_state.setdefault("fetch_logs", [])
    st.session_state["fetch_logs"].append({
        "ts": ts_now(),
        "ticker": ticker.upper(),
        "summary": f"Fetched Yahoo: price={out['Aktuell kurs']}, shares(mn)={out['Utestående aktier']:.2f}, ps_ttm_now={out['P/S']}",
        "ps": ps_pack.get("meta", {}),
    })

    return out
