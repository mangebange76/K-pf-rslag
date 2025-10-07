# data_sources.py
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

# Tidsstämplar
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp(): return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
except Exception:
    def now_stamp(): return datetime.now().strftime("%Y-%m-%d %H:%M")

# ---- små helpers för datum/etiketter ----
def _to_naive_ts(x):
    if isinstance(x, pd.Timestamp):
        try: return x.tz_localize(None)
        except Exception: return pd.Timestamp(x)
    try: return pd.to_datetime(x).tz_localize(None)
    except Exception: return None

def _clean_iso(d):
    try: return pd.to_datetime(d).date().isoformat()
    except Exception: return ""

# --------- Live FX (Yahoo) ---------
FX_TICKERS = {"USD":"USDSEK=X","NOK":"NOKSEK=X","CAD":"CADSEK=X","EUR":"EURSEK=X","SEK":None}
def hamta_live_valutakurser() -> dict:
    out = {}
    for cc, yt in FX_TICKERS.items():
        if yt is None: out[cc] = 1.0; continue
        try:
            t = yf.Ticker(yt)
            info = {}
            try: info = t.info or {}
            except Exception: info = {}
            px = info.get("regularMarketPrice")
            if px is None:
                h = t.history(period="1d", auto_adjust=False)
                if not h.empty: px = float(h["Close"].iloc[-1])
            if px: out[cc] = float(px)
        except Exception: pass
    return out

# --------- CAGR ---------
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
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
        if series.empty or len(series) < 2: return 0.0
        series = series.sort_index()
        start = float(series.iloc[0]); end = float(series.iloc[-1]); years = max(1, len(series)-1)
        if start <= 0: return 0.0
        return round(((end/start)**(1/years)-1)*100, 2)
    except Exception:
        return 0.0

# --------- SEC helpers ----------
SEC_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

def _sec_headers():
    ua = st.secrets.get("SEC_USER_AGENT", "").strip() or "StreamlitApp/1.0 (contact@example.com)"
    return {"User-Agent": ua, "Accept": "application/json"}

@st.cache_data(show_spinner=False, ttl=60*60*24)
def _sec_load_ticker_map() -> dict:
    try:
        r = requests.get(SEC_TICKERS_URL, headers=_sec_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
        out = {}
        for _, row in data.items():
            t = str(row.get("ticker","")).upper().strip()
            cik_str = str(row.get("cik_str","")).strip()
            if t and cik_str.isdigit():
                out[t] = str(cik_str).zfill(10)
        return out
    except Exception:
        return {}

def _ticker_to_cik_any(ticker: str) -> str | None:
    tk = str(ticker).upper().strip()
    # overrides via secrets
    try:
        import json
        overrides_raw = st.secrets.get("CIK_OVERRIDES", "")
        if isinstance(overrides_raw, str) and overrides_raw.strip():
            overrides = json.loads(overrides_raw)
        elif isinstance(overrides_raw, dict):
            overrides = overrides_raw
        else:
            overrides = {}
        if tk in overrides:
            cik = "".join(ch for ch in str(overrides[tk]) if ch.isdigit())
            if cik: return cik.zfill(10)
    except Exception:
        pass
    # yfinance
    try:
        info = yf.Ticker(tk).info or {}
        cik = str(info.get("cik") or "")
        cik = "".join(ch for ch in cik if ch.isdigit())
        if cik: return cik.zfill(10)
    except Exception:
        pass
    # SEC-lista
    m = _sec_load_ticker_map()
    return m.get(tk)

@st.cache_data(show_spinner=False, ttl=60*60*6)
def _fetch_companyfacts(cik: str) -> dict | None:
    try:
        url = SEC_BASE.format(cik=cik)
        r = requests.get(url, headers=_sec_headers(), timeout=20)
        if r.ok: return r.json()
    except Exception:
        return None
    return None

def _extract_shares_history(facts_json: dict) -> dict:
    out = {}
    try:
        facts = facts_json.get("facts", {}).get("us-gaap", {})
        for tag in ["CommonStockSharesOutstanding","EntityCommonStockSharesOutstanding","SharesOutstanding"]:
            if tag not in facts: continue
            units = facts[tag].get("units", {})
            for unit_name, values in units.items():
                if "share" not in unit_name.lower(): continue
                for v in values:
                    end = v.get("end") or v.get("instant")
                    val = v.get("val")
                    if end and val is not None:
                        try: out[pd.to_datetime(end).date().isoformat()] = float(val)
                        except Exception: pass
            if out: break
    except Exception:
        pass
    return out

def _nearest_shares_for_date(date_iso: str, shares_map: dict) -> float | None:
    if not shares_map: return None
    target = pd.to_datetime(date_iso)
    items = sorted(((pd.to_datetime(d), v) for d,v in shares_map.items()), key=lambda x: x[0])
    le = [v for d,v in items if d <= target]
    if le: return le[-1]
    return items[0][1] if items else None

# --------- Yahoo helpers ----------
def _try_get_quarterly_revenue_df(t: yf.Ticker) -> pd.DataFrame | None:
    qfin = getattr(t, "quarterly_income_stmt", None)
    if not (isinstance(qfin, pd.DataFrame) and not qfin.empty and "Total Revenue" in qfin.index):
        qfin = getattr(t, "quarterly_financials", None)
    if isinstance(qfin, pd.DataFrame) and not qfin.empty and "Total Revenue" in qfin.index:
        return qfin.copy()
    return None

def _nearest_price_on_or_after(t: yf.Ticker, dt: pd.Timestamp, max_days: int = 10) -> float | None:
    start = (dt - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")
    end   = (dt + pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")
    try:
        h = t.history(start=start, end=end, auto_adjust=False, interval="1d")
        if h.empty: return None
        after = h[h.index >= dt]
        if not after.empty: return float(after["Close"].iloc[0])
        before = h[h.index < dt]
        if not before.empty: return float(before["Close"].iloc[-1])
        return float(h["Close"].iloc[-1])
    except Exception:
        return None

# --------- P/S-historik + källor + DEBUG ---------
def hamta_ps_kvartal(ticker: str) -> dict:
    out = {
        "P/S": 0.0,
        "P/S Q1": 0.0, "P/S Q2": 0.0, "P/S Q3": 0.0, "P/S Q4": 0.0,
        "P/S Q1 datum":"", "P/S Q2 datum":"", "P/S Q3 datum":"", "P/S Q4 datum":"",
        "Källa P/S":"", "Källa P/S Q1":"", "Källa P/S Q2":"", "Källa P/S Q3":"", "Källa P/S Q4":"",
        "_DEBUG_PS": {}
    }
    dbg = {"ticker": ticker, "ps_source": "-", "q_cols": 0, "ttm_points": 0, "price_hits": 0, "sec_cik": None, "sec_shares_pts": 0}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try: info = t.info or {}
        except Exception: info = {}
        shares_now = float(info.get("sharesOutstanding") or 0.0)

        qfin = _try_get_quarterly_revenue_df(t)
        if qfin is None or qfin.empty or "Total Revenue" not in qfin.index:
            ps_ttm = info.get("priceToSalesTrailing12Months")
            if ps_ttm and ps_ttm > 0:
                out["P/S"] = float(ps_ttm); out["Källa P/S"] = "Yahoo/ps_ttm"; dbg["ps_source"] = "yahoo_ps_ttm"
            out["_DEBUG_PS"] = dbg
            return out

        # normalisera datumetiketter
        cols_raw = list(qfin.columns)
        cols = []
        for c in cols_raw:
            ts = _to_naive_ts(c)
            if ts is not None: cols.append(ts)
        cols = sorted(cols, reverse=True)  # senaste → äldst
        dbg["q_cols"] = len(cols)
        if not cols:
            out["_DEBUG_PS"] = dbg
            return out

        # revenue-map på normaliserade timestamps
        rev = qfin.loc["Total Revenue"][cols_raw].astype(float)
        rev_map = { _to_naive_ts(c): float(rev[c]) if pd.notna(rev[c]) else float("nan") for c in cols_raw }

        # SEC shares
        cik = _ticker_to_cik_any(ticker)
        dbg["sec_cik"] = cik
        shares_map = {}
        if cik:
            facts = _fetch_companyfacts(cik)
            if facts:
                shares_map = _extract_shares_history(facts)
                dbg["sec_shares_pts"] = len(shares_map)

        labels = [("P/S Q1","P/S Q1 datum","Källa P/S Q1"),
                  ("P/S Q2","P/S Q2 datum","Källa P/S Q2"),
                  ("P/S Q3","P/S Q3 datum","Källa P/S Q3"),
                  ("P/S Q4","P/S Q4 datum","Källa P/S Q4")]
        used_ps_q1 = None

        for i, (lab_ps, lab_dt, lab_src) in enumerate(labels):
            if i >= len(cols): break
            c = cols[i]
            window_vals = []
            for j in range(i, min(i+4, len(cols))):
                window_vals.append(rev_map.get(cols[j], float("nan")))
            if len(window_vals) < 4 or any(pd.isna(x) for x in window_vals):
                continue

            px = _nearest_price_on_or_after(t, c)
            if px is None:
                continue
            dbg["price_hits"] += 1

            end_iso = _clean_iso(c)
            sh_sec = _nearest_shares_for_date(end_iso, shares_map)
            if sh_sec and sh_sec > 0:
                sh = sh_sec; src = "Computed/SEC-shares"
            else:
                sh = shares_now; src = "Computed/Current-shares"

            ttm = float(sum(window_vals))
            if ttm <= 0 or not sh:
                continue
            dbg["ttm_points"] += 1
            ps = round((px * float(sh))/ttm, 3)
            out[lab_ps] = ps
            out[lab_dt] = end_iso
            out[lab_src] = src
            if i == 0:
                used_ps_q1 = (ps, src)

        ps_ttm = info.get("priceToSalesTrailing12Months")
        if ps_ttm and ps_ttm > 0:
            out["P/S"] = float(ps_ttm); out["Källa P/S"] = "Yahoo/ps_ttm"; dbg["ps_source"] = "yahoo_ps_ttm"
        elif used_ps_q1:
            out["P/S"] = float(used_ps_q1[0]); out["Källa P/S"] = used_ps_q1[1]; dbg["ps_source"] = "computed_q1"
        else:
            try:
                mc = info.get("marketCap")
                if not mc:
                    px_now = info.get("regularMarketPrice")
                    if not px_now:
                        h = t.history(period="5d")
                        if not h.empty: px_now = float(h["Close"].iloc[-1])
                    if px_now and shares_now:
                        mc = float(px_now) * float(shares_now)
                fin = getattr(t, "financials", None)
                ttm_rev = None
                if isinstance(fin, pd.DataFrame) and not fin.empty and "Total Revenue" in fin.index:
                    ttm_rev = float(fin.loc["Total Revenue"].dropna().iloc[-1])
                if mc and ttm_rev and ttm_rev > 0:
                    out["P/S"] = round(float(mc)/float(ttm_rev), 3); out["Källa P/S"] = "Fallback/MC_over_TTM"; dbg["ps_source"] = "fallback_mc_over_ttm"
            except Exception:
                pass
    except Exception:
        pass
    out["_DEBUG_PS"] = dbg
    return out

# --------- Yahoo fält + logg till session_state ---------
def hamta_yahoo_fält(ticker: str) -> dict:
    out = {
        "Bolagsnamn":"", "Aktuell kurs":0.0, "Valuta":"USD",
        "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0,
        "Utestående aktier":0.0,
        "Källa Aktuell kurs":"", "Källa Utestående aktier":""
    }
    log = {"ticker": ticker, "ts": now_stamp(), "source": "yahoo+sec", "yahoo": {}, "sec": {}, "ps": {}, "summary": ""}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
            log["yahoo"]["info_keys_sample"] = sorted(list(info.keys()))[:20]
        except Exception as e:
            log["yahoo"]["info_error"] = str(e)

        pris = info.get("regularMarketPrice")
        if pris is None:
            try:
                h = t.history(period="1d")
                if not h.empty and "Close" in h:
                    pris = float(h["Close"].iloc[-1])
                    out["Källa Aktuell kurs"] = "Yahoo/history"
                    log["yahoo"]["got_price_from"] = "history"
            except Exception as e:
                log["yahoo"]["history_error"] = str(e)
        else:
            out["Källa Aktuell kurs"] = "Yahoo/info"
            log["yahoo"]["got_price_from"] = "info"
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency")
        if valuta: out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn: out["Bolagsnamn"] = str(namn)

        div_rate = info.get("dividendRate")
        if div_rate is not None: out["Årlig utdelning"] = float(div_rate)

        shares = info.get("sharesOutstanding")
        if shares and float(shares) > 0:
            out["Utestående aktier"] = round(float(shares)/1e6, 2)
            out["Källa Utestående aktier"] = "Yahoo/info"
        log["yahoo"]["has_sharesOutstanding"] = bool(shares)

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

        ps_data = hamta_ps_kvartal(ticker)
        out.update(ps_data)

        log["ps"] = ps_data.get("_DEBUG_PS", {})
        log["sec"]["cik"] = log["ps"].get("sec_cik")
        log["sec"]["shares_points"] = log["ps"].get("sec_shares_pts", 0)
        log["summary"] = f"kurs:{log['yahoo'].get('got_price_from','-')}, ps_src:{log['ps'].get('ps_source','-')}, qcols:{log['ps'].get('q_cols',0)}, ttm_pts:{log['ps'].get('ttm_points',0)}, px_hits:{log['ps'].get('price_hits',0)}, sec_shares:{log['ps'].get('sec_shares_pts',0)}"
    except Exception as e:
        log["error"] = str(e)
    finally:
        logs = st.session_state.get("fetch_logs", [])
        logs.append(log)
        st.session_state["fetch_logs"] = logs[-200:]
    return out
