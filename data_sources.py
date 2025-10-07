# data_sources.py
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

# ---------- Tidsstämplar ----------
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

def _naive_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception:
        pass
    return df

# ---------- Live FX (Yahoo) ----------
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

# ---------- CAGR ----------
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

# ---------- SEC helpers ----------
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
    """end_date_iso -> shares"""
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

# ---------- SEC revenue fallback ----------
def _extract_sec_quarter_revenues(facts_json: dict) -> list[tuple[pd.Timestamp, float]]:
    """
    Hämta kvartalsintäkter från SEC:
    - fp (Q1..Q4) eller frame med Qx, annars duration 80–100 dagar
    Returnerar [(end_timestamp_naiv, quarter_revenue_float)] (senaste först).
    """
    tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet"
    ]
    out = []
    try:
        facts = facts_json.get("facts", {}).get("us-gaap", {})
        for tag in tags:
            if tag not in facts: continue
            units = facts[tag].get("units", {})
            for unit_name, values in units.items():
                if "share" in unit_name.lower(): 
                    continue
                for v in values:
                    val = v.get("val")
                    end = v.get("end")
                    start = v.get("start")
                    fp = (v.get("fp") or "").upper()
                    frame = (v.get("frame") or "")
                    if val is None or not end:
                        continue
                    is_quarter = False
                    if fp in {"Q1","Q2","Q3","Q4"}:
                        is_quarter = True
                    elif any(q in frame for q in ["Q1","Q2","Q3","Q4"]):
                        is_quarter = True
                    else:
                        try:
                            sd = pd.to_datetime(start) if start else None
                            ed = pd.to_datetime(end)
                            if sd is not None:
                                days = (ed - sd).days
                                if 80 <= days <= 100:
                                    is_quarter = True
                        except Exception:
                            pass
                    if not is_quarter:
                        continue
                    try:
                        ts = _to_naive_ts(end)
                        if ts is None: 
                            continue
                        out.append((ts, float(val)))
                    except Exception:
                        pass
            if out:
                break
    except Exception:
        return []
    if not out:
        return []
    # dedupe på end-date – behåll senast angivna
    ded = {}
    for ts, v in out:
        ded[ts] = v
    out2 = sorted(ded.items(), key=lambda x: x[0], reverse=True)
    return out2  # [(ts, val)] senaste först

# ---------- Yahoo helpers ----------
def _try_get_quarterly_revenue_df(t: yf.Ticker) -> pd.DataFrame | None:
    qfin = getattr(t, "quarterly_income_stmt", None)
    if not (isinstance(qfin, pd.DataFrame) and not qfin.empty and "Total Revenue" in qfin.index):
        qfin = getattr(t, "quarterly_financials", None)
    if isinstance(qfin, pd.DataFrame) and not qfin.empty and "Total Revenue" in qfin.index:
        return qfin.copy()
    return None

def _nearest_price_on_or_after(t: yf.Ticker, dt: pd.Timestamp, max_days: int = 30) -> tuple[float | None, str]:
    """
    Hitta ett pris nära 'dt'. Returnerar (pris, källa-tag):
      - 1d-after, 1d-before, 1wk-after, 1wk-nearest, 6mo-nearest, spot
    """
    try:
        # 1) på/efter (daglig)
        start1 = dt.strftime("%Y-%m-%d")
        end1   = (dt + pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")
        h1 = t.history(start=start1, end=end1, auto_adjust=False, interval="1d")
        h1 = _naive_index(h1)
        if not h1.empty:
            after = h1[h1.index >= dt]
            if not after.empty:
                return float(after["Close"].iloc[0]), "1d-after"

        # 2) före (daglig)
        start2 = (dt - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")
        end2   = dt.strftime("%Y-%m-%d")
        h2 = t.history(start=start2, end=end2, auto_adjust=False, interval="1d")
        h2 = _naive_index(h2)
        if not h2.empty:
            before = h2[h2.index < dt]
            if not before.empty:
                return float(before["Close"].iloc[-1]), "1d-before"

        # 3) veckodata
        start3 = (dt - pd.Timedelta(days=max_days+20)).strftime("%Y-%m-%d")
        end3   = (dt + pd.Timedelta(days=max_days+20)).strftime("%Y-%m-%d")
        hw = t.history(start=start3, end=end3, auto_adjust=False, interval="1wk")
        hw = _naive_index(hw)
        if not hw.empty:
            afterw = hw[hw.index >= dt]
            if not afterw.empty:
                return float(afterw["Close"].iloc[0]), "1wk-after"
            # närmast i veckan
            idx = (hw.index - dt).abs().argmin()
            return float(hw["Close"].iloc[idx]), "1wk-nearest"

        # 4) brett fönster 6 mån (daglig) – närmast
        h4 = t.history(period="6mo", auto_adjust=False, interval="1d")
        h4 = _naive_index(h4)
        if not h4.empty:
            idx = (h4.index - dt).abs().argmin()
            return float(h4["Close"].iloc[idx]), "6mo-nearest"

        # 5) spot från info
        try:
            info = t.info or {}
            spot = info.get("regularMarketPrice")
            if spot:
                return float(spot), "spot"
        except Exception:
            pass

    except Exception:
        return None, ""
    return None, ""

# ---------- P/S-historik + källor + DEBUG ----------
def hamta_ps_kvartal(ticker: str) -> dict:
    out = {
        "P/S": 0.0,
        "P/S Q1": 0.0, "P/S Q2": 0.0, "P/S Q3": 0.0, "P/S Q4": 0.0,
        "P/S Q1 datum":"", "P/S Q2 datum":"", "P/S Q3 datum":"", "P/S Q4 datum":"",
        "Källa P/S":"", "Källa P/S Q1":"", "Källa P/S Q2":"", "Källa P/S Q3":"", "Källa P/S Q4":"",
        "_DEBUG_PS": {}
    }
    # cutoff-år (default 6) kan styras i secrets
    try:
        SEC_CUTOFF_YEARS = int(st.secrets.get("SEC_CUTOFF_YEARS", 6))
    except Exception:
        SEC_CUTOFF_YEARS = 6
    cutoff_ts = pd.Timestamp.today().tz_localize(None) - pd.DateOffset(years=SEC_CUTOFF_YEARS)

    dbg = {"ticker": ticker, "ps_source": "-", "q_cols": 0, "ttm_points": 0, "price_hits": 0,
           "sec_cik": None, "sec_shares_pts": 0, "sec_rev_pts": 0, "sec_rev_pts_after_cutoff": 0,
           "cutoff_years": SEC_CUTOFF_YEARS}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try: info = t.info or {}
        except Exception: info = {}
        shares_now = float(info.get("sharesOutstanding") or 0.0)

        # ---- Yahoo-kvartal (primärt) ----
        qfin = _try_get_quarterly_revenue_df(t)
        used_ps_q1 = None
        if qfin is not None and not qfin.empty and "Total Revenue" in qfin.index:
            cols_raw = list(qfin.columns)
            cols = []
            for c in cols_raw:
                ts = _to_naive_ts(c)
                if ts is not None: cols.append(ts)
            cols = sorted(cols, reverse=True)
            dbg["q_cols"] = len(cols)

            rev = qfin.loc["Total Revenue"][cols_raw].astype(float)
            rev_map = { _to_naive_ts(c): float(rev[c]) if pd.notna(rev[c]) else float("nan") for c in cols_raw }

            # SEC shares (kan komma till användning redan här)
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

            for i, (lab_ps, lab_dt, lab_src) in enumerate(labels):
                if i >= len(cols): break
                c = cols[i]
                # hoppa över om för gammalt
                if c < cutoff_ts:
                    continue

                window_vals = []
                for j in range(i, min(i+4, len(cols))):
                    if cols[j] < cutoff_ts:
                        break
                    window_vals.append(rev_map.get(cols[j], float("nan")))
                if len(window_vals) < 4 or any(pd.isna(x) for x in window_vals):
                    continue

                px, px_src = _nearest_price_on_or_after(t, c)
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
                out[lab_src] = f"{src}+{px_src}"
                if i == 0:
                    used_ps_q1 = (ps, out[lab_src])

        # P/S TTM direkt från Yahoo om tillgängligt
        ps_ttm = info.get("priceToSalesTrailing12Months")
        if ps_ttm and ps_ttm > 0:
            out["P/S"] = float(ps_ttm); out["Källa P/S"] = "Yahoo/ps_ttm"; dbg["ps_source"] = "yahoo_ps_ttm"

        # ---- SEC fallback för kvartal (om Yahoo inte gav tillräckligt) ----
        q_filled = sum(1 for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"] if out[k] and out[k] > 0)
        if q_filled < 4:
            cik = dbg.get("sec_cik") or _ticker_to_cik_any(ticker)
            if cik:
                facts = _fetch_companyfacts(cik)
                if facts:
                    shares_map = _extract_shares_history(facts)
                    dbg["sec_shares_pts"] = len(shares_map)

                    sec_quarters = _extract_sec_quarter_revenues(facts)  # [(ts, val)] nyast->äldst
                    dbg["sec_rev_pts"] = len(sec_quarters)
                    # ---- NYTT: filtrera bort för gamla kvartal ----
                    try:
                        sec_quarters = [(ts, v) for ts, v in sec_quarters if ts >= cutoff_ts]
                    except Exception:
                        pass
                    dbg["sec_rev_pts_after_cutoff"] = len(sec_quarters)

                    if sec_quarters:
                        labels = [("P/S Q1","P/S Q1 datum","Källa P/S Q1"),
                                  ("P/S Q2","P/S Q2 datum","Källa P/S Q2"),
                                  ("P/S Q3","P/S Q3 datum","Källa P/S Q3"),
                                  ("P/S Q4","P/S Q4 datum","Källa P/S Q4")]
                        for i, (lab_ps, lab_dt, lab_src) in enumerate(labels):
                            if out[lab_ps] and out[lab_ps] > 0:
                                continue
                            if i >= len(sec_quarters):
                                break
                            end_ts, _ = sec_quarters[i]
                            if end_ts < cutoff_ts:
                                continue
                            window = sec_quarters[i:i+4]
                            if len(window) < 4:
                                continue
                            ttm = sum(v for _, v in window)
                            if ttm <= 0:
                                continue
                            px, px_src = _nearest_price_on_or_after(t, end_ts)
                            if px is None:
                                continue
                            dbg["price_hits"] += 1

                            end_iso = _clean_iso(end_ts)
                            sh_sec = _nearest_shares_for_date(end_iso, shares_map)
                            if sh_sec and sh_sec > 0:
                                sh = sh_sec; src = "SEC/revenue+SEC-shares"
                            else:
                                sh = shares_now; src = "SEC/revenue+Current-shares"

                            if not sh:
                                continue
                            ps = round((px * float(sh))/float(ttm), 3)
                            out[lab_ps] = ps
                            out[lab_dt] = end_iso
                            out[lab_src] = f"{src}+{px_src}"

                        # Om P/S (TTM) saknas helt – använd Q1 TTM från SEC
                        if (not out["P/S"] or out["P/S"] <= 0) and out["P/S Q1"] and out["P/S Q1"] > 0:
                            out["P/S"] = float(out["P/S Q1"])
                            out["Källa P/S"] = "SEC_fallback/q1_ttm"
                            dbg["ps_source"] = "sec_fallback"

        if dbg["ps_source"] == "-" and used_ps_q1:
            dbg["ps_source"] = "computed_q1"

        # sista fallback om allt annat misslyckas
        if (not out["P/S"] or out["P/S"] <= 0):
            try:
                mc = info.get("marketCap")
                px_now = info.get("regularMarketPrice")
                if not mc:
                    if not px_now:
                        h = t.history(period="5d")
                        h = _naive_index(h)
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

# ---------- Yahoo fält + logg ----------
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
                h = _naive_index(h)
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
        log["sec"]["rev_points"] = log["ps"].get("sec_rev_pts", 0)
        log["sec"]["rev_points_after_cutoff"] = log["ps"].get("sec_rev_pts_after_cutoff", 0)
        log["summary"] = (
            f"kurs:{log['yahoo'].get('got_price_from','-')}, ps_src:{log['ps'].get('ps_source','-')}, "
            f"qcols:{log['ps'].get('q_cols',0)}, ttm_pts:{log['ps'].get('ttm_points',0)}, "
            f"px_hits:{log['ps'].get('price_hits',0)}, sec_shares:{log['ps'].get('sec_shares_pts',0)}, "
            f"sec_rev:{log['ps'].get('sec_rev_pts',0)}/{log['ps'].get('sec_rev_pts_after_cutoff',0)} "
            f"(cutoff {log['ps'].get('cutoff_years',6)}y)"
        )
    except Exception as e:
        log["error"] = str(e)
    finally:
        logs = st.session_state.get("fetch_logs", [])
        logs.append(log)
        st.session_state["fetch_logs"] = logs[-200:]
    return out
