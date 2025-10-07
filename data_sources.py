# data_sources.py
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import re
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

def _secret_bool(name: str, default: bool = False) -> bool:
    try:
        v = st.secrets.get(name, default)
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)): return bool(v)
        if isinstance(v, str): return v.strip().lower() in {"1","true","yes","y"}
    except Exception:
        pass
    return default

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

# ---------- SEC revenues (companyfacts, kvartal) ----------
def _extract_sec_quarter_revenues(facts_json: dict) -> list[tuple[pd.Timestamp, float]]:
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
    # dedupe + sort nyast->äldst
    ded = {ts: v for ts, v in out}
    return sorted(ded.items(), key=lambda x: x[0], reverse=True)

# ---------- Yahoo helpers ----------
def _try_get_quarterly_revenue_df(t: yf.Ticker) -> pd.DataFrame | None:
    qfin = getattr(t, "quarterly_income_stmt", None)
    if not (isinstance(qfin, pd.DataFrame) and not qfin.empty and "Total Revenue" in qfin.index):
        qfin = getattr(t, "quarterly_financials", None)
    if isinstance(qfin, pd.DataFrame) and not qfin.empty and "Total Revenue" in qfin.index:
        return qfin.copy()
    return None

def _nearest_price_on_or_after(t: yf.Ticker, dt: pd.Timestamp, max_days: int = 30) -> tuple[float | None, str]:
    """Returnerar (pris, källa-tag). Provar 1d-after, 1d-before, 1wk-after, 1wk-nearest, 6mo-nearest, spot."""
    try:
        start1 = dt.strftime("%Y-%m-%d")
        end1   = (dt + pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")
        h1 = t.history(start=start1, end=end1, auto_adjust=False, interval="1d")
        h1 = _naive_index(h1)
        if not h1.empty:
            after = h1[h1.index >= dt]
            if not after.empty:
                return float(after["Close"].iloc[0]), "1d-after"

        start2 = (dt - pd.Timedelta(days=max_days)).strftime("%Y-%m-%d")
        end2   = dt.strftime("%Y-%m-%d")
        h2 = t.history(start=start2, end=end2, auto_adjust=False, interval="1d")
        h2 = _naive_index(h2)
        if not h2.empty:
            before = h2[h2.index < dt]
            if not before.empty:
                return float(before["Close"].iloc[-1]), "1d-before"

        start3 = (dt - pd.Timedelta(days=max_days+20)).strftime("%Y-%m-%d")
        end3   = (dt + pd.Timedelta(days=max_days+20)).strftime("%Y-%m-%d")
        hw = t.history(start=start3, end=end3, auto_adjust=False, interval="1wk")
        hw = _naive_index(hw)
        if not hw.empty:
            afterw = hw[hw.index >= dt]
            if not afterw.empty:
                return float(afterw["Close"].iloc[0]), "1wk-after"
            idx = (hw.index - dt).abs().argmin()
            return float(hw["Close"].iloc[idx]), "1wk-nearest"

        h4 = t.history(period="6mo", auto_adjust=False, interval="1d")
        h4 = _naive_index(h4)
        if not h4.empty:
            idx = (h4.index - dt).abs().argmin()
            return float(h4["Close"].iloc[idx]), "6mo-nearest"

        info = t.info or {}
        spot = info.get("regularMarketPrice")
        if spot: return float(spot), "spot"
    except Exception:
        return None, ""
    return None, ""

# ---------- iXBRL parsing ----------
def hamta_sec_filing_lankar(ticker: str, forms=("10-Q","10-K","8-K"), limit: int = 6) -> list[dict]:
    """Senaste filing-länkar för ticker → [{form,date,url,viewer,cik,accession}]"""
    cik = _ticker_to_cik_any(ticker)
    if not cik:
        return []
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=_sec_headers(), timeout=20)
        r.raise_for_status()
        j = r.json()
        recent = (j.get("filings", {}) or {}).get("recent", {})
        forms_list = list(recent.get("form", []))
        dates      = list(recent.get("filingDate", []))
        acc        = list(recent.get("accessionNumber", []))
        prim       = list(recent.get("primaryDocument", []))
        out = []
        for i, f in enumerate(forms_list):
            if f not in forms:
                continue
            an = str(acc[i]); primdoc = str(prim[i])
            archive = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{an.replace('-','')}/{primdoc}"
            viewer  = f"https://www.sec.gov/ixviewer/doc?action=display&source={archive}"
            out.append({"form": f, "date": str(dates[i]), "url": archive, "viewer": viewer, "cik": cik, "accession": an})
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []

def _parse_ixbrl_contexts(html: str) -> dict:
    """
    Returnerar {context_id: (start_ts, end_ts, is_duration)} från iXBRL-dokumentet.
    """
    ctx = {}
    for m in re.finditer(r'<xbrli:context[^>]*id="([^"]+)"[^>]*>(.*?)</xbrli:context>', html, flags=re.I|re.S):
        cid, block = m.group(1), m.group(2)
        inst = re.search(r'<xbrli:instant>([^<]+)</xbrli:instant>', block, flags=re.I)
        if inst:
            try:
                end = pd.to_datetime(inst.group(1)).tz_localize(None)
                ctx[cid] = (None, end, False)
            except Exception:
                pass
            continue
        start = re.search(r'<xbrli:startDate>([^<]+)</xbrli:startDate>', block, flags=re.I)
        end   = re.search(r'<xbrli:endDate>([^<]+)</xbrli:endDate>', block, flags=re.I)
        if start and end:
            try:
                s = pd.to_datetime(start.group(1)).tz_localize(None)
                e = pd.to_datetime(end.group(1)).tz_localize(None)
                ctx[cid] = (s, e, True)
            except Exception:
                pass
    return ctx

def _parse_ixbrl_revenues(html: str) -> list[tuple[pd.Timestamp, float]]:
    """
    Plockar kvartalsintäkter ur iXBRL (ix:nonFraction) för vanliga revenue-taggar.
    Returnerar [(end_ts, value), ...], sorterad nyast->äldst.
    """
    tags = [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueGoodsNet",
    ]
    name_re = r'(' + "|".join(re.escape(t) for t in tags) + r')'
    contexts = _parse_ixbrl_contexts(html)
    out = []

    # fånga ix:nonFraction med value i text eller i value-attribut
    for m in re.finditer(
        rf'<ix:nonFraction[^>]*name="(?P<name>{name_re})"[^>]*contextRef="(?P<ctx>[^"]+)"[^>]*([^>]*?scale="(?P<scale>-?\d+)")?[^>]*?(?:value="(?P<val_attr>[^"]+)")?[^>]*>(?P<val_text>.*?)</ix:nonFraction>',
        html, flags=re.I|re.S
    ):
        name = m.group("name")
        ctx_id = m.group("ctx")
        scale = int(m.group("scale") or 0)
        raw = (m.group("val_attr") or m.group("val_text") or "").strip()
        raw = re.sub(r'<[^>]+>', '', raw)  # ta bort ev. taggar i textnoder
        # hantera (1,234) och 1,234.56
        neg = raw.startswith("(") and raw.endswith(")")
        raw = raw.strip("()").replace("\xa0","").replace(",", "")
        try:
            val = float(raw)
            if neg: val = -val
            if scale: val = val * (10 ** scale)
        except Exception:
            continue
        if ctx_id not in contexts: 
            continue
        s, e, is_dur = contexts[ctx_id]
        if not is_dur or e is None:
            continue
        # kvartalsfilter (ca 90 dagar)
        days = None if s is None else (e - s).days
        if days is None or not (80 <= days <= 100):
            continue
        out.append((e, float(val)))

    if not out:
        return []
    # dedupe per end-date, behåll största värdet (skydd mot dubbletter på flera revenue-taggar)
    ded = {}
    for ts, v in out:
        if (ts not in ded) or (abs(v) > abs(ded[ts])):
            ded[ts] = v
    return sorted(ded.items(), key=lambda x: x[0], reverse=True)

def hamta_sec_ixbrl_quarter_revenues(ticker: str, max_docs: int = 6) -> list[tuple[pd.Timestamp, float]]:
    """
    Hämtar HTML för senaste 10-Q/10-K (ev. 8-K om iXBRL finns), parser iXBRL och
    returnerar [(end_ts, revenue)], nyast->äldst.
    """
    links = hamta_sec_filing_lankar(ticker, forms=("10-Q","10-K","8-K"), limit=max_docs)
    all_pairs = []
    for L in links:
        try:
            r = requests.get(L["url"], headers=_sec_headers(), timeout=20)
            if not r.ok: continue
            html = r.text
            pairs = _parse_ixbrl_revenues(html)
            all_pairs.extend(pairs)
        except Exception:
            continue
    if not all_pairs:
        return []
    # dedupe nyast->äldst
    ded = {}
    for ts, v in all_pairs:
        if (ts not in ded) or (abs(v) > abs(ded[ts])):
            ded[ts] = v
    return sorted(ded.items(), key=lambda x: x[0], reverse=True)

# ---------- P/S-historik + källor + DEBUG ----------
def hamta_ps_kvartal(ticker: str) -> dict:
    """
    Hämtar/beräknar P/S (TTM) per kvartal för de 4 senaste kvartalen.
    Prioritet: Yahoo kvartalsdata → SEC iXBRL → SEC companyfacts. Därefter MC/TTM fallback.
    Cutoff styrs av SEC_CUTOFF_YEARS (default 6). Om SEC_ALLOW_BACKFILL_BEYOND_CUTOFF är True
    fylls resterande kvartal med äldre data (före cutoff) och källan märks [pre-cutoff].
    """
    out = {
        "P/S": 0.0,
        "P/S Q1": 0.0, "P/S Q2": 0.0, "P/S Q3": 0.0, "P/S Q4": 0.0,
        "P/S Q1 datum":"", "P/S Q2 datum":"", "P/S Q3 datum":"", "P/S Q4 datum":"",
        "Källa P/S":"", "Källa P/S Q1":"", "Källa P/S Q2":"", "Källa P/S Q3":"", "Källa P/S Q4":"",
        "_DEBUG_PS": {}
    }
    try:
        try:
            SEC_CUTOFF_YEARS = int(st.secrets.get("SEC_CUTOFF_YEARS", 6))
        except Exception:
            SEC_CUTOFF_YEARS = 6
        ALLOW_BACKFILL = _secret_bool("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF", False)

        cutoff_ts = pd.Timestamp.today().tz_localize(None) - pd.DateOffset(years=SEC_CUTOFF_YEARS)
        dbg = {"ticker": ticker, "ps_source": "-", "q_cols": 0, "ttm_points": 0, "price_hits": 0,
               "sec_cik": None, "sec_shares_pts": 0, "sec_rev_pts": 0, "sec_rev_pts_after_cutoff": 0,
               "sec_ixbrl_pts": 0, "cutoff_years": SEC_CUTOFF_YEARS, "backfill_used": False}

        t = yf.Ticker(ticker)
        info = {}
        try: info = t.info or {}
        except Exception: info = {}
        shares_now = float(info.get("sharesOutstanding") or 0.0)

        # gemensam fyllare
        def _fill_from_quarters(seq: list[tuple[pd.Timestamp, float]], base_src: str,
                                apply_cutoff: bool, mark_old: bool,
                                shares_map: dict | None, q_already: set[str]) -> int:
            filled = 0
            labels = [("P/S Q1","P/S Q1 datum","Källa P/S Q1"),
                      ("P/S Q2","P/S Q2 datum","Källa P/S Q2"),
                      ("P/S Q3","P/S Q3 datum","Källa P/S Q3"),
                      ("P/S Q4","P/S Q4 datum","Källa P/S Q4")]
            for i, (lab_ps, lab_dt, lab_src) in enumerate(labels):
                if i >= len(seq): break
                end_ts, _ = seq[i]
                if apply_cutoff and end_ts < cutoff_ts: 
                    continue
                # bygg TTM-fönster
                window = seq[i:i+4]
                if len(window) < 4: 
                    continue
                ttm = sum(v for _, v in window)
                if ttm <= 0: 
                    continue
                # pris vid/efter periodslut
                px, px_src = _nearest_price_on_or_after(t, end_ts)
                if px is None: 
                    continue
                dbg["price_hits"] += 1
                # aktier (SEC-shares när tillgängligt)
                end_iso = _clean_iso(end_ts)
                sh = None
                if shares_map:
                    sh = _nearest_shares_for_date(end_iso, shares_map)
                if not sh or sh <= 0:
                    sh = shares_now
                    src = f"{base_src}+Current-shares+{px_src}"
                else:
                    src = f"{base_src}+SEC-shares+{px_src}"
                if end_iso in q_already:
                    continue
                ps = round((px * float(sh))/float(ttm), 3)
                out[lab_ps] = ps
                out[lab_dt] = end_iso
                out[lab_src] = src + (" [pre-cutoff]" if mark_old else "")
                q_already.add(end_iso)
                dbg["ttm_points"] += 1
                filled += 1
            return filled

        used_q_dates = set()

        # ---- 1) Yahoo quarterly primärt ----
        qfin = _try_get_quarterly_revenue_df(t)
        if qfin is not None and not qfin.empty and "Total Revenue" in qfin.index:
            cols_raw = list(qfin.columns)
            cols = [ _to_naive_ts(c) for c in cols_raw ]
            cols = [c for c in cols if c is not None]
            cols = sorted(cols, reverse=True)
            dbg["q_cols"] = len(cols)
            rev = qfin.loc["Total Revenue"][cols_raw].astype(float)
            yahoo_quarters = []
            for c in cols:
                try:
                    v = float(rev[pd.Timestamp(c)])
                    yahoo_quarters.append((pd.Timestamp(c), v))
                except Exception:
                    continue

            # hämta ev. SEC-shares för bättre kvot
            cik = _ticker_to_cik_any(ticker); dbg["sec_cik"] = cik
            shares_map = {}
            if cik:
                facts = _fetch_companyfacts(cik)
                if facts:
                    shares_map = _extract_shares_history(facts)
                    dbg["sec_shares_pts"] = len(shares_map)

            # fyll med cutoff, sedan backfill om tillåtet
            _fill_from_quarters(yahoo_quarters, base_src="Computed/Yahoo-revenue", 
                                apply_cutoff=True, mark_old=False,
                                shares_map=shares_map, q_already=used_q_dates)
            if sum(out[k] > 0 for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]) < 4 and ALLOW_BACKFILL:
                add = _fill_from_quarters(yahoo_quarters, base_src="Computed/Yahoo-revenue",
                                          apply_cutoff=False, mark_old=True,
                                          shares_map=shares_map, q_already=used_q_dates)
                if add > 0: dbg["backfill_used"] = True

        # P/S TTM direkt från Yahoo
        ps_ttm = info.get("priceToSalesTrailing12Months")
        if ps_ttm and ps_ttm > 0:
            out["P/S"] = float(ps_ttm); out["Källa P/S"] = "Yahoo/ps_ttm"; dbg["ps_source"] = "yahoo_ps_ttm"

        # ---- 2) SEC iXBRL fallback (fångar allra senaste kvartalet) ----
        if sum(out[k] > 0 for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]) < 4:
            ix_quarters = hamta_sec_ixbrl_quarter_revenues(ticker, max_docs=6)
            dbg["sec_ixbrl_pts"] = len(ix_quarters)
            # shares via companyfacts om möjligt
            cik = dbg.get("sec_cik") or _ticker_to_cik_any(ticker)
            shares_map = {}
            if cik:
                facts = _fetch_companyfacts(cik)
                if facts:
                    shares_map = _extract_shares_history(facts)
                    dbg["sec_shares_pts"] = max(dbg["sec_shares_pts"], len(shares_map))
            # fyll med cutoff → ev. backfill
            _fill_from_quarters(ix_quarters, base_src="SEC/ixbrl-revenue",
                                apply_cutoff=True, mark_old=False,
                                shares_map=shares_map, q_already=used_q_dates)
            if sum(out[k] > 0 for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]) < 4 and ALLOW_BACKFILL:
                add = _fill_from_quarters(ix_quarters, base_src="SEC/ixbrl-revenue",
                                          apply_cutoff=False, mark_old=True,
                                          shares_map=shares_map, q_already=used_q_dates)
                if add > 0: dbg["backfill_used"] = True

        # ---- 3) SEC companyfacts (som tidigare) ----
        if sum(out[k] > 0 for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]) < 4:
            cik = dbg.get("sec_cik") or _ticker_to_cik_any(ticker)
            if cik:
                facts = _fetch_companyfacts(cik)
                if facts:
                    shares_map = _extract_shares_history(facts)
                    dbg["sec_shares_pts"] = max(dbg["sec_shares_pts"], len(shares_map))
                    sec_quarters = _extract_sec_quarter_revenues(facts)
                    dbg["sec_rev_pts"] = len(sec_quarters)
                    seq_after = [(ts, v) for ts, v in sec_quarters if ts >= cutoff_ts]
                    dbg["sec_rev_pts_after_cutoff"] = len(seq_after)

                    _fill_from_quarters(seq_after, base_src="SEC/revenue",
                                        apply_cutoff=False, mark_old=False,
                                        shares_map=shares_map, q_already=used_q_dates)
                    if sum(out[k] > 0 for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]) < 4 and ALLOW_BACKFILL:
                        add = _fill_from_quarters(sec_quarters, base_src="SEC/revenue",
                                                  apply_cutoff=False, mark_old=True,
                                                  shares_map=shares_map, q_already=used_q_dates)
                        if add > 0: dbg["backfill_used"] = True

                    if (not out["P/S"] or out["P/S"] <= 0) and out["P/S Q1"] and out["P/S Q1"] > 0:
                        out["P/S"] = float(out["P/S Q1"])
                        out["Källa P/S"] = "SEC_fallback/q1_ttm" + (" [pre-cutoff]" if dbg["backfill_used"] else "")
                        dbg["ps_source"] = "sec_fallback"

        # ---- 4) Sista fallback: MC/TTM ----
        if (not out["P/S"] or out["P/S"] <= 0):
            try:
                mc = info.get("marketCap")
                px_now = info.get("regularMarketPrice")
                if not mc:
                    if not px_now:
                        h = t.history(period="5d"); h = _naive_index(h)
                        if not h.empty: px_now = float(h["Close"].iloc[-1])
                    if px_now and shares_now: mc = float(px_now) * float(shares_now)
                fin = getattr(t, "financials", None)
                ttm_rev = None
                if isinstance(fin, pd.DataFrame) and not fin.empty and "Total Revenue" in fin.index:
                    ttm_rev = float(fin.loc["Total Revenue"].dropna().iloc[-1])
                if mc and ttm_rev and ttm_rev > 0:
                    out["P/S"] = round(float(mc)/float(ttm_rev), 3)
                    out["Källa P/S"] = "Fallback/MC_over_TTM"; dbg["ps_source"] = "fallback_mc_over_ttm"
            except Exception:
                pass

        out["_DEBUG_PS"] = dbg
        return out

    except Exception:
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
        log["sec"]["ixbrl_points"] = log["ps"].get("sec_ixbrl_pts", 0)
        log["summary"] = (
            f"kurs:{log['yahoo'].get('got_price_from','-')}, ps_src:{log['ps'].get('ps_source','-')}, "
            f"qcols:{log['ps'].get('q_cols',0)}, ttm_pts:{log['ps'].get('ttm_points',0)}, "
            f"px_hits:{log['ps'].get('price_hits',0)}, sec_ixbrl:{log['ps'].get('sec_ixbrl_pts',0)}, "
            f"sec_shares:{log['ps'].get('sec_shares_pts',0)}, "
            f"sec_rev:{log['ps'].get('sec_rev_pts',0)}/{log['ps'].get('sec_rev_pts_after_cutoff',0)} "
            f"(cutoff {log['ps'].get('cutoff_years',6)}y, backfill={log['ps'].get('backfill_used',False)})"
        )
    except Exception as e:
        log["error"] = str(e)
    finally:
        logs = st.session_state.get("fetch_logs", [])
        logs.append(log)
        st.session_state["fetch_logs"] = logs[-200:]
    return out
