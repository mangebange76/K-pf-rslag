# app.py — Del 1/7
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
import requests
from datetime import datetime, timedelta
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build  # Drive-kopior/städning

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# --- Lokal Stockholm-tid om pytz finns (annars systemtid) ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
    def _ts_datetime():
        return datetime.now(TZ_STHLM)
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")
    def _ts_datetime():
        return datetime.now()

def _ts_str():
    return _ts_datetime().strftime("%Y%m%d-%H%M%S")

# --- Google Sheets-koppling ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
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
        ws.update([["Valuta","Kurs"]])
        return ws

def hamta_data():
    sheet = skapa_koppling()
    data = _with_backoff(sheet.get_all_records)
    return pd.DataFrame(data)

# --- Snapshot-backup till ny flik + CSV-export ---
def backup_snapshot_sheet(df: pd.DataFrame, base_sheet_name: str = SHEET_NAME) -> str:
    ss = get_spreadsheet()
    snap_title = f"BACKUP_{base_sheet_name}_{_ts_str()}"
    rows = max(2, len(df) + 5)
    cols = max(2, len(df.columns) + 2)
    ws = ss.add_worksheet(title=snap_title, rows=rows, cols=cols)
    values = [list(df.columns)]
    if not df.empty:
        values += df.astype(str).values.tolist()
    _with_backoff(ws.update, values)
    return snap_title

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# --- Standardkurser (fallback) ---
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

# --- Tidsstämplar: vilka fält vi spårar ---
TS_FIELDS = {
    "Utestående aktier":  "TS Utestående aktier",
    "P/S":                "TS P/S",
    "P/S Q1":             "TS P/S Q1",
    "P/S Q2":             "TS P/S Q2",
    "P/S Q3":             "TS P/S Q3",
    "P/S Q4":             "TS P/S Q4",
    "Omsättning idag":    "TS Omsättning idag",
    "Omsättning nästa år":"TS Omsättning nästa år",
}

# --- Kolumnschema (inkl. TS-kolumner + meta) ---
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    *TS_FIELDS.values(),
    "Senast manuellt uppdaterad",
    "Senast auto-uppdaterad",
    "Senast uppdaterad källa",
]

# app.py — Del 2/7
@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    ws = skapa_rates_sheet_if_missing()
    rows = _with_backoff(ws.get_all_records)
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

def hamta_valutakurser(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# --- TS/diff-hjälpare ---
def _ts_human():
    try:
        return _ts_datetime().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M")

def _is_close(a, b, tol=1e-9):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a) == str(b)

def mark_field_if_changed(df: pd.DataFrame, row_idx: int, field: str, new_value) -> bool:
    """
    Sätt df[field] = new_value ENDAST om värdet ändrats. Stämpla TS-kolumnen om fältet spåras.
    Returnerar True om någon verklig ändring skedde.
    """
    old = df.at[row_idx, field] if field in df.columns else None
    changed = False
    try:
        changed = not _is_close(float(old), float(new_value))
    except Exception:
        changed = str(old) != str(new_value)
    if changed:
        df.at[row_idx, field] = new_value
        ts_col = TS_FIELDS.get(field)
        if ts_col:
            df.at[row_idx, ts_col] = _ts_human()
    return changed

# Fält vi diffar i körningsrapporten
UPPDATERBARA_FALT = [
    "Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)",
    "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
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
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta",
              "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
              *TS_FIELDS.values()]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# app.py — Del 3/7
# --- FMP: konfiguration + hjälpare ---
FMP_BASE = st.secrets.get("FMP_BASE", "https://financialmodelingprep.com")
FMP_KEY  = st.secrets.get("FMP_API_KEY", "")
FMP_CALL_DELAY = float(st.secrets.get("FMP_CALL_DELAY", 2.5))      # skonsam default
FMP_BLOCK_MINUTES = float(st.secrets.get("FMP_BLOCK_MINUTES", 20)) # paus efter 429

def _fmp_get(path: str, params=None, stable: bool = True):
    """
    Throttlad GET med enkel backoff + 'circuit breaker' vid 429.
    Returnerar (json, statuscode). Anropa med path t.ex. 'api/v3/quote/AAPL'.
    """
    # blockera om vi nyligen fått 429
    block_until = st.session_state.get("fmp_block_until")
    if block_until and _ts_datetime() < block_until:
        return None, 429

    params = (params or {}).copy()
    if FMP_KEY:
        params["apikey"] = FMP_KEY
    url = f"{FMP_BASE}/{path}"

    delays = [0.0, 1.2, 2.5]  # backoff
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
    """
    Försök validera symbol via quote-short, annars search. Faller tillbaka till upper().
    """
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
    Fullare variant: försök hämta namn/valuta/pris/shares, P/S (TTM, key-metrics, beräkning),
    P/S Q1–Q4 (ratios quarterly) samt analytikerestimat (om plan tillåter).
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

    # Shares fallback
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

    # P/S = marketCap / revenueTTM
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

    # P/S Q1–Q4 (enl. FMP ratios, om tillgängligt – annars räknas via SEC/Yahoo i vår SEC-funktion)
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

    # Analytikerestimat (kan kräva betalplan) – fyll om >0
    est, est_sc = _fmp_get("api/v3/analyst-estimates", {"symbol": sym, "period": "annual", "limit": 2}, stable=False)
    out["_debug"]["analyst_estimates_sc"] = est_sc
    def _pick_rev(obj: dict) -> float:
        for k in ("revenueAvg", "revenueMean", "revenue", "revenueEstimateAvg"):
            v = obj.get(k)
            if v is not None:
                try: return float(v)
                except: return 0.0
        return 0.0
    if isinstance(est, list) and est:
        cur = est[0] if len(est) >= 1 else {}
        nxt = est[1] if len(est) >= 2 else {}
        r_cur = _pick_rev(cur); r_nxt = _pick_rev(nxt)
        if r_cur > 0: out["Omsättning idag"] = r_cur / 1e6
        if r_nxt > 0: out["Omsättning nästa år"] = r_nxt / 1e6
    out["_est_status"] = est_sc
    return out

# --- SEC (US) & Finnhub helpers ---------------------------------------------
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

def _sec_cik_for(ticker: str) -> str | None:
    return _sec_ticker_map().get(str(ticker).upper())

def _sec_companyfacts(cik10: str):
    return _sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json")

def _pick_fact_number(fact: dict, unit_keys=("USD", "USD_shares", "shares"), want_quarterly=True):
    """
    Returnerar lista av datapunkter (period_end, value) för önskad unit.
    Väljer helst kvartalsdata (10-Q), men godtar årliga om kvartal saknas.
    """
    if not isinstance(fact, dict): return []
    units = fact.get("units") or {}
    for uk in unit_keys:
        arr = units.get(uk)
        if not isinstance(arr, list):
            continue
        rows = []
        for it in arr:
            form = (it.get("form") or "").upper()
            end = it.get("end")
            val = it.get("val")
            if end and val is not None:
                rows.append((end, form, val))
        if not rows:
            continue
        rows.sort(key=lambda x: x[0], reverse=True)
        if want_quarterly:
            q = [(e, v) for (e, f, v) in rows if "10-Q" in f or "10-Q/A" in f]
            if q:
                return q
        return [(e, v) for (e, _, v) in rows]
    return []

def _sec_latest_shares(facts: dict) -> float:
    """
    Försöker först point-in-time aktier. Faller tillbaka till weighted average om nödvändigt.
    Returnerar antal aktier (styck), inte miljoner.
    """
    dei = (facts.get("facts") or {}).get("dei", {})
    gaap = (facts.get("facts") or {}).get("us-gaap", {})

    # 1) Point-in-time (bäst)
    point_candidates = [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
        "EntityCommonSharesOutstanding",
        "ShareIssued",
    ]
    for c in point_candidates:
        fact = dei.get(c) or gaap.get(c)
        if fact:
            pts = _pick_fact_number(fact, unit_keys=("shares","USD_shares","shares"), want_quarterly=False)
            for _, v in pts:
                try:
                    v = float(v)
                    if v > 0:
                        return v
                except Exception:
                    pass

    # 2) Weighted average (näst bäst)
    wa_candidates = [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstandingBasic",
    ]
    for c in wa_candidates:
        fact = gaap.get(c)
        if fact:
            pts = _pick_fact_number(fact, unit_keys=("shares","USD_shares","shares"), want_quarterly=True)
            for _, v in pts:
                try:
                    v = float(v)
                    if v > 0:
                        return v
                except Exception:
                    pass
    return 0.0

# --- PATCH: historisk TTM-P/S per kvartal -----------------------------------
from datetime import datetime as _dt, timedelta as _td  # för datumtolkning och prisfönster

def _parse_iso(d: str):
    try:
        return _dt.fromisoformat(d.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return _dt.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

def _sec_quarterly_revenues_dated(facts: dict, max_quarters: int = 20) -> list[tuple]:
    """
    Hämtar upp till max_quarters *kvartalsvisa* (3-mån) intäkter från SEC (10-Q),
    filtrerar på duration ~70–100 dagar. Returnerar lista [(end_date, value)] sorterad nyast→äldst.
    """
    gaap = (facts.get("facts") or {}).get("us-gaap", {})
    cand_names = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "Revenue",
    ]
    rows = []
    for name in cand_names:
        fact = gaap.get(name)
        if not fact:
            continue
        units = (fact.get("units") or {}).get("USD", [])
        tmp = []
        for it in units:
            form = (it.get("form") or "").upper()
            if "10-Q" not in form:
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
        if tmp:
            # dedupe per end-datum, behåll senaste
            ded = {}
            for end, v in tmp:
                ded[end] = v
            rows = sorted(ded.items(), key=lambda t: t[0], reverse=True)
            break  # använd första GAAP-nyckeln som gav träff
    return rows[:max_quarters]

def _sec_quarterly_revenues(facts: dict) -> list[float]:
    """Bakåtkompatibel wrapper: returnerar bara värden (4 senaste)."""
    rows = _sec_quarterly_revenues_dated(facts, max_quarters=4)
    return [v for (_, v) in rows]

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
            price = None
            for j in range(len(idx)-1, -1, -1):
                if idx[j] <= d:
                    try:
                        price = float(closes[j])
                    except Exception:
                        price = None
                    break
            if price is not None:
                out[d] = price
        return out
    except Exception:
        return {}

def _ttm_windows(values: list[tuple], need: int = 4) -> list[tuple]:
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

def hamta_sec_yahoo_combo(ticker: str) -> dict:
    """
    US-bolag: Shares + kvartalsintäkter från SEC; pris/valuta/namn från Yahoo.
    Beräknar:
      - Utestående aktier (M)
      - P/S (TTM) med *aktuellt* pris/MCAP
      - P/S Q1–Q4 som *historisk TTM-P/S* vid respektive kvartals slut (pris nära datumet × shares).
    """
    out = {}
    cik = _sec_cik_for(ticker)
    if not cik:
        return out  # ej US

    facts, sc = _sec_companyfacts(cik)
    if sc != 200 or not isinstance(facts, dict):
        return out

    # Yahoo-basics (namn/valuta/aktuellt pris)
    y = hamta_yahoo_fält(ticker)
    for k in ("Bolagsnamn", "Valuta", "Aktuell kurs"):
        if y.get(k): out[k] = y[k]

    # Shares: SEC point-in-time → fallback Yahoo
    shares = _sec_latest_shares(facts)
    if shares <= 0:
        try:
            y_sh = _yfi_get(yf.Ticker(ticker), "shares_outstanding", "shares", "sharesOutstanding")
            if y_sh and float(y_sh) > 0:
                shares = float(y_sh)
        except Exception:
            pass
    if shares > 0:
        out["Utestående aktier"] = shares / 1e6

    # Market cap (nu) från Yahoo (helst), annars pris*shares
    mcap_now = _yfi_get(yf.Ticker(ticker), "market_cap", "marketCap")
    try:
        mcap_now = float(mcap_now or 0.0)
    except Exception:
        mcap_now = 0.0
    if mcap_now <= 0 and out.get("Aktuell kurs", 0) > 0 and shares > 0:
        mcap_now = float(out["Aktuell kurs"]) * shares

    # SEC kvartalsintäkter (daterade) → TTM-lista
    q_rows = _sec_quarterly_revenues_dated(facts, max_quarters=20)  # många för att kunna bygga 4 TTM-fönster
    if q_rows:
        ttm_list = _ttm_windows(q_rows, need=4)  # [(end_date, ttm_revenue), ...] nyast→
    else:
        ttm_list = []

    # P/S (TTM) nu
    if mcap_now > 0 and ttm_list:
        ltm_now = ttm_list[0][1]  # senaste TTM
        if ltm_now > 0:
            out["P/S"] = mcap_now / ltm_now

    # P/S Q1–Q4 = historisk TTM-P/S vid respektive kvartals slut
    # Hämta historiska priser nära varje TTM-end
    if shares > 0 and ttm_list:
        q_dates = [d for (d, _) in ttm_list]  # upp till 4
        px_map = _yahoo_prices_for_dates(ticker, q_dates)
        for idx, (d_end, ttm_rev) in enumerate(ttm_list[:4], start=1):
            if ttm_rev and ttm_rev > 0:
                px = px_map.get(d_end, None)
                if px and px > 0:
                    mcap_hist = shares * float(px)  # approx: antar samma shares (närmast kända)
                    ps_hist = mcap_hist / ttm_rev
                    out[f"P/S Q{idx}"] = float(ps_hist)

    return out

# --- Finnhub (valfritt) för revenue estimates (annual) -----------------------
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", "")

def hamta_finnhub_revenue_estimates(ticker: str) -> dict:
    """
    Kräver FINNHUB_API_KEY i secrets.
    Hämtar annual revenue estimates: current FY + next FY (om finns).
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

# app.py — Del 4/7
# --- FX via FMP + fallback (ECB / exchangerate.host) ---
def _parse_fx_mid(obj: dict) -> float:
    if not isinstance(obj, dict):
        return 0.0
    b = obj.get("bid"); a = obj.get("ask"); p = obj.get("price")
    try:
        if b is not None and a is not None:
            return (float(b) + float(a)) / 2.0
        if p is not None:
            return float(p)
    except Exception:
        pass
    return 0.0

def hamta_valutakurser_via_fmp():
    result = {"SEK": 1.0}
    pairs = {"USD": "USDSEK", "NOK": "NOKSEK", "CAD": "CADSEK", "EUR": "EURSEK"}
    miss = []
    for cc, pair in pairs.items():
        js, sc = _fmp_get(f"api/v3/forex/{pair}", stable=False)
        rate = 0.0
        if isinstance(js, dict) and js:
            rate = _parse_fx_mid(js)
        elif isinstance(js, list) and js:
            rate = _parse_fx_mid(js[0])
        if rate > 0:
            result[cc] = rate
        else:
            miss.append(f"{pair} (HTTP {sc})")
    return result, miss

def _fx_frankfurter(base: str, quote: str) -> float:
    try:
        r = requests.get("https://api.frankfurter.app/latest",
                         params={"from": base, "to": quote}, timeout=15)
        if r.status_code == 200:
            j = r.json()
            return float(j["rates"][quote])
    except Exception:
        pass
    return 0.0

def _fx_exchangerate_host(base: str, quote: str) -> float:
    try:
        r = requests.get("https://api.exchangerate.host/latest",
                         params={"base": base, "symbols": quote}, timeout=15)
        if r.status_code == 200:
            j = r.json()
            return float(j["rates"][quote])
    except Exception:
        pass
    return 0.0

def hamta_valutakurser_auto():
    rates, miss = hamta_valutakurser_via_fmp()
    provider = "FMP"
    got = [k for k in ("USD","NOK","CAD","EUR") if rates.get(k)]
    if len(got) == 0:
        provider = "Frankfurter (ECB)"
        rates_out = {"SEK": 1.0}; miss2 = []
        for cc in ("USD","NOK","CAD","EUR"):
            rate = _fx_frankfurter(cc, "SEK")
            if rate > 0: rates_out[cc] = rate
            else: miss2.append(f"{cc}SEK (Frankfurter)")
        rates, miss = rates_out, miss2
        if not any(rates.get(k) for k in ("USD","NOK","CAD","EUR")):
            provider = "exchangerate.host"
            rates_out = {"SEK": 1.0}; miss3 = []
            for cc in ("USD","NOK","CAD","EUR"):
                rate = _fx_exchangerate_host(cc, "SEK")
                if rate > 0: rates_out[cc] = rate
                else: miss3.append(f"{cc}SEK (exchangerate.host)")
            rates, miss = rates_out, miss3
    return rates, miss, provider

# --- Drive-kopior & städning ---
DRIVE_BACKUP_FOLDER_ID = st.secrets.get("DRIVE_BACKUP_FOLDER_ID")  # valfritt

def _build_drive_service():
    creds = Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"],
        scopes=[
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def _current_backup_prefix() -> str:
    try:
        ss = get_spreadsheet()
        return f"{ss.title} – BACKUP "
    except Exception:
        return "BACKUP "

def list_drive_backups(limit: int = 500) -> list:
    service = _build_drive_service()
    prefix = _current_backup_prefix()
    safe_prefix = prefix.replace("'", "\\'")
    base_q = "mimeType='application/vnd.google-apps.spreadsheet' and 'me' in owners"
    q = f"{base_q} and name contains '{safe_prefix}'"
    if DRIVE_BACKUP_FOLDER_ID:
        q += f" and '{DRIVE_BACKUP_FOLDER_ID}' in parents"

    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=q, spaces="drive",
            fields="nextPageToken, files(id,name,createdTime,size)",
            orderBy="createdTime desc",
            pageToken=page_token, pageSize=min(1000, limit),
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token or len(items) >= limit:
            break
    return items

def cleanup_drive_backups(keep_last: int = 5, older_than_days: int = None, dry_run: bool = True) -> dict:
    from datetime import timezone
    service = _build_drive_service()
    files = list_drive_backups()
    kept, candidates = [], []

    kept_ids = set([f["id"] for f in files[:max(0, keep_last)]])
    for idx, f in enumerate(files):
        f["_index"] = idx + 1
        is_old = False
        if older_than_days is not None:
            try:
                ct = datetime.fromisoformat(f["createdTime"].replace("Z", "+00:00")).astimezone(timezone.utc)
                is_old = (datetime.now(timezone.utc) - ct) > timedelta(days=older_than_days)
            except Exception:
                pass
        if f["id"] in kept_ids and not is_old:
            kept.append(f)
        else:
            candidates.append(f)

    deleted = []
    if not dry_run:
        for f in candidates:
            try:
                service.files().delete(fileId=f["id"]).execute()
                deleted.append(f)
            except Exception as e:
                f["_delete_error"] = str(e)

    return {"deleted": deleted, "kept": kept, "candidates": candidates}

def _friendly_drive_error(e: Exception) -> str:
    s = str(e)
    if "The user's Drive storage quota has been exceeded" in s:
        return "Drive-kvoten är full på servicekontot. Städa gamla kopior eller byt konto/mapp."
    if "[403]" in s and "drive.googleapis.com" in s:
        return "Drive API är inte aktiverat för projektet som credentials tillhör. Aktivera och försök igen."
    return s

def backup_copy_spreadsheet():
    ss = get_spreadsheet()
    new_title = f"{ss.title} – BACKUP {now_stamp()} {_ts_datetime().strftime('%H%M%S')}"
    try:
        copied = client.copy(
            ss.id, title=new_title, copy_permissions=False,
            folder_id=DRIVE_BACKUP_FOLDER_ID,
        )
        try:
            return copied.id, copied.title
        except Exception:
            try:
                return copied.get("id"), copied.get("name", new_title)
            except Exception:
                return None, new_title
    except Exception as e:
        raise RuntimeError(_friendly_drive_error(e))

# --- Spara data (snapshot före skrivning) ---
def spara_data(df: pd.DataFrame, do_snapshot: bool = True):
    if do_snapshot:
        try:
            snap_name = backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)
            st.sidebar.info(f"Backup-snapshot skapad: {snap_name}")
        except Exception as e:
            st.sidebar.warning(f"Backup-snapshot misslyckades: {e}")
    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    _with_backoff(sheet.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# app.py — Del 5/7
# --- Helper för att flattena kolumner från yfinance ---
def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" ".join([str(x) for x in tup if str(x) != ""]).strip().lower() for tup in out.columns.values]
    else:
        out.columns = [str(c).strip().lower() for c in out.columns]
    return out

# --- yfinance-hjälpare (fast_info + info) ---
def _yfi_get(t: yf.Ticker, *keys):
    """
    Försöker först i fast_info (snabbt, mer robust), sedan i info.
    Returnerar första icke-None av givna keys.
    """
    try:
        fi = dict(getattr(t, "fast_info", {}) or {})
    except Exception:
        fi = {}
    for k in keys:
        v = fi.get(k, None)
        if v is not None:
            return v

    inf = {}
    try:
        inf = t.info or {}
    except Exception:
        inf = {}
    for k in keys:
        v = inf.get(k, None)
        if v is not None:
            return v
    return None

def _extract_quarterly_revenues(t: yf.Ticker):
    """
    Hämtar kvartalsvisa intäkter. Försöker med .quarterly_income_stmt (nyare)
    och faller tillbaka till .quarterly_financials (äldre).
    Returnerar lista med de 4 senaste värdena (float), samt summerad LTM.
    """
    q_revs = []

    def _pick_rev_row(df: pd.DataFrame) -> pd.Series:
        keys = {str(ix).strip().lower().replace(" ", ""): ix for ix in df.index}
        for cand in ("totalrevenue", "total revenue"):
            if cand in keys:
                return df.loc[keys[cand]]
        for cand in ("revenue", "sales"):
            if cand in keys:
                return df.loc[keys[cand]]
        return pd.Series(dtype=float)

    try:
        qis = getattr(t, "quarterly_income_stmt", None)
        if isinstance(qis, pd.DataFrame) and not qis.empty:
            row = _pick_rev_row(qis)
            if not row.empty:
                for c in list(qis.columns)[:4]:
                    v = row.get(c, None)
                    if pd.notna(v):
                        try: q_revs.append(float(v))
                        except: pass
    except Exception:
        pass

    if not q_revs:
        try:
            qfin = getattr(t, "quarterly_financials", None)
            if isinstance(qfin, pd.DataFrame) and not qfin.empty:
                if "Total Revenue" in qfin.index:
                    row = qfin.loc["Total Revenue"]
                    for c in list(qfin.columns)[:4]:
                        v = row.get(c, None)
                        if pd.notna(v):
                            try: q_revs.append(float(v))
                            except: pass
        except Exception:
            pass

    rev_ltm = sum(q_revs) if q_revs else 0.0
    return q_revs, rev_ltm

# --- CAGR via yfinance ---
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
        if series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start, end = float(series.iloc[0]), float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

# --- Yahoo: namn/kurs/valuta/utdelning/CAGR ---
def hamta_yahoo_fält(ticker: str) -> dict:
    out = {"Bolagsnamn": "", "Aktuell kurs": 0.0, "Valuta": "USD", "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0}
    try:
        t = yf.Ticker(ticker)

        pris = _yfi_get(t, "last_price", "regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = _yfi_get(t, "currency", "financialCurrency")
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = None
        try:
            inf = t.info or {}
            namn = inf.get("shortName") or inf.get("longName") or inf.get("displayName")
        except Exception:
            namn = None
        if namn:
            out["Bolagsnamn"] = str(namn)

        try:
            inf = t.info or {}
            div_rate = inf.get("dividendRate", None)
            if div_rate is None and inf.get("dividendYield") and out["Aktuell kurs"] > 0:
                div_rate = float(inf["dividendYield"]) * out["Aktuell kurs"]
            if div_rate is not None:
                out["Årlig utdelning"] = float(div_rate)
        except Exception:
            pass

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# --- Yahoo P/S-fallback (stark) ---
def yahoo_ps_fallback(ticker: str) -> dict:
    """
    Beräknar:
      - Utestående aktier (M) från fast_info/info, eller market_cap / price
      - P/S (TTM) = market_cap / LTM-omsättning
      - P/S Q1–Q4 ≈ market_cap / respektive kvartalsintäkt (om finns)
    """
    out = {}
    try:
        t = yf.Ticker(ticker)

        mcap = _yfi_get(t, "market_cap", "marketCap")
        pris = _yfi_get(t, "last_price", "regularMarketPrice")
        shares = _yfi_get(t, "shares_outstanding", "shares", "sharesOutstanding")

        try: mcap = float(mcap or 0.0)
        except: mcap = 0.0
        try: pris = float(pris or 0.0)
        except: pris = 0.0
        try: shares = float(shares or 0.0)
        except: shares = 0.0

        if shares <= 0 and mcap > 0 and pris > 0:
            shares = mcap / pris

        if shares > 0:
            out["Utestående aktier"] = shares / 1e6

        q_revs, rev_ltm = _extract_quarterly_revenues(t)

        if rev_ltm <= 0:
            try:
                ain = getattr(t, "income_stmt", None)
                if isinstance(ain, pd.DataFrame) and not ain.empty:
                    idxkeys = {str(ix).strip().lower().replace(" ", ""): ix for ix in ain.index}
                    ix = None
                    for cand in ("totalrevenue", "total revenue", "revenue", "sales"):
                        if cand in idxkeys:
                            ix = idxkeys[cand]; break
                    if ix is not None:
                        ser = ain.loc[ix].dropna()
                        if not ser.empty:
                            rev_ltm = float(ser.iloc[0])
            except Exception:
                pass

        if mcap > 0 and rev_ltm and rev_ltm > 0:
            out["P/S"] = float(mcap / rev_ltm)

        if mcap > 0 and q_revs:
            for i, rv in enumerate(q_revs[:4], start=1):
                if rv and rv > 0:
                    out[f"P/S Q{i}"] = float(mcap / rv)
    except Exception:
        pass
    return out

# --- Yahoo omsättningsestimat (robust) ---
def hamta_yahoo_omsattningsestimat(ticker: str) -> dict:
    out = {}
    try:
        t = yf.Ticker(ticker)
        et = getattr(t, "earnings_trend", None)
        if (et is None or (isinstance(et, pd.DataFrame) and et.empty)) and hasattr(t, "get_earnings_trend"):
            try:
                et = t.get_earnings_trend()
            except Exception:
                pass

        if isinstance(et, pd.DataFrame) and not et.empty:
            df = _flatten_cols(et)
            if "period" not in df.columns:
                df["period"] = et.get("period", None)
            df["period"] = df["period"].astype(str).str.lower()

            cand_keys = [
                "revenueestimate avg",
                "revenue_estimate avg",
                "revenueestimate",
                "revenue_estimate",
            ]

            def _extract_avg(row):
                for k in cand_keys[:2]:
                    if k in row.index:
                        try:
                            v = float(row[k])
                            if v > 0:
                                return v
                        except Exception:
                            pass
                for k in cand_keys[2:]:
                    if k in row.index:
                        cell = row[k]
                        if isinstance(cell, dict):
                            v = cell.get("avg", None)
                            try:
                                v = float(v)
                                if v > 0:
                                    return v
                            except Exception:
                                pass
                return 0.0

            def _pick_period(names):
                m = df[df["period"].isin([p.lower() for p in names])]
                if not m.empty:
                    return _extract_avg(m.iloc[0])
                m2 = df[df["period"].apply(lambda x: any(p.lower() in x for p in names))]
                if not m2.empty:
                    return _extract_avg(m2.iloc[0])
                return 0.0

            cur = _pick_period(["currentyear", "current year", "0y", "thisyear"])
            nxt = _pick_period(["nextyear", "next year", "+1y", "nextfiscalyear"])
            if cur > 0: out["Omsättning idag"] = cur / 1e6
            if nxt > 0: out["Omsättning nästa år"] = nxt / 1e6
    except Exception:
        pass
    return out

# --- Beräkningar ---
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    for i, rad in df.iterrows():
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# app.py — Del 6/7
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict, source: str = "Yahoo") -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från " + source, key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []
        estimat_miss = []
        change_summaries = []

        ps_count = 0
        psq_count = 0
        est_count = 0

        updated_tickers = []
        unchanged_tickers = []
        changes_map = {}

        total = len(df)
        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip().upper()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")

            before = {f: row.get(f, 0.0) for f in UPPDATERBARA_FALT}

            # 1) Hämta basdata enligt källa
            if source == "FMP":
                data = hamta_fmp_falt_light(tkr) if st.session_state.get("fmp_light_mode", True) else hamta_fmp_falt(tkr)
            elif source == "SEC":
                data = hamta_sec_yahoo_combo(tkr)
            else:  # Yahoo
                data = hamta_yahoo_fält(tkr)

            failed_fields = []

            # 2) Sätt namn/valuta/kurs om finns
            if data.get("Bolagsnamn"):
                mark_field_if_changed(df, i, "Bolagsnamn", data["Bolagsnamn"])
            else: failed_fields.append("Bolagsnamn")

            if data.get("Aktuell kurs", 0) > 0:
                mark_field_if_changed(df, i, "Aktuell kurs", float(data["Aktuell kurs"]))
            else: failed_fields.append("Aktuell kurs")

            if data.get("Valuta"):
                mark_field_if_changed(df, i, "Valuta", data["Valuta"])
            else: failed_fields.append("Valuta")

            # 3) Fältspecifikt per källa
            if source in ("FMP", "SEC"):
                # Shares + P/S + kvartal
                if data.get("Utestående aktier", 0) > 0:
                    mark_field_if_changed(df, i, "Utestående aktier", float(data["Utestående aktier"]))
                if data.get("P/S", 0) > 0:
                    if mark_field_if_changed(df, i, "P/S", float(data["P/S"])):
                        ps_count += 1
                psq_touched = False
                for q in (1, 2, 3, 4):
                    key = f"P/S Q{q}"
                    if data.get(key, 0) > 0:
                        if mark_field_if_changed(df, i, key, float(data[key])):
                            psq_touched = True
                if psq_touched:
                    psq_count += 1

                # Estimat: prova FMP (om inte light), annars Yahoo, sedan Finnhub
                est_touched = False
                if source == "FMP" and not st.session_state.get("fmp_light_mode", True):
                    cur_est = data.get("Omsättning idag", 0.0)
                    nxt_est = data.get("Omsättning nästa år", 0.0)
                    if cur_est and cur_est > 0:
                        if mark_field_if_changed(df, i, "Omsättning idag", float(cur_est)):
                            est_touched = True
                    if nxt_est and nxt_est > 0:
                        if mark_field_if_changed(df, i, "Omsättning nästa år", float(nxt_est)):
                            est_touched = True

                if not est_touched:
                    y_est = hamta_yahoo_omsattningsestimat(tkr)
                    y_ok = False
                    v = float(y_est.get("Omsättning idag", 0.0))
                    if v > 0:
                        if mark_field_if_changed(df, i, "Omsättning idag", v):
                            y_ok = True; est_touched = True
                    v = float(y_est.get("Omsättning nästa år", 0.0))
                    if v > 0:
                        if mark_field_if_changed(df, i, "Omsättning nästa år", v):
                            y_ok = True; est_touched = True

                    # Finnhub sista hand om båda saknas
                    if not y_ok and float(df.at[i, "Omsättning idag"]) == 0.0 and float(df.at[i, "Omsättning nästa år"]) == 0.0:
                        fh = hamta_finnhub_revenue_estimates(tkr)
                        v = float(fh.get("Omsättning idag", 0.0))
                        if v > 0:
                            if mark_field_if_changed(df, i, "Omsättning idag", v):
                                est_touched = True
                        v = float(fh.get("Omsättning nästa år", 0.0))
                        if v > 0:
                            if mark_field_if_changed(df, i, "Omsättning nästa år", v):
                                est_touched = True

                if est_touched:
                    est_count += 1
                else:
                    estimat_miss.append(f"{tkr} (estimat saknas eller ej tillgängligt)")

                # Fallbacks om P/S eller shares saknas → SEC+Yahoo och/eller Yahoo PS fallback
                got_ps = float(df.at[i, "P/S"]) if "P/S" in df.columns else 0.0
                got_sh = float(df.at[i, "Utestående aktier"]) if "Utestående aktier" in df.columns else 0.0
                if got_ps == 0.0 or got_sh == 0.0:
                    secd = hamta_sec_yahoo_combo(tkr)
                    touched = False
                    if secd.get("Utestående aktier", 0) > 0 and got_sh == 0.0:
                        if mark_field_if_changed(df, i, "Utestående aktier", float(secd["Utestående aktier"])):
                            touched = True
                    if secd.get("P/S", 0) > 0 and got_ps == 0.0:
                        if mark_field_if_changed(df, i, "P/S", float(secd["P/S"])):
                            ps_count += 1; touched = True
                    for q in (1, 2, 3, 4):
                        key = f"P/S Q{q}"
                        if secd.get(key, 0) > 0 and float(df.at[i, key]) == 0.0:
                            if mark_field_if_changed(df, i, key, float(secd[key])):
                                touched = True
                    if not touched:
                        yb = yahoo_ps_fallback(tkr)
                        if yb.get("Utestående aktier", 0) > 0 and got_sh == 0.0:
                            mark_field_if_changed(df, i, "Utestående aktier", float(yb["Utestående aktier"]))
                        if yb.get("P/S", 0) > 0 and got_ps == 0.0:
                            if mark_field_if_changed(df, i, "P/S", float(yb["P/S"])):
                                ps_count += 1
                        for q in (1, 2, 3, 4):
                            key = f"P/S Q{q}"
                            if yb.get(key, 0) > 0 and float(df.at[i, key]) == 0.0:
                                mark_field_if_changed(df, i, key, float(yb[key]))

            else:
                # Källa: Yahoo
                # Försök estimat via Yahoo, annars Finnhub
                y_est = hamta_yahoo_omsattningsestimat(tkr)
                est_touched = False
                v = float(y_est.get("Omsättning idag", 0.0))
                if v > 0:
                    if mark_field_if_changed(df, i, "Omsättning idag", v):
                        est_touched = True
                v = float(y_est.get("Omsättning nästa år", 0.0))
                if v > 0:
                    if mark_field_if_changed(df, i, "Omsättning nästa år", v):
                        est_touched = True

                if not est_touched and float(df.at[i, "Omsättning idag"]) == 0.0 and float(df.at[i, "Omsättning nästa år"]) == 0.0:
                    fh = hamta_finnhub_revenue_estimates(tkr)
                    v = float(fh.get("Omsättning idag", 0.0))
                    if v > 0:
                        if mark_field_if_changed(df, i, "Omsättning idag", v):
                            est_touched = True
                    v = float(fh.get("Omsättning nästa år", 0.0))
                    if v > 0:
                        if mark_field_if_changed(df, i, "Omsättning nästa år", v):
                            est_touched = True

                if not est_touched:
                    estimat_miss.append(f"{tkr} (Yahoo/Finnhub saknar estimat)")
                else:
                    est_count += 1

                # Shares/P-S via SEC eller Yahoo-fallback om saknas
                got_ps = float(df.at[i, "P/S"]) if "P/S" in df.columns else 0.0
                got_sh = float(df.at[i, "Utestående aktier"]) if "Utestående aktier" in df.columns else 0.0
                if got_ps == 0.0 or got_sh == 0.0:
                    secd = hamta_sec_yahoo_combo(tkr)
                    touched = False
                    if secd.get("Utestående aktier", 0) > 0 and got_sh == 0.0:
                        if mark_field_if_changed(df, i, "Utestående aktier", float(secd["Utestående aktier"])):
                            touched = True
                    if secd.get("P/S", 0) > 0 and got_ps == 0.0:
                        if mark_field_if_changed(df, i, "P/S", float(secd["P/S"])):
                            ps_count += 1; touched = True
                    for q in (1, 2, 3, 4):
                        key = f"P/S Q{q}"
                        if secd.get(key, 0) > 0 and float(df.at[i, key]) == 0.0:
                            if mark_field_if_changed(df, i, key, float(secd[key])):
                                touched = True
                    if not touched:
                        yb = yahoo_ps_fallback(tkr)
                        if yb.get("Utestående aktier", 0) > 0 and got_sh == 0.0:
                            mark_field_if_changed(df, i, "Utestående aktier", float(yb["Utestående aktier"]))
                        if yb.get("P/S", 0) > 0 and got_ps == 0.0:
                            if mark_field_if_changed(df, i, "P/S", float(yb["P/S"])):
                                ps_count += 1
                        for q in (1, 2, 3, 4):
                            key = f"P/S Q{q}"
                            if yb.get(key, 0) > 0 and float(df.at[i, key]) == 0.0:
                                mark_field_if_changed(df, i, key, float(yb[key]))

            # 4) Diffs och metadata
            after = {f: df.at[i, f] if f in df.columns else 0.0 for f in UPPDATERBARA_FALT}
            changed_fields = []
            for f in UPPDATERBARA_FALT:
                a, b = before.get(f, None), after.get(f, None)
                try:
                    same = _is_close(a, b)
                except Exception:
                    same = str(a) == str(b)
                if not same:
                    changed_fields.append(f)

            if changed_fields:
                df.at[i, "Senast auto-uppdaterad"] = _ts_human()
                if not df.at[i, "Senast uppdaterad källa"]:
                    df.at[i, "Senast uppdaterad källa"] = source
                change_summaries.append(f"{tkr}: " + ", ".join(changed_fields))
                updated_tickers.append(tkr)
                changes_map[tkr] = changed_fields
            else:
                unchanged_tickers.append(tkr)

            if failed_fields:
                misslyckade.append(f"{tkr}: {', '.join(failed_fields)}")

            time.sleep(1.0)
            bar.progress((i+1)/total)

        # 5) Spara endast om något faktiskt ändrats
        if change_summaries:
            df = uppdatera_berakningar(df, user_rates)
            spara_data(df, do_snapshot=True)
            st.sidebar.success(f"Klart! {len(change_summaries)} av {total} bolag fick ändringar och sparades.")
        else:
            st.sidebar.info("Ingen faktisk ändring upptäcktes – ingen skrivning/snapshot gjordes.")

        st.sidebar.info(f"P/S uppdaterades på {ps_count} bolag; P/S kvartal på {psq_count}; Estimat på {est_count}.")

        # 6) Körningslogg
        log_parts = []
        if change_summaries:
            log_parts.append("[Ändringar]")
            log_parts.extend(change_summaries); log_parts.append("")
        if misslyckade:
            log_parts.append("[Fält som saknades]")
            log_parts.extend(misslyckade); log_parts.append("")
        if estimat_miss:
            log_parts.append("[Analytikerestimat saknades – fält lämnades orörda]")
            log_parts.extend(estimat_miss)
        if log_parts:
            payload = "\n".join(log_parts)
            st.sidebar.text_area("Ändringslogg", payload, height=240, key=f"{key_prefix}_chglog")
            st.sidebar.download_button(
                label="⬇️ Ladda ner ändringslogg",
                data=payload.encode("utf-8"),
                file_name=f"update_log_{_ts_str()}.txt",
                mime="text/plain",
                key=f"{key_prefix}_chglog_dl"
            )

        st.session_state["run_report"] = {
            "ts": _ts_human(),
            "source": source,
            "updated": updated_tickers,
            "unchanged": unchanged_tickers,
            "errors": misslyckade,
            "estimat_miss": estimat_miss,
            "changes_map": changes_map,
            "total": total,
            "updated_count": len(updated_tickers),
            "unchanged_count": len(unchanged_tickers),
        }

    return df

MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict, datakalla_default: str = "Yahoo") -> pd.DataFrame:
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
        row_index = df.index[df["Ticker"]==namn_map[valt_label]][0]
    else:
        bef = pd.Series({}, dtype=object)
        row_index = None

    st.session_state.setdefault("datakalla_form", datakalla_default)
    st.session_state["datakalla_form"] = st.radio("Hämtningskälla", ["Yahoo","FMP","SEC"],
                                                  index=(0 if datakalla_default=="Yahoo" else (1 if datakalla_default=="FMP" else 2)),
                                                  horizontal=True, key="form_src")

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()

            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS Utestående aktier','–')}")

            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS P/S','–')}")

            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS P/S Q1','–')}")

            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS P/S Q2','–')}")

            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS P/S Q3','–')}")

            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS P/S Q4','–')}")
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS Omsättning idag','–')}")

            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            st.caption(f"Senast ändrad: {bef.get('TS Omsättning nästa år','–')}")

            st.markdown("**Uppdateras automatiskt vid spara:**")
            st.write("- Yahoo: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) + Omsättningsestimat (om finns)")
            st.write("- FMP: Utestående aktier, P/S (TTM), P/S Q1–Q4, ev. omsättningsestimat (om > 0)")
            st.write("- SEC: Utestående aktier (US), P/S (TTM), P/S Q1–Q4 beräknade från kvartalsintäkter + pris")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas om")

        spar = st.form_submit_button("💾 Spara & hämta från vald källa")

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        changed_manual = False
        if row_index is not None:
            for k, v in ny.items():
                if k in TS_FIELDS:
                    if mark_field_if_changed(df, row_index, k, v):
                        changed_manual = True
                else:
                    df.at[row_index, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa", *TS_FIELDS.values()] else "") for c in FINAL_COLS}
            tom.update(ny)
            now_h = _ts_human()
            for f, ts_col in TS_FIELDS.items():
                try:
                    if float(tom.get(f, 0.0)) != 0.0:
                        tom[ts_col] = now_h
                except Exception:
                    if str(tom.get(f,"")) != "":
                        tom[ts_col] = now_h
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            changed_manual = any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM)
            row_index = df.index[-1]

        if changed_manual:
            df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta från vald källa + fallbacks
        src = st.session_state["datakalla_form"]
        if src == "FMP":
            data = hamta_fmp_falt_light(ticker) if st.session_state.get("fmp_light_mode", True) else hamta_fmp_falt(ticker)
        elif src == "SEC":
            data = hamta_sec_yahoo_combo(ticker)
        else:
            data = hamta_yahoo_fält(ticker)

        # basics
        if data.get("Bolagsnamn"): df.loc[df["Ticker"]==ticker, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[df["Ticker"]==ticker, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[df["Ticker"]==ticker, "Aktuell kurs"] = float(data["Aktuell kurs"])
        if src == "Yahoo":
            if "Årlig utdelning" in data: df.loc[df["Ticker"]==ticker, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            if "CAGR 5 år (%)" in data:   df.loc[df["Ticker"]==ticker, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

        # shares/P-S/kvartal
        touched_ps = False
        for key in ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
            if data.get(key, 0) and row_index is not None:
                if mark_field_if_changed(df, row_index, key, float(data[key])):
                    touched_ps = True

        # Estimat: Yahoo → Finnhub → (FMP om ej light och vi valde FMP)
        est_touched = False
        y_est = hamta_yahoo_omsattningsestimat(ticker)
        v = float(y_est.get("Omsättning idag", 0.0))
        if v > 0 and row_index is not None:
            if mark_field_if_changed(df, row_index, "Omsättning idag", v):
                est_touched = True
        v = float(y_est.get("Omsättning nästa år", 0.0))
        if v > 0 and row_index is not None:
            if mark_field_if_changed(df, row_index, "Omsättning nästa år", v):
                est_touched = True

        if not est_touched:
            fh = hamta_finnhub_revenue_estimates(ticker)
            v = float(fh.get("Omsättning idag", 0.0))
            if v > 0 and row_index is not None:
                if mark_field_if_changed(df, row_index, "Omsättning idag", v):
                    est_touched = True
            v = float(fh.get("Omsättning nästa år", 0.0))
            if v > 0 and row_index is not None:
                if mark_field_if_changed(df, row_index, "Omsättning nästa år", v):
                    est_touched = True

        if src == "FMP" and not est_touched and not st.session_state.get("fmp_light_mode", True):
            v = float(data.get("Omsättning idag", 0.0))
            if v > 0 and row_index is not None:
                if mark_field_if_changed(df, row_index, "Omsättning idag", v):
                    est_touched = True
            v = float(data.get("Omsättning nästa år", 0.0))
            if v > 0 and row_index is not None:
                if mark_field_if_changed(df, row_index, "Omsättning nästa år", v):
                    est_touched = True

        # Shares/P-S fortfarande tomt? Testa SEC → Yahoo-fallback
        got_ps = float(df.at[row_index, "P/S"]) if "P/S" in df.columns else 0.0
        got_sh = float(df.at[row_index, "Utestående aktier"]) if "Utestående aktier" in df.columns else 0.0
        if got_ps == 0.0 or got_sh == 0.0:
            secd = hamta_sec_yahoo_combo(ticker)
            if secd.get("Utestående aktier", 0) > 0 and got_sh == 0.0:
                mark_field_if_changed(df, row_index, "Utestående aktier", float(secd["Utestående aktier"]))
            if secd.get("P/S", 0) > 0 and got_ps == 0.0:
                mark_field_if_changed(df, row_index, "P/S", float(secd["P/S"]))
            for q in (1,2,3,4):
                key = f"P/S Q{q}"
                if secd.get(key, 0) > 0 and float(df.at[row_index, key]) == 0.0:
                    mark_field_if_changed(df, row_index, key, float(secd[key]))
            if float(df.at[row_index, "P/S"]) == 0.0 or float(df.at[row_index, "Utestående aktier"]) == 0.0:
                yb = yahoo_ps_fallback(ticker)
                if yb.get("Utestående aktier", 0) > 0 and float(df.at[row_index, "Utestående aktier"]) == 0.0:
                    mark_field_if_changed(df, row_index, "Utestående aktier", float(yb["Utestående aktier"]))
                if yb.get("P/S", 0) > 0 and float(df.at[row_index, "P/S"]) == 0.0:
                    mark_field_if_changed(df, row_index, "P/S", float(yb["P/S"]))
                for q in (1,2,3,4):
                    key = f"P/S Q{q}"
                    if yb.get(key, 0) > 0 and float(df.at[row_index, key]) == 0.0:
                        mark_field_if_changed(df, row_index, key, float(yb[key]))

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från vald källa.")

    # Listor för kontroll
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[[
            "Ticker","Bolagsnamn","Senast manuellt uppdaterad",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Omsättning idag","Omsättning nästa år"
        ]],
        use_container_width=True
    )

    st.markdown("### 🤖 Äldst auto-uppdaterade (topp 10)")
    auto_df = df.copy()
    auto_df["_auto_dt"] = pd.to_datetime(auto_df["Senast auto-uppdaterad"], errors="coerce")\
                        .fillna(pd.Timestamp("1900-01-01"))
    auto_oldest = auto_df.sort_values(by=["_auto_dt","Bolagsnamn"], ascending=[True, True]).head(10)
    st.dataframe(
        auto_oldest[[
            "Ticker","Bolagsnamn","Senast auto-uppdaterad","Senast uppdaterad källa",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Omsättning idag","Omsättning nästa år"
        ]],
        use_container_width=True
    )

    st.markdown("### ⚠️ Bolag utan auto-uppdatering sedan valt datum")
    default_cutoff = (_ts_datetime() - timedelta(days=90)).date()
    cutoff_date = st.date_input("Visa bolag med auto-uppdatering äldre än:", value=default_cutoff)

    stale = df.copy()
    stale["_auto_dt"] = pd.to_datetime(stale["Senast auto-uppdaterad"], errors="coerce")
    mask_stale = (stale["_auto_dt"].isna()) | (stale["_auto_dt"].dt.date < cutoff_date)
    stale = stale[mask_stale].sort_values(by=["_auto_dt","Bolagsnamn"], ascending=[True, True])

    st.caption(f"Hittade {len(stale)} bolag äldre än {cutoff_date}.")
    st.dataframe(
        stale[[
            "Ticker","Bolagsnamn",
            "Senast auto-uppdaterad","Senast uppdaterad källa",
            "Senast manuellt uppdaterad"
        ]],
        use_container_width=True
    )

    return df

# app.py — Del 7/7
# --- Analysvy ---
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")

    show_ts = st.checkbox("Visa tidsstämplar för spårade fält", value=False)

    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
                "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"]
        if show_ts:
            cols += list(TS_FIELDS.values())
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

# --- Portfölj ---
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurser(v, user_rates))
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

# --- Investeringsförslag ---
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
                                index=1)

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    if subset == "Endast portfölj":
        base = df[df["Antal aktier"] > 0].copy()
    else:
        base = df.copy()

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

    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurser(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = hamta_valutakurser(rad["Valuta"], user_rates)
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
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# --- Kontroll & reconciliation ---
def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧪 Kontroll & Reconciliation")

    rr = st.session_state.get("run_report")
    st.subheader("Senaste körningsrapport")
    if not rr:
        st.info("Ingen körning i denna session ännu. Kör en massuppdatering (Yahoo, FMP eller SEC) så fylls rapporten.")
    else:
        col_top1, col_top2, col_top3 = st.columns(3)
        col_top1.metric("Källa", rr["source"])
        col_top2.metric("Tid", rr["ts"])
        col_top3.metric("Totalt bolag", rr["total"])

        st.markdown("#### ✅ Uppdaterade (värden ändrades)")
        upd = rr.get("updated", [])
        if upd:
            df_upd = df[df["Ticker"].isin(upd)].copy().sort_values("Bolagsnamn")
            ch_map = rr.get("changes_map", {})
            df_upd["Ändrade fält (senaste körningen)"] = df_upd["Ticker"].map(lambda x: ", ".join(ch_map.get(x, [])))
            st.dataframe(df_upd[[
                "Ticker","Bolagsnamn","Senast auto-uppdaterad","Senast uppdaterad källa",
                "Ändrade fält (senaste körningen)"
            ]], use_container_width=True)
            st.download_button(
                "⬇️ Exportera uppdaterade (CSV)",
                data=df_upd.to_csv(index=False).encode("utf-8"),
                file_name=f"uppdaterade_{_ts_str()}.csv",
                mime="text/csv"
            )
        else:
            st.caption("Inga bolag ändrades i senaste körningen.")

        st.markdown("#### 💤 Inte uppdaterade (ingen värdeförändring)")
        unch = rr.get("unchanged", [])
        if unch:
            df_unch = df[df["Ticker"].isin(unch)].copy().sort_values("Bolagsnamn")
            st.dataframe(df_unch[[
                "Ticker","Bolagsnamn","Senast auto-uppdaterad","Senast uppdaterad källa",
                "Senast manuellt uppdaterad"
            ]], use_container_width=True)
            st.download_button(
                "⬇️ Exportera ej uppdaterade (CSV)",
                data=df_unch.to_csv(index=False).encode("utf-8"),
                file_name=f"ej_uppdaterade_{_ts_str()}.csv",
                mime="text/csv"
            )
        else:
            st.caption("Alla bolag ändrades i senaste körningen.")

        if rr.get("errors") or rr.get("estimat_miss"):
            st.markdown("#### ⚠️ Varningar / Estimat saknades")
            if rr.get("errors"):
                st.write("- Fält som saknades:"); st.code("\n".join(rr["errors"]), language="text")
            if rr.get("estimat_miss"):
                st.write("- Estimat saknades:"); st.code("\n".join(rr["estimat_miss"]), language="text")

    st.markdown("---")

    st.subheader("🤖 Äldst auto-uppdaterade (topp 10)")
    auto_df = df.copy()
    auto_df["_auto_dt"] = pd.to_datetime(auto_df["Senast auto-uppdaterad"], errors="coerce")\
                            .fillna(pd.Timestamp("1900-01-01"))
    auto_oldest = auto_df.sort_values(by=["_auto_dt","Bolagsnamn"], ascending=[True, True]).head(10)
    st.dataframe(
        auto_oldest[[
            "Ticker","Bolagsnamn","Senast auto-uppdaterad","Senast uppdaterad källa",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Omsättning idag","Omsättning nästa år"
        ]],
        use_container_width=True
    )

    st.subheader("📅 Utan auto-uppdatering sedan valt datum")
    default_cutoff = (_ts_datetime() - timedelta(days=90)).date()
    cutoff_date = st.date_input("Visa bolag med auto-uppdatering äldre än:", value=default_cutoff)

    stale = df.copy()
    stale["_auto_dt"] = pd.to_datetime(stale["Senast auto-uppdaterad"], errors="coerce")
    mask_stale = (stale["_auto_dt"].isna()) | (stale["_auto_dt"].dt.date < cutoff_date)
    stale = stale[mask_stale].sort_values(by=["_auto_dt","Bolagsnamn"], ascending=[True, True])

    st.caption(f"Hittade {len(stale)} bolag äldre än {cutoff_date}.")
    st.dataframe(
        stale[[
            "Ticker","Bolagsnamn",
            "Senast auto-uppdaterad","Senast uppdaterad källa",
            "Senast manuellt uppdaterad"
        ]],
        use_container_width=True
    )
    st.download_button(
        "⬇️ Exportera listan (CSV)",
        data=stale.drop(columns=["_auto_dt"]).to_csv(index=False).encode("utf-8"),
        file_name=f"utan_auto_sedan_{cutoff_date}.csv",
        mime="text/csv"
    )

# --- main ---
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: valutakurser
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, format="%.4f")
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    with st.sidebar.expander("🌐 Hämta valutakurser automatiskt"):
        if st.button("Hämta valutakurser nu"):
            rates, miss, provider = hamta_valutakurser_auto()
            usd = rates.get("USD", usd); nok = rates.get("NOK", nok)
            cad = rates.get("CAD", cad); eur = rates.get("EUR", eur)
            user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            if miss:
                st.warning(f"Vissa par kunde inte hämtas via {provider}:\n- " + "\n- ".join(miss))
            else:
                st.success(f"Valutakurser uppdaterade via {provider}.")
            if provider != "FMP":
                st.info("FMP:s valutadata kan kräva högre plan. Automatisk fallback användes.")

    # ⚙️ FMP-inställningar
    with st.sidebar.expander("⚙️ FMP-inställningar", expanded=False):
        st.session_state["fmp_light_mode"] = st.checkbox("FMP Light-mode (färre API-anrop)", value=True)
        key_present = bool(FMP_KEY)
        key_preview = (FMP_KEY[:4] + "…") if key_present else "(saknas)"
        st.caption(f"FMP_API_KEY: {key_preview} | BASE: {FMP_BASE} | CALL_DELAY: {FMP_CALL_DELAY}s")

        blocked_until = st.session_state.get("fmp_block_until")
        if blocked_until:
            if _ts_datetime() < blocked_until:
                st.warning(f"FMP är pausad till {blocked_until.strftime('%Y-%m-%d %H:%M')}")
                if st.button("Återuppta nu", key="fmp_unblock"):
                    st.session_state.pop("fmp_block_until", None)
                    st.rerun()
            else:
                st.session_state.pop("fmp_block_until", None)

        if not key_present:
            st.warning("Ingen FMP_API_KEY hittades i secrets. Lägg till den för att undvika 429/403.")

    # 🧪 API-hälsokoll
    with st.sidebar.expander("🧪 API-hälsokoll", expanded=False):
        test_tkr = st.text_input("Testa FMP/SEC/Yahoo för ticker (Yahoo-format):", key="fmp_test_ticker")
        if st.button("Kör test", disabled=not bool(st.session_state.get("fmp_test_ticker"))):
            tkr = st.session_state["fmp_test_ticker"].strip().upper()
            d = hamta_fmp_falt_light(tkr) if st.session_state.get("fmp_light_mode", True) else hamta_fmp_falt(tkr)
            sec_sample = hamta_sec_yahoo_combo(tkr)
            st.write({
                "FMP": {
                    "api_key": ("OK" if bool(FMP_KEY) else "saknas"),
                    "symbol": d.get("_symbol"),
                    "Bolagsnamn": d.get("Bolagsnamn"),
                    "Valuta": d.get("Valuta"),
                    "Aktuell kurs": d.get("Aktuell kurs"),
                    "Utestående aktier (M)": d.get("Utestående aktier"),
                    "P/S": d.get("P/S"),
                    "ps_source": d.get("_debug", {}).get("ps_source"),
                    "est_http": d.get("_est_status"),
                    "http_statusar": {
                        k: d.get("_debug", {}).get(k) for k in [
                            "profile_sc","quote_sc","shares_float_sc",
                            "ratios_ttm_sc","key_metrics_ttm_sc","income_ttm_sc","ratios_quarter_sc",
                            "analyst_estimates_sc"
                        ] if d.get("_debug", {}).get(k) is not None
                    }
                },
                "SEC+Yahoo": {
                    "Utestående aktier (M)": sec_sample.get("Utestående aktier"),
                    "P/S": sec_sample.get("P/S"),
                    "P/S Q1": sec_sample.get("P/S Q1"),
                    "P/S Q2": sec_sample.get("P/S Q2"),
                    "P/S Q3": sec_sample.get("P/S Q3"),
                    "P/S Q4": sec_sample.get("P/S Q4"),
                }
            })

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    # Läs data
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df, do_snapshot=False)

    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Global massuppdatering (tre knappar: Yahoo, FMP, SEC)
    df = massuppdatera(df, key_prefix="global_y", user_rates=user_rates, source="Yahoo")
    df = massuppdatera(df, key_prefix="global_f", user_rates=user_rates, source="FMP")
    df = massuppdatera(df, key_prefix="global_s", user_rates=user_rates, source="SEC")

    # Backup & återställning
    with st.sidebar.expander("🛟 Backup & återställning", expanded=False):
        st.caption("Skapa kopior innan större körningar/ändringar.")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("📄 Snapshot (ny flik)"):
                try:
                    name = backup_snapshot_sheet(df, base_sheet_name=SHEET_NAME)
                    st.success(f"Snapshot skapad: {name}")
                except Exception as e:
                    st.error(f"Misslyckades: {e}")
        with col_b2:
            if st.button("📁 Fullständig filkopia"):
                try:
                    new_id, new_title = backup_copy_spreadsheet()
                    if new_id:
                        st.success(f"Kopia skapad: {new_title}")
                        st.write("Öppna i Drive och spara URL om du vill kunna byta tillbaka via `SHEET_URL`.")
                    else:
                        st.warning(f"Kopia skapad: {new_title} (okänt id)")
                except Exception as e:
                    st.error(f"Misslyckades: {e}")

        st.download_button(
            "⬇️ Ladda ner CSV av aktuell data",
            data=df_to_csv_bytes(df),
            file_name=f"databas_backup_{_ts_str()}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader("🧹 Städa filkopior (Drive)")
        keep_last = st.number_input("Behåll senaste N kopior", min_value=0, value=5, step=1)
        older_than_days = st.number_input("...och / eller radera kopior äldre än (dagar)", min_value=0, value=0, step=1)
        dry_run = st.checkbox("Dry-run (lista utan att radera)", value=True)

        col_clean1, col_clean2 = st.columns(2)
        with col_clean1:
            if st.button("🔎 Förhandsgranska kandidater"):
                res = cleanup_drive_backups(
                    keep_last=int(keep_last),
                    older_than_days=int(older_than_days) if older_than_days > 0 else None,
                    dry_run=True,
                )
                cands = res["candidates"]
                if not cands:
                    st.success("Inga kandidater att radera enligt dina regler.")
                else:
                    df_cands = pd.DataFrame([{
                        "Ordning": f.get("_index"),
                        "Namn": f.get("name"),
                        "Skapad": f.get("createdTime"),
                        "Id": f.get("id"),
                    } for f in cands])
                    st.dataframe(df_cands, use_container_width=True)

        with col_clean2:
            if st.button("🗑️ Radera nu (enligt reglerna)"):
                try:
                    res = cleanup_drive_backups(
                        keep_last=int(keep_last),
                        older_than_days=int(older_than_days) if older_than_days > 0 else None,
                        dry_run=bool(dry_run)
                    )
                    if dry_run:
                        st.info("Dry-run aktiv: ingen radering utfördes.")
                    else:
                        st.success(f"Raderade {len(res['deleted'])} filer.")
                except Exception as e:
                    st.error(f"Misslyckades: {e}")

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj","Kontroll"])

    if meny == "Analys":
        df = uppdatera_berakningar(df, user_rates)
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates, datakalla_default="Yahoo")
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)
    elif meny == "Kontroll":
        kontrollvy(df)

if __name__ == "__main__":
    main()
