# app.py  — monolitisk Streamlit-app (Google Sheets + CSV fallback)
from __future__ import annotations
import os, io, time, math, json, datetime as dt
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import requests
import streamlit as st

# ============================== Konfiguration ==============================

APP_TITLE = "📊 Aktieanalys & P/S (monolit)"
CSV_PATH = "db_cache.csv"                      # fallback-fil lokalt
SHEET_TAB = "Data"                             # bladnamn i Google Sheet
HTTP_TIMEOUT = 20
YF_BASE = "https://query1.finance.yahoo.com"
SEC_BASE = "https://data.sec.gov"
UA = "ps-app/1.0 (contact: example@example.com)"  # byt gärna till din mail för SEC

# ============================== Kolumnschema ==============================

FINAL_COLS = [
    "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning",
    "Utestående aktier","Antal aktier","P/S",
    "P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
    "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S",
    "Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "CAGR 5 år (%)","P/S-snitt",
    "Senast manuellt uppdaterad","Senast auto uppdaterad",
    "TS P/S","TS Utestående aktier","TS Omsättning"
]

def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")

# ============================== Hjälpare ==============================

def _safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in FINAL_COLS:
        if c not in df.columns:
            if any(k in c.lower() for k in ["kurs","omsättning","p/s","utdelning","cagr","aktier","riktkurs","snitt","antal"]):
                df[c] = 0.0
            else:
                df[c] = ""
    # typer
    num_cols = ["Aktuell kurs","Årlig utdelning","Utestående aktier","Antal aktier","P/S",
                "P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "CAGR 5 år (%)","P/S-snitt"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _headers_json():
    return {"User-Agent": UA, "Accept": "application/json"}

def _get_json(url: str, headers: dict | None = None, params: dict | None = None) -> dict:
    if st.session_state.get("offline_mode", False):
        raise RuntimeError("Offline-läge: nätanrop avstängda")
    r = requests.get(url, headers=headers or _headers_json(), params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def _yahoo_qs(ticker: str, modules: List[str]) -> dict:
    url = f"{YF_BASE}/v10/finance/quoteSummary/{ticker}"
    params = {"modules": ",".join(modules)}
    js = _get_json(url, params=params)
    return js.get("quoteSummary", {}).get("result", [{}])[0] or {}

def _yahoo_price_at(ticker: str, date: dt.date) -> Optional[float]:
    """Pris på (eller närmast efter) datumet (daglig)."""
    start = int(dt.datetime(date.year, date.month, date.day).timestamp())
    end = start + 60*60*24*5  # upp till 5 dagar framåt (nästa handelsdag)
    url = f"{YF_BASE}/v8/finance/chart/{ticker}"
    js = _get_json(url, params={"period1": start, "period2": end, "interval": "1d"})
    res = js.get("chart", {}).get("result", [])
    if not res: return None
    timestamps = res[0].get("timestamp", [])
    closes = res[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
    for ts, cl in zip(timestamps, closes):
        if cl is not None:
            return float(cl)
    return None

def _yahoo_quarter_revenues(ticker: str) -> List[Tuple[dt.date, float]]:
    """[(kvartalsdatum, revenue)] – nyast först. Yahoo quarterly incomeStatement."""
    modules = ["incomeStatementHistoryQuarterly"]
    js = _yahoo_qs(ticker, modules)
    items = (js.get("incomeStatementHistoryQuarterly", {}) or {}).get("incomeStatementHistory", []) or []
    out = []
    for it in items:
        end_date = it.get("endDate", {}).get("fmt")
        rev = _safe_float((it.get("totalRevenue") or {}).get("raw"))
        if end_date:
            try:
                d = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
                out.append((d, rev))
            except Exception:
                pass
    # säkerställ nyast först
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _yahoo_quote_basics(ticker: str) -> dict:
    js = _yahoo_qs(ticker, ["price","summaryDetail","defaultKeyStatistics"])
    price = (js.get("price") or {})
    dks = (js.get("defaultKeyStatistics") or {})
    sd  = (js.get("summaryDetail") or {})
    return {
        "name": price.get("longName") or price.get("shortName") or "",
        "currency": price.get("currency") or "",
        "last": _safe_float(price.get("regularMarketPrice", {}).get("raw")),
        "div": _safe_float(sd.get("trailingAnnualDividendRate", {}).get("raw")),
        "shares_yahoo": _safe_float(dks.get("sharesOutstanding", {}).get("raw")),
    }

# ----------------------- SEC helpers -----------------------

def _sec_cik_from_ticker(ticker: str) -> Optional[str]:
    try:
        js = _get_json(f"{SEC_BASE}/files/company_tickers.json")
        # filen är {index:{cik, ticker, title}}
        for _, row in js.items():
            if (row.get("ticker") or "").upper() == ticker.upper():
                return str(row.get("cik_str")).zfill(10)
    except Exception:
        pass
    return None

def _sec_companyfacts(cik: str) -> dict:
    return _get_json(f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json", headers=_headers_json())

def _sec_shares_near(cik: str, around: dt.date) -> Optional[float]:
    """Försök hitta utestående aktier inom ±7 dagar efter 'around'."""
    try:
        facts = _sec_companyfacts(cik).get("facts", {}).get("dei", {})
        candidates = []
        for key in ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding", "EntityPublicFloatShares"]:
            if key in facts:
                for unit, arr in (facts[key].get("units") or {}).items():
                    for v in arr:
                        d = v.get("end") or v.get("fy")  # end (YYYY-MM-DD)
                        if not d: continue
                        try:
                            ddate = dt.datetime.strptime(d[:10], "%Y-%m-%d").date()
                        except Exception:
                            continue
                        val = _safe_float(v.get("val"))
                        candidates.append((ddate, val))
        if not candidates: return None
        # välj närmast efter rapportdatum (upp till +7 dagar), annars närmast före
        after = [(d,v) for d,v in candidates if d >= around and (d-around).days <= 7]
        if after:
            after.sort(key=lambda x: x[0])
            return float(after[0][1])
        before = [(d,v) for d,v in candidates if d <= around]
        if before:
            before.sort(key=lambda x: x[0], reverse=True)
            return float(before[0][1])
    except Exception:
        return None
    return None

# ----------------------- P/S-beräkning -----------------------

def _compute_quarter_ps(ticker: str) -> Dict[str, dict]:
    """
    Returnerar:
    {
      'Q1': {'ps': float, 'date': 'YYYY-MM-DD', 'src': 'Computed/FY26 Q2/price@2025-09-01'},
      'Q2': {...}, ...
    }
    Där Q1 = senaste kvartalet, Q2 = näst senaste, etc.
    """
    out: Dict[str, dict] = {"Q1": {}, "Q2": {}, "Q3": {}, "Q4": {}}

    basics = _yahoo_quote_basics(ticker)
    revs = _yahoo_quarter_revenues(ticker)  # nyast först
    if len(revs) < 4:
        return out

    # bygg TTM per kvartal i ordningen Q1..Q4
    # revs[0] nyast → TTM_1 = sum(revs[0:4]), TTM_2 = sum(revs[1:5]), ...
    ttm_list: List[Tuple[dt.date, float]] = []
    for i in range(0, min(4, len(revs)-3)):
        window = revs[i:i+4]
        ttm = sum(x[1] for x in window)
        end_date = window[0][0]  # senaste kvartalets slut i fönstret
        ttm_list.append((end_date, ttm))  # index 0 → Q1

    # SEC shares vid respektive kvartalsslut (+1 dag)
    cik = None
    try:
        cik = _sec_cik_from_ticker(ticker)
    except Exception:
        cik = None

    for idx, (q_date, ttm_rev) in enumerate(ttm_list[:4], start=1):
        # pris nästa handelsdag (eller samma)
        price_date = q_date + dt.timedelta(days=1)
        price = _yahoo_price_at(ticker, price_date) or basics["last"]

        shares = None
        if cik:
            shares = _sec_shares_near(cik, q_date)
        if not shares or shares <= 0:
            shares = basics.get("shares_yahoo", 0.0)
        # P/S
        ps = 0.0
        if ttm_rev and price and shares:
            ps = (price * shares) / ttm_rev
        # text
        # gissa FY-etikett från månad (FY slutar ofta med Q4), enkel FY-beteckning
        fy_year = q_date.year if q_date.month >= 2 else q_date.year - 1
        q_slot = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}  # ej exakt men räcker som etikett
        label = f"Computed/FY{str(fy_year)[-2:]} {q_slot.get(((q_date.month-1)//3+1), 'Q?')}/price@{price_date.isoformat()}"

        out[f"Q{idx}"] = {
            "ps": round(float(ps), 2),
            "date": q_date.isoformat(),
            "src": label
        }
    return out

# ============================== Google Sheets/CSV ==============================

def _gs_context():
    """Försök bygga gspread-klient från secrets. Returnerar (gc, sh) eller (None, None)."""
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception:
        return None, None

    # Stöd för två sätt:
    # 1) st.secrets["gs"]["service_account"] = { ... json ... }, st.secrets["gs"]["sheet_key"]
    # 2) st.secrets["GS_SERVICE_ACCOUNT_JSON"] = "...json str...", st.secrets["GS_SHEET_KEY"]
    svc = None
    key = None
    if "gs" in st.secrets:
        block = st.secrets["gs"]
        svc = dict(block.get("service_account", {}))
        key = block.get("sheet_key", "")
    else:
        if "GS_SERVICE_ACCOUNT_JSON" in st.secrets:
            try:
                svc = json.loads(st.secrets["GS_SERVICE_ACCOUNT_JSON"])
            except Exception:
                svc = None
        key = st.secrets.get("GS_SHEET_KEY", "")

    if not svc or not key:
        return None, None

    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(svc, scopes=scopes)
    try:
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(key)
        return gc, sh
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def gs_read_df() -> Tuple[pd.DataFrame, str]:
    gc, sh = _gs_context()
    if not gc or not sh:
        # CSV fallback
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            return _ensure_schema(df), "CSV (fallback, saknas)"
        else:
            return _ensure_schema(pd.DataFrame()), "CSV (fallback, tom)"
    try:
        ws = sh.worksheet(SHEET_TAB)
    except Exception:
        # skapa blad om det saknas
        try:
            ws = sh.add_worksheet(SHEET_TAB, rows=1000, cols=50)
            ws.append_row(FINAL_COLS)
        except Exception:
            return _ensure_schema(pd.DataFrame()), "Google Sheets (kunde ej skapa blad)"
    try:
        values = ws.get_all_values()
        if not values:
            return _ensure_schema(pd.DataFrame()), "Google Sheets (tomt)"
        hdr = values[0]
        rows = values[1:]
        df = pd.DataFrame(rows, columns=hdr)
        return _ensure_schema(df), "Google Sheets"
    except Exception:
        return _ensure_schema(pd.DataFrame()), "Google Sheets (läsfel)"

def gs_write_df(df: pd.DataFrame) -> str:
    df = _ensure_schema(df)
    gc, sh = _gs_context()
    if not gc or not sh:
        # CSV fallback
        df.to_csv(CSV_PATH, index=False)
        return "CSV (fallback)"
    try:
        ws = None
        try:
            ws = sh.worksheet(SHEET_TAB)
        except Exception:
            ws = sh.add_worksheet(SHEET_TAB, rows=1000, cols=50)
        # skriv om helt
        ws.clear()
        ws.append_row(FINAL_COLS)
        # chunkvis
        data = df[FINAL_COLS].astype(str).values.tolist()
        # gspread batch begränsning → skriv i bitar
        step = 500
        for i in range(0, len(data), step):
            ws.append_rows(data[i:i+step], value_input_option="RAW")
        return "Google Sheets"
    except Exception:
        df.to_csv(CSV_PATH, index=False)
        return "CSV (fallback)"

# ============================== Valutakurser ==============================

@st.cache_data(show_spinner=False)
def load_saved_rates() -> Dict[str, float]:
    try:
        if os.path.exists("rates.json"):
            with open("rates.json","r",encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"USD":10.0,"NOK":1.0,"CAD":7.5,"EUR":11.0,"SEK":1.0}

def save_rates(d: Dict[str, float]) -> None:
    with open("rates.json","w",encoding="utf-8") as f:
        json.dump(d,f)

def fx_sidebar() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")
    # Offline-läge
    off = st.sidebar.toggle("🔌 Säkerhetsläge (offline)", value=st.session_state.get("offline_mode", False),
                            help="Stoppar alla nätanrop (Yahoo/SEC/FX). Använd sparade värden.")
    st.session_state["offline_mode"] = bool(off)

    saved = load_saved_rates()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD", 10.0)), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK", 1.0)), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD", 7.5)), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR", 11.0)), step=0.01, format="%.4f")
    rates = {"USD":usd,"NOK":nok,"CAD":cad,"EUR":eur,"SEK":1.0}

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara kurser"):
            save_rates(rates)
            st.sidebar.success("Valutakurser sparade.")
    with c2:
        if st.button("🌐 Live-kurser"):
            if st.session_state.get("offline_mode", False):
                st.sidebar.warning("Offline-läge är på. Stäng av för att hämta live-kurser.")
            else:
                live = live_rates_yahoo()
                if live:
                    save_rates(live)
                    st.sidebar.success("Live-kurser hämtade & sparade.")
                    st.rerun()
                else:
                    st.sidebar.error("Kunde inte hämta live-kurser just nu. Behåller dina sparade värden.")

    st.sidebar.markdown("---")
    # Datakälla-indikator + test
    df_tmp, src = gs_read_df()
    st.sidebar.caption(f"Datakälla nu: **{src}** • {now_stamp()}")
    if "Google Sheets" not in src:
        st.sidebar.warning("GS problem: GS-secrets saknas; använder CSV.")
    if st.sidebar.button("🧪 Testa GS-anslutning"):
        _, src2 = gs_read_df()
        st.sidebar.info(f"Resultat: {src2}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Importera CSV")
    up = st.sidebar.file_uploader("Drag and drop file here", type=["csv"])
    if up is not None:
        try:
            dfi = pd.read_csv(up)
            dfi = _ensure_schema(dfi)
            srcw = gs_write_df(dfi)
            st.sidebar.success(f"Importerade {len(dfi)} rader → {srcw}")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Importfel: {e}")

    if st.sidebar.button("⬇️ Exportera nuvarande som CSV"):
        try:
            cur, _ = gs_read_df()
            cur.to_csv(CSV_PATH, index=False)
            st.sidebar.success(f"Exporterade till {CSV_PATH}")
        except Exception as e:
            st.sidebar.error(f"Exportfel: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data"):
        st.cache_data.clear()
        st.rerun()

    return rates

def live_rates_yahoo() -> Dict[str, float] | None:
    try:
        pairs = {
            "USD": "USDSEK=X",
            "NOK": "NOKSEK=X",
            "CAD": "CADSEK=X",
            "EUR": "EURSEK=X",
        }
        out = {"SEK":1.0}
        for k, sym in pairs.items():
            js = _yahoo_qs(sym, ["price"])
            pr = (js.get("price") or {}).get("regularMarketPrice", {}).get("raw")
            if pr is None:
                return None
            out[k] = float(pr)
        return out
    except Exception:
        return None

# ============================== Beräkningar ==============================

def apply_calculations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i, r in df.iterrows():
        ps_vals = [r.get("P/S Q1",0.0), r.get("P/S Q2",0.0), r.get("P/S Q3",0.0), r.get("P/S Q4",0.0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        df.at[i, "P/S-snitt"] = round(np.mean(ps_clean), 2) if ps_clean else float(r.get("P/S",0.0))

        # CAGR clamp
        cagr = float(r.get("CAGR 5 år (%)", 0.0))
        if cagr > 100.0: cagr = 50.0
        if cagr < 0.0:   cagr = 2.0
        g = cagr/100.0

        # omsättning-projektioner
        next_rev = float(r.get("Omsättning nästa år",0.0))
        if next_rev > 0:
            df.at[i,"Omsättning om 2 år"] = round(next_rev*(1+g),2)
            df.at[i,"Omsättning om 3 år"] = round(next_rev*((1+g)**2),2)

        # riktkurs (använder miljoner i oms & aktier)
        ps_use = float(df.at[i,"P/S-snitt"]) if float(df.at[i,"P/S-snitt"])>0 else float(r.get("P/S",0.0))
        shares_mn = float(r.get("Utestående aktier",0.0))
        if ps_use>0 and shares_mn>0:
            df.at[i,"Riktkurs idag"]    = round((float(r.get("Omsättning idag",0.0))     * ps_use) / shares_mn, 2)
            df.at[i,"Riktkurs om 1 år"] = round((float(r.get("Omsättning nästa år",0.0)) * ps_use) / shares_mn, 2)
            df.at[i,"Riktkurs om 2 år"] = round((float(df.at[i,"Omsättning om 2 år"])    * ps_use) / shares_mn, 2)
            df.at[i,"Riktkurs om 3 år"] = round((float(df.at[i,"Omsättning om 3 år"])    * ps_use) / shares_mn, 2)
        else:
            for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i,k] = 0.0
    return df

# ============================== Yahoo + SEC orchestration ==============================

def fetch_and_fill_ticker(df: pd.DataFrame, ticker: str, row_index: Optional[int]=None) -> Tuple[pd.DataFrame, dict]:
    """
    Hämtar alla data + beräknar P/S Q1..Q4 och uppdaterar df.
    Returnerar (ny_df, log_dict)
    """
    log = {"ticker": ticker, "steps": []}
    t = ticker.upper().strip()
    # basics
    try:
        b = _yahoo_quote_basics(t)
        log["steps"].append("Yahoo basics OK")
    except Exception as e:
        b = {"name":"","currency":"","last":0.0,"div":0.0,"shares_yahoo":0.0}
        log["steps"].append(f"Yahoo basics FAIL: {e}")

    # P/S kvartal
    try:
        qps = _compute_quarter_ps(t)
        log["steps"].append("Compute Q-PS OK")
    except Exception as e:
        qps = {"Q1":{},"Q2":{},"Q3":{},"Q4":{}}
        log["steps"].append(f"Compute Q-PS FAIL: {e}")

    # skapa eller uppdatera rad
    if row_index is None:
        exists = df[df["Ticker"].str.upper()==t]
        if not exists.empty:
            row_index = int(exists.index[0])
        else:
            # ny rad
            empty_row = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta"] else "") for c in FINAL_COLS}
            empty_row["Ticker"] = t
            df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
            row_index = len(df)-1

    i = row_index
    # skriv basics
    df.at[i,"Ticker"] = t
    if b["name"]:
        df.at[i,"Bolagsnamn"] = b["name"]
    if b["currency"]:
        df.at[i,"Valuta"] = b["currency"]
    if b["last"]>0:
        df.at[i,"Aktuell kurs"] = b["last"]
        df.at[i,"Källa Aktuell kurs"] = "Yahoo/price"
    df.at[i,"Årlig utdelning"] = b.get("div", 0.0)

    # Utest. aktier → miljoner
    shares = b.get("shares_yahoo", 0.0)
    if shares>0:
        df.at[i,"Utestående aktier"] = round(shares/1_000_000.0, 2)
        df.at[i,"Källa Utestående aktier"] = "Yahoo/keyStats"
        df.at[i,"TS Utestående aktier"] = now_stamp()

    # P/S (TTM) från Yahoo finns inte robust – lämna 0 och använd snitt av Q
    df.at[i,"P/S"] = 0.0
    df.at[i,"TS P/S"] = now_stamp()
    df.at[i,"Källa P/S"] = "Yahoo/ps_ttm (n/a)"

    # skriv Q1..Q4
    for q in (1,2,3,4):
        blk = qps.get(f"Q{q}", {}) or {}
        df.at[i, f"P/S Q{q}"] = float(blk.get("ps", 0.0))
        df.at[i, f"P/S Q{q} datum"] = blk.get("date","")
        df.at[i, f"Källa P/S Q{q}"] = blk.get("src","")

    # metadata
    df.at[i,"Senast auto uppdaterad"] = now_stamp()

    return df, log

# ============================== UI – vyer ==============================

def view_add_update(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / uppdatera bolag")

    # välj/scroll
    vis = df.sort_values(["Bolagsnamn","Ticker"]).reset_index(drop=True)
    options = [""] + [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis.iterrows()]
    sel = st.selectbox("Välj bolag (tomt = nytt)", options)
    row_index = None
    if sel:
        t = sel.split("(")[-1].rstrip(")")
        row = df[df["Ticker"]==t]
        if not row.empty:
            row_index = int(row.index[0])

    with st.form("form_bolag"):
        ticker = st.text_input("Ticker (Yahoo)", value=(df.iloc[row_index]["Ticker"] if row_index is not None else "")).upper()
        cols = st.columns(2)
        with cols[0]:
            utest = st.number_input("Utestående aktier (miljoner)", value=float(df.iloc[row_index]["Utestående aktier"]) if row_index is not None else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(df.iloc[row_index]["Antal aktier"]) if row_index is not None else 0.0)
            ps_ttm = st.number_input("P/S (TTM)", value=float(df.iloc[row_index]["P/S"]) if row_index is not None else 0.0)
            ps1 = st.number_input(f"P/S Q1 — {df.iloc[row_index]['P/S Q1 datum'] if row_index is not None else '–'}", value=float(df.iloc[row_index]["P/S Q1"]) if row_index is not None else 0.0)
            ps2 = st.number_input(f"P/S Q2 — {df.iloc[row_index]['P/S Q2 datum'] if row_index is not None else '–'}", value=float(df.iloc[row_index]["P/S Q2"]) if row_index is not None else 0.0)
        with cols[1]:
            ps3 = st.number_input(f"P/S Q3 — {df.iloc[row_index]['P/S Q3 datum'] if row_index is not None else '–'}", value=float(df.iloc[row_index]["P/S Q3"]) if row_index is not None else 0.0)
            ps4 = st.number_input(f"P/S Q4 — {df.iloc[row_index]['P/S Q4 datum'] if row_index is not None else '–'}", value=float(df.iloc[row_index]["P/S Q4"]) if row_index is not None else 0.0)
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(df.iloc[row_index]["Omsättning idag"]) if row_index is not None else 0.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(df.iloc[row_index]["Omsättning nästa år"]) if row_index is not None else 0.0)

        c1, c2 = st.columns(2)
        with c1:
            submit = st.form_submit_button("💾 Spara")
        with c2:
            fetch = st.form_submit_button("💾 Spara & hämta (Yahoo + SEC)")

    if submit and ticker:
        # spara manuellt (utan hämt)
        if row_index is None:
            new_row = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta"] else "") for c in FINAL_COLS}
            new_row.update({
                "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
                "P/S": ps_ttm,"P/S Q1": ps1,"P/S Q2": ps2,"P/S Q3": ps3,"P/S Q4": ps4,
                "Omsättning idag": oms_idag,"Omsättning nästa år": oms_next,
                "Senast manuellt uppdaterad": now_stamp()
            })
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df.loc[row_index, ["Ticker","Utestående aktier","Antal aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                               "Omsättning idag","Omsättning nästa år"]] = [
                ticker, utest, antal, ps_ttm, ps1, ps2, ps3, ps4, oms_idag, oms_next
            ]
            df.loc[row_index, "Senast manuellt uppdaterad"] = now_stamp()
        st.success("Sparat.")
        return df

    if fetch and ticker:
        df2, log = fetch_and_fill_ticker(df, ticker, row_index)
        st.session_state["last_log"] = log
        st.success("Sparat och uppdaterat från Yahoo/SEC.")
        return df2

    if row_index is not None and st.button("↻ Hämta igen denna ticker (Yahoo + SEC)"):
        t = df.iloc[row_index]["Ticker"]
        df2, log = fetch_and_fill_ticker(df, t, row_index)
        st.session_state["last_log"] = log
        st.success("Hämtat igen.")
        return df2

    # visa senaste logg
    if "last_log" in st.session_state:
        lg = st.session_state["last_log"]
        with st.expander("🔎 Senaste hämtlogg"):
            st.json(lg)

    # lista
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    tmp = df.copy()
    tmp["_sort"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    st.dataframe(tmp.sort_values(by=["_sort","Bolagsnamn"]).head(10)[
        ["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    ], use_container_width=True)
    return df

def view_analysis(df: pd.DataFrame):
    st.subheader("📈 Analys")
    vis = df.sort_values(["Bolagsnamn","Ticker"]).reset_index(drop=True)
    if vis.empty:
        st.info("Inga poster ännu.")
        return
    idx = st.number_input("Visa bolag #", min_value=0, max_value=len(vis)-1, value=0, step=1)
    st.dataframe(vis.iloc[[idx]][[
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S",
        "P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
        "Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"
    ]], use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

def view_mass_update(df: pd.DataFrame) -> pd.DataFrame:
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo/SEC"):
        bar = st.sidebar.progress(0.0)
        logs = []
        tickers = [t for t in df["Ticker"].astype(str).tolist() if t]
        total = len(tickers)
        for k, t in enumerate(tickers, start=1):
            df, lg = fetch_and_fill_ticker(df, t, None)
            logs.append(lg)
            bar.progress(k/total)
            time.sleep(0.15)
        st.session_state["last_mass_logs"] = logs
        st.sidebar.success("Massuppdatering klar.")
        return df
    if "last_mass_logs" in st.session_state:
        with st.expander("📜 Loggar senaste massuppdatering"):
            st.json(st.session_state["last_mass_logs"])
    return df

# ============================== MAIN ==============================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    rates = fx_sidebar()

    # Läs data
    df, src = gs_read_df()
    if df.empty:
        df = _ensure_schema(pd.DataFrame(columns=FINAL_COLS))

    # Massuppdateringsknapp i sidopanelen
    df = view_mass_update(df)

    # Meny
    tab = st.sidebar.radio("📌 Välj vy", ["Lägg till / uppdatera bolag","Analys"])
    if tab == "Lägg till / uppdatera bolag":
        df = view_add_update(df)
    else:
        df = apply_calculations(df)
        view_analysis(df)

    # Beräkningar + spara
    df = apply_calculations(df)
    target = gs_write_df(df)
    st.caption(f"Senast sparad till: **{target}** • {now_stamp()}")

if __name__ == "__main__":
    main()
