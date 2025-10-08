# app.py — MONOLIT, robust Google Sheets + CSV fallback, inga externa moduler utöver gspread (valfritt)
from __future__ import annotations

import os, json, math, traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Aktieanalys & P/S", layout="wide")

# ---------- Hjälpfunktioner ----------
def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

def _round2(x: float) -> float:
    try: return round(float(x), 2)
    except: return 0.0

def _to_float(x: Any) -> float:
    try:
        if x is None: return 0.0
        if isinstance(x, float) and math.isnan(x): return 0.0
        return float(str(x).replace(",", "."))
    except: return 0.0

def _headers_yahoo() -> Dict[str,str]:
    return {"User-Agent":"Mozilla/5.0","Accept":"application/json, text/plain, */*","Connection":"close"}

def _headers_sec() -> Dict[str,str]:
    return {"User-Agent":"ps-analyzer/1.0 (contact: you@example.com)","Accept":"application/json","Connection":"close"}

def _safe_json(url: str, headers: Optional[Dict[str,str]]=None, timeout:int=12) -> Tuple[bool,Dict[str,Any],str]:
    if st.session_state.get("offline_mode", False):
        return False, {}, "Offline-läge aktivt"
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        r.raise_for_status()
        return True, r.json(), ""
    except Exception as e:
        return False, {}, f"{type(e).__name__}: {e}"

# ---------- Globalt UI-läge ----------
if "offline_mode" not in st.session_state:
    st.session_state.offline_mode = False
if "force_csv" not in st.session_state:
    st.session_state.force_csv = False
if "ds_last_error" not in st.session_state:
    st.session_state.ds_last_error = ""

# ---------- Google Sheets + CSV ----------
def _gs_params() -> Tuple[dict,str,str]:
    """Returnerar (service_account_dict|{}, sheet_url, worksheet_name)."""
    sa = st.secrets.get("gsheets", {}).get("service_account", {})
    url = st.secrets.get("gsheets", {}).get("spreadsheet_url", os.environ.get("SHEET_URL",""))
    ws  = st.secrets.get("gsheets", {}).get("worksheet_name", os.environ.get("SHEET_NAME","Aktier"))
    return sa, url, ws

def _service_email() -> str:
    sa, _, _ = _gs_params()
    return sa.get("client_email","")

def _read_gs_dataframe() -> Tuple[pd.DataFrame, str]:
    """Läs GS utan caching av objekt. Return (df, info) eller (tom, fel)."""
    try:
        import gspread  # valfritt, men om ej installerat -> CSV fallback
    except Exception as e:
        return pd.DataFrame(), f"gspread saknas ({e}); använder CSV."

    sa, sheet_url, ws_name = _gs_params()
    if not sa or not sheet_url:
        return pd.DataFrame(), "GS-secrets saknas; använder CSV."

    try:
        gc = gspread.service_account_from_dict(sa)
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(ws_name)
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame(), "Arket är tomt."
        header, rows = values[0], values[1:]
        df = pd.DataFrame(rows, columns=header)
        return df, f"OK ({ws_name})"
    except Exception as e:
        return pd.DataFrame(), f"GS-fel: {e}"

def _write_gs_dataframe(df: pd.DataFrame) -> Optional[str]:
    try:
        import gspread
    except Exception as e:
        return f"gspread saknas ({e})."

    sa, sheet_url, ws_name = _gs_params()
    if not sa or not sheet_url:
        return "GS-secrets saknas."

    try:
        gc = gspread.service_account_from_dict(sa)
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(ws_name)
        body = [list(df.columns)] + df.astype(object).fillna("").values.tolist()
        ws.clear()
        # chunkad update om det är mycket data
        CHUNK = 500
        for i in range(0, len(body), CHUNK):
            ws.update(f"A{i+1}", body[i:i+CHUNK])
        return None
    except Exception as e:
        return f"GS-skrivfel: {e}"

def _csv_path() -> str:
    return os.environ.get("DATA_CSV","data.csv")

def hamta_data() -> Tuple[pd.DataFrame, str]:
    """Läser data. Väljer GS eller CSV beroende på toggle/åtkomst. Return (df, mode_info)."""
    if st.session_state.get("force_csv", False):
        p = _csv_path()
        if os.path.exists(p):
            try:
                return pd.read_csv(p), "CSV"
            except Exception as e:
                st.session_state.ds_last_error = f"CSV-fel: {e}"
                return pd.DataFrame(), "CSV (fel)"
        return pd.DataFrame(), "CSV (saknas)"

    df_gs, info = _read_gs_dataframe()
    if not df_gs.empty:
        return df_gs, "Google Sheets"
    # Om GS tomt/fel -> CSV fallback:
    st.session_state.ds_last_error = info or ""
    p = _csv_path()
    if os.path.exists(p):
        try:
            return pd.read_csv(p), "CSV (fallback)"
        except Exception as e:
            st.session_state.ds_last_error = f"{info} | CSV-fel: {e}"
            return pd.DataFrame(), "CSV (fallback, fel)"
    return pd.DataFrame(), "CSV (fallback, saknas)"

def spara_data(df: pd.DataFrame) -> None:
    """Sparar till primärkällan (GS om möjligt, annars CSV)."""
    if not st.session_state.get("force_csv", False):
        err = _write_gs_dataframe(df)
        if err is None:
            return
        # misslyckades -> spara CSV och visa fel
        st.session_state.ds_last_error = err

    # CSV
    try:
        df.to_csv(_csv_path(), index=False)
    except Exception as e:
        st.session_state.ds_last_error = f"CSV-skrivfel: {e}"

# ---------- Valuta ----------
def las_sparade_valutakurser() -> Dict[str,float]:
    p = "fx.json"
    if os.path.exists(p):
        try: return json.load(open(p,"r",encoding="utf-8"))
        except: pass
    return {"USD":10.0,"NOK":1.0,"CAD":7.5,"EUR":11.0,"SEK":1.0}

def spara_valutakurser(d: Dict[str,float]) -> None:
    json.dump(d, open("fx.json","w",encoding="utf-8"))

@st.cache_data(ttl=3600, show_spinner=False)
def _fx_from_yahoo() -> Dict[str,float]:
    pairs = {"USD":"SEK=X","NOK":"NOKSEK=X","CAD":"CADSEK=X","EUR":"EURSEK=X"}
    out = {"SEK":1.0}
    for code, tick in pairs.items():
        ok, js, err = _safe_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{tick}?interval=1d&range=5d", _headers_yahoo())
        if not ok: continue
        try:
            close = js["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            vals = [x for x in close if isinstance(x,(int,float)) and x]
            if vals: out[code] = float(vals[-1])
        except: pass
    base = las_sparade_valutakurser()
    for k in ["USD","NOK","CAD","EUR"]:
        out[k] = float(out.get(k, base.get(k, 1.0)))
    return out

def hamta_valutakurs(valuta:str, rates:Dict[str,float]) -> float:
    return float(rates.get((valuta or "SEK").upper(), 1.0))

def sidebar_rates() -> Dict[str,float]:
    st.sidebar.header("💱 Valutakurser → SEK")
    st.session_state.offline_mode = st.sidebar.toggle(
        "🔌 Offline (blockera Yahoo/SEC/FX)", value=st.session_state.get("offline_mode", False)
    )
    st.session_state.force_csv = st.sidebar.toggle(
        "🧰 Tvinga CSV-läge", value=st.session_state.get("force_csv", False),
        help="Kringgå Google Sheets helt och läs/skriv endast 'data.csv'."
    )

    saved = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD",10.0)), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK",1.0)), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD",7.5)), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR",11.0)), step=0.01, format="%.4f")
    rates = {"USD":usd,"NOK":nok,"CAD":cad,"EUR":eur,"SEK":1.0}

    c1,c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(rates); st.sidebar.success("Sparat.")
    with c2:
        if st.button("🌐 Live-kurser"):
            if st.session_state.offline_mode:
                st.sidebar.warning("Offline-läge är på.")
            else:
                live = _fx_from_yahoo()
                spara_valutakurser(live); st.sidebar.success("Hämtat."); st.rerun()

    st.sidebar.markdown("---")
    # Datakälla status
    sa_email = _service_email()
    st.sidebar.caption(f"**Datakälla:** {'CSV' if st.session_state.force_csv else 'Google Sheets → CSV fallback'}")
    if sa_email: st.sidebar.caption(f"Service-konto: `{sa_email}` (måste ha delning till arket)")

    if st.sidebar.button("🧪 Testa GS-anslutning"):
        df_test, info = _read_gs_dataframe()
        if not df_test.empty:
            st.sidebar.success(f"GS OK – rader: {len(df_test)}")
        else:
            st.sidebar.error(f"GS problem: {info}")

    uploaded = st.sidebar.file_uploader("⬆️ Importera CSV", type=["csv"])
    if uploaded is not None:
        try:
            imp = pd.read_csv(uploaded)
            spara_data(imp); st.sidebar.success("Importerad & sparad."); st.rerun()
        except Exception as e:
            st.sidebar.error(f"Importfel: {e}")

    if st.sidebar.button("⬇️ Exportera nuvarande som CSV"):
        df_curr, _ = hamta_data()
        st.sidebar.download_button("Ladda ner data.csv", df_curr.to_csv(index=False).encode("utf-8"), file_name="data.csv", mime="text/csv")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Ladda om data"):
        st.cache_data.clear(); st.rerun()

    return rates

# ---------- Yahoo/SEC P/S ----------
def _unix(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_quote_summary(ticker:str) -> Dict[str,Any]:
    ok, js, err = _safe_json(
        f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=price,summaryDetail,defaultKeyStatistics",
        _headers_yahoo()
    )
    if not ok: raise RuntimeError(err)
    try: return js["quoteSummary"]["result"][0]
    except: return {}

@st.cache_data(ttl=1800, show_spinner=False)
def yahoo_timeseries_revenue(ticker:str) -> Dict[str,Any]:
    ok, js, err = _safe_json(
        f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}?type=quarterlyTotalRevenue,trailingTotalRevenue&merge=false",
        _headers_yahoo()
    )
    if not ok: raise RuntimeError(err)
    return js

@st.cache_data(ttl=3600, show_spinner=False)
def yahoo_price_on(ticker:str, date:datetime) -> float:
    if st.session_state.offline_mode: return 0.0
    p1 = _unix(date - timedelta(days=2)); p2 = _unix(date + timedelta(days=3))
    ok, js, err = _safe_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={p1}&period2={p2}&interval=1d", _headers_yahoo())
    if not ok: return 0.0
    try:
        closes = js["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        ts = js["chart"]["result"][0]["timestamp"]
        pairs = [(datetime.fromtimestamp(t, tz=timezone.utc).date(), c or 0.0) for t,c in zip(ts, closes)]
        pairs = [p for p in pairs if p[1] > 0]
        for d,c in pairs:
            if d >= date.date(): return float(c)
        return float(pairs[-1][1]) if pairs else 0.0
    except: return 0.0

@st.cache_data(ttl=86400, show_spinner=False)
def sec_company_tickers() -> Dict[str,int]:
    ok, js, err = _safe_json("https://www.sec.gov/files/company_tickers.json", _headers_sec())
    if not ok: return {}
    out = {}
    try:
        for _, rec in js.items():
            out[str(rec["ticker"]).upper()] = int(rec["cik_str"])
    except: pass
    return out

@st.cache_data(ttl=86400, show_spinner=False)
def sec_company_facts(cik:int) -> Dict[str,Any]:
    ok, js, err = _safe_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", _headers_sec())
    if not ok: return {}
    return js

def sec_shares_series(cik:int) -> List[Tuple[datetime,float]]:
    js = sec_company_facts(cik)
    out: List[Tuple[datetime,float]] = []
    try:
        facts = js["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]
        unit = facts.get("shares") or facts.get("SHRS") or list(facts.values())[0]
        for item in unit:
            d = datetime.fromisoformat(item["end"] + "T00:00:00")
            val = _to_float(item.get("val"))
            if val > 0:
                out.append((d, val/1_000_000.0))
    except: pass
    out.sort(key=lambda x: x[0])
    return out

def _nearest_value(series: List[Tuple[datetime,float]], target: datetime, max_days:int=30) -> float:
    if not series: return 0.0
    best, diff = None, 10**9
    for d,v in series:
        td = abs((d-target).days)
        if td < diff: best, diff = v, td
    return float(best) if best is not None and diff <= max_days else 0.0

def compute_quarter_ps(ticker:str) -> Dict[str,Any]:
    if st.session_state.offline_mode:
        raise RuntimeError("Offline-läge: inga nätanrop.")
    out: Dict[str,Any] = {}

    qs = yahoo_quote_summary(ticker)
    out["Bolagsnamn"] = qs.get("price",{}).get("shortName") or qs.get("price",{}).get("longName") or ""
    out["Valuta"] = qs.get("price",{}).get("currency","USD")
    out["Aktuell kurs"] = _to_float(qs.get("price",{}).get("regularMarketPrice",0))
    out["Årlig utdelning"] = _to_float(qs.get("summaryDetail",{}).get("dividendRate",0))

    # shares now (miljoner)
    sh = _to_float(qs.get("defaultKeyStatistics",{}).get("sharesOutstanding")) or _to_float(qs.get("price",{}).get("sharesOutstanding"))
    if sh > 0:
        out["Utestående aktier"] = _round2(sh/1_000_000.0)
        out["Källa Utestående aktier"] = "Yahoo/info"
        out["TS Utestående aktier"] = _now_iso()

    # ps ttm
    ps_ttm = _to_float(qs.get("summaryDetail",{}).get("priceToSalesTrailing12Months")) or _to_float(qs.get("defaultKeyStatistics",{}).get("priceToSalesTrailing12Months"))
    if ps_ttm > 0:
        out["P/S"] = _round2(ps_ttm); out["Källa P/S"] = "Yahoo/ps_ttm"; out["TS P/S"] = _now_iso()

    # quarterly revenue
    qjs = yahoo_timeseries_revenue(ticker)
    quarters: List[Tuple[datetime,float]] = []
    try:
        arr = qjs["timeseries"]["result"][0]["quarterlyTotalRevenue"]
        for it in arr:
            val = _to_float(it.get("reportedValue",{}).get("raw"))
            asof = it.get("asOfDate")
            if val > 0 and asof:
                quarters.append((datetime.fromisoformat(asof+"T00:00:00"), val/1_000_000.0))
    except: pass
    quarters.sort(key=lambda x:x[0], reverse=True)
    quarters = quarters[:4]

    cik = sec_company_tickers().get(ticker.upper(), 0)
    sec_series = sec_shares_series(int(cik)) if cik else []

    for idx, (d, rev_mn) in enumerate(quarters):
        d_plus = d + timedelta(days=1)
        px = yahoo_price_on(ticker, d_plus)
        sh_mn = _nearest_value(sec_series, d_plus, 30) if sec_series else (out.get("Utestående aktier",0.0))
        ps = (px * sh_mn) / rev_mn if (px>0 and sh_mn>0 and rev_mn>0) else 0.0
        qn = idx+1
        out[f"P/S Q{qn}"] = _round2(ps)
        out[f"P/S Q{qn} datum"] = d.date().isoformat()
        out[f"Källa P/S Q{qn}"] = "Computed/Yahoo-revenue+SEC-shares+1d-after" if sec_series else "Computed/Yahoo-revenue+Yahoo-shares+1d-after"
    return out

# ---------- Data-schema & beräkningar ----------
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

def to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = to_num(df, ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","CAGR 5 år (%)",
                     "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                     "Utestående aktier","Aktuell kurs","Årlig utdelning","Antal aktier"])
    for i, r in df.iterrows():
        vals = [r.get("P/S Q1",0), r.get("P/S Q2",0), r.get("P/S Q3",0), r.get("P/S Q4",0)]
        vals = [float(x) for x in vals if float(x) > 0]
        ps_snitt = _round2(np.mean(vals)) if vals else _round2(float(r.get("P/S",0)))
        df.at[i, "P/S-snitt"] = ps_snitt

        cagr = float(r.get("CAGR 5 år (%)",0.0))
        cagr = 50.0 if cagr>100 else (-20.0 if cagr<-50 else cagr)
        g = cagr/100.0
        next_rev = float(r.get("Omsättning nästa år",0.0))
        if next_rev>0:
            df.at[i,"Omsättning om 2 år"] = _round2(next_rev*(1+g))
            df.at[i,"Omsättning om 3 år"] = _round2(next_rev*((1+g)**2))

        a_mn = float(r.get("Utestående aktier",0.0))
        if a_mn>0 and ps_snitt>0:
            df.at[i,"Riktkurs idag"]    = _round2(float(r.get("Omsättning idag",0.0))*ps_snitt/a_mn)
            df.at[i,"Riktkurs om 1 år"] = _round2(float(r.get("Omsättning nästa år",0.0))*ps_snitt/a_mn)
            df.at[i,"Riktkurs om 2 år"] = _round2(float(df.at[i,"Omsättning om 2 år"])*ps_snitt/a_mn)
            df.at[i,"Riktkurs om 3 år"] = _round2(float(df.at[i,"Omsättning om 3 år"])*ps_snitt/a_mn)
        else:
            for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                df.at[i,k]=0.0
    return df

# ---------- UI: Lägg till / uppdatera ----------
def _apply_row(df: pd.DataFrame, ticker:str, data:Dict[str,Any]) -> pd.DataFrame:
    if ticker not in df["Ticker"].astype(str).values:
        base = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Källa Aktuell kurs","Senast manuellt uppdaterad","Senast auto uppdaterad"] else "") for c in df.columns}
        base["Ticker"] = ticker
        df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
    for k,v in data.items():
        if k in df.columns: df.loc[df["Ticker"]==ticker, k] = v
    return df

def _fetch_and_apply(df: pd.DataFrame, ticker:str) -> Tuple[pd.DataFrame,bool,str]:
    try:
        with st.spinner(f"Hämtar Yahoo/SEC för {ticker} …"):
            st.cache_data.clear()
            data = compute_quarter_ps(ticker)
            if not data: return df, False, "Inga data."
            if data.get("Aktuell kurs",0)>0: data["Källa Aktuell kurs"]="Yahoo/price"
            df = _apply_row(df, ticker, data)
            df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = _now_iso()
            df = uppdatera_berakningar(df)
            spara_data(df)
        return df, True, "Klart."
    except Exception as e:
        return df, False, f"Fel: {type(e).__name__}: {e}"

def edit_view(df: pd.DataFrame) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [""] + [f"{r['Bolagsnamn']} ({r['Ticker']})" for _,r in vis.iterrows()]
    val = st.selectbox("Välj bolag (tomt = nytt)", labels, index=0)
    bef = pd.Series({}, dtype=object)
    if val:
        tkr = val[val.rfind("(")+1:val.rfind(")")]
        r = df[df["Ticker"]==tkr]
        if not r.empty: bef = r.iloc[0]

    with st.form("frm"):
        c1,c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo)", value=bef.get("Ticker","")).upper()
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
        ok = st.form_submit_button("💾 Spara & hämta (Yahoo + SEC)")

    if ok and ticker:
        base = {"Ticker":ticker,"Utestående aktier":utest,"Antal aktier":antal,"P/S":ps_ttm,"P/S Q1":ps1,"P/S Q2":ps2,"P/S Q3":ps3,"P/S Q4":ps4,
                "Omsättning idag":oms_i,"Omsättning nästa år":oms_n,"Senast manuellt uppdaterad":_now_iso()}
        df = _apply_row(df, ticker, base); spara_data(df)
        df, ok2, msg = _fetch_and_apply(df, ticker)
        (st.success if ok2 else st.error)(msg); st.rerun()

    if not bef.empty and st.button("↻ Hämta igen denna ticker (Yahoo + SEC)"):
        tkr = str(bef.get("Ticker","")).upper()
        df, ok2, msg = _fetch_and_apply(df, tkr)
        (st.success if ok2 else st.error)(msg); st.rerun()

    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
                 use_container_width=True)
    return df

# ---------- Analys & Portfölj ----------
def analys_view(df: pd.DataFrame) -> None:
    st.header("📈 Analys")
    if df.empty: st.info("Tom databas."); return
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _,r in vis.iterrows()]
    if "idx" not in st.session_state: st.session_state.idx = 0
    st.session_state.idx = st.number_input("Visa bolag #", 0, max(0,len(labels)-1), st.session_state.idx, 1)
    st.selectbox("Eller välj i lista", labels, index=st.session_state.idx if labels else 0, key="sel")
    r = vis.iloc[st.session_state.idx]
    cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Senast manuellt uppdaterad","Senast auto uppdaterad"]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)
    st.markdown("### Hela databasen"); st.dataframe(df, use_container_width=True)

def portfolio_view(df: pd.DataFrame, rates: Dict[str,float]) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"]>0].copy()
    if port.empty: st.info("Du äger inga aktier."); return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, rates))
    port["Värde (SEK)"] = port["Antal aktier"]*port["Aktuell kurs"]*port["Växelkurs"]
    tot = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = (port["Värde (SEK)"]/tot*100).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"]*port["Årlig utdelning"]*port["Växelkurs"]
    st.markdown(f"**Totalt portföljvärde:** {_round2(tot)} SEK")
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    st.markdown(f"**Total kommande utdelning:** {_round2(tot_utd)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {_round2(tot_utd/12)} SEK")
    st.dataframe(port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
                 use_container_width=True)

# ---------- MAIN ----------
def main():
    st.title("📊 Aktieanalys & P/S (monolit)")
    rates = sidebar_rates()

    df, mode = hamta_data()
    st.caption(f"Datakälla nu: **{mode}** • {_now_iso()}")
    if st.session_state.ds_last_error:
        st.warning(f"Datakälla-info: {st.session_state.ds_last_error}")

    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    df = ensure_schema(df)

    meny = st.sidebar.radio("📌 Välj vy", ["Lägg till / uppdatera","Analys","Portfölj"], index=0)
    if meny == "Lägg till / uppdatera":
        df = edit_view(df)
    elif meny == "Analys":
        analys_view(df)
    else:
        df = uppdatera_berakningar(df)
        portfolio_view(df, rates)

if __name__ == "__main__":
    main()
