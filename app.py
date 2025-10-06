# app.py – stabil version som normaliserar bladet, hanterar Google creds säkert,
# och inte bråkar om kolumner. Bara "Ticker" krävs i arket; resten skapas.

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ---- Tredjeparts (måste finnas i requirements.txt) ----
# streamlit, pandas, numpy, gspread, google-auth, yfinance, pytz (valfritt)
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

try:
    import yfinance as yf
except Exception:
    yf = None

# ============= Grundinställningar =============
st.set_page_config(page_title="K-pf-rslag – stabil", layout="wide")

# Lokal Stockholm-tidstempel (tål avsaknad av pytz)
def now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


# ============= Google Sheets – robust klient =============
def _load_credentials_dict() -> dict:
    """
    Hämtar GOOGLE_CREDENTIALS från st.secrets. Tål:
      - dict (rekommenderat)
      - JSON-sträng
      - private_key med \\n → \n
    """
    if "GOOGLE_CREDENTIALS" not in st.secrets:
        raise RuntimeError("Saknar GOOGLE_CREDENTIALS i secrets.")

    raw = st.secrets["GOOGLE_CREDENTIALS"]

    if isinstance(raw, dict):
        cred = dict(raw)
    elif isinstance(raw, str):
        try:
            cred = json.loads(raw)
        except Exception:
            raise RuntimeError("GOOGLE_CREDENTIALS kunde inte tolkas (ej giltig JSON).")
    else:
        raise RuntimeError("GOOGLE_CREDENTIALS hade okänt format (varken dict eller JSON-sträng).")

    # Fixa escaped newlines i private_key
    pk = cred.get("private_key")
    if isinstance(pk, str) and "\\n" in pk and "\n" not in pk:
        cred["private_key"] = pk.replace("\\n", "\n")

    # Minimikrav: client_email + private_key
    if not cred.get("client_email") or not cred.get("private_key"):
        raise RuntimeError("GOOGLE_CREDENTIALS saknar client_email eller private_key.")
    return cred


def _client():
    if gspread is None or Credentials is None:
        raise RuntimeError("gspread/google-auth saknas i miljön.")
    cred_dict = _load_credentials_dict()
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(cred_dict, scopes=scope)
    return gspread.authorize(creds)


def _spreadsheet():
    if "SHEET_URL" not in st.secrets or not st.secrets["SHEET_URL"]:
        raise RuntimeError("SHEET_URL saknas i secrets.")
    cli = _client()
    return cli.open_by_url(st.secrets["SHEET_URL"])


def _with_backoff(func, *args, **kwargs):
    delays = [0, 0.6, 1.2, 2.5]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err


def _get_or_create_ws(title: str):
    ss = _spreadsheet()
    try:
        return _with_backoff(ss.worksheet, title)
    except Exception:
        # Skapa bladet med rimlig storlek
        _with_backoff(ss.add_worksheet, title=title, rows=1000, cols=50)
        return _with_backoff(ss.worksheet, title)


def list_worksheet_titles() -> List[str]:
    try:
        ss = _spreadsheet()
        sheets = _with_backoff(ss.worksheets)
        return [ws.title for ws in sheets]
    except Exception:
        return []


def ws_read_df(title: str) -> pd.DataFrame:
    """
    Läser alla värden. Om bladet saknar header → tom DF.
    """
    ws = _get_or_create_ws(title)
    rows = _with_backoff(ws.get_all_values)
    if not rows:
        return pd.DataFrame()
    # Ta första raden som header
    header = [h.strip() for h in rows[0]] if rows else []
    data = rows[1:] if len(rows) > 1 else []
    if not header:
        return pd.DataFrame()
    # Säkerställ att alla rader har samma längd
    w = len(header)
    norm = [r[:w] + [""]*(w - len(r)) for r in data]
    df = pd.DataFrame(norm, columns=header)
    return df


def ws_write_df(title: str, df: pd.DataFrame):
    ws = _get_or_create_ws(title)
    out = df.copy()
    # konvertera allt till str (Google Sheets API)
    values = [list(out.columns)]
    values += [[("" if pd.isna(v) else str(v)) for v in row] for row in out.itertuples(index=False)]
    _with_backoff(ws.clear)
    _with_backoff(ws.update, values)


# ============= Valutakurser-blad =============
RATES_SHEET = "Valutakurser"
DEFAULT_RATES = {"USD": 10.00, "NOK": 1.00, "CAD": 7.50, "EUR": 11.00, "SEK": 1.0}

def _get_or_create_rates_ws():
    return _get_or_create_ws(RATES_SHEET)

def read_rates() -> Dict[str, float]:
    try:
        ws = _get_or_create_rates_ws()
        rows = _with_backoff(ws.get_all_records)
        out: Dict[str, float] = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).strip().upper()
            val = str(r.get("Kurs", "")).replace(",", ".").strip()
            try:
                out[cur] = float(val)
            except Exception:
                pass
        # lägg in defaults om något saknas
        for k, v in DEFAULT_RATES.items():
            out.setdefault(k, v)
        return out
    except Exception:
        return dict(DEFAULT_RATES)

def save_rates(rates: Dict[str, float]):
    ws = _get_or_create_rates_ws()
    body = [["Valuta", "Kurs"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        body.append([k, str(rates.get(k, DEFAULT_RATES[k]))])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)


# ============= Datablad – schema & normalisering =============
DATA_COLS: List[str] = [
    "Ticker", "Bolagsnamn", "Sektor", "Valuta",
    "Antal aktier", "GAV (SEK)", "Aktuell kurs",
    "Utestående aktier",  # miljoner
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt (Q1..Q4)",
    "P/B", "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Årlig utdelning", "Payout (%)", "CAGR 5 år (%)",
    "Senast manuellt uppdaterad", "Senast auto uppdaterad", "Auto källa", "Senast beräknad",
    "DA (%)", "Uppsida idag (%)", "Uppsida 1 år (%)", "Uppsida 2 år (%)", "Uppsida 3 år (%)",
    "Score (Growth)", "Score (Dividend)", "Score (Financials)", "Score (Total)", "Confidence",
]

NUMERIC_COLS = {
    "Antal aktier","GAV (SEK)","Aktuell kurs","Utestående aktier",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
    "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "Årlig utdelning","Payout (%)","CAGR 5 år (%)",
    "DA (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
    "Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence",
}

TEXT_COLS = set(DATA_COLS) - NUMERIC_COLS

def _to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace(" ", "")
        s = s.replace("\u00a0", "")  # no-break-space
        s = s.replace(",", ".")
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return 0.0
        return float(s)
    except Exception:
        return 0.0

def ensure_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliserar ett godtyckligt ark till vårt schema. Vi läser *endast*
    kolumner vi känner till. Finns bara 'Ticker' → skapas resten.
    """
    out = pd.DataFrame(columns=DATA_COLS)
    # Om 'Ticker' finns: ta den; annars tom
    if "Ticker" in df_in.columns:
        out["Ticker"] = df_in["Ticker"].astype(str).str.upper().str.strip()
    # Kopiera kända numeriska kolumner om de råkar finnas
    for c in DATA_COLS:
        if c == "Ticker":
            continue
        if c in df_in.columns:
            if c in NUMERIC_COLS:
                out[c] = pd.to_numeric(df_in[c].apply(_to_float), errors="coerce").fillna(0.0)
            else:
                out[c] = df_in[c].astype(str)
        else:
            # default
            if c in NUMERIC_COLS:
                out[c] = 0.0
            else:
                out[c] = ""

    # Dedup tickers (behåll första)
    if "Ticker" in out.columns:
        out = out.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
    return out


# ============= IO med cache-busting =============
@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str, _nonce: int) -> pd.DataFrame:
    return ws_read_df(ws_title)

def load_df(ws_title: str) -> pd.DataFrame:
    n = st.session_state.get("_reload_nonce", 0)
    raw = load_df_cached(ws_title, n)
    # Normalisera alltid
    return ensure_columns(raw)

def save_df(ws_title: str, df: pd.DataFrame):
    ws_write_df(ws_title, df)
    st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1


# ============= Yahoo (snabb) =============
def yahoo_quick(ticker: str) -> Dict[str, float | str]:
    out = {
        "Bolagsnamn": "", "Valuta": "USD",
        "Aktuell kurs": 0.0, "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0, "Utestående aktier": 0.0,
        "P/S": 0.0, "P/B": 0.0,
    }
    if yf is None or not ticker:
        return out
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        price = info.get("regularMarketPrice")
        if price is None:
            h = t.history(period="5d")
            if not h.empty and "Close" in h.columns:
                price = float(h["Close"].iloc[-1])
        if price is not None:
            out["Aktuell kurs"] = float(price)

        # Valuta / namn
        if info.get("currency"): out["Valuta"] = str(info["currency"]).upper()
        nm = info.get("shortName") or info.get("longName")
        if nm: out["Bolagsnamn"] = str(nm)

        # Utdelning (årlig)
        if info.get("dividendRate") is not None:
            out["Årlig utdelning"] = float(info["dividendRate"] or 0.0)

        # Utestående aktier (absolut) → vi lagrar i miljoner senare
        if info.get("sharesOutstanding"):
            out["Utestående aktier"] = float(info["sharesOutstanding"]) / 1e6

        # Multiplar
        if info.get("priceToSalesTrailing12Months") is not None:
            out["P/S"] = float(info["priceToSalesTrailing12Months"] or 0.0)
        if info.get("priceToBook") is not None:
            out["P/B"] = float(info["priceToBook"] or 0.0)

        # CAGR 5 år (grov på "Total Revenue")
        rev_df = getattr(t, "financials", None)
        if isinstance(rev_df, pd.DataFrame) and not rev_df.empty and "Total Revenue" in rev_df.index:
            s = rev_df.loc["Total Revenue"].dropna().sort_index()
        else:
            inc = getattr(t, "income_stmt", None)
            if isinstance(inc, pd.DataFrame) and not inc.empty and "Total Revenue" in inc.index:
                s = inc.loc["Total Revenue"].dropna().sort_index()
            else:
                s = pd.Series(dtype=float)
        if len(s) >= 2:
            start, end = float(s.iloc[0]), float(s.iloc[-1])
            years = max(1, len(s) - 1)
            if start > 0:
                out["CAGR 5 år (%)"] = round(((end / start) ** (1.0 / years) - 1.0) * 100.0, 2)
    except Exception:
        pass
    return out


# ============= Beräkningar =============
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def recompute(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # P/S-snitt
    ps_cols = ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]
    for c in ps_cols:
        if c not in df.columns:
            df[c] = 0.0
    ps_vals = df[ps_cols].applymap(_to_float)
    ps_avg = ps_vals.replace(0.0, np.nan).mean(axis=1).fillna(0.0)
    df["P/S-snitt (Q1..Q4)"] = ps_avg.round(2)

    # CAGR clamp (2..50%)
    cagr = df["CAGR 5 år (%)"].apply(_to_float)
    cagr_clamped = cagr.clip(lower=2.0, upper=50.0) / 100.0

    # Omsättning år 2/3 från "Omsättning nästa år"
    oms_next = df["Omsättning nästa år"].apply(_to_float)
    df["Omsättning om 2 år"] = (oms_next * (1.0 + cagr_clamped)).round(2)
    df["Omsättning om 3 år"] = (oms_next * ((1.0 + cagr_clamped) ** 2)).round(2)

    # Riktkurser (kräver Utestående aktier (M) > 0 och P/S-snitt > 0)
    sh = df["Utestående aktier"].apply(_to_float)
    psn = df["P/S-snitt (Q1..Q4)"].apply(_to_float)

    def rk(oms):
        ok = (sh > 0) & (psn > 0)
        val = (oms * psn / sh).where(ok, 0.0)
        return val.round(2)

    df["Riktkurs idag"]    = rk(df["Omsättning idag"].apply(_to_float))
    df["Riktkurs om 1 år"] = rk(df["Omsättning nästa år"].apply(_to_float))
    df["Riktkurs om 2 år"] = rk(df["Omsättning om 2 år"].apply(_to_float))
    df["Riktkurs om 3 år"] = rk(df["Omsättning om 3 år"].apply(_to_float))

    # DA och uppsidor
    price = df["Aktuell kurs"].apply(_to_float).replace(0.0, np.nan)
    df["DA (%)"] = ((df["Årlig utdelning"].apply(_to_float) / price) * 100.0).fillna(0.0).round(2)
    for col, outc in [
        ("Riktkurs idag", "Uppsida idag (%)"),
        ("Riktkurs om 1 år", "Uppsida 1 år (%)"),
        ("Riktkurs om 2 år", "Uppsida 2 år (%)"),
        ("Riktkurs om 3 år", "Uppsida 3 år (%)"),
    ]:
        rkser = df[col].apply(_to_float).replace(0.0, np.nan)
        df[outc] = (((rkser - price) / price) * 100.0).fillna(0.0).round(2)

    df["Senast beräknad"] = now_stamp()
    return df


# ============= Sidopanel – Valutakurser =============
def sidebar_rates() -> Dict[str, float]:
    st.sidebar.subheader("💱 Valutakurser → SEK")

    # Ladda (en gång)
    if "rates_loaded" not in st.session_state:
        saved = read_rates()
        for k in ["USD","NOK","CAD","EUR"]:
            st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
        st.session_state["rates_loaded"] = True

    colA, colB = st.sidebar.columns(2)
    if colA.button("↻ Läs sparade"):
        saved = read_rates()
        for k in ["USD","NOK","CAD","EUR"]:
            st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
        st.sidebar.success("Sparade kurser inlästa.")
    if colB.button("💾 Spara"):
        try:
            to_save = {
                "USD": float(st.session_state.get("rate_usd", DEFAULT_RATES["USD"])),
                "NOK": float(st.session_state.get("rate_nok", DEFAULT_RATES["NOK"])),
                "CAD": float(st.session_state.get("rate_cad", DEFAULT_RATES["CAD"])),
                "EUR": float(st.session_state.get("rate_eur", DEFAULT_RATES["EUR"])),
                "SEK": 1.0,
            }
            save_rates(to_save)
            st.sidebar.success("Valutakurser sparade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.0001, format="%.6f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.0001, format="%.6f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.0001, format="%.6f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.0001, format="%.6f")

    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}


# ============= Vyer =============
def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Datablad")
    st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)
    if c1.button("💾 Spara hela bladet (beräkna först)"):
        df2 = recompute(df)
        try:
            save_df(ws_title, df2)
            st.success("Sparat.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")
    if c2.button("↻ Läs om från Google Sheets"):
        st.cache_data.clear()
        st.experimental_rerun()


def view_manual(df: pd.DataFrame, ws_title: str) -> pd.DataFrame:
    st.subheader("🧩 Manuell insamling")

    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = ["➕ Lägg till nytt bolag..."] + [
        (f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r.get("Bolagsnamn","")).strip() else r["Ticker"])
        for _, r in vis.iterrows()
    ]
    if "manual_idx" not in st.session_state:
        st.session_state["manual_idx"] = 0
    st.session_state["manual_idx"] = min(st.session_state["manual_idx"], len(labels)-1)

    sel = st.selectbox("Välj rad", list(range(len(labels))), format_func=lambda i: labels[i],
                       index=st.session_state["manual_idx"])
    st.session_state["manual_idx"] = sel
    is_new = (sel == 0)

    row = (pd.Series({c: 0.0 for c in df.columns if c in NUMERIC_COLS}) if is_new
           else vis.iloc[sel-1])

    st.markdown("### Obligatoriska fält (dina)")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker (Yahoo)", value=(row.get("Ticker","") if not is_new else "")).upper().strip()
        antal  = st.number_input("Antal aktier", value=float(row.get("Antal aktier",0.0) or 0.0), step=1.0, min_value=0.0)
        gav    = st.number_input("GAV (SEK)", value=float(row.get("GAV (SEK)",0.0) or 0.0), step=0.01, min_value=0.0, format="%.4f")
    with c2:
        oms_idag = st.number_input("Omsättning idag (M)", value=float(row.get("Omsättning idag",0.0) or 0.0), step=1.0, min_value=0.0)
        oms_next = st.number_input("Omsättning nästa år (M)", value=float(row.get("Omsättning nästa år",0.0) or 0.0), step=1.0, min_value=0.0)

    with st.expander("🌐 Auto-fält (kan justeras)"):
        cL, cR = st.columns(2)
        with cL:
            namn   = st.text_input("Bolagsnamn", value=str(row.get("Bolagsnamn","")))
            sektor = st.text_input("Sektor", value=str(row.get("Sektor","")))
            valuta = st.text_input("Valuta", value=str(row.get("Valuta","") or "USD")).upper()
            kurs   = st.number_input("Aktuell kurs", value=float(row.get("Aktuell kurs",0.0) or 0.0), step=0.01, min_value=0.0)
            utd    = st.number_input("Årlig utdelning", value=float(row.get("Årlig utdelning",0.0) or 0.0), step=0.01, min_value=0.0)
            payout = st.number_input("Payout (%)", value=float(row.get("Payout (%)",0.0) or 0.0), step=1.0, min_value=0.0)
        with cR:
            sh_m = st.number_input("Utestående aktier (miljoner)", value=float(row.get("Utestående aktier",0.0) or 0.0), step=0.01, min_value=0.0)
            ps   = st.number_input("P/S", value=float(row.get("P/S",0.0) or 0.0), step=0.01, min_value=0.0)
            ps1  = st.number_input("P/S Q1", value=float(row.get("P/S Q1",0.0) or 0.0), step=0.01, min_value=0.0)
            ps2  = st.number_input("P/S Q2", value=float(row.get("P/S Q2",0.0) or 0.0), step=0.01, min_value=0.0)
            ps3  = st.number_input("P/S Q3", value=float(row.get("P/S Q3",0.0) or 0.0), step=0.01, min_value=0.0)
            ps4  = st.number_input("P/S Q4", value=float(row.get("P/S Q4",0.0) or 0.0), step=0.01, min_value=0.0)
            pb   = st.number_input("P/B", value=float(row.get("P/B",0.0) or 0.0), step=0.01, min_value=0.0)
            pb1  = st.number_input("P/B Q1", value=float(row.get("P/B Q1",0.0) or 0.0), step=0.01, min_value=0.0)
            pb2  = st.number_input("P/B Q2", value=float(row.get("P/B Q2",0.0) or 0.0), step=0.01, min_value=0.0)
            pb3  = st.number_input("P/B Q3", value=float(row.get("P/B Q3",0.0) or 0.0), step=0.01, min_value=0.0)
            pb4  = st.number_input("P/B Q4", value=float(row.get("P/B Q4",0.0) or 0.0), step=0.01, min_value=0.0)
            cagr = st.number_input("CAGR 5 år (%)", value=float(row.get("CAGR 5 år (%)",0.0) or 0.0), step=0.1)

    cA, cB, cC = st.columns(3)
    do_save  = cA.button("💾 Spara")
    do_quick = cB.button("⚡ Snabb Yahoo (valda)")
    do_all   = cC.button("🔭 Full Yahoo för alla (0.5s delay)")

    df2 = df.copy()
    if do_save:
        if not ticker:
            st.error("Ticker krävs."); return df
        mask = (df2["Ticker"] == ticker)
        payload = {
            "Ticker": ticker, "Antal aktier": float(antal), "GAV (SEK)": float(gav),
            "Omsättning idag": float(oms_idag), "Omsättning nästa år": float(oms_next),
            "Bolagsnamn": namn, "Sektor": sektor, "Valuta": valuta,
            "Aktuell kurs": float(kurs), "Årlig utdelning": float(utd), "Payout (%)": float(payout),
            "Utestående aktier": float(sh_m),
            "P/S": float(ps), "P/S Q1": float(ps1), "P/S Q2": float(ps2), "P/S Q3": float(ps3), "P/S Q4": float(ps4),
            "P/B": float(pb), "P/B Q1": float(pb1), "P/B Q2": float(pb2), "P/B Q3": float(pb3), "P/B Q4": float(pb4),
            "CAGR 5 år (%)": float(cagr),
            "Senast manuellt uppdaterad": now_stamp(),
        }
        if mask.any():
            for k, v in payload.items():
                df2.loc[mask, k] = v
        else:
            base = {c: (0.0 if c in NUMERIC_COLS else "") for c in DATA_COLS}
            base.update(payload)
            df2 = pd.concat([df2, pd.DataFrame([base])], ignore_index=True)
        df2 = recompute(df2)
        try:
            save_df(ws_title, df2)
            st.success("Sparat.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    if do_quick:
        if not ticker:
            st.error("Ange ticker först.")
        else:
            y = yahoo_quick(ticker)
            mask = (df2["Ticker"] == ticker)
            if not mask.any():
                st.error("Tickern finns inte i tabellen – spara först.")
            else:
                for src, dst in [
                    ("Bolagsnamn","Bolagsnamn"), ("Valuta","Valuta"),
                    ("Aktuell kurs","Aktuell kurs"), ("Årlig utdelning","Årlig utdelning"),
                    ("CAGR 5 år (%)","CAGR 5 år (%)"), ("Utestående aktier","Utestående aktier"),
                    ("P/S","P/S"), ("P/B","P/B"),
                ]:
                    v = y.get(src)
                    if v is not None:
                        df2.loc[mask, dst] = v
                df2.loc[mask, "Senast auto uppdaterad"] = now_stamp()
                df2.loc[mask, "Auto källa"] = "Yahoo (snabb)"
                df2 = recompute(df2)
                try:
                    save_df(ws_title, df2)
                    st.success("Snabbdata uppdaterad.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Kunde inte spara: {e}")

    if do_all:
        if yf is None:
            st.error("yfinance saknas i miljön.")
        else:
            total = max(1, len(df2))
            status = st.empty(); bar = st.progress(0.0)
            for i, r in df2.iterrows():
                tkr = str(r["Ticker"]).strip().upper()
                if not tkr:
                    bar.progress((i+1)/total); continue
                status.write(f"Hämtar {i+1}/{len(df2)} – {tkr}")
                y = yahoo_quick(tkr)
                for src, dst in [
                    ("Bolagsnamn","Bolagsnamn"), ("Valuta","Valuta"),
                    ("Aktuell kurs","Aktuell kurs"), ("Årlig utdelning","Årlig utdelning"),
                    ("CAGR 5 år (%)","CAGR 5 år (%)"), ("Utestående aktier","Utestående aktier"),
                    ("P/S","P/S"), ("P/B","P/B"),
                ]:
                    v = y.get(src)
                    if v is not None:
                        df2.at[i, dst] = v
                df2.at[i, "Senast auto uppdaterad"] = now_stamp()
                df2.at[i, "Auto källa"] = "Yahoo (snabb)"
                time.sleep(0.5)
                bar.progress((i+1)/total)
            df2 = recompute(df2)
            try:
                save_df(ws_title, df2)
                st.success("Klart.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Kunde inte spara: {e}")

    return df2


def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier."); return

    # Växelkurs → SEK
    def vx(v):
        cur = str(v).upper().strip()
        return float(rates.get(cur, rates.get("SEK", 1.0)))

    port["Växelkurs"]    = port["Valuta"].apply(vx)
    port["Värde (SEK)"]  = port["Antal aktier"].apply(_to_float) * port["Aktuell kurs"].apply(_to_float) * port["Växelkurs"]
    port["Köpvärde (SEK)"] = port["Antal aktier"].apply(_to_float) * port["GAV (SEK)"].apply(_to_float)
    port["Vinst (SEK)"]  = port["Värde (SEK)"] - port["Köpvärde (SEK)"]
    port["Vinst (%)"]    = np.where(port["Köpvärde (SEK)"]>0, (port["Vinst (SEK)"]/port["Köpvärde (SEK)"])*100.0, 0.0)

    tot_val = float(port["Värde (SEK)"].sum())
    tot_cost= float(port["Köpvärde (SEK)"].sum())
    tot_pnl = tot_val - tot_cost
    tot_pnl_pct = (tot_pnl / tot_cost * 100.0) if tot_cost > 0 else 0.0

    st.markdown(f"**Portföljvärde:** {round(tot_val,2)} SEK")
    st.markdown(f"**Anskaffningsvärde:** {round(tot_cost,2)} SEK")
    st.markdown(f"**Vinst:** {round(tot_pnl,2)} SEK ({round(tot_pnl_pct,2)} %)")

    port["Andel (%)"] = np.where(tot_val>0, (port["Värde (SEK)"]/tot_val)*100.0, 0.0).round(2)
    port["DA (%)"]    = np.where(port["Aktuell kurs"]>0, (port["Årlig utdelning"]/port["Aktuell kurs"])*100.0, 0.0).round(2)

    cols = ["Ticker","Bolagsnamn","Antal aktier","GAV (SEK)","Aktuell kurs","Valuta","Växelkurs",
            "Värde (SEK)","Köpvärde (SEK)","Vinst (SEK)","Vinst (%)","Årlig utdelning","DA (%)","Andel (%)"]
    st.dataframe(port[cols].sort_values("Andel (%)", ascending=False), use_container_width=True)


def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag")

    horizon = st.selectbox("Riktkurs-horisont", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    subset = st.radio("Visa", ["Alla bolag","Endast portfölj"], horizontal=True)

    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()

    base = base[(base[horizon] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inget att visa."); return

    base["Potential (%)"] = ((base[horizon] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0).round(2)
    sort_on = st.selectbox("Sortera på", ["Potential (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)"], index=0)
    trim_mode = st.checkbox("Trim/sälj-läge (minst uppsida först)", value=False)

    base = base.sort_values(sort_on, ascending=trim_mode).reset_index(drop=True)
    st.dataframe(base[["Ticker","Bolagsnamn","Aktuell kurs",horizon,"Potential (%)","DA (%)","P/S-snitt (Q1..Q4)","Utestående aktier"]], use_container_width=True)

    # Kortvisning
    st.markdown("---")
    if "idea_idx" not in st.session_state: st.session_state["idea_idx"] = 0
    st.session_state["idea_idx"] = st.number_input("Visa rad #", min_value=0, max_value=max(0, len(base)-1),
                                                   value=st.session_state["idea_idx"], step=1)
    r = base.iloc[st.session_state["idea_idx"]]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- **Aktuell kurs:** {round(float(r['Aktuell kurs']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs idag / 1 / 2 / 3 år:** {r['Riktkurs idag']} / {r['Riktkurs om 1 år']} / {r['Riktkurs om 2 år']} / {r['Riktkurs om 3 år']}")
        st.write(f"- **Uppsida (valda):** {r['Potential (%)']} %")
    with c2:
        st.write(f"- **P/S-snitt (Q1..Q4):** {r['P/S-snitt (Q1..Q4)']}")
        st.write(f"- **Omsättning (M):** idag {r['Omsättning idag']}, nästa {r['Omsättning nästa år']}, 2y {r['Omsättning om 2 år']}, 3y {r['Omsättning om 3 år']}")
        st.write(f"- **Årlig utdelning / DA:** {r['Årlig utdelning']} / {r['DA (%)']} %")


# ============= main =============
def main():
    st.title("📊 K-pf-rslag – stabil")

    # Välj datablad
    titles = list_worksheet_titles() or ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → blad", titles, index=0)

    # Snabb cachebust-läsning
    if st.sidebar.button("↻ Läs om från Google Sheets"):
        st.cache_data.clear()
        st.experimental_rerun()

    # Valutakurser
    rates = sidebar_rates()

    # Läs & normalisera
    df = load_df(ws_title)

    # Tabbar
    tabs = st.tabs(["📄 Data", "🧩 Manuell", "📦 Portfölj", "💡 Förslag"])
    with tabs[0]:
        view_data(df, ws_title)
    with tabs[1]:
        df2 = view_manual(df, ws_title)   # sparar själv
    with tabs[2]:
        # Läs om efter eventuell sparning
        dfp = load_df(ws_title)
        view_portfolio(dfp, rates)
    with tabs[3]:
        dfi = load_df(ws_title)
        dfi = recompute(dfi)
        view_ideas(dfi)


if __name__ == "__main__":
    main()
