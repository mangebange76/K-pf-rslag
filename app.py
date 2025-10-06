# app.py (Del 1/5)
from __future__ import annotations

import re, time, json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Egna moduler (ligger i stockapp/)
from stockapp.sheets import (
    ws_read_df, ws_write_df, list_worksheet_titles, delete_worksheet,
    with_backoff, get_spreadsheet
)
from stockapp.rates import (
    read_rates, save_rates, fetch_live_rates, repair_rates_sheet, DEFAULT_RATES
)
from stockapp.dividends import build_dividend_calendar

# (Valfri) fetchers för full uppdatering
try:
    from stockapp.fetchers.yahoo import get_all as y_overview
except Exception:
    y_overview = None
try:
    from stockapp.fetchers.finviz import get_overview as fz_overview
except Exception:
    fz_overview = None
try:
    from stockapp.fetchers.morningstar import get_overview as ms_overview
except Exception:
    ms_overview = None
try:
    from stockapp.fetchers.sec import get_pb_quarters as sec_pb_quarters
except Exception:
    sec_pb_quarters = None

st.set_page_config(page_title="K-pf-rslag", layout="wide")

# ---------------- Tidsstämplar ----------------
def _now_sthlm() -> datetime:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz)
    except Exception:
        return datetime.now()

def now_stamp() -> str:
    return _now_sthlm().strftime("%Y-%m-%d")

# ---------------- Snapshot ----------------
SNAP_PREFIX = "SNAP__"
def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

def _parse_snap_title(title: str) -> Optional[datetime]:
    m = re.search(r"(\d{8}_\d{6})$", str(title))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    except Exception:
        return None

def safe_snapshot(df: pd.DataFrame, base_ws_title: str, max_cells: int = 800_000):
    """Skapa snapshot om möjligt; rensa äldre än 5 dagar."""
    if st.session_state.get("_snapshot_done"):
        return
    st.session_state["_snapshot_done"] = True
    try:
        rows, cols = df.shape
        cells = (rows + 1) * (cols + 1)
        if cells > max_cells:
            st.sidebar.warning(f"Snapshot hoppades över (cells {cells:,} > {max_cells:,}).")
            return
        now = _now_sthlm()
        snap_title = f"{SNAP_PREFIX}{base_ws_title}__{_format_ts(now)}"
        ws_write_df(snap_title, df)
        st.sidebar.success(f"Snapshot sparat: {snap_title}")

        # Rensa >5 dagar
        cutoff = now - timedelta(days=5)
        for t in list_worksheet_titles() or []:
            if not str(t).startswith(SNAP_PREFIX):
                continue
            ts = _parse_snap_title(t)
            if ts and ts < cutoff.replace(tzinfo=None):
                try:
                    delete_worksheet(t)
                except Exception:
                    pass
    except Exception as e:
        st.sidebar.warning(f"Kunde inte spara snapshot: {e}")

# ---------------- Numerik ----------------
def _clean_number(x) -> float:
    if x is None:
        return 0.0
    try:
        s = str(x).strip().replace("\u00a0", "").replace(" ", "")
        s = s.replace(",", ".")
        if s in ("", "-", "nan", "None"):
            return 0.0
        return float(s)
    except Exception:
        try:
            return float(x)
        except Exception:
            return 0.0

# ---------------- Kolumnschema ----------------
FINAL_COLS: List[str] = [
    # Bas
    "Ticker", "Bolagsnamn", "Sektor", "Valuta",
    "Antal aktier", "GAV (SEK)", "Aktuell kurs",
    "Utestående aktier",

    # Multiplar
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt (Q1..Q4)",
    "P/B", "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",

    # Omsättning (M)
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",

    # Riktkurser (bolagets valuta)
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Utdelning
    "Årlig utdelning", "Payout (%)",

    # Övrigt
    "CAGR 5 år (%)",

    # Tidsstämplar
    "Senast manuellt uppdaterad",
    "Senast auto uppdaterad",
    "Auto källa",
    "Senast beräknad",

    # Visning/score
    "DA (%)", "Uppsida idag (%)", "Uppsida 1 år (%)", "Uppsida 2 år (%)", "Uppsida 3 år (%)",
    "Score (Growth)", "Score (Dividend)", "Score (Financials)", "Score (Total)", "Confidence",

    # Sparade score per horisont
    "Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)",
    "Score Growth (Idag)", "Score Dividend (Idag)", "Score Financials (Idag)",
    "Score Growth (1 år)", "Score Dividend (1 år)", "Score Financials (1 år)",
    "Score Growth (2 år)", "Score Dividend (2 år)", "Score Financials (2 år)",
    "Score Growth (3 år)", "Score Dividend (3 år)", "Score Financials (3 år)",

    # Utdelningsschema
    "Div_Frekvens/år", "Div_Månader", "Div_Vikter",
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Skapa saknade kolumner
    for c in FINAL_COLS:
        if c not in out.columns:
            if any(k in c.lower() for k in [
                "kurs","omsättning","p/s","p/b","utdelning","cagr","aktier",
                "riktkurs","payout","score","uppsida","da","confidence",
                "frekvens","gav","utestående"
            ]):
                out[c] = 0.0
            else:
                out[c] = ""
    # Numeriska typer
    num_cols = [
        "Antal aktier","GAV (SEK)","Aktuell kurs","Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
        "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Årlig utdelning","Payout (%)","CAGR 5 år (%)",
        "DA (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
        "Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence",
        "Score Total (Idag)","Score Total (1 år)","Score Total (2 år)","Score Total (3 år)",
        "Score Growth (Idag)","Score Dividend (Idag)","Score Financials (Idag)",
        "Score Growth (1 år)","Score Dividend (1 år)","Score Financials (1 år)",
        "Score Growth (2 år)","Score Dividend (2 år)","Score Financials (2 år)",
        "Score Growth (3 år)","Score Dividend (3 år)","Score Financials (3 år)",
        "Div_Frekvens/år",
    ]
    for c in num_cols:
        out[c] = out[c].map(_clean_number)

    for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Auto källa",
              "Senast manuellt uppdaterad","Senast auto uppdaterad","Senast beräknad",
              "Div_Månader","Div_Vikter"]:
        out[c] = out[c].astype(str)

    # Trimma helt tomma rader (inga värden + ingen ticker)
    if "Ticker" in out.columns:
        mask_all_empty = out[["Ticker"]].fillna("").eq("").all(axis=1) & (out.select_dtypes(include=[float,int]).fillna(0).sum(axis=1) == 0)
        out = out[~mask_all_empty].copy()

    return out

# app.py (Del 2/5)

# --------- Beräkningar & score ----------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_ps_pb_snitt(row: pd.Series) -> Tuple[float, float]:
    ps_vals = [row.get("P/S Q1", 0), row.get("P/S Q2", 0), row.get("P/S Q3", 0), row.get("P/S Q4", 0)]
    ps_clean = [float(x) for x in ps_vals if float(x) > 0]
    ps_avg = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0

    pb_vals = [row.get("P/B Q1", 0), row.get("P/B Q2", 0), row.get("P/B Q3", 0), row.get("P/B Q4", 0)]
    pb_clean = [float(x) for x in pb_vals if float(x) > 0]
    pb_avg = round(float(np.mean(pb_clean)), 2) if pb_clean else 0.0
    return ps_avg, pb_avg

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i, r in out.iterrows():
        ps_avg, pb_avg = compute_ps_pb_snitt(r)
        out.at[i, "P/S-snitt (Q1..Q4)"] = ps_avg
        out.at[i, "P/B-snitt (Q1..Q4)"] = pb_avg

        cagr = float(r.get("CAGR 5 år (%)", 0.0))
        g = clamp(cagr, 2.0, 50.0) / 100.0

        next_rev = float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            out.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            out.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)
        else:
            out.at[i, "Omsättning om 2 år"] = float(r.get("Omsättning om 2 år", 0.0))
            out.at[i, "Omsättning om 3 år"] = float(r.get("Omsättning om 3 år", 0.0))

        shares_m = float(r.get("Utestående aktier", 0.0))
        if shares_m <= 0 or ps_avg <= 0:
            out.at[i, "Riktkurs idag"] = out.at[i, "Riktkurs om 1 år"] = out.at[i, "Riktkurs om 2 år"] = out.at[i, "Riktkurs om 3 år"] = 0.0
            continue

        out.at[i, "Riktkurs idag"]    = round(float(r.get("Omsättning idag", 0.0))     * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 1 år"] = round(float(r.get("Omsättning nästa år", 0.0)) * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 2 år"] = round(float(out.at[i, "Omsättning om 2 år"])   * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 3 år"] = round(float(out.at[i, "Omsättning om 3 år"])   * ps_avg / shares_m, 2)
    return out

def add_multi_uppsida(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = out["Aktuell kurs"].replace(0, np.nan)
    for col, label in [
        ("Riktkurs idag", "Uppsida idag (%)"),
        ("Riktkurs om 1 år", "Uppsida 1 år (%)"),
        ("Riktkurs om 2 år", "Uppsida 2 år (%)"),
        ("Riktkurs om 3 år", "Uppsida 3 år (%)"),
    ]:
        rk = out[col].replace(0, np.nan)
        out[label] = ((rk - price) / price * 100.0).fillna(0.0)
    out["DA (%)"] = np.where(out["Aktuell kurs"]>0, (out["Årlig utdelning"]/out["Aktuell kurs"])*100.0, 0.0)
    return out

def _horizon_to_tag(h: str) -> str:
    if "om 1 år" in h: return "1 år"
    if "om 2 år" in h: return "2 år"
    if "om 3 år" in h: return "3 år"
    return "Idag"

def score_rows(df: pd.DataFrame, horizon: str, strategy: str) -> pd.DataFrame:
    out = df.copy()
    out["DA (%)"] = np.where(out["Aktuell kurs"] > 0, (out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0, 0.0)
    out["Uppsida (%)"] = np.where(out["Aktuell kurs"] > 0, (out[horizon] - out["Aktuell kurs"]) / out["Aktuell kurs"] * 100.0, 0.0)

    cur_ps = out["P/S"].replace(0, np.nan)
    ps_avg = out["P/S-snitt (Q1..Q4)"].replace(0, np.nan)
    cheap_ps = (ps_avg / (cur_ps * 2.0)).clip(upper=1.0).fillna(0.0)
    g_norm = (out["CAGR 5 år (%)"] / 30.0).clip(0, 1)
    u_norm = (out["Uppsida (%)"] / 50.0).clip(0, 1)
    out["Score (Growth)"] = (0.4 * g_norm + 0.4 * u_norm + 0.2 * cheap_ps) * 100.0

    payout = out["Payout (%)"]
    payout_health = 1 - (abs(payout - 60.0) / 60.0)
    payout_health = payout_health.clip(0, 1)
    payout_health = np.where(out["Payout (%)"] <= 0, 0.85, payout_health)
    y_norm = (out["DA (%)"] / 8.0).clip(0, 1)
    grow_ok = np.where(out["CAGR 5 år (%)"] >= 0, 1.0, 0.6)
    out["Score (Dividend)"] = (0.6 * y_norm + 0.3 * payout_health + 0.1 * grow_ok) * 100.0

    cur_pb = out["P/B"].replace(0, np.nan)
    pb_avg = out["P/B-snitt (Q1..Q4)"].replace(0, np.nan)
    cheap_pb = (pb_avg / (cur_pb * 2.0)).clip(upper=1.0).fillna(0.0)
    out["Score (Financials)"] = (0.7 * cheap_pb + 0.3 * u_norm) * 100.0

    def weights_for_row(sektor: str, strategy: str) -> Tuple[float, float, float]:
        if strategy == "Tillväxt":   return (0.70, 0.10, 0.20)
        if strategy == "Utdelning":  return (0.15, 0.70, 0.15)
        if strategy == "Finans":     return (0.20, 0.20, 0.60)
        s = (sektor or "").lower()
        if any(k in s for k in ["bank","finans","insurance","financial"]): return (0.25, 0.25, 0.50)
        if any(k in s for k in ["utility","utilities","consumer staples","telecom"]): return (0.20, 0.60, 0.20)
        if any(k in s for k in ["tech","information technology","semiconductor","software"]): return (0.70, 0.10, 0.20)
        return (0.45, 0.35, 0.20)

    Wg, Wd, Wf = [], [], []
    for _, r in out.iterrows():
        wg, wd, wf = weights_for_row(r.get("Sektor",""), strategy)
        Wg.append(wg); Wd.append(wd); Wf.append(wf)
    Wg = np.array(Wg); Wd = np.array(Wd); Wf = np.array(Wf)
    out["Score (Total)"] = (Wg*out["Score (Growth)"] + Wd*out["Score (Dividend)"] + Wf*out["Score (Financials)"]).round(2)

    need = [
        out["Aktuell kurs"] > 0,
        out["P/S-snitt (Q1..Q4)"] > 0,
        out["Omsättning idag"] >= 0,
        out["Omsättning nästa år"] >= 0,
    ]
    present = np.stack(need, axis=0).astype(float)
    out["Confidence"] = (present.mean(axis=0) * 100.0).round(0)
    return out

def compute_scores_all_horizons(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    mapping = [("Riktkurs idag","Idag"), ("Riktkurs om 1 år","1 år"), ("Riktkurs om 2 år","2 år"), ("Riktkurs om 3 år","3 år")]
    strat = "Auto" if str(strategy).startswith("Auto") else strategy
    for horizon, tag in mapping:
        tmp = score_rows(out, horizon=horizon, strategy=strat)
        out[f"Score Growth ({tag})"]     = tmp["Score (Growth)"].round(2)
        out[f"Score Dividend ({tag})"]   = tmp["Score (Dividend)"].round(2)
        out[f"Score Financials ({tag})"] = tmp["Score (Financials)"].round(2)
        out[f"Score Total ({tag})"]      = tmp["Score (Total)"].round(2)
    return out

def enrich_for_save(df: pd.DataFrame, horizon_for_score: str = "Riktkurs idag", strategy: str = "Auto") -> pd.DataFrame:
    df2 = update_calculations(df)
    df2 = add_multi_uppsida(df2)
    df2 = score_rows(df2, horizon=horizon_for_score, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    df2 = compute_scores_all_horizons(df2, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    df2["Senast beräknad"] = now_stamp()
    return df2

# app.py (Del 3/5)

# ---------- Snabb Yahoo (pris mm) ----------
@st.cache_data(show_spinner=False, ttl=600)
def yahoo_fetch_one_quick(ticker: str) -> Dict[str, float | str]:
    out = {"Bolagsnamn":"", "Valuta":"USD", "Aktuell kurs":0.0, "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try: info = t.info or {}
        except Exception: info = {}
        nm = info.get("shortName") or info.get("longName");  out["Bolagsnamn"] = str(nm or "")
        cur = info.get("currency");                          out["Valuta"]     = str(cur or "USD").upper()
        px  = info.get("regularMarketPrice")
        if px is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h: px = float(h["Close"].iloc[-1])
        if px is not None: out["Aktuell kurs"] = float(px)
        dr = info.get("dividendRate");  out["Årlig utdelning"] = float(dr or 0.0)

        # CAGR 5y från revenue
        try:
            df_is = getattr(t, "income_stmt", None)
            ser = None
            if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
                ser = df_is.loc["Total Revenue"].dropna()
            else:
                df_fin = getattr(t, "financials", None)
                if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                    ser = df_fin.loc["Total Revenue"].dropna()
            if ser is not None and len(ser) >= 2:
                ser = ser.sort_index()
                start, end = float(ser.iloc[0]), float(ser.iloc[-1])
                years = max(1, len(ser)-1)
                if start > 0:
                    out["CAGR 5 år (%)"] = round(((end/start)**(1.0/years) - 1.0) * 100.0, 2)
        except Exception:
            pass
    except Exception:
        pass
    return out


# ---------- Etiketttext bredvid fält ----------
def _m_tag(df_row: pd.Series) -> str:
    d = str(df_row.get("Senast manuellt uppdaterad","")).strip()
    return f"〔M: {d or '—'}〕"

def _a_tag(df_row: pd.Series) -> str:
    d = str(df_row.get("Senast auto uppdaterad","")).strip()
    src = str(df_row.get("Auto källa","")).strip()
    return f"〔A: {d or '—'}{(' · '+src) if src else ''}〕"

def _b_tag(df_row: pd.Series) -> str:
    d = str(df_row.get("Senast beräknad","")).strip()
    return f"〔B: {d or '—'}〕"


# ---------- IO (med cache-nonce) ----------
@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str, _nonce: int) -> pd.DataFrame:
    return ws_read_df(ws_title)

def load_df(ws_title: str) -> pd.DataFrame:
    n = st.session_state.get("_reload_nonce", 0)
    df = load_df_cached(ws_title, n)
    return ensure_columns(df)

def save_df(ws_title: str, df: pd.DataFrame, bust_cache: bool = True):
    ws_write_df(ws_title, df)
    if bust_cache:
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1


# ---------- Sidopanel: valutakurser ----------
def sidebar_rates() -> Dict[str, float]:
    st.sidebar.subheader("💱 Valutakurser → SEK")

    if "rates_loaded" not in st.session_state:
        saved = read_rates()
        st.session_state["rate_usd"] = float(saved.get("USD", DEFAULT_RATES["USD"]))
        st.session_state["rate_nok"] = float(saved.get("NOK", DEFAULT_RATES["NOK"]))
        st.session_state["rate_cad"] = float(saved.get("CAD", DEFAULT_RATES["CAD"]))
        st.session_state["rate_eur"] = float(saved.get("EUR", DEFAULT_RATES["EUR"]))
        st.session_state["rates_loaded"] = True

    colA, colB = st.sidebar.columns(2)
    if colA.button("🌐 Hämta livekurser"):
        try:
            live = fetch_live_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(live[k])
            st.sidebar.success("Livekurser hämtade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta livekurser: {e}")

    if colB.button("↻ Läs sparade kurser"):
        try:
            saved = read_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
            st.sidebar.success("Sparade kurser inlästa.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte läsa sparade kurser: {e}")

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.000001, format="%.6f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.000001, format="%.6f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.000001, format="%.6f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.000001, format="%.6f")

    colC, colD = st.sidebar.columns(2)
    if colC.button("💾 Spara valutakurser"):
        try:
            save_rates({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.sidebar.success("Valutakurser sparade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")

    if colD.button("🛠 Reparera bladet"):
        try:
            repair_rates_sheet()
            st.sidebar.success("Bladet reparerat.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte reparera: {e}")

    return {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

# app.py (Del 4/5)

# ---------- Full uppdatering (alla fetchers) ----------
def _update_one_all_fetchers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df

    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        return df

    # Yahoo bred
    if y_overview:
        try:
            y = y_overview(tkr) or {}
            mapping = {
                "name": "Bolagsnamn",
                "currency": "Valuta",
                "price": "Aktuell kurs",
                "dividend_rate": "Årlig utdelning",
                "ps_ttm": "P/S",
                "pb": "P/B",
                "shares_outstanding": "Utestående aktier",  # absolut → M
                "cagr5_pct": "CAGR 5 år (%)",
            }
            for k_src, k_dst in mapping.items():
                v = y.get(k_src)
                if v is None: continue
                if k_dst == "Utestående aktier":
                    df.loc[mask, k_dst] = float(v) / 1e6
                else:
                    df.loc[mask, k_dst] = float(v) if isinstance(v, (int,float)) else str(v)
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Yahoo"
        except Exception:
            pass

    # Finviz
    if fz_overview:
        try:
            fz = fz_overview(tkr) or {}
            if float(fz.get("ps_ttm", 0.0)) > 0: df.loc[mask, "P/S"] = float(fz["ps_ttm"])
            if float(fz.get("pb", 0.0))     > 0: df.loc[mask, "P/B"] = float(fz["pb"])
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Finviz"
        except Exception:
            pass

    # Morningstar
    if ms_overview:
        try:
            ms = ms_overview(tkr) or {}
            if float(ms.get("ps_ttm", 0.0)) > 0: df.loc[mask, "P/S"] = float(ms["ps_ttm"])
            if float(ms.get("pb", 0.0))     > 0: df.loc[mask, "P/B"] = float(ms["pb"])
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Morningstar"
        except Exception:
            pass

    # SEC → P/B kvartal
    if sec_pb_quarters:
        try:
            sec = sec_pb_quarters(tkr) or {}
            pairs = sec.get("pb_quarters") or []  # list[(date, pb)]
            if pairs:
                pairs = pairs[:4]  # Q1..Q4
                for idx, (_, pbv) in enumerate(pairs, start=1):
                    if idx > 4: break
                    df.loc[mask, f"P/B Q{idx}"] = float(pbv or 0.0)
                row = df.loc[mask].iloc[0]
                _ps_avg, pb_avg = compute_ps_pb_snitt(row)
                df.loc[mask, "P/B-snitt (Q1..Q4)"] = pb_avg
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "SEC"
        except Exception:
            pass

    return df


# ---------- Manuell insamling ----------
def view_manual(df: pd.DataFrame, ws_title: str) -> pd.DataFrame:
    st.subheader("🧩 Manuell insamling")

    # Navigering & val
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r.get("Bolagsnamn","")).strip() else str(r["Ticker"]) for _, r in vis.iterrows()]
    labels = ["➕ Lägg till nytt bolag..."] + labels

    if "manual_idx" not in st.session_state:
        st.session_state["manual_idx"] = 0

    sel = st.selectbox("Välj bolag att redigera", list(range(len(labels))), format_func=lambda i: labels[i], index=st.session_state["manual_idx"])
    st.session_state["manual_idx"] = sel

    c_prev, c_next = st.columns([1,1])
    with c_prev:
        if st.button("⬅️ Föregående", use_container_width=True, disabled=(sel<=0)):
            st.session_state["manual_idx"] = max(0, sel-1); st.rerun()
    with c_next:
        if st.button("➡️ Nästa", use_container_width=True, disabled=(sel>=len(labels)-1)):
            st.session_state["manual_idx"] = min(len(labels)-1, sel+1); st.rerun()

    is_new = (sel == 0)
    if not is_new and len(vis) > 0:
        row = vis.iloc[sel-1]
    else:
        # Säker init
        row = pd.Series({c: (0.0 if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else "") for c in df.columns})

    # Sidokolumn: snabb + full uppdatering
    st.markdown("#### Uppdatera vald ticker")
    c_up1, c_up2 = st.columns(2)
    with c_up1:
        if not is_new and st.button("⚡ Snabb (Yahoo pris + utd + valuta + CAGR)"):
            try:
                quick = yahoo_fetch_one_quick(str(row["Ticker"]).upper())
                m = (df["Ticker"].astype(str).str.upper() == str(row["Ticker"]).upper())
                if quick.get("Bolagsnamn"): df.loc[m, "Bolagsnamn"] = str(quick["Bolagsnamn"])
                if quick.get("Valuta"):     df.loc[m, "Valuta"]     = str(quick["Valuta"])
                if float(quick.get("Aktuell kurs",0.0))>0: df.loc[m, "Aktuell kurs"] = float(quick["Aktuell kurs"])
                df.loc[m, "Årlig utdelning"] = float(quick.get("Årlig utdelning",0.0))
                df.loc[m, "CAGR 5 år (%)"]   = float(quick.get("CAGR 5 år (%)",0.0))
                df.loc[m, "Senast auto uppdaterad"] = now_stamp()
                df.loc[m, "Auto källa"] = "Yahoo (snabb)"
                df = enrich_for_save(df)
                save_df(ws_title, df, bust_cache=True)
                st.success("Snabb uppdatering OK."); st.rerun()
            except Exception as e:
                st.error(f"Kunde inte göra snabb uppdatering: {e}")
    with c_up2:
        if not is_new and st.button("🔭 Full (alla fetchers)"):
            df2 = _update_one_all_fetchers(df, str(row["Ticker"]).upper())
            df2 = enrich_for_save(df2)
            try:
                save_df(ws_title, df2, bust_cache=True)
                st.success("Full uppdatering OK."); st.rerun()
            except Exception as e:
                st.error(f"Kunde inte spara: {e}")

    # Obligatoriska fält (dina manuella)
    mtag = _m_tag(row)
    st.markdown("### Obligatoriska fält (manuella)")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input(f"Ticker (Yahoo-format) {mtag}", value=str(row.get("Ticker","")).upper() if not is_new else "", placeholder="t.ex. AAPL")
        antal  = st.number_input(f"Antal aktier (du äger) {mtag}", value=float(_clean_number(row.get("Antal aktier",0.0)) if not is_new else 0.0), step=1.0, min_value=0.0)
        gav    = st.number_input(f"GAV (SEK) {mtag}", value=float(_clean_number(row.get("GAV (SEK)",0.0)) if not is_new else 0.0), step=0.01, min_value=0.0, format="%.2f")
    with c2:
        oms_idag = st.number_input(f"Omsättning idag (M) {mtag}", value=float(_clean_number(row.get("Omsättning idag",0.0)) if not is_new else 0.0), step=1.0, min_value=0.0)
        oms_nxt  = st.number_input(f"Omsättning nästa år (M) {mtag}", value=float(_clean_number(row.get("Omsättning nästa år",0.0)) if not is_new else 0.0), step=1.0, min_value=0.0)

    # Auto-fält (kan korrigeras)
    atag = _a_tag(row)
    with st.expander("🌐 Fält som hämtas (auto)"):
        cL, cR = st.columns(2)
        with cL:
            bolagsnamn  = st.text_input(f"Bolagsnamn {atag}", value=str(row.get("Bolagsnamn","")))
            sektor      = st.text_input(f"Sektor {atag}", value=str(row.get("Sektor","")))
            valuta      = st.text_input(f"Valuta (t.ex. USD, SEK) {atag}", value=str(row.get("Valuta","") or "USD").upper())
            aktuell_kurs = st.number_input(f"Aktuell kurs {atag}", value=float(_clean_number(row.get("Aktuell kurs",0.0))), step=0.01, min_value=0.0)
            utd_arlig    = st.number_input(f"Årlig utdelning {atag}", value=float(_clean_number(row.get("Årlig utdelning",0.0))), step=0.01, min_value=0.0)
            payout_pct   = st.number_input(f"Payout (%) {atag}", value=float(_clean_number(row.get("Payout (%)",0.0))), step=1.0, min_value=0.0)
        with cR:
            utest_m = st.number_input(f"Utestående aktier (miljoner) {atag}", value=float(_clean_number(row.get("Utestående aktier",0.0))), step=1.0, min_value=0.0)
            ps  = st.number_input(f"P/S {atag}",   value=float(_clean_number(row.get("P/S",0.0))), step=0.01, min_value=0.0)
            ps1 = st.number_input(f"P/S Q1 {atag}", value=float(_clean_number(row.get("P/S Q1",0.0))), step=0.01, min_value=0.0)
            ps2 = st.number_input(f"P/S Q2 {atag}", value=float(_clean_number(row.get("P/S Q2",0.0))), step=0.01, min_value=0.0)
            ps3 = st.number_input(f"P/S Q3 {atag}", value=float(_clean_number(row.get("P/S Q3",0.0))), step=0.01, min_value=0.0)
            ps4 = st.number_input(f"P/S Q4 {atag}", value=float(_clean_number(row.get("P/S Q4",0.0))), step=0.01, min_value=0.0)
            pb  = st.number_input(f"P/B {atag}",   value=float(_clean_number(row.get("P/B",0.0))), step=0.01, min_value=0.0)
            pb1 = st.number_input(f"P/B Q1 {atag}", value=float(_clean_number(row.get("P/B Q1",0.0))), step=0.01, min_value=0.0)
            pb2 = st.number_input(f"P/B Q2 {atag}", value=float(_clean_number(row.get("P/B Q2",0.0))), step=0.01, min_value=0.0)
            pb3 = st.number_input(f"P/B Q3 {atag}", value=float(_clean_number(row.get("P/B Q3",0.0))), step=0.01, min_value=0.0)
            pb4 = st.number_input(f"P/B Q4 {atag}", value=float(_clean_number(row.get("P/B Q4",0.0))), step=0.01, min_value=0.0)

    # Beräknat (read-only)
    btag = _b_tag(row)
    with st.expander("🧮 Beräknade fält (auto)"):
        cA, cB = st.columns(2)
        with cA:
            st.number_input(f"P/S-snitt (Q1..Q4) {btag}", value=float(_clean_number(row.get("P/S-snitt (Q1..Q4)",0.0))), step=0.01, disabled=True)
            st.number_input(f"Omsättning om 2 år (M) {btag}", value=float(_clean_number(row.get("Omsättning om 2 år",0.0))), step=1.0, disabled=True)
            st.number_input(f"Riktkurs idag {btag}", value=float(_clean_number(row.get("Riktkurs idag",0.0))), step=0.01, disabled=True)
            st.number_input(f"Riktkurs om 2 år {btag}", value=float(_clean_number(row.get("Riktkurs om 2 år",0.0))), step=0.01, disabled=True)
        with cB:
            st.number_input(f"P/B-snitt (Q1..Q4) {btag}", value=float(_clean_number(row.get("P/B-snitt (Q1..Q4)",0.0))), step=0.01, disabled=True)
            st.number_input(f"Omsättning om 3 år (M) {btag}", value=float(_clean_number(row.get("Omsättning om 3 år",0.0))), step=1.0, disabled=True)
            st.number_input(f"Riktkurs om 1 år {btag}", value=float(_clean_number(row.get("Riktkurs om 1 år",0.0))), step=0.01, disabled=True)
            st.number_input(f"Riktkurs om 3 år {btag}", value=float(_clean_number(row.get("Riktkurs om 3 år",0.0))), step=0.01, disabled=True)

    # Spara
    def _any_core_change(before: pd.Series, after: dict) -> bool:
        core = ["Antal aktier","GAV (SEK)","Omsättning idag","Omsättning nästa år"]
        for k in core:
            b = float(_clean_number(before.get(k,0.0))); a = float(_clean_number(after.get(k,0.0)))
            if abs(a-b) > 1e-12: return True
        return False

    if st.button("💾 Spara"):
        errors = []
        if not ticker.strip(): errors.append("Ticker saknas.")
        if antal < 0: errors.append("Antal aktier kan inte vara negativt.")
        if gav < 0: errors.append("GAV (SEK) kan inte vara negativt.")
        if oms_idag < 0 or oms_nxt < 0: errors.append("Omsättning idag/nästa år kan inte vara negativt.")
        if errors:
            st.error(" | ".join(errors)); return df

        exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())
        exists = bool(exists_mask.any())

        update = {
            # manuella
            "Ticker": ticker.upper(),
            "Antal aktier": float(antal),
            "GAV (SEK)": float(gav),
            "Omsättning idag": float(oms_idag),
            "Omsättning nästa år": float(oms_nxt),
            # auto-fält (tillåt korr)
            "Bolagsnamn": str(bolagsnamn or "").strip(),
            "Sektor": str(sektor or "").strip(),
            "Valuta": str(valuta or "").strip().upper(),
            "Aktuell kurs": float(aktuell_kurs),
            "Årlig utdelning": float(utd_arlig),
            "Payout (%)": float(payout_pct),
            "Utestående aktier": float(utest_m),
            "P/S": float(ps), "P/S Q1": float(ps1), "P/S Q2": float(ps2), "P/S Q3": float(ps3), "P/S Q4": float(ps4),
            "P/B": float(pb), "P/B Q1": float(pb1), "P/B Q2": float(pb2), "P/B Q3": float(pb3), "P/B Q4": float(pb4),
        }

        def _apply_update(df_in: pd.DataFrame, mask, data: dict) -> pd.DataFrame:
            out = df_in.copy()
            for k,v in data.items():
                if k not in out.columns: continue
                if isinstance(v, str):
                    if v.strip(): out.loc[mask, k] = v
                else:
                    out.loc[mask, k] = v
            return out

        if exists:
            before_row = df.loc[exists_mask].iloc[0].copy()
            df = _apply_update(df, exists_mask, update)
            if _any_core_change(before_row, update):
                df.loc[exists_mask, "Senast manuellt uppdaterad"] = now_stamp()
        else:
            base = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Sektor","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad","Auto källa","Senast beräknad","Div_Månader","Div_Vikter"] else "") for c in FINAL_COLS}
            base.update(update)
            base["Senast manuellt uppdaterad"] = now_stamp()
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
            exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())

        # Direkt efter spara → snabb Yahoo (pris etc) + auto-stämpel
        try:
            quick = yahoo_fetch_one_quick(ticker.upper())
            if quick.get("Bolagsnamn"): df.loc[exists_mask, "Bolagsnamn"] = str(quick["Bolagsnamn"])
            if quick.get("Valuta"):     df.loc[exists_mask, "Valuta"]     = str(quick["Valuta"])
            if float(quick.get("Aktuell kurs",0.0))>0: df.loc[exists_mask, "Aktuell kurs"] = float(quick["Aktuell kurs"])
            df.loc[exists_mask, "Årlig utdelning"] = float(quick.get("Årlig utdelning",0.0))
            df.loc[exists_mask, "CAGR 5 år (%)"]   = float(quick.get("CAGR 5 år (%)",0.0))
            df.loc[exists_mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[exists_mask, "Auto källa"] = "Yahoo (snabb)"
        except Exception:
            pass

        # Beräkna & spara
        try:
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)  # BUST cache
            st.success("Sparat, snabbdata hämtad, beräkningar uppdaterade.")
            st.rerun()
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    return df

# app.py (Del 5/5)

def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Data (hela bladet)")
    st.dataframe(df, use_container_width=True)

    st.markdown("**Spara alla beräkningar till Google Sheets**")
    c1, c2 = st.columns(2)
    horizon = c1.selectbox("Score-horisont vid sparning", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    strategy = c2.selectbox("Strategi för score vid sparning", ["Auto (via sektor)","Tillväxt","Utdelning","Finans"], index=0)

    if st.button("💾 Spara beräkningar → Google Sheets"):
        try:
            df2 = enrich_for_save(df, horizon_for_score=horizon, strategy=("Auto" if strategy.startswith("Auto") else strategy))
            save_df(ws_title, df2, bust_cache=True)
            st.success("Beräkningar sparade till Google Sheets.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")


def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Vx"] = port["Valuta"].apply(lambda v: rates.get(str(v).upper(), 1.0))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Vx"]
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]
    port["Vinst (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst (%)"] = np.where(port["Anskaffningsvärde (SEK)"]>0, (port["Vinst (SEK)"]/port["Anskaffningsvärde (SEK)"])*100.0, 0.0)
    tot = float(port["Värde (SEK)"].sum())
    tot_av = float(port["Anskaffningsvärde (SEK)"].sum())
    tot_v = tot - tot_av
    tot_v_pct = (tot_v / tot_av * 100.0) if tot_av > 0 else 0.0

    port["Andel (%)"] = np.where(tot > 0, (port["Värde (SEK)"]/tot)*100.0, 0.0).round(2)
    port["DA (%)"] = np.where(port["Aktuell kurs"]>0, (port["Årlig utdelning"]/port["Aktuell kurs"])*100.0, 0.0).round(2)
    port["YOC (%)"] = np.where(port["GAV (SEK)"]>0, (port["Årlig utdelning"]/port["GAV (SEK)"])*100.0, 0.0).round(2)

    st.markdown(
        f"**Totalt portföljvärde:** {round(tot,2)} SEK  \n"
        f"**Anskaffningsvärde:** {round(tot_av,2)} SEK  \n"
        f"**Vinst:** {round(tot_v,2)} SEK ({round(tot_v_pct,2)} %)"
    )
    st.dataframe(
        port[["Ticker","Bolagsnamn","Sektor","Antal aktier","GAV (SEK)","Aktuell kurs","Valuta",
              "Värde (SEK)","Anskaffningsvärde (SEK)","Vinst (SEK)","Vinst (%)","Årlig utdelning","DA (%)","YOC (%)","Andel (%)"]]
            .sort_values("Andel (%)", ascending=False),
        use_container_width=True
    )


def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag")

    if df.empty:
        st.info("Inga rader."); return

    horizon = st.selectbox("Riktkurs-horisont", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    strategy = st.selectbox("Strategi", ["Auto (via sektor)","Tillväxt","Utdelning","Finans"], index=0)

    subset = st.radio("Visa", ["Alla bolag","Endast portfölj"], horizontal=True)
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()

    base = update_calculations(base)
    base = base[(base[horizon] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inget att visa."); return

    base = score_rows(base, horizon=horizon, strategy=("Auto" if strategy.startswith("Auto") else strategy))

    tag = _horizon_to_tag(horizon)
    base["Uppsida (%)"] = ((base[horizon] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0).round(2)
    base["DA (%)"] = np.where(base["Aktuell kurs"] > 0, (base["Årlig utdelning"]/base["Aktuell kurs"])*100.0, 0.0).round(2)

    sort_on = st.selectbox("Sortera på", ["Score (Total)", "Uppsida (%)", "DA (%)"], index=0)
    ascending = st.checkbox("Omvänd sortering", value=False)

    cols = ["Ticker","Bolagsnamn","Sektor","Aktuell kurs",horizon,"Uppsida (%)","DA (%)","Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence"]
    base = base.sort_values(by=[sort_on], ascending=ascending).reset_index(drop=True)
    st.dataframe(base[cols], use_container_width=True)

def view_dividend_calendar(df: pd.DataFrame, ws_title: str, rates: Dict[str, float]):
    st.subheader("📅 Utdelningskalender (12 månader framåt)")
    months_forward = st.number_input("Antal månader framåt", min_value=3, max_value=24, value=12, step=1)
    write_back = st.checkbox("Skriv tillbaka schema till databasen (Div_Frekvens/år, Div_Månader, Div_Vikter)", value=True)

    if st.button("Bygg kalender"):
        summ, det, df_out = build_dividend_calendar(df, rates, months_forward=int(months_forward), write_back_schedule=bool(write_back))
        st.session_state["div_summ"] = summ
        st.session_state["div_det"] = det
        st.session_state["div_df_out"] = df_out
        st.success("Kalender skapad.")

    if "div_summ" in st.session_state:
        st.markdown("### Summering per månad (SEK)")
        st.dataframe(st.session_state["div_summ"], use_container_width=True)
    if "div_det" in st.session_state:
        st.markdown("### Detalj per bolag/månad (SEK)")
        st.dataframe(st.session_state["div_det"], use_container_width=True)

    c1, c2 = st.columns(2)
    if c1.button("💾 Spara schema + kalender till Google Sheets"):
        try:
            df_to_save = st.session_state.get("div_df_out", df)
            df2 = enrich_for_save(df_to_save, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)
            summ = st.session_state.get("div_summ", pd.DataFrame())
            det  = st.session_state.get("div_det", pd.DataFrame())
            ws_write_df("Utdelningskalender – Summering", summ if not summ.empty else pd.DataFrame(columns=["År","Månad","Månad (sv)","Summa (SEK)"]))
            ws_write_df("Utdelningskalender – Detalj", det if not det.empty else pd.DataFrame(columns=[
                "År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier","Valuta","Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"]))
            st.success("Schema + kalender sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    if c2.button("↻ Rensa kalender-cache"):
        for k in ["div_summ","div_det","div_df_out"]:
            if k in st.session_state: del st.session_state[k]
        st.info("Kalender-cache rensad.")


def main():
    st.title("K-pf-rslag")

    # Välj data-blad
    try:
        titles = list_worksheet_titles() or ["Blad1"]
    except Exception:
        titles = ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    # Sidopanel: tvångsläs om
    if st.sidebar.button("↻ Läs om data nu"):
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
        st.rerun()

    # Valutakurser
    user_rates = sidebar_rates()

    # Läs data + snapshot + beräkna
    df = load_df(ws_title)
    if st.sidebar.checkbox("Skapa snapshot vid start", value=True):
        safe_snapshot(df, ws_title)
    df = update_calculations(df)

    # Globala uppdateringsknappar i sidopanelen
    st.sidebar.markdown("---")
    if st.sidebar.button("⚡ Snabb uppdatering (Yahoo) – alla"):
        try:
            miss = []
            for i, r in df.iterrows():
                tkr = str(r["Ticker"]).strip()
                if not tkr: continue
                q = yahoo_fetch_one_quick(tkr)
                if q.get("Bolagsnamn"): df.at[i, "Bolagsnamn"] = str(q["Bolagsnamn"])
                if q.get("Valuta"):     df.at[i, "Valuta"]     = str(q["Valuta"])
                if float(q.get("Aktuell kurs",0.0))>0: df.at[i, "Aktuell kurs"] = float(q["Aktuell kurs"])
                df.at[i, "Årlig utdelning"] = float(q.get("Årlig utdelning",0.0))
                df.at[i, "CAGR 5 år (%)"]   = float(q.get("CAGR 5 år (%)",0.0))
                df.at[i, "Senast auto uppdaterad"] = now_stamp()
                df.at[i, "Auto källa"] = "Yahoo (snabb)"
                time.sleep(0.8)
            df2 = enrich_for_save(df)
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Snabb uppdatering för alla: OK")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Fel i snabb uppdatering: {e}")

    if st.sidebar.button("🔭 Full uppdatering (alla fetchers) – alla"):
        try:
            for i, r in df.iterrows():
                t = str(r.get("Ticker","")).strip().upper()
                if not t: continue
                df = _update_one_all_fetchers(df, t)
                time.sleep(0.8)
            df2 = enrich_for_save(df)
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Full uppdatering för alla: OK")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Fel i full uppdatering: {e}")

    tabs = st.tabs(["📄 Data", "🧩 Manuell insamling", "📦 Portfölj", "💡 Köpförslag", "📅 Utdelningskalender"])
    with tabs[0]:
        view_data(df, ws_title)
    with tabs[1]:
        df_new = view_manual(df, ws_title)
        if df_new is not None and not df_new.equals(df):
            df = df_new
    with tabs[2]:
        view_portfolio(df, user_rates)
    with tabs[3]:
        view_ideas(df)
    with tabs[4]:
        view_dividend_calendar(df, ws_title, user_rates)


if __name__ == "__main__":
    main()
