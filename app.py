# app.py  — Del 1/6
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Tredjeparts (valfria fallbacks i fetchers hanteras där)
try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

# Egna moduler (måste finnas i stockapp/)
from stockapp.sheets import (
    ws_read_df, ws_write_df, list_worksheet_titles, delete_worksheet
)
from stockapp.rates import (
    read_rates, save_rates, fetch_live_rates, repair_rates_sheet, DEFAULT_RATES
)
from stockapp.dividends import build_dividend_calendar

# (Valfritt) fetchers för enskild/full uppdatering
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


# ---------- tids-hjälpare ----------
def _now_sthlm() -> datetime:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz)
    except Exception:
        return datetime.now()

def now_stamp() -> str:
    return _now_sthlm().strftime("%Y-%m-%d")


# ---------- SNAPSHOT (robust + cleanup) ----------
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

def snapshot_on_start(df: pd.DataFrame, base_ws_title: str):
    """
    Skapar snapshotblad vid första körningen och rensar äldre än 5 dagar.
    Tål 10M-cellsgräns och kvotfel: varnar i sidopanelen men låter appen rulla vidare.
    """
    if st.session_state.get("_snapshot_done"):
        return
    st.session_state["_snapshot_done"] = True

    now = _now_sthlm()
    snap_title = f"{SNAP_PREFIX}{base_ws_title}__{_format_ts(now)}"

    # Light-snapshot om bladet är väldigt stort (undvik cell-limit)
    preferred_cols = [
        "Ticker","Bolagsnamn","Valuta","Sektor",
        "Antal aktier","GAV (SEK)","Aktuell kurs","Årlig utdelning",
        "Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
        "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Score (Total)","Senast manuellt uppdaterad","Senast auto uppdaterad","Senast beräknad"
    ]
    try:
        df_to_save = df
        total_cells = int(df.shape[0]) * int(df.shape[1])
        if total_cells > 200_000:
            keep = [c for c in preferred_cols if c in df.columns]
            if keep:
                df_to_save = df[keep].copy()

        ws_write_df(snap_title, df_to_save)
        st.sidebar.success(f"Snapshot sparat: {snap_title}")
    except Exception as e:
        st.sidebar.warning(f"Kunde inte spara snapshot: {e}")

    # Rensa snapshots äldre än 5 dagar
    try:
        titles = list_worksheet_titles() or []
        cutoff = now - timedelta(days=5)
        for t in titles:
            if not str(t).startswith(SNAP_PREFIX):
                continue
            ts = _parse_snap_title(t)
            if ts and ts < cutoff.replace(tzinfo=None):
                try:
                    delete_worksheet(t)
                except Exception:
                    pass
    except Exception as e:
        st.sidebar.warning(f"Kunde inte rensa gamla snapshot-blad: {e}")


# ---------- Kolumnschema ----------
FINAL_COLS: List[str] = [
    # Bas
    "Ticker", "Bolagsnamn", "Sektor", "Valuta",
    "Antal aktier", "GAV (SEK)", "Aktuell kurs",
    "Utestående aktier",  # miljoner

    # Multiplar (P/S & P/B inkl kvartal och snitt)
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
    "Senast manuellt uppdaterad",      # M
    "Senast auto uppdaterad",          # A (fetchers/kurs)
    "Auto källa",                      # text
    "Senast beräknad",                 # B (beräkningar)

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
    for c in FINAL_COLS:
        if c not in out.columns:
            if any(k in c.lower() for k in [
                "kurs","omsättning","p/s","p/b","utdelning","cagr","aktier",
                "riktkurs","payout","score","uppsida","da","confidence",
                "frekvens","gav"
            ]):
                out[c] = 0.0
            else:
                out[c] = ""
    # typer
    float_cols = [
        "Antal aktier", "GAV (SEK)", "Aktuell kurs", "Utestående aktier",
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
    for c in float_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Auto källa",
              "Senast manuellt uppdaterad","Senast auto uppdaterad","Senast beräknad",
              "Div_Månader","Div_Vikter"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
        else:
            out[c] = ""
    return out


# ---------- IO (med cache-nonce) ----------
@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str, _nonce: int) -> pd.DataFrame:
    # _nonce används endast för att bust:a cachen efter sparning
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

    # Initiera sessionstate EN gång innan widgets skapas
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
            # Rerun så widgets skapas med nya defaults (undviker Streamlit-varning)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta livekurser: {e}")

    if colB.button("↻ Läs sparade kurser"):
        try:
            saved = read_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
            st.rerun()
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
            st.sidebar.success("Valutakurser sparade till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")

    if colD.button("🛠 Reparera bladet"):
        try:
            repair_rates_sheet()
            st.sidebar.success("Bladet reparerat.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte reparera: {e}")

    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}

# app.py — Del 2/6

# ---------- Numerik-hjälpare ----------
def _to_float(x) -> float:
    """Robust konvertering: str med ',' eller ' ' → float. Tomt/N/A → 0.0"""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.floating)):
            if pd.isna(x):
                return 0.0
            return float(x)
        s = str(x).strip()
        if not s or s.lower() in {"nan", "na", "n/a", "-", "—"}:
            return 0.0
        # ta bort tusentalsavgränsare & valutatecken
        s = s.replace(" ", "").replace("€", "").replace("$", "")
        # normalisera decimaltecken
        if s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        return float(s)
    except Exception:
        return 0.0


# ---------- Beräkningar ----------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_ps_pb_snitt(row: pd.Series) -> Tuple[float, float]:
    ps_vals = [row.get("P/S Q1", 0), row.get("P/S Q2", 0), row.get("P/S Q3", 0), row.get("P/S Q4", 0)]
    ps_clean = [float(_to_float(x)) for x in ps_vals if float(_to_float(x)) > 0]
    ps_avg = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0

    pb_vals = [row.get("P/B Q1", 0), row.get("P/B Q2", 0), row.get("P/B Q3", 0), row.get("P/B Q4", 0)]
    pb_clean = [float(_to_float(x)) for x in pb_vals if float(_to_float(x)) > 0]
    pb_avg = round(float(np.mean(pb_clean)), 2) if pb_clean else 0.0
    return ps_avg, pb_avg

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i, r in out.iterrows():
        ps_avg, pb_avg = compute_ps_pb_snitt(r)
        out.at[i, "P/S-snitt (Q1..Q4)"] = ps_avg
        out.at[i, "P/B-snitt (Q1..Q4)"] = pb_avg

        cagr = _to_float(r.get("CAGR 5 år (%)", 0.0))
        # dämpning: begränsa intervall 2–50%
        g = clamp(cagr, 2.0, 50.0) / 100.0

        next_rev = _to_float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            out.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            out.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)
        else:
            out.at[i, "Omsättning om 2 år"] = _to_float(r.get("Omsättning om 2 år", 0.0))
            out.at[i, "Omsättning om 3 år"] = _to_float(r.get("Omsättning om 3 år", 0.0))

        shares_m = _to_float(r.get("Utestående aktier", 0.0))  # antas i miljoner
        if shares_m <= 0 or ps_avg <= 0:
            out.at[i, "Riktkurs idag"] = out.at[i, "Riktkurs om 1 år"] = out.at[i, "Riktkurs om 2 år"] = out.at[i, "Riktkurs om 3 år"] = 0.0
            continue

        out.at[i, "Riktkurs idag"]    = round(_to_float(r.get("Omsättning idag", 0.0))    * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 1 år"] = round(_to_float(r.get("Omsättning nästa år", 0.0)) * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 2 år"] = round(_to_float(out.at[i, "Omsättning om 2 år"])   * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 3 år"] = round(_to_float(out.at[i, "Omsättning om 3 år"])   * ps_avg / shares_m, 2)
    return out

def add_multi_uppsida(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price = out["Aktuell kurs"].map(_to_float).replace(0, np.nan)
    for col, label in [
        ("Riktkurs idag", "Uppsida idag (%)"),
        ("Riktkurs om 1 år", "Uppsida 1 år (%)"),
        ("Riktkurs om 2 år", "Uppsida 2 år (%)"),
        ("Riktkurs om 3 år", "Uppsida 3 år (%)"),
    ]:
        rk = out[col].map(_to_float).replace(0, np.nan)
        out[label] = ((rk - price) / price * 100.0).fillna(0.0).round(2)
    out["DA (%)"] = np.where(
        out["Aktuell kurs"].map(_to_float) > 0,
        (out["Årlig utdelning"].map(_to_float) / out["Aktuell kurs"].map(_to_float)) * 100.0,
        0.0
    ).round(2)
    return out

def _horizon_to_tag(h: str) -> str:
    if "om 1 år" in h: return "1 år"
    if "om 2 år" in h: return "2 år"
    if "om 3 år" in h: return "3 år"
    return "Idag"

def score_rows(df: pd.DataFrame, horizon: str, strategy: str) -> pd.DataFrame:
    out = df.copy()
    out["DA (%)"] = np.where(
        out["Aktuell kurs"].map(_to_float) > 0,
        (out["Årlig utdelning"].map(_to_float) / out["Aktuell kurs"].map(_to_float)) * 100.0,
        0.0
    ).round(2)
    out["Uppsida (%)"] = np.where(
        out["Aktuell kurs"].map(_to_float) > 0,
        (out[horizon].map(_to_float) - out["Aktuell kurs"].map(_to_float)) / out["Aktuell kurs"].map(_to_float) * 100.0,
        0.0
    ).round(2)

    cur_ps = out["P/S"].map(_to_float).replace(0, np.nan)
    ps_avg = out["P/S-snitt (Q1..Q4)"].map(_to_float).replace(0, np.nan)
    cheap_ps = (ps_avg / (cur_ps * 2.0)).clip(upper=1.0).fillna(0.0)

    g_norm = (out["CAGR 5 år (%)"].map(_to_float) / 30.0).clip(0, 1)
    u_norm = (out["Uppsida (%)"] / 50.0).clip(0, 1)
    out["Score (Growth)"] = (0.4 * g_norm + 0.4 * u_norm + 0.2 * cheap_ps) * 100.0

    payout = out["Payout (%)"].map(_to_float)
    payout_health = 1 - (abs(payout - 60.0) / 60.0)
    payout_health = payout_health.clip(0, 1)
    payout_health = np.where(out["Payout (%)"].map(_to_float) <= 0, 0.85, payout_health)
    y_norm = (out["DA (%)"] / 8.0).clip(0, 1)
    grow_ok = np.where(out["CAGR 5 år (%)"].map(_to_float) >= 0, 1.0, 0.6)
    out["Score (Dividend)"] = (0.6 * y_norm + 0.3 * payout_health + 0.1 * grow_ok) * 100.0

    cur_pb = out["P/B"].map(_to_float).replace(0, np.nan)
    pb_avg = out["P/B-snitt (Q1..Q4)"].map(_to_float).replace(0, np.nan)
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
        wg, wd, wf = weights_for_row(str(r.get("Sektor","")), strategy)
        Wg.append(wg); Wd.append(wd); Wf.append(wf)
    Wg = np.array(Wg); Wd = np.array(Wd); Wf = np.array(Wf)
    out["Score (Total)"] = (Wg*out["Score (Growth)"] + Wd*out["Score (Dividend)"] + Wf*out["Score (Financials)"]).round(2)

    need = [
        out["Aktuell kurs"].map(_to_float) > 0,
        out["P/S-snitt (Q1..Q4)"].map(_to_float) > 0,
        out["Omsättning idag"].map(_to_float) >= 0,
        out["Omsättning nästa år"].map(_to_float) >= 0,
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


# ---------- Snabb Yahoo (pris mm) ----------
@st.cache_data(show_spinner=False, ttl=600)
def yahoo_fetch_one_quick(ticker: str) -> Dict[str, float | str]:
    """
    Hämtar snabb-data från Yahoo/yfinance: namn, valuta, aktuell kurs, dividend rate, enkel CAGR.
    """
    out = {"Bolagsnamn":"", "Valuta":"USD", "Aktuell kurs":0.0, "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0}
    if yf is None or not ticker:
        return out
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

        # Enkel CAGR 5y (Total Revenue, årlig)
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

# app.py — Del 3/6

# ---------- Äldst-tabeller ----------
def _oldest_tables(df: pd.DataFrame):
    st.markdown("### ⏱️ Äldst uppdaterade")

    def _to_date(s: str) -> Optional[pd.Timestamp]:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return pd.to_datetime(s)
        except Exception:
            return None

    tmp = df.copy()
    tmp["d_man"]  = tmp["Senast manuellt uppdaterad"].apply(_to_date)
    tmp["d_auto"] = tmp["Senast auto uppdaterad"].apply(_to_date)
    tmp["d_any"]  = tmp[["d_man", "d_auto"]].min(axis=1)

    any_sorted = tmp.dropna(subset=["d_any"]).sort_values("d_any", ascending=True)
    if any_sorted.empty:
        st.info("Inga tidsstämplar ännu.")
    else:
        st.dataframe(
            any_sorted.head(10)[["Ticker", "Bolagsnamn", "d_any"]]
            .rename(columns={"d_any": "Äldst (valfri)"}),
            use_container_width=True
        )

    man_sorted = tmp.dropna(subset=["d_man"]).sort_values("d_man", ascending=True)
    if not man_sorted.empty:
        st.dataframe(
            man_sorted.head(10)[["Ticker", "Bolagsnamn", "d_man"]]
            .rename(columns={"d_man": "Äldst (manuell)"}),
            use_container_width=True
        )
    else:
        st.caption("Inga manuella uppdateringar stämplade ännu.")

    auto_sorted = tmp.dropna(subset=["d_auto"]).sort_values("d_auto", ascending=True)
    if not auto_sorted.empty:
        st.dataframe(
            auto_sorted.head(10)[["Ticker", "Bolagsnamn", "d_auto", "Auto källa"]]
            .rename(columns={"d_auto": "Äldst (auto)"}),
            use_container_width=True
        )
    else:
        st.caption("Inga automatiska uppdateringar stämplade ännu.")

    st.markdown("---")


# ---------- Enskild full uppdatering (alla fetchers) ----------
def _update_one_all_fetchers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    tkr = (ticker or "").strip().upper()
    if not tkr:
        return df

    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        return df

    # Yahoo bred översikt (om modul tillgänglig)
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
                if v is None:
                    continue
                if k_dst == "Utestående aktier":
                    df.loc[mask, k_dst] = float(v) / 1e6
                else:
                    df.loc[mask, k_dst] = float(v) if isinstance(v, (int, float)) else str(v)
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Yahoo"
        except Exception:
            pass

    # Finviz
    if fz_overview:
        try:
            fz = fz_overview(tkr) or {}
            if float(fz.get("ps_ttm", 0.0)) > 0:
                df.loc[mask, "P/S"] = float(fz["ps_ttm"])
            if float(fz.get("pb", 0.0)) > 0:
                df.loc[mask, "P/B"] = float(fz["pb"])
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "Finviz"
        except Exception:
            pass

    # Morningstar
    if ms_overview:
        try:
            ms = ms_overview(tkr) or {}
            if float(ms.get("ps_ttm", 0.0)) > 0:
                df.loc[mask, "P/S"] = float(ms["ps_ttm"])
            if float(ms.get("pb", 0.0)) > 0:
                df.loc[mask, "P/B"] = float(ms["pb"])
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
                    if idx > 4:
                        break
                    df.loc[mask, f"P/B Q{idx}"] = float(pbv or 0.0)
                # uppdatera snitt
                row = df.loc[mask].iloc[0]
                _ps_avg, pb_avg = compute_ps_pb_snitt(row)
                df.loc[mask, "P/B-snitt (Q1..Q4)"] = pb_avg
            df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            df.loc[mask, "Auto källa"] = "SEC"
        except Exception:
            pass

    return df


# ---------- Sidopanel: global uppdatering (mass) ----------
def sidebar_updaters(df: pd.DataFrame, ws_title: str) -> pd.DataFrame:
    st.sidebar.markdown("### 🔄 Massuppdatering")

    delay = st.sidebar.slider("Fördröjning mellan anrop (sek)", min_value=0.2, max_value=1.5, value=0.6, step=0.1)
    do_quick = st.sidebar.button("⚡ Snabbuppdatera alla (Yahoo)")
    do_full  = st.sidebar.button("🛰️ Full uppdatera alla (Yahoo+Finviz+Morningstar+SEC)")

    if do_quick:
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        tickers = df["Ticker"].astype(str).str.upper().tolist()
        miss = []
        for i, tkr in enumerate(tickers, start=1):
            status.write(f"Yahoo snabb för {tkr} ({i}/{len(tickers)})")
            try:
                q = yahoo_fetch_one_quick(tkr)
                m = (df["Ticker"].astype(str).str.upper() == tkr)
                if not m.any():
                    continue
                if q.get("Bolagsnamn"):
                    df.loc[m, "Bolagsnamn"] = str(q["Bolagsnamn"])
                if q.get("Valuta"):
                    df.loc[m, "Valuta"] = str(q["Valuta"])
                if float(q.get("Aktuell kurs", 0.0)) > 0:
                    df.loc[m, "Aktuell kurs"] = float(q["Aktuell kurs"])
                df.loc[m, "Årlig utdelning"] = _to_float(q.get("Årlig utdelning", 0.0))
                df.loc[m, "CAGR 5 år (%)"] = _to_float(q.get("CAGR 5 år (%)", 0.0))
                df.loc[m, "Senast auto uppdaterad"] = now_stamp()
                df.loc[m, "Auto källa"] = "Yahoo (snabb)"
            except Exception as e:
                miss.append(f"{tkr}: {e}")
            time.sleep(delay)
            bar.progress(i / max(1, len(tickers)))

        df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
        try:
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Snabbuppdatering klar & sparad.")
            if miss:
                st.sidebar.warning("Vissa tickers misslyckades:")
                st.sidebar.text("\n".join(miss))
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")

        return df2

    if do_full:
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        tickers = df["Ticker"].astype(str).str.upper().tolist()
        miss = []
        for i, tkr in enumerate(tickers, start=1):
            status.write(f"Full uppdatering för {tkr} ({i}/{len(tickers)})")
            try:
                df = _update_one_all_fetchers(df, tkr)
            except Exception as e:
                miss.append(f"{tkr}: {e}")
            time.sleep(delay)
            bar.progress(i / max(1, len(tickers)))

        df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
        try:
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Full uppdatering klar & sparad.")
            if miss:
                st.sidebar.warning("Vissa tickers misslyckades:")
                st.sidebar.text("\n".join(miss))
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")

        return df2

    return df


# ---------- Manuell insamling ----------
def view_manual(df: pd.DataFrame, ws_title: str):
    st.subheader("🧩 Manuell insamling")

    # Överst: äldst-tabeller
    _oldest_tables(df)

    # Navigering & val
    vis = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    labels = [
        f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r.get("Bolagsnamn", "")).strip() else str(r["Ticker"])
        for _, r in vis.iterrows()
    ]
    labels = ["➕ Lägg till nytt bolag..."] + labels

    if "manual_idx" not in st.session_state:
        st.session_state["manual_idx"] = 0

    sel = st.selectbox(
        "Välj bolag att redigera",
        list(range(len(labels))),
        format_func=lambda i: labels[i],
        index=st.session_state["manual_idx"]
    )
    st.session_state["manual_idx"] = sel

    c_prev, c_next = st.columns([1, 1])
    with c_prev:
        if st.button("⬅️ Föregående", use_container_width=True, disabled=(sel <= 0)):
            st.session_state["manual_idx"] = max(0, sel - 1)
            st.rerun()
    with c_next:
        if st.button("➡️ Nästa", use_container_width=True, disabled=(sel >= len(labels) - 1)):
            st.session_state["manual_idx"] = min(len(labels) - 1, sel + 1)
            st.rerun()

    is_new = (sel == 0)
    if not is_new:
        row = vis.iloc[sel - 1]
    else:
        row = pd.Series({c: (0.0 if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else "") for c in df.columns})

    # Enskild full uppdatering
    col_up1, col_up2 = st.columns([1, 1])
    with col_up1:
        if not is_new and st.button("🔭 Full uppdatering (alla fetchers) för vald ticker"):
            df = _update_one_all_fetchers(df, row["Ticker"])
            df = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            try:
                save_df(ws_title, df, bust_cache=True)
                st.success("Fetchers körda, beräkningar uppdaterade och sparade.")
                st.rerun()
            except Exception as e:
                st.error(f"Kunde inte spara: {e}")
    with col_up2:
        st.caption("Kör Yahoo/Finviz/Morningstar/SEC (om tillgängliga) för just den här tickern.")

    # Obligatoriska fält
    st.markdown("### Obligatoriska fält")
    mtag = _m_tag(row)

    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input(f"Ticker (Yahoo-format) {mtag}", value=str(row.get("Ticker", "")).upper() if not is_new else "", placeholder="t.ex. AAPL")
        antal  = st.number_input(f"Antal aktier (du äger) {mtag}", value=_to_float(row.get("Antal aktier", 0.0) or 0.0), step=1.0, min_value=0.0)
        gav    = st.number_input(f"GAV (SEK) {mtag}", value=_to_float(row.get("GAV (SEK)", 0.0) or 0.0), step=0.01, min_value=0.0, format="%.2f")
    with c2:
        oms_idag = st.number_input(f"Omsättning idag (M) {mtag}", value=_to_float(row.get("Omsättning idag", 0.0) or 0.0), step=1.0, min_value=0.0)
        oms_nxt  = st.number_input(f"Omsättning nästa år (M) {mtag}", value=_to_float(row.get("Omsättning nästa år", 0.0) or 0.0), step=1.0, min_value=0.0)

    # Övriga fält: FETCHERS
    atag = _a_tag(row)
    with st.expander("🌐 Fält som hämtas (auto)"):
        cL, cR = st.columns(2)
        with cL:
            bolagsnamn = st.text_input(f"Bolagsnamn {atag}", value=str(row.get("Bolagsnamn", "")))
            sektor     = st.text_input(f"Sektor {atag}", value=str(row.get("Sektor", "")))
            valuta     = st.text_input(f"Valuta (t.ex. USD, SEK) {atag}", value=str(row.get("Valuta", "") or "USD").upper())
            aktuell_kurs = st.number_input(f"Aktuell kurs {atag}", value=_to_float(row.get("Aktuell kurs", 0.0) or 0.0), step=0.01, min_value=0.0)
            utd_arlig    = st.number_input(f"Årlig utdelning {atag}", value=_to_float(row.get("Årlig utdelning", 0.0) or 0.0), step=0.01, min_value=0.0)
            payout_pct   = st.number_input(f"Payout (%) {atag}", value=_to_float(row.get("Payout (%)", 0.0) or 0.0), step=1.0, min_value=0.0)
        with cR:
            utest_m = st.number_input(f"Utestående aktier (miljoner) {atag}", value=_to_float(row.get("Utestående aktier", 0.0) or 0.0), step=1.0, min_value=0.0)
            ps  = st.number_input(f"P/S {atag}",   value=_to_float(row.get("P/S", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps1 = st.number_input(f"P/S Q1 {atag}", value=_to_float(row.get("P/S Q1", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps2 = st.number_input(f"P/S Q2 {atag}", value=_to_float(row.get("P/S Q2", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps3 = st.number_input(f"P/S Q3 {atag}", value=_to_float(row.get("P/S Q3", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps4 = st.number_input(f"P/S Q4 {atag}", value=_to_float(row.get("P/S Q4", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb  = st.number_input(f"P/B {atag}",   value=_to_float(row.get("P/B", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb1 = st.number_input(f"P/B Q1 {atag}", value=_to_float(row.get("P/B Q1", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb2 = st.number_input(f"P/B Q2 {atag}", value=_to_float(row.get("P/B Q2", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb3 = st.number_input(f"P/B Q3 {atag}", value=_to_float(row.get("P/B Q3", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb4 = st.number_input(f"P/B Q4 {atag}", value=_to_float(row.get("P/B Q4", 0.0) or 0.0), step=0.01, min_value=0.0)

    # Övriga fält: BERÄKNADE (read-only)
    btag = _b_tag(row)
    with st.expander("🧮 Beräknade fält (auto)"):
        cA, cB = st.columns(2)
        with cA:
            st.number_input(f"P/S-snitt (Q1..Q4) {btag}", value=_to_float(row.get("P/S-snitt (Q1..Q4)", 0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Omsättning om 2 år (M) {btag}", value=_to_float(row.get("Omsättning om 2 år", 0.0) or 0.0), step=1.0, disabled=True)
            st.number_input(f"Riktkurs idag {btag}", value=_to_float(row.get("Riktkurs idag", 0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Riktkurs om 2 år {btag}", value=_to_float(row.get("Riktkurs om 2 år", 0.0) or 0.0), step=0.01, disabled=True)
        with cB:
            st.number_input(f"P/B-snitt (Q1..Q4) {btag}", value=_to_float(row.get("P/B-snitt (Q1..Q4)", 0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Omsättning om 3 år (M) {btag}", value=_to_float(row.get("Omsättning om 3 år", 0.0) or 0.0), step=1.0, disabled=True)
            st.number_input(f"Riktkurs om 1 år {btag}", value=_to_float(row.get("Riktkurs om 1 år", 0.0) or 0.0), step=0.01, disabled=True)
            st.number_input(f"Riktkurs om 3 år {btag}", value=_to_float(row.get("Riktkurs om 3 år", 0.0) or 0.0), step=0.01, disabled=True)

    # Spara
    def _any_core_change(before: pd.Series, after: dict) -> bool:
        core = ["Antal aktier", "GAV (SEK)", "Omsättning idag", "Omsättning nästa år"]
        for k in core:
            b = _to_float(before.get(k, 0.0) or 0.0)
            a = _to_float(after.get(k, 0.0) or 0.0)
            if abs(a - b) > 1e-12:
                return True
        return False

    if st.button("💾 Spara"):
        errors = []
        if not str(ticker).strip():
            errors.append("Ticker saknas.")
        if antal < 0:
            errors.append("Antal aktier kan inte vara negativt.")
        if gav < 0:
            errors.append("GAV (SEK) kan inte vara negativt.")
        if oms_idag < 0 or oms_nxt < 0:
            errors.append("Omsättning idag/nästa år kan inte vara negativt.")
        if errors:
            st.error(" | ".join(errors))
            return

        exists_mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
        exists = bool(exists_mask.any())

        update = {
            # manuella
            "Ticker": str(ticker).upper(),
            "Antal aktier": _to_float(antal),
            "GAV (SEK)": _to_float(gav),
            "Omsättning idag": _to_float(oms_idag),
            "Omsättning nästa år": _to_float(oms_nxt),
            # auto-fält (tillåt korr)
            "Bolagsnamn": str(bolagsnamn or "").strip(),
            "Sektor": str(sektor or "").strip(),
            "Valuta": str(valuta or "").strip().upper(),
            "Aktuell kurs": _to_float(aktuell_kurs),
            "Årlig utdelning": _to_float(utd_arlig),
            "Payout (%)": _to_float(payout_pct),
            "Utestående aktier": _to_float(utest_m),
            "P/S": _to_float(ps), "P/S Q1": _to_float(ps1), "P/S Q2": _to_float(ps2), "P/S Q3": _to_float(ps3), "P/S Q4": _to_float(ps4),
            "P/B": _to_float(pb), "P/B Q1": _to_float(pb1), "P/B Q2": _to_float(pb2), "P/B Q3": _to_float(pb3), "P/B Q4": _to_float(pb4),
        }

        def _apply_update(df_in: pd.DataFrame, mask, data: dict) -> pd.DataFrame:
            out = df_in.copy()
            for k, v in data.items():
                if k not in out.columns:
                    continue
                if isinstance(v, str):
                    if v.strip():
                        out.loc[mask, k] = v
                else:
                    out.loc[mask, k] = v
            return out

        if exists:
            before_row = df.loc[exists_mask].iloc[0].copy()
            df = _apply_update(df, exists_mask, update)
            if _any_core_change(before_row, update):
                df.loc[exists_mask, "Senast manuellt uppdaterad"] = now_stamp()
        else:
            base = {c: (0.0 if c not in ["Ticker", "Bolagsnamn", "Sektor", "Valuta",
                                         "Senast manuellt uppdaterad", "Senast auto uppdaterad",
                                         "Auto källa", "Senast beräknad", "Div_Månader", "Div_Vikter"]
                        else "")
                    for c in FINAL_COLS}
            base.update(update)
            base["Senast manuellt uppdaterad"] = now_stamp()
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
            exists_mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())

        # Direkt efter spara → snabb Yahoo (pris etc) + auto-stämpel
        try:
            quick = yahoo_fetch_one_quick(str(ticker).upper())
            if quick.get("Bolagsnamn"):
                df.loc[exists_mask, "Bolagsnamn"] = str(quick["Bolagsnamn"])
            if quick.get("Valuta"):
                df.loc[exists_mask, "Valuta"] = str(quick["Valuta"])
            if _to_float(quick.get("Aktuell kurs", 0.0)) > 0:
                df.loc[exists_mask, "Aktuell kurs"] = _to_float(quick["Aktuell kurs"])
            df.loc[exists_mask, "Årlig utdelning"] = _to_float(quick.get("Årlig utdelning", 0.0))
            df.loc[exists_mask, "CAGR 5 år (%)"]   = _to_float(quick.get("CAGR 5 år (%)", 0.0))
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

# app.py — Del 4/6

# ---------- Data-flik ----------
def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Data (hela bladet)")
    st.dataframe(df, use_container_width=True)

    st.markdown("**Spara alla beräkningar till Google Sheets**")
    c1, c2 = st.columns(2)
    horizon = c1.selectbox(
        "Score-horisont vid sparning",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=0
    )
    strategy = c2.selectbox(
        "Strategi för score vid sparning",
        ["Auto (via sektor)", "Tillväxt", "Utdelning", "Finans"],
        index=0
    )

    if st.button("💾 Spara beräkningar → Google Sheets"):
        try:
            strat = "Auto" if strategy.startswith("Auto") else strategy
            df2 = enrich_for_save(df, horizon_for_score=horizon, strategy=strat)
            save_df(ws_title, df2, bust_cache=True)
            st.success("Beräkningar sparade till Google Sheets.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")


# ---------- Portfölj ----------
def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")

    port = df.copy()
    # säkerställa numerik
    for col in ["Antal aktier", "Aktuell kurs", "Årlig utdelning", "GAV (SEK)"]:
        if col in port.columns:
            port[col] = port[col].map(_to_float)

    port = port[port["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs per rad
    port["Vx"] = port["Valuta"].apply(lambda v: rates.get(str(v).upper(), 1.0))

    # Värden
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Vx"]
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]
    port["Vinst (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst (%)"] = np.where(
        port["Anskaffningsvärde (SEK)"] > 0,
        (port["Vinst (SEK)"] / port["Anskaffningsvärde (SEK)"]) * 100.0,
        0.0
    )

    # Utdelning
    port["DA (%)"] = np.where(
        port["Aktuell kurs"] > 0,
        (port["Årlig utdelning"] / port["Aktuell kurs"]) * 100.0,
        0.0
    )
    # YOC baseras på GAV (SEK) och utdelning i SEK (via växelkurs)
    port["YOC (%)"] = np.where(
        port["GAV (SEK)"] > 0,
        ((port["Årlig utdelning"] * port["Vx"]) / port["GAV (SEK)"]) * 100.0,
        0.0
    )
    port["Årsutdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Vx"]

    tot_val = float(port["Värde (SEK)"].sum())
    tot_cost = float(port["Anskaffningsvärde (SEK)"].sum())
    tot_gain = tot_val - tot_cost
    tot_gain_pct = (tot_gain / tot_cost * 100.0) if tot_cost > 0 else 0.0
    tot_div_sek = float(port["Årsutdelning (SEK)"].sum())

    port["Andel (%)"] = np.where(tot_val > 0, (port["Värde (SEK)"] / tot_val) * 100.0, 0.0)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Portföljvärde (SEK)", f"{tot_val:,.0f}".replace(",", " "), delta=None)
    col_b.metric("Anskaffningsvärde (SEK)", f"{tot_cost:,.0f}".replace(",", " "), delta=None)
    col_c.metric("Vinst (SEK)", f"{tot_gain:,.0f}".replace(",", " "), delta=f"{tot_gain_pct:.2f}%")
    col_d.metric("Årlig utdelning (SEK)", f"{tot_div_sek:,.0f}".replace(",", " "), delta=None)

    show_cols = [
        "Ticker", "Bolagsnamn", "Sektor",
        "Antal aktier", "Valuta", "Aktuell kurs", "GAV (SEK)",
        "Värde (SEK)", "Anskaffningsvärde (SEK)", "Vinst (SEK)", "Vinst (%)",
        "Årlig utdelning", "DA (%)", "YOC (%)", "Årsutdelning (SEK)", "Andel (%)"
    ]
    existing = [c for c in show_cols if c in port.columns]

    st.dataframe(
        port[existing].sort_values("Andel (%)", ascending=False),
        use_container_width=True
    )


# ---------- Köpförslag ----------
def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag")

    if df.empty:
        st.info("Inga rader.")
        return

    horizon = st.selectbox(
        "Riktkurs-horisont",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=0
    )
    strategy = st.selectbox(
        "Strategi",
        ["Auto (via sektor)", "Tillväxt", "Utdelning", "Finans"],
        index=0
    )

    subset = st.radio("Visa", ["Alla bolag", "Endast portfölj"], horizontal=True)
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()

    base = update_calculations(base)
    base = base[(base[horizon] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inget att visa.")
        return

    base = score_rows(base, horizon=horizon, strategy=("Auto" if strategy.startswith("Auto") else strategy))

    show_components = st.checkbox("Visa komponentpoäng (Growth/Dividend/Financials) för vald horisont", True)
    show_saved = st.checkbox("Visa sparade horisontpoäng från Google Sheets", True)

    tag = _horizon_to_tag(horizon)
    saved_cols_all = [
        "Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)",
        "Score Growth (Idag)", "Score Dividend (Idag)", "Score Financials (Idag)",
        "Score Growth (1 år)", "Score Dividend (1 år)", "Score Financials (1 år)",
        "Score Growth (2 år)", "Score Dividend (2 år)", "Score Financials (2 år)",
        "Score Growth (3 år)", "Score Dividend (3 år)", "Score Financials (3 år)",
    ]
    available_saved = [c for c in saved_cols_all if c in base.columns]

    default_saved = [f"Score Total ({tag})"]
    for group in ["Growth", "Dividend", "Financials"]:
        colname = f"Score {group} ({tag})"
        if colname in available_saved:
            default_saved.append(colname)

    selected_saved_cols = []
    if show_saved and available_saved:
        selected_saved_cols = st.multiselect(
            "Välj sparade score-kolumner att visa",
            options=available_saved,
            default=default_saved
        )

    sort_options = ["Score (Total)", "Uppsida (%)", "DA (%)"]
    for c in ["Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)"]:
        if c in base.columns and c not in sort_options:
            sort_options.append(c)
    sort_on = st.selectbox("Sortera på", sort_options, index=0)

    base["Uppsida (%)"] = ((base[horizon] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0).round(2)
    base["DA (%)"] = np.where(base["Aktuell kurs"] > 0, (base["Årlig utdelning"] / base["Aktuell kurs"]) * 100.0, 0.0).round(2)

    ascending = False
    if sort_on == "Uppsida (%)":
        trim_mode = st.checkbox("Visa trim/sälj-läge (minst uppsida först)", value=False)
        if trim_mode:
            ascending = True
    reverse_global = st.checkbox("Omvänd sortering (gäller valt fält)", value=False)
    if reverse_global:
        ascending = not ascending

    cols = ["Ticker", "Bolagsnamn", "Sektor", "Aktuell kurs", horizon, "Uppsida (%)", "DA (%)"]
    if show_components:
        cols += ["Score (Growth)", "Score (Dividend)", "Score (Financials)", "Score (Total)", "Confidence"]
    else:
        cols += ["Score (Total)", "Confidence"]
    if show_saved and selected_saved_cols:
        cols += selected_saved_cols

    base = base.sort_values(by=[sort_on], ascending=ascending).reset_index(drop=True)
    st.dataframe(base[cols], use_container_width=True)

    st.markdown("---")
    st.markdown("### Kortvisning (bläddra)")
    if "idea_idx" not in st.session_state:
        st.session_state["idea_idx"] = 0
    st.session_state["idea_idx"] = st.number_input(
        "Visa rad #", min_value=0, max_value=max(0, len(base) - 1),
        value=st.session_state["idea_idx"], step=1
    )
    r = base.iloc[st.session_state["idea_idx"]]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- **Sektor:** {r.get('Sektor', '—')}")
        st.write(f"- **Aktuell kurs:** {round(_to_float(r['Aktuell kurs']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs idag:** {round(_to_float(r['Riktkurs idag']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 1 år:** {round(_to_float(r['Riktkurs om 1 år']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 2 år:** {round(_to_float(r['Riktkurs om 2 år']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 3 år:** {round(_to_float(r['Riktkurs om 3 år']), 2)} {r['Valuta']}")
        st.write(f"- **Uppsida ({horizon}):** {round(_to_float(r['Uppsida (%)']), 2)} %")
    with c2:
        st.write(f"- **P/S-snitt (Q1..Q4):** {round(_to_float(r['P/S-snitt (Q1..Q4)']), 2)}")
        st.write(f"- **P/B-snitt (Q1..Q4):** {round(_to_float(r['P/B-snitt (Q1..Q4)']), 2)}")
        st.write(f"- **Omsättning idag (M):** {round(_to_float(r['Omsättning idag']), 2)}")
        st.write(f"- **Omsättning nästa år (M):** {round(_to_float(r['Omsättning nästa år']), 2)}")
        st.write(f"- **Årlig utdelning:** {round(_to_float(r['Årlig utdelning']), 2)}")
        st.write(f"- **Payout:** {round(_to_float(r['Payout (%)']), 2)} %")
        st.write(f"- **DA (egen):** {round(_to_float(r['DA (%)']), 2)} %")
        st.write(f"- **CAGR 5 år:** {round(_to_float(r['CAGR 5 år (%)']), 2)} %")
        st.write(
            "- **Score – Growth / Dividend / Financials / Total:** "
            f"{round(_to_float(r['Score (Growth)']), 1)} / "
            f"{round(_to_float(r['Score (Dividend)']), 1)} / "
            f"{round(_to_float(r['Score (Financials)']), 1)} / "
            f"**{round(_to_float(r['Score (Total)']), 1)}** "
            f"(Conf {int(_to_float(r['Confidence']))}%)"
        )

# app.py — Del 5/6

# ---------- Utdelningskalender ----------
def view_dividend_calendar(df: pd.DataFrame, ws_title: str, rates: Dict[str, float]):
    st.subheader("📅 Utdelningskalender (12–24 månader framåt)")

    months_forward = st.number_input(
        "Antal månader framåt",
        min_value=3, max_value=24, value=12, step=1
    )
    write_back = st.checkbox(
        "Skriv tillbaka schema till databasen (Div_Frekvens/år, Div_Månader, Div_Vikter)",
        value=True
    )

    if st.button("Bygg kalender"):
        try:
            summ, det, df_out = build_dividend_calendar(
                df, rates,
                months_forward=int(months_forward),
                write_back_schedule=bool(write_back)
            )
            st.session_state["div_summ"] = summ
            st.session_state["div_det"] = det
            st.session_state["div_df_out"] = df_out
            st.success("Kalender skapad.")
        except Exception as e:
            st.error(f"Kunde inte bygga kalender: {e}")

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
            ws_write_df("Utdelningskalender – Summering", summ if not summ.empty else pd.DataFrame(
                columns=["År", "Månad", "Månad (sv)", "Summa (SEK)"]
            ))
            ws_write_df("Utdelningskalender – Detalj", det if not det.empty else pd.DataFrame(
                columns=[
                    "År", "Månad", "Månad (sv)", "Ticker", "Bolagsnamn",
                    "Antal aktier", "Valuta", "Per utbetalning (valuta)",
                    "SEK-kurs", "Summa (SEK)"
                ]
            ))
            st.success("Schema + kalender sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    if c2.button("↻ Rensa kalender-cache"):
        for k in ["div_summ", "div_det", "div_df_out"]:
            if k in st.session_state:
                del st.session_state[k]
        st.info("Kalender-cache rensad.")


# ---------- Massuppdatering (sidopanel) ----------
def _apply_quick_yahoo_row(df: pd.DataFrame, i: int, tkr: str):
    """Uppdatera en rad med snabb Yahoo (pris/valuta/utdelning/CAGR)."""
    try:
        q = yahoo_fetch_one_quick(tkr)
        if q.get("Bolagsnamn"):
            df.at[i, "Bolagsnamn"] = str(q["Bolagsnamn"])
        if q.get("Valuta"):
            df.at[i, "Valuta"] = str(q["Valuta"]).upper()
        if _to_float(q.get("Aktuell kurs", 0.0)) > 0:
            df.at[i, "Aktuell kurs"] = _to_float(q["Aktuell kurs"])
        # Årlig utdelning & CAGR kan vara 0, skriv in explicit
        df.at[i, "Årlig utdelning"] = _to_float(q.get("Årlig utdelning", 0.0))
        df.at[i, "CAGR 5 år (%)"] = _to_float(q.get("CAGR 5 år (%)", 0.0))

        df.at[i, "Senast auto uppdaterad"] = now_stamp()
        df.at[i, "Auto källa"] = "Yahoo (snabb)"
    except Exception:
        # Swallow; lämna raden orörd
        pass


def _mass_update_quick(df: pd.DataFrame, ws_title: str):
    """Snabbuppdatera kurser mm för alla tickers (Yahoo)."""
    st.sidebar.write("Startar snabbuppdatering (Yahoo)…")
    total = len(df)
    if total == 0:
        st.sidebar.info("Inga rader i bladet.")
        return

    bar = st.sidebar.progress(0)
    status = st.sidebar.empty()
    failed = []

    for idx, row in df.reset_index(drop=True).iterrows():
        tkr = str(row.get("Ticker", "")).strip().upper()
        if not tkr:
            bar.progress((idx + 1) / total)
            continue

        status.write(f"{idx+1}/{total} – {tkr}")
        try:
            _apply_quick_yahoo_row(df, row.name if "name" in dir(row) else idx, tkr)
        except Exception as e:
            failed.append(f"{tkr}: {e}")

        # 0.5s delay för att vara snäll
        import time as _time
        _time.sleep(0.5)
        bar.progress((idx + 1) / total)

    # Beräkna och spara en gång i slutet
    try:
        df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
        save_df(ws_title, df2, bust_cache=True)
        st.sidebar.success("Snabbuppdatering klar och sparad.")
        if failed:
            st.sidebar.warning("Vissa poster misslyckades; se lista nedan.")
            st.sidebar.text_area("Fel", "\n".join(failed), height=150)
    except Exception as e:
        st.sidebar.error(f"Kunde inte spara snabbuppdateringen: {e}")


def _mass_update_full(df: pd.DataFrame, ws_title: str):
    """Full massuppdatering: Yahoo+Finviz+Morningstar+SEC (om tillgängligt)."""
    st.sidebar.write("Startar full massuppdatering (Yahoo/Finviz/Morningstar/SEC)…")
    total = len(df)
    if total == 0:
        st.sidebar.info("Inga rader i bladet.")
        return

    bar = st.sidebar.progress(0)
    status = st.sidebar.empty()
    failed = []

    for idx, row in df.reset_index(drop=True).iterrows():
        tkr = str(row.get("Ticker", "")).strip().upper()
        if not tkr:
            bar.progress((idx + 1) / total)
            continue

        status.write(f"{idx+1}/{total} – {tkr}")
        try:
            df[:] = _update_one_all_fetchers(df, tkr)  # uppdaterar in-place mot mask
        except Exception as e:
            failed.append(f"{tkr}: {e}")

        # 0.6s delay
        import time as _time
        _time.sleep(0.6)
        bar.progress((idx + 1) / total)

    try:
        df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
        save_df(ws_title, df2, bust_cache=True)
        st.sidebar.success("Full massuppdatering klar och sparad.")
        if failed:
            st.sidebar.warning("Vissa poster misslyckades; se lista nedan.")
            st.sidebar.text_area("Fel", "\n".join(failed), height=150)
    except Exception as e:
        st.sidebar.error(f"Kunde inte spara fulluppdateringen: {e}")


def sidebar_massupdate_controls(df: pd.DataFrame, ws_title: str):
    st.sidebar.markdown("### 🔄 Massuppdatering")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("⚡ Snabb (Yahoo)"):
        _mass_update_quick(df, ws_title)
    if c2.button("🧰 Full (alla fetchers)"):
        _mass_update_full(df, ws_title)

# app.py — Del 6/6

# ---------- Portfölj ----------
def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")

    def vx(cur: str) -> float:
        return rates.get(str(cur).upper(), 1.0)

    out = df.copy()

    # Robust typkonvertering (komma/punkt + tomma strängar)
    for c in ["Antal aktier", "Aktuell kurs", "Årlig utdelning", "GAV (SEK)"]:
        if c in out.columns:
            out[c] = (
                out[c]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace(["", " ", "nan", "NaN", None], "0")
                .astype(float)
            )
        else:
            out[c] = 0.0

    out = out[out["Antal aktier"] > 0].copy()
    if out.empty:
        st.info("Du äger inga aktier ännu.")
        return

    out["Växelkurs"] = out["Valuta"].apply(vx)
    out["Värde (SEK)"] = out["Antal aktier"] * out["Aktuell kurs"] * out["Växelkurs"]
    out["Anskaffningsvärde (SEK)"] = out["Antal aktier"] * out["GAV (SEK)"]
    out["Vinst (SEK)"] = out["Värde (SEK)"] - out["Anskaffningsvärde (SEK)"]
    out["Vinst (%)"] = np.where(
        out["Anskaffningsvärde (SEK)"] > 0,
        out["Vinst (SEK)"] / out["Anskaffningsvärde (SEK)"] * 100.0,
        0.0,
    ).round(2)

    # Direktavkastning på dagens kurs
    out["DA (%)"] = np.where(
        out["Aktuell kurs"] > 0,
        (out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0,
        0.0,
    ).round(2)

    # Utdelning i SEK och YOC (Yield on Cost)
    out["Utd/aktie (SEK)"] = out["Årlig utdelning"] * out["Växelkurs"]
    out["YOC (%)"] = np.where(
        out["GAV (SEK)"] > 0,
        (out["Utd/aktie (SEK)"] / out["GAV (SEK)"]) * 100.0,
        0.0,
    ).round(2)

    total_värde = float(out["Värde (SEK)"].sum())
    total_anskv = float(out["Anskaffningsvärde (SEK)"].sum())
    total_vinst = total_värde - total_anskv
    total_vinst_pct = (total_vinst / total_anskv * 100.0) if total_anskv > 0 else 0.0
    total_utd_år_sek = float((out["Antal aktier"] * out["Utd/aktie (SEK)"]).sum())

    st.markdown(
        f"""
**Totalt portföljvärde:** {total_värde:,.2f} SEK  
**Anskaffningsvärde:** {total_anskv:,.2f} SEK  
**Vinst:** {total_vinst:,.2f} SEK ({total_vinst_pct:.2f}%)  
**Total årlig utdelning (SEK):** {total_utd_år_sek:,.2f}
""".replace(",", " ")
    )

    show_cols = [
        "Ticker", "Bolagsnamn", "Sektor", "Valuta",
        "Antal aktier", "GAV (SEK)", "Aktuell kurs",
        "Värde (SEK)", "Anskaffningsvärde (SEK)", "Vinst (SEK)", "Vinst (%)",
        "Årlig utdelning", "Utd/aktie (SEK)", "DA (%)", "YOC (%)"
    ]
    existing_cols = [c for c in show_cols if c in out.columns]
    st.dataframe(out[existing_cols].sort_values("Värde (SEK)", ascending=False), use_container_width=True)


# ---------- Huvudprogram ----------
def main():
    st.title("K-pf-rslag")

    # Välj data-blad
    try:
        titles = list_worksheet_titles() or ["Blad1"]
    except Exception:
        titles = ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    # Snabbläs-om vid behov
    if st.sidebar.button("↻ Läs om data nu"):
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
        st.rerun()

    # Valutakurser (läs/spara/live + inputs)
    user_rates = sidebar_rates()

    # Läs data, säkerställ schema, snapshot och grundberäkna
    df = load_df(ws_title)
    try:
        snapshot_on_start(df, ws_title)   # skapar snapshot & städar äldre; ignorerar fel internt
    except Exception:
        pass
    df = update_calculations(df)

    # Knappar för massuppdatering i sidopanelen
    sidebar_massupdate_controls(df, ws_title)

    # Flikar
    tabs = st.tabs([
        "📄 Data",
        "🧩 Manuell insamling",
        "📦 Portfölj",
        "💡 Köpförslag",
        "📅 Utdelningskalender",
    ])
    with tabs[0]:
        view_data(df, ws_title)
    with tabs[1]:
        view_manual(df, ws_title)
    with tabs[2]:
        view_portfolio(df, user_rates)
    with tabs[3]:
        view_ideas(df)
    with tabs[4]:
        view_dividend_calendar(df, ws_title, user_rates)


if __name__ == "__main__":
    main()
