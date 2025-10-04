# app.py
from __future__ import annotations

import time
import datetime as _dt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="K-pf-rslag", layout="wide")
st.title("K-pf-rslag")

# Valfri manuell insamlingsvy
try:
    from stockapp.manual_collect import manual_collect_view
except Exception:
    manual_collect_view = None  # type: ignore

# Våra moduler
from stockapp.sheets import get_ws, ws_read_df, save_dataframe, list_sheet_names
from stockapp.rates import read_rates, save_rates, DEFAULT_RATES, fetch_live_rates
from stockapp.fetchers.yahoo import get_all as yahoo_get
from stockapp.fetchers.sec import get_pb_quarters  # SEC-P/B 4 kvartal (+ bvps i details)
# Valfria kompletteringskällor
from stockapp.fetchers.finviz import get_overview as finviz_get
from stockapp.fetchers.morningstar import get_overview as ms_get
from stockapp.fetchers.stocktwits import get_symbol_summary as stw_get


# ---------------- Schema ----------------
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Sektor", "Industri", "Valuta", "Aktuell kurs",
    "Utestående aktier (milj.)", "Market Cap",
    "P/S (TTM)", "P/B", "EV/EBITDA (ttm)",
    "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)",
    "Revenue TTM (M)", "Revenue growth (%)", "Book value / share",
    "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
    # P/B historik (SEC)
    "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",
    # PB-riktkurser (NYTT)
    "PB Riktkurs idag", "PB Riktkurs om 1 år", "PB Riktkurs om 2 år", "PB Riktkurs om 3 år",
    "BVPS CAGR (%)",
    # PS-modellen & omsättningar
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    # Portfölj
    "Antal aktier",
    # Övrigt
    "CAGR 5 år (%)", "Senast manuellt uppdaterad",
    # Råfält för EV/EBITDA-strategi
    "_y_ev_now", "_y_ebitda_now",
    # Stocktwits (om hämtat)
    "STW meddelanden 24h", "STW bull 24h", "STW bear 24h",
    "STW bull ratio (%)", "STW bevakare", "STW meddelanden/h (24h)", "STW senast",
    # Info
    "Senast uppdaterad källa",
]

NUMERIC_COLS = [
    "Aktuell kurs", "Utestående aktier (milj.)", "Market Cap",
    "P/S (TTM)", "P/B", "EV/EBITDA (ttm)",
    "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)",
    "Revenue TTM (M)", "Revenue growth (%)", "Book value / share",
    "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
    "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",
    "PB Riktkurs idag", "PB Riktkurs om 1 år", "PB Riktkurs om 2 år", "PB Riktkurs om 3 år",
    "BVPS CAGR (%)",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "CAGR 5 år (%)",
    "_y_ev_now", "_y_ebitda_now",
    "STW meddelanden 24h", "STW bull 24h", "STW bear 24h",
    "STW bull ratio (%)", "STW bevakare", "STW meddelanden/h (24h)",
]

# ---- Dividend scoring baseline ----
DIV_BASE_YIELD = 15.0   # % yield som ger full pott vid baseline
DIV_BASE_PAYOUT = 80.0  # % payout som baseline
DIV_MIN_PAYOUT = 30.0   # clamp
DIV_MAX_PAYOUT = 200.0  # clamp

# ---- Stödkonstanter för vyer ----
HORIZONS = ["Idag", "Om 1 år", "Om 2 år", "Om 3 år"]


# ---------------- Hjälpare ----------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({c: [] for c in FINAL_COLS})
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = 0.0 if c in NUMERIC_COLS else ""
    return df

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Sektor","Industri","Valuta","Senast manuellt uppdaterad","Senast uppdaterad källa","STW senast"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def current_yield_pct(row: pd.Series) -> float:
    try:
        div = float(row.get("Årlig utdelning", 0.0))
        price = float(row.get("Aktuell kurs", 0.0))
        if div > 0 and price > 0:
            return round(100.0 * div / price, 2)
    except Exception:
        pass
    return float(row.get("Dividend yield (%)", 0.0))

def _ps_avg(row: pd.Series) -> float:
    pss = float(row.get("P/S-snitt", 0.0))
    if pss > 0:
        return pss
    vals = [float(row.get(k, 0.0)) for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"] if float(row.get(k,0.0))>0]
    return float(np.mean(vals)) if vals else 0.0

def _pb_avg(row: pd.Series) -> float:
    pbs = float(row.get("P/B-snitt (Q1..Q4)", 0.0))
    if pbs > 0:
        return pbs
    vals = [float(row.get(k, 0.0)) for k in ["P/B Q1","P/B Q2","P/B Q3","P/B Q4"] if float(row.get(k,0.0))>0]
    return float(np.mean(vals)) if vals else float(row.get("P/B", 0.0))

# ---- Prognos av omsättning (sparas) från "Nästa år" via CAGR5 + dämpning
def _proj_from_next_year(row: pd.Series) -> tuple[float, float]:
    next_year = float(row.get("Omsättning nästa år", 0.0))
    cagr5 = float(row.get("CAGR 5 år (%)", 0.0)) / 100.0
    g1 = _clamp(cagr5, -0.20, 0.50)
    if next_year <= 0:
        return 0.0, 0.0
    rev2 = round(next_year * (1.0 + g1), 2)
    rev3 = round(rev2 * (1.0 + g1*0.7), 2)
    return rev2, rev3

def _revenue_for_horizon(row: pd.Series, horizon: str) -> float:
    if horizon == "Idag":
        return float(row.get("Omsättning idag", 0.0))
    if horizon == "Om 1 år":
        return float(row.get("Omsättning nästa år", 0.0))
    rev2, rev3 = _proj_from_next_year(row)
    if horizon == "Om 2 år":
        return rev2
    if horizon == "Om 3 år":
        return rev3
    return 0.0

def target_price_ps(row: pd.Series, horizon: str) -> float:
    ps = _ps_avg(row)
    rev_m = _revenue_for_horizon(row, horizon)
    shares_m = float(row.get("Utestående aktier (milj.)", 0.0))
    if ps > 0 and rev_m > 0 and shares_m > 0:
        return round((ps * rev_m) / shares_m, 2)
    return 0.0

def ps_targets_all(row: pd.Series) -> dict:
    return {hz: target_price_ps(row, hz) for hz in HORIZONS}

# ---- PB via BVPS CAGR ----
def _annual_years_between(d0: _dt.date, d1: _dt.date) -> float:
    return max(1e-6, abs((d1 - d0).days) / 365.25)

def _cagr_from_points(v0: float, v1: float, years: float) -> float:
    if v0 <= 0 or v1 <= 0 or years <= 0:
        return 0.0
    return (v1 / v0) ** (1.0 / years) - 1.0

def _bvps_cagr_from_sec_details(details: list[dict]) -> float:
    """
    details: från sec.get_pb_quarters() → [{"date": "YYYY-MM-DD", "bvps": 12.34}, ...]
    Returnerar CAGR (andel) och klampar till [-0.20, +0.30].
    """
    if not details or len(details) < 2:
        return 0.0
    d_sorted = sorted(details, key=lambda x: x.get("date", ""))
    try:
        d0 = _dt.date.fromisoformat(d_sorted[0]["date"])
        d1 = _dt.date.fromisoformat(d_sorted[-1]["date"])
    except Exception:
        return 0.0
    v0 = float(d_sorted[0].get("bvps") or 0.0)
    v1 = float(d_sorted[-1].get("bvps") or 0.0)
    yrs = _annual_years_between(d0, d1)
    g = _cagr_from_points(v0, v1, yrs)
    return _clamp(g, -0.20, 0.30)

def pb_targets_all_from_bvps(pb_avg: float, bvps0: float, g_bv: float) -> dict:
    """
    Returnerar PB-riktkurser för Idag/1y/2y/3y baserat på BVPS-tillväxt (dämpad 0.7/0.5).
    """
    if pb_avg <= 0 or bvps0 <= 0:
        return {"Idag": 0.0, "Om 1 år": 0.0, "Om 2 år": 0.0, "Om 3 år": 0.0}
    bvps1 = bvps0 * (1.0 + g_bv)
    bvps2 = bvps1 * (1.0 + 0.7*g_bv)
    bvps3 = bvps2 * (1.0 + 0.5*g_bv)
    return {
        "Idag": round(pb_avg * bvps0, 2),
        "Om 1 år": round(pb_avg * bvps1, 2),
        "Om 2 år": round(pb_avg * bvps2, 2),
        "Om 3 år": round(pb_avg * bvps3, 2),
    }

def pb_targets_from_row(row: pd.Series) -> dict:
    """Använd sparade PB-riktkurser om de finns – annars försök räkna 'Idag' från P/B * BVPS."""
    have_all = all(float(row.get(f"PB Riktkurs {hz}", 0.0)) > 0 for hz in ["idag", "om 1 år", "om 2 år", "om 3 år"])
    if have_all:
        return {
            "Idag": float(row.get("PB Riktkurs idag", 0.0)),
            "Om 1 år": float(row.get("PB Riktkurs om 1 år", 0.0)),
            "Om 2 år": float(row.get("PB Riktkurs om 2 år", 0.0)),
            "Om 3 år": float(row.get("PB Riktkurs om 3 år", 0.0)),
        }
    # fallback: endast idag
    pb = _pb_avg(row)
    bvps0 = float(row.get("Book value / share", 0.0))
    return {
        "Idag": round(pb * bvps0, 2) if (pb > 0 and bvps0 > 0) else 0.0,
        "Om 1 år": 0.0, "Om 2 år": 0.0, "Om 3 år": 0.0
    }

def upsides_from(price: float, targets: dict) -> dict:
    return {hz: (round(((tp - price)/price*100.0), 2) if (price>0 and tp>0) else 0.0) for hz, tp in targets.items()}

def sund_utdelning_80(row: pd.Series) -> float:
    div = float(row.get("Årlig utdelning", 0.0))
    p = float(row.get("Payout ratio (%)", 0.0))
    if div > 0 and p > 0:
        return round(div * (0.80 / (p/100.0)), 4)
    return 0.0

# --- Format-hjälpare ---
def _fmt_num(v, nd=2):
    try:
        f = float(v)
        if np.isnan(f):
            return "—"
        return f"{f:.{nd}f}"
    except Exception:
        return "—"

def _fmt_pct(v, nd=2):
    try:
        f = float(v)
        if np.isnan(f):
            return "—"
        return f"{f:.{nd}f}%"
    except Exception:
        return "—"


# ---------------- Beräkningar + persist ----------------
def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        df.at[i, "P/S-snitt"] = round(np.mean(ps_clean), 2) if ps_clean else float(rad.get("P/S-snitt", 0.0))

        # P/B-snitt
        pbs = [rad.get("P/B Q1", 0), rad.get("P/B Q2", 0), rad.get("P/B Q3", 0), rad.get("P/B Q4", 0)]
        pb_clean = [float(x) for x in pbs if float(x) > 0]
        df.at[i, "P/B-snitt (Q1..Q4)"] = round(np.mean(pb_clean), 2) if pb_clean else float(rad.get("P/B-snitt (Q1..Q4)", 0.0))

        # Dividend yield (%) – beräkna själv (pris-känslig)
        df.at[i, "Dividend yield (%)"] = current_yield_pct(rad)

        # Omsättning om 2/3 år – från "Omsättning nästa år"
        rev2, rev3 = _proj_from_next_year(rad)
        df.at[i, "Omsättning om 2 år"] = rev2
        df.at[i, "Omsättning om 3 år"] = rev3

        # Riktkurser (PS) – ALLTID beräknade (sparas)
        df.at[i, "Riktkurs idag"]    = target_price_ps(df.loc[i], "Idag")
        df.at[i, "Riktkurs om 1 år"] = target_price_ps(df.loc[i], "Om 1 år")
        df.at[i, "Riktkurs om 2 år"] = target_price_ps(df.loc[i], "Om 2 år")
        df.at[i, "Riktkurs om 3 år"] = target_price_ps(df.loc[i], "Om 3 år")
    return df


# ---------------- I/O Sheets ----------------
def _load_df(worksheet_name: str | None) -> pd.DataFrame:
    try:
        ws = get_ws(worksheet_name=worksheet_name)
        df = ws_read_df(ws)
    except Exception as e:
        st.warning(f"🚫 Kunde inte läsa från Google Sheet: {e}")
        df = pd.DataFrame()
    df = ensure_columns(df)
    df = to_numeric(df)
    df = update_calculations(df)
    return df

def _save_df(df: pd.DataFrame, worksheet_name: str | None) -> None:
    df = update_calculations(df.copy())
    try:
        save_dataframe(df, worksheet_name=worksheet_name)
        st.success("Sparat till Google Sheets.")
    except Exception as e:
        st.warning(f"⚠️ Kunde inte spara: {e}")


# ---------------- Scoring (poäng) ----------------
def _norm(val: float, top: float) -> float:
    if top <= 0: return 0.0
    return _clamp(val / top, 0.0, 1.0)

def score_tillvaxt(row: pd.Series, horizon: str) -> float:
    price = float(row.get("Aktuell kurs", 0.0))
    tps = target_price_ps(row, horizon)
    ps_up = ((tps - price) / price * 100.0) if price > 0 and tps > 0 else 0.0
    g = float(row.get("Revenue growth (%)", 0.0))
    gm = float(row.get("Gross margin (%)", 0.0))
    payout = float(row.get("Payout ratio (%)", 0.0))

    s_ps    = _norm(ps_up, 100.0)
    s_g     = _norm(g, 40.0)
    s_gm    = _norm(gm, 60.0)
    penalty = _norm(max(0.0, payout - 80), 40.0)

    score = 100.0 * (0.55*s_ps + 0.25*s_g + 0.15*s_gm + 0.05*(1-penalty))
    return round(score, 1)

def score_utdelning(row: pd.Series, horizon: str) -> float:
    yld = current_yield_pct(row)
    payout = float(row.get("Payout ratio (%)", 0.0))

    base_ratio = DIV_BASE_YIELD / DIV_BASE_PAYOUT  # 15/80 = 0.1875
    p = _clamp(payout, DIV_MIN_PAYOUT, DIV_MAX_PAYOUT)
    ratio = (yld / p) if p > 0 else 0.0
    score_main = 100.0 * _clamp(ratio / base_ratio, 0.0, 1.0)

    # Värdebonus
    price = float(row.get("Aktuell kurs", 0.0))
    t_ps = target_price_ps(row, horizon)
    up_ps = ((t_ps - price) / price * 100.0) if (price > 0 and t_ps > 0) else 0.0
    pb_t = pb_targets_from_row(row)
    up_pb = ((pb_t.get(horizon,0.0) - price) / price * 100.0) if (price>0 and pb_t.get(horizon,0.0)>0) else 0.0
    value_up = max(up_ps, up_pb)
    value_bonus = 10.0 * _norm(value_up, 60.0)

    # Risk-straff för mycket hög payout
    risk_penalty = 10.0 * _norm(max(0.0, payout - 100.0), 50.0)

    score = score_main + value_bonus - risk_penalty
    return round(_clamp(score, 0.0, 100.0), 1)

def score_finans_pb(row: pd.Series, horizon: str) -> float:
    price = float(row.get("Aktuell kurs", 0.0))
    pb_t = pb_targets_from_row(row)
    tpb = pb_t.get(horizon, 0.0)
    pb_up = ((tpb - price) / price * 100.0) if price > 0 and tpb > 0 else 0.0

    nm = float(row.get("Net margin (%)", 0.0))
    payout = float(row.get("Payout ratio (%)", 0.0))

    s_pb    = _norm(pb_up, 80.0)
    s_nm    = _norm(nm, 25.0)
    payout_bonus = 1.0 - _norm(max(0.0, payout - 80), 40.0)

    score = 100.0 * (0.65*s_pb + 0.20*s_nm + 0.15*payout_bonus)
    return round(score, 1)

def infer_mode_from_sector(sector: str) -> str:
    s = (sector or "").lower()
    if any(x in s for x in ["bank", "financial", "insurance", "finans"]):
        return "Finans (P/B)"
    if any(x in s for x in ["utility", "telecom", "real estate", "staples", "försörjning", "fastigheter"]):
        return "Utdelning"
    return "Tillväxt"


# ---------------- Kortvy-render ----------------
def _get_score_for_mode(row: pd.Series, mode: str, horizon: str) -> float:
    m = mode
    if m.startswith("Auto"):
        m = infer_mode_from_sector(row.get("Sektor", ""))
    if m == "Tillväxt":
        return score_tillvaxt(row, horizon)
    elif m == "Utdelning":
        return score_utdelning(row, horizon)
    else:
        return score_finans_pb(row, horizon)

def render_company_card(row: pd.Series, horizon: str, mode: str):
    tkr = row.get("Ticker","")
    namn = row.get("Bolagsnamn","")
    sektor = row.get("Sektor","")
    industri = row.get("Industri","")
    st.subheader(f"{namn} ({tkr})")
    meta = " • ".join([x for x in [f"Sektor: {sektor}" if sektor else "", f"Industri: {industri}" if industri else ""] if x])
    if meta:
        st.caption(meta)

    price = float(row.get("Aktuell kurs", 0.0))
    # PS
    tps_all = ps_targets_all(row)
    ups_ps_all = upsides_from(price, tps_all)
    tp_sel = tps_all.get(horizon, 0.0)
    up_sel = ups_ps_all.get(horizon, 0.0)
    # PB
    pb_targets = pb_targets_from_row(row)
    pb_ups = upsides_from(price, pb_targets)

    yld_calc = current_yield_pct(row)
    payout = float(row.get("Payout ratio (%)", 0.0))
    adj_yield = yld_calc * (DIV_BASE_PAYOUT / payout) if (yld_calc>0 and payout>0) else 0.0
    sund = sund_utdelning_80(row)

    score = _get_score_for_mode(row, mode, horizon)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Aktuell kurs", _fmt_num(price))
        st.write("Valuta:", row.get("Valuta",""))
        st.write("Utest. aktier (M):", _fmt_num(row.get("Utestående aktier (milj.)",0)))
    with c2:
        st.metric(f"Riktkurs ({horizon}) [PS]", _fmt_num(tp_sel))
        st.metric("Uppsida (Vald) [PS]", _fmt_pct(up_sel))
        st.write("P/S-snitt (Q1-4):", _fmt_num(row.get("P/S-snitt",0)))
    with c3:
        st.metric(f"Riktkurs ({horizon}) [PB]", _fmt_num(pb_targets.get(horizon,0)))
        st.metric("Uppsida (Vald) [PB]", _fmt_pct(pb_ups.get(horizon,0)))
        st.write("P/B-snitt (Q1-4):", _fmt_num(row.get("P/B-snitt (Q1..Q4)",0)))
    with c4:
        st.metric("Poäng", _fmt_num(score, 1))
        st.write("P/S (TTM):", _fmt_num(row.get("P/S (TTM)",0)))
        st.write("P/B (nu):", _fmt_num(row.get("P/B",0)))

    with st.expander("📌 Riktkurser & uppsidor (PS & PB)", expanded=True):
        rk_cols = st.columns(4)
        with rk_cols[0]:
            st.write("**PS – Riktkurs**")
            for hz in HORIZONS:
                st.write(f"{hz}:", _fmt_num(tps_all[hz]))
        with rk_cols[1]:
            st.write("**PS – Uppsida**")
            for hz in HORIZONS:
                st.write(f"{hz}:", _fmt_pct(ups_ps_all[hz]))
        with rk_cols[2]:
            st.write("**PB – Riktkurs**")
            for hz in HORIZONS:
                st.write(f"{hz}:", _fmt_num(pb_targets.get(hz,0)))
        with rk_cols[3]:
            st.write("**PB – Uppsida**")
            for hz in HORIZONS:
                st.write(f"{hz}:", _fmt_pct(pb_ups.get(hz,0)))

    with st.expander("💵 Omsättning & tillväxt", expanded=False):
        c = st.columns(4)
        with c[0]: st.write("Idag:", _fmt_num(row.get("Omsättning idag",0)))
        with c[1]: st.write("Nästa år:", _fmt_num(row.get("Omsättning nästa år",0)))
        with c[2]: st.write("Om 2 år:", _fmt_num(row.get("Omsättning om 2 år",0)))
        with c[3]: st.write("Om 3 år:", _fmt_num(row.get("Omsättning om 3 år",0)))
        c2 = st.columns(3)
        with c2[0]: st.write("Revenue TTM (M):", _fmt_num(row.get("Revenue TTM (M)",0)))
        with c2[1]: st.write("Revenue growth:", _fmt_pct(row.get("Revenue growth (%)",0)))
        with c2[2]: st.write("CAGR 5 år:", _fmt_pct(row.get("CAGR 5 år (%)",0)))

    with st.expander("💸 Utdelning", expanded=False):
        c = st.columns(4)
        with c[0]: st.write("Årlig utdelning:", _fmt_num(row.get("Årlig utdelning",0), 4))
        with c[1]: st.write("Dividend yield (ber.):", _fmt_pct(yld_calc))
        with c[2]: st.write("Payout ratio:", _fmt_pct(payout))
        with c[3]: st.write("Just. yield (80%):", _fmt_pct(adj_yield))
        st.write("Sund utd (80%):", _fmt_num(sund, 4))

    with st.expander("📊 Marginaler & värdering", expanded=False):
        c = st.columns(4)
        with c[0]:
            st.write("Gross margin:", _fmt_pct(row.get("Gross margin (%)",0)))
            st.write("Operating margin:", _fmt_pct(row.get("Operating margin (%)",0)))
            st.write("Net margin:", _fmt_pct(row.get("Net margin (%)",0)))
        with c[1]:
            st.write("EV/EBITDA (ttm):", _fmt_num(row.get("EV/EBITDA (ttm)",0)))
            st.write("Book value / share:", _fmt_num(row.get("Book value / share",0)))
            st.write("BVPS CAGR:", _fmt_pct(row.get("BVPS CAGR (%)",0)))
        with c[2]:
            st.write("P/S Q1:", _fmt_num(row.get("P/S Q1",0)))
            st.write("P/S Q2:", _fmt_num(row.get("P/S Q2",0)))
            st.write("P/S Q3:", _fmt_num(row.get("P/S Q3",0)))
            st.write("P/S Q4:", _fmt_num(row.get("P/S Q4",0)))
        with c[3]:
            st.write("P/B Q1:", _fmt_num(row.get("P/B Q1",0)))
            st.write("P/B Q2:", _fmt_num(row.get("P/B Q2",0)))
            st.write("P/B Q3:", _fmt_num(row.get("P/B Q3",0)))
            st.write("P/B Q4:", _fmt_num(row.get("P/B Q4",0)))

    with st.expander("🗣 Stocktwits (om hämtat)", expanded=False):
        st.write("Meddelanden 24h:", row.get("STW meddelanden 24h","—"))
        st.write("Bull 24h:", row.get("STW bull 24h","—"))
        st.write("Bear 24h:", row.get("STW bear 24h","—"))
        st.write("Bull ratio (%):", row.get("STW bull ratio (%)","—"))
        st.write("Bevakare:", row.get("STW bevakare","—"))
        st.write("Medd/h (24h):", row.get("STW meddelanden/h (24h)","—"))
        st.write("Senast:", row.get("STW senast","—"))

    with st.expander("ℹ️ Källor", expanded=False):
        st.write(row.get("Senast uppdaterad källa",""))


# ---------------- Prisuppdatering (alla) ----------------
def update_all_prices(df: pd.DataFrame, worksheet_name: str, delay_sec: float = 0.5) -> pd.DataFrame:
    if df.empty:
        st.warning("Ingen data i vyn.")
        return df

    status = st.empty()
    bar = st.progress(0.0)
    total = len(df)

    for i, (idx, row) in enumerate(df.iterrows()):
        tkr = str(row.get("Ticker", "")).strip()
        if not tkr:
            bar.progress((i+1)/total)
            continue

        try:
            y = yahoo_get(tkr)
            price = float(y.get("price") or 0.0)
            if price:
                df.at[idx, "Aktuell kurs"] = price
            if y.get("name"):       df.at[idx, "Bolagsnamn"] = y["name"]
            if y.get("currency"):   df.at[idx, "Valuta"] = y["currency"]
            if y.get("market_cap"): df.at[idx, "Market Cap"] = float(y["market_cap"])
            if y.get("shares_outstanding"):
                df.at[idx, "Utestående aktier (milj.)"] = float(y["shares_outstanding"]) / 1e6
            if y.get("sector"):     df.at[idx, "Sektor"] = y["sector"]
            if y.get("industry"):   df.at[idx, "Industri"] = y["industry"]
        except Exception as e:
            st.write(f"⚠️ {tkr}: {e}")

        status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
        bar.progress((i+1)/total)
        time.sleep(delay_sec)

    df = update_calculations(df)
    st.session_state["_df_ref"] = df
    _save_df(df, worksheet_name)
    st.success("✅ Aktiekurser + projektioner + riktkurser uppdaterade och sparade.")
    return df


# ---------------- Sidopanel ----------------
with st.sidebar:
    st.header("Google Sheets")
    blad = []
    try:
        blad = list_sheet_names()
    except Exception as e:
        st.info(f"Kunde inte lista blad: {e}")
    default_name = st.secrets.get("WORKSHEET_NAME") or "Blad1"
    idx = blad.index(default_name) if (blad and default_name in blad) else 0
    ws_name = st.selectbox("Välj data-blad:", blad or [default_name], index=idx)

    # --- Valutakurser ---
    st.markdown("---")
    st.subheader("💱 Valutakurser → SEK")

    rates_saved = read_rates()
    pref = st.session_state.get("_live_rates") or rates_saved

    usd = st.number_input("USD → SEK", key="rate_usd", value=float(pref.get("USD", DEFAULT_RATES["USD"])), step=0.0001, format="%.6f")
    nok = st.number_input("NOK → SEK", key="rate_nok", value=float(pref.get("NOK", DEFAULT_RATES["NOK"])), step=0.0001, format="%.6f")
    cad = st.number_input("CAD → SEK", key="rate_cad", value=float(pref.get("CAD", DEFAULT_RATES["CAD"])), step=0.0001, format="%.6f")
    eur = st.number_input("EUR → SEK", key="rate_eur", value=float(pref.get("EUR", DEFAULT_RATES["EUR"])), step=0.0001, format="%.6f")

    def _apply_rates_to_widgets_and_rerun(rates: dict):
        # Uppdatera widget-värden och rerun för att undvika "cannot be modified..."-felet
        st.session_state["rate_usd"] = float(rates.get("USD", DEFAULT_RATES["USD"]))
        st.session_state["rate_nok"] = float(rates.get("NOK", DEFAULT_RATES["NOK"]))
        st.session_state["rate_cad"] = float(rates.get("CAD", DEFAULT_RATES["CAD"]))
        st.session_state["rate_eur"] = float(rates.get("EUR", DEFAULT_RATES["EUR"]))
        st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🌐 Hämta livekurser"):
            try:
                live = fetch_live_rates()
                st.session_state["_live_rates"] = live
                _apply_rates_to_widgets_and_rerun(live)
            except Exception as e:
                st.error(f"Kunde inte hämta livekurser: {e}")
    with c2:
        if st.button("💾 Spara valutakurser"):
            to_save = {
                "USD": float(st.session_state.get("rate_usd", DEFAULT_RATES["USD"])),
                "NOK": float(st.session_state.get("rate_nok", DEFAULT_RATES["NOK"])),
                "CAD": float(st.session_state.get("rate_cad", DEFAULT_RATES["CAD"])),
                "EUR": float(st.session_state.get("rate_eur", DEFAULT_RATES["EUR"])),
                "SEK": 1.0,
            }
            try:
                save_rates(to_save)
                st.success("Valutakurser sparade till Google Sheets.")
            except Exception as e:
                st.error(f"Kunde inte spara kurser: {e}")
    with c3:
        if st.button("↻ Läs sparade kurser"):
            try:
                saved = read_rates()
                st.session_state.pop("_live_rates", None)
                _apply_rates_to_widgets_and_rerun(saved)
            except Exception as e:
                st.error(f"Kunde inte läsa sparade kurser: {e}")

    src = st.session_state.get("_rates_source")
    if src:
        st.caption(f"Källa för livekurser: {src}")

    st.markdown("---")
    # Datakällor vid uppdatering
    use_finvi = st.checkbox("Komplettera med Finviz", value=False, help="Fyller luckor där Yahoo saknar värden.")
    use_mstar = st.checkbox("Komplettera med Morningstar", value=False, help="Fyller kvarvarande luckor.")
    use_sec_pb = st.checkbox("Beräkna P/B 4Q + PB-riktkurs via SEC", value=True, help="Hämtar equity & shares per period från SEC och pris från Yahoo.")
    use_stw   = st.checkbox("Hämta Stocktwits (sentiment)", value=False, help="Meddelanden 24h, bull/bear, watchlist m.m.")

    # Force fresh
    force_fresh = st.checkbox("Ignorera cache (hämta färskt)", value=False)

    if st.button("🔄 Läs in data-bladet"):
        st.session_state["_df_ref"] = _load_df(ws_name)
        st.toast(f"Inläst '{ws_name}'", icon="✅")

    uploaded = st.file_uploader("Importera CSV (ersätter vy)", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state["_df_ref"] = pd.read_csv(uploaded)
            st.success("CSV inläst.")
        except Exception as e:
            st.error(f"Kunde inte läsa CSV: {e}")

    if st.button("💾 Spara vy"):
        _save_df(st.session_state.get("_df_ref", pd.DataFrame()), ws_name)

    st.markdown("---")
    if st.button("📈 Uppdatera aktiekurser (alla)"):
        if force_fresh:
            st.cache_data.clear()
        df0 = st.session_state.get("_df_ref", pd.DataFrame()).copy()
        if df0.empty:
            st.warning("Ingen data i vyn.")
        else:
            update_all_prices(df0, ws_name, delay_sec=0.5)

    if st.button("🚀 Uppdatera ALLA rader (Yahoo + valfria källor)"):
        if force_fresh:
            st.cache_data.clear()

        df0 = st.session_state.get("_df_ref", pd.DataFrame()).copy()
        if df0.empty:
            st.warning("Ingen data i vyn.")
        else:
            status = st.empty()
            bar = st.progress(0.0)
            total = len(df0)

            for i, row in df0.iterrows():
                tkr = str(row.get("Ticker", "")).strip()
                if not tkr:
                    bar.progress((i+1)/total); continue

                used_sources = []

                # --- Yahoo (bas) ---
                try:
                    y = yahoo_get(tkr)
                    used_sources.append("Yahoo")
                    if y.get("name"):       df0.at[i, "Bolagsnamn"] = y["name"]
                    if y.get("currency"):   df0.at[i, "Valuta"] = y["currency"]
                    if y.get("price", 0)>0: df0.at[i, "Aktuell kurs"] = float(y["price"])
                    if y.get("sector"):     df0.at[i, "Sektor"] = y["sector"]
                    if y.get("industry"):   df0.at[i, "Industri"] = y["industry"]

                    sh_out = float(y.get("shares_outstanding") or 0.0)
                    if sh_out > 0: df0.at[i, "Utestående aktier (milj.)"] = sh_out / 1e6
                    if y.get("market_cap", 0)>0: df0.at[i, "Market Cap"] = float(y["market_cap"])

                    for src_key, dst_col in [
                        ("ps_ttm", "P/S (TTM)"),
                        ("pb", "P/B"),
                        ("ev_ebitda", "EV/EBITDA (ttm)"),
                        ("dividend_rate", "Årlig utdelning"),
                        ("dividend_yield_pct", "Dividend yield (%)"),
                        ("payout_ratio_pct", "Payout ratio (%)"),
                        ("book_value_per_share", "Book value / share"),
                        ("gross_margins_pct", "Gross margin (%)"),
                        ("operating_margins_pct", "Operating margin (%)"),
                        ("profit_margins_pct", "Net margin (%)"),
                    ]:
                        v = float(y.get(src_key) or 0.0)
                        if v != 0.0:
                            df0.at[i, dst_col] = v

                    # Rå EV/EBITDA
                    df0.at[i, "_y_ev_now"] = float(y.get("enterprise_value") or 0.0)
                    df0.at[i, "_y_ebitda_now"] = float(y.get("ebitda") or 0.0)

                    rev_ttm = float(y.get("revenue_ttm") or 0.0)
                    if rev_ttm > 0:
                        df0.at[i, "Revenue TTM (M)"] = rev_ttm / 1e6
                    df0.at[i, "Revenue growth (%)"] = float(y.get("revenue_growth_pct") or 0.0)
                    df0.at[i, "CAGR 5 år (%)"] = float(y.get("cagr5_pct") or 0.0)
                except Exception as e:
                    st.write(f"Yahoo misslyckades för {tkr}: {e}")

                # --- Finviz (fyll luckor) ---
                if use_finvi:
                    try:
                        fv = finviz_get(tkr)
                        used_sources.append("Finviz")
                        def fill(col, key):
                            cur = float(df0.at[i, col]) if col in df0.columns else 0.0
                            val = float(fv.get(key) or 0.0)
                            if (cur == 0.0) and (val != 0.0):
                                df0.at[i, col] = val
                        fill("Aktuell kurs", "price")
                        fill("P/S (TTM)", "ps_ttm")
                        fill("P/B", "pb")
                        fill("Dividend yield (%)", "dividend_yield_pct")
                        fill("Payout ratio (%)", "payout_ratio_pct")
                        fill("Market Cap", "market_cap")
                        if float(df0.at[i, "Utestående aktier (milj.)"] or 0.0) == 0.0 and fv.get("shares_outstanding"):
                            df0.at[i, "Utestående aktier (milj.)"] = float(fv["shares_outstanding"]) / 1e6
                        fill("Gross margin (%)", "gross_margins_pct")
                        fill("Operating margin (%)", "operating_margins_pct")
                        fill("Net margin (%)", "profit_margins_pct")
                    except Exception as e:
                        st.write(f"Finviz misslyckades för {tkr}: {e}")

                # --- Morningstar (fyll kvarvarande luckor) ---
                if use_mstar:
                    try:
                        ms = ms_get(tkr)
                        used_sources.append("Morningstar")
                        def fill(col, key):
                            cur = float(df0.at[i, col]) if col in df0.columns else 0.0
                            val = float(ms.get(key) or 0.0)
                            if (cur == 0.0) and (val != 0.0):
                                df0.at[i, col] = val
                        fill("Aktuell kurs", "price")
                        fill("P/S (TTM)", "ps_ttm")
                        fill("P/B", "pb")
                        fill("Dividend yield (%)", "dividend_yield_pct")
                        fill("Payout ratio (%)", "payout_ratio_pct")
                        fill("Market Cap", "market_cap")
                        if float(df0.at[i, "Utestående aktier (milj.)"] or 0.0) == 0.0 and ms.get("shares_outstanding"):
                            df0.at[i, "Utestående aktier (milj.)"] = float(ms["shares_outstanding"]) / 1e6
                        fill("Book value / share", "book_value_per_share")
                        fill("Gross margin (%)", "gross_margins_pct")
                        fill("Operating margin (%)", "operating_margins_pct")
                        fill("Net margin (%)", "profit_margins_pct")
                    except Exception as e:
                        st.write(f"Morningstar misslyckades för {tkr}: {e}")

                # --- SEC P/B historik + PB-riktkurser (valfritt) ---
                if use_sec_pb:
                    try:
                        pbdata = get_pb_quarters(tkr)
                        pbs = pbdata.get("pb_quarters", [])
                        details = pbdata.get("details", []) or []

                        # P/B kvartal (4 senaste → Q1..Q4)
                        p_values = [float(x[1]) for x in pbs]
                        q = [0.0, 0.0, 0.0, 0.0]
                        for idx_, val in enumerate(reversed(p_values[-4:])):
                            q[idx_] = round(val, 2)
                        df0.at[i, "P/B Q1"] = q[0]
                        df0.at[i, "P/B Q2"] = q[1]
                        df0.at[i, "P/B Q3"] = q[2]
                        df0.at[i, "P/B Q4"] = q[3]

                        # BVPS0 (nu) från SEC-details senaste, annars från Book value / share
                        bvps0 = 0.0
                        if details:
                            latest = max(details, key=lambda d: d.get("date",""))
                            bvps0 = float(latest.get("bvps") or 0.0)
                        if bvps0 <= 0:
                            bvps0 = float(df0.at[i, "Book value / share"]) if "Book value / share" in df0.columns else 0.0

                        # BVPS-CAGR (%)
                        g_bv = _bvps_cagr_from_sec_details(details)  # andel
                        if g_bv != 0:
                            df0.at[i, "BVPS CAGR (%)"] = round(g_bv * 100.0, 2)

                        # PB-snitt och PB-riktkurser (Idag/1y/2y/3y)
                        pb_avg = _pb_avg(df0.loc[i])
                        pb_tps = pb_targets_all_from_bvps(pb_avg, bvps0, g_bv)
                        df0.at[i, "PB Riktkurs idag"]    = pb_tps["Idag"]
                        df0.at[i, "PB Riktkurs om 1 år"] = pb_tps["Om 1 år"]
                        df0.at[i, "PB Riktkurs om 2 år"] = pb_tps["Om 2 år"]
                        df0.at[i, "PB Riktkurs om 3 år"] = pb_tps["Om 3 år"]

                        used_sources.append("SEC")
                    except Exception as e:
                        st.write(f"SEC/PB misslyckades för {tkr}: {e}")

                # --- Stocktwits (valfritt) ---
                if use_stw:
                    try:
                        stw = stw_get(tkr)
                        used_sources.append("Stocktwits")
                        df0.at[i, "STW meddelanden 24h"] = int(stw.get("stw_messages_24h", 0))
                        df0.at[i, "STW bull 24h"] = int(stw.get("stw_bull_24h", 0))
                        df0.at[i, "STW bear 24h"] = int(stw.get("stw_bear_24h", 0))
                        df0.at[i, "STW bull ratio (%)"] = float(stw.get("stw_bull_ratio", 0.0))
                        df0.at[i, "STW bevakare"] = int(stw.get("stw_watchlist_count", 0))
                        df0.at[i, "STW meddelanden/h (24h)"] = float(stw.get("stw_avg_msgs_per_hour_24h", 0.0))
                        df0.at[i, "STW senast"] = stw.get("last_message_at", "")
                    except Exception as e:
                        st.write(f"Stocktwits misslyckades för {tkr}: {e}")

                df0.at[i, "Senast uppdaterad källa"] = ", ".join(dict.fromkeys(used_sources))  # unika i ordning

                status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
                bar.progress((i+1)/total)
                time.sleep(0.5)

            df0 = update_calculations(df0)
            st.session_state["_df_ref"] = df0
            _save_df(df0, ws_name)


# ---------------- Första läsning ----------------
if "_df_ref" not in st.session_state:
    st.session_state["_df_ref"] = _load_df(st.secrets.get("WORKSHEET_NAME") or "Blad1")


# ---------------- Flikar ----------------
tab_data, tab_collect, tab_port, tab_suggest = st.tabs(["📄 Data", "🧩 Manuell insamling", "📦 Portfölj", "💡 Köpförslag"])

with tab_data:
    df = st.session_state.get("_df_ref", pd.DataFrame())
    if df.empty:
        st.info("Ingen data att visa.")
    else:
        st.dataframe(df, use_container_width=True)

with tab_collect:
    if manual_collect_view is None:
        st.info("Insamlingsvyn är inte aktiverad i denna bas.")
    else:
        df_in = st.session_state.get("_df_ref", pd.DataFrame())
        df_out = manual_collect_view(df_in)
        if isinstance(df_out, pd.DataFrame) and not df_out.equals(df_in):
            st.session_state["_df_ref"] = update_calculations(df_out.copy())
            st.success("Uppdaterade sessionens data (inkl. projektioner + riktkurser).")

with tab_port:
    df = st.session_state.get("_df_ref", pd.DataFrame())
    if df.empty:
        st.info("Ingen data.")
    else:
        saved_for_fallback = read_rates()
        rates = {
            "USD": float(st.session_state.get("rate_usd", saved_for_fallback.get("USD", DEFAULT_RATES["USD"]))),
            "NOK": float(st.session_state.get("rate_nok", saved_for_fallback.get("NOK", DEFAULT_RATES["NOK"]))),
            "CAD": float(st.session_state.get("rate_cad", saved_for_fallback.get("CAD", DEFAULT_RATES["CAD"]))),
            "EUR": float(st.session_state.get("rate_eur", saved_for_fallback.get("EUR", DEFAULT_RATES["EUR"]))),
            "SEK": 1.0,
        }
        port = df[df["Antal aktier"] > 0].copy()
        if port.empty:
            st.info("Du äger inga aktier.")
        else:
            port["Växelkurs"] = port["Valuta"].apply(lambda v: rates.get(str(v).upper(), 1.0))
            port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
            total_v = float(port["Värde (SEK)"].sum())
            port["Andel (%)"] = (port["Värde (SEK)"] / total_v * 100.0).round(2) if total_v > 0 else 0.0
            port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
            tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

            st.markdown(f"**Totalt portföljvärde:** {round(total_v,2)} SEK")
            st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
            st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

            st.dataframe(
                port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
                      "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
                use_container_width=True
            )

with tab_suggest:
    df = st.session_state.get("_df_ref", pd.DataFrame())
    if df.empty:
        st.info("Ingen data.")
    else:
        st.subheader("Regelstyrda köpförslag")

        c1, c2, c3 = st.columns(3)
        mode = c1.selectbox("Typ", ["Auto (bransch)", "Tillväxt", "Utdelning", "Finans (P/B)"])
        horizon = c2.selectbox("Riktkurs baseras på", HORIZONS, index=0)  # standard: Idag
        min_score = c3.slider("Min. Poäng (endast Poäng-läget)", 0, 100, 60, 1)

        # Sektorfilter och endast innehav
        if "Sektor" in df.columns:
            sectors_all = sorted([s for s in df["Sektor"].astype(str).unique() if s and s != "nan"])
            selected_sectors = st.multiselect("Filtrera sektor(er)", sectors_all, default=sectors_all)
        else:
            selected_sectors = []
        only_holdings = st.checkbox("Visa endast innehav (Antal aktier > 0)", value=False)

        base = df.copy()
        if selected_sectors and "Sektor" in base.columns:
            base = base[base["Sektor"].isin(selected_sectors)].copy()
        if only_holdings and "Antal aktier" in base.columns:
            base = base[base["Antal aktier"] > 0].copy()

        rows = []
        for _, r in base.iterrows():
            price = float(r.get("Aktuell kurs", 0.0))

            # PS
            tps = ps_targets_all(r)
            ups_ps = upsides_from(price, tps)
            # PB
            pb_t = pb_targets_from_row(r)
            pb_ups = upsides_from(price, pb_t)

            tp_sel_ps = tps.get(horizon, 0.0)
            up_sel_ps = ups_ps.get(horizon, 0.0)
            tp_sel_pb = pb_t.get(horizon, 0.0)
            up_sel_pb = pb_ups.get(horizon, 0.0)

            payout = float(r.get("Payout ratio (%)", 0.0))
            yld_calc = current_yield_pct(r)
            adj_yield_80 = round(yld_calc * (DIV_BASE_PAYOUT / payout), 2) if (yld_calc>0 and payout>0) else 0.0

            m = mode
            if m.startswith("Auto"):
                m = infer_mode_from_sector(r.get("Sektor", ""))

            if m == "Tillväxt":
                score = score_tillvaxt(r, horizon)
            elif m == "Utdelning":
                score = score_utdelning(r, horizon)
            else:
                score = score_finans_pb(r, horizon)

            rows.append({
                "Typ": m,
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Sektor": r.get("Sektor","") if "Sektor" in r else "",
                "Valuta": r.get("Valuta",""),
                "Aktuell kurs": price,

                f"Riktkurs (Vald={horizon}) [PS]": tp_sel_ps,
                "Uppsida (Vald) [PS] %": up_sel_ps,
                f"Riktkurs (Vald={horizon}) [PB]": tp_sel_pb,
                "Uppsida (Vald) [PB] %": up_sel_pb,

                "Riktkurs (Idag) [PS]": tps["Idag"],
                "Riktkurs (Om 1 år) [PS]": tps["Om 1 år"],
                "Riktkurs (Om 2 år) [PS]": tps["Om 2 år"],
                "Riktkurs (Om 3 år) [PS]": tps["Om 3 år"],

                "Riktkurs (Idag) [PB]": pb_t["Idag"],
                "Riktkurs (Om 1 år) [PB]": pb_t["Om 1 år"],
                "Riktkurs (Om 2 år) [PB]": pb_t["Om 2 år"],
                "Riktkurs (Om 3 år) [PB]": pb_t["Om 3 år"],

                "Dividend yield (%)": yld_calc,
                "Payout ratio (%)": payout,
                "Just. yield (80%)": adj_yield_80,
                "Sund utd (80%)": sund_utdelning_80(r),

                "Revenue growth (%)": float(r.get("Revenue growth (%)", 0.0)),
                "Gross margin (%)": float(r.get("Gross margin (%)", 0.0)),
                "Net margin (%)": float(r.get("Net margin (%)", 0.0)),

                "Poäng": score,
            })

        prop = pd.DataFrame(rows)
        if prop.empty:
            st.info("Inga förslag – saknas data för urvalet.")
        else:
            mode_filter = st.radio("Filtrera på", ["Poäng", "Endast riktkurs"], horizontal=True)

            if mode_filter == "Poäng":
                prop_f = prop[prop["Poäng"] >= min_score].copy()
                prop_f = prop_f.sort_values(by=["Poäng", "Uppsida (Vald) [PS] %"], ascending=[False, False]).reset_index(drop=True)
            else:
                st.markdown("**Riktkurs-filter (standard: Idag)**")
                rk_basis = st.selectbox("Riktkurs-basis", ["PS", "PB"], index=0)
                max_up = st.slider("Max uppsida (%)", 0.0, 100.0, 5.0, 0.5)
                max_price = st.number_input("Max aktuell kurs (valfritt)", value=0.0, step=1.0)
                ups_col = ("Uppsida (Vald) [PS] %" if rk_basis == "PS" else "Uppsida (Vald) [PB] %")
                prop_f = prop[prop[ups_col] <= max_up].copy()
                if max_price and max_price > 0:
                    prop_f = prop_f[prop_f["Aktuell kurs"] <= max_price]
                prop_f = prop_f.sort_values(by=[ups_col], ascending=True).reset_index(drop=True)
                if "Poäng" in prop_f.columns:
                    prop_f = prop_f.drop(columns=["Poäng"])

            st.caption(f"Visar {len(prop_f)}/{len(prop)} förslag efter filter.")
            view_mode = st.radio("Visningsläge", ["Tabell", "Kortvy (1 i taget)"], horizontal=True, index=0)

            if view_mode == "Tabell":
                st.dataframe(prop_f, use_container_width=True)
            else:
                if prop_f.empty:
                    st.info("Inget att visa i kortvy.")
                else:
                    tickers_order = list(prop_f["Ticker"].astype(str).values)
                    if "suggest_idx" not in st.session_state:
                        st.session_state["suggest_idx"] = 0
                    st.session_state["suggest_idx"] = min(st.session_state["suggest_idx"], len(tickers_order)-1)

                    col_nav = st.columns([1,2,1])
                    with col_nav[0]:
                        if st.button("⬅️ Föregående"):
                            st.session_state["suggest_idx"] = max(0, st.session_state["suggest_idx"] - 1)
                    with col_nav[1]:
                        st.write(f"Post {st.session_state['suggest_idx']+1} / {len(tickers_order)}")
                    with col_nav[2]:
                        if st.button("➡️ Nästa"):
                            st.session_state["suggest_idx"] = min(len(tickers_order)-1, st.session_state["suggest_idx"] + 1)

                    sel_tkr = tickers_order[st.session_state["suggest_idx"]]
                    base_df = st.session_state.get("_df_ref", pd.DataFrame())
                    row = base_df[base_df["Ticker"].astype(str) == sel_tkr]
                    if row.empty:
                        st.warning("Kunde inte hitta raden i huvudtabellen.")
                    else:
                        render_company_card(row.iloc[0], horizon, mode)

                st.markdown("---")
                st.dataframe(prop_f.head(50), use_container_width=True)  # mini-lista

            st.markdown("---")
            csave, cdl = st.columns([1,1])
            with csave:
                if st.button("💾 Spara förslag till Google Sheet"):
                    try:
                        sheet_name = f"Förslag-{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"
                        save_dataframe(prop_f, worksheet_name=sheet_name)
                        st.success(f"Sparat i blad: **{sheet_name}**")
                    except Exception as e:
                        st.error(f"Kunde inte spara förslagen: {e}")
            with cdl:
                csv_bytes = prop_f.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ Ladda ned CSV", data=csv_bytes, file_name="förslag.csv", mime="text/csv")
