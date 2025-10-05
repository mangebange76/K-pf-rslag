# app.py — Del 1/6
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --- Google Sheets-hjälpare (våra wrappers i stockapp/) ---
from stockapp.sheets import (
    ws_read_df, ws_write_df, list_worksheet_titles, delete_worksheet
)

# --- Valutakurser (valfri modul). Faller tillbaka till defaults om import misslyckas. ---
DEFAULT_RATES: Dict[str, float] = {"USD": 10.0, "NOK": 1.0, "CAD": 7.0, "EUR": 11.0, "SEK": 1.0}
RATES_OK = False
try:
    from stockapp.rates import (
        read_rates, save_rates, fetch_live_rates, repair_rates_sheet, DEFAULT_RATES as _DEF_RATES
    )
    # använd defaults från modulen om de finns
    if isinstance(_DEF_RATES, dict):
        DEFAULT_RATES.update({k: float(v) for k, v in _DEF_RATES.items() if k in DEFAULT_RATES})
    RATES_OK = True
except Exception:
    # dummy-funktioner när rate-modulen saknas
    def read_rates() -> Dict[str, float]:
        return DEFAULT_RATES.copy()
    def save_rates(x: Dict[str, float]):  # no-op
        pass
    def fetch_live_rates() -> Dict[str, float]:
        return DEFAULT_RATES.copy()
    def repair_rates_sheet():
        pass

st.set_page_config(page_title="K-pf-rslag", layout="wide")

# ---------- Tids- & parsinghjälp ----------
def _now_sthlm() -> datetime:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz)
    except Exception:
        return datetime.now()

def now_stamp() -> str:
    return _now_sthlm().strftime("%Y-%m-%d")

def _to_float(x) -> float:
    """Tål både komma- och punkt-decimaler, tomma strängar och None."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return 0.0
    # ta bort mellanslag och icke-brytande mellanslag
    s = s.replace("\u00A0", "").replace(" ", "")
    # om både punkt och komma finns: anta format som "1 234,56" → ta bort punkt, ersätt komma
    if "," in s and "." in s:
        # välj sista symbolen som decimal; ta bort den andra
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

# ---------- Kolumnschema (KANONISKA namn som appen arbetar med) ----------
# Basen du har i Google Sheet (vi mappar alias → dessa namn)
BASE_COLS: List[str] = [
    "Ticker",
    "Antal aktier",          # alias: "Antal Aktier"
    "GAV (SEK)",
    "Omsättning idag",       # alias: "Omsättning i år"
    "Omsättning nästa år",
]

# Fält som hämtas/kan fyllas automatiskt
AUTO_COLS: List[str] = [
    "Bolagsnamn",
    "Sektor",
    "Valuta",
    "Aktuell kurs",
    "Årlig utdelning",
    "CAGR 5 år (%)",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/B", "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4",
    "Utestående aktier",  # miljoner
]

# Beräknade fält
CALC_COLS: List[str] = [
    "P/S-snitt (Q1..Q4)", "P/B-snitt (Q1..Q4)",
    "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "DA (%)",
]

# Tidsstämplar
TS_COLS: List[str] = [
    "Senast manuellt uppdaterad",
    "Senast auto uppdaterad",
    "Auto källa",
    "Senast beräknad",
]

ALL_COLS: List[str] = BASE_COLS + AUTO_COLS + CALC_COLS + TS_COLS

# Datatyper
NUMERIC_COLS = {
    "Antal aktier","GAV (SEK)","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)",
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4",
    "Utestående aktier",
    "P/S-snitt (Q1..Q4)","P/B-snitt (Q1..Q4)",
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
    "DA (%)",
}
STR_COLS = {"Ticker","Bolagsnamn","Sektor","Valuta","Auto källa",
            "Senast manuellt uppdaterad","Senast auto uppdaterad","Senast beräknad"}

# Alias (så att rubriker i arket får funka direkt)
ALIASES = {
    "Antal Aktier": "Antal aktier",
    "Omsättning i år": "Omsättning idag",
}

# app.py — Del 2/6

# ---------- Normalisering & kolumnsäkring ----------

def _rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Byt rubriker i arket till våra kanoniska namn."""
    ren = {}
    for k, v in ALIASES.items():
        if k in df.columns and v not in df.columns:
            ren[k] = v
    if ren:
        df = df.rename(columns=ren)
    return df

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Se till att alla kolumner finns och att typer är rimliga."""
    df = _rename_aliases(df).copy()

    # lägg till saknade kolumner
    for c in ALL_COLS:
        if c not in df.columns:
            df[c] = "" if c in STR_COLS else 0.0

    # typkonvertering
    for c in df.columns:
        if c in NUMERIC_COLS:
            df[c] = df[c].apply(_to_float).astype(float)
        elif c in STR_COLS:
            df[c] = df[c].astype(str)
        else:
            # okända kolumner rör vi inte – men de kan följa med i visningar
            pass

    # säkerställ ordning (våra kända kolumner först)
    known = [c for c in ALL_COLS if c in df.columns]
    other = [c for c in df.columns if c not in ALL_COLS]
    return df[known + other]


# ---------- Cachead IO mot Google Sheets ----------

@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str, _nonce: int) -> pd.DataFrame:
    """Läs rådata från Google Sheets. _nonce bustar cachen efter sparning."""
    raw = ws_read_df(ws_title)          # läser rubrikrad + värden
    if raw is None or raw.empty:
        return pd.DataFrame(columns=ALL_COLS)
    return ensure_columns(raw)

def load_df(ws_title: str) -> pd.DataFrame:
    n = st.session_state.get("_reload_nonce", 0)
    return load_df_cached(ws_title, n)

def save_df(ws_title: str, df: pd.DataFrame, *, bust_cache: bool = True):
    ws_write_df(ws_title, ensure_columns(df))
    if bust_cache:
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1


# ---------- Sidopanel: valutakurser ----------

def sidebar_rates() -> Dict[str, float]:
    st.sidebar.subheader("💱 Valutakurser → SEK")

    # startvärden (läs sparade om möjligt)
    if "rates_loaded" not in st.session_state:
        try:
            saved = read_rates() if RATES_OK else DEFAULT_RATES
        except Exception:
            saved = DEFAULT_RATES
        for k in ["USD","NOK","CAD","EUR"]:
            st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
        st.session_state["rates_loaded"] = True

    colA, colB = st.sidebar.columns(2)
    if colA.button("🌐 Hämta live"):
        try:
            live = fetch_live_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(live.get(k, st.session_state[f"rate_{k.lower()}"]))
            st.sidebar.success("Livekurser hämtade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta livekurser: {e}")

    if colB.button("↻ Läs sparade"):
        try:
            saved = read_rates()
            for k in ["USD","NOK","CAD","EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(saved.get(k, DEFAULT_RATES[k]))
            st.sidebar.success("Sparade kurser inlästa.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte läsa sparade kurser: {e}")

    usd = st.sidebar.number_input("USD → SEK", key="rate_usd", step=0.0001, format="%.6f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok", step=0.0001, format="%.6f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad", step=0.0001, format="%.6f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur", step=0.0001, format="%.6f")

    colC, colD = st.sidebar.columns(2)
    if colC.button("💾 Spara kurser"):
        try:
            save_rates({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
            st.sidebar.success("Valutakurser sparade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")

    if colD.button("🛠 Reparera bladet"):
        try:
            repair_rates_sheet()
            st.sidebar.success("Bladet reparerat.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte reparera: {e}")

    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": 1.0}

# app.py — Del 3/6
# ---------- Beräkningar & snabbhämtning ----------

def clamp(v: float, lo: float, hi: float) -> float:
    try:
        v = float(v)
    except Exception:
        v = 0.0
    return max(lo, min(hi, v))

def compute_ps_pb_snitt(row: pd.Series) -> Tuple[float, float]:
    """Snitt av positiva Q1..Q4 för P/S och P/B."""
    def _avg(cols: List[str]) -> float:
        vals = []
        for c in cols:
            vals.append(_to_float(row.get(c, 0)))
        vals = [x for x in vals if x > 0]
        return round(float(np.mean(vals)), 2) if vals else 0.0

    ps_avg = _avg(["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"])
    pb_avg = _avg(["P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4"])
    return ps_avg, pb_avg

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """Härled snitt, framtida omsättning, riktkurser osv."""
    if df is None or df.empty:
        return df

    out = df.copy()

    for i, r in out.iterrows():
        # Snitt
        ps_avg, pb_avg = compute_ps_pb_snitt(r)
        out.at[i, "P/S-snitt (Q1..Q4)"] = ps_avg
        out.at[i, "P/B-snitt (Q1..Q4)"] = pb_avg

        # CAGR clamp 2..50%
        cagr = _to_float(r.get("CAGR 5 år (%)", 0.0))
        g = clamp(cagr, 2.0, 50.0) / 100.0

        # Omsättning år 2–3
        next_rev = _to_float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            out.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            out.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)
        else:
            out.at[i, "Omsättning om 2 år"] = _to_float(r.get("Omsättning om 2 år", 0.0))
            out.at[i, "Omsättning om 3 år"] = _to_float(r.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver utestående aktier (M) > 0 och ps-snitt > 0)
        shares_m = _to_float(r.get("Utestående aktier", 0.0))
        psn = _to_float(out.at[i, "P/S-snitt (Q1..Q4)"])
        if shares_m > 0 and psn > 0:
            out.at[i, "Riktkurs idag"]    = round(_to_float(r.get("Omsättning idag", 0.0))     * psn / shares_m, 2)
            out.at[i, "Riktkurs om 1 år"] = round(_to_float(r.get("Omsättning nästa år", 0.0)) * psn / shares_m, 2)
            out.at[i, "Riktkurs om 2 år"] = round(_to_float(out.at[i, "Omsättning om 2 år"])   * psn / shares_m, 2)
            out.at[i, "Riktkurs om 3 år"] = round(_to_float(out.at[i, "Omsättning om 3 år"])   * psn / shares_m, 2)
        else:
            out.at[i, "Riktkurs idag"] = 0.0
            out.at[i, "Riktkurs om 1 år"] = 0.0
            out.at[i, "Riktkurs om 2 år"] = 0.0
            out.at[i, "Riktkurs om 3 år"] = 0.0

    return out

def add_multi_uppsida(df: pd.DataFrame) -> pd.DataFrame:
    """Beräkna uppsida (%) och DA (%) från aktuell kurs."""
    out = df.copy()
    price = out["Aktuell kurs"].map(_to_float).replace(0, np.nan)
    for col, label in [
        ("Riktkurs idag", "Uppsida idag (%)"),
        ("Riktkurs om 1 år", "Uppsida 1 år (%)"),
        ("Riktkurs om 2 år", "Uppsida 2 år (%)"),
        ("Riktkurs om 3 år", "Uppsida 3 år (%)"),
    ]:
        rk = out[col].map(_to_float).replace(0, np.nan)
        out[label] = ((rk - price) / price * 100.0).fillna(0.0)

    out["DA (%)"] = np.where(
        out["Aktuell kurs"].map(_to_float) > 0,
        (out["Årlig utdelning"].map(_to_float) / out["Aktuell kurs"].map(_to_float)) * 100.0,
        0.0
    )
    return out

def _horizon_to_tag(h: str) -> str:
    if "om 1 år" in h: return "1 år"
    if "om 2 år" in h: return "2 år"
    if "om 3 år" in h: return "3 år"
    return "Idag"

def score_rows(df: pd.DataFrame, *, horizon: str, strategy: str) -> pd.DataFrame:
    """En enkel scoring (Growth/Dividend/Financials/Total) + Confidence."""
    out = df.copy()
    # Normaliserade fält
    out["DA (%)"] = np.where(
        out["Aktuell kurs"].map(_to_float) > 0,
        (out["Årlig utdelning"].map(_to_float) / out["Aktuell kurs"].map(_to_float)) * 100.0,
        0.0
    )
    out["Uppsida (%)"] = np.where(
        out["Aktuell kurs"].map(_to_float) > 0,
        (out[horizon].map(_to_float) - out["Aktuell kurs"].map(_to_float)) / out["Aktuell kurs"].map(_to_float) * 100.0,
        0.0
    )

    # Growth
    cur_ps = out["P/S"].map(_to_float).replace(0, np.nan)
    ps_avg = out["P/S-snitt (Q1..Q4)"].map(_to_float).replace(0, np.nan)
    cheap_ps = (ps_avg / (cur_ps * 2.0)).clip(upper=1.0).fillna(0.0)
    g_norm = (out["CAGR 5 år (%)"].map(_to_float) / 30.0).clip(0, 1)
    u_norm = (out["Uppsida (%)"] / 50.0).clip(0, 1)
    out["Score (Growth)"] = (0.4 * g_norm + 0.4 * u_norm + 0.2 * cheap_ps) * 100.0

    # Dividend
    payout = out["Payout (%)"].map(_to_float)
    payout_health = 1 - (abs(payout - 60.0) / 60.0)
    payout_health = payout_health.clip(0, 1)
    payout_health = np.where(payout <= 0, 0.85, payout_health)
    y_norm = (out["DA (%)"] / 8.0).clip(0, 1)
    grow_ok = np.where(out["CAGR 5 år (%)"].map(_to_float) >= 0, 1.0, 0.6)
    out["Score (Dividend)"] = (0.6 * y_norm + 0.3 * payout_health + 0.1 * grow_ok) * 100.0

    # Financials
    cur_pb = out["P/B"].map(_to_float).replace(0, np.nan)
    pb_avg = out["P/B-snitt (Q1..Q4)"].map(_to_float).replace(0, np.nan)
    cheap_pb = (pb_avg / (cur_pb * 2.0)).clip(upper=1.0).fillna(0.0)
    out["Score (Financials)"] = (0.7 * cheap_pb + 0.3 * u_norm) * 100.0

    # Vikter per strategi
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
        wg, wd, wf = weights_for_row(r.get("Sektor", ""), "Auto" if strategy.startswith("Auto") else strategy)
        Wg.append(wg); Wd.append(wd); Wf.append(wf)

    Wg = np.array(Wg); Wd = np.array(Wd); Wf = np.array(Wf)
    out["Score (Total)"] = (Wg*out["Score (Growth)"] + Wd*out["Score (Dividend)"] + Wf*out["Score (Financials)"]).round(2)

    # Confidence = hur mycket kärndata som finns
    need = [
        out["Aktuell kurs"].map(_to_float) > 0,
        out["P/S-snitt (Q1..Q4)"].map(_to_float) > 0,
        out["Omsättning idag"].map(_to_float) >= 0,
        out["Omsättning nästa år"].map(_to_float) >= 0,
    ]
    present = np.stack(need, axis=0).astype(float)
    out["Confidence"] = (present.mean(axis=0) * 100.0).round(0)

    return out

def compute_scores_all_horizons(df: pd.DataFrame, *, strategy: str) -> pd.DataFrame:
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

def enrich_for_save(df: pd.DataFrame, *, horizon_for_score: str = "Riktkurs idag", strategy: str = "Auto") -> pd.DataFrame:
    df2 = update_calculations(df)
    df2 = add_multi_uppsida(df2)
    df2 = score_rows(df2, horizon=horizon_for_score, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    df2 = compute_scores_all_horizons(df2, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    df2["Senast beräknad"] = today_stamp()
    return df2


# ---------- Snabb Yahoo (pris, namn, valuta, utd, CAGR) ----------

@st.cache_data(show_spinner=False, ttl=600)
def yahoo_fetch_one_quick(ticker: str) -> Dict[str, float | str]:
    out = {"Bolagsnamn":"", "Valuta":"USD", "Aktuell kurs":0.0, "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0}
    if not ticker:
        return out
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        nm = info.get("shortName") or info.get("longName")
        if nm: out["Bolagsnamn"] = str(nm)

        cur = info.get("currency")
        if cur: out["Valuta"] = str(cur).upper()

        px = info.get("regularMarketPrice")
        if px is None:
            h = t.history(period="1d")
            if isinstance(h, pd.DataFrame) and not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        if px is not None:
            out["Aktuell kurs"] = float(px)

        dr = info.get("dividendRate")
        if dr is not None:
            out["Årlig utdelning"] = float(dr)

        # CAGR 5y från Total Revenue (årsdata)
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

def _m_tag(row: pd.Series) -> str:
    d = str(row.get("Senast manuellt uppdaterad","")).strip()
    return f"〔M: {d or '—'}〕"

def _a_tag(row: pd.Series) -> str:
    d = str(row.get("Senast auto uppdaterad","")).strip()
    src = str(row.get("Auto källa","")).strip()
    return f"〔A: {d or '—'}{(' · '+src) if src else ''}〕"

def _b_tag(row: pd.Series) -> str:
    d = str(row.get("Senast beräknad","")).strip()
    return f"〔B: {d or '—'}〕"


# ---------- Små tabeller: äldst uppdaterade ----------

def _oldest_tables(df: pd.DataFrame):
    st.markdown("### ⏱️ Äldst uppdaterade")

    def _to_dt(s: str) -> Optional[pd.Timestamp]:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return pd.to_datetime(s)
        except Exception:
            return None

    tmp = df.copy()
    tmp["d_man"]  = tmp["Senast manuellt uppdaterad"].apply(_to_dt)
    tmp["d_auto"] = tmp["Senast auto uppdaterad"].apply(_to_dt)
    tmp["d_any"]  = tmp[["d_man","d_auto"]].min(axis=1)

    any_sorted = tmp.dropna(subset=["d_any"]).sort_values("d_any", ascending=True)
    if any_sorted.empty:
        st.info("Inga tidsstämplar ännu.")
    else:
        st.dataframe(any_sorted.head(10)[["Ticker","Bolagsnamn","d_any"]].rename(columns={"d_any":"Äldst (valfri)"}), use_container_width=True)

    man_sorted = tmp.dropna(subset=["d_man"]).sort_values("d_man", ascending=True)
    if not man_sorted.empty:
        st.dataframe(man_sorted.head(10)[["Ticker","Bolagsnamn","d_man"]].rename(columns={"d_man":"Äldst (manuell)"}), use_container_width=True)

    auto_sorted = tmp.dropna(subset=["d_auto"]).sort_values("d_auto", ascending=True)
    if not auto_sorted.empty:
        st.dataframe(auto_sorted.head(10)[["Ticker","Bolagsnamn","d_auto","Auto källa"]].rename(columns={"d_auto":"Äldst (auto)"}), use_container_width=True)

# app.py — Del 4/6
# ---------- Enskild full uppdatering (alla fetchers) ----------

def _update_one_all_fetchers(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Uppdatera vald ticker via Yahoo/Finviz/Morningstar/SEC (om tillgängligt)."""
    tkr = (ticker or "").strip().upper()
    if not tkr or df.empty:
        return df

    mask = (df["Ticker"].astype(str).str.upper() == tkr)
    if not mask.any():
        return df

    # Yahoo – bred översikt
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
                elif isinstance(v, (int, float)):
                    df.loc[mask, k_dst] = float(v)
                else:
                    df.loc[mask, k_dst] = str(v)
            df.loc[mask, "Senast auto uppdaterad"] = today_stamp()
            df.loc[mask, "Auto källa"] = "Yahoo"
        except Exception:
            pass

    # Finviz
    if fz_overview:
        try:
            fz = fz_overview(tkr) or {}
            if _to_float(fz.get("ps_ttm", 0.0)) > 0:
                df.loc[mask, "P/S"] = float(fz["ps_ttm"])
            if _to_float(fz.get("pb", 0.0)) > 0:
                df.loc[mask, "P/B"] = float(fz["pb"])
            if _to_float(fz.get("price", 0.0)) > 0:
                df.loc[mask, "Aktuell kurs"] = float(fz["price"])
            df.loc[mask, "Senast auto uppdaterad"] = today_stamp()
            df.loc[mask, "Auto källa"] = "Finviz"
        except Exception:
            pass

    # Morningstar
    if ms_overview:
        try:
            ms = ms_overview(tkr) or {}
            if _to_float(ms.get("ps_ttm", 0.0)) > 0:
                df.loc[mask, "P/S"] = float(ms["ps_ttm"])
            if _to_float(ms.get("pb", 0.0)) > 0:
                df.loc[mask, "P/B"] = float(ms["pb"])
            if _to_float(ms.get("price", 0.0)) > 0:
                df.loc[mask, "Aktuell kurs"] = float(ms["price"])
            df.loc[mask, "Senast auto uppdaterad"] = today_stamp()
            df.loc[mask, "Auto källa"] = "Morningstar"
        except Exception:
            pass

    # SEC → P/B kvartal (beräkna P/B Q1..Q4 + snitt)
    if sec_pb_quarters:
        try:
            sec = sec_pb_quarters(tkr) or {}
            pairs = sec.get("pb_quarters") or []  # list[(date, pb)]
            if pairs:
                pairs = pairs[:4]  # Q1..Q4
                for idx, (_, pbv) in enumerate(pairs, start=1):
                    if idx > 4:
                        break
                    df.loc[mask, f"P/B Q{idx}"] = _to_float(pbv)
                # uppdatera snitt
                row = df.loc[mask].iloc[0]
                _, pb_avg = compute_ps_pb_snitt(row)
                df.loc[mask, "P/B-snitt (Q1..Q4)"] = pb_avg
            df.loc[mask, "Senast auto uppdaterad"] = today_stamp()
            df.loc[mask, "Auto källa"] = "SEC"
        except Exception:
            pass

    return df


# ---------- Manuell insamling (redigering + enskild uppdatering) ----------

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
        index=st.session_state["manual_idx"],
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
    if not is_new and len(vis) > 0:
        row = vis.iloc[sel - 1]
    else:
        row = pd.Series({c: (0.0 if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else "") for c in df.columns})

    # Enskild uppdatering
    c_up1, c_up2, c_up3 = st.columns([1, 1, 2])
    with c_up1:
        if not is_new and st.button("⚡ Snabbuppdatera (Yahoo)"):
            tkr = str(row.get("Ticker", "")).upper()
            if tkr:
                try:
                    q = yahoo_fetch_one_quick(tkr)
                    m = (df["Ticker"].astype(str).str.upper() == tkr)
                    if q.get("Bolagsnamn"): df.loc[m, "Bolagsnamn"] = str(q["Bolagsnamn"])
                    if q.get("Valuta"):     df.loc[m, "Valuta"]     = str(q["Valuta"])
                    if _to_float(q.get("Aktuell kurs", 0.0)) > 0: df.loc[m, "Aktuell kurs"] = float(q["Aktuell kurs"])
                    df.loc[m, "Årlig utdelning"]     = _to_float(q.get("Årlig utdelning", 0.0))
                    df.loc[m, "CAGR 5 år (%)"]       = _to_float(q.get("CAGR 5 år (%)", 0.0))
                    df.loc[m, "Senast auto uppdaterad"] = today_stamp()
                    df.loc[m, "Auto källa"] = "Yahoo (snabb)"
                    # Beräkna & spara
                    df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
                    save_df(ws_title, df2, bust_cache=True)
                    st.success("Snabbuppdaterad, sparad och beräknad.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Kunde inte snabbuppdatera: {e}")

    with c_up2:
        if not is_new and st.button("🔭 Full uppdatering (alla fetchers)"):
            tkr = str(row.get("Ticker", "")).upper()
            if tkr:
                try:
                    df = _update_one_all_fetchers(df, tkr)
                    df = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
                    save_df(ws_title, df, bust_cache=True)
                    st.success("Fetchers körda, beräkningar uppdaterade och sparade.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Kunde inte spara: {e}")
    with c_up3:
        st.caption("Tips: Snabbuppdatering hämtar pris/valuta/namn/utdelning/CAGR från Yahoo. Full uppdatering försöker även Finviz, Morningstar och SEC (P/B-kvartal).")

    # Obligatoriska fält
    st.markdown("### Obligatoriska fält")
    mtag = _m_tag(row)

    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input(f"Ticker (Yahoo-format) {mtag}", value=str(row.get("Ticker", "")).upper() if not is_new else "", placeholder="t.ex. AAPL")
        antal  = st.text_input(f"Antal aktier (du äger) {mtag}", value=str(row.get("Antal aktier", 0)))
        gav    = st.text_input(f"GAV (SEK) {mtag}", value=str(row.get("GAV (SEK)", 0)))
    with c2:
        oms_idag = st.text_input(f"Omsättning idag (M) {mtag}", value=str(row.get("Omsättning idag", 0)))
        oms_nxt  = st.text_input(f"Omsättning nästa år (M) {mtag}", value=str(row.get("Omsättning nästa år", 0)))

    # Auto-fält (fetchers)
    atag = _a_tag(row)
    with st.expander("🌐 Fält som hämtas (auto)"):
        cL, cR = st.columns(2)
        with cL:
            bolagsnamn   = st.text_input(f"Bolagsnamn {atag}", value=str(row.get("Bolagsnamn", "")))
            sektor       = st.text_input(f"Sektor {atag}", value=str(row.get("Sektor", "")))
            valuta       = st.text_input(f"Valuta (t.ex. USD, SEK) {atag}", value=str(row.get("Valuta", "") or "USD").upper())
            aktuell_kurs = st.text_input(f"Aktuell kurs {atag}", value=str(row.get("Aktuell kurs", 0)))
            utd_arlig    = st.text_input(f"Årlig utdelning {atag}", value=str(row.get("Årlig utdelning", 0)))
            payout_pct   = st.text_input(f"Payout (%) {atag}", value=str(row.get("Payout (%)", 0)))
        with cR:
            utest_m = st.text_input(f"Utestående aktier (miljoner) {atag}", value=str(row.get("Utestående aktier", 0)))
            ps  = st.text_input(f"P/S {atag}",   value=str(row.get("P/S", 0)))
            ps1 = st.text_input(f"P/S Q1 {atag}", value=str(row.get("P/S Q1", 0)))
            ps2 = st.text_input(f"P/S Q2 {atag}", value=str(row.get("P/S Q2", 0)))
            ps3 = st.text_input(f"P/S Q3 {atag}", value=str(row.get("P/S Q3", 0)))
            ps4 = st.text_input(f"P/S Q4 {atag}", value=str(row.get("P/S Q4", 0)))
            pb  = st.text_input(f"P/B {atag}",   value=str(row.get("P/B", 0)))
            pb1 = st.text_input(f"P/B Q1 {atag}", value=str(row.get("P/B Q1", 0)))
            pb2 = st.text_input(f"P/B Q2 {atag}", value=str(row.get("P/B Q2", 0)))
            pb3 = st.text_input(f"P/B Q3 {atag}", value=str(row.get("P/B Q3", 0)))
            pb4 = st.text_input(f"P/B Q4 {atag}", value=str(row.get("P/B Q4", 0)))

    # Beräknade (read-only)
    btag = _b_tag(row)
    with st.expander("🧮 Beräknade fält (auto)"):
        cA, cB = st.columns(2)
        with cA:
            st.text_input(f"P/S-snitt (Q1..Q4) {btag}", value=str(row.get("P/S-snitt (Q1..Q4)", 0)), disabled=True)
            st.text_input(f"Omsättning om 2 år (M) {btag}", value=str(row.get("Omsättning om 2 år", 0)), disabled=True)
            st.text_input(f"Riktkurs idag {btag}", value=str(row.get("Riktkurs idag", 0)), disabled=True)
            st.text_input(f"Riktkurs om 2 år {btag}", value=str(row.get("Riktkurs om 2 år", 0)), disabled=True)
        with cB:
            st.text_input(f"P/B-snitt (Q1..Q4) {btag}", value=str(row.get("P/B-snitt (Q1..Q4)", 0)), disabled=True)
            st.text_input(f"Omsättning om 3 år (M) {btag}", value=str(row.get("Omsättning om 3 år", 0)), disabled=True)
            st.text_input(f"Riktkurs om 1 år {btag}", value=str(row.get("Riktkurs om 1 år", 0)), disabled=True)
            st.text_input(f"Riktkurs om 3 år {btag}", value=str(row.get("Riktkurs om 3 år", 0)), disabled=True)

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
        if not ticker.strip():
            errors.append("Ticker saknas.")
        if errors:
            st.error(" | ".join(errors))
            return

        exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())
        exists = bool(exists_mask.any())

        update = {
            # manuella
            "Ticker": ticker.upper(),
            "Antal aktier": _to_float(antal),
            "GAV (SEK)": _to_float(gav),
            "Omsättning idag": _to_float(oms_idag),
            "Omsättning nästa år": _to_float(oms_nxt),
            # auto (kan överstyras manuellt om man vill)
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
                df.loc[exists_mask, "Senast manuellt uppdaterad"] = today_stamp()
        else:
            base = {c: (0.0 if c not in ["Ticker", "Bolagsnamn", "Sektor", "Valuta",
                                         "Senast manuellt uppdaterad", "Senast auto uppdaterad", "Auto källa",
                                         "Senast beräknad", "Div_Månader", "Div_Vikter"] else "")
                    for c in FINAL_COLS}
            base.update(update)
            base["Senast manuellt uppdaterad"] = today_stamp()
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
            exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())

        # Efter spara → snabb Yahoo för att fräscha kurs etc
        try:
            quick = yahoo_fetch_one_quick(ticker.upper())
            if quick.get("Bolagsnamn"):
                df.loc[exists_mask, "Bolagsnamn"] = str(quick["Bolagsnamn"])
            if quick.get("Valuta"):
                df.loc[exists_mask, "Valuta"] = str(quick["Valuta"])
            if _to_float(quick.get("Aktuell kurs", 0.0)) > 0:
                df.loc[exists_mask, "Aktuell kurs"] = float(quick["Aktuell kurs"])
            df.loc[exists_mask, "Årlig utdelning"] = _to_float(quick.get("Årlig utdelning", 0.0))
            df.loc[exists_mask, "CAGR 5 år (%)"] = _to_float(quick.get("CAGR 5 år (%)", 0.0))
            df.loc[exists_mask, "Senast auto uppdaterad"] = today_stamp()
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

# app.py — Del 5/6
# ---------- Data-vy (hela bladet + spara beräkningar) ----------

def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Data (hela bladet)")
    st.dataframe(df, use_container_width=True)

    st.markdown("**Spara alla beräkningar till Google Sheets**")
    c1, c2 = st.columns(2)
    horizon = c1.selectbox(
        "Score-horisont vid sparning",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=0,
    )
    strategy = c2.selectbox(
        "Strategi för score vid sparning",
        ["Auto (via sektor)", "Tillväxt", "Utdelning", "Finans"],
        index=0,
    )

    if st.button("💾 Spara beräkningar → Google Sheets"):
        try:
            df2 = enrich_for_save(df, horizon_for_score=horizon,
                                  strategy=("Auto" if strategy.startswith("Auto") else strategy))
            save_df(ws_title, df2, bust_cache=True)
            st.success("Beräkningar sparade till Google Sheets.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")


# ---------- Massuppdatering i sidopanel ----------

def _sidebar_massupdate_controls(df: pd.DataFrame, ws_title: str):
    st.sidebar.markdown("### 🔁 Massuppdatering")

    # Snabb (Yahoo)
    if st.sidebar.button("⚡ Snabbuppdatera alla (Yahoo)"):
        try:
            total = len(df)
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            fails = []
            for i, row in df.iterrows():
                tkr = str(row.get("Ticker", "")).strip().upper()
                if not tkr:
                    prog.progress((i + 1) / max(1, total))
                    continue
                status.write(f"Yahoo snabb: {i+1}/{total} – {tkr}")
                try:
                    q = yahoo_fetch_one_quick(tkr)
                    m = (df["Ticker"].astype(str).str.upper() == tkr)
                    if q.get("Bolagsnamn"): df.loc[m, "Bolagsnamn"] = str(q["Bolagsnamn"])
                    if q.get("Valuta"):     df.loc[m, "Valuta"]     = str(q["Valuta"])
                    px = _to_float(q.get("Aktuell kurs", 0.0))
                    if px > 0: df.loc[m, "Aktuell kurs"] = px
                    df.loc[m, "Årlig utdelning"] = _to_float(q.get("Årlig utdelning", 0.0))
                    df.loc[m, "CAGR 5 år (%)"]   = _to_float(q.get("CAGR 5 år (%)", 0.0))
                    df.loc[m, "Senast auto uppdaterad"] = today_stamp()
                    df.loc[m, "Auto källa"] = "Yahoo (snabb)"
                except Exception as e:
                    fails.append(f"{tkr}: {e}")
                time.sleep(0.6)  # artig paus
                prog.progress((i + 1) / max(1, total))

            # Re-beräkna, spara och bust:a cache
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Snabbuppdatering klar och sparad.")
            if fails:
                st.sidebar.warning("Vissa tickers kunde inte uppdateras:")
                st.sidebar.text_area("Fel", "\n".join(map(str, fails)), height=120)
        except Exception as e:
            st.sidebar.error(f"Kunde inte snabbuppdatera: {e}")

    # Full (Yahoo+Finviz+Morningstar+SEC)
    if st.sidebar.button("🔭 Full uppdatering (alla fetchers)"):
        try:
            total = len(df)
            prog = st.sidebar.progress(0.0)
            status = st.sidebar.empty()
            fails = []
            for i, row in df.iterrows():
                tkr = str(row.get("Ticker", "")).strip().upper()
                if not tkr:
                    prog.progress((i + 1) / max(1, total))
                    continue
                status.write(f"Full uppd: {i+1}/{total} – {tkr}")
                try:
                    df = _update_one_all_fetchers(df, tkr)
                except Exception as e:
                    fails.append(f"{tkr}: {e}")
                time.sleep(0.6)
                prog.progress((i + 1) / max(1, total))

            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2, bust_cache=True)
            st.sidebar.success("Full uppdatering klar och sparad.")
            if fails:
                st.sidebar.warning("Vissa tickers fick fel:")
                st.sidebar.text_area("Fel", "\n".join(map(str, fails)), height=120)
        except Exception as e:
            st.sidebar.error(f"Kunde inte köra full uppdatering: {e}")


# ---------- Portfölj ----------

def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")
    port = df.copy()

    # Koercera viktig numerik (hantera kommatecken/punkt & tomma strängar)
    for c in ["Antal aktier", "GAV (SEK)", "Aktuell kurs", "Årlig utdelning"]:
        if c in port.columns:
            port[c] = port[c].apply(_to_float)

    # Bara rader med ägd mängd
    port = port[port["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # FX
    def _fx(v):
        v = str(v).upper().strip()
        return rates.get(v, 1.0)

    port["FX→SEK"] = port["Valuta"].apply(_fx)

    # Värden
    port["Värde (SEK)"] = port["Antal aktier"].map(_to_float) * port["Aktuell kurs"].map(_to_float) * port["FX→SEK"]
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"].map(_to_float) * port["GAV (SEK)"].map(_to_float)

    total_value = float(port["Värde (SEK)"].sum())
    total_cost  = float(port["Anskaffningsvärde (SEK)"].sum())

    # DA & YOC
    port["DA (%)"]  = np.where(
        port["Aktuell kurs"].map(_to_float) > 0,
        (port["Årlig utdelning"].map(_to_float) / port["Aktuell kurs"].map(_to_float)) * 100.0,
        0.0,
    ).round(2)

    port["YOC (%)"] = np.where(
        port["GAV (SEK)"].map(_to_float) > 0,
        (port["Årlig utdelning"].map(_to_float) * port["FX→SEK"]) / port["GAV (SEK)"].map(_to_float) * 100.0,
        0.0,
    ).round(2)

    # Utdelning total SEK
    port["Årsutdelning (SEK)"] = port["Antal aktier"].map(_to_float) * port["Årlig utdelning"].map(_to_float) * port["FX→SEK"]
    total_div = float(port["Årsutdelning (SEK)"].sum())

    # Andelar & resultat
    port["Andel (%)"] = np.where(total_value > 0, (port["Värde (SEK)"] / total_value) * 100.0, 0.0).round(2)
    gain_sek = total_value - total_cost
    gain_pct = (gain_sek / total_cost * 100.0) if total_cost > 0 else 0.0

    st.markdown(
        f"""
**Totalt portföljvärde:** {round(total_value, 2)} SEK  
**Anskaffningsvärde:** {round(total_cost, 2)} SEK  
**Vinst:** {round(gain_sek, 2)} SEK (**{round(gain_pct, 2)} %**)  
**Total årlig utdelning:** {round(total_div, 2)} SEK  (≈ {round(total_div/12.0, 2)} SEK/mån)
"""
    )

    cols = [
        "Ticker", "Bolagsnamn", "Sektor", "Antal aktier", "Valuta",
        "Aktuell kurs", "GAV (SEK)", "Värde (SEK)", "Anskaffningsvärde (SEK)",
        "DA (%)", "YOC (%)", "Årsutdelning (SEK)", "Andel (%)"
    ]
    cols = [c for c in cols if c in port.columns]
    st.dataframe(
        port[cols].sort_values("Andel (%)", ascending=False),
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
        "Strategi", ["Auto (via sektor)", "Tillväxt", "Utdelning", "Finans"],
        index=0
    )

    subset = st.radio("Visa", ["Alla bolag", "Endast portfölj"], horizontal=True)
    base = df.copy()
    for c in ["Aktuell kurs", horizon, "Årlig utdelning", "CAGR 5 år (%)",
              "P/S", "P/S-snitt (Q1..Q4)", "P/B", "P/B-snitt (Q1..Q4)"]:
        if c in base.columns:
            base[c] = base[c].apply(_to_float)

    if subset == "Endast portfölj":
        base = base[base["Antal aktier"].apply(_to_float) > 0].copy()

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
            f"- **Score – Growth / Dividend / Financials / Total:** "
            f"{round(_to_float(r['Score (Growth)']), 1)} / "
            f"{round(_to_float(r['Score (Dividend)']), 1)} / "
            f"{round(_to_float(r['Score (Financials)']), 1)} / "
            f"**{round(_to_float(r['Score (Total)']), 1)}** "
            f"(Conf {int(_to_float(r['Confidence']))}%)"
        )


# ---------- Utdelningskalender ----------

def view_dividend_calendar(df: pd.DataFrame, ws_title: str, rates: Dict[str, float]):
    st.subheader("📅 Utdelningskalender (12 månader framåt)")
    months_forward = st.number_input("Antal månader framåt", min_value=3, max_value=24, value=12, step=1)
    write_back = st.checkbox("Skriv tillbaka schema till databasen (Div_Frekvens/år, Div_Månader, Div_Vikter)", value=True)

    if 'build_dividend_calendar' not in globals() or build_dividend_calendar is None:
        st.info("Dividendmodulen är inte laddad. Kalendern kan inte genereras i denna miljö.")
        return

    if st.button("Bygg kalender"):
        try:
            summ, det, df_out = build_dividend_calendar(
                df, rates, months_forward=int(months_forward),
                write_back_schedule=bool(write_back)
            )
            st.session_state["div_summ"] = summ
            st.session_state["div_det"] = det
            st.session_state["div_df_out"] = df_out
            st.success("Kalender skapad.")
        except Exception as e:
            st.error(f"Kunde inte skapa kalender: {e}")

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
            ws_write_df("Utdelningskalender – Summering",
                        summ if not summ.empty else pd.DataFrame(columns=["År", "Månad", "Månad (sv)", "Summa (SEK)"]))
            ws_write_df("Utdelningskalender – Detalj",
                        det if not det.empty else pd.DataFrame(columns=[
                            "År", "Månad", "Månad (sv)", "Ticker", "Bolagsnamn",
                            "Antal aktier", "Valuta", "Per utbetalning (valuta)", "SEK-kurs", "Summa (SEK)"
                        ]))
            st.success("Schema + kalender sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    if c2.button("↻ Rensa kalender-cache"):
        for k in ["div_summ", "div_det", "div_df_out"]:
            if k in st.session_state:
                del st.session_state[k]
        st.info("Kalender-cache rensad.")

# app.py — Del 6/6

# Se till att 'time' och ev. today_stamp finns när massuppd. körs
import time

if "today_stamp" not in globals():
    def today_stamp() -> str:
        # fallback till bef. now_stamp om den redan finns
        return now_stamp() if "now_stamp" in globals() else datetime.now().strftime("%Y-%m-%d")


def main():
    st.title("K-pf-rslag")

    # --- Välj Google Sheets-blad ---
    try:
        titles = list_worksheet_titles() or ["Blad1"]
    except Exception:
        titles = ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    # --- Snabb omläsning-knapp (bust cache) ---
    if st.sidebar.button("↻ Läs om data nu"):
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
        st.rerun()

    # --- Valutakurser i sidopanelen ---
    user_rates = sidebar_rates()

    # --- Läs data + snapshot + basberäkningar ---
    df = load_df(ws_title)              # använder cache + ensure_columns
    snapshot_on_start(df, ws_title)     # skapar 5-dagars snapshot & städar äldre
    df = update_calculations(df)        # P/S/P/B-snitt, riktkurser, omsättning år 2/3

    # --- Sidopanel: massuppdateringsknappar (Snabb/Full) ---
    _sidebar_massupdate_controls(df, ws_title)

    # --- Flikar ---
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
