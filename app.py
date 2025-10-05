# app.py
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Egna moduler
from stockapp.sheets import (
    ws_read_df,
    ws_write_df,
    list_worksheet_titles,
    delete_worksheet,
)
from stockapp.rates import (
    read_rates,
    save_rates,
    fetch_live_rates,
    repair_rates_sheet,
    DEFAULT_RATES,
)
from stockapp.dividends import build_dividend_calendar
from stockapp.collectors import mass_collect_and_apply as collectors_mass_update


# ------------------------------------------------------------
# App-config
# ------------------------------------------------------------
st.set_page_config(page_title="K-pf-rslag", layout="wide")


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _horizon_to_tag(h: str) -> str:
    if "om 1 år" in h: return "1 år"
    if "om 2 år" in h: return "2 år"
    if "om 3 år" in h: return "3 år"
    return "Idag"


def _now_sthlm():
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz)
    except Exception:
        return datetime.now()


# ------------------------------------------------------------
# Snapshot – spara df vid appstart och städa äldre än 5 dagar
# ------------------------------------------------------------
SNAP_PREFIX = "SNAP__"  # blad börjar med detta

def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

def _parse_snap_title(title: str) -> datetime | None:
    """
    SNAP__{WS}__YYYYMMDD_HHMMSS → plocka ut timestamp.
    """
    m = re.search(r"(\d{8}_\d{6})$", title)
    if not m:
        return None
    s = m.group(1)
    try:
        return datetime.strptime(s, "%Y%m%d_%H%M%S")
    except Exception:
        return None

def snapshot_on_start(df: pd.DataFrame, base_ws_title: str):
    """
    Körs en gång per sessionstart:
      - skapar SNAP-blad med timestamp
      - tar bort SNAP-blad äldre än 5 dygn
    """
    if st.session_state.get("_snapshot_done"):
        return
    st.session_state["_snapshot_done"] = True

    now = _now_sthlm()
    snap_title = f"{SNAP_PREFIX}{base_ws_title}__{_format_ts(now)}"
    try:
        ws_write_df(snap_title, df)
        st.sidebar.success(f"Snapshot sparat: {snap_title}")
    except Exception as e:
        st.sidebar.warning(f"Kunde inte spara snapshot: {e}")

    # Städa gamla (>5 dygn)
    try:
        titles = list_worksheet_titles() or []
        cutoff = now - timedelta(days=5)
        to_delete: List[str] = []
        for t in titles:
            if not str(t).startswith(SNAP_PREFIX):
                continue
            ts = _parse_snap_title(str(t))
            if ts is None:
                continue
            if ts < cutoff.replace(tzinfo=None):
                to_delete.append(t)
        for t in to_delete:
            try:
                delete_worksheet(t)
            except Exception:
                pass
        if to_delete:
            st.sidebar.info(f"Rensade {len(to_delete)} gamla snapshot-blad.")
    except Exception as e:
        st.sidebar.warning(f"Kunde inte rensa gamla snapshot-blad: {e}")


# ------------------------------------------------------------
# Kolumnschema
# ------------------------------------------------------------
FINAL_COLS: List[str] = [
    # Bas
    "Ticker", "Bolagsnamn", "Sektor", "Valuta",
    "Antal aktier", "GAV (SEK)", "Aktuell kurs",
    "Utestående aktier",  # miljoner

    # P/S
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt (Q1..Q4)",

    # P/B
    "P/B", "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",

    # Omsättning (M)
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",

    # Riktkurser (valuta)
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",

    # Utdelning
    "Årlig utdelning", "Payout (%)",

    # Övrigt
    "CAGR 5 år (%)", "Senast manuellt uppdaterad",

    # Dynamiska/visuella score-kolumner
    "DA (%)",
    "Uppsida idag (%)", "Uppsida 1 år (%)", "Uppsida 2 år (%)", "Uppsida 3 år (%)",
    "Score (Growth)", "Score (Dividend)", "Score (Financials)", "Score (Total)", "Confidence",

    # Sparade totalpoäng per horisont
    "Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)",

    # Sparade komponentpoäng per horisont
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
                "riktkurs","payout","score","uppsida","da","confidence","frekvens","gav"
            ]):
                out[c] = 0.0
            else:
                out[c] = ""
    float_cols = [
        "Antal aktier", "GAV (SEK)", "Aktuell kurs", "Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
        "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Årlig utdelning","Payout (%)","CAGR 5 år (%)",
        "DA (%)", "Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
        "Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence",
        "Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)",
        "Score Growth (Idag)", "Score Dividend (Idag)", "Score Financials (Idag)",
        "Score Growth (1 år)", "Score Dividend (1 år)", "Score Financials (1 år)",
        "Score Growth (2 år)", "Score Dividend (2 år)", "Score Financials (2 år)",
        "Score Growth (3 år)", "Score Dividend (3 år)", "Score Financials (3 år)",
        "Div_Frekvens/år",
    ]
    for c in float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Senast manuellt uppdaterad","Div_Månader","Div_Vikter"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
        else:
            out[c] = ""
    return out


# ------------------------------------------------------------
# Sidopanel – valutakurser
# ------------------------------------------------------------
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
            st.sidebar.success("Valutakurser sparade till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")

    if colD.button("🛠 Reparera bladet"):
        try:
            repair_rates_sheet()
            st.sidebar.success("Bladet reparerat.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte reparera: {e}")

    return {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}


# ------------------------------------------------------------
# Databas-IO
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str) -> pd.DataFrame:
    return ws_read_df(ws_title)

def load_df(ws_title: str) -> pd.DataFrame:
    df = ensure_columns(load_df_cached(ws_title))
    return df

def save_df(ws_title: str, df: pd.DataFrame):
    ws_write_df(ws_title, df)


# ------------------------------------------------------------
# Yahoo helpers – live + långsamma (snabbknappar)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def yahoo_fetch_one(ticker: str) -> Dict[str, float | str]:
    out = {"Bolagsnamn": "", "Valuta": "USD", "Aktuell kurs": 0.0, "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        name = info.get("shortName") or info.get("longName") or ""
        if name:
            out["Bolagsnamn"] = str(name)

        cur = info.get("currency")
        if cur:
            out["Valuta"] = str(cur).upper()

        price = info.get("regularMarketPrice")
        if price is None:
            try:
                fi = getattr(t, "fast_info", None) or {}
                price = fi.get("lastPrice")
            except Exception:
                price = None
        if price is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        if price is not None:
            out["Aktuell kurs"] = float(price)

        div_rate = info.get("dividendRate")
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        out["CAGR 5 år (%)"] = calc_cagr_5y(t)
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def yahoo_fetch_slow(ticker: str) -> Dict[str, float | str]:
    out = {
        "P/B": 0.0, "P/S": 0.0, "Payout (%)": 0.0,
        "Utestående aktier": 0.0, "Sektor": "", "Bolagsnamn": "", "Årlig utdelning": 0.0, "Valuta": ""
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pb = info.get("priceToBook")
        if pb is not None:
            out["P/B"] = float(pb)

        ps = info.get("priceToSalesTrailing12Months")
        if ps is not None:
            out["P/S"] = float(ps)

        div_rate = info.get("dividendRate")
        eps = info.get("trailingEps")
        payout = 0.0
        if div_rate is not None and eps is not None and float(eps) > 0:
            payout = (float(div_rate) / float(eps)) * 100.0
        out["Payout (%)"] = float(payout)

        shares = info.get("sharesOutstanding")
        if shares is not None:
            out["Utestående aktier"] = float(shares) / 1e6

        sector = info.get("sector") or ""
        out["Sektor"] = str(sector)

        name = info.get("shortName") or info.get("longName") or ""
        if name:
            out["Bolagsnamn"] = str(name)

        cur = info.get("currency")
        if cur:
            out["Valuta"] = str(cur).upper()

        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)
    except Exception:
        pass
    return out


def calc_cagr_5y(t: yf.Ticker) -> float:
    try:
        df_is = getattr(t, "income_stmt", None)
        ser = None
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            ser = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(t, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                ser = df_fin.loc["Total Revenue"].dropna()
        if ser is None or ser.empty or len(ser) < 2:
            return 0.0
        ser = ser.sort_index()
        start, end = float(ser.iloc[0]), float(ser.iloc[-1])
        years = max(1, len(ser) - 1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


def mass_update_yahoo(df: pd.DataFrame, ws_title: str):
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo (pris+snabbt)"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        tickers = df["Ticker"].astype(str).tolist()
        n = len(tickers)
        for i, tkr in enumerate(tickers):
            status.write(f"Hämtar {i+1}/{n} – {tkr}")
            data = yahoo_fetch_one(tkr)
            if data.get("Bolagsnamn"): df.loc[df["Ticker"] == tkr, "Bolagsnamn"] = data["Bolagsnamn"]
            if data.get("Valuta"):     df.loc[df["Ticker"] == tkr, "Valuta"]     = data["Valuta"]
            if float(data.get("Aktuell kurs", 0.0)) > 0: df.loc[df["Ticker"] == tkr, "Aktuell kurs"] = float(data["Aktuell kurs"])
            df.loc[df["Ticker"] == tkr, "Årlig utdelning"] = float(data.get("Årlig utdelning", 0.0))
            df.loc[df["Ticker"] == tkr, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)", 0.0))
            time.sleep(0.5)
            bar.progress((i + 1) / max(1, n))
        try:
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2)
            st.sidebar.success("Beräkningar + kurser sparade till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")


def mass_update_fundamentals(df: pd.DataFrame, ws_title: str):
    if st.sidebar.button("🧩 Uppdatera långsamma nyckeltal (Yahoo)"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        tickers = df["Ticker"].astype(str).tolist()
        n = len(tickers)
        for i, tkr in enumerate(tickers):
            status.write(f"Nyckeltal {i+1}/{n} – {tkr}")
            data = yahoo_fetch_slow(tkr)
            for k in ["P/B","P/S","Payout (%)","Utestående aktier","Årlig utdelning"]:
                if float(data.get(k, 0.0)) > 0:
                    df.loc[df["Ticker"] == tkr, k] = float(data[k])
            if data.get("Sektor"):
                df.loc[df["Ticker"] == tkr, "Sektor"] = str(data["Sektor"])
            if data.get("Bolagsnamn"):
                df.loc[df["Ticker"] == tkr, "Bolagsnamn"] = str(data["Bolagsnamn"])
            if data.get("Valuta"):
                df.loc[df["Ticker"] == tkr, "Valuta"] = str(data["Valuta"])
            time.sleep(0.5)
            bar.progress((i + 1) / max(1, n))
        try:
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2)
            st.sidebar.success("Långsamma nyckeltal sparade till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")


# ------------------------------------------------------------
# Beräkningar (P/S-snitt, oms år2/3, riktkurser)
# ------------------------------------------------------------
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
        out.at[i, "Riktkurs om 1 år"] = round(float(r.get("Omsättning nästa år", 0.0))  * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 2 år"] = round(float(out.at[i, "Omsättning om 2 år"])    * ps_avg / shares_m, 2)
        out.at[i, "Riktkurs om 3 år"] = round(float(out.at[i, "Omsättning om 3 år"])    * ps_avg / shares_m, 2)
    return out


# ------------------------------------------------------------
# Poängmotor (kompakt)
# ------------------------------------------------------------
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
        if any(k in s for k in ["bank", "finans", "insurance", "financial"]): return (0.25, 0.25, 0.50)
        if any(k in s for k in ["utility", "utilities", "consumer staples", "telecom"]): return (0.20, 0.60, 0.20)
        if any(k in s for k in ["tech", "information technology", "semiconductor", "software"]): return (0.70, 0.10, 0.20)
        return (0.45, 0.35, 0.20)

    Wg, Wd, Wf = [], [], []
    for _, r in out.iterrows():
        wg, wd, wf = weights_for_row(r.get("Sektor", ""), strategy)
        Wg.append(wg); Wd.append(wd); Wf.append(wf)
    Wg = np.array(Wg); Wd = np.array(Wd); Wf = np.array(Wf)

    out["Score (Total)"] = (Wg * out["Score (Growth)"] + Wd * out["Score (Dividend)"] + Wf * out["Score (Financials)"]).round(2)

    need = [
        out["Aktuell kurs"] > 0,
        out["P/S-snitt (Q1..Q4)"] > 0,
        out["Omsättning idag"] >= 0,
        out["Omsättning nästa år"] >= 0,
    ]
    present = np.stack(need, axis=0).astype(float)
    out["Confidence"] = (present.mean(axis=0) * 100.0).round(0)
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
        up = ((rk - price) / price) * 100.0
        out[label] = up.fillna(0.0)
    out["DA (%)"] = np.where(out["Aktuell kurs"] > 0, (out["Årlig utdelning"]/out["Aktuell kurs"])*100.0, 0.0)
    return out


def compute_scores_all_horizons(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    out = df.copy()
    mapping = [
        ("Riktkurs idag",   "Idag"),
        ("Riktkurs om 1 år","1 år"),
        ("Riktkurs om 2 år","2 år"),
        ("Riktkurs om 3 år","3 år"),
    ]
    strat = "Auto" if str(strategy).startswith("Auto") else strategy
    for horizon, tag in mapping:
        tmp = score_rows(out, horizon=horizon, strategy=strat)
        out[f"Score Growth ({tag})"]      = tmp["Score (Growth)"].round(2)
        out[f"Score Dividend ({tag})"]    = tmp["Score (Dividend)"].round(2)
        out[f"Score Financials ({tag})"]  = tmp["Score (Financials)"].round(2)
        out[f"Score Total ({tag})"]       = tmp["Score (Total)"].round(2)
    return out


def enrich_for_save(df: pd.DataFrame, horizon_for_score: str = "Riktkurs idag", strategy: str = "Auto") -> pd.DataFrame:
    df2 = update_calculations(df)
    df2 = add_multi_uppsida(df2)
    df2 = score_rows(df2, horizon=horizon_for_score, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    df2 = compute_scores_all_horizons(df2, strategy=("Auto" if str(strategy).startswith("Auto") else strategy))
    return df2


# ------------------------------------------------------------
# Vyer
# ------------------------------------------------------------
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
            save_df(ws_title, df2)
            st.success("Beräkningar sparade till Google Sheets.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")


def view_manual(df: pd.DataFrame, ws_title: str):
    """
    Manuell insamling:
      - Välj ticker + bläddra
      - Lägg till/uppdatera bolag
      - Obligatoriska fält: Ticker, Antal aktier, GAV (SEK), Omsättning idag, Omsättning nästa år
      - Övriga fält i expander
      - Sätter 'Senast manuellt uppdaterad' vid ändringar i obligatoriska fält
    """
    st.subheader("🧩 Manuell insamling")

    vis = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r.get("Bolagsnamn","")).strip() else str(r["Ticker"]) for _, r in vis.iterrows()]
    labels = ["➕ Lägg till nytt bolag..."] + labels

    if "manual_idx" not in st.session_state:
        st.session_state["manual_idx"] = 0  # 0 = nytt bolag

    sel = st.selectbox("Välj bolag att redigera", list(range(len(labels))), format_func=lambda i: labels[i], index=st.session_state["manual_idx"])
    st.session_state["manual_idx"] = sel

    col_nav1, col_nav2, col_nav3 = st.columns([1,2,1])
    with col_nav1:
        if st.button("⬅️ Föregående", use_container_width=True, disabled=(sel <= 0)):
            st.session_state["manual_idx"] = max(0, sel - 1)
            st.rerun()
    with col_nav3:
        if st.button("➡️ Nästa", use_container_width=True, disabled=(sel >= len(labels)-1)):
            st.session_state["manual_idx"] = min(len(labels)-1, sel + 1)
            st.rerun()

    is_new = (sel == 0)
    if not is_new:
        row = vis.iloc[sel-1]
    else:
        row = pd.Series({c: (0.0 if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) else "") for c in df.columns})

    # OBLIGATORISKA FÄLT
    st.markdown("### Obligatoriska fält")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker (Yahoo-format)", value=str(row.get("Ticker","")).upper() if not is_new else "", placeholder="t.ex. AAPL")
        antal  = st.number_input("Antal aktier (du äger)", value=float(row.get("Antal aktier", 0.0) or 0.0), step=1.0, min_value=0.0)
        gav    = st.number_input("GAV (SEK)", value=float(row.get("GAV (SEK)", 0.0) or 0.0), step=0.01, min_value=0.0, format="%.2f")
    with c2:
        oms_idag = st.number_input("Omsättning idag (M)", value=float(row.get("Omsättning idag", 0.0) or 0.0), step=1.0, min_value=0.0)
        oms_nxt  = st.number_input("Omsättning nästa år (M)", value=float(row.get("Omsättning nästa år", 0.0) or 0.0), step=1.0, min_value=0.0)

    # ÖVRIGA FÄLT
    with st.expander("➕ Visa/ändra övriga fält"):
        cL, cR = st.columns(2)
        with cL:
            bolagsnamn = st.text_input("Bolagsnamn", value=str(row.get("Bolagsnamn","")))
            sektor     = st.text_input("Sektor", value=str(row.get("Sektor","")))
            valuta     = st.text_input("Valuta (t.ex. USD, SEK)", value=str(row.get("Valuta","") or "USD").upper())

            aktuell_kurs = st.number_input("Aktuell kurs", value=float(row.get("Aktuell kurs", 0.0) or 0.0), step=0.01, min_value=0.0)
            utd_arlig    = st.number_input("Årlig utdelning", value=float(row.get("Årlig utdelning", 0.0) or 0.0), step=0.01, min_value=0.0)
            payout_pct   = st.number_input("Payout (%)", value=float(row.get("Payout (%)", 0.0) or 0.0), step=1.0, min_value=0.0)
            cagr5        = st.number_input("CAGR 5 år (%)", value=float(row.get("CAGR 5 år (%)", 0.0) or 0.0), step=0.1)
        with cR:
            utest_m = st.number_input("Utestående aktier (miljoner)", value=float(row.get("Utestående aktier", 0.0) or 0.0), step=1.0, min_value=0.0)
            ps  = st.number_input("P/S",   value=float(row.get("P/S", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps1 = st.number_input("P/S Q1", value=float(row.get("P/S Q1", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps2 = st.number_input("P/S Q2", value=float(row.get("P/S Q2", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps3 = st.number_input("P/S Q3", value=float(row.get("P/S Q3", 0.0) or 0.0), step=0.01, min_value=0.0)
            ps4 = st.number_input("P/S Q4", value=float(row.get("P/S Q4", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb  = st.number_input("P/B",   value=float(row.get("P/B", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb1 = st.number_input("P/B Q1", value=float(row.get("P/B Q1", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb2 = st.number_input("P/B Q2", value=float(row.get("P/B Q2", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb3 = st.number_input("P/B Q3", value=float(row.get("P/B Q3", 0.0) or 0.0), step=0.01, min_value=0.0)
            pb4 = st.number_input("P/B Q4", value=float(row.get("P/B Q4", 0.0) or 0.0), step=0.01, min_value=0.0)

    def _any_core_change(before: pd.Series, after: dict) -> bool:
        core = ["Antal aktier", "GAV (SEK)", "Omsättning idag", "Omsättning nästa år"]
        for k in core:
            b = float(before.get(k, 0.0) or 0.0)
            a = float(after.get(k, 0.0) or 0.0)
            if abs(a - b) > 1e-12:
                return True
        return False

    if st.button("💾 Spara"):
        errors = []
        if not ticker.strip():
            errors.append("Ticker saknas.")
        if antal < 0: errors.append("Antal aktier kan inte vara negativt.")
        if gav < 0: errors.append("GAV (SEK) kan inte vara negativt.")
        if oms_idag < 0 or oms_nxt < 0: errors.append("Omsättning idag/nästa år kan inte vara negativt.")
        if errors:
            st.error(" | ".join(errors))
            return

        exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())
        exists = bool(exists_mask.any())

        update = {
            "Ticker": ticker.upper(),
            "Antal aktier": float(antal),
            "GAV (SEK)": float(gav),
            "Omsättning idag": float(oms_idag),
            "Omsättning nästa år": float(oms_nxt),
            "Bolagsnamn": (bolagsnamn or "").strip(),
            "Sektor": (sektor or "").strip(),
            "Valuta": (valuta or "").strip().upper(),
            "Aktuell kurs": float(aktuell_kurs),
            "Årlig utdelning": float(utd_arlig),
            "Payout (%)": float(payout_pct),
            "CAGR 5 år (%)": float(cagr5),
            "Utestående aktier": float(utest_m),
            "P/S": float(ps), "P/S Q1": float(ps1), "P/S Q2": float(ps2), "P/S Q3": float(ps3), "P/S Q4": float(ps4),
            "P/B": float(pb), "P/B Q1": float(pb1), "P/B Q2": float(pb2), "P/B Q3": float(pb3), "P/B Q4": float(pb4),
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
            # ny rad – se till att alla kolumner finns med default
            new_row = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Sektor","Valuta","Senast manuellt uppdaterad","Div_Månader","Div_Vikter"] else "") for c in FINAL_COLS}
            new_row.update(update)
            new_row["Senast manuellt uppdaterad"] = now_stamp()
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        try:
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2)
            st.success("Sparat och beräkningar uppdaterade.")
            # hoppa till ny/uppdaterad rad i listan
            try:
                vis2 = df2.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
                new_idx = vis2.index[vis2["Ticker"].astype(str).str.upper() == ticker.upper()].tolist()
                if new_idx:
                    st.session_state["manual_idx"] = new_idx[0] + 1  # +1 pga "ny" först
            except Exception:
                pass
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    st.markdown("### Förhandsgranskning (beräknade fält)")
    tmp = update_calculations(df.copy())
    if not is_new:
        mask = (tmp["Ticker"].astype(str).str.upper() == str(row["Ticker"]).upper())
        prev = tmp.loc[mask, [
            "Ticker","Bolagsnamn","P/S-snitt (Q1..Q4)","P/B-snitt (Q1..Q4)",
            "Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"
        ]]
        st.dataframe(prev, use_container_width=True)
    else:
        st.info("Ny post: fyll i obligatoriska fält och spara för att se beräkningar.")


def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Vx"] = port["Valuta"].apply(lambda v: rates.get(str(v).upper(), 1.0))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Vx"]
    tot = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(tot > 0, (port["Värde (SEK)"]/tot)*100.0, 0.0).round(2)
    port["DA (%)"] = np.where(port["Aktuell kurs"]>0, (port["Årlig utdelning"]/port["Aktuell kurs"])*100.0, 0.0).round(2)

    st.markdown(f"**Totalt portföljvärde:** {round(tot,2)} SEK")
    st.dataframe(
        port[["Ticker","Bolagsnamn","Sektor","Antal aktier","GAV (SEK)","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","DA (%)"]]
            .sort_values("Andel (%)", ascending=False),
        use_container_width=True
    )


def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag")

    if df.empty:
        st.info("Inga rader.")
        return

    horizon = st.selectbox("Riktkurs-horisont", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    strategy = st.selectbox("Strategi", ["Auto (via sektor)","Tillväxt","Utdelning","Finans"], index=0)

    subset = st.radio("Visa", ["Alla bolag","Endast portfölj"], horizontal=True)
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
    for group in ["Growth","Dividend","Financials"]:
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
    base["DA (%)"] = np.where(base["Aktuell kurs"] > 0, (base["Årlig utdelning"]/base["Aktuell kurs"])*100.0, 0.0).round(2)

    ascending = False
    if sort_on == "Uppsida (%)":
        trim_mode = st.checkbox("Visa trim/sälj-läge (minst uppsida först)", value=False)
        if trim_mode:
            ascending = True

    reverse_global = st.checkbox("Omvänd sortering (gäller valt fält)", value=False)
    if reverse_global:
        ascending = not ascending

    cols = ["Ticker","Bolagsnamn","Sektor","Aktuell kurs",horizon,"Uppsida (%)","DA (%)"]
    if show_components:
        cols += ["Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence"]
    else:
        cols += ["Score (Total)","Confidence"]
    if show_saved and selected_saved_cols:
        cols += selected_saved_cols

    base = base.sort_values(by=[sort_on], ascending=ascending).reset_index(drop=True)
    st.dataframe(base[cols], use_container_width=True)

    st.markdown("---")
    st.markdown("### Kortvisning (bläddra)")
    if "idea_idx" not in st.session_state:
        st.session_state["idea_idx"] = 0
    st.session_state["idea_idx"] = st.number_input(
        "Visa rad #", min_value=0, max_value=max(0, len(base)-1),
        value=st.session_state["idea_idx"], step=1
    )

    r = base.iloc[st.session_state["idea_idx"]]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- **Sektor:** {r.get('Sektor','—')}")
        st.write(f"- **Aktuell kurs:** {round(float(r['Aktuell kurs']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs idag:** {round(float(r['Riktkurs idag']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 1 år:** {round(float(r['Riktkurs om 1 år']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 2 år:** {round(float(r['Riktkurs om 2 år']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 3 år:** {round(float(r['Riktkurs om 3 år']),2)} {r['Valuta']}")
        st.write(f"- **Uppsida ({horizon}):** {round(float(r['Uppsida (%)']),2)} %")
    with c2:
        st.write(f"- **P/S-snitt (Q1..Q4):** {round(float(r['P/S-snitt (Q1..Q4)']),2)}")
        st.write(f"- **P/B-snitt (Q1..Q4):** {round(float(r['P/B-snitt (Q1..Q4)']),2)}")
        st.write(f"- **Omsättning idag (M):** {round(float(r['Omsättning idag']),2)}")
        st.write(f"- **Omsättning nästa år (M):** {round(float(r['Omsättning nästa år']),2)}")
        st.write(f"- **Årlig utdelning:** {round(float(r['Årlig utdelning']),2)}")
        st.write(f"- **Payout:** {round(float(r['Payout (%)']),2)} %")
        st.write(f"- **DA (egen):** {round(float(r['DA (%)']),2)} %")
        st.write(f"- **CAGR 5 år:** {round(float(r['CAGR 5 år (%)']),2)} %")
        st.write(f"- **Score – Growth / Dividend / Financials / Total:** "
                 f"{round(float(r['Score (Growth)']),1)} / {round(float(r['Score (Dividend)']),1)} / "
                 f"{round(float(r['Score (Financials)']),1)} / **{round(float(r['Score (Total)']),1)}** "
                 f"(Conf {int(r['Confidence'])}%)")


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
            save_df(ws_title, df2)
            summ = st.session_state.get("div_summ", pd.DataFrame())
            det = st.session_state.get("div_det", pd.DataFrame())
            ws_write_df("Utdelningskalender – Summering", summ if not summ.empty else pd.DataFrame(columns=["År","Månad","Månad (sv)","Summa (SEK)"]))
            ws_write_df("Utdelningskalender – Detalj", det if not det.empty else pd.DataFrame(columns=[
                "År","Månad","Månad (sv)","Ticker","Bolagsnamn","Antal aktier","Valuta","Per utbetalning (valuta)","SEK-kurs","Summa (SEK)"]))
            st.success("Schema + kalender sparat till Google Sheets.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    if c2.button("↻ Rensa kalender-cache"):
        for k in ["div_summ","div_det","div_df_out"]:
            if k in st.session_state:
                del st.session_state[k]
        st.info("Kalender-cache rensad.")


# ------------------------------------------------------------
# Massuppdatering alla källor (Yahoo+Finviz+Morningstar+SEC)
# ------------------------------------------------------------
def mass_update_all_sources(df: pd.DataFrame, ws_title: str):
    if st.sidebar.button("🔭 Massuppdatera (Yahoo + Finviz + Morningstar + SEC)"):
        out = collectors_mass_update(
            df,
            ws_title=ws_title,
            delay_sec=0.5,
            save_fn=ws_write_df
        )
        try:
            out2 = enrich_for_save(out, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, out2)
            st.sidebar.success("✅ All källdata uppdaterad och beräkningar sparade till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"⚠️ Kunde inte spara beräkningar efter uppdatering: {e}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.title("K-pf-rslag")

    titles = list_worksheet_titles() or ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    user_rates = sidebar_rates()

    # Läs data
    df = load_df(ws_title)

    # Snapshot direkt vid appstart (en gång per session)
    snapshot_on_start(df, ws_title)

    st.sidebar.markdown("---")
    # Knappar för uppdateringar
    mass_update_yahoo(df, ws_title)
    mass_update_fundamentals(df, ws_title)
    mass_update_all_sources(df, ws_title)

    df = update_calculations(df)

    tabs = st.tabs(["📄 Data", "🧩 Manuell insamling", "📦 Portfölj", "💡 Köpförslag", "📅 Utdelningskalender"])
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
