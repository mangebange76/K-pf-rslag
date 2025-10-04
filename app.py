# app.py
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Egna moduler (oförändrade gränssnitt)
from stockapp.sheets import ws_read_df, ws_write_df, list_worksheet_titles
from stockapp.rates import (
    read_rates,
    save_rates,
    fetch_live_rates,
    repair_rates_sheet,
    DEFAULT_RATES,
)

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


# ------------------------------------------------------------
# Kolumnschema
# ------------------------------------------------------------
FINAL_COLS: List[str] = [
    # Bas
    "Ticker", "Bolagsnamn", "Sektor", "Valuta",
    "Antal aktier", "Aktuell kurs",
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
]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FINAL_COLS:
        if c not in out.columns:
            if any(k in c.lower() for k in ["kurs", "omsättning", "p/s", "p/b", "utdelning", "cagr", "aktier", "riktkurs", "payout"]):
                out[c] = 0.0
            else:
                out[c] = ""
    # typer
    float_cols = [
        "Antal aktier", "Aktuell kurs", "Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
        "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Årlig utdelning","Payout (%)","CAGR 5 år (%)",
    ]
    for c in float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Senast manuellt uppdaterad"]:
        out[c] = out[c].astype(str)
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
# Yahoo helpers
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
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo"):
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
        # Spara med beräknade kolumner + score
        try:
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2)
            st.sidebar.success("Beräkningar + kurser sparade till Google Sheets.")
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
# Poängmotor + sektorsvikter
# ------------------------------------------------------------
def score_rows(df: pd.DataFrame, horizon: str, strategy: str) -> pd.DataFrame:
    """
    Lägger till kolumner:
      - 'DA (%)'
      - 'Uppsida (%)'
      - 'Score (Growth)', 'Score (Dividend)', 'Score (Financials)', 'Score (Total)', 'Confidence'
    """
    out = df.copy()
    out["DA (%)"] = np.where(out["Aktuell kurs"] > 0, (out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0, 0.0)
    out["Uppsida (%)"] = np.where(out["Aktuell kurs"] > 0, (out[horizon] - out["Aktuell kurs"]) / out["Aktuell kurs"] * 100.0, 0.0)

    # Growth
    cur_ps = out["P/S"].replace(0, np.nan)
    ps_avg = out["P/S-snitt (Q1..Q4)"].replace(0, np.nan)
    cheap_ps = (ps_avg / (cur_ps * 2.0)).clip(upper=1.0).fillna(0.0)
    g_norm = (out["CAGR 5 år (%)"] / 30.0).clip(0, 1)
    u_norm = (out["Uppsida (%)"] / 50.0).clip(0, 1)
    out["Score (Growth)"] = (0.4 * g_norm + 0.4 * u_norm + 0.2 * cheap_ps) * 100.0

    # Dividend
    payout = out["Payout (%)"]
    payout_health = 1 - (abs(payout - 60.0) / 60.0)     # topp nära 60%
    payout_health = payout_health.clip(0, 1)
    payout_health = np.where(payout <= 0, 0.85, payout_health)  # ok om okänt
    y_norm = (out["DA (%)"] / 8.0).clip(0, 1)
    grow_ok = np.where(out["CAGR 5 år (%)"] >= 0, 1.0, 0.6)
    out["Score (Dividend)"] = (0.6 * y_norm + 0.3 * payout_health + 0.1 * grow_ok) * 100.0

    # Financials
    cur_pb = out["P/B"].replace(0, np.nan)
    pb_avg = out["P/B-snitt (Q1..Q4)"].replace(0, np.nan)
    cheap_pb = (pb_avg / (cur_pb * 2.0)).clip(upper=1.0).fillna(0.0)
    out["Score (Financials)"] = (0.7 * cheap_pb + 0.3 * u_norm) * 100.0

    # Vikter
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

    # Confidence
    need = [
        out["Aktuell kurs"] > 0,
        out["P/S-snitt (Q1..Q4)"] > 0,
        out["Omsättning idag"] >= 0,
        out["Omsättning nästa år"] >= 0,
    ]
    present = np.stack(need, axis=0).astype(float)
    out["Confidence"] = (present.mean(axis=0) * 100.0).round(0)
    return out


# ------------------------------------------------------------
# Enrichment vid sparning – DA, uppsidor för alla horisonter & score
# ------------------------------------------------------------
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


def enrich_for_save(df: pd.DataFrame, horizon_for_score: str = "Riktkurs idag", strategy: str = "Auto") -> pd.DataFrame:
    """Gör alla beräkningar + score och returnerar ett DF redo att skrivas till Sheets."""
    df2 = update_calculations(df)
    df2 = add_multi_uppsida(df2)
    # Score baseras på vald horisont (standard 'Riktkurs idag')
    df2 = score_rows(df2, horizon=horizon_for_score, strategy=("Auto" if strategy.startswith("Auto") else strategy))
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
    st.subheader("🧩 Manuell insamling (light)")
    vis = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" if r["Bolagsnamn"] else r["Ticker"] for _, r in vis.iterrows()]
    idx = st.selectbox("Välj ticker", list(range(len(vis))), format_func=lambda i: labels[i] if labels else "", index=0 if len(vis) else 0)
    if len(vis) == 0:
        st.info("Inga rader i databasen.")
        return

    row = vis.iloc[idx]
    tkr = st.text_input("Ticker", value=str(row["Ticker"]).upper())
    sektor = st.text_input("Sektor", value=str(row["Sektor"]))
    shares_m = st.number_input("Utestående aktier (miljoner)", value=float(row["Utestående aktier"]), step=1.0)
    ps = st.number_input("P/S", value=float(row["P/S"]), step=0.01)
    ps1 = st.number_input("P/S Q1", value=float(row["P/S Q1"]), step=0.01)
    ps2 = st.number_input("P/S Q2", value=float(row["P/S Q2"]), step=0.01)
    ps3 = st.number_input("P/S Q3", value=float(row["P/S Q3"]), step=0.01)
    ps4 = st.number_input("P/S Q4", value=float(row["P/S Q4"]), step=0.01)
    pb  = st.number_input("P/B", value=float(row["P/B"]), step=0.01)
    pb1 = st.number_input("P/B Q1", value=float(row["P/B Q1"]), step=0.01)
    pb2 = st.number_input("P/B Q2", value=float(row["P/B Q2"]), step=0.01)
    pb3 = st.number_input("P/B Q3", value=float(row["P/B Q3"]), step=0.01)
    pb4 = st.number_input("P/B Q4", value=float(row["P/B Q4"]), step=0.01)
    rev0 = st.number_input("Omsättning idag (M)", value=float(row["Omsättning idag"]), step=1.0)
    rev1 = st.number_input("Omsättning nästa år (M)", value=float(row["Omsättning nästa år"]), step=1.0)
    payout = st.number_input("Payout (%)", value=float(row["Payout (%)"]), step=1.0)

    if st.button("💾 Spara rad + beräkna"):
        mask = df["Ticker"] == row["Ticker"]
        df.loc[mask, ["Ticker","Sektor","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                      "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4",
                      "Omsättning idag","Omsättning nästa år","Payout (%)"]] = [[
            tkr, sektor, shares_m, ps, ps1, ps2, ps3, ps4,
            pb, pb1, pb2, pb3, pb4,
            rev0, rev1, payout
        ]]
        df.loc[mask, "Senast manuellt uppdaterad"] = now_stamp()

        # Sparas alltid med full enrichment (inkl. score)
        try:
            df2 = enrich_for_save(df, horizon_for_score="Riktkurs idag", strategy="Auto")
            save_df(ws_title, df2)
            st.success("Rad sparad och alla beräkningar skrivna till Google Sheets.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    st.markdown("### Förhandsgranskning (beräkningar)")
    tmp = update_calculations(vis.copy())
    st.dataframe(
        tmp.loc[[idx], ["Ticker","P/S-snitt (Q1..Q4)","P/B-snitt (Q1..Q4)","Omsättning om 2 år","Omsättning om 3 år",
                        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]],
        use_container_width=True,
    )


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
        port[["Ticker","Bolagsnamn","Sektor","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","DA (%)"]]
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

    sort_on = st.radio("Sortera på", ["Score (Total)", "Uppsida (%)", "DA (%)"], horizontal=True)
    base = base.sort_values(by=[sort_on], ascending=False).reset_index(drop=True)

    st.dataframe(
        base[["Ticker","Bolagsnamn","Sektor","Aktuell kurs",horizon,"Uppsida (%)",
              "DA (%)","Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence"]],
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### Kortvisning (bläddra)")
    if "idea_idx" not in st.session_state:
        st.session_state["idea_idx"] = 0
    st.session_state["idea_idx"] = st.number_input("Visa rad #", min_value=0, max_value=max(0, len(base)-1),
                                                   value=st.session_state["idea_idx"], step=1)
    r = base.iloc[st.session_state["idea_idx"]]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"- **Sektor:** {r['Sektor'] or '—'}")
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
        st.write(f"- **Score – Growth/Div/Fin/Total:** {round(float(r['Score (Growth)']),1)} / "
                 f"{round(float(r['Score (Dividend)']),1)} / {round(float(r['Score (Financials)']),1)} / "
                 f"**{round(float(r['Score (Total)']),1)}** (Conf {int(r['Confidence'])}%)")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    st.title("K-pf-rslag")

    # Välj blad
    titles = list_worksheet_titles() or ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    # Valutakurser i sidopanel
    user_rates = sidebar_rates()

    # Läs & beräkna
    df = load_df(ws_title)
    st.sidebar.markdown("---")
    mass_update_yahoo(df, ws_title)  # knapp i sidopanelen
    df = update_calculations(df)

    tabs = st.tabs(["📄 Data", "🧩 Manuell insamling", "📦 Portfölj", "💡 Köpförslag"])
    with tabs[0]:
        view_data(df, ws_title)
    with tabs[1]:
        view_manual(df, ws_title)
    with tabs[2]:
        view_portfolio(df, user_rates)
    with tabs[3]:
        view_ideas(df)


if __name__ == "__main__":
    main()
