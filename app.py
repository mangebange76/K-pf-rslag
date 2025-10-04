# app.py
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Våra moduler
from stockapp.sheets import ws_read_df, ws_write_df, list_worksheet_titles
from stockapp.rates import (
    read_rates,
    save_rates,
    fetch_live_rates,
    repair_rates_sheet,
    DEFAULT_RATES,
)

# ------------------------------- App config -------------------------------
st.set_page_config(page_title="K-pf-rslag", layout="wide")


# ------------------------------- Hjälpare -------------------------------
def now_stamp() -> str:
    try:
        import pytz

        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ------------------------------- Kolumnschema -------------------------------
FINAL_COLS: List[str] = [
    # Bas
    "Ticker",
    "Bolagsnamn",
    "Valuta",
    "Antal aktier",
    "Aktuell kurs",
    "Utestående aktier",  # i miljoner
    # P/S
    "P/S",
    "P/S Q1",
    "P/S Q2",
    "P/S Q3",
    "P/S Q4",
    "P/S-snitt (Q1..Q4)",
    # P/B (manuellt/later)
    "P/B",
    "P/B Q1",
    "P/B Q2",
    "P/B Q3",
    "P/B Q4",
    "P/B-snitt (Q1..Q4)",
    # Omsättning
    "Omsättning idag",  # M
    "Omsättning nästa år",  # M (manuell)
    "Omsättning om 2 år",
    "Omsättning om 3 år",
    # Riktkurser
    "Riktkurs idag",
    "Riktkurs om 1 år",
    "Riktkurs om 2 år",
    "Riktkurs om 3 år",
    # Övrigt
    "Årlig utdelning",
    "CAGR 5 år (%)",
    "Senast manuellt uppdaterad",
]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FINAL_COLS:
        if c not in out.columns:
            if any(
                key in c.lower()
                for key in [
                    "kurs",
                    "omsättning",
                    "p/s",
                    "p/b",
                    "utdelning",
                    "cagr",
                    "aktier",
                    "riktkurs",
                    "andel",
                ]
            ):
                out[c] = 0.0
            else:
                out[c] = ""
    # typer
    float_cols = [
        "Antal aktier",
        "Aktuell kurs",
        "Utestående aktier",
        "P/S",
        "P/S Q1",
        "P/S Q2",
        "P/S Q3",
        "P/S Q4",
        "P/S-snitt (Q1..Q4)",
        "P/B",
        "P/B Q1",
        "P/B Q2",
        "P/B Q3",
        "P/B Q4",
        "P/B-snitt (Q1..Q4)",
        "Omsättning idag",
        "Omsättning nästa år",
        "Omsättning om 2 år",
        "Omsättning om 3 år",
        "Riktkurs idag",
        "Riktkurs om 1 år",
        "Riktkurs om 2 år",
        "Riktkurs om 3 år",
        "Årlig utdelning",
        "CAGR 5 år (%)",
    ]
    for c in float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    for c in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad"]:
        out[c] = out[c].astype(str)
    return out


# ------------------------------- Rates-panel -------------------------------
def sidebar_rates() -> Dict[str, float]:
    st.sidebar.subheader("💱 Valutakurser → SEK")

    # Ladda en gång i session
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
            for k in ["USD", "NOK", "CAD", "EUR"]:
                st.session_state[f"rate_{k.lower()}"] = float(live[k])
            st.sidebar.success("Livekurser hämtade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta livekurser: {e}")

    if colB.button("↻ Läs sparade kurser"):
        try:
            saved = read_rates()
            for k in ["USD", "NOK", "CAD", "EUR"]:
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


# ------------------------------- Databas-IO -------------------------------
@st.cache_data(show_spinner=False)
def load_df_cached(ws_title: str) -> pd.DataFrame:
    df = ws_read_df(ws_title)
    return df


def load_df(ws_title: str) -> pd.DataFrame:
    df = load_df_cached(ws_title)
    df = ensure_columns(df)
    return df


def save_df(ws_title: str, df: pd.DataFrame):
    ws_write_df(ws_title, df)


# ------------------------------- Yahoo helpers -------------------------------
@st.cache_data(show_spinner=False, ttl=600)
def yahoo_fetch_one(ticker: str) -> Dict[str, float | str]:
    """
    Hämtar minimal men robust uppsättning fält från Yahoo.
    """
    out = {"Bolagsnamn": "", "Valuta": "USD", "Aktuell kurs": 0.0, "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0}
    try:
        t = yf.Ticker(ticker)

        # Namn/valuta/price – robust
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        name = info.get("shortName") or info.get("longName") or ""
        if name:
            out["Bolagsnamn"] = str(name)

        cur = info.get("currency", None)
        if cur:
            out["Valuta"] = str(cur).upper()

        price = info.get("regularMarketPrice", None)
        if price is None:
            try:
                fi = getattr(t, "fast_info", None) or {}
                price = fi.get("lastPrice", None)
            except Exception:
                price = None
        if price is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        if price is not None:
            out["Aktuell kurs"] = float(price)

        # Utdelning (årstakt)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        # CAGR 5y – baserat på financials/income_stmt (Total Revenue)
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
        start = float(ser.iloc[0])
        end = float(ser.iloc[-1])
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
        miss = []
        tickers = df["Ticker"].astype(str).tolist()
        n = len(tickers)
        for i, tkr in enumerate(tickers):
            status.write(f"Hämtar {i+1}/{n} – {tkr}")
            data = yahoo_fetch_one(tkr)

            if data.get("Bolagsnamn"):
                df.loc[df["Ticker"] == tkr, "Bolagsnamn"] = data["Bolagsnamn"]
            if data.get("Valuta"):
                df.loc[df["Ticker"] == tkr, "Valuta"] = data["Valuta"]
            if float(data.get("Aktuell kurs", 0.0)) > 0:
                df.loc[df["Ticker"] == tkr, "Aktuell kurs"] = float(data["Aktuell kurs"])
            df.loc[df["Ticker"] == tkr, "Årlig utdelning"] = float(data.get("Årlig utdelning", 0.0))
            df.loc[df["Ticker"] == tkr, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)", 0.0))

            time.sleep(0.5)  # throttling
            bar.progress((i + 1) / max(1, n))

        try:
            save_df(ws_title, df)
            st.sidebar.success("Sparat till Google Sheets.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")


# ------------------------------- Beräkningar -------------------------------
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
        # Snitt
        ps_avg, pb_avg = compute_ps_pb_snitt(r)
        out.at[i, "P/S-snitt (Q1..Q4)"] = ps_avg
        out.at[i, "P/B-snitt (Q1..Q4)"] = pb_avg

        # CAGR clamp → dämpning för år 2/3 (2%–50%)
        cagr = float(r.get("CAGR 5 år (%)", 0.0))
        g = clamp(cagr, 2.0, 50.0) / 100.0

        # Omsättning 2 & 3 år – utgår från "Omsättning nästa år" om det finns
        next_rev = float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            out.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            out.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)
        else:
            out.at[i, "Omsättning om 2 år"] = float(r.get("Omsättning om 2 år", 0.0))
            out.at[i, "Omsättning om 3 år"] = float(r.get("Omsättning om 3 år", 0.0))

        shares_m = float(r.get("Utestående aktier", 0.0))
        if shares_m <= 0:
            # kan inte räkna riktkurs
            for c in ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]:
                out.at[i, c] = 0.0
            continue

        # Riktkurser via P/S-snitt
        out.at[i, "Riktkurs idag"] = round(float(r.get("Omsättning idag", 0.0)) * ps_avg / shares_m, 2) if ps_avg > 0 else 0.0
        out.at[i, "Riktkurs om 1 år"] = (
            round(float(r.get("Omsättning nästa år", 0.0)) * ps_avg / shares_m, 2) if ps_avg > 0 else 0.0
        )
        out.at[i, "Riktkurs om 2 år"] = (
            round(float(out.at[i, "Omsättning om 2 år"]) * ps_avg / shares_m, 2) if ps_avg > 0 else 0.0
        )
        out.at[i, "Riktkurs om 3 år"] = (
            round(float(out.at[i, "Omsättning om 3 år"]) * ps_avg / shares_m, 2) if ps_avg > 0 else 0.0
        )
    return out


# ------------------------------- Vyer -------------------------------
def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Data (hela bladet)")
    st.dataframe(df, use_container_width=True)

    if st.button("💾 Spara beräkningar → Google Sheets"):
        try:
            save_df(ws_title, df)
            st.success("Sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")


def view_manual(df: pd.DataFrame, ws_title: str):
    st.subheader("🧩 Manuell insamling (light)")

    # Sorteringslista
    vis = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" if r["Bolagsnamn"] else r["Ticker"] for _, r in vis.iterrows()]
    idx = st.selectbox("Välj ticker", list(range(len(vis))), format_func=lambda i: labels[i] if labels else "", index=0 if len(vis) else 0)

    if len(vis) == 0:
        st.info("Inga rader i databasen.")
        return

    row = vis.iloc[idx]
    tkr = st.text_input("Ticker", value=str(row["Ticker"]).upper())
    shares_m = st.number_input("Utestående aktier (miljoner)", value=float(row["Utestående aktier"]), step=1.0)
    ps = st.number_input("P/S", value=float(row["P/S"]), step=0.01)
    ps1 = st.number_input("P/S Q1", value=float(row["P/S Q1"]), step=0.01)
    ps2 = st.number_input("P/S Q2", value=float(row["P/S Q2"]), step=0.01)
    ps3 = st.number_input("P/S Q3", value=float(row["P/S Q3"]), step=0.01)
    ps4 = st.number_input("P/S Q4", value=float(row["P/S Q4"]), step=0.01)
    rev0 = st.number_input("Omsättning idag (M)", value=float(row["Omsättning idag"]), step=1.0)
    rev1 = st.number_input("Omsättning nästa år (M)", value=float(row["Omsättning nästa år"]), step=1.0)
    antal = st.number_input("Antal aktier du äger", value=float(row["Antal aktier"]), step=1.0)

    if st.button("💾 Spara rad + beräkna"):
        # Skriv tillbaka till df via ticker-key
        mask = df["Ticker"] == row["Ticker"]
        df.loc[mask, ["Ticker", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år", "Antal aktier"]] = [
            [tkr, shares_m, ps, ps1, ps2, ps3, ps4, rev0, rev1, antal]
        ]
        df.loc[mask, "Senast manuellt uppdaterad"] = now_stamp()
        df[:] = update_calculations(df)
        try:
            save_df(ws_title, df)
            st.success("Sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    st.markdown("### Förhandsgranskning (beräkningar)")
    tmp = update_calculations(vis.copy())
    st.dataframe(
        tmp.loc[[idx], ["Ticker", "P/S-snitt (Q1..Q4)", "Omsättning om 2 år", "Omsättning om 3 år", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]],
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
    port["Andel (%)"] = np.where(tot > 0, (port["Värde (SEK)"] / tot) * 100.0, 0.0).round(2)
    port["DA (%)"] = np.where(port["Aktuell kurs"] > 0, (port["Årlig utdelning"] / port["Aktuell kurs"]) * 100.0, 0.0).round(2)

    st.markdown(f"**Totalt portföljvärde:** {round(tot, 2)} SEK")
    st.dataframe(
        port[
            [
                "Ticker",
                "Bolagsnamn",
                "Antal aktier",
                "Aktuell kurs",
                "Valuta",
                "Värde (SEK)",
                "Andel (%)",
                "Årlig utdelning",
                "DA (%)",
            ]
        ].sort_values("Andel (%)", ascending=False),
        use_container_width=True,
    )


def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag")

    if df.empty:
        st.info("Inga rader.")
        return

    # Förbered
    base = update_calculations(df.copy())
    base["DA (%)"] = np.where(base["Aktuell kurs"] > 0, (base["Årlig utdelning"] / base["Aktuell kurs"]) * 100.0, 0.0)

    subset = st.radio("Visa", ["Alla bolag", "Endast portfölj"], horizontal=True)
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()

    horizon = st.selectbox("Riktkurs-horisont", ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"], index=0)
    base = base[(base[horizon] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inget att visa för valt filter.")
        return

    base["Uppsida (%)"] = ((base[horizon] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0).round(2)

    mode = st.radio("Sortering", ["Störst uppsida", "Minst uppsida (trim/sälj)"], horizontal=True, index=0)
    asc = True if mode == "Minst uppsida (trim/sälj)" else False
    base = base.sort_values(by=["Uppsida (%)"], ascending=asc).reset_index(drop=True)

    st.dataframe(
        base[
            [
                "Ticker",
                "Bolagsnamn",
                "Aktuell kurs",
                horizon,
                "Uppsida (%)",
                "P/S-snitt (Q1..Q4)",
                "Omsättning idag",
                "Omsättning nästa år",
                "Omsättning om 2 år",
                "Omsättning om 3 år",
                "DA (%)",
            ]
        ],
        use_container_width=True,
    )

    # Kortvisning – bläddra ett bolag i taget
    st.markdown("---")
    st.markdown("### Kortvisning")
    if "idea_idx" not in st.session_state:
        st.session_state["idea_idx"] = 0
    st.session_state["idea_idx"] = st.number_input(
        "Visa rad #", min_value=0, max_value=max(0, len(base) - 1), value=st.session_state["idea_idx"], step=1
    )

    r = base.iloc[st.session_state["idea_idx"]]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    cols = st.columns(2)
    with cols[0]:
        st.write(f"- **Aktuell kurs:** {round(float(r['Aktuell kurs']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs idag:** {round(float(r['Riktkurs idag']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 1 år:** {round(float(r['Riktkurs om 1 år']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 2 år:** {round(float(r['Riktkurs om 2 år']), 2)} {r['Valuta']}")
        st.write(f"- **Riktkurs om 3 år:** {round(float(r['Riktkurs om 3 år']), 2)} {r['Valuta']}")
        st.write(f"- **Uppsida ({horizon}):** {round(float(r['Uppsida (%)']),2)} %")
    with cols[1]:
        st.write(f"- **P/S-snitt (Q1..Q4):** {round(float(r['P/S-snitt (Q1..Q4)']),2)}")
        st.write(f"- **Omsättning idag (M):** {round(float(r['Omsättning idag']),2)}")
        st.write(f"- **Omsättning nästa år (M):** {round(float(r['Omsättning nästa år']),2)}")
        st.write(f"- **Årlig utdelning:** {round(float(r['Årlig utdelning']),2)}")
        st.write(f"- **DA (egen):** {round(float(r['DA (%)']),2)} %")
        st.write(f"- **CAGR 5 år:** {round(float(r['CAGR 5 år (%)']),2)} %")


# ------------------------------- MAIN -------------------------------
def main():
    st.title("K-pf-rslag")

    # Välj blad
    titles = list_worksheet_titles() or ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    # Rates i sidopanel
    user_rates = sidebar_rates()

    # Läs data
    df = load_df(ws_title)

    # Massuppdatering från Yahoo i sidopanelen
    st.sidebar.markdown("---")
    mass_update_yahoo(df, ws_title)

    # Auto-beräkna (utan att skriva)
    df = update_calculations(df)

    # Vyer
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
