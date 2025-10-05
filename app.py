# app.py
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# yfinance för snabbdata
try:
    import yfinance as yf
except Exception:
    yf = None  # type: ignore

# Egna moduler
from stockapp.sheets import ws_read_df, ws_write_df, list_worksheet_titles
from stockapp.rates import read_rates, save_rates, fetch_live_rates, repair_rates_sheet, DEFAULT_RATES

st.set_page_config(page_title="K-pf-rslag – Stabil baseline", layout="wide")


# ------------------- Hjälpare -------------------
def _to_float(x) -> float:
    try:
        if x is None:
            return 0.0
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return 0.0
        s = s.replace(" ", "").replace("\u00A0", "")
        # ta bort tusentals-separatorer
        s = s.replace(",", ".")
        # om både punkt och komma fanns från början blir det redan fixat
        return float(s)
    except Exception:
        return 0.0


def _now_stamp() -> str:
    try:
        import pytz
        from datetime import datetime
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")


# ------------------- Kolumnschema -------------------
FINAL_COLS: List[str] = [
    "Ticker", "Bolagsnamn", "Sektor", "Valuta",
    "Antal aktier", "GAV (SEK)", "Aktuell kurs", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt (Q1..Q4)",
    "P/B", "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Årlig utdelning", "Payout (%)", "CAGR 5 år (%)",
    "Senast manuellt uppdaterad", "Senast auto uppdaterad", "Auto källa", "Senast beräknad",
    "DA (%)", "Uppsida idag (%)", "Uppsida 1 år (%)", "Uppsida 2 år (%)", "Uppsida 3 år (%)",
    "Score (Growth)", "Score (Dividend)", "Score (Financials)", "Score (Total)", "Confidence",
    "Score Total (Idag)", "Score Total (1 år)", "Score Total (2 år)", "Score Total (3 år)",
    "Score Growth (Idag)", "Score Dividend (Idag)", "Score Financials (Idag)",
    "Score Growth (1 år)", "Score Dividend (1 år)", "Score Financials (1 år)",
    "Score Growth (2 år)", "Score Dividend (2 år)", "Score Financials (2 år)",
    "Score Growth (3 år)", "Score Dividend (3 år)", "Score Financials (3 år)",
    "Div_Frekvens/år", "Div_Månader", "Div_Vikter",
    "Uppsida (%)"
]

NUM_COLS = [
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
    "Div_Frekvens/år","Uppsida (%)"
]


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # lägg till saknade
    for c in FINAL_COLS:
        if c not in out.columns:
            out[c] = "" if c not in NUM_COLS else 0.0
    # kasta bort okända (för att undvika felmappningar)
    out = out[FINAL_COLS].copy()
    # typ-konvertera
    for c in NUM_COLS:
        out[c] = out[c].map(_to_float)
    for c in ["Ticker", "Bolagsnamn", "Sektor", "Valuta", "Auto källa",
              "Senast manuellt uppdaterad", "Senast auto uppdaterad", "Senast beräknad",
              "Div_Månader", "Div_Vikter"]:
        out[c] = out[c].astype(str)
    return out


# ------------------- Yahoo snabbdata -------------------
def yahoo_quick(ticker: str) -> Dict[str, float | str]:
    out = {"Bolagsnamn":"", "Valuta":"USD", "Aktuell kurs":0.0, "Årlig utdelning":0.0, "CAGR 5 år (%)":0.0, "Utestående aktier":0.0}
    if yf is None or not ticker:
        return out
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # pris
        px = info.get("regularMarketPrice")
        if px is None:
            hist = t.history(period="1d")
            if not hist.empty and "Close" in hist.columns:
                px = float(hist["Close"].iloc[-1])
        if px is not None:
            out["Aktuell kurs"] = float(px)

        # namn/valuta
        out["Bolagsnamn"] = str(info.get("shortName") or info.get("longName") or "")
        out["Valuta"] = str(info.get("currency") or "USD").upper()

        # utdelning (årlig)
        dr = info.get("dividendRate")
        out["Årlig utdelning"] = float(dr or 0.0)

        # shares outstanding (absoluta)
        sh = info.get("sharesOutstanding")
        out["Utestående aktier"] = float(sh or 0.0) / 1e6  # konvertera till miljoner

        # CAGR 5 år på revenue (enkel approx)
        cagr = 0.0
        try:
            df = getattr(t, "income_stmt", None)
            ser = None
            if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                ser = df.loc["Total Revenue"].dropna()
            else:
                df2 = getattr(t, "financials", None)
                if isinstance(df2, pd.DataFrame) and not df2.empty and "Total Revenue" in df2.index:
                    ser = df2.loc["Total Revenue"].dropna()
            if ser is not None and len(ser) >= 2:
                ser = ser.sort_index()
                start, end = float(ser.iloc[0]), float(ser.iloc[-1])
                years = max(1, len(ser)-1)
                if start > 0:
                    cagr = ((end/start)**(1.0/years) - 1.0) * 100.0
        except Exception:
            pass
        out["CAGR 5 år (%)"] = round(float(cagr), 2)
    except Exception:
        pass
    return out


# ------------------- Beräkningar -------------------
def compute_avgs_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i, r in out.iterrows():
        # P/S-snitt
        ps_vals = [r.get("P/S Q1",0), r.get("P/S Q2",0), r.get("P/S Q3",0), r.get("P/S Q4",0)]
        ps_clean = [float(x) for x in ps_vals if _to_float(x) > 0]
        ps_avg = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        out.at[i, "P/S-snitt (Q1..Q4)"] = ps_avg

        # CAGR clamp (2–50%)
        cagr = _to_float(r.get("CAGR 5 år (%)", 0.0))
        g = min(50.0, max(2.0, cagr)) / 100.0

        # Omsättning om 2/3 år
        next_rev = _to_float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            out.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            out.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)

        shares_m = _to_float(r.get("Utestående aktier", 0.0))
        if shares_m > 0 and ps_avg > 0:
            out.at[i, "Riktkurs idag"]    = round(_to_float(r.get("Omsättning idag", 0.0))    * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 1 år"] = round(_to_float(r.get("Omsättning nästa år", 0.0)) * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 2 år"] = round(_to_float(out.at[i, "Omsättning om 2 år"])  * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 3 år"] = round(_to_float(out.at[i, "Omsättning om 3 år"])  * ps_avg / shares_m, 2)
        else:
            out.at[i, "Riktkurs idag"] = out.at[i, "Riktkurs om 1 år"] = out.at[i, "Riktkurs om 2 år"] = out.at[i, "Riktkurs om 3 år"] = 0.0

        # Uppsidor + DA
        price = _to_float(out.at[i, "Aktuell kurs"])
        if price > 0:
            out.at[i, "DA (%)"] = round((_to_float(out.at[i, "Årlig utdelning"]) / price) * 100.0, 2)
            for col_in, col_out in [
                ("Riktkurs idag", "Uppsida idag (%)"),
                ("Riktkurs om 1 år", "Uppsida 1 år (%)"),
                ("Riktkurs om 2 år", "Uppsida 2 år (%)"),
                ("Riktkurs om 3 år", "Uppsida 3 år (%)"),
            ]:
                rk = _to_float(out.at[i, col_in])
                out.at[i, col_out] = round(((rk - price) / price) * 100.0, 2) if rk > 0 else 0.0
    out["Senast beräknad"] = _now_stamp()
    return out


# ------------------- IO med cachebust -------------------
@st.cache_data(show_spinner=False)
def _load_df_cached(ws_title: str, nonce: int) -> pd.DataFrame:
    return ws_read_df(ws_title)

def load_df(ws_title: str) -> pd.DataFrame:
    n = st.session_state.get("_reload_nonce", 0)
    df = _load_df_cached(ws_title, n)
    if df.empty:
        # skapa tom DF med endast rubriker
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        ws_write_df(ws_title, df)
    return ensure_columns(df)

def save_df(ws_title: str, df: pd.DataFrame):
    ws_write_df(ws_title, df)
    st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1


# ------------------- Sidopanel: Valutor + Massuppd -------------------
def sidebar_rates_and_actions() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")
    # Läs sparade
    saved = read_rates()
    # Widgets (session-kommando för att hålla värden)
    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD", DEFAULT_RATES["USD"])), step=0.0001, format="%.6f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK", DEFAULT_RATES["NOK"])), step=0.0001, format="%.6f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD", DEFAULT_RATES["CAD"])), step=0.0001, format="%.6f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR", DEFAULT_RATES["EUR"])), step=0.0001, format="%.6f")
    rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2 = st.sidebar.columns(2)
    if c1.button("🌐 Hämta live"):
        try:
            live = fetch_live_rates()
            rates.update(live)
            save_rates(rates)
            st.sidebar.success("Livekurser hämtade och sparade.")
            st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta/spara: {e}")
    if c2.button("💾 Spara"):
        try:
            save_rates(rates)
            st.sidebar.success("Sparat.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data nu"):
        st.session_state["_reload_nonce"] = st.session_state.get("_reload_nonce", 0) + 1
        st.rerun()

    return rates


# ------------------- Manuell vy -------------------
def view_manual(df: pd.DataFrame, ws_title: str):
    st.subheader("🧩 Manuell insamling")

    # Välj/ny
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = ["➕ Lägg till nytt bolag..."] + [
        f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r["Bolagsnamn"]).strip() else str(r["Ticker"])
        for _, r in vis.iterrows()
    ]
    if "manual_idx" not in st.session_state:
        st.session_state["manual_idx"] = 0
    sel = st.selectbox("Välj bolag", list(range(len(labels))), index=st.session_state["manual_idx"],
                       format_func=lambda i: labels[i])
    st.session_state["manual_idx"] = sel
    is_new = (sel == 0)
    row = (vis.iloc[sel-1] if not is_new and not vis.empty else pd.Series({c: "" for c in FINAL_COLS}))

    # Enskild snabb uppdatering (Yahoo)
    c_upd1, c_upd2 = st.columns([1,1])
    with c_upd1:
        if not is_new and st.button("⚡ Snabbuppdatera vald (Yahoo)"):
            try:
                tkr = str(row.get("Ticker","")).upper().strip()
                if tkr:
                    quick = yahoo_quick(tkr)
                    mask = (df["Ticker"].astype(str).str.upper() == tkr)
                    if quick.get("Bolagsnamn"): df.loc[mask, "Bolagsnamn"] = str(quick["Bolagsnamn"])
                    if quick.get("Valuta"):     df.loc[mask, "Valuta"]     = str(quick["Valuta"])
                    if _to_float(quick.get("Aktuell kurs",0))>0: df.loc[mask, "Aktuell kurs"] = _to_float(quick["Aktuell kurs"])
                    df.loc[mask, "Årlig utdelning"]   = _to_float(quick.get("Årlig utdelning",0))
                    df.loc[mask, "Utestående aktier"] = _to_float(quick.get("Utestående aktier",0))
                    df.loc[mask, "CAGR 5 år (%)"]     = _to_float(quick.get("CAGR 5 år (%)",0))
                    df.loc[mask, "Senast auto uppdaterad"] = _now_stamp()
                    df.loc[mask, "Auto källa"] = "Yahoo (snabb)"
                    df2 = compute_avgs_and_targets(df)
                    save_df(ws_title, df2)
                    st.success("Vald rad uppdaterad.")
                    st.rerun()
            except Exception as e:
                st.error(f"Kunde inte uppdatera vald: {e}")
    with c_upd2:
        st.caption("Hämtar namn/valuta/kurs/utdelning/shares/CAGR från Yahoo.")

    # Form: obligatoriska överst
    st.markdown("### Obligatoriska fält")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker (Yahoo)", value=str(row.get("Ticker","")).upper() if not is_new else "")
        antal  = st.number_input("Antal aktier", value=_to_float(row.get("Antal aktier",0.0)), step=1.0, min_value=0.0)
        gav    = st.number_input("GAV (SEK)", value=_to_float(row.get("GAV (SEK)",0.0)), step=0.01, min_value=0.0, format="%.4f")
    with c2:
        oms_idag = st.number_input("Omsättning idag (M)", value=_to_float(row.get("Omsättning idag",0.0)), step=1.0, min_value=0.0)
        oms_next = st.number_input("Omsättning nästa år (M)", value=_to_float(row.get("Omsättning nästa år",0.0)), step=1.0, min_value=0.0)

    with st.expander("🌐 Fält som kan hämtas (kan lämnas tomma)"):
        cL, cR = st.columns(2)
        with cL:
            bolagsnamn = st.text_input("Bolagsnamn", value=str(row.get("Bolagsnamn","")))
            sektor     = st.text_input("Sektor", value=str(row.get("Sektor","")))
            valuta     = st.text_input("Valuta (USD/SEK/…)", value=str(row.get("Valuta","") or "USD").upper())
            aktuell    = st.number_input("Aktuell kurs", value=_to_float(row.get("Aktuell kurs",0.0)), step=0.01, min_value=0.0)
            utd_arlig  = st.number_input("Årlig utdelning", value=_to_float(row.get("Årlig utdelning",0.0)), step=0.01, min_value=0.0)
            payout     = st.number_input("Payout (%)", value=_to_float(row.get("Payout (%)",0.0)), step=1.0, min_value=0.0)
        with cR:
            utest_m    = st.number_input("Utestående aktier (M)", value=_to_float(row.get("Utestående aktier",0.0)), step=1.0, min_value=0.0)
            ps  = st.number_input("P/S", value=_to_float(row.get("P/S",0.0)), step=0.01, min_value=0.0)
            ps1 = st.number_input("P/S Q1", value=_to_float(row.get("P/S Q1",0.0)), step=0.01, min_value=0.0)
            ps2 = st.number_input("P/S Q2", value=_to_float(row.get("P/S Q2",0.0)), step=0.01, min_value=0.0)
            ps3 = st.number_input("P/S Q3", value=_to_float(row.get("P/S Q3",0.0)), step=0.01, min_value=0.0)
            ps4 = st.number_input("P/S Q4", value=_to_float(row.get("P/S Q4",0.0)), step=0.01, min_value=0.0)
            pb  = st.number_input("P/B", value=_to_float(row.get("P/B",0.0)), step=0.01, min_value=0.0)
            pb1 = st.number_input("P/B Q1", value=_to_float(row.get("P/B Q1",0.0)), step=0.01, min_value=0.0)
            pb2 = st.number_input("P/B Q2", value=_to_float(row.get("P/B Q2",0.0)), step=0.01, min_value=0.0)
            pb3 = st.number_input("P/B Q3", value=_to_float(row.get("P/B Q3",0.0)), step=0.01, min_value=0.0)
            pb4 = st.number_input("P/B Q4", value=_to_float(row.get("P/B Q4",0.0)), step=0.01, min_value=0.0)

    if st.button("💾 Spara"):
        errs = []
        if not str(ticker).strip():
            errs.append("Ticker saknas.")
        if errs:
            st.error(" | ".join(errs))
            return

        # uppdatera DF strikt per fältnamn
        if is_new:
            base = {c: ("" if c not in NUM_COLS else 0.0) for c in FINAL_COLS}
            base.update({
                "Ticker": ticker.strip().upper(),
                "Antal aktier": antal, "GAV (SEK)": gav,
                "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
                "Bolagsnamn": bolagsnamn, "Sektor": sektor, "Valuta": valuta,
                "Aktuell kurs": aktuell, "Årlig utdelning": utd_arlig, "Payout (%)": payout,
                "Utestående aktier": utest_m,
                "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
                "P/B": pb, "P/B Q1": pb1, "P/B Q2": pb2, "P/B Q3": pb3, "P/B Q4": pb4,
                "Senast manuellt uppdaterad": _now_stamp()
            })
            df = pd.concat([df, pd.DataFrame([base])], ignore_index=True)
        else:
            mask = (df["Ticker"].astype(str).str.upper() == str(ticker).upper())
            upd = {
                "Ticker": ticker.strip().upper(),
                "Antal aktier": antal, "GAV (SEK)": gav,
                "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
                "Bolagsnamn": bolagsnamn, "Sektor": sektor, "Valuta": valuta,
                "Aktuell kurs": aktuell, "Årlig utdelning": utd_arlig, "Payout (%)": payout,
                "Utestående aktier": utest_m,
                "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
                "P/B": pb, "P/B Q1": pb1, "P/B Q2": pb2, "P/B Q3": pb3, "P/B Q4": pb4,
            }
            for k, v in upd.items():
                df.loc[mask, k] = v
            df.loc[mask, "Senast manuellt uppdaterad"] = _now_stamp()

        # direkt efter spara → hämta snabbdata (namn/valuta/kurs/utd/shares/CAGR)
        try:
            tkr = str(ticker).upper().strip()
            quick = yahoo_quick(tkr)
            mask = (df["Ticker"].astype(str).str.upper() == tkr)
            if quick.get("Bolagsnamn"): df.loc[mask, "Bolagsnamn"] = str(quick["Bolagsnamn"])
            if quick.get("Valuta"):     df.loc[mask, "Valuta"]     = str(quick["Valuta"])
            if _to_float(quick.get("Aktuell kurs",0))>0: df.loc[mask, "Aktuell kurs"] = _to_float(quick["Aktuell kurs"])
            df.loc[mask, "Årlig utdelning"]   = _to_float(quick.get("Årlig utdelning",0))
            df.loc[mask, "Utestående aktier"] = _to_float(quick.get("Utestående aktier",0))
            df.loc[mask, "CAGR 5 år (%)"]     = _to_float(quick.get("CAGR 5 år (%)",0))
            df.loc[mask, "Senast auto uppdaterad"] = _now_stamp()
            df.loc[mask, "Auto källa"] = "Yahoo (snabb)"
        except Exception:
            pass

        # beräkningar + spara
        df2 = ensure_columns(df)
        df2 = compute_avgs_and_targets(df2)
        save_df(ws_title, df2)
        st.success("Sparat.")
        st.rerun()


# ------------------- Data & Massuppdatering -------------------
def view_data(df: pd.DataFrame, ws_title: str):
    st.subheader("📄 Data")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Uppdatera ALLA (Yahoo – pris/utdelning/shares/CAGR)"):
        try:
            status = st.empty()
            bar = st.progress(0.0)
            total = max(1, len(df))
            for i, r in df.iterrows():
                tkr = str(r["Ticker"]).strip().upper()
                if not tkr:
                    bar.progress((i+1)/total); continue
                status.write(f"Hämtar {tkr} …")
                q = yahoo_quick(tkr)
                if q.get("Bolagsnamn"): df.at[i, "Bolagsnamn"] = str(q["Bolagsnamn"])
                if q.get("Valuta"):     df.at[i, "Valuta"]     = str(q["Valuta"])
                if _to_float(q.get("Aktuell kurs",0))>0: df.at[i, "Aktuell kurs"] = _to_float(q["Aktuell kurs"])
                df.at[i, "Årlig utdelning"]   = _to_float(q.get("Årlig utdelning",0))
                df.at[i, "Utestående aktier"] = _to_float(q.get("Utestående aktier",0))
                df.at[i, "CAGR 5 år (%)"]     = _to_float(q.get("CAGR 5 år (%)",0))
                df.at[i, "Senast auto uppdaterad"] = _now_stamp()
                df.at[i, "Auto källa"] = "Yahoo (snabb)"
                time.sleep(0.6)  # artigt
                bar.progress((i+1)/total)
            df2 = compute_avgs_and_targets(df)
            save_df(ws_title, df2)
            st.success("Alla rader uppdaterade.")
            st.rerun()
        except Exception as e:
            st.error(f"Massuppdatering misslyckades: {e}")


# ------------------- Portfölj -------------------
def view_portfolio(df: pd.DataFrame, rates: Dict[str, float]):
    st.subheader("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier ännu.")
        return
    # växelkurs
    port["VX"] = port["Valuta"].map(lambda v: rates.get(str(v).upper(), 1.0))
    # värden
    port["Värde (SEK)"] = port["Antal aktier"].map(_to_float) * port["Aktuell kurs"].map(_to_float) * port["VX"].map(_to_float)
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"].map(_to_float) * port["GAV (SEK)"].map(_to_float)
    port["Vinst (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst (%)"] = np.where(port["Anskaffningsvärde (SEK)"]>0,
                                 (port["Vinst (SEK)"]/port["Anskaffningsvärde (SEK)"])*100.0, 0.0)
    tot = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(tot>0, (port["Värde (SEK)"]/tot)*100.0, 0.0)

    st.markdown(f"**Totalt portföljvärde:** {round(tot,2)} SEK")
    st.markdown(f"**Totalt anskaffningsvärde:** {round(float(port['Anskaffningsvärde (SEK)'].sum()),2)} SEK")
    st.markdown(f"**Total vinst:** {round(float(port['Vinst (SEK)'].sum()),2)} SEK "
                f"({round(float(np.where(float(port['Anskaffningsvärde (SEK)'].sum())>0, (port['Vinst (SEK)'].sum()/port['Anskaffningsvärde (SEK)'].sum())*100.0, 0.0)),2)} %)")

    cols = ["Ticker","Bolagsnamn","Antal aktier","GAV (SEK)","Aktuell kurs","Valuta",
            "Värde (SEK)","Anskaffningsvärde (SEK)","Vinst (SEK)","Vinst (%)","Andel (%)"]
    st.dataframe(port[cols].sort_values("Andel (%)", ascending=False), use_container_width=True)


# ------------------- Idévy (enkel) -------------------
def view_ideas(df: pd.DataFrame):
    st.subheader("💡 Köpförslag (enkel)")
    if df.empty:
        st.info("Inga rader.")
        return
    horizon = st.selectbox("Riktkurs-horisont", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    subset = st.radio("Visa", ["Alla bolag","Endast portfölj"], horizontal=True)
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()
    base = base[(base[horizon] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inget att visa.")
        return
    base["Uppsida (%)"] = ((base[horizon] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0).round(2)
    st.dataframe(base[["Ticker","Bolagsnamn","Sektor","Aktuell kurs",horizon,"Uppsida (%)"]].sort_values("Uppsida (%)", ascending=False), use_container_width=True)


# ------------------- main -------------------
def main():
    st.title("K-pf-rslag – Stabil baseline")

    # Välj blad
    try:
        titles = list_worksheet_titles() or ["Blad1"]
    except Exception:
        titles = ["Blad1"]
    ws_title = st.sidebar.selectbox("Google Sheets → välj data-blad", titles, index=0)

    # Valutor & globala actions
    user_rates = sidebar_rates_and_actions()

    # Läs data + säkerställ kolumner + beräkna
    df = load_df(ws_title)
    df = ensure_columns(df)
    df = compute_avgs_and_targets(df)

    tabs = st.tabs(["📄 Data", "🧩 Manuell", "📦 Portfölj", "💡 Förslag"])
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
