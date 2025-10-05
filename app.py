from __future__ import annotations

import io
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# yfinance för snabbdata
try:
    import yfinance as yf
except Exception:
    yf = None  # hanteras i UI

# ------------------------------------------------------------
# Tidsstämplar (Stockholm om tillgängligt)
# ------------------------------------------------------------
def now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

# ------------------------------------------------------------
# Kolumnschema (fast, inga “smart tolkningar”)
# ------------------------------------------------------------
FINAL_COLS: List[str] = [
    # Identitet & portfölj
    "Ticker","Bolagsnamn","Sektor","Valuta",
    "Antal aktier","GAV (SEK)","Aktuell kurs","Utestående aktier",  # Utestående i miljoner

    # P/S & P/B (kvartal + snitt)
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt (Q1..Q4)",
    "P/B","P/B Q1","P/B Q2","P/B Q3","P/B Q4","P/B-snitt (Q1..Q4)",

    # Omsättning (M)
    "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",

    # Riktkurser (i bolagets valuta)
    "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",

    # Utdelning
    "Årlig utdelning","Payout (%)",

    # Övrigt
    "CAGR 5 år (%)",

    # Tidsstämplar
    "Senast manuellt uppdaterad","Senast auto uppdaterad","Auto källa","Senast beräknad",

    # Derived/visning
    "DA (%)","Uppsida idag (%)","Uppsida 1 år (%)","Uppsida 2 år (%)","Uppsida 3 år (%)",
    "Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence",

    # Sparade score (alla horisonter)
    "Score Total (Idag)","Score Total (1 år)","Score Total (2 år)","Score Total (3 år)",
    "Score Growth (Idag)","Score Dividend (Idag)","Score Financials (Idag)",
    "Score Growth (1 år)","Score Dividend (1 år)","Score Financials (1 år)",
    "Score Growth (2 år)","Score Dividend (2 år)","Score Financials (2 år)",
    "Score Growth (3 år)","Score Dividend (3 år)","Score Financials (3 år)",

    # Utdelningsschema-plats (kan vara tomt)
    "Div_Frekvens/år","Div_Månader","Div_Vikter",
]

NUMERIC_DEFAULTS = {
    "Antal aktier": 0.0, "GAV (SEK)": 0.0, "Aktuell kurs": 0.0, "Utestående aktier": 0.0,
    "P/S": 0.0, "P/S Q1": 0.0, "P/S Q2": 0.0, "P/S Q3": 0.0, "P/S Q4": 0.0, "P/S-snitt (Q1..Q4)": 0.0,
    "P/B": 0.0, "P/B Q1": 0.0, "P/B Q2": 0.0, "P/B Q3": 0.0, "P/B Q4": 0.0, "P/B-snitt (Q1..Q4)": 0.0,
    "Omsättning idag": 0.0, "Omsättning nästa år": 0.0, "Omsättning om 2 år": 0.0, "Omsättning om 3 år": 0.0,
    "Riktkurs idag": 0.0, "Riktkurs om 1 år": 0.0, "Riktkurs om 2 år": 0.0, "Riktkurs om 3 år": 0.0,
    "Årlig utdelning": 0.0, "Payout (%)": 0.0, "CAGR 5 år (%)": 0.0,
    "DA (%)": 0.0, "Uppsida idag (%)": 0.0, "Uppsida 1 år (%)": 0.0, "Uppsida 2 år (%)": 0.0, "Uppsida 3 år (%)": 0.0,
    "Score (Growth)": 0.0, "Score (Dividend)": 0.0, "Score (Financials)": 0.0, "Score (Total)": 0.0, "Confidence": 0.0,
    "Score Total (Idag)": 0.0,"Score Total (1 år)": 0.0,"Score Total (2 år)": 0.0,"Score Total (3 år)": 0.0,
    "Score Growth (Idag)": 0.0,"Score Dividend (Idag)": 0.0,"Score Financials (Idag)": 0.0,
    "Score Growth (1 år)": 0.0,"Score Dividend (1 år)": 0.0,"Score Financials (1 år)": 0.0,
    "Score Growth (2 år)": 0.0,"Score Dividend (2 år)": 0.0,"Score Financials (2 år)": 0.0,
    "Score Growth (3 år)": 0.0,"Score Dividend (3 år)": 0.0,"Score Financials (3 år)": 0.0,
    "Div_Frekvens/år": 0.0,
}

TEXT_DEFAULTS = {
    "Ticker":"", "Bolagsnamn":"", "Sektor":"", "Valuta":"",
    "Senast manuellt uppdaterad":"", "Senast auto uppdaterad":"", "Auto källa":"", "Senast beräknad":"",
    "Div_Månader":"", "Div_Vikter":""
}

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # skapa alla kolumner som saknas
    for c in FINAL_COLS:
        if c not in out.columns:
            if c in NUMERIC_DEFAULTS:
                out[c] = NUMERIC_DEFAULTS[c]
            elif c in TEXT_DEFAULTS:
                out[c] = TEXT_DEFAULTS[c]
            else:
                # default: text
                out[c] = ""
    # typning: numeriska
    for c, d in NUMERIC_DEFAULTS.items():
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(d)
    # typning: text
    for c in TEXT_DEFAULTS.keys():
        out[c] = out[c].astype(str)
    # rensa Ticker
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    return out

# ------------------------------------------------------------
# CSV IO (ingen Sheets)
# ------------------------------------------------------------
def empty_db() -> pd.DataFrame:
    cols = {c: (NUMERIC_DEFAULTS[c] if c in NUMERIC_DEFAULTS else TEXT_DEFAULTS.get(c,"")) for c in FINAL_COLS}
    return pd.DataFrame([cols]).iloc[0:0]

def load_csv_uploaded(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    return ensure_columns(raw)

def download_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ------------------------------------------------------------
# Yahoo helpers
# ------------------------------------------------------------
def _safe_float(x) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def yahoo_quick(ticker: str) -> Dict[str, float | str]:
    """
    Hämtar: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%),
    Utestående aktier (miljoner), P/S, P/B
    """
    out: Dict[str, float | str] = {
        "Bolagsnamn":"", "Valuta":"USD", "Aktuell kurs":0.0, "Årlig utdelning":0.0,
        "CAGR 5 år (%)":0.0, "Utestående aktier":0.0, "P/S":0.0, "P/B":0.0
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

        # namn/valuta/price
        out["Bolagsnamn"] = str(info.get("shortName") or info.get("longName") or "")
        out["Valuta"] = str(info.get("currency") or "USD").upper()

        price = info.get("regularMarketPrice")
        if price is None:
            h = t.history(period="1d")
            if hasattr(h, "empty") and not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        out["Aktuell kurs"] = _safe_float(price)

        # utdelning
        dr = info.get("dividendRate")
        out["Årlig utdelning"] = _safe_float(dr)

        # shares (miljoner)
        sh = _safe_float(info.get("sharesOutstanding"))
        out["Utestående aktier"] = sh / 1e6 if sh > 0 else 0.0

        # multiplar
        out["P/S"] = _safe_float(info.get("priceToSalesTrailing12Months"))
        out["P/B"] = _safe_float(info.get("priceToBook"))

        # CAGR 5 år (~Total Revenue, annual)
        cagr = 0.0
        for attr in ("income_stmt", "financials"):
            try:
                df = getattr(t, attr, None)
                if isinstance(df, pd.DataFrame) and not df.empty and "Total Revenue" in df.index:
                    s = df.loc["Total Revenue"].dropna().sort_index()
                    if len(s) >= 2:
                        start = float(s.iloc[0]); end = float(s.iloc[-1]); years = max(1, len(s)-1)
                        if start > 0:
                            cagr = ((end/start)**(1.0/years) - 1.0)*100.0
                            break
            except Exception:
                pass
        out["CAGR 5 år (%)"] = round(cagr, 2) if cagr != 0.0 else 0.0
    except Exception:
        pass
    return out

# ------------------------------------------------------------
# Beräkningar (ingen “smartness”, exakt som definierat)
# ------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def compute_ps_pb_snitt(row: pd.Series) -> Tuple[float, float]:
    ps = [row.get("P/S Q1",0), row.get("P/S Q2",0), row.get("P/S Q3",0), row.get("P/S Q4",0)]
    ps = [float(x) for x in ps if float(x) > 0]
    pb = [row.get("P/B Q1",0), row.get("P/B Q2",0), row.get("P/B Q3",0), row.get("P/B Q4",0)]
    pb = [float(x) for x in pb if float(x) > 0]
    return (round(np.mean(ps),2) if ps else 0.0, round(np.mean(pb),2) if pb else 0.0)

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for i, r in out.iterrows():
        ps_avg, pb_avg = compute_ps_pb_snitt(r)
        out.at[i, "P/S-snitt (Q1..Q4)"] = ps_avg
        out.at[i, "P/B-snitt (Q1..Q4)"] = pb_avg

        # CAGR clamp (2%–50%)
        cagr = float(r.get("CAGR 5 år (%)", 0.0))
        g = clamp(cagr, 2.0, 50.0) / 100.0

        # Omsättning år 2/3
        next_rev = float(r.get("Omsättning nästa år", 0.0))
        if next_rev > 0:
            out.at[i, "Omsättning om 2 år"] = round(next_rev * (1.0 + g), 2)
            out.at[i, "Omsättning om 3 år"] = round(next_rev * ((1.0 + g) ** 2), 2)
        else:
            out.at[i, "Omsättning om 2 år"] = float(r.get("Omsättning om 2 år", 0.0))
            out.at[i, "Omsättning om 3 år"] = float(r.get("Omsättning om 3 år", 0.0))

        # Riktkurser (P/S-snitt * oms / utest aktier)
        shares_m = float(r.get("Utestående aktier", 0.0))
        if shares_m > 0 and ps_avg > 0:
            out.at[i, "Riktkurs idag"]    = round(float(r.get("Omsättning idag", 0.0))     * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 1 år"] = round(float(r.get("Omsättning nästa år", 0.0)) * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 2 år"] = round(float(out.at[i, "Omsättning om 2 år"])   * ps_avg / shares_m, 2)
            out.at[i, "Riktkurs om 3 år"] = round(float(out.at[i, "Omsättning om 3 år"])   * ps_avg / shares_m, 2)
        else:
            out.at[i, "Riktkurs idag"] = out.at[i, "Riktkurs om 1 år"] = out.at[i, "Riktkurs om 2 år"] = out.at[i, "Riktkurs om 3 år"] = 0.0
    # Uppsida + DA
    price = out["Aktuell kurs"].replace(0, np.nan)
    for col, lab in [("Riktkurs idag","Uppsida idag (%)"),
                     ("Riktkurs om 1 år","Uppsida 1 år (%)"),
                     ("Riktkurs om 2 år","Uppsida 2 år (%)"),
                     ("Riktkurs om 3 år","Uppsida 3 år (%)")]:
        rk = out[col].replace(0, np.nan)
        out[lab] = ((rk - price) / price * 100.0).fillna(0.0)

    out["DA (%)"] = np.where(out["Aktuell kurs"] > 0, (out["Årlig utdelning"] / out["Aktuell kurs"]) * 100.0, 0.0)
    out["Senast beräknad"] = now_stamp()
    return out

# ------------------------------------------------------------
# Poäng (enkel, som tidigare – inga “hemliga” regler)
# ------------------------------------------------------------
def score_rows(df: pd.DataFrame, horizon: str, strategy: str) -> pd.DataFrame:
    out = df.copy()
    out["Uppsida (%)"] = np.where(out["Aktuell kurs"]>0, (out[horizon]-out["Aktuell kurs"])/out["Aktuell kurs"]*100.0, 0.0)
    cur_ps = out["P/S"].replace(0, np.nan)
    ps_avg = out["P/S-snitt (Q1..Q4)"].replace(0, np.nan)
    cheap_ps = (ps_avg / (cur_ps * 2.0)).clip(upper=1.0).fillna(0.0)
    g_norm = (out["CAGR 5 år (%)"] / 30.0).clip(0, 1)
    u_norm = (out["Uppsida (%)"] / 50.0).clip(0, 1)
    out["Score (Growth)"] = (0.4*g_norm + 0.4*u_norm + 0.2*cheap_ps) * 100.0

    payout = out["Payout (%)"]
    payout_health = 1 - (abs(payout - 60.0) / 60.0)
    payout_health = payout_health.clip(0, 1)
    payout_health = np.where(out["Payout (%)"]<=0, 0.85, payout_health)
    y_norm = (out["DA (%)"] / 8.0).clip(0, 1)
    grow_ok = np.where(out["CAGR 5 år (%)"] >= 0, 1.0, 0.6)
    out["Score (Dividend)"] = (0.6*y_norm + 0.3*payout_health + 0.1*grow_ok) * 100.0

    cur_pb = out["P/B"].replace(0, np.nan)
    pb_avg = out["P/B-snitt (Q1..Q4)"].replace(0, np.nan)
    cheap_pb = (pb_avg / (cur_pb * 2.0)).clip(upper=1.0).fillna(0.0)
    out["Score (Financials)"] = (0.7*cheap_pb + 0.3*u_norm) * 100.0

    # viktning
    def w(strategy: str, sektor: str) -> Tuple[float,float,float]:
        s = strategy.lower()
        if "tillväxt" in s: return (0.70,0.10,0.20)
        if "utdelning" in s: return (0.15,0.70,0.15)
        if "finans" in s:    return (0.20,0.20,0.60)
        # auto via sektor:
        sec = (sektor or "").lower()
        if any(k in sec for k in ["bank","finans","insurance","financial"]): return (0.25,0.25,0.50)
        if any(k in sec for k in ["utility","telecom","consumer staples"]):  return (0.20,0.60,0.20)
        if any(k in sec for k in ["tech","semiconductor","software"]):       return (0.70,0.10,0.20)
        return (0.45,0.35,0.20)

    totals = []
    for _, r in out.iterrows():
        wg, wd, wf = w(strategy, str(r.get("Sektor","")))
        t = wg*r["Score (Growth)"] + wd*r["Score (Dividend)"] + wf*r["Score (Financials)"]
        totals.append(round(t, 2))
    out["Score (Total)"] = totals

    need = [
        out["Aktuell kurs"]>0,
        out["P/S-snitt (Q1..Q4)"]>0,
        out["Omsättning idag"]>=0,
        out["Omsättning nästa år"]>=0,
    ]
    present = np.stack(need, axis=0).astype(float)
    out["Confidence"] = (present.mean(axis=0)*100.0).round(0)
    return out

# ------------------------------------------------------------
# UI: Valutakurser (manuella, ingen Sheets)
# ------------------------------------------------------------
def sidebar_rates() -> Dict[str,float]:
    st.sidebar.subheader("💱 Valutakurser → SEK")
    if "rates" not in st.session_state:
        st.session_state["rates"] = {"USD":9.75,"NOK":0.95,"CAD":7.05,"EUR":11.18,"SEK":1.0}
    r = st.session_state["rates"]
    r["USD"] = st.sidebar.number_input("USD → SEK", value=float(r["USD"]), step=0.0001, format="%.6f")
    r["NOK"] = st.sidebar.number_input("NOK → SEK", value=float(r["NOK"]), step=0.0001, format="%.6f")
    r["CAD"] = st.sidebar.number_input("CAD → SEK", value=float(r["CAD"]), step=0.0001, format="%.6f")
    r["EUR"] = st.sidebar.number_input("EUR → SEK", value=float(r["EUR"]), step=0.0001, format="%.6f")
    st.session_state["rates"] = r
    return r

def rate_for(currency: str, rates: Dict[str,float]) -> float:
    return rates.get(str(currency).upper(), 1.0)

# ------------------------------------------------------------
# UI: Data, manuell insamling, portfölj, idéer
# ------------------------------------------------------------
def view_data(df: pd.DataFrame):
    st.subheader("📄 Data")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Ladda ned CSV", data=download_csv(df), file_name="data.csv", mime="text/csv")

def view_manual(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🧩 Manuell insamling")
    vis = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" if str(r.get("Bolagsnamn","")).strip() else r["Ticker"] for _, r in vis.iterrows()]
    labels = ["➕ Lägg till nytt bolag..."] + labels

    if "manual_idx" not in st.session_state:
        st.session_state["manual_idx"] = 0

    sel = st.selectbox("Välj bolag att redigera", list(range(len(labels))), format_func=lambda i: labels[i],
                       index=st.session_state["manual_idx"])
    st.session_state["manual_idx"] = sel
    is_new = (sel == 0)
    row = (pd.Series({c: df.iloc[0][c]*0 for c in df.columns}) if is_new else vis.iloc[sel-1])

    # Obligatoriska fält – DINA
    st.markdown("### Obligatoriska fält (dina)")
    c1,c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker (Yahoo)", value=str(row.get("Ticker","") if not is_new else ""))
        antal  = st.number_input("Antal aktier (du äger)", value=float(row.get("Antal aktier",0.0)), step=1.0, min_value=0.0)
        gav    = st.number_input("GAV (SEK)", value=float(row.get("GAV (SEK)",0.0)), step=0.01, min_value=0.0, format="%.4f")
    with c2:
        oms_idag = st.number_input("Omsättning idag (M)", value=float(row.get("Omsättning idag",0.0)), step=1.0, min_value=0.0)
        oms_nxt  = st.number_input("Omsättning nästa år (M)", value=float(row.get("Omsättning nästa år",0.0)), step=1.0, min_value=0.0)

    # Auto-fält i expander (visas, men kan korrigeras)
    with st.expander("🌐 Auto-fält (kan överstyras)"):
        cL,cR = st.columns(2)
        with cL:
            bolagsnamn = st.text_input("Bolagsnamn", value=str(row.get("Bolagsnamn","")))
            sektor     = st.text_input("Sektor", value=str(row.get("Sektor","")))
            valuta     = st.text_input("Valuta", value=str(row.get("Valuta","") or "USD").upper())
            aktuell    = st.number_input("Aktuell kurs", value=float(row.get("Aktuell kurs",0.0)), step=0.01, min_value=0.0)
            utd        = st.number_input("Årlig utdelning", value=float(row.get("Årlig utdelning",0.0)), step=0.01, min_value=0.0)
            payout     = st.number_input("Payout (%)", value=float(row.get("Payout (%)",0.0)), step=1.0, min_value=0.0)
        with cR:
            utest_m    = st.number_input("Utestående aktier (miljoner)", value=float(row.get("Utestående aktier",0.0)), step=0.01, min_value=0.0)
            ps         = st.number_input("P/S", value=float(row.get("P/S",0.0)), step=0.01, min_value=0.0)
            ps1        = st.number_input("P/S Q1", value=float(row.get("P/S Q1",0.0)), step=0.01, min_value=0.0)
            ps2        = st.number_input("P/S Q2", value=float(row.get("P/S Q2",0.0)), step=0.01, min_value=0.0)
            ps3        = st.number_input("P/S Q3", value=float(row.get("P/S Q3",0.0)), step=0.01, min_value=0.0)
            ps4        = st.number_input("P/S Q4", value=float(row.get("P/S Q4",0.0)), step=0.01, min_value=0.0)
            pb         = st.number_input("P/B", value=float(row.get("P/B",0.0)), step=0.01, min_value=0.0)
            pb1        = st.number_input("P/B Q1", value=float(row.get("P/B Q1",0.0)), step=0.01, min_value=0.0)
            pb2        = st.number_input("P/B Q2", value=float(row.get("P/B Q2",0.0)), step=0.01, min_value=0.0)
            pb3        = st.number_input("P/B Q3", value=float(row.get("P/B Q3",0.0)), step=0.01, min_value=0.0)
            pb4        = st.number_input("P/B Q4", value=float(row.get("P/B Q4",0.0)), step=0.01, min_value=0.0)
            cagr       = st.number_input("CAGR 5 år (%)", value=float(row.get("CAGR 5 år (%)",0.0)), step=0.1)

    # Spara & snabb Yahoo
    cA,cB,cC = st.columns([1,1,1])
    do_save = cA.button("💾 Spara")
    do_quick = cB.button("⚡ Snabb Yahoo (pris/valuta/namn/utd/multiplar)")
    do_all = cC.button("🔭 Full Yahoo för alla rader (0.5s delay)")

    updated_df = df.copy()

    # Spara en rad
    if do_save:
        if not ticker.strip():
            st.error("Ticker krävs.")
            return df
        mask = (updated_df["Ticker"] == ticker.upper())
        data = {
            "Ticker": ticker.upper(),
            "Antal aktier": float(antal),
            "GAV (SEK)": float(gav),
            "Omsättning idag": float(oms_idag),
            "Omsättning nästa år": float(oms_nxt),
            "Bolagsnamn": bolagsnamn, "Sektor": sektor, "Valuta": valuta,
            "Aktuell kurs": float(aktuell), "Årlig utdelning": float(utd), "Payout (%)": float(payout),
            "Utestående aktier": float(utest_m),
            "P/S": float(ps), "P/S Q1": float(ps1), "P/S Q2": float(ps2), "P/S Q3": float(ps3), "P/S Q4": float(ps4),
            "P/B": float(pb), "P/B Q1": float(pb1), "P/B Q2": float(pb2), "P/B Q3": float(pb3), "P/B Q4": float(pb4),
            "CAGR 5 år (%)": float(cagr),
            "Senast manuellt uppdaterad": now_stamp(),
        }
        if mask.any():
            for k,v in data.items():
                updated_df.loc[mask, k] = v
        else:
            base = {c: (NUMERIC_DEFAULTS[c] if c in NUMERIC_DEFAULTS else TEXT_DEFAULTS.get(c,"")) for c in FINAL_COLS}
            base.update(data)
            updated_df = pd.concat([updated_df, pd.DataFrame([base])], ignore_index=True)
        st.success("Sparat.")
        updated_df = update_calculations(updated_df)

    # Snabb Yahoo på vald ticker
    if do_quick and ticker.strip():
        y = yahoo_quick(ticker.strip().upper())
        mask = (updated_df["Ticker"] == ticker.strip().upper())
        if not mask.any():
            st.error("Ticker finns inte i tabellen – spara först.")
        else:
            for src, dst in [
                ("Bolagsnamn","Bolagsnamn"), ("Valuta","Valuta"),
                ("Aktuell kurs","Aktuell kurs"), ("Årlig utdelning","Årlig utdelning"),
                ("CAGR 5 år (%)","CAGR 5 år (%)"), ("Utestående aktier","Utestående aktier"),
                ("P/S","P/S"), ("P/B","P/B"),
            ]:
                v = y.get(src)
                if v is not None:
                    updated_df.loc[mask, dst] = v
            updated_df.loc[mask, "Senast auto uppdaterad"] = now_stamp()
            updated_df.loc[mask, "Auto källa"] = "Yahoo (snabb)"
            st.success("Snabbdata hämtad.")
            updated_df = update_calculations(updated_df)

    # Full Yahoo för alla (pris/valuta/namn/utd/multiplar/shares/CAGR)
    if do_all:
        if yf is None:
            st.error("yfinance saknas i miljön.")
        else:
            status = st.empty(); bar = st.progress(0.0)
            for i, r in updated_df.iterrows():
                tkr = str(r["Ticker"]).strip().upper()
                if not tkr: continue
                status.write(f"Hämtar {i+1}/{len(updated_df)} – {tkr}")
                y = yahoo_quick(tkr)
                for src, dst in [
                    ("Bolagsnamn","Bolagsnamn"), ("Valuta","Valuta"),
                    ("Aktuell kurs","Aktuell kurs"), ("Årlig utdelning","Årlig utdelning"),
                    ("CAGR 5 år (%)","CAGR 5 år (%)"), ("Utestående aktier","Utestående aktier"),
                    ("P/S","P/S"), ("P/B","P/B"),
                ]:
                    v = y.get(src)
                    if v is not None:
                        updated_df.at[i, dst] = v
                updated_df.at[i, "Senast auto uppdaterad"] = now_stamp()
                updated_df.at[i, "Auto källa"] = "Yahoo (snabb)"
                time.sleep(0.5)
                bar.progress((i+1)/len(updated_df))
            st.success("Klart.")
            updated_df = update_calculations(updated_df)

    return ensure_columns(updated_df)

def view_portfolio(df: pd.DataFrame, rates: Dict[str,float]):
    st.subheader("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["SEK-kurs"] = port["Valuta"].apply(lambda v: rate_for(v, rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["SEK-kurs"]
    port["Anskaffning (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]
    total_value = float(port["Värde (SEK)"].sum())
    total_cost  = float(port["Anskaffning (SEK)"].sum())
    pnl_kr = total_value - total_cost
    pnl_pct = (pnl_kr / total_cost * 100.0) if total_cost > 0 else 0.0

    port["P/L (SEK)"] = (port["Värde (SEK)"] - port["Anskaffning (SEK)"])
    port["P/L (%)"]   = np.where(port["Anskaffning (SEK)"]>0, (port["P/L (SEK)"]/port["Anskaffning (SEK)"])*100.0, 0.0)
    port["Andel (%)"] = np.where(total_value>0, (port["Värde (SEK)"]/total_value)*100.0, 0.0)

    st.markdown(f"**Portföljvärde:** {round(total_value,2)} SEK")
    st.markdown(f"**Anskaffningsvärde:** {round(total_cost,2)} SEK")
    st.markdown(f"**Resultat:** {round(pnl_kr,2)} SEK ({round(pnl_pct,2)} %)")

    cols = ["Ticker","Bolagsnamn","Antal aktier","Valuta","Aktuell kurs","SEK-kurs",
            "GAV (SEK)","Anskaffning (SEK)","Värde (SEK)","P/L (SEK)","P/L (%)","Andel (%)"]
    st.dataframe(port[cols].sort_values("Andel (%)", ascending=False), use_container_width=True)

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
    base["Uppsida (%)"] = ((base[horizon] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0).round(2)
    base["DA (%)"] = np.where(base["Aktuell kurs"] > 0, (base["Årlig utdelning"]/base["Aktuell kurs"])*100.0, 0.0).round(2)

    sort_on = st.selectbox("Sortera på", ["Score (Total)","Uppsida (%)","DA (%)"], index=0)
    ascending = st.checkbox("Omvänd sortering", value=False)
    base = base.sort_values(by=[sort_on], ascending=ascending).reset_index(drop=True)

    cols = ["Ticker","Bolagsnamn","Sektor","Aktuell kurs",horizon,"Uppsida (%)","DA (%)",
            "P/S-snitt (Q1..Q4)","P/B-snitt (Q1..Q4)","CAGR 5 år (%)","Score (Growth)","Score (Dividend)","Score (Financials)","Score (Total)","Confidence"]
    st.dataframe(base[cols], use_container_width=True)

    st.markdown("---")
    st.markdown("### Kortvisning")
    if "idea_idx" not in st.session_state:
        st.session_state["idea_idx"] = 0
    st.session_state["idea_idx"] = st.number_input("Visa #", min_value=0, max_value=max(0, len(base)-1),
                                                   value=st.session_state["idea_idx"], step=1)
    r = base.iloc[st.session_state["idea_idx"]]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    c1,c2 = st.columns(2)
    with c1:
        st.write(f"- **Sektor:** {r.get('Sektor','—')}")
        st.write(f"- **Aktuell kurs:** {round(float(r['Aktuell kurs']),2)} {r['Valuta']}")
        st.write(f"- **Riktkurs ({horizon}):** {round(float(r[horizon]),2)} {r['Valuta']}")
        st.write(f"- **Uppsida ({horizon}):** {round(float(r['Uppsida (%)']),2)} %")
    with c2:
        st.write(f"- **P/S-snitt (Q1..Q4):** {round(float(r['P/S-snitt (Q1..Q4)']),2)}")
        st.write(f"- **P/B-snitt (Q1..Q4):** {round(float(r['P/B-snitt (Q1..Q4)']),2)}")
        st.write(f"- **CAGR 5 år:** {round(float(r['CAGR 5 år (%)']),2)} %")
        st.write(f"- **DA:** {round(float(r['DA (%)']),2)} %")
        st.write(f"- **Score – G/D/F/T:** "
                 f"{round(float(r['Score (Growth)']),1)} / {round(float(r['Score (Dividend)']),1)} / "
                 f"{round(float(r['Score (Financials)']),1)} / **{round(float(r['Score (Total)']),1)}** "
                 f"(Conf {int(r['Confidence'])}%)")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="K-pf-rslag (standalone)", layout="wide")
    st.title("K-pf-rslag – Standalone (utan Sheets)")

    # Valutakurser
    rates = sidebar_rates()

    # Data in / out
    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader("Ladda upp CSV (valfritt)", type=["csv"])
    if uploaded and "db" not in st.session_state:
        try:
            st.session_state["db"] = load_csv_uploaded(uploaded)
            st.sidebar.success("CSV inläst.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte läsa CSV: {e}")

    if "db" not in st.session_state:
        st.session_state["db"] = empty_db()

    if st.sidebar.button("🧹 Starta om (töm i minnet)"):
        st.session_state["db"] = empty_db()
        st.experimental_rerun()

    # Meny
    tab_data, tab_manual, tab_port, tab_ideas = st.tabs(["📄 Data","🧩 Manuell insamling","📦 Portfölj","💡 Köpförslag"])
    with tab_data:
        st.session_state["db"] = ensure_columns(st.session_state["db"])
        st.session_state["db"] = update_calculations(st.session_state["db"])
        view_data(st.session_state["db"])
    with tab_manual:
        st.session_state["db"] = ensure_columns(st.session_state["db"])
        st.session_state["db"] = view_manual(st.session_state["db"])
    with tab_port:
        st.session_state["db"] = ensure_columns(st.session_state["db"])
        st.session_state["db"] = update_calculations(st.session_state["db"])
        view_portfolio(st.session_state["db"], rates)
    with tab_ideas:
        st.session_state["db"] = ensure_columns(st.session_state["db"])
        st.session_state["db"] = update_calculations(st.session_state["db"])
        view_ideas(st.session_state["db"])

if __name__ == "__main__":
    main()
