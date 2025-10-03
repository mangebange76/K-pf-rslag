# app.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="K-pf-rslag", layout="wide")
st.title("K-pf-rslag")

# Valfri insamlingsvy
try:
    from stockapp.manual_collect import manual_collect_view
except Exception:
    manual_collect_view = None  # type: ignore

from stockapp.sheets import get_ws, ws_read_df, save_dataframe, list_sheet_names
from stockapp.rates import read_rates, save_rates, DEFAULT_RATES
from stockapp.fetchers.yahoo import get_all as yahoo_get
from stockapp.fetchers.sec import get_pb_quarters  # <-- NYTT

# ---------------- Schema ----------------
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs",
    "Utestående aktier (milj.)", "Market Cap",
    "P/S (TTM)", "P/B", "EV/EBITDA (ttm)",
    "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)",
    "Revenue TTM (M)", "Revenue growth (%)", "Book value / share",
    "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
    # NYTT: P/B historik
    "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",
    # Befintliga fält för kompatibilitet
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "CAGR 5 år (%)", "P/S-snitt", "Senast manuellt uppdaterad",
]

NUMERIC_COLS = [
    "Aktuell kurs", "Utestående aktier (milj.)", "Market Cap",
    "P/S (TTM)", "P/B", "EV/EBITDA (ttm)",
    "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)",
    "Revenue TTM (M)", "Revenue growth (%)", "Book value / share",
    "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
    "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4", "P/B-snitt (Q1..Q4)",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "CAGR 5 år (%)", "P/S-snitt",
]

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
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for i, rad in df.iterrows():
        # P/S-snitt (om du använder de manuella Q1..Q4-fälten)
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        df.at[i, "P/S-snitt"] = round(np.mean(ps_clean), 2) if ps_clean else 0.0

        # P/B-snitt – NYTT
        pbs = [rad.get("P/B Q1", 0), rad.get("P/B Q2", 0), rad.get("P/B Q3", 0), rad.get("P/B Q4", 0)]
        pb_clean = [float(x) for x in pbs if float(x) > 0]
        df.at[i, "P/B-snitt (Q1..Q4)"] = round(np.mean(pb_clean), 2) if pb_clean else 0.0
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
    return df

def _save_df(df: pd.DataFrame, worksheet_name: str | None) -> None:
    try:
        save_dataframe(df, worksheet_name=worksheet_name)
        st.success("Sparat till Google Sheets.")
    except Exception as e:
        st.warning(f"⚠️ Kunde inte spara: {e}")

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

    # Valutakurser
    st.markdown("---")
    st.subheader("💱 Valutakurser → SEK")
    rates = read_rates()
    usd = st.number_input("USD → SEK", value=float(rates.get("USD", DEFAULT_RATES["USD"])), step=0.01, format="%.4f")
    nok = st.number_input("NOK → SEK", value=float(rates.get("NOK", DEFAULT_RATES["NOK"])), step=0.01, format="%.4f")
    cad = st.number_input("CAD → SEK", value=float(rates.get("CAD", DEFAULT_RATES["CAD"])), step=0.01, format="%.4f")
    eur = st.number_input("EUR → SEK", value=float(rates.get("EUR", DEFAULT_RATES["EUR"])), step=0.01, format="%.4f")
    new_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Spara valutakurser"):
            save_rates(new_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.success("Valutakurser sparade.")
    with c2:
        if st.button("↻ Läs sparade kurser"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    use_sec_pb = st.checkbox("Beräkna P/B 4Q via SEC", value=True, help="Hämtar equity & aktier per period från SEC och pris från Yahoo.")

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
    if st.button("🚀 Uppdatera ALLA rader (Yahoo + ev. SEC P/B)"):
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

                # --- Yahoo ---
                y = yahoo_get(tkr)
                if y.get("name"):      df0.at[i, "Bolagsnamn"] = y["name"]
                if y.get("currency"):  df0.at[i, "Valuta"] = y["currency"]
                if y.get("price", 0)>0: df0.at[i, "Aktuell kurs"] = y["price"]

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
                    df0.at[i, dst_col] = float(y.get(src_key) or 0.0)

                rev_ttm = float(y.get("revenue_ttm") or 0.0)
                if rev_ttm > 0:
                    df0.at[i, "Revenue TTM (M)"] = rev_ttm / 1e6
                df0.at[i, "Revenue growth (%)"] = float(y.get("revenue_growth_pct") or 0.0)
                df0.at[i, "CAGR 5 år (%)"] = float(y.get("cagr5_pct") or 0.0)

                # --- SEC P/B historik (valfritt) ---
                if use_sec_pb:
                    pbdata = get_pb_quarters(tkr)
                    pbs = pbdata.get("pb_quarters", [])
                    # pbs är lista [(datum, pb), ...] nyast först upp till 4 st
                    # Vi mappar till Q1..Q4 där Q4 = nyast för konsekvens med dina bilder
                    # (Om du hellre vill Q1=nyast, byt ordning nedan.)
                    p_values = [float(x[1]) for x in pbs]
                    # Fyll bakifrån så vi alltid har 4 kolumner
                    q = [0.0, 0.0, 0.0, 0.0]
                    for idx, val in enumerate(reversed(p_values[-4:])):
                        q[idx] = round(val, 2)
                    df0.at[i, "P/B Q1"] = q[0]
                    df0.at[i, "P/B Q2"] = q[1]
                    df0.at[i, "P/B Q3"] = q[2]
                    df0.at[i, "P/B Q4"] = q[3]

                status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
                bar.progress((i+1)/total)

            df0 = update_calculations(df0)
            st.session_state["_df_ref"] = df0
            _save_df(df0, ws_name)

# ---------------- Första läsning ----------------
if "_df_ref" not in st.session_state:
    st.session_state["_df_ref"] = _load_df(st.secrets.get("WORKSHEET_NAME") or "Blad1")

# ---------------- Flikar ----------------
tab_data, tab_collect, tab_port = st.tabs(["📄 Data", "🧩 Manuell insamling", "📦 Portfölj"])

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
            st.session_state["_df_ref"] = df_out
            st.success("Uppdaterade sessionens data.")

with tab_port:
    df = st.session_state.get("_df_ref", pd.DataFrame())
    if df.empty:
        st.info("Ingen data.")
    else:
        rates = read_rates()
        port = df[df.get("Antal aktier", 0) > 0].copy()
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
