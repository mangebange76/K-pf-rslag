# app.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="K-pf-rslag", layout="wide")
st.title("K-pf-rslag")

# (Valfri) insamlingsvy – om filen saknas visar vi ett snällt meddelande.
try:
    from stockapp.manual_collect import manual_collect_view
except Exception:
    manual_collect_view = None  # type: ignore

# Sheets-helpers (URL-baserade – som din gamla app)
from stockapp.sheets import get_ws, ws_read_df, save_dataframe, list_sheet_names

# Valutakurser (ny modul)
from stockapp.rates import read_rates, save_rates, DEFAULT_RATES

# Datakällor
from stockapp.fetchers.yahoo import get_all as yahoo_get
from stockapp.fetchers.sec import get_all as sec_get  # stub

# ---------------- Hjälpare för DataFrame-schema ----------------
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad",
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({c: [] for c in FINAL_COLS})
    for kol in FINAL_COLS:
        if kol not in df.columns:
            df[kol] = 0.0 if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs"]) else ""
    return df

def to_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for i, rad in df.iterrows():
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# ---------------- Laddning & sparning ----------------
def _load_df(worksheet_name: str | None) -> pd.DataFrame:
    try:
        ws = get_ws(worksheet_name=worksheet_name)
        df = ws_read_df(ws)
    except Exception as e:
        st.warning(f"🚫 Kunde inte läsa från Google Sheet: {e}")
        df = pd.DataFrame()
    df = ensure_columns(df)
    df = to_numeric_cols(df)
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

    # Datakällor
    st.markdown("---")
    st.subheader("Datakällor")
    use_yahoo = st.checkbox("Använd Yahoo", value=True)
    use_sec   = st.checkbox("Använd SEC", value=False)

    st.markdown("---")
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
    if st.button("🚀 Uppdatera ALLA rader från valda källor"):
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

                if use_yahoo:
                    y = yahoo_get(tkr)
                    if y.get("name"):     df0.at[i, "Bolagsnamn"] = y["name"]
                    if y.get("price",0)>0: df0.at[i, "Aktuell kurs"] = y["price"]
                    if y.get("currency"): df0.at[i, "Valuta"] = y["currency"]
                    if "dividend" in y:   df0.at[i, "Årlig utdelning"] = float(y.get("dividend") or 0.0)
                    if "cagr5" in y:      df0.at[i, "CAGR 5 år (%)"] = float(y.get("cagr5") or 0.0)

                if use_sec:
                    s = sec_get(tkr)  # stub just nu
                    # här kan du mappa ev. SEC-fält när vi aktiverar det

                status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
                bar.progress((i+1)/total)

            df0 = update_calculations(df0)
            st.session_state["_df_ref"] = df0
            _save_df(df0, ws_name)

# ---------------- Första inläsning ----------------
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
        # Enkel portföljvisning med valutakurser
        rates = read_rates()
        port = df[df.get("Antal aktier", 0) > 0].copy()
        if port.empty:
            st.info("Du äger inga aktier.")
        else:
            port["Växelkurs"] = port["Valuta"].apply(lambda v: rates.get(str(v).upper(), 1.0))
            port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
            total_v = float(port["Värde (SEK)"].sum())
            if total_v > 0:
                port["Andel (%)"] = (port["Värde (SEK)"] / total_v * 100.0).round(2)
            else:
                port["Andel (%)"] = 0.0
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
