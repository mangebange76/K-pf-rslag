# app.py
from __future__ import annotations

import math
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

# Valutakurser
from stockapp.rates import read_rates, save_rates, DEFAULT_RATES

# Datakällor
from stockapp.fetchers.yahoo import get_all as yahoo_get
from stockapp.fetchers.sec import get_all as sec_get  # stub (ej använd nu)

# -------------------------------- Schema --------------------------------
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs",
    "Utestående aktier (milj.)",
    "Market Cap",
    "P/S (TTM)", "P/B", "EV/EBITDA (ttm)",
    "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)",
    "Revenue TTM (M)", "Revenue growth (%)",
    "Book value / share",
    "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
    # Din äldre modell – låter dem vara kvar för kompatibilitet
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad",
]

NUMERIC_COLS = [
    "Aktuell kurs",
    "Utestående aktier (milj.)",
    "Market Cap",
    "P/S (TTM)", "P/B", "EV/EBITDA (ttm)",
    "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)",
    "Revenue TTM (M)", "Revenue growth (%)",
    "Book value / share",
    "Gross margin (%)", "Operating margin (%)", "Net margin (%)",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier",
    "CAGR 5 år (%)", "P/S-snitt",
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

# ---------------------------- Kalkyl-hjälpare ----------------------------
def update_calculations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    # P/S-snitt och riktkurser enligt din tidigare modell (om du använder dem)
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

        aktier_ut_m = float(rad.get("Utestående aktier (milj.)", 0.0))
        if aktier_ut_m > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut_m, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut_m, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut_m, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut_m, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# ----------------------------- I/O mot Sheets ----------------------------
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

# ------------------------------- Sidopanel -------------------------------
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
    if st.button("🚀 Uppdatera ALLA rader från Yahoo"):
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

                y = yahoo_get(tkr)
                # Bas
                if y.get("name"):      df0.at[i, "Bolagsnamn"] = y["name"]
                if y.get("currency"):  df0.at[i, "Valuta"] = y["currency"]
                if y.get("price", 0)>0: df0.at[i, "Aktuell kurs"] = y["price"]

                # Storlek/aktier
                sh_out = float(y.get("shares_outstanding") or 0.0)
                if sh_out > 0:
                    df0.at[i, "Utestående aktier (milj.)"] = sh_out / 1e6
                if y.get("market_cap", 0)>0:
                    df0.at[i, "Market Cap"] = float(y["market_cap"])

                # Multiplar/nyckeltal
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
                    df0.at[i, dst_col] = v

                # TTM & growth (M = miljoner i aktiens valuta)
                rev_ttm = float(y.get("revenue_ttm") or 0.0)
                if rev_ttm > 0:
                    df0.at[i, "Revenue TTM (M)"] = rev_ttm / 1e6
                df0.at[i, "Revenue growth (%)"] = float(y.get("revenue_growth_pct") or 0.0)

                # CAGR 5 år (från finansiella när möjligt)
                df0.at[i, "CAGR 5 år (%)"] = float(y.get("cagr5_pct") or 0.0)

                status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
                bar.progress((i+1)/total)

            df0 = update_calculations(df0)
            st.session_state["_df_ref"] = df0
            _save_df(df0, ws_name)

# ------------------------------- Första läsning --------------------------
if "_df_ref" not in st.session_state:
    st.session_state["_df_ref"] = _load_df(st.secrets.get("WORKSHEET_NAME") or "Blad1")

# ------------------------------- Strategi-mål ----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def target_price_growth(row: pd.Series, ps_target: float) -> float:
    rev_m = float(row.get("Revenue TTM (M)", 0.0))
    gr_pct = float(row.get("Revenue growth (%)", 0.0))
    sh_m  = float(row.get("Utestående aktier (milj.)", 0.0))
    if rev_m <= 0 or sh_m <= 0 or ps_target <= 0:
        return 0.0
    g = clamp(gr_pct / 100.0, -0.20, 0.50)  # -20% .. +50%
    rev_next_m = rev_m * (1.0 + g)
    return (ps_target * rev_next_m) / sh_m

def target_price_dividend(row: pd.Series, target_yield_pct: float) -> float:
    div = float(row.get("Årlig utdelning", 0.0))
    yld = target_yield_pct / 100.0
    if div <= 0 or yld <= 0:
        return 0.0
    return div / yld

def target_price_ev_ebitda(row: pd.Series, target_multiple: float) -> float:
    """Använder relationen: Equity_target = Mcap + (Target*EBITDA - EV_now)"""
    ebitda = float(row.get("EV/EBITDA (ttm)", 0.0))  # OBS: detta är EV/EBITDA, inte EBITDA
    # Vi behöver EV och EBITDA separat – de finns i Yahoo-responsen men vi lagrar inte EBITDA solo.
    # Approximation: EV_now/EBITDA_now = EV/EBITDA (ttm) -> EV_now = multiple_now * EBITDA
    # Men utan "multiple_now" blir det känsligt. Därför beräknar vi direkt med fälten vid uppdatering:
    # För säkerhets skull: om vi saknar depositionsfält, returnera 0.
    # Läs back-fyllda råfält från uppdateringsrundan (om de fanns).
    ev_now = float(row.get("_y_ev_now", 0.0))
    ebitda_now = float(row.get("_y_ebitda_now", 0.0))
    mcap_now = float(row.get("Market Cap", 0.0))
    sh_m = float(row.get("Utestående aktier (milj.)", 0.0))
    if ebitda_now <= 0 or ev_now <= 0 or mcap_now <= 0 or sh_m <= 0 or target_multiple <= 0:
        return 0.0
    ev_target = target_multiple * ebitda_now
    equity_target = mcap_now + (ev_target - ev_now)
    return (equity_target / (sh_m * 1e6)) if (sh_m > 0) else 0.0  # pris per aktie

def target_price_pb(row: pd.Series, pb_target: float) -> float:
    bvps = float(row.get("Book value / share", 0.0))
    if bvps <= 0 or pb_target <= 0:
        return 0.0
    return pb_target * bvps

def proposal_table(df: pd.DataFrame, strategy: str,
                   ps_target: float, div_target_yield: float,
                   ev_ebitda_target: float, pb_target: float,
                   rates: dict) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    tmp = df.copy()

    # Targetpris per strategi
    if strategy == "Tillväxt (P/S)":
        tmp["Målpris"] = tmp.apply(lambda r: target_price_growth(r, ps_target), axis=1)
    elif strategy == "Utdelning":
        tmp["Målpris"] = tmp.apply(lambda r: target_price_dividend(r, div_target_yield), axis=1)
    elif strategy == "EV/EBITDA":
        tmp["Målpris"] = tmp.apply(lambda r: target_price_ev_ebitda(r, ev_ebitda_target), axis=1)
    else:  # "P/B (Finans)"
        tmp["Målpris"] = tmp.apply(lambda r: target_price_pb(r, pb_target), axis=1)

    # Uppsida
    tmp["Uppsida (%)"] = np.where(tmp["Aktuell kurs"] > 0,
                                  (tmp["Målpris"] - tmp["Aktuell kurs"]) / tmp["Aktuell kurs"] * 100.0,
                                  0.0)

    # SEK-konvertering (visning)
    tmp["Växelkurs"] = tmp["Valuta"].apply(lambda v: rates.get(str(v).upper(), 1.0))
    tmp["Kurs (SEK)"] = (tmp["Aktuell kurs"] * tmp["Växelkurs"]).round(2)
    tmp["Målpris (SEK)"] = (tmp["Målpris"] * tmp["Växelkurs"]).round(2)

    # Relevanta kolumner per strategi
    common = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Kurs (SEK)","Målpris","Målpris (SEK)","Uppsida (%)"]
    if strategy == "Tillväxt (P/S)":
        cols = common + ["P/S (TTM)", "Revenue TTM (M)", "Revenue growth (%)", "Utestående aktier (milj.)", "Gross margin (%)"]
    elif strategy == "Utdelning":
        cols = common + ["Årlig utdelning","Dividend yield (%)","Payout ratio (%)","Profit margin (%)"] if "Profit margin (%)" in tmp.columns else common + ["Årlig utdelning","Dividend yield (%)","Payout ratio (%)"]
    elif strategy == "EV/EBITDA":
        cols = common + ["EV/EBITDA (ttm)","Market Cap","Utestående aktier (milj.)"]
    else:
        cols = common + ["P/B","Book value / share"]

    # Sortera på uppsida
    tmp = tmp.sort_values(by="Uppsida (%)", ascending=False)
    return tmp[ [c for c in cols if c in tmp.columns] ].reset_index(drop=True)

# ------------------------------------ UI ---------------------------------
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

with tab_suggest:
    df = st.session_state.get("_df_ref", pd.DataFrame())
    if df.empty:
        st.info("Ingen data.")
    else:
        st.subheader("Strategi & parametrar")
        c1, c2, c3, c4 = st.columns(4)
        strategy = c1.selectbox("Strategi", ["Tillväxt (P/S)", "Utdelning", "EV/EBITDA", "P/B (Finans)"])

        ps_target = c2.slider("PS-mål (Tillväxt)", 1.0, 15.0, 6.0, 0.5)
        div_target = c3.slider("Målyield % (Utdelning)", 2.0, 10.0, 4.0, 0.1)
        ev_target = c4.slider("EV/EBITDA-mål", 6.0, 20.0, 12.0, 0.5)

        pb_target = st.slider("P/B-mål (Finans)", 0.5, 2.5, 1.2, 0.1)

        st.markdown("---")
        rates = read_rates()
        table = proposal_table(
            df, strategy,
            ps_target=ps_target,
            div_target_yield=div_target,
            ev_ebitda_target=ev_target,
            pb_target=pb_target,
            rates=rates
        )
        if table.empty:
            st.info("Inget målpris kunde beräknas – saknas nyckeltal för valda bolag/strategi.")
        else:
            st.dataframe(table, use_container_width=True)
