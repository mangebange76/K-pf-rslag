from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# --------- Externa fetchers ---------
try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="K-pf-rslag", layout="wide")

# --------- Projektmoduler ---------
from stockapp.sheets import ws_read_df, ws_write_df, list_worksheet_titles

from stockapp.rates import (
    read_rates, save_rates, fetch_live_rates, repair_rates_sheet, DEFAULT_RATES
)

try:
    from stockapp.manual_collect import manual_collect_view
except Exception:
    manual_collect_view = None


# ============ Hjälpare & konstanter ============
def _num(x, default=0.0) -> float:
    """Robust str->float som hanterar svensk formatering (tusentalspunkt/komma)."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(" ", "")
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "")
                s = s.replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        return float(default)


def _get_rate(cur: str, rates: Dict[str, float]) -> float:
    if not cur:
        return 1.0
    return float(rates.get(str(cur).upper(), DEFAULT_RATES.get(str(cur).upper(), 1.0)))


BASE_COLS: List[str] = [
    "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs",
    "Utestående aktier (milj.)", "Utestående aktier",  # stöd båda
    "Omsättning idag (M)", "Omsättning nästa år (M)",
    "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "P/B Q1", "P/B Q2", "P/B Q3", "P/B Q4",
    "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)",
    "CAGR 5 år (%)", "Antal aktier",
    # beräknade:
    "P/S-snitt (Q1..Q4)", "P/B-snitt (Q1..Q4)",
    "Omsättning om 2 år (M)", "Omsättning om 3 år (M)",
    "Riktkurs idag (PS)", "Riktkurs om 1 år (PS)", "Riktkurs om 2 år (PS)", "Riktkurs om 3 år (PS)",
    "Riktkurs idag (PB)", "Riktkurs om 1 år (PB)", "Riktkurs om 2 år (PB)", "Riktkurs om 3 år (PB)",
]


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in BASE_COLS:
        if c not in df.columns:
            if any(k in c.lower() for k in ["ticker", "bolags", "valuta"]):
                df[c] = ""
            else:
                df[c] = 0.0
    return df


# ============ Google Sheets IO ============
def select_worksheet() -> str:
    try:
        titles = list_worksheet_titles()
        titles = [t for t in titles if t] or ["Blad1"]
    except Exception:
        titles = ["Blad1"]
    return st.sidebar.selectbox("Välj data-blad:", titles, index=0, key="ws_title")


@st.cache_data(ttl=300, show_spinner=False)
def load_df_cached(ws_title: str) -> pd.DataFrame:
    df = ws_read_df(ws_title)
    if df is None or df.empty:
        df = pd.DataFrame(columns=BASE_COLS)
    return ensure_cols(df)


def load_df(ws_title: str) -> pd.DataFrame:
    return load_df_cached(ws_title)


def save_df(ws_title: str, df: pd.DataFrame):
    ws_write_df(ws_title, df)
    try:
        st.cache_data.clear()
    except Exception:
        pass


# ============ Beräkningar ============
def ps_snitt(row: pd.Series) -> float:
    vals = [_num(row.get("P/S Q1")), _num(row.get("P/S Q2")),
            _num(row.get("P/S Q3")), _num(row.get("P/S Q4"))]
    vals = [v for v in vals if v > 0]
    return round(float(np.mean(vals)) if vals else 0.0, 4)


def pb_snitt(row: pd.Series) -> float:
    vals = [_num(row.get("P/B Q1")), _num(row.get("P/B Q2")),
            _num(row.get("P/B Q3")), _num(row.get("P/B Q4"))]
    vals = [v for v in vals if v > 0]
    return round(float(np.mean(vals)) if vals else 0.0, 4)


def project_revenues(next_year_M: float, cagr_pct: float):
    """Dämpad CAGR: år2 = next*(1+g), år3 = next*(1+0.7g)."""
    g = float(cagr_pct) / 100.0
    if next_year_M <= 0:
        return 0.0, 0.0
    y2 = next_year_M * (1.0 + g)
    y3 = next_year_M * (1.0 + min(g * 0.7, g))
    return round(y2, 6), round(y3, 6)


def shares_millions(row: pd.Series) -> float:
    v = _num(row.get("Utestående aktier (milj.)", 0.0))
    if v <= 0:
        raw = _num(row.get("Utestående aktier", 0.0))
        if raw > 0:
            v = raw / 1_000_000.0
    return max(0.0, v)


def price_target_ps(revenue_M: float, ps: float, sh_mill: float) -> float:
    if revenue_M <= 0 or ps <= 0 or sh_mill <= 0:
        return 0.0
    return round((revenue_M * ps) / sh_mill, 4)


def price_target_pb(pb: float, book_value_per_share: float) -> float:
    if pb <= 0 or book_value_per_share <= 0:
        return 0.0
    return round(pb * book_value_per_share, 4)


def recalc_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["P/S-snitt (Q1..Q4)"] = df.apply(ps_snitt, axis=1)
    df["P/B-snitt (Q1..Q4)"] = df.apply(pb_snitt, axis=1)

    y2, y3 = [], []
    for _, r in df.iterrows():
        a, b = project_revenues(_num(r.get("Omsättning nästa år (M)")),
                                _num(r.get("CAGR 5 år (%)")))
        y2.append(a); y3.append(b)
    df["Omsättning om 2 år (M)"] = y2
    df["Omsättning om 3 år (M)"] = y3

    rk_id, rk_1, rk_2, rk_3 = [], [], [], []
    for _, r in df.iterrows():
        sh = shares_millions(r)
        ps = _num(r.get("P/S-snitt (Q1..Q4)"))
        rk_id.append(price_target_ps(_num(r.get("Omsättning idag (M)")), ps, sh))
        rk_1.append(price_target_ps(_num(r.get("Omsättning nästa år (M)")), ps, sh))
        rk_2.append(price_target_ps(_num(r.get("Omsättning om 2 år (M)")), ps, sh))
        rk_3.append(price_target_ps(_num(r.get("Omsättning om 3 år (M)")), ps, sh))
    df["Riktkurs idag (PS)"] = rk_id
    df["Riktkurs om 1 år (PS)"] = rk_1
    df["Riktkurs om 2 år (PS)"] = rk_2
    df["Riktkurs om 3 år (PS)"] = rk_3

    # PB-riktkurser om BVPS finns
    if "BVPS" in df.columns:
        rkpb_id, rkpb_1, rkpb_2, rkpb_3 = [], [], [], []
        for _, r in df.iterrows():
            pb = _num(r.get("P/B-snitt (Q1..Q4)"))
            bvps = _num(r.get("BVPS"))
            rkpb_id.append(price_target_pb(pb, bvps))
            rkpb_1.append(price_target_pb(pb, bvps))
            rkpb_2.append(price_target_pb(pb, bvps))
            rkpb_3.append(price_target_pb(pb, bvps))
        df["Riktkurs idag (PB)"] = rkpb_id
        df["Riktkurs om 1 år (PB)"] = rkpb_1
        df["Riktkurs om 2 år (PB)"] = rkpb_2
        df["Riktkurs om 3 år (PB)"] = rkpb_3
    else:
        for c in ["Riktkurs idag (PB)", "Riktkurs om 1 år (PB)", "Riktkurs om 2 år (PB)", "Riktkurs om 3 år (PB)"]:
            if c not in df.columns:
                df[c] = 0.0

    return df


def update_all_prices(df: pd.DataFrame, delay_sec: float = 0.5) -> pd.DataFrame:
    """Massuppdatera 'Aktuell kurs' från yfinance med lagom throttling."""
    if yf is None:
        st.warning("yfinance saknas – kan inte hämta livekurser.")
        return df

    tickers = [str(t).strip() for t in df["Ticker"].fillna("").tolist() if str(t).strip()]
    if not tickers:
        st.info("Inga tickers i bladet.")
        return df

    bar = st.progress(0.0)
    status = st.empty()
    out = df.copy()

    for i, tkr in enumerate(tickers, start=1):
        try:
            status.write(f"Hämtar {i}/{len(tickers)} – {tkr}")
            t = yf.Ticker(tkr)

            price = None
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
                out.loc[out["Ticker"] == tkr, "Aktuell kurs"] = float(price)

        except Exception:
            pass

        bar.progress(i / len(tickers))
        time.sleep(max(0.0, delay_sec))

    status.empty()
    return out


# ============ UI ============
def build_rates_sidebar() -> Dict[str, float]:
    st.sidebar.markdown("### 💱 Valutakurser → SEK")

    if st.sidebar.button("🌐 Hämta livekurser"):
        try:
            live = fetch_live_rates()
            st.session_state["rates_prefill"] = {k: float(live.get(k, DEFAULT_RATES[k]))
                                                 for k in ["USD", "NOK", "CAD", "EUR", "SEK"]}
            st.sidebar.success("Livekurser hämtade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte hämta livekurser: {e}")
        st.rerun()

    if st.sidebar.button("↻ Läs sparade kurser"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        saved = read_rates()
        st.session_state["rates_prefill"] = {k: float(saved.get(k, DEFAULT_RATES[k]))
                                             for k in ["USD", "NOK", "CAD", "EUR", "SEK"]}
        st.sidebar.success("Sparade kurser inlästa.")
        st.rerun()

    prefill = st.session_state.get("rates_prefill") or read_rates()

    usd = st.sidebar.number_input("USD → SEK", value=float(prefill.get("USD", DEFAULT_RATES["USD"])),
                                  step=0.0001, format="%.6f", key="rate_usd")
    nok = st.sidebar.number_input("NOK → SEK", value=float(prefill.get("NOK", DEFAULT_RATES["NOK"])),
                                  step=0.0001, format="%.6f", key="rate_nok")
    cad = st.sidebar.number_input("CAD → SEK", value=float(prefill.get("CAD", DEFAULT_RATES["CAD"])),
                                  step=0.0001, format="%.6f", key="rate_cad")
    eur = st.sidebar.number_input("EUR → SEK", value=float(prefill.get("EUR", DEFAULT_RATES["EUR"])),
                                  step=0.0001, format="%.6f", key="rate_eur")

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara valutakurser"):
            try:
                save_rates({"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0})
                st.session_state["rates_prefill"] = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}
                st.sidebar.success("Valutakurser sparade till Google Sheets.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte spara kurser: {e}")
    with c2:
        if st.button("🛠 Reparera bladet"):
            try:
                repair_rates_sheet()
                st.cache_data.clear()
                st.sidebar.success("Rates-bladet reparerat.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte reparera: {e}")

    return {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}


def main():
    st.title("K-pf-rslag")

    # Sidopanel – valutakurser
    rates = build_rates_sidebar()

    # Välj Google Sheets-blad & läs data
    ws_title = select_worksheet()
    df = load_df(ws_title)

    tabs = st.tabs(["📄 Data", "🧩 Manuell insamling", "📦 Portfölj", "💡 Köpförslag"])

    # ----- DATA -----
    with tabs[0]:
        st.subheader("Databasen")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("🔁 Läs in från Sheets igen"):
                st.cache_data.clear()
                st.rerun()
        with col_b:
            if st.button("📈 Uppdatera alla aktiekurser (0.5s/ticker)"):
                df2 = update_all_prices(df, delay_sec=0.5)
                if not df2.equals(df):
                    save_df(ws_title, df2)
                    st.success("Kurser uppdaterade och sparade.")
                    df = df2
        with col_c:
            if st.button("🧮 Räkna om & spara"):
                df2 = recalc_df(df)
                save_df(ws_title, df2)
                st.success("Beräkningar uppdaterade och sparade.")
                df = df2

        st.dataframe(df, use_container_width=True)

    # ----- MANUELL INSAMLING -----
    with tabs[1]:
        st.subheader("Manuell insamling")
        if manual_collect_view is None:
            st.info("Yahoo/SEC-fetchers/modul för manuell insamling är inte laddad i denna miljö.")
        else:
            try:
                df_after = manual_collect_view(df)
                if isinstance(df_after, pd.DataFrame):
                    save_df(ws_title, df_after)
                    st.success("Sparat uppdaterad data från manuell insamling.")
                    df = df_after
            except Exception as e:
                st.error(f"Fel i manuell insamling: {e}")

    # ----- PORTFÖLJ -----
    with tabs[2]:
        st.subheader("Min portfölj")
        port = df.copy()
        if "Antal aktier" in port.columns:
            port = port[port["Antal aktier"].apply(_num) > 0].copy()
        else:
            port = port.iloc[0:0].copy()

        if port.empty:
            st.info("Du har inga innehav registrerade (kolumnen 'Antal aktier').")
        else:
            port["Kurs SEK"] = port.apply(lambda r: _num(r.get("Aktuell kurs")) * _get_rate(r.get("Valuta"), rates), axis=1)
            port["Värde (SEK)"] = port.apply(lambda r: _num(r.get("Antal aktier")) * r.get("Kurs SEK", 0.0), axis=1)
            total = float(port["Värde (SEK)"].sum())
            if total > 0:
                port["Andel (%)"] = (port["Värde (SEK)"] / total * 100.0).round(2)

            st.metric("Totalt portföljvärde (SEK)", f"{round(total,2):,.2f}")
            st.dataframe(
                port[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Kurs SEK", "Värde (SEK)", "Andel (%)"]],
                use_container_width=True
            )

    # ----- KÖPFÖRSLAG -----
    with tabs[3]:
        st.subheader("Köpförslag")

        horizon = st.radio("Riktkurs", ["Idag", "Om 1 år", "Om 2 år", "Om 3 år"], horizontal=True, index=0)
        rk_col = {
            "Idag": "Riktkurs idag (PS)",
            "Om 1 år": "Riktkurs om 1 år (PS)",
            "Om 2 år": "Riktkurs om 2 år (PS)",
            "Om 3 år": "Riktkurs om 3 år (PS)",
        }[horizon]

        base = recalc_df(df)
        base = base[(base[rk_col] > 0) & (base["Aktuell kurs"] > 0)].copy()

        if base.empty:
            st.info("Inga bolag med nödvändiga data för förslag ännu.")
        else:
            base["Uppsida (%)"] = (base[rk_col] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
            base = base.sort_values(by="Uppsida (%)", ascending=False).reset_index(drop=True)

            if "suggest_idx" not in st.session_state:
                st.session_state.suggest_idx = 0

            colp, coltxt, coln = st.columns([1, 2, 1])
            with colp:
                if st.button("⬅️ Föregående"):
                    st.session_state.suggest_idx = max(0, st.session_state.suggest_idx - 1)
            with coln:
                if st.button("➡️ Nästa"):
                    st.session_state.suggest_idx = min(len(base) - 1, st.session_state.suggest_idx + 1)
            st.caption(f"Post {st.session_state.suggest_idx+1}/{len(base)}")

            r = base.iloc[st.session_state.suggest_idx]
            cols_show = [
                "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs",
                "P/S-snitt (Q1..Q4)", "P/B-snitt (Q1..Q4)",
                "Omsättning idag (M)", "Omsättning nästa år (M)", "Omsättning om 2 år (M)", "Omsättning om 3 år (M)",
                "Riktkurs idag (PS)", "Riktkurs om 1 år (PS)", "Riktkurs om 2 år (PS)", "Riktkurs om 3 år (PS)",
                "Riktkurs idag (PB)", "Riktkurs om 1 år (PB)", "Riktkurs om 2 år (PB)", "Riktkurs om 3 år (PB)",
                "Årlig utdelning", "Dividend yield (%)", "Payout ratio (%)", "CAGR 5 år (%)", "Uppsida (%)"
            ]
            st.dataframe(pd.DataFrame([r[cols_show]]), use_container_width=True)

            st.markdown("### Topp 25 efter uppsida")
            st.dataframe(
                base[["Ticker", "Bolagsnamn", "Aktuell kurs", rk_col, "Uppsida (%)",
                      "P/S-snitt (Q1..Q4)", "P/B-snitt (Q1..Q4)"]].head(25),
                use_container_width=True
            )


if __name__ == "__main__":
    main()
