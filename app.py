# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Robust imports med fallback/guards
# ------------------------------------------------------------
# config
try:
    from stockapp.config import (
        APP_TITLE,
        FINAL_COLS,
        MANUAL_PROGNOS_FIELDS,
        DISPLAY_CURRENCY,
    )
except Exception:
    # Fallback – minimikrav för att appen ska gå igång
    APP_TITLE = "K-pf-rslag"
    DISPLAY_CURRENCY = "SEK"
    FINAL_COLS = [
        "Ticker",
        "Bolagsnamn",
        "Valuta",
        "Kurs",
        "Aktuell kurs",
        "Antal aktier",
        "Market Cap",
        "Market Cap (SEK)",
        "P/S",
        "P/S Q1",
        "P/S Q2",
        "P/S Q3",
        "P/S Q4",
        "P/S-snitt (Q1..Q4)",
        "GAV (SEK)",
        "Omsättning i år (est.)",
        "Omsättning nästa år (est.)",
        "Sektor",
        "Risklabel",
        "Senast manuellt uppdaterad",
        "Senast auto-uppdaterad",
        "Senast uppdaterad källa",
    ]
    MANUAL_PROGNOS_FIELDS = ["Omsättning i år (est.)", "Omsättning nästa år (est.)"]

# storage / utils / rates
from stockapp.storage import hamta_data, spara_data
from stockapp.utils import (
    add_oldest_ts_col,
    dedupe_tickers,
    ensure_schema,
    format_large_number,
    now_stamp,
    parse_date,
    safe_float,
    stamp_fields_ts,
    risk_label_from_mcap,
)
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)

# batch (ersätter tidigare "update")
from stockapp.batch import (
    sidebar_batch_controls,
    runner_price,
    runner_full,
)

# investeringsförslag
from stockapp.invest import visa_investeringsforslag


# ------------------------------------------------------------
# Hjälpfunktioner (lokala)
# ------------------------------------------------------------
def _init_state_defaults():
    """Sätt upp session_state nycklar innan widgets skapas."""
    if "_df_ref" not in st.session_state:
        st.session_state["_df_ref"] = pd.DataFrame(columns=FINAL_COLS)

    # valutakurser – seedas EN gång från sparade
    if "_rates_seeded" not in st.session_state:
        saved = las_sparade_valutakurser()
        for key in ("USD", "EUR", "NOK", "CAD", "SEK"):
            st.session_state[f"rate_{key}"] = float(saved.get(key, 1.0))
        st.session_state["_rates_seeded"] = True

    # batchkö
    st.session_state.setdefault("batch_queue", [])
    st.session_state.setdefault("batch_order_mode", "Äldst först")
    st.session_state.setdefault("batch_size", 10)

    # vyer
    st.session_state.setdefault("view", "Investeringsförslag")
    st.session_state.setdefault("page", 1)
    st.session_state.setdefault("page_size", 5)

    # edit/bläddra
    st.session_state.setdefault("edit_index", 0)


def _load_df() -> pd.DataFrame:
    """Hämta df från Google Sheet – säkra schema och varna om problem."""
    try:
        df = hamta_data()
        df = ensure_schema(df, FINAL_COLS)
    except Exception as e:
        st.warning(f"⚠️ Kunde inte läsa data från Google Sheet: {e}")
        df = pd.DataFrame(columns=FINAL_COLS)

    # dubblettskydd (i minnet)
    df2, dups = dedupe_tickers(df)
    if dups:
        st.info(f"ℹ️ Dubbletter ignoreras i minnet: {', '.join(dups)}")
    return df2


def _save_df(df: pd.DataFrame):
    """Spara df till Google Sheet – robust med backoff."""
    try:
        spara_data(df)
        st.success("✅ Ändringar sparade.")
    except Exception as e:
        st.error(f"🚫 Kunde inte spara till Google Sheet: {e}")


def _recompute_derived(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """Beräkna enkla derivatkolumner: P/S-snitt och Market Cap (SEK) m.m."""
    out = df.copy()

    # P/S-snitt (Q1..Q4)
    for c in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
        if c not in out.columns:
            out[c] = np.nan
    out["P/S-snitt (Q1..Q4)"] = pd.to_numeric(
        out[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1), errors="coerce"
    )

    # Market Cap (SEK) från Market Cap & Valuta
    def _mcap_sek(row):
        mcap = safe_float(row.get("Market Cap"), np.nan)
        cur = str(row.get("Valuta", "SEK")).upper()
        if math.isnan(mcap):
            return np.nan
        rate = hamta_valutakurs(cur, user_rates)
        return mcap * float(rate)

    if "Market Cap (SEK)" in out.columns:
        out["Market Cap (SEK)"] = out.apply(_mcap_sek, axis=1)
    else:
        out.insert(len(out.columns), "Market Cap (SEK)", out.apply(_mcap_sek, axis=1))

    # Risklabel om saknas
    if "Risklabel" not in out.columns:
        out["Risklabel"] = out["Market Cap"].apply(risk_label_from_mcap) if "Market Cap" in out.columns else "Unknown"
    else:
        # fyll saknade
        mask = out["Risklabel"].isna() | (out["Risklabel"].astype(str).str.strip() == "")
        if "Market Cap" in out.columns:
            out.loc[mask, "Risklabel"] = out.loc[mask, "Market Cap"].apply(risk_label_from_mcap)

    return out


def _sidebar_rates() -> Dict[str, float]:
    """Sidopanel för valutakurser (utan experimental_rerun)."""
    with st.sidebar.expander("💱 Valutakurser (→ SEK)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Hämta automatiskt"):
                fetched, misses, provider = hamta_valutakurser_auto()
                # seed state innan widgets ritas om:
                for k, v in fetched.items():
                    st.session_state[f"rate_{k}"] = float(v)
                spara_valutakurser(fetched)
                if misses:
                    st.warning("Kunde inte hämta: " + ", ".join(misses))
                st.toast(f"Valutor uppdaterade via {provider}.")
                st.rerun()

        usd = st.number_input(
            "USD",
            key="rate_USD",
            value=float(st.session_state["rate_USD"]),
            step=0.01,
        )
        eur = st.number_input(
            "EUR",
            key="rate_EUR",
            value=float(st.session_state["rate_EUR"]),
            step=0.01,
        )
        nok = st.number_input(
            "NOK",
            key="rate_NOK",
            value=float(st.session_state["rate_NOK"]),
            step=0.01,
        )
        cad = st.number_input(
            "CAD",
            key="rate_CAD",
            value=float(st.session_state["rate_CAD"]),
            step=0.01,
        )
        sek = st.number_input(
            "SEK",
            key="rate_SEK",
            value=float(st.session_state["rate_SEK"]),
            step=0.01,
        )

        rates = {"USD": usd, "EUR": eur, "NOK": nok, "CAD": cad, "SEK": sek}
        if st.button("Spara kurser"):
            spara_valutakurser(rates)
            st.toast("Sparade valutakurser.")
    return {"USD": usd, "EUR": eur, "NOK": nok, "CAD": cad, "SEK": sek}


def _sidebar_batch(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Sidopanel – batchkö och körning, via stockapp.batch.sidebar_batch_controls.
    """
    def _save_cb(dfx: pd.DataFrame):
        _save_df(dfx)

    def _recompute_cb(dfx: pd.DataFrame) -> pd.DataFrame:
        return _recompute_derived(dfx, user_rates)

    df_out = sidebar_batch_controls(
        df,
        user_rates,
        default_batch_size=int(st.session_state.get("batch_size", 10)),
        save_cb=_save_cb,
        recompute_cb=_recompute_cb,
    )
    return df_out


# ------------------------------------------------------------
# Vyer
# ------------------------------------------------------------
def vy_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]):
    """
    Delegerar själva rangordning/rendering till stockapp.invest.visa_investeringsforslag,
    men ser till att derivatkolumner finns (P/S-snitt, Risklabel, Market Cap (SEK)).
    """
    work = _recompute_derived(df, user_rates)
    visa_investeringsforslag(work, user_rates)


def vy_edit(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.header("✏️ Lägg till / uppdatera bolag")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return df

    tickers = df["Ticker"].astype(str).tolist()
    st.session_state["edit_index"] = min(max(0, st.session_state["edit_index"]), len(tickers) - 1)

    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("◀ Föregående", disabled=st.session_state["edit_index"] <= 0):
        st.session_state["edit_index"] -= 1
        st.rerun()
    c2.markdown(
        f"<div style='text-align:center'>**{st.session_state['edit_index']+1} / {len(tickers)}**</div>",
        unsafe_allow_html=True,
    )
    if c3.button("Nästa ▶", disabled=st.session_state["edit_index"] >= len(tickers) - 1):
        st.session_state["edit_index"] += 1
        st.rerun()

    current_tkr = tickers[st.session_state["edit_index"]]
    st.write(f"**Ticker:** {current_tkr}")

    colx, coly = st.columns(2)
    if colx.button("Uppdatera kurs"):
        df2, msg = runner_price(df, current_tkr, user_rates)
        st.toast(msg)
        if df2 is not None:
            _save_df(df2)
            st.session_state["_df_ref"] = df2
            st.rerun()
    if coly.button("Full uppdatering"):
        df2, msg = runner_full(df, current_tkr, user_rates)
        st.toast(msg)
        if df2 is not None:
            _save_df(df2)
            st.session_state["_df_ref"] = df2
            st.rerun()

    # “Manuell prognoslista” direkt här: äldst prognos först
    st.subheader("📝 Manuell prognoslista (äldst först)")
    need = _build_requires_manual_df(df, older_than_days=None)
    st.dataframe(need, use_container_width=True, hide_index=True)

    return df


def vy_portfolio(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("💼 Portfölj")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # värde i SEK
    def _to_sek(row):
        # stöder både "Kurs" och "Aktuell kurs"
        price = safe_float(row.get("Kurs"), np.nan)
        if math.isnan(price):
            price = safe_float(row.get("Aktuell kurs"), np.nan)
        qty = safe_float(row.get("Antal aktier"), 0.0)
        cur = str(row.get("Valuta", "SEK")).upper()
        rate = hamta_valutakurs(cur, user_rates)
        if math.isnan(price):
            return np.nan
        return price * qty * float(rate)

    port = df.copy()
    port["Värde (SEK)"] = port.apply(_to_sek, axis=1)
    total = port["Värde (SEK)"].sum(skipna=True)

    st.markdown("**Totalt portföljvärde:** " + format_large_number(total, "SEK"))

    show_cols = ["Bolagsnamn", "Ticker", "Antal aktier", "Kurs", "Aktuell kurs", "Valuta", "Värde (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Hjälptabeller / kontroll
# ------------------------------------------------------------
def _build_requires_manual_df(df: pd.DataFrame, older_than_days: Optional[int]) -> pd.DataFrame:
    """
    Lista över tickers där **prognosfält** behöver manuell uppdatering.
    Sorterar äldst datum först (eller filtrerar på ålder om older_than_days anges).
    """
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Bolagsnamn", "Fält", "Senast uppdaterad"])

    rows = []
    for _, r in df.iterrows():
        for f in MANUAL_PROGNOS_FIELDS:
            ts_col = f"{f} TS"
            ts_val = r.get(ts_col)
            rows.append(
                {
                    "Ticker": r.get("Ticker"),
                    "Bolagsnamn": r.get("Bolagsnamn"),
                    "Fält": f,
                    "Senast uppdaterad": parse_date(ts_val),
                }
            )
    need = pd.DataFrame(rows)
    need = need.sort_values(by="Senast uppdaterad", ascending=True, na_position="first")
    if older_than_days is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(older_than_days))
        need = need[(need["Senast uppdaterad"].isna()) | (need["Senast uppdaterad"] < cutoff)]
    return need.reset_index(drop=True)


# ------------------------------------------------------------
# Huvudprogram
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    _init_state_defaults()

    # Läs data
    df = _load_df()
    st.session_state["_df_ref"] = df

    # Sidopanel – valutor & batch
    user_rates = _sidebar_rates()
    df2 = _sidebar_batch(st.session_state["_df_ref"], user_rates)
    if df2 is not st.session_state["_df_ref"]:
        st.session_state["_df_ref"] = df2

    # Välj vy
    st.session_state["view"] = st.sidebar.radio(
        "Välj vy",
        ["Investeringsförslag", "Lägg till / uppdatera", "Portfölj"],
        index=["Investeringsförslag", "Lägg till / uppdatera", "Portfölj"].index(st.session_state["view"]),
    )

    if st.session_state["view"] == "Investeringsförslag":
        vy_investeringsforslag(st.session_state["_df_ref"], user_rates)
    elif st.session_state["view"] == "Lägg till / uppdatera":
        df3 = vy_edit(st.session_state["_df_ref"], user_rates)
        if df3 is not st.session_state["_df_ref"]:
            st.session_state["_df_ref"] = df3
    else:
        vy_portfolio(st.session_state["_df_ref"], user_rates)


if __name__ == "__main__":
    main()
