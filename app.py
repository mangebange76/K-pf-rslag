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
try:
    from stockapp.config import (
        APP_TITLE,
        FINAL_COLS,
        TS_FIELDS,
        DISPLAY_CURRENCY,
        PROPOSALS_PAGE_SIZE,
        BATCH_DEFAULT_SIZE,
    )
except Exception:
    APP_TITLE = "K-pf-rslag"
    DISPLAY_CURRENCY = "SEK"
    PROPOSALS_PAGE_SIZE = 5
    BATCH_DEFAULT_SIZE = 10
    FINAL_COLS = [
        "Ticker",
        "Bolagsnamn",
        "Valuta",
        "Sektor",
        "Risklabel",
        "Antal aktier",
        "Kurs",
        "P/S",
        "P/S Q1",
        "P/S Q2",
        "P/S Q3",
        "P/S Q4",
        "Market Cap",
    ]
    TS_FIELDS = ["TS Omsättning i år", "TS Omsättning nästa år"]

from stockapp.storage import hamta_data, spara_data
from stockapp.utils import (
    add_oldest_ts_col,
    dedupe_tickers,
    ensure_schema,
    format_large_number,
    now_stamp,
    parse_date,
    safe_float,
    with_backoff,
    risk_label_from_mcap,
)
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)

# valfria fetchers
try:
    from stockapp.fetchers.orchestrator import run_update_full  # (df, tkr, rates) -> (df, msg)
except Exception:
    run_update_full = None  # type: ignore

try:
    from stockapp.fetchers.yahoo import get_live_price as _yahoo_price
except Exception:
    _yahoo_price = None  # type: ignore


# ------------------------------------------------------------
# Hjälpfunktioner (lokala)
# ------------------------------------------------------------
def _init_state_defaults():
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
    st.session_state.setdefault("batch_size", int(BATCH_DEFAULT_SIZE))

    # vyer
    st.session_state.setdefault("view", "Investeringsförslag")
    st.session_state.setdefault("page", 1)
    st.session_state.setdefault("page_size", int(PROPOSALS_PAGE_SIZE))

    # edit/bläddra
    st.session_state.setdefault("edit_index", 0)


def _normalize_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mappa vanliga alias → kanoniska kolumnnamn INNAN ensure_schema stramar upp.
    (Påverkar inte befintliga kolumner om kanoniskt namn redan finns.)
    """
    alias_map = {
        # Namn
        "Namn": "Bolagsnamn",
        # Kurs
        "Aktuell kurs": "Kurs",
        "Pris": "Kurs",
        # Antal
        "Antal": "Antal aktier",
        "Antal du äger": "Antal aktier",
        # Utestående aktier
        "Utestående aktier": "Utestående aktier (milj.)",
        "Shares Outstanding (M)": "Utestående aktier (milj.)",
        # P/S-snitt
        "P/S-snitt": "P/S-snitt (Q1..Q4)",
        "P/S snitt": "P/S-snitt (Q1..Q4)",
        "PS-snitt": "P/S-snitt (Q1..Q4)",
    }
    have = set(df.columns.astype(str))
    ren = {}
    for alias, canonical in alias_map.items():
        if alias in have and canonical not in have:
            ren[alias] = canonical
    if ren:
        df = df.rename(columns=ren)
    return df


def _load_df() -> pd.DataFrame:
    """Hämta df från Google Sheet – säkra schema och varna om problem."""
    try:
        df = hamta_data()
        df = _normalize_aliases(df)
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


def _sidebar_rates() -> Dict[str, float]:
    """Sidopanel för valutakurser (utan experimental_rerun)."""
    with st.sidebar.expander("💱 Valutakurser (→ SEK)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Hämta automatiskt"):
                fetched, misses, provider = hamta_valutakurser_auto()
                for k, v in fetched.items():
                    st.session_state[f"rate_{k}"] = float(v)
                spara_valutakurser(fetched)
                if misses:
                    st.warning("Kunde inte hämta: " + ", ".join(misses))
                st.toast(f"Valutor uppdaterade via {provider}.")
                st.rerun()

        usd = st.number_input("USD", key="rate_USD", value=float(st.session_state["rate_USD"]), step=0.01)
        eur = st.number_input("EUR", key="rate_EUR", value=float(st.session_state["rate_EUR"]), step=0.01)
        nok = st.number_input("NOK", key="rate_NOK", value=float(st.session_state["rate_NOK"]), step=0.01)
        cad = st.number_input("CAD", key="rate_CAD", value=float(st.session_state["rate_CAD"]), step=0.01)
        sek = st.number_input("SEK", key="rate_SEK", value=float(st.session_state["rate_SEK"]), step=0.01)

        rates = {"USD": usd, "EUR": eur, "NOK": nok, "CAD": cad, "SEK": sek}
        if st.button("Spara kurser"):
            spara_valutakurser(rates)
            st.toast("Sparade valutakurser.")
    return {"USD": usd, "EUR": eur, "NOK": nok, "CAD": cad, "SEK": sek}


def _sidebar_batch(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """Sidopanel – batchkö och körning."""
    with st.sidebar.expander("⚙️ Batch", expanded=True):
        st.session_state["batch_order_mode"] = st.selectbox(
            "Sortering",
            ["Äldst först", "A–Ö", "Z–A"],
            index=["Äldst först", "A–Ö", "Z–A"].index(st.session_state["batch_order_mode"]),
        )
        st.session_state["batch_size"] = st.number_input(
            "Antal i batch", min_value=1, max_value=200, value=int(st.session_state["batch_size"])
        )

        if st.button("Skapa batchkö"):
            order = _pick_order(df, st.session_state["batch_order_mode"])
            queue = [t for t in order if t not in st.session_state["batch_queue"]]
            st.session_state["batch_queue"] = queue[: st.session_state["batch_size"]]
            st.toast(f"Skapade batch ({len(st.session_state['batch_queue'])} tickers).")

        if st.session_state["batch_queue"]:
            st.write("Kö:", ", ".join(st.session_state["batch_queue"]))

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Kör batch – endast kurs"):
                df2 = _run_batch(df, st.session_state["batch_queue"], mode="price", user_rates=user_rates)
                return df2
        with c2:
            if st.button("Kör batch – full"):
                df2 = _run_batch(df, st.session_state["batch_queue"], mode="full", user_rates=user_rates)
                return df2
    return df


def _pick_order(df: pd.DataFrame, mode: str) -> List[str]:
    """Välj ordning för batch."""
    work = df.copy()
    work["Ticker"] = work["Ticker"].astype(str)
    if mode == "Äldst först":
        work = add_oldest_ts_col(work, dest_col="__oldest_ts__")
        work = work.sort_values(by="__oldest_ts__", ascending=True, na_position="first")
    elif mode == "A–Ö":
        work = work.sort_values(by="Ticker", ascending=True)
    else:
        work = work.sort_values(by="Ticker", ascending=False)
    return work["Ticker"].tolist()


def _runner_price(df: pd.DataFrame, tkr: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """Uppdatera ENDAST kurs för ticker (Yahoo fallback)."""
    if _yahoo_price is None:
        return df, "Yahoo-källa saknas"
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns inte i tabellen"
    try:
        price = _yahoo_price(str(tkr))
        if price and price > 0:
            df.loc[ridx, "Kurs"] = float(price)
            # stämpla både prefix- och suffix-variant för max kompatibilitet
            ts = now_stamp()
            df.loc[ridx, "TS Kurs"] = ts
            df.loc[ridx, "Kurs TS"] = ts
            return df, "OK"
        return df, "Pris saknas"
    except Exception as e:
        return df, f"Fel: {e}"


def _runner_full(df: pd.DataFrame, tkr: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """Uppdatera ALLT för ticker via orchestrator om den finns, annars fallback till pris."""
    if run_update_full is None:
        return _runner_price(df, tkr, user_rates)
    try:
        out = run_update_full(df, tkr, user_rates)  # type: ignore
        if isinstance(out, tuple) and len(out) == 2:
            df2, msg = out
            return df2, str(msg)
        if isinstance(out, pd.DataFrame):
            return out, "OK"
        return df, "Orchestrator: oväntat svar"
    except Exception as e:
        return df, f"Fel: {e}"


def _run_batch(df: pd.DataFrame, queue: List[str], mode: str, user_rates: Dict[str, float]) -> pd.DataFrame:
    """Kör batch mot kö – visar progress 1/X och sparar var 5:e."""
    if not queue:
        st.info("Ingen batchkö.")
        return df

    total = len(queue)
    bar = st.sidebar.progress(0, text=f"0/{total}")
    done = 0
    log_lines = []
    work = df.copy()

    for tkr in list(queue):  # iterera över en kopia
        if mode == "price":
            work, msg = _runner_price(work, tkr, user_rates)
        else:
            work, msg = _runner_full(work, tkr, user_rates)
        done += 1
        bar.progress(done / total, text=f"{done}/{total}")
        log_lines.append(f"{tkr}: {msg}")
        st.session_state["batch_queue"] = [x for x in st.session_state["batch_queue"] if x != tkr]
        if done % 5 == 0:
            _save_df(work)

    _save_df(work)
    st.sidebar.write("Logg:")
    for ln in log_lines:
        st.sidebar.write("• " + ln)
    return work


# ------------------------------------------------------------
# Vyer
# ------------------------------------------------------------
def vy_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("📈 Investeringsförslag")

    sortcol = None
    if "TotalScore" in df.columns:
        sortcol = "TotalScore"
    elif "P/S-snitt (Q1..Q4)" in df.columns:
        sortcol = "P/S-snitt (Q1..Q4)"
    elif "P/S" in df.columns:
        sortcol = "P/S"

    work = df.copy()
    if "Risklabel" not in work.columns:
        if "Market Cap" in work.columns:
            work["Risklabel"] = work["Market Cap"].apply(risk_label_from_mcap)
        else:
            work["Risklabel"] = "Unknown"

    c1, c2, c3 = st.columns([1, 1, 1])
    sektorer = ["Alla"]
    if "Sektor" in work.columns:
        sektorer += sorted([s for s in work["Sektor"].dropna().astype(str).unique() if s and s != "nan"])
    val_sektor = c1.selectbox("Sektor", sektorer)
    risk_opts = ["Alla", "Mega", "Large", "Mid", "Small", "Micro", "Unknown"]
    val_risk = c2.selectbox("Risklabel", risk_opts)
    st.session_state["page_size"] = c3.number_input(
        "Poster per sida", min_value=1, max_value=20, value=int(st.session_state["page_size"])
    )

    if val_sektor != "Alla" and "Sektor" in work.columns:
        work = work[work["Sektor"].astype(str) == val_sektor]
    if val_risk != "Alla":
        work = work[work["Risklabel"].astype(str) == val_risk]

    if sortcol:
        asc = False if sortcol == "TotalScore" else True
        work = work.sort_values(by=sortcol, ascending=asc, na_position="last")

    total = len(work)
    if total == 0:
        st.info("Inga träffar.")
        return
    pages = max(1, math.ceil(total / st.session_state["page_size"]))
    st.session_state["page"] = max(1, min(st.session_state.get("page", 1), pages))

    colp1, colp2, colp3 = st.columns([1, 2, 1])
    if colp1.button("◀ Föregående", disabled=st.session_state["page"] <= 1):
        st.session_state["page"] -= 1
        st.rerun()
    colp2.markdown(
        f"<div style='text-align:center'><strong>{st.session_state['page']} / {pages}</strong></div>",
        unsafe_allow_html=True,
    )
    if colp3.button("Nästa ▶", disabled=st.session_state["page"] >= pages):
        st.session_state["page"] += 1
        st.rerun()

    start = (st.session_state["page"] - 1) * st.session_state["page_size"]
    end = start + st.session_state["page_size"]
    page_df = work.iloc[start:end].reset_index(drop=True)

    for _, row in page_df.iterrows():
        with st.container(border=True):
            st.subheader(f"{row.get('Bolagsnamn', '')} ({row.get('Ticker', '')})")
            cols = st.columns(4)
            ps_val = safe_float(row.get("P/S"), np.nan)
            ps_avg = safe_float(row.get("P/S-snitt (Q1..Q4)"), np.nan)
            mcap_disp = format_large_number(row.get("Market Cap", np.nan), "USD")
            cols[0].metric("P/S (TTM)", "–" if math.isnan(ps_val) else f"{ps_val:.2f}")
            cols[1].metric("P/S-snitt (4Q)", "–" if math.isnan(ps_avg) else f"{ps_avg:.2f}")
            cols[2].metric("Market Cap (nu)", mcap_disp)
            cols[3].write(f"**Risklabel:** {row.get('Risklabel', 'Unknown')}")

            with st.expander("Visa nyckeltal / historik"):
                info = []
                for c in [
                    "Sektor",
                    "Valuta",
                    "Debt/Equity",
                    "Gross margin (%)",
                    "Operating margin (%)",
                    "Net margin (%)",
                    "Utestående aktier (milj.)",
                    "P/S Q1",
                    "P/S Q2",
                    "P/S Q3",
                    "P/S Q4",
                ]:
                    if c in df.columns:
                        info.append((c, row.get(c)))
                info.insert(0, ("Market Cap (nu)", mcap_disp))
                for k, v in info:
                    if isinstance(v, (int, float)) and not math.isnan(float(v)):
                        st.write(f"- **{k}:** {v}")
                    else:
                        st.write(f"- **{k}:** –")

            if "TotalScore" in df.columns and not pd.isna(row.get("TotalScore")):
                sc = float(row.get("TotalScore"))
                if sc >= 85:
                    tag = "✅ Mycket bra"
                elif sc >= 70:
                    tag = "👍 Bra"
                elif sc >= 55:
                    tag = "🙂 Okej"
                elif sc >= 40:
                    tag = "⚠️ Något övervärderad"
                else:
                    tag = "🛑 Övervärderad / Sälj"
                st.markdown(f"**Betyg:** {sc:.1f} – {tag}")


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
        f"<div style='text-align:center'><strong>{st.session_state['edit_index']+1} / {len(tickers)}</strong></div>",
        unsafe_allow_html=True,
    )
    if c3.button("Nästa ▶", disabled=st.session_state["edit_index"] >= len(tickers) - 1):
        st.session_state["edit_index"] += 1
        st.rerun()

    current_tkr = tickers[st.session_state["edit_index"]]
    st.write(f"**Ticker:** {current_tkr}")

    colx, coly = st.columns(2)
    if colx.button("Uppdatera kurs"):
        df2, msg = _runner_price(df, current_tkr, user_rates)
        st.toast(f"{current_tkr}: {msg}")
        if df2 is not None:
            _save_df(df2)
            st.session_state["_df_ref"] = df2
    if coly.button("Full uppdatering"):
        df2, msg = _runner_full(df, current_tkr, user_rates)
        st.toast(f"{current_tkr}: {msg}")
        if df2 is not None:
            _save_df(df2)
            st.session_state["_df_ref"] = df2

    return df


def vy_portfolio(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("💼 Portfölj")

    # Om tomt – gör ett försök att läsa om direkt från arket
    if df is None or df.empty:
        df_reload = _load_df()
        if df_reload is not None and not df_reload.empty:
            df = df_reload
            st.session_state["_df_ref"] = df_reload

    # Statusrad alltid
    st.caption(f"Rader: {len(df)} • Kolumner: {len(df.columns)}")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # säkerställ kolumner
    if "Antal aktier" not in df.columns:
        df["Antal aktier"] = 0.0
    if "Valuta" not in df.columns:
        df["Valuta"] = "SEK"

    def _to_sek(row):
        price = safe_float(row.get("Kurs"), np.nan)
        qty = safe_float(row.get("Antal aktier"), 0.0)
        cur = str(row.get("Valuta", "SEK")).upper()
        rate = hamta_valutakurs(cur, user_rates)
        if math.isnan(price):
            return np.nan
        return price * qty * float(rate)

    port = df.copy()
    port["Värde (SEK)"] = port.apply(_to_sek, axis=1)
    total = float(port["Värde (SEK)"].sum(skipna=True))

    st.markdown(f"**Totalt portföljvärde:** {format_large_number(total, 'SEK')}")

    show_cols = ["Bolagsnamn", "Ticker", "Antal aktier", "Kurs", "Valuta", "Värde (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(
        port[show_cols].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


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

    # liten diag-rad
    with st.expander("⚠️ Diagnostik (import/konfiguration)", expanded=False):
        nrows = len(df)
        ncols = len(df.columns)
        tickers = ", ".join(df["Ticker"].astype(str).head(5).tolist()) if "Ticker" in df.columns else "-"
        st.caption(f"Lästa rader: {nrows} • Kolumner: {ncols} • Tickers (5): {tickers}")

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
