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
        MANUAL_PROGNOS_FIELDS,
    )
except Exception:
    APP_TITLE = "K-pf-rslag"
    DISPLAY_CURRENCY = "SEK"
    FINAL_COLS = [
        "Ticker",
        "Bolagsnamn",
        "Valuta",
        "Kurs",
        "Antal aktier",
        "Market Cap",
        "P/S",
        "P/S Q1",
        "P/S Q2",
        "P/S Q3",
        "P/S Q4",
        "GAV (SEK)",
        "Omsättning i år (est.)",
        "Omsättning nästa år (est.)",
    ]
    TS_FIELDS = [
        "Kurs",
        "P/S",
        "P/S Q1",
        "P/S Q2",
        "P/S Q3",
        "P/S Q4",
        "Market Cap",
        "Omsättning i år (est.)",
        "Omsättning nästa år (est.)",
    ]
    MANUAL_PROGNOS_FIELDS = ["Omsättning i år (est.)", "Omsättning nästa år (est.)"]

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
    from stockapp.fetchers.orchestrator import run_update_full
except Exception:
    run_update_full = None  # type: ignore

try:
    from stockapp.fetchers.yahoo import get_live_price as _yahoo_price
except Exception:
    _yahoo_price = None  # type: ignore


# ------------------------------------------------------------
# Lokala helpers
# ------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mappar dina rubriker till appens standard, t.ex.:
      - 'Aktuell kurs'  -> 'Kurs'
      - 'Market Cap (valuta)' -> 'Market Cap'
      - 'Utestående aktier' / 'Utestående aktier (milj.)' -> 'Utestående aktier'
      - 'Bruttomarginal (%)' -> 'Gross margin (%)'
      - 'Nettomarginal (%)'  -> 'Net margin (%)'
      - 'Dividend Yield (%)' -> 'Dividend yield (%)'
      - 'Payout Ratio CF (%)'-> 'Dividend payout (FCF) (%)'
      - TS-prefix 'TS_'/'TS-' konverteras till suffix ' ... TS' för kända fält
    """
    if df is None or df.empty:
        return df

    rename_map = {
        "Aktuell kurs": "Kurs",
        "Pris": "Kurs",
        "Market Cap (valuta)": "Market Cap",
        "Market Cap (SEK)": "Market Cap (SEK)",  # behåll separat
        "Utestående aktier (milj.)": "Utestående aktier",
        "Utestående aktier (miljoner)": "Utestående aktier",
        "Utestående aktier": "Utestående aktier",
        "Bruttomarginal (%)": "Gross margin (%)",
        "Nettomarginal (%)": "Net margin (%)",
        "EV/EBITDA": "EV/EBITDA (ttm)",
        "EV/EBITDA (TTM)": "EV/EBITDA (ttm)",
        "Dividend Yield (%)": "Dividend yield (%)",
        "Payout Ratio CF (%)": "Dividend payout (FCF) (%)",
        "Omsättning idag": "Omsättning i år (est.)",
        "Omsättning i år": "Omsättning i år (est.)",
        "Omsättning nästa år": "Omsättning nästa år (est.)",
        "Bolagsnamn": "Bolagsnamn",  # already ok
        "Namn": "Bolagsnamn",
    }

    out = df.copy()
    # 1) vanliga byten
    for old, new in rename_map.items():
        if old in out.columns and new != old:
            out = out.rename(columns={old: new})

    # 2) P/S kvartal – acceptera varianter "PS Q1", "P/SQ1", etc.
    for q in ("Q1", "Q2", "Q3", "Q4"):
        for cand in [f"P/S {q}", f"PS {q}", f"P/S{q}", f"PS{q}"]:
            if cand in out.columns:
                out = out.rename(columns={cand: f"P/S {q}"})

    # 3) TS-kolumner: konvertera "TS_P/S", "TS-P/S", "TS P/S" -> "P/S TS"
    ts_targets = [
        "Kurs", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Market Cap", "Omsättning i år (est.)", "Omsättning nästa år (est.)"
    ]
    colmap = {}
    for c in out.columns:
        cu = str(c).strip()
        if cu.upper().startswith("TS") and cu not in ts_targets:
            # Ta bort 'TS', separations-tecken och trimma
            rest = cu[2:]
            while rest and rest[0] in (" ", "_", "-", ":"):
                rest = rest[1:]
            # Försök hitta match mot kända fält
            for tgt in ts_targets:
                if rest.lower() == tgt.lower():
                    colmap[c] = f"{tgt} TS"
                    break
    if colmap:
        out = out.rename(columns=colmap)

    return out


def _init_state_defaults():
    if "_df_ref" not in st.session_state:
        st.session_state["_df_ref"] = pd.DataFrame(columns=FINAL_COLS)

    if "_rates_seeded" not in st.session_state:
        saved = las_sparade_valutakurser()
        for key in ("USD", "EUR", "NOK", "CAD", "SEK"):
            st.session_state[f"rate_{key}"] = float(saved.get(key, 1.0))
        st.session_state["_rates_seeded"] = True

    st.session_state.setdefault("batch_queue", [])
    st.session_state.setdefault("batch_order_mode", "Äldst först")
    st.session_state.setdefault("batch_size", 10)

    st.session_state.setdefault("view", "Investeringsförslag")
    st.session_state.setdefault("page", 1)
    st.session_state.setdefault("page_size", 5)
    st.session_state.setdefault("edit_index", 0)


def _load_df() -> pd.DataFrame:
    """Hämta df, normalisera kolumner, säkra schema, visa diagnostik."""
    try:
        df = hamta_data()
    except Exception as e:
        st.warning(f"⚠️ Kunde inte läsa data från Google Sheet: {e}")
        df = pd.DataFrame(columns=FINAL_COLS)

    df = _normalize_columns(df)
    df = ensure_schema(df, FINAL_COLS)
    df2, dups = dedupe_tickers(df)
    if dups:
        st.info(f"ℹ️ Dubbletter ignoreras i minnet: {', '.join(dups)}")

    # diagnostik
    tickers = []
    if "Ticker" in df2.columns:
        tickers = df2["Ticker"].astype(str).head(5).tolist()
    st.caption(f"📥 Lästa rader: {len(df2)} • Kolumner: {len(df2.columns)} • Tickers (5): {', '.join(tickers)}")
    return df2


def _save_df(df: pd.DataFrame):
    try:
        spara_data(df)
        st.success("✅ Ändringar sparade.")
    except Exception as e:
        st.error(f"🚫 Kunde inte spara till Google Sheet: {e}")


def _sidebar_rates() -> Dict[str, float]:
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


def _pick_order(df: pd.DataFrame, mode: str) -> List[str]:
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
    if _yahoo_price is None:
        return df, "Yahoo-källa saknas"
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns inte i tabellen"
    try:
        price = _yahoo_price(str(tkr))
        if price and price > 0:
            df.loc[ridx, "Kurs"] = float(price)
            df = stamp_fields_ts(df, ["Kurs"], ts_suffix=" TS")
            return df, "OK"
        return df, "Pris saknas"
    except Exception as e:
        return df, f"Fel: {e}"


def _runner_full(df: pd.DataFrame, tkr: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
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
    if not queue:
        st.info("Ingen batchkö.")
        return df

    total = len(queue)
    bar = st.sidebar.progress(0, text=f"0/{total}")
    done = 0
    log_lines = []
    work = df.copy()

    for tkr in list(queue):
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


def _sidebar_batch(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    with st.sidebar.expander("⚙️ Batch", expanded=True):
        st.session_state["batch_order_mode"] = st.selectbox(
            "Sortering", ["Äldst först", "A–Ö", "Z–A"], index=["Äldst först", "A–Ö", "Z–A"].index(st.session_state["batch_order_mode"])
        )
        st.session_state["batch_size"] = st.number_input("Antal i batch", min_value=1, max_value=200, value=int(st.session_state["batch_size"]))
        if st.button("Skapa batchkö"):
            order = _pick_order(df, st.session_state["batch_order_mode"])
            queue = [t for t in order if t not in st.session_state["batch_queue"]]
            st.session_state["batch_queue"] = queue[: st.session_state["batch_size"]]
            st.toast(f"Skapade batch ({len(st.session_state['batch_queue'])} tickers).")
        if st.session_state["batch_queue"]:
            st.write("Kö:", ", ".join(st.session_state["batch_queue"]))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Kör batch – endast kurs"):
                return _run_batch(df, st.session_state["batch_queue"], mode="price", user_rates=user_rates)
        with col2:
            if st.button("Kör batch – full"):
                return _run_batch(df, st.session_state["batch_queue"], mode="full", user_rates=user_rates)
    return df


# ------------------------------------------------------------
# Vyer (lite kortade för fokus på läsning)
# ------------------------------------------------------------
def vy_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("📈 Investeringsförslag")
    sortcol = "Score" if "Score" in df.columns else ("P/S-snitt (Q1..Q4)" if "P/S-snitt (Q1..Q4)" in df.columns else None)
    if sortcol is None:
        for c in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
            if c not in df.columns:
                df[c] = np.nan
        df["P/S-snitt (Q1..Q4)"] = pd.to_numeric(df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1), errors="coerce")
        sortcol = "P/S-snitt (Q1..Q4)"

    work = df.copy()
    if "Risklabel" not in work.columns:
        work["Risklabel"] = work["Market Cap"].apply(risk_label_from_mcap) if "Market Cap" in work.columns else "Unknown"

    c1, c2, c3 = st.columns([1, 1, 1])
    sektorer = ["Alla"]
    if "Sektor" in work.columns:
        sektorer += sorted([s for s in work["Sektor"].dropna().astype(str).unique() if s and s != "nan"])
    val_sektor = c1.selectbox("Sektor", sektorer)
    risk_opts = ["Alla", "Mega", "Large", "Mid", "Small", "Micro", "Unknown"]
    val_risk = c2.selectbox("Risklabel", risk_opts)
    st.session_state["page_size"] = c3.number_input("Poster per sida", min_value=1, max_value=20, value=int(st.session_state["page_size"]))

    if val_sektor != "Alla" and "Sektor" in work.columns:
        work = work[work["Sektor"].astype(str) == val_sektor]
    if val_risk != "Alla":
        work = work[work["Risklabel"].astype(str) == val_risk]

    if sortcol == "Score":
        work = work.sort_values(by=sortcol, ascending=False, na_position="last")
    else:
        work = work.sort_values(by=sortcol, ascending=True, na_position="last")

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
    colp2.markdown(f"<div style='text-align:center'>**{st.session_state['page']} / {pages}**</div>", unsafe_allow_html=True)
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
            ps_ttm = safe_float(row.get('P/S'), np.nan)
            ps_avg = safe_float(row.get('P/S-snitt (Q1..Q4)'), np.nan)
            cols[0].metric("P/S (TTM)", "–" if math.isnan(ps_ttm) else f"{ps_ttm:.2f}")
            cols[1].metric("P/S-snitt (4Q)", "–" if math.isnan(ps_avg) else f"{ps_avg:.2f}")
            mcap_disp = format_large_number(row.get("Market Cap", np.nan), "USD")
            cols[2].metric("Market Cap (nu)", mcap_disp)
            cols[3].write(f"**Risklabel:** {row.get('Risklabel', 'Unknown')}")

            with st.expander("Visa nyckeltal / historik"):
                info = []
                for c in ["Sektor", "Valuta", "Debt/Equity", "Gross margin (%)", "Net margin (%)", "Utestående aktier", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
                    if c in df.columns:
                        info.append((c, row.get(c)))
                info.insert(0, ("Market Cap (nu)", mcap_disp))
                for k, v in info:
                    if isinstance(v, (int, float)) and not math.isnan(float(v)):
                        st.write(f"- **{k}:** {v}")
                    else:
                        st.write(f"- **{k}:** –")


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
    c2.markdown(f"<div style='text-align:center'>**{st.session_state['edit_index']+1} / {len(tickers)}**</div>", unsafe_allow_html=True)
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

    st.subheader("📝 Manuell prognoslista (äldst först)")
    need = _build_requires_manual_df(df, older_than_days=None)
    st.dataframe(need, use_container_width=True, hide_index=True)
    return df


def vy_portfolio(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("💼 Portfölj")
    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

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
    total = port["Värde (SEK)"].sum(skipna=True)

    st.markdown(f"**Totalt portföljvärde:** {format_large_number(total, 'SEK')}")

    show_cols = ["Bolagsnamn", "Ticker", "Antal aktier", "Kurs", "Valuta", "Värde (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)


def _build_requires_manual_df(df: pd.DataFrame, older_than_days: Optional[int]) -> pd.DataFrame:
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
# Huvud
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    _init_state_defaults()

    df = _load_df()
    st.session_state["_df_ref"] = df

    user_rates = _sidebar_rates()
    df2 = _sidebar_batch(st.session_state["_df_ref"], user_rates)
    if df2 is not st.session_state["_df_ref"]:
        st.session_state["_df_ref"] = df2

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
