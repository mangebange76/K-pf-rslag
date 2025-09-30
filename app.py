# app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

# Våra moduler
from stockapp.config import (
    FINAL_COLS,
    TS_FIELDS,
    STANDARD_VALUTAKURSER,
)
from stockapp.utils import (
    ensure_schema,
    add_oldest_ts_col,
    now_stamp,
    safe_float,
    dedupe_tickers,
)
from stockapp.storage import hamta_data, spara_data
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)
from stockapp.orchestrator import (
    run_update_price,
    run_update_full,
    run_batch_update,
)
from stockapp.invest import visa_investeringsforslag


# ------------------------------------------------------------
# Hjälpfunktioner (UI)
# ------------------------------------------------------------
def _init_session():
    # Engångs-init
    st.session_state.setdefault("_df_ref", None)
    st.session_state.setdefault("_last_log", [])
    st.session_state.setdefault("_msg", "")
    st.session_state.setdefault("_nav_index", 0)  # för bläddring i Lägg till/uppdatera

    # Init valutainput-keys med sparade värden (inte mixa med andra nycklar)
    if "rate_usd_in" not in st.session_state:
        saved = las_sparade_valutakurser()
        st.session_state["rate_usd_in"] = float(saved.get("USD", STANDARD_VALUTAKURSER["USD"]))
        st.session_state["rate_eur_in"] = float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"]))
        st.session_state["rate_cad_in"] = float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
        st.session_state["rate_nok_in"] = float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
        st.session_state["rate_sek_in"] = float(saved.get("SEK", 1.0))

    # Batch UI-state
    st.session_state.setdefault("_batch_count", 10)
    st.session_state.setdefault("_batch_order", "Äldst först")
    st.session_state.setdefault("_batch_mode", "Full uppdatering")


def _read_df() -> pd.DataFrame:
    try:
        df = hamta_data()
        df = ensure_schema(df)
        # dedupe tickers (skydd), men skriv inte tillbaka direkt – visa varningen först
        dupes = dedupe_tickers(df)
        if dupes:
            st.warning(f"Dubbletter upptäckta (tickers): {', '.join(sorted(dupes))}. Jag visar dem bara en gång i UI.")
            # håll första förekomsten
            df = df.loc[~df["Ticker"].str.upper().duplicated()].copy()
        return df
    except Exception as e:
        st.error(f"Kunde inte läsa data från Google Sheet: {e}")
        # tom df med schema så att UI lever
        return ensure_schema(pd.DataFrame(columns=FINAL_COLS))


def _save_df(df: pd.DataFrame, snapshot=False):
    try:
        spara_data(df, do_snapshot=snapshot)
        st.success("Tabellen sparades.")
    except Exception as e:
        st.error(f"Kunde inte spara till Google Sheet: {e}")


def _sidebar_rates() -> dict:
    st.sidebar.header("Valutakurser")
    with st.sidebar.expander("Kursinställningar", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Hämta auto (FMP→ECB→ERH)"):
                rates, misses, provider = hamta_valutakurser_auto()
                st.session_state["rate_usd_in"] = float(rates.get("USD", st.session_state["rate_usd_in"]))
                st.session_state["rate_eur_in"] = float(rates.get("EUR", st.session_state["rate_eur_in"]))
                st.session_state["rate_cad_in"] = float(rates.get("CAD", st.session_state["rate_cad_in"]))
                st.session_state["rate_nok_in"] = float(rates.get("NOK", st.session_state["rate_nok_in"]))
                st.session_state["rate_sek_in"] = float(rates.get("SEK", st.session_state["rate_sek_in"]))
                if misses:
                    st.info(f"Auto-kurser hämtade från {provider}, saknades: {', '.join(misses)}")
                else:
                    st.success(f"Auto-kurser hämtade från {provider}")

        usd = st.number_input("USD → SEK", min_value=0.0001, value=float(st.session_state["rate_usd_in"]), key="rate_usd_in")
        eur = st.number_input("EUR → SEK", min_value=0.0001, value=float(st.session_state["rate_eur_in"]), key="rate_eur_in")
        cad = st.number_input("CAD → SEK", min_value=0.0001, value=float(st.session_state["rate_cad_in"]), key="rate_cad_in")
        nok = st.number_input("NOK → SEK", min_value=0.0001, value=float(st.session_state["rate_nok_in"]), key="rate_nok_in")
        sek = st.number_input("SEK → SEK", min_value=0.0001, value=float(st.session_state["rate_sek_in"]), key="rate_sek_in")

        if st.button("Spara kurser till Google Sheet"):
            spara_valutakurser({"USD": usd, "EUR": eur, "CAD": cad, "NOK": nok, "SEK": sek})
            st.success("Valutakurser sparade.")

    return {"USD": usd, "EUR": eur, "CAD": cad, "NOK": nok, "SEK": sek}


def _pick_batch(df: pd.DataFrame) -> list[str]:
    order = st.session_state["_batch_order"]
    count = int(st.session_state["_batch_count"])

    work = add_oldest_ts_col(df.copy())
    if order == "A–Ö":
        work = work.sort_values(by="Ticker", ascending=True)
    else:
        # Äldst först
        work = work.sort_values(by="_oldest_ts", ascending=True, na_position="first")

    tickers = work["Ticker"].head(count).tolist()
    return tickers


def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: dict):
    st.sidebar.header("Uppdatera data")
    with st.sidebar.expander("Enskild ticker", expanded=True):
        t = st.text_input("Ticker (t.ex. NVDA)", key="_single_ticker").strip().upper()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Uppdatera enbart kurs"):
                if t:
                    df2, changed, msg = run_update_price(df.copy(), t, save=True)
                    st.session_state["_df_ref"] = df2
                    if changed:
                        st.success(f"{t}: {', '.join(next(iter(changed.values())))}")
                    else:
                        st.info(msg)
                else:
                    st.warning("Ange ticker.")
        with c2:
            if st.button("Full uppdatering"):
                if t:
                    df2, changed, dbg = run_update_full(df.copy(), t, save=True)
                    st.session_state["_df_ref"] = df2
                    if changed:
                        st.success(f"{t}: {', '.join(next(iter(changed.values())))}")
                    else:
                        st.info(dbg.get("status", "Inga ändringar."))
                else:
                    st.warning("Ange ticker.")

    with st.sidebar.expander("Batch-uppdatering", expanded=True):
        st.selectbox("Sortering", ["Äldst först", "A–Ö"], key="_batch_order")
        st.number_input("Antal att köra nu", min_value=1, max_value=1000, value=int(st.session_state["_batch_count"]), step=1, key="_batch_count")
        mode = st.selectbox("Vad ska köras?", ["Full uppdatering", "Endast kurs"], key="_batch_mode")

        tickers = _pick_batch(df)
        st.caption(f"Valda tickers ({len(tickers)}): {', '.join(tickers)}")

        if st.button("Kör batch"):
            if not tickers:
                st.warning("Inget att köra.")
            else:
                txt = st.sidebar.empty()
                pbar = st.sidebar.progress(0.0)

                def _cb(i, n, ticker, status):
                    txt.write(f"**{i}/{n}** – {ticker}: {status}")
                    pbar.progress(i / n)

                run_mode = "full" if st.session_state["_batch_mode"] == "Full uppdatering" else "price"
                df2, changed_map, errors = run_batch_update(
                    df.copy(), tickers, mode=run_mode, progress_cb=_cb, save=True
                )
                st.session_state["_df_ref"] = df2
                if changed_map:
                    st.success(f"Ändringar på {len(changed_map)} tickers.")
                if errors:
                    st.error(f"Fel för {len(errors)} tickers. Se sidopanelens logg.")
                pbar.progress(1.0)


# ------------------------------------------------------------
# VY: Portfölj
# ------------------------------------------------------------
def visa_portfolj(df: pd.DataFrame, user_rates: dict):
    st.subheader("Portföljöversikt")

    # Försök räkna värde baserat på Antal * Kurs * valutakurs
    if "Antal" not in df.columns:
        st.info("Kolumnen 'Antal' saknas – lägg till innehav i Lägg till/uppdatera.")
        return

    port = df.copy()
    # Valutakurs per rad
    def _rate(row):
        return float(user_rates.get(str(row.get("Valuta", "SEK")).upper(), 1.0))

    port["_rate"] = port.apply(_rate, axis=1)
    port["_pris_sek"] = pd.to_numeric(port["Aktuell kurs"], errors="coerce").fillna(0.0) * port["_rate"]
    port["Värde (SEK)"] = pd.to_numeric(port["Antal"], errors="coerce").fillna(0.0) * port["_pris_sek"]

    total = float(port["Värde (SEK)"].sum())
    st.metric("Portföljvärde (SEK)", f"{total:,.0f}".replace(",", " "))

    show = ["Ticker", "Bolagsnamn", "Antal", "Valuta", "Aktuell kurs", "Värde (SEK)", "Sector", "Industry"]
    show = [c for c in show if c in port.columns]
    st.dataframe(port[show].sort_values("Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# VY: Lägg till / uppdatera bolag (inkl. manuell prognoslista)
# ------------------------------------------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.subheader("Lägg till / uppdatera bolag")

    # --- Bläddringsfält över tickers
    tickers = sorted(df["Ticker"].astype(str).str.upper().tolist())
    if tickers:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀ Föregående"):
                st.session_state["_nav_index"] = max(0, st.session_state["_nav_index"] - 1)
        with c3:
            if st.button("Nästa ▶"):
                st.session_state["_nav_index"] = min(len(tickers) - 1, st.session_state["_nav_index"] + 1)

        st.session_state["_nav_index"] = int(st.number_input("Index", min_value=0, max_value=len(tickers) - 1,
                                                             value=int(st.session_state["_nav_index"]), step=1))
        current_ticker = tickers[st.session_state["_nav_index"]]
    else:
        current_ticker = ""

    # --- Form
    with st.form("addupd"):
        t_in = st.text_input("Ticker", value=current_ticker).strip().upper()
        namn = st.text_input("Bolagsnamn", value=str(df.loc[df["Ticker"].str.upper() == t_in, "Bolagsnamn"].iloc[0]) if t_in in tickers else "")
        valuta = st.text_input("Valuta", value=str(df.loc[df["Ticker"].str.upper() == t_in, "Valuta"].iloc[0]) if t_in in tickers else "USD").upper()
        antal = st.number_input("Antal", min_value=0.0, value=float(df.loc[df["Ticker"].str.upper() == t_in, "Antal"].iloc[0]) if t_in in tickers else 0.0)
        gav_sek = st.number_input("GAV (SEK)", min_value=0.0, value=float(df.loc[df["Ticker"].str.upper() == t_in, "GAV (SEK)"].iloc[0]) if (t_in in tickers and "GAV (SEK)" in df.columns) else 0.0)
        oms_idag = st.number_input("Omsättning idag (manuell, milj. i bolagets valuta)", min_value=0.0,
                                   value=float(df.loc[df["Ticker"].str.upper() == t_in, "Omsättning idag"].iloc[0]) if (t_in in tickers and "Omsättning idag" in df.columns and pd.notna(df.loc[df["Ticker"].str.upper() == t_in, "Omsättning idag"].iloc[0])) else 0.0)
        oms_next = st.number_input("Omsättning nästa år (manuell, milj. i bolagets valuta)", min_value=0.0,
                                   value=float(df.loc[df["Ticker"].str.upper() == t_in, "Omsättning nästa år"].iloc[0]) if (t_in in tickers and "Omsättning nästa år" in df.columns and pd.notna(df.loc[df["Ticker"].str.upper() == t_in, "Omsättning nästa år"].iloc[0])) else 0.0)

        c1, c2, c3 = st.columns([1, 1, 1])
        submit = c1.form_submit_button("Spara rad")
        upd_px = c2.form_submit_button("Uppdatera enbart kurs (denna)")
        upd_full = c3.form_submit_button("Full uppdatering (denna)")

    df2 = df.copy()

    if submit:
        if not t_in:
            st.warning("Ange ticker.")
            return df2
        # Dubbelticker-skydd: om ny ticker och redan finns -> fel
        exists_idx = df2.index[df2["Ticker"].astype(str).str.upper() == t_in]
        if len(exists_idx) == 0:
            # ny rad
            nr = {c: "" for c in FINAL_COLS}
            nr["Ticker"] = t_in
            nr["Bolagsnamn"] = namn
            nr["Valuta"] = valuta or "USD"
            nr["Antal"] = antal
            nr["GAV (SEK)"] = gav_sek
            nr["Omsättning idag"] = oms_idag if oms_idag > 0 else ""
            nr["Omsättning nästa år"] = oms_next if oms_next > 0 else ""
            df2 = pd.concat([df2, pd.DataFrame([nr])], ignore_index=True)
            # tidsstämpla manuella fält om satta
            if oms_idag > 0 and "TS_Omsättning idag" in df2.columns:
                df2.loc[df2["Ticker"] == t_in, "TS_Omsättning idag"] = now_stamp()
            if oms_next > 0 and "TS_Omsättning nästa år" in df2.columns:
                df2.loc[df2["Ticker"] == t_in, "TS_Omsättning nästa år"] = now_stamp()
            _save_df(df2, snapshot=False)
            st.success(f"Lade till {t_in}.")
            st.session_state["_df_ref"] = df2
        else:
            ridx = int(exists_idx[0])
            df2.at[ridx, "Bolagsnamn"] = namn
            df2.at[ridx, "Valuta"] = valuta or "USD"
            df2.at[ridx, "Antal"] = antal
            if "GAV (SEK)" in df2.columns:
                df2.at[ridx, "GAV (SEK)"] = gav_sek
            if oms_idag > 0:
                df2.at[ridx, "Omsättning idag"] = oms_idag
                if "TS_Omsättning idag" in df2.columns:
                    df2.at[ridx, "TS_Omsättning idag"] = now_stamp()
            if oms_next > 0:
                df2.at[ridx, "Omsättning nästa år"] = oms_next
                if "TS_Omsättning nästa år" in df2.columns:
                    df2.at[ridx, "TS_Omsättning nästa år"] = now_stamp()
            _save_df(df2, snapshot=False)
            st.success(f"Uppdaterade {t_in}.")
            st.session_state["_df_ref"] = df2

    if upd_px and current_ticker:
        df3, changed, msg = run_update_price(df2.copy(), t_in, save=True)
        st.session_state["_df_ref"] = df3
        if changed:
            st.success(f"{t_in}: {', '.join(next(iter(changed.values())))}")
        else:
            st.info(msg)

    if upd_full and current_ticker:
        df3, changed, dbg = run_update_full(df2.copy(), t_in, save=True)
        st.session_state["_df_ref"] = df3
        if changed:
            st.success(f"{t_in}: {', '.join(next(iter(changed.values())))}")
        else:
            st.info(dbg.get("status", "Inga ändringar."))

    # --- Manuell prognoslista
    st.markdown("---")
    st.markdown("### Manuell prognoslista (äldre uppdateringar först)")
    need = _build_requires_manual_df(df2, older_than_days=None)  # None => bara sortera på äldst TS
    if need.empty:
        st.info("Alla tickers har färska prognos-fält eller saknar krav.")
    else:
        st.dataframe(need, use_container_width=True, hide_index=True)

    return st.session_state.get("_df_ref", df2)


def _build_requires_manual_df(df: pd.DataFrame, older_than_days: int | None = None) -> pd.DataFrame:
    work = df.copy()
    out_rows = []
    from datetime import datetime, timedelta

    cutoff = None
    if older_than_days is not None:
        cutoff = datetime.utcnow() - timedelta(days=int(older_than_days))

    for _, row in work.iterrows():
        t = str(row.get("Ticker", "")).upper()
        d1 = str(row.get("TS_Omsättning idag", "")).strip()
        d2 = str(row.get("TS_Omsättning nästa år", "")).strip()

        def _parse(dt):
            try:
                return pd.to_datetime(dt)
            except Exception:
                return None

        p1 = _parse(d1)
        p2 = _parse(d2)

        oldest = None
        if p1 and p2:
            oldest = min(p1, p2)
        elif p1:
            oldest = p1
        elif p2:
            oldest = p2

        too_old = False
        if cutoff is not None and oldest is not None:
            too_old = oldest < cutoff
        elif cutoff is not None and oldest is None:
            too_old = True  # saknas -> betrakta som för gammal

        out_rows.append({
            "Ticker": t,
            "Senaste TS (min av två)": oldest.isoformat() if oldest else "",
            "Saknar Omsättning idag": "" if pd.notna(row.get("Omsättning idag")) and safe_float(row.get("Omsättning idag"), 0) > 0 else "Ja",
            "Saknar Omsättning nästa år": "" if pd.notna(row.get("Omsättning nästa år")) and safe_float(row.get("Omsättning nästa år"), 0) > 0 else "Ja",
            "Behöver manuell": "Ja" if (too_old or oldest is None) else "",
        })

    need = pd.DataFrame(out_rows)
    need = need.sort_values(by="Senaste TS (min av två)", ascending=True, na_position="first")
    return need


# ------------------------------------------------------------
# VY: Kontroll
# ------------------------------------------------------------
def kontrollvy(df: pd.DataFrame):
    st.subheader("Kontroll / Kvalitet")

    older_days = st.number_input("Visa poster äldre än (dagar)", min_value=1, value=60, step=1)
    need = _build_requires_manual_df(df, older_than_days=int(older_days))
    st.write("**Manuell prognoslista (filtrerad på ålder):**")
    st.dataframe(need, use_container_width=True, hide_index=True)

    # Visa äldst TS per ticker
    st.markdown("---")
    st.write("**Äldsta TS per ticker (för batch-kö):**")
    work = add_oldest_ts_col(df.copy())
    st.dataframe(work[["Ticker", "_oldest_ts"]].sort_values("_oldest_ts", ascending=True), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Huvud
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Aktieverktyg", layout="wide")
    _init_session()

    # Läs data (eller använd cache i session)
    if st.session_state["_df_ref"] is None:
        st.session_state["_df_ref"] = _read_df()
    df = st.session_state["_df_ref"].copy()

    # Sidopanel – valutakurser & uppdatering/batch
    user_rates = _sidebar_rates()
    _sidebar_batch_and_actions(df, user_rates)

    # Flikar
    tab_names = ["Investeringsförslag", "Portfölj", "Lägg till/uppdatera bolag", "Kontroll"]
    t1, t2, t3, t4 = st.tabs(tab_names)

    with t1:
        # Investeringsförslag (separat modul)
        try:
            visa_investeringsforslag(st.session_state["_df_ref"], user_rates)
        except Exception as e:
            st.error(f"Investeringsförslag kunde inte visas: {e}")
            # Fallback: enkel tabell
            cols = [c for c in ["Ticker", "Bolagsnamn", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Market Cap"] if c in df.columns]
            if cols:
                st.dataframe(df[cols].sort_values(by="P/S", ascending=True, na_position="last").head(20), use_container_width=True, hide_index=True)

    with t2:
        visa_portfolj(st.session_state["_df_ref"], user_rates)

    with t3:
        st.session_state["_df_ref"] = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates)

    with t4:
        kontrollvy(st.session_state["_df_ref"])


if __name__ == "__main__":
    main()
