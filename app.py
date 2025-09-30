 # app.py — komplett
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---- Våra moduler -----------------------------------------------------------
# (Alla dessa ska finnas i stockapp/-paketet. Appen har robusta fallbackar.)
from stockapp.config import (
    APP_TITLE,
    SHEET_NAME,
    FINAL_COLS,
    TS_FIELDS,                     # nyckel: fältnamn -> "TS: ..."
    STANDARD_VALUTAKURSER,
)

from stockapp.storage import hamta_data, spara_data
from stockapp.rates import las_sparade_valutakurser, spara_valutakurser, hamta_valutakurser_auto, hamta_valutakurs
from stockapp.utils import (
    ensure_schema,
    now_stamp,
    with_backoff,
    format_large_number,
    currency_fmt,
    dedupe_tickers,
    ts_now_ymd,
)
from stockapp.manuals import build_requires_manual_df

# Hämt-kedja och enskilda fetchers
from stockapp.fetchers.orchestrator import (
    run_update_full,          # (ticker: str, df: pd.DataFrame) -> (df, changed_fields, log)
    run_update_price_only,    # (ticker: str, df: pd.DataFrame) -> (df, changed_fields, log)
)

# Investering/poängsättning
try:
    from stockapp.scoring import compute_company_score, sector_weighted_score
except Exception:
    compute_company_score = None
    sector_weighted_score = None

try:
    from stockapp.invest import visa_investeringsforslag  # integrerad vy
except Exception:
    visa_investeringsforslag = None


# ---- Streamlit grundinställningar -------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")


# =============================================================================
# Hjälpare för session state och init
# =============================================================================
def _init_session():
    keys_defaults = {
        "_df_ref": None,                  # aktiv DataFrame i minnet
        "rate_usd": float(STANDARD_VALUTAKURSER["USD"]),
        "rate_eur": float(STANDARD_VALUTAKURSER["EUR"]),
        "rate_cad": float(STANDARD_VALUTAKURSER["CAD"]),
        "rate_nok": float(STANDARD_VALUTAKURSER["NOK"]),
        "batch_queue": [],                # lista med tickers som står i kö
        "batch_done": [],                 # färdiga tickers denna omgång
        "batch_last_log": [],
        "batch_order": "Äldst TS först", # default sortering
        "batch_chunk_size": 10,           # antal att köra per “kör”
        "last_single_log": None,          # logg efter enskild uppdatering
        "selected_row_idx": 0,            # index i redigeringsvyn
    }
    for k, v in keys_defaults.items():
        st.session_state.setdefault(k, v)


# =============================================================================
# Sidopanel: Valutor (utan experimental_rerun)
# =============================================================================
def _sidebar_rates() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")

    # Init widget inputs med sparade/senaste (engångs)
    saved = {}
    try:
        saved = las_sparade_valutakurser()
    except Exception as e:
        st.sidebar.warning(f"Kunde inte läsa sparade kurser: {e}")

    # Synka session default om första körningen
    for cur, key in [("USD", "rate_usd"), ("EUR", "rate_eur"), ("CAD", "rate_cad"), ("NOK", "rate_nok")]:
        if key not in st.session_state or st.session_state[key] is None:
            st.session_state[key] = float(saved.get(cur, STANDARD_VALUTAKURSER[cur]))

    # Egna inmatningar
    col1, col2 = st.sidebar.columns(2)
    with col1:
        usd = st.number_input("USD → SEK", value=float(st.session_state["rate_usd"]), step=0.01, format="%.4f", key="rate_usd_input")
        cad = st.number_input("CAD → SEK", value=float(st.session_state["rate_cad"]), step=0.01, format="%.4f", key="rate_cad_input")
    with col2:
        eur = st.number_input("EUR → SEK", value=float(st.session_state["rate_eur"]), step=0.01, format="%.4f", key="rate_eur_input")
        nok = st.number_input("NOK → SEK", value=float(st.session_state["rate_nok"]), step=0.01, format="%.4f", key="rate_nok_input")

    # Knapp: Auto-hämta
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        try:
            auto_rates, misses, provider = hamta_valutakurser_auto()
            st.session_state["rate_usd"] = float(auto_rates.get("USD", usd))
            st.session_state["rate_eur"] = float(auto_rates.get("EUR", eur))
            st.session_state["rate_cad"] = float(auto_rates.get("CAD", cad))
            st.session_state["rate_nok"] = float(auto_rates.get("NOK", nok))
            st.sidebar.success(f"Kurser uppdaterade från {provider}.")
            if misses:
                st.sidebar.warning("Vissa par misslyckades:\n- " + "\n- ".join(misses))
            # återspegla i inputs direkt
            st.session_state["rate_usd_input"] = st.session_state["rate_usd"]
            st.session_state["rate_eur_input"] = st.session_state["rate_eur"]
            st.session_state["rate_cad_input"] = st.session_state["rate_cad"]
            st.session_state["rate_nok_input"] = st.session_state["rate_nok"]
        except Exception as e:
            st.sidebar.error(f"Misslyckades auto-hämta: {e}")

    # Knapp: Spara
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("💾 Spara kurser"):
            rates = {
                "USD": float(st.session_state["rate_usd_input"]),
                "EUR": float(st.session_state["rate_eur_input"]),
                "CAD": float(st.session_state["rate_cad_input"]),
                "NOK": float(st.session_state["rate_nok_input"]),
                "SEK": 1.0,
            }
            try:
                spara_valutakurser(rates)
                # även uppdatera sessionens “senast anv.” värden
                st.session_state["rate_usd"] = rates["USD"]
                st.session_state["rate_eur"] = rates["EUR"]
                st.session_state["rate_cad"] = rates["CAD"]
                st.session_state["rate_nok"] = rates["NOK"]
                st.sidebar.success("Sparat.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte spara: {e}")
    with col_s2:
        if st.button("↻ Ladda om sparade"):
            try:
                sr = las_sparade_valutakurser()
                st.session_state["rate_usd"] = float(sr.get("USD", st.session_state["rate_usd"]))
                st.session_state["rate_eur"] = float(sr.get("EUR", st.session_state["rate_eur"]))
                st.session_state["rate_cad"] = float(sr.get("CAD", st.session_state["rate_cad"]))
                st.session_state["rate_nok"] = float(sr.get("NOK", st.session_state["rate_nok"]))
                st.session_state["rate_usd_input"] = st.session_state["rate_usd"]
                st.session_state["rate_eur_input"] = st.session_state["rate_eur"]
                st.session_state["rate_cad_input"] = st.session_state["rate_cad"]
                st.session_state["rate_nok_input"] = st.session_state["rate_nok"]
                st.sidebar.success("Laddade om sparade kurser.")
            except Exception as e:
                st.sidebar.error(f"Misslyckades ladda om: {e}")

    return {
        "USD": float(st.session_state["rate_usd"]),
        "EUR": float(st.session_state["rate_eur"]),
        "CAD": float(st.session_state["rate_cad"]),
        "NOK": float(st.session_state["rate_nok"]),
        "SEK": 1.0,
    }


# =============================================================================
# Batch-körning (i sidopanel)
# =============================================================================
def _order_oldest_ts_first(df: pd.DataFrame) -> List[str]:
    # sortera på minsta TS över alla spårade fält
    ts_cols = [c for c in df.columns if str(c).startswith("TS:")]
    work = df.copy()
    for c in ts_cols:
        work[c] = pd.to_datetime(work[c], errors="coerce")
    if ts_cols:
        work["_min_ts"] = work[ts_cols].min(axis=1, skipna=True)
    else:
        work["_min_ts"] = pd.NaT
    work["_min_ts_fill"] = work["_min_ts"].fillna(pd.Timestamp("1900-01-01"))
    work = work.sort_values(by=["_min_ts_fill", "Bolagsnamn", "Ticker"], ascending=[True, True, True])
    return [str(t).upper() for t in work["Ticker"].tolist()]

def _order_az(df: pd.DataFrame) -> List[str]:
    work = df.copy()
    work = work.sort_values(by=["Bolagsnamn", "Ticker"])
    return [str(t).upper() for t in work["Ticker"].tolist()]

def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: Dict[str, float]) -> Optional[pd.DataFrame]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Batch-uppdatering")

    sort_mode = st.sidebar.radio("Ordning", ["Äldst TS först", "A–Ö (bolagsnamn)"], index=0, key="batch_order_radio")
    chunk = st.sidebar.number_input("Antal per körning", min_value=1, max_value=100, value=int(st.session_state["batch_chunk_size"]), step=1)

    # Välj Tickers att köa
    if st.sidebar.button("Ladda kö (ersätter)"):
        if sort_mode.startswith("Äldst"):
            order = _order_oldest_ts_first(df)
        else:
            order = _order_az(df)
        st.session_state["batch_queue"] = order[:]  # ersätt
        st.session_state["batch_done"] = []
        st.session_state["batch_last_log"] = []
        st.session_state["batch_chunk_size"] = int(chunk)
        st.sidebar.success(f"Kö skapad ({len(order)} tickers).")

    # Kör nästa chunk
    if st.sidebar.button("▶️ Kör nästa"):
        queue = st.session_state.get("batch_queue", [])
        if not queue:
            st.sidebar.info("Kön är tom.")
            return None

        to_run = queue[: int(st.session_state["batch_chunk_size"])]
        st.sidebar.write(f"Kör {len(to_run)} st...")

        progress = st.sidebar.progress(0.0, text="Startar...")
        total = len(to_run)
        new_df = df.copy()
        logs = []

        for i, tkr in enumerate(to_run, start=1):
            try:
                new_df, changed, log = run_update_full(tkr, new_df)
                logs.append({tkr: {"changed": changed, "log": log}})
                st.session_state["batch_done"].append(tkr)
                progress.progress(i / total, text=f"{i}/{total}: {tkr}")
            except Exception as e:
                logs.append({tkr: {"error": str(e)}})
                progress.progress(i / total, text=f"{i}/{total}: {tkr} (fel)")

        # Ta bort körda ur kön
        st.session_state["batch_queue"] = queue[int(st.session_state["batch_chunk_size"]):]
        st.session_state["batch_last_log"] = logs

        # Spara om något ändrats (run_update_full ska spara själv om det görs internt; men här gör vi inte autospar för att vara snäll mot kvoter)
        st.session_state["_df_ref"] = new_df
        st.sidebar.success(f"Klart {len(to_run)}/{total}. Kvar i kö: {len(st.session_state['batch_queue'])}")
        st.sidebar.caption("Körlogg (senaste chunk):")
        st.sidebar.json(logs)

    # Visning av köstatus + quick-actions
    with st.sidebar.expander("Köstatus", expanded=False):
        q = st.session_state.get("batch_queue", [])
        d = st.session_state.get("batch_done", [])
        st.write(f"🧾 I kö: {len(q)} | ✅ Klara denna omgång: {len(d)}")
        if d:
            st.write(", ".join(d[:50]) + (" ..." if len(d) > 50 else ""))

    return st.session_state.get("_df_ref", df)


# =============================================================================
# Vyer
# =============================================================================
def kontrollvy(df: pd.DataFrame):
    st.header("🧭 Kontroll")

    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    ts_cols = [c for c in df.columns if str(c).startswith("TS:")]
    work = df.copy()
    for c in ts_cols:
        work[c] = pd.to_datetime(work[c], errors="coerce")
    if ts_cols:
        work["_min_ts"] = work[ts_cols].min(axis=1, skipna=True)
    else:
        work["_min_ts"] = pd.NaT
    work = work.sort_values(by=["_min_ts", "Bolagsnamn"]).head(20)
    cols_show = ["Ticker", "Bolagsnamn"] + ts_cols
    st.dataframe(work[cols_show], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📝 Manuell prognoslista (Omsättning idag/nästa år)")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30)
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inget ser urgammalt ut just nu.")
    else:
        st.warning(f"{len(need)} rader behöver troligen manuell uppdatering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📒 Senaste körlogg (Batch)")
    log = st.session_state.get("batch_last_log") or []
    if not log:
        st.info("Ingen batch-körning ännu.")
    else:
        st.json(log)


def analysvy(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("📈 Analys")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    st.session_state["analysis_idx"] = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter) - 1),
                                                      value=int(st.session_state.get("analysis_idx", 0)), step=1)
    st.selectbox("Eller välj i lista", etiketter, index=int(st.session_state["analysis_idx"]) if etiketter else 0, key="analysis_select")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("⬅️ Föregående", key="analysis_prev"):
            st.session_state["analysis_idx"] = max(0, int(st.session_state["analysis_idx"]) - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analysis_next"):
            st.session_state["analysis_idx"] = min(len(etiketter) - 1, int(st.session_state["analysis_idx"]) + 1)

    st.write(f"Post {int(st.session_state['analysis_idx']) + 1}/{len(etiketter)}")

    r = vis_df.iloc[int(st.session_state["analysis_idx"])]
    cols = [c for c in [
        "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Utestående aktier", "P/S",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "CAGR 5 år (%)", "Antal aktier", "Årlig utdelning", "Market Cap",
        "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
    ] if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)


def _save_row(df: pd.DataFrame, row_idx: int, payload: Dict[str, float | str]) -> pd.DataFrame:
    """
    Skriver in payload i df på rad row_idx. Sätter dubblettskydd för Ticker.
    Sätter 'Senast manuellt uppdaterad' och uppdaterar TS: för P/S- och Omsättning-fält.
    """
    # Dubblettskydd för ticker
    new_t = str(payload.get("Ticker", "")).upper().strip()
    if new_t:
        dups = df.index[(df["Ticker"].str.upper() == new_t) & (df.index != row_idx)].tolist()
        if dups:
            raise ValueError(f"Ticker {new_t} finns redan i tabellen (rad {dups[0]+1}).")

    for k, v in payload.items():
        if k not in df.columns:
            continue
        df.at[row_idx, k] = v

    # Sätt “Senast manuellt uppdaterad”
    if "Senast manuellt uppdaterad" in df.columns:
        df.at[row_idx, "Senast manuellt uppdaterad"] = ts_now_ymd()

    # Stämpla TS: fält
    track_fields = ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år", "Utestående aktier"]
    for f in track_fields:
        if f in payload and f in TS_FIELDS and TS_FIELDS[f] in df.columns:
            df.at[row_idx, TS_FIELDS[f]] = ts_now_ymd()

    return df


def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorterbar redigeringslista
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)", "Äldst uppdaterade först (alla fält)"], index=0)
    if sort_val.startswith("Äldst"):
        ts_cols = [c for c in df.columns if str(c).startswith("TS:")]
        w = df.copy()
        for c in ts_cols:
            w[c] = pd.to_datetime(w[c], errors="coerce")
        w["_min_ts"] = w[ts_cols].min(axis=1, skipna=True) if ts_cols else pd.NaT
        vis_df = w.sort_values(by=["_min_ts", "Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"])

    etiketter = [""] + [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    st.session_state["selected_row_idx"] = int(st.session_state.get("selected_row_idx", 0))
    valt = st.selectbox("Välj bolag (lämna tomt för nytt)", etiketter, index=min(st.session_state["selected_row_idx"], len(etiketter) - 1), key="edit_select")

    # Prev/Next
    col_prev, col_cnt, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state["selected_row_idx"] = max(0, int(st.session_state["selected_row_idx"]) - 1)
    with col_cnt:
        st.write(f"Post {int(st.session_state['selected_row_idx'])}/{max(0, len(etiketter) - 1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state["selected_row_idx"] = min(len(etiketter) - 1, int(st.session_state["selected_row_idx"]) + 1)

    # Hitta rad eller skapa ny
    if valt and valt in etiketter:
        tkr = valt[valt.rfind("(") + 1 : valt.rfind(")")]
        sel = df.index[df["Ticker"].str.upper() == str(tkr).upper()].tolist()
        if sel:
            row_idx = sel[0]
        else:
            row_idx = None
    else:
        row_idx = None

    # Form
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=(df.at[row_idx, "Ticker"] if row_idx is not None else "")).upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(df.at[row_idx, "Utestående aktier"]) if row_idx is not None else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(df.at[row_idx, "Antal aktier"]) if row_idx is not None else 0.0)
            gav_sek = st.number_input("GAV (SEK)", value=float(df.at[row_idx, "GAV (SEK)"]) if (row_idx is not None and "GAV (SEK)" in df.columns) else 0.0)

            ps  = st.number_input("P/S",   value=float(df.at[row_idx, "P/S"]) if row_idx is not None else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(df.at[row_idx, "P/S Q1"]) if row_idx is not None else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(df.at[row_idx, "P/S Q2"]) if row_idx is not None else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(df.at[row_idx, "P/S Q3"]) if row_idx is not None else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(df.at[row_idx, "P/S Q4"]) if row_idx is not None else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)", value=float(df.at[row_idx, "Omsättning idag"]) if row_idx is not None else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(df.at[row_idx, "Omsättning nästa år"]) if row_idx is not None else 0.0)
            st.caption("OBS! Dessa två fält uppdateras **inte** automatiskt — manuellt ansvar.")

            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över manuella 0-värden):**")
            st.write("- Tidsstämpling för ändrade fält (inkl. manuell TS).")
            st.write("- Dubblettskydd för ticker.")

        spar = st.form_submit_button("💾 Spara")

    # Spara
    if spar:
        if not ticker:
            st.error("Ticker kan inte vara tom.")
            return df
        payload = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "GAV (SEK)": gav_sek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
        }

        df2 = df.copy()
        if row_idx is None:
            # ny rad
            blank = {c: "" for c in df2.columns}
            # numeriska default nollor
            for c in df2.columns:
                if any(x in c.lower() for x in ["kurs", "omsättning", "p/s", "utdelning", "cagr", "antal", "riktkurs", "aktier", "snitt", "market cap", "gav"]):
                    blank[c] = 0.0
            df2 = pd.concat([df2, pd.DataFrame([blank])], ignore_index=True)
            row_idx = df2.index[-1]

        try:
            df2 = _save_row(df2, row_idx, payload)
            st.success("Sparar hela arket…")
            spara_data(df2, do_snapshot=False)
            st.session_state["_df_ref"] = df2
            df = df2
            st.success("Sparat.")
        except Exception as e:
            st.error(f"Kunde inte spara: {e}")

    # Enskild uppdatering – knappar
    st.subheader("⚡ Enskild uppdatering")
    colu1, colu2 = st.columns(2)
    this_tkr = ticker if spar or row_idx is None else (df.at[row_idx, "Ticker"] if row_idx is not None else "")
    this_tkr = this_tkr or (df.at[row_idx, "Ticker"] if row_idx is not None else "")

    with colu1:
        if st.button("🔄 Uppdatera kurs (endast pris)"):
            t = str(this_tkr).upper().strip()
            if not t:
                st.warning("Välj eller ange en ticker först.")
            else:
                try:
                    df2, changed, log = run_update_price_only(t, st.session_state["_df_ref"])
                    st.session_state["_df_ref"] = df2
                    st.session_state["last_single_log"] = {t: {"changed": changed, "log": log}}
                    st.success(f"Pris uppdaterat för {t}.")
                except Exception as e:
                    st.error(f"{t} : Fel: {e}")

    with colu2:
        if st.button("🧠 Full auto (alla fält)"):
            t = str(this_tkr).upper().strip()
            if not t:
                st.warning("Välj eller ange en ticker först.")
            else:
                try:
                    df2, changed, log = run_update_full(t, st.session_state["_df_ref"])
                    st.session_state["_df_ref"] = df2
                    st.session_state["last_single_log"] = {t: {"changed": changed, "log": log}}
                    st.success(f"Auto-uppdatering klar för {t}.")
                except Exception as e:
                    st.error(f"{t} : Fel: {e}")

    if st.session_state.get("last_single_log"):
        with st.expander("Senaste enskilda uppdateringslogg", expanded=False):
            st.json(st.session_state["last_single_log"])

    st.divider()
    st.markdown("### 📝 Manuell prognoslista (snabbvy)")
    need = build_requires_manual_df(df, older_than_days=None)
    st.dataframe(need, use_container_width=True, hide_index=True)

    return st.session_state.get("_df_ref", df)


def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("📦 Min portfölj")
    port = df[df.get("Antal aktier", 0) > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]

    st.markdown(f"**Totalt portföljvärde:** {format_large_number(total_värde, 'SEK')}")
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    st.markdown(f"**Total kommande utdelning:** {format_large_number(tot_utd, 'SEK')}")
    st.markdown(f"**Ungefärlig månadsutdelning:** {format_large_number(tot_utd/12.0, 'SEK')}")

    show_cols = [c for c in [
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
        "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning (SEK)", "GAV (SEK)"
    ] if c in port.columns]

    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    _init_session()
    st.title(APP_TITLE)

    # Valutor
    user_rates = _sidebar_rates()

    # Ladda data (en gång per körning om inte redan i session)
    if st.session_state["_df_ref"] is None:
        df = hamta_data()
        df = ensure_schema(df)
        # Dedupe tickers (inte hårdbryt – bara info)
        df2, dupes = dedupe_tickers(df)
        if dupes:
            st.sidebar.warning(f"{len(dupes)} dubbletter av tickers upptäcktes och ignoreras i beräkningar.")
        st.session_state["_df_ref"] = df2

    # Batchkontroller i sidopanel (uppdaterar ev sessionens df)
    _sidebar_batch_and_actions(st.session_state["_df_ref"], user_rates)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Kontroll", "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"], index=0)

    if meny == "Kontroll":
        kontrollvy(st.session_state["_df_ref"])
    elif meny == "Analys":
        analysvy(st.session_state["_df_ref"], user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        st.session_state["_df_ref"] = lagg_till_eller_uppdatera(st.session_state["_df_ref"], user_rates)
    elif meny == "Investeringsförslag":
        if callable(visa_investeringsforslag):
            # investeringsförslagsvyn sköter poäng, sektor-vikter och navigation
            visa_investeringsforslag(st.session_state["_df_ref"], user_rates)
        else:
            st.warning("Vyn 'Investeringsförslag' saknas. Fallback: topp-20 på P/S-snitt.")
            base = st.session_state["_df_ref"].copy()
            if "P/S-snitt" in base.columns:
                base = base[base["P/S-snitt"] > 0].sort_values(by="P/S-snitt", ascending=True).head(20)
                st.dataframe(base[["Ticker", "Bolagsnamn", "P/S-snitt"]], use_container_width=True, hide_index=True)
            else:
                st.info("Inget P/S-snitt i data.")
    elif meny == "Portfölj":
        visa_portfolj(st.session_state["_df_ref"], user_rates)


if __name__ == "__main__":
    main()
