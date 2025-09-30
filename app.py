# app.py — komplett
# -*- coding: utf-8 -*-

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

# =========================
# Importera våra moduler
# =========================
from stockapp.config import FINAL_COLS, TS_FIELDS, STANDARD_VALUTAKURSER
from stockapp.utils import (
    ensure_schema,
    now_stamp,
    stamp_fields_ts,
    uppdatera_berakningar,
)
from stockapp.storage import hamta_data, spara_data
from stockapp.rates import (
    las_sparade_valutakurser,
    spara_valutakurser,
    hamta_valutakurser_auto,
    hamta_valutakurs,
)
from stockapp.orchestrator import (
    run_update_full,
    run_update_price_only,
)

# Valfria (om du har dem)
_has_views = True
try:
    from stockapp.views import (
        kontrollvy,       # def kontrollvy(df)
        analysvy,         # def analysvy(df, user_rates)
        lagg_till_eller_uppdatera,  # def lagg_till_eller_uppdatera(df, user_rates)
        visa_portfolj,    # def visa_portfolj(df, user_rates)
    )
except Exception:
    _has_views = False

_has_invest = True
try:
    from stockapp.invest import visa_investeringsforslag  # def visa_investeringsforslag(df, user_rates)
except Exception:
    _has_invest = False

_has_batch = True
try:
    from stockapp.batch import sidebar_batch_controls, run_batch_update
except Exception:
    _has_batch = False


# =========================================
# Streamlit grundinställningar
# =========================================
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")


# =========================================
# Hjälpare (lokalt i app.py)
# =========================================
def _init_session_state_rates(saved: Dict[str, float]) -> None:
    """Preinit av state-nycklar för valutafälten så vi inte skriver efter widget-instansiering."""
    st.session_state.setdefault("rate_usd_input", float(saved.get("USD", STANDARD_VALUTAKURSER["USD"])))
    st.session_state.setdefault("rate_eur_input", float(saved.get("EUR", STANDARD_VALUTAKURSER["EUR"])))
    st.session_state.setdefault("rate_cad_input", float(saved.get("CAD", STANDARD_VALUTAKURSER["CAD"])))
    st.session_state.setdefault("rate_nok_input", float(saved.get("NOK", STANDARD_VALUTAKURSER["NOK"])))

def _load_df() -> pd.DataFrame:
    """Läs från Sheets en gång och buffra i session_state för att skona kvoter."""
    if "_df_ref" not in st.session_state:
        try:
            df = hamta_data()
        except Exception as e:
            st.error(f"⚠️ Kunde inte läsa Google Sheet: {e}")
            df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = ensure_schema(df)
        st.session_state["_df_ref"] = df
    return st.session_state["_df_ref"]

def _save_df(df: pd.DataFrame, snapshot: bool = False) -> None:
    """Spara och uppdatera referensen i session_state."""
    df2 = ensure_schema(df.copy())
    try:
        spara_data(df2, snapshot=snapshot)  # vår storage tar snapshot=True/False
        st.session_state["_df_ref"] = df2
    except TypeError:
        # om storage.spara_data inte accepterar snapshot-namnparametern i din version:
        spara_data(df2)
        st.session_state["_df_ref"] = df2

def _apply_vals_to_df(df: pd.DataFrame, tkr: str, vals: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Skriver in nyckel/värde i DF för given ticker. Skapar kolumner vid behov.
    TS-kolumner fylls via stamp_fields_ts (uppdaterar datum även om värdet ej ändrats).
    Returnerar (df, changed_fields) — changed_fields visar vilka fält som faktiskt skrevs (värde jämförs som str).
    """
    meta = meta or {}
    changed: List[str] = []
    if "Ticker" not in df.columns:
        df["Ticker"] = ""

    # Finn rad eller skapa
    mask = (df["Ticker"].astype(str).str.upper() == str(tkr).upper())
    if not mask.any():
        # skapa tom rad
        empty_row = {c: (0.0 if c not in ("Ticker",) and not str(c).startswith("TS_") else "") for c in df.columns}
        empty_row["Ticker"] = tkr.upper()
        df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
        mask = (df["Ticker"].astype(str).str.upper() == str(tkr).upper())

    idx = df.index[mask][0]

    # Skriv värden; skapa saknade kolumner vid behov
    for k, v in vals.items():
        if k not in df.columns:
            # skapa kolumn (gissa typ)
            if isinstance(v, (int, float, np.floating)):
                df[k] = 0.0
            else:
                df[k] = ""
        old = df.at[idx, k]
        # skriv alltid (vi vill ha färska datum även om värde = samma)
        df.at[idx, k] = v
        if str(old) != str(v):
            changed.append(k)

    # Tidsstämpla fält via TS_FIELDS (uppdatera alltid datum)
    df = stamp_fields_ts(df, idx, list(vals.keys()))

    # Uppdatera metadata-kolumner om de finns
    if "Senast auto-uppdaterad" in df.columns:
        df.at[idx, "Senast auto-uppdaterad"] = now_stamp()
    if "Senast uppdaterad källa" in df.columns and "sources" in meta:
        df.at[idx, "Senast uppdaterad källa"] = " + ".join(meta["sources"]) if isinstance(meta["sources"], list) else str(meta["sources"])

    return df, changed


# =========================================
# Sidopanel: Valutor, Snabbkörningar, Batch
# =========================================
def _sidebar_rates() -> Dict[str, float]:
    st.sidebar.header("💱 Valutakurser → SEK")

    saved = {}
    try:
        saved = las_sparade_valutakurser()
    except Exception as e:
        st.sidebar.warning(f"⚠️ Kunde inte läsa sparade kurser: {e}")
        saved = {k: float(v) for k, v in STANDARD_VALUTAKURSER.items()}

    # Init keys innan widgets skapas
    _init_session_state_rates(saved)

    # Auto-hämtning — uppdatera state först, rendera inputs efteråt
    if st.sidebar.button("🌐 Hämta kurser automatiskt"):
        try:
            auto_rates, misses, provider = hamta_valutakurser_auto()
            st.sidebar.success(f"Valutakurser (källa: {provider}) hämtade.")
            if misses:
                st.sidebar.warning("Missade par:\n- " + "\n- ".join(misses))
            st.session_state.rate_usd_input = float(auto_rates.get("USD", st.session_state.rate_usd_input))
            st.session_state.rate_eur_input = float(auto_rates.get("EUR", st.session_state.rate_eur_input))
            st.session_state.rate_cad_input = float(auto_rates.get("CAD", st.session_state.rate_cad_input))
            st.session_state.rate_nok_input = float(auto_rates.get("NOK", st.session_state.rate_nok_input))
        except Exception as e:
            st.sidebar.error(f"Fel vid automatisk hämtning: {e}")

    # Widgets (bundna till session_state)
    usd = st.sidebar.number_input("USD → SEK", key="rate_usd_input", step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", key="rate_eur_input", step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", key="rate_cad_input", step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", key="rate_nok_input", step=0.01, format="%.4f")

    cols = st.sidebar.columns(2)
    with cols[0]:
        if st.button("💾 Spara kurser"):
            try:
                spara_valutakurser({"USD": usd, "EUR": eur, "CAD": cad, "NOK": nok, "SEK": 1.0})
                st.sidebar.success("Valutakurser sparade.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte spara kurser: {e}")
    with cols[1]:
        if st.button("↺ Läs sparade"):
            try:
                sr = las_sparade_valutakurser()
                st.session_state.rate_usd_input = float(sr.get("USD", usd))
                st.session_state.rate_eur_input = float(sr.get("EUR", eur))
                st.session_state.rate_cad_input = float(sr.get("CAD", cad))
                st.session_state.rate_nok_input = float(sr.get("NOK", nok))
                st.sidebar.info("Hämtade från bladet.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte läsa sparade: {e}")

    return {"USD": float(usd), "EUR": float(eur), "CAD": float(cad), "NOK": float(nok), "SEK": 1.0}


def _sidebar_quick_updates(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Snabb uppdatering")

    # Uppdatera endast KURS för alla
    if st.sidebar.button("📈 Uppdatera KURS (alla)"):
        tickers = list(df["Ticker"].astype(str))
        n = len(tickers)
        prog = st.sidebar.progress(0.0)
        status = st.sidebar.empty()
        changed_total = 0
        for i, tkr in enumerate(tickers, start=1):
            status.write(f"Uppdaterar kurs {i}/{n}: {tkr}")
            vals, dbg, meta = run_update_price_only(tkr)
            if vals:
                df, changed = _apply_vals_to_df(df, tkr, vals, meta)
                changed_total += len(changed)
            prog.progress(i / max(n, 1))
        if changed_total > 0:
            _save_df(df, snapshot=False)
            st.sidebar.success(f"Klart. Uppdaterade fält: {changed_total}")
        else:
            st.sidebar.info("Ingen förändring att spara.")

    # Uppdatera specifik ticker (pris eller full)
    st.sidebar.markdown("**Uppdatera en ticker**")
    tkr_one = st.sidebar.text_input("Ticker (Yahoo-format)", key="one_ticker").strip().upper()
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Endast KURS (1 st)") and tkr_one:
            vals, dbg, meta = run_update_price_only(tkr_one)
            if vals:
                df, changed = _apply_vals_to_df(df, tkr_one, vals, meta)
                _save_df(df, snapshot=False)
                st.sidebar.success(f"{tkr_one}: kursuppdatering sparad ({len(changed)} fält).")
            else:
                st.sidebar.warning(f"{tkr_one}: inga fält att skriva.")
    with c2:
        if st.button("FULL uppdatering (1 st)") and tkr_one:
            vals, dbg, meta = run_update_full(tkr_one)
            if vals:
                df, changed = _apply_vals_to_df(df, tkr_one, vals, meta)
                df = uppdatera_berakningar(df, user_rates)
                _save_df(df, snapshot=False)
                st.sidebar.success(f"{tkr_one}: full uppdatering sparad ({len(changed)} fält).")
            else:
                st.sidebar.warning(f"{tkr_one}: inga fält att skriva.")
    return df


def _sidebar_batch_and_actions(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Batch-körning")
    if not _has_batch:
        st.sidebar.info("Batch-modulen är inte installerad. (stockapp.batch)")
        return df

    # Ställ in runners i state (om ej redan satta)
    st.session_state.setdefault("_runner_full", run_update_full)
    st.session_state.setdefault("_runner_price_only", run_update_price_only)

    # Låter modulens panel bygga kön och köra
    df2 = sidebar_batch_controls(
        df,
        user_rates,
        save_cb=_save_df,
        recompute_cb=lambda d: uppdatera_berakningar(d, user_rates),
        runner=st.session_state["_runner_full"],
        runner_price_only=st.session_state["_runner_price_only"],
    )
    return df2


# =========================================
# Huvudprogram
# =========================================
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Valutor (sidopanel)
    user_rates = _sidebar_rates()

    # 2) Läs data (buffers i session)
    df = _load_df()

    # 3) Snabbuppdateringar (kurs alla + singel)
    df = _sidebar_quick_updates(df, user_rates)

    # 4) Batch-körning (sidopanel)
    df = _sidebar_batch_and_actions(df, user_rates)

    # 5) Meny
    st.sidebar.markdown("---")
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Kontroll", "Analys", "Lägg till / uppdatera", "Investeringsförslag", "Portfölj"],
    )

    # 6) Visa vald vy
    if meny == "Kontroll":
        if _has_views:
            kontrollvy(df)
        else:
            st.info("Kontroll-vy saknas (installera stockapp.views).")

    elif meny == "Analys":
        if _has_views:
            # Ofta vill vi räkna om på visningsdata (utan att spara)
            df_calc = uppdatera_berakningar(df.copy(), user_rates)
            analysvy(df_calc, user_rates)
        else:
            st.info("Analys-vy saknas (installera stockapp.views).")

    elif meny == "Lägg till / uppdatera":
        if _has_views:
            df2 = lagg_till_eller_uppdatera(df, user_rates)
            if df2 is not None and not df2.equals(df):
                _save_df(df2, snapshot=False)
        else:
            st.info("Redigerings-vy saknas (installera stockapp.views).")

    elif meny == "Investeringsförslag":
        if _has_invest:
            df_calc = uppdatera_berakningar(df.copy(), user_rates)
            visa_investeringsforslag(df_calc, user_rates)
        else:
            st.info("Investeringsförslag saknas (installera stockapp.invest).")

    elif meny == "Portfölj":
        if _has_views:
            df_calc = uppdatera_berakningar(df.copy(), user_rates)
            visa_portfolj(df_calc, user_rates)
        else:
            st.info("Portfölj-vy saknas (installera stockapp.views).")


if __name__ == "__main__":
    main()
