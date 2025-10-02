# -*- coding: utf-8 -*-
"""
stockapp.editor
---------------
Lägg till / uppdatera bolag:
- Bläddra mellan tickers (◀/▶).
- Lägg till ny ticker (Ticker, Bolagsnamn, Valuta).
- Redigera fält: Antal aktier, GAV (SEK), Bolagsnamn, Valuta.
- Uppdatera kurs (Yahoo) / Full uppdatering (orchestrator).
- Stämplar TS-kolumner (TS Kurs, TS Full, TS Omsättning i år / nästa år).
- Visar manuell prognoslista (äldst uppdaterad först).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Imports med fallback
# ------------------------------------------------------------
try:
    from .config import FINAL_COLS
except Exception:
    FINAL_COLS = []

from .utils import (
    ensure_schema,
    to_float,
    parse_date,
    now_stamp,
    stamp_fields_ts,
)

# Orchestrator (full uppdatering) & Yahoo (pris)
try:
    from .fetchers.orchestrator import run_update_full  # type: ignore
except Exception:  # pragma: no cover
    run_update_full = None  # type: ignore

try:
    from .fetchers.yahoo import get_live_price as _yahoo_price  # type: ignore
except Exception:  # pragma: no cover
    _yahoo_price = None  # type: ignore


# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
_MAN_PROG_FIELDS = ["Omsättning i år (M)", "Omsättning nästa år (M)"]
_TS_FOR_FIELD = {
    "Omsättning i år (M)": "TS Omsättning i år",
    "Omsättning nästa år (M)": "TS Omsättning nästa år",
}

def _ensure_editor_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Se till att vi har de fält som behövs i editorn."""
    work = ensure_schema(df.copy(), FINAL_COLS if FINAL_COLS else df.columns.tolist())

    # Nyttiga basfält
    for c in ["Ticker", "Bolagsnamn", "Valuta", "Kurs", "Antal aktier", "GAV (SEK)"]:
        if c not in work.columns:
            work[c] = np.nan if c in ("Kurs", "GAV (SEK)") else ""

    # TS-fält vi använder
    for ts in ["TS Kurs", "TS Full", "TS Omsättning i år", "TS Omsättning nästa år"]:
        if ts not in work.columns:
            work[ts] = ""

    # Manuell prognos-fält
    for c in _MAN_PROG_FIELDS:
        if c not in work.columns:
            work[c] = np.nan

    # Typer
    work["Ticker"] = work["Ticker"].astype(str).str.upper()
    # undvik "None" i UI
    work["Bolagsnamn"] = work["Bolagsnamn"].fillna("")
    work["Valuta"] = work["Valuta"].fillna("USD")

    # numeriska
    for c in ["Kurs", "Antal aktier", "GAV (SEK)"] + _MAN_PROG_FIELDS:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    return work


def _runner_price(df: pd.DataFrame, tkr: str) -> Tuple[pd.DataFrame, str]:
    """Uppdatera ENDAST kurs (Yahoo) och stämpla TS Kurs."""
    if _yahoo_price is None:
        return df, "Yahoo-källa saknas"
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns ej i tabellen"
    try:
        price = _yahoo_price(str(tkr))
        if price and price > 0:
            out = df.copy()
            out.loc[ridx, "Kurs"] = float(price)
            # Stämpla
            out.loc[ridx, "TS Kurs"] = now_stamp()
            return out, "OK"
        return df, "Pris saknas"
    except Exception as e:  # pragma: no cover
        return df, f"Fel: {e}"


def _runner_full(df: pd.DataFrame, tkr: str) -> Tuple[pd.DataFrame, str]:
    """Full uppdatering (orchestrator) och TS Full."""
    if run_update_full is None:
        return _runner_price(df, tkr)

    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns ej i tabellen"
    try:
        out = run_update_full(df.copy(), tkr, {})  # user_rates skickas normalt inte behövas här
        if isinstance(out, tuple) and len(out) == 2:
            out_df, msg = out
        elif isinstance(out, pd.DataFrame):
            out_df, msg = out, "OK"
        else:
            return df, "Orchestrator: oväntat svar"

        out_df.loc[ridx, "TS Full"] = now_stamp()
        return out_df, msg
    except Exception as e:  # pragma: no cover
        return df, f"Fel: {e}"


def _manual_queue(df: pd.DataFrame, older_than_days: Optional[int] = None) -> pd.DataFrame:
    """Bygg lista över rader där manuell prognos bör ses över (äldst först)."""
    if df.empty:
        return pd.DataFrame(columns=["Ticker", "Bolagsnamn", "Fält", "Senast uppdaterad"])

    rows = []
    for _, r in df.iterrows():
        for f in _MAN_PROG_FIELDS:
            ts = _TS_FOR_FIELD.get(f, "")
            rows.append(
                {
                    "Ticker": r.get("Ticker"),
                    "Bolagsnamn": r.get("Bolagsnamn"),
                    "Fält": f,
                    "Senast uppdaterad": parse_date(r.get(ts)),
                }
            )
    need = pd.DataFrame(rows)
    need = need.sort_values(by="Senast uppdaterad", ascending=True, na_position="first")
    if older_than_days is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(older_than_days))
        need = need[(need["Senast uppdaterad"].isna()) | (need["Senast uppdaterad"] < cutoff)]
    return need.reset_index(drop=True)


# ------------------------------------------------------------
# Publik vy
# ------------------------------------------------------------
def visa_editor(df: pd.DataFrame, user_rates: Dict[str, float], on_save=None) -> pd.DataFrame:
    """
    Interaktiv vy för att lägga till / uppdatera bolag.
    Returnerar *alltid* df (kan vara samma objekt om inget ändrats).
    Om `on_save` ges (callable), anropas den med df när något sparas.
    """
    st.header("✏️ Lägg till / uppdatera bolag")

    work = _ensure_editor_schema(df)

    # --- Lägg till ny ticker -------------------------------------------------
    with st.expander("➕ Lägg till ny ticker", expanded=False):
        c1, c2, c3 = st.columns([1, 2, 1])
        new_tkr = c1.text_input("Ticker", value="", placeholder="t.ex. NVDA").strip().upper()
        new_name = c2.text_input("Bolagsnamn", value="", placeholder="valfritt")
        new_ccy = c3.text_input("Valuta", value="USD").strip().upper() or "USD"

        if st.button("Lägg till"):
            if not new_tkr:
                st.warning("Ange ticker.")
            elif new_tkr in work["Ticker"].astype(str).str.upper().tolist():
                st.warning("Tickern finns redan.")
            else:
                add_row = {
                    "Ticker": new_tkr,
                    "Bolagsnamn": new_name or new_tkr,
                    "Valuta": new_ccy,
                    "Antal aktier": 0.0,
                    "GAV (SEK)": np.nan,
                }
                work = pd.concat([work, pd.DataFrame([add_row])], ignore_index=True)
                st.success(f"La till {new_tkr}.")
                if callable(on_save):
                    on_save(work)

    if work.empty:
        st.info("Inga bolag i databasen ännu.")
        return work

    # --- Bläddra mellan befintliga ------------------------------------------
    tickers: List[str] = work["Ticker"].astype(str).tolist()
    if "_edit_idx" not in st.session_state:
        st.session_state["_edit_idx"] = 0
    st.session_state["_edit_idx"] = int(
        max(0, min(st.session_state["_edit_idx"], len(tickers) - 1))
    )

    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("◀ Föregående", disabled=st.session_state["_edit_idx"] <= 0):
        st.session_state["_edit_idx"] -= 1
        st.rerun()
    c2.markdown(
        f"<div style='text-align:center'>**{st.session_state['_edit_idx']+1} / {len(tickers)}**</div>",
        unsafe_allow_html=True,
    )
    if c3.button("Nästa ▶", disabled=st.session_state["_edit_idx"] >= len(tickers) - 1):
        st.session_state["_edit_idx"] += 1
        st.rerun()

    current_tkr = tickers[st.session_state["_edit_idx"]]
    ridx = work.index[work["Ticker"].astype(str) == current_tkr][0]
    row = work.loc[ridx]

    st.subheader(f"{row.get('Bolagsnamn','')} ({current_tkr})")

    # --- Snabbredigering av fält --------------------------------------------
    ec1, ec2, ec3, ec4 = st.columns([2, 1, 1, 1])
    name_new = ec1.text_input("Bolagsnamn", value=str(row.get("Bolagsnamn") or ""))
    val_new = ec2.text_input("Valuta", value=str(row.get("Valuta") or "USD")).upper()
    qty_new = ec3.number_input("Antal aktier", value=float(to_float(row.get("Antal aktier"), 0.0)), min_value=0.0, step=1.0)
    gav_new = ec4.number_input("GAV (SEK)", value=float(to_float(row.get("GAV (SEK)"), 0.0)), min_value=0.0, step=0.01, format="%.2f")

    if st.button("💾 Spara rad"):
        work.loc[ridx, "Bolagsnamn"] = name_new
        work.loc[ridx, "Valuta"] = val_new or "USD"
        work.loc[ridx, "Antal aktier"] = qty_new
        work.loc[ridx, "GAV (SEK)"] = gav_new
        st.success("Sparat.")
        if callable(on_save):
            on_save(work)

    # --- Uppdateringar ------------------------------------------------------
    uc1, uc2 = st.columns(2)
    if uc1.button("⚡ Uppdatera kurs (Yahoo)"):
        work2, msg = _runner_price(work, current_tkr)
        st.toast(f"{current_tkr}: {msg}")
        if work2 is not work:
            work = work2
            if callable(on_save):
                on_save(work)

    if uc2.button("🧩 Full uppdatering (orchestrator)"):
        work2, msg = _runner_full(work, current_tkr)
        st.toast(f"{current_tkr}: {msg}")
        if work2 is not work:
            work = work2
            if callable(on_save):
                on_save(work)

    # --- Manuell prognoslista ----------------------------------------------
    st.subheader("📝 Manuell prognoslista (äldst först)")
    need = _manual_queue(work, older_than_days=None)
    st.dataframe(need, use_container_width=True, hide_index=True)

    # --- Snabbuppdatering av manuella prognoser för aktuell ticker ----------
    st.markdown("**Uppdatera prognoser för valt bolag**")
    mp1, mp2, mp3 = st.columns([1, 1, 1])
    y_now = mp1.number_input(
        "Omsättning i år (M)",
        value=float(to_float(row.get("Omsättning i år (M)"), np.nan)) if not pd.isna(row.get("Omsättning i år (M)")) else 0.0,
        min_value=0.0,
        step=1.0,
    )
    y_next = mp2.number_input(
        "Omsättning nästa år (M)",
        value=float(to_float(row.get("Omsättning nästa år (M)"), np.nan)) if not pd.isna(row.get("Omsättning nästa år (M)")) else 0.0,
        min_value=0.0,
        step=1.0,
    )
    if mp3.button("💾 Spara prognoser"):
        work.loc[ridx, "Omsättning i år (M)"] = float(y_now) if y_now > 0 else np.nan
        work.loc[ridx, "Omsättning nästa år (M)"] = float(y_next) if y_next > 0 else np.nan
        # TS-stämpla respektive fält
        fields = []
        if y_now > 0:
            fields.append("Omsättning i år (M)")
        if y_next > 0:
            fields.append("Omsättning nästa år (M)")
        if fields:
            # bygg dynamiska TS-kolumner: "TS Omsättning i år" etc
            ts_cols = [_TS_FOR_FIELD[f] for f in fields if f in _TS_FOR_FIELD]
            work = stamp_fields_ts(work, fields=ts_cols, ts_suffix="")  # TS-kolumnerna är redan fulla namn
        st.success("Prognoser sparade.")
        if callable(on_save):
            on_save(work)

    return work


# Bakåtkompatibelt alias (tidigare namn i appen)
lagg_till_eller_uppdatera = visa_editor
