# stockapp/editor.py
# -*- coding: utf-8 -*-
"""
Lägg till / uppdatera bolag:
- Bläddring 1/X mellan tickers
- Uppdatera kurs (Yahoo) / Full uppdatering (orchestrator)
- Manuell uppdatering av 'Omsättning i år (est.)' och 'Omsättning nästa år (est.)' med TS-stämpling
- Manuell prognoslista (äldst först)
- Lägg till nytt bolag (dubblettskydd)

Publikt API:
    lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st

# Konfiguration (valfri – vi skyddar med fallback)
try:
    from .config import FINAL_COLS
except Exception:
    FINAL_COLS = [
        "Ticker","Bolagsnamn","Valuta","Kurs","Antal aktier","Market Cap",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "GAV (SEK)","Omsättning i år (est.)","Omsättning nästa år (est.)",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"
    ]

# Utils vi förlitar oss på
from .utils import (
    ensure_schema,
    stamp_fields_ts,
    now_stamp,
    parse_date,
    safe_float,
    dedupe_tickers,
)

# Orchestrator och Yahoo-fallback (valfria)
try:
    from .fetchers.orchestrator import run_update_full as _run_update_full
except Exception:
    _run_update_full = None

try:
    from .fetchers.yahoo import get_live_price as _yahoo_price
except Exception:
    _yahoo_price = None


# -----------------------------
# Kolumnalias / hjälp
# -----------------------------
ALIAS_PRICE = ["Kurs", "Aktuell kurs"]

MANUAL_PROGNOS_FIELDS = ["Omsättning i år (est.)", "Omsättning nästa år (est.)"]

TS_SUFFIX = " TS"  # ex: "Omsättning i år (est.) TS"


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_min_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Skapa saknade nyckelkolumner som editor-vyn behöver."""
    base_cols = set(FINAL_COLS) | set(ALIAS_PRICE) | set([f + TS_SUFFIX for f in MANUAL_PROGNOS_FIELDS])
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan
    # Säkerställ datatyper på några centrala fält
    for c in ["Antal aktier", "GAV (SEK)"] + ALIAS_PRICE:
        if c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
    return df


def _runner_price(df: pd.DataFrame, tkr: str) -> Tuple[pd.DataFrame, str]:
    """Uppdatera ENDAST kurs (Yahoo-fallback). Stämplar TS för prisfältet."""
    if _yahoo_price is None:
        return df, "Yahoo-priskälla saknas"
    price_col = _first_existing_col(df, ALIAS_PRICE) or "Kurs"
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns ej i tabellen"
    try:
        p = _yahoo_price(str(tkr))
        if p and p > 0:
            df.loc[ridx, price_col] = float(p)
            df = stamp_fields_ts(df, [price_col], ts_suffix=TS_SUFFIX)
            df.loc[ridx, "Senast auto-uppdaterad"] = now_stamp()
            df.loc[ridx, "Senast uppdaterad källa"] = "Yahoo (pris)"
            return df, f"OK – {price_col}={float(p):.4f}"
        return df, "Pris saknas"
    except Exception as e:
        return df, f"Fel: {e}"


def _runner_full(df: pd.DataFrame, tkr: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    """Full uppdatering via orchestrator (SEC/FMP/Yahoo)."""
    if _run_update_full is None:
        # fallback till endast pris om orchestrator saknas
        return _runner_price(df, tkr)

    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(tkr).upper()]
    if len(ridx) == 0:
        return df, "Ticker finns ej i tabellen"

    try:
        out = _run_update_full(df.copy(), str(tkr), user_rates)  # API: (df, ticker, rates) -> (df_out, log) eller df_out
        if isinstance(out, tuple) and len(out) == 2:
            df2, log = out
        elif isinstance(out, pd.DataFrame):
            df2, log = out, "OK"
        else:
            return df, "Orchestrator: oväntat svar"

        # säkerställ att minst “Senast auto-uppdaterad” & källa sätts om orkestratorn inte gjorde det
        if "Senast auto-uppdaterad" in df2.columns:
            df2.loc[ridx, "Senast auto-uppdaterad"] = now_stamp()
        if "Senast uppdaterad källa" in df2.columns and (df2.loc[ridx, "Senast uppdaterad källa"].isna().any()):
            df2.loc[ridx, "Senast uppdaterad källa"] = "Orchestrator"

        # stämpla alla tekniska fält som uppdaterades? Låt orchestrator göra det primärt.
        return df2, str(log)
    except Exception as e:
        return df, f"Fel: {e}"


def _build_requires_manual_df(df: pd.DataFrame, older_than_days: Optional[int] = None) -> pd.DataFrame:
    """
    Lista för de fält som kräver manuell prognos (två fält).
    Sortering på äldst TS, NAs först. older_than_days = None -> alltid sorterad äldst först.
    """
    if df.empty:
        return pd.DataFrame(columns=["Ticker","Bolagsnamn","Fält","Senast uppdaterad"])

    rows = []
    for _, r in df.iterrows():
        for f in MANUAL_PROGNOS_FIELDS:
            ts_col = f + TS_SUFFIX
            rows.append({
                "Ticker": r.get("Ticker"),
                "Bolagsnamn": r.get("Bolagsnamn"),
                "Fält": f,
                "Senast uppdaterad": parse_date(r.get(ts_col))
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(by="Senast uppdaterad", ascending=True, na_position="first")
    if older_than_days is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(older_than_days))
        out = out[(out["Senast uppdaterad"].isna()) | (out["Senast uppdaterad"] < cutoff)]
    return out.reset_index(drop=True)


def _edit_current_card(df: pd.DataFrame, idx: int) -> Tuple[pd.DataFrame, bool]:
    """
    Renderar kortet för nuvarande rad/ticker och låter uppdatera:
      - Kurs (pris) – knapp
      - Full uppdatering – knapp
      - Manuell: Omsättning i år / nästa år + TS-stämpling
    Returnerar (df, changed)
    """
    changed = False
    row = df.iloc[idx]
    tkr = str(row.get("Ticker", ""))
    namn = str(row.get("Bolagsnamn", ""))
    st.subheader(f"{namn} ({tkr})")

    col1, col2 = st.columns(2)
    if col1.button("Uppdatera kurs", key=f"btn_price_{idx}"):
        ndf, msg = _runner_price(df.copy(), tkr)
        st.toast(f"{tkr}: {msg}")
        if not ndf.equals(df):
            df = ndf
            changed = True
            st.experimental_set_query_params()  # no-op reflow
            st.rerun()
    if col2.button("Full uppdatering", key=f"btn_full_{idx}"):
        ndf, msg = _runner_full(df.copy(), tkr, user_rates={})
        st.toast(f"{tkr}: {msg}")
        if not ndf.equals(df):
            df = ndf
            changed = True
            st.experimental_set_query_params()
            st.rerun()

    # Visa och uppdatera manuella prognoser
    st.markdown("**Manuella prognoser**")
    c3, c4 = st.columns(2)
    val_iy = c3.number_input(
        "Omsättning i år (est.)",
        value=float(safe_float(row.get("Omsättning i år (est.)"), np.nan)) if not math.isnan(safe_float(row.get("Omsättning i år (est.)"), np.nan)) else 0.0,
        step=1.0,
        key=f"iy_{idx}"
    )
    val_ny = c4.number_input(
        "Omsättning nästa år (est.)",
        value=float(safe_float(row.get("Omsättning nästa år (est.)"), np.nan)) if not math.isnan(safe_float(row.get("Omsättning nästa år (est.)"), np.nan)) else 0.0,
        step=1.0,
        key=f"ny_{idx}"
    )
    if st.button("Spara manuella prognoser", key=f"save_manu_{idx}"):
        df.at[df.index[idx], "Omsättning i år (est.)"] = float(val_iy) if val_iy else np.nan
        df.at[df.index[idx], "Omsättning nästa år (est.)"] = float(val_ny) if val_ny else np.nan
        df = stamp_fields_ts(df, MANUAL_PROGNOS_FIELDS, ts_suffix=TS_SUFFIX, row_index=df.index[idx])
        # uppdatera “Senast manuellt uppdaterad”
        df.at[df.index[idx], "Senast manuellt uppdaterad"] = now_stamp()
        st.success("Sparat manuella prognoser.")
        changed = True

    return df, changed


def _add_new_company(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Form för att lägga till nytt bolag, med dubblettskydd."""
    st.subheader("➕ Lägg till nytt bolag")
    c1, c2, c3 = st.columns([1,1,1])
    tkr = c1.text_input("Ticker", value="")
    namn = c2.text_input("Bolagsnamn", value="")
    valuta = c3.selectbox("Valuta", ["USD","SEK","EUR","NOK","CAD"], index=0)

    c4, c5 = st.columns(2)
    qty = c4.number_input("Antal aktier", min_value=0.0, step=1.0, value=0.0)
    gav = c5.number_input("GAV (SEK)", min_value=0.0, step=0.01, value=0.0)

    if st.button("Lägg till"):
        if not tkr.strip():
            st.warning("Ange en ticker.")
            return df, False

        # dubblettskydd (case-insensitive)
        if (df["Ticker"].astype(str).str.upper() == tkr.strip().upper()).any():
            st.error("Ticker finns redan – dubbletter ej tillåtna.")
            return df, False

        # Lägg till rad
        new_row = {c: np.nan for c in df.columns}
        new_row["Ticker"] = tkr.strip().upper()
        new_row["Bolagsnamn"] = namn.strip() if namn else tkr.strip().upper()
        new_row["Valuta"] = valuta
        # skapa pris-fält om saknas
        price_col = _first_existing_col(df, ALIAS_PRICE) or "Kurs"
        new_row[price_col] = np.nan
        new_row["Antal aktier"] = float(qty) if qty else np.nan
        new_row["GAV (SEK)"] = float(gav) if gav else np.nan
        # stämpla “Senast manuellt uppdaterad”
        new_row["Senast manuellt uppdaterad"] = now_stamp()

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.success(f"La till {new_row['Ticker']}.")
        return df, True

    return df, False


# -------------------------------------------------
# Publikt gränssnitt
# -------------------------------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Renderar vyn och returnerar ev. uppdaterat df (ingen automatisk sparning här).
    """
    st.header("✏️ Lägg till / uppdatera bolag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        # Ge möjlighet att lägga till första bolaget ändå
        df = pd.DataFrame(columns=FINAL_COLS)
        df = _ensure_min_schema(df)
        df, _ = _add_new_company(df)
        return df

    # Säkerställ schema
    df = ensure_schema(df, FINAL_COLS)
    df = _ensure_min_schema(df)

    # Dubbletter bort i minnet (informativt)
    df, dups = dedupe_tickers(df)
    if dups:
        st.info("Dubbletter ignoreras i minnet: " + ", ".join(dups))

    # Bläddringsindex
    st.session_state.setdefault("edit_index", 0)
    N = len(df)
    st.session_state["edit_index"] = max(0, min(st.session_state["edit_index"], N - 1))

    # Ticker-lista och hoppa till
    tickers = df["Ticker"].astype(str).tolist()
    cjump1, cjump2 = st.columns([2,1])
    sel = cjump1.selectbox("Välj bolag", options=tickers, index=st.session_state["edit_index"])
    if sel != tickers[st.session_state["edit_index"]]:
        st.session_state["edit_index"] = tickers.index(sel)
    cjump2.write(f"**{st.session_state['edit_index']+1} / {N}**")

    # Navigation
    colp1, colp2, colp3 = st.columns([1,2,1])
    if colp1.button("◀ Föregående", disabled=st.session_state["edit_index"] <= 0):
        st.session_state["edit_index"] -= 1
        st.rerun()
    colp2.markdown("<div style='text-align:center'><b>Bläddra</b></div>", unsafe_allow_html=True)
    if colp3.button("Nästa ▶", disabled=st.session_state["edit_index"] >= N - 1):
        st.session_state["edit_index"] += 1
        st.rerun()

    # Rendera kortet
    df, changed = _edit_current_card(df, st.session_state["edit_index"])

    st.divider()
    # Manuell prognoslista – äldst först (oavsett ålder)
    st.subheader("📝 Manuell prognoslista (äldst först)")
    need = _build_requires_manual_df(df, older_than_days=None)
    st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()
    # Lägg till nytt bolag
    df, added = _add_new_company(df)

    # Avslut
    return df
