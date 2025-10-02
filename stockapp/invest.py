# -*- coding: utf-8 -*-
"""
stockapp.invest
---------------
Investeringsförslag med sektorsviktad scoring, coverage-bonus och sidbläddring.
Visar tydliga kort per bolag med expander för nyckeltal.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Konfiguration / utils / scoring
try:
    from .config import PROPOSALS_PAGE_SIZE, FINAL_COLS
except Exception:
    PROPOSALS_PAGE_SIZE = 10
    FINAL_COLS = []

from .utils import (
    ensure_schema,
    to_float,
    format_large_number,
    risk_label_from_mcap,
)
from .scoring import score_dataframe


# ---------------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------------

SECTOR_COL_CANDIDATES = ["Sektor", "Sector"]
NAME_COL_CANDIDATES = ["Bolagsnamn", "Namn", "Name", "Company"]
TICKER_COL = "Ticker"

PS_Q_COLS = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]


def _sector_col(df: pd.DataFrame) -> str:
    for c in SECTOR_COL_CANDIDATES:
        if c in df.columns:
            return c
    # skapa svenskt namn om saknas
    df["Sektor"] = ""
    return "Sektor"


def _name_col(df: pd.DataFrame) -> str:
    for c in NAME_COL_CANDIDATES:
        if c in df.columns:
            return c
    df["Bolagsnamn"] = df.get(TICKER_COL, "")
    return "Bolagsnamn"


def _prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Säkerställ minsta schema + beräkna härledda kolumner som används i vyn & scoring.
    """
    work = ensure_schema(df.copy(), FINAL_COLS if FINAL_COLS else df.columns.tolist())

    # P/S-snitt (Q1..Q4)
    for c in PS_Q_COLS:
        if c not in work.columns:
            work[c] = np.nan
    if "P/S-snitt (Q1..Q4)" not in work.columns:
        work["P/S-snitt (Q1..Q4)"] = pd.to_numeric(
            work[PS_Q_COLS].mean(axis=1), errors="coerce"
        )

    # Risklabel (från Market Cap) om saknas
    if "Risklabel" not in work.columns:
        # skapa tom kolumn först för att undvika SettingWithCopy
        work["Risklabel"] = ""
    if "Market Cap" in work.columns:
        work["Risklabel"] = work["Market Cap"].apply(risk_label_from_mcap).astype(str)
    else:
        work["Risklabel"] = work["Risklabel"].replace("", "Unknown")

    # Konsekventa namn/typer
    if TICKER_COL not in work.columns:
        work[TICKER_COL] = ""
    work[TICKER_COL] = work[TICKER_COL].astype(str)

    # Pris/Kurs normaliserat fält
    if "Kurs" not in work.columns and "Aktuell kurs" in work.columns:
        work["Kurs"] = work["Aktuell kurs"]
    if "Kurs" in work.columns:
        work["Kurs"] = pd.to_numeric(work["Kurs"], errors="coerce")

    return work


def _metrics_for_expander(work: pd.DataFrame) -> List[str]:
    """
    Lista med nyckeltal som visas i expandern (bara de som faktiskt finns).
    Ordningen är *avsiktligt* kuraterad.
    """
    preferred = [
        "Valuta",
        "Sektor",
        "Risklabel",
        "Market Cap",
        "Utestående aktier (milj.)",
        "P/S",
        "P/S Q1",
        "P/S Q2",
        "P/S Q3",
        "P/S Q4",
        "P/S-snitt (Q1..Q4)",
        "EV/EBITDA (ttm)",
        "Debt/Equity",
        "Net debt / EBITDA",
        "P/B",
        "Gross margin (%)",
        "Operating margin (%)",
        "Net margin (%)",
        "ROE (%)",
        "FCF Yield (%)",
        "Dividend yield (%)",
        "Dividend payout (FCF) (%)",
        "Omsättning i år (M)",
        "Omsättning nästa år (M)",
        "Uppsida (%)",
        "Riktkurs (valuta)",
    ]
    return [c for c in preferred if c in work.columns]


def _fmt_val(v) -> str:
    if isinstance(v, (int, float)) and not pd.isna(v):
        return f"{v:.2f}"
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    return str(v)


# ---------------------------------------------------------------------
# Publik vy
# ---------------------------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    """
    Investeringsförslag med filter, sidbläddring och scoring.
    """
    st.header("📈 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga träffar.")
        return

    work = _prep_dataframe(df)

    # Scoring – välj stil (growth/dividend)
    with st.expander("⚙️ Poängsättning", expanded=False):
        style = st.radio(
            "Välj stil",
            options=["growth", "dividend"],
            horizontal=True,
            index=0,
        )
        st.caption("Scoren vägs sektor- och stilberoende (se scoring.py).")

    scored = score_dataframe(work, style=style)

    # Filter & UI-val
    sector_col = _sector_col(scored)
    name_col = _name_col(scored)

    sectors = ["Alla"] + sorted(
        [s for s in scored[sector_col].astype(str).dropna().unique() if s and s != "nan"]
    )
    sel_sector = st.selectbox("Sektor", sectors)

    risk_opts = ["Alla"] + sorted(
        [s for s in scored["Risklabel"].astype(str).dropna().unique() if s and s != "nan"]
    )
    sel_risk = st.selectbox("Risklabel", risk_opts)

    page_size = st.number_input(
        "Poster per sida", min_value=1, max_value=50, value=int(PROPOSALS_PAGE_SIZE)
    )

    # Tillämpa filter
    filt = scored.copy()
    if sel_sector != "Alla":
        filt = filt[filt[sector_col].astype(str) == sel_sector]
    if sel_risk != "Alla":
        filt = filt[filt["Risklabel"].astype(str) == sel_risk]

    # Sortera: om TotalScore finns → desc, annars på lägst P/S-snitt
    if "TotalScore" in filt.columns:
        filt = filt.sort_values(by="TotalScore", ascending=False, na_position="last")
    else:
        filt = filt.sort_values(by="P/S-snitt (Q1..Q4)", ascending=True, na_position="last")

    # Sidbläddring
    total = len(filt)
    if total == 0:
        st.info("Inga träffar.")
        return

    pages = max(1, math.ceil(total / page_size))
    page_key = "_invest_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    st.session_state[page_key] = max(1, min(st.session_state[page_key], pages))

    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("◀ Föregående", disabled=st.session_state[page_key] <= 1):
        st.session_state[page_key] -= 1
        st.rerun()
    c2.markdown(
        f"<div style='text-align:center'>**{st.session_state[page_key]} / {pages}**</div>",
        unsafe_allow_html=True,
    )
    if c3.button("Nästa ▶", disabled=st.session_state[page_key] >= pages):
        st.session_state[page_key] += 1
        st.rerun()

    start = (st.session_state[page_key] - 1) * page_size
    end = start + page_size
    page_df = filt.iloc[start:end].reset_index(drop=True)

    # Kortsvisning
    for _, row in page_df.iterrows():
        with st.container(border=True):
            title = f"{row.get(name_col, '')} ({row.get(TICKER_COL, '')})"
            st.subheader(title)

            cols = st.columns(4)
            # P/S (TTM) / snitt
            ps = to_float(row.get("P/S"))
            ps_avg = to_float(row.get("P/S-snitt (Q1..Q4)"))
            cols[0].metric("P/S", "–" if np.isnan(ps) else f"{ps:.2f}")
            cols[1].metric("P/S-snitt (4Q)", "–" if np.isnan(ps_avg) else f"{ps_avg:.2f}")

            # Market Cap
            mcap = row.get("Market Cap", np.nan)
            cols[2].metric("Market Cap", format_large_number(mcap, "USD"))

            # Score + rekommendation
            tag = ""
            score = row.get("TotalScore", np.nan)
            rec = row.get("Recommendation", "")
            if not pd.isna(score):
                if score >= 85:
                    tag = "✅"
                elif score >= 70:
                    tag = "👍"
                elif score >= 55:
                    tag = "🙂"
                elif score >= 40:
                    tag = "⚠️"
                else:
                    tag = "🛑"
            cols[3].markdown(
                f"**Score:** {('–' if pd.isna(score) else f'{float(score):.1f}')}  \n"
                f"**Rek:** {tag} {rec}"
            )

            # Expander med nyckeltal
            with st.expander("Visa nyckeltal / prognoser"):
                fields = _metrics_for_expander(scored)
                kv = []
                for f in fields:
                    val = row.get(f)
                    # Snyggare tal för kassa/market cap etc
                    if f in ("Market Cap",) and not pd.isna(val):
                        kv.append((f, format_large_number(val, "USD")))
                    else:
                        kv.append((f, _fmt_val(val)))

                # Två kolumner för läsbarhet
                mid = (len(kv) + 1) // 2
                left, right = st.columns(2)
                with left:
                    for k, v in kv[:mid]:
                        st.write(f"- **{k}:** {v}")
                with right:
                    for k, v in kv[mid:]:
                        st.write(f"- **{k}:** {v}")
