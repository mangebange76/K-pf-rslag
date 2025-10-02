# -*- coding: utf-8 -*-
"""
stockapp.invest
----------------
Investeringsförslag med sektors-/stilbaserad poängsättning,
bläddringsfunktion (1/X), filtrering på sektor och risklabel,
samt expander per bolag med nyckeltal.

Förutsätter:
- score_dataframe()/score_row()/compute_ps_target i stockapp.scoring
- FINAL_COLS, PROPOSALS_PAGE_SIZE i stockapp.config
- format_large_number, ensure_schema, to_float, risk_label_from_mcap i stockapp.utils
"""

from __future__ import annotations
from typing import List, Dict
import math

import numpy as np
import pandas as pd
import streamlit as st

from .config import FINAL_COLS, PROPOSALS_PAGE_SIZE
from .utils import ensure_schema, to_float, format_large_number, risk_label_from_mcap
from .scoring import score_dataframe, compute_ps_target


# ------------------------------------------------------------
# Hjälpare lokalt
# ------------------------------------------------------------
def _alias_columns_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skapa alias som scoring-modulen förväntar sig:
      - "P/S-snitt" från "P/S-snitt (Q1..Q4)" om finns
      - "Omsättning i år" från "Omsättning i år (M)" om finns
      - "Utestående aktier" från "Utestående aktier (milj.)" (miljoner)
    Muterar en kopia; original df lämnas orörd.
    """
    out = df.copy()

    if "P/S-snitt" not in out.columns and "P/S-snitt (Q1..Q4)" in out.columns:
        out["P/S-snitt"] = pd.to_numeric(out["P/S-snitt (Q1..Q4)"], errors="coerce")

    if "Omsättning i år" not in out.columns and "Omsättning i år (M)" in out.columns:
        # användarens input i miljoner (bolagets valuta)
        out["Omsättning i år"] = pd.to_numeric(out["Omsättning i år (M)"], errors="coerce")

    if "Utestående aktier" not in out.columns and "Utestående aktier (milj.)" in out.columns:
        out["Utestående aktier"] = pd.to_numeric(out["Utestående aktier (milj.)"], errors="coerce")

    # Sektor-fält – scoring läser "Sector" (eng); aliasa från "Sektor" om behövs
    if "Sector" not in out.columns and "Sektor" in out.columns:
        out["Sector"] = out["Sektor"]

    return out


def _page_controls(total: int, key_prefix: str = "inv") -> int:
    """
    Visar bläddringsknappar och returnerar nuvarande sida (1-baserad).
    Lagrar state under nycklar: f"{key_prefix}_page" och f"{key_prefix}_page_size".
    """
    if f"{key_prefix}_page_size" not in st.session_state:
        st.session_state[f"{key_prefix}_page_size"] = int(PROPOSALS_PAGE_SIZE)

    # sidstorlek
    csz1, csz2 = st.columns([3, 1])
    with csz2:
        new_size = st.number_input(
            "Poster / sida",
            min_value=1,
            max_value=50,
            value=int(st.session_state[f"{key_prefix}_page_size"]),
            key=f"{key_prefix}_page_size_input",
        )
        st.session_state[f"{key_prefix}_page_size"] = int(new_size)

    page_size = int(st.session_state[f"{key_prefix}_page_size"])
    pages = max(1, math.ceil(max(0, total) / max(1, page_size)))

    if f"{key_prefix}_page" not in st.session_state:
        st.session_state[f"{key_prefix}_page"] = 1

    # knappar
    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("◀ Föregående", disabled=st.session_state[f"{key_prefix}_page"] <= 1, key=f"{key_prefix}_prev"):
        st.session_state[f"{key_prefix}_page"] = max(1, st.session_state[f"{key_prefix}_page"] - 1)
        st.rerun()
    c2.markdown(
        f"<div style='text-align:center'>**{st.session_state[f'{key_prefix}_page']} / {pages}**</div>",
        unsafe_allow_html=True,
    )
    if c3.button("Nästa ▶", disabled=st.session_state[f"{key_prefix}_page"] >= pages, key=f"{key_prefix}_next"):
        st.session_state[f"{key_prefix}_page"] = min(pages, st.session_state[f"{key_prefix}_page"] + 1)
        st.rerun()

    return int(st.session_state[f"{key_prefix}_page"])


def _format_metric(val) -> str:
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return "–"
    try:
        return f"{float(val):.2f}"
    except Exception:
        return str(val)


# ------------------------------------------------------------
# Publik vy-funktion
# ------------------------------------------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    """
    Visar investeringsförslag:
      - Stil: Tillväxt / Utdelning (påverkar viktningen)
      - Filter: Sektor + Risklabel
      - Sortering: TotalScore (desc)
      - Paging med 1/X
      - Expander per bolag med nyckeltal och riktkurs/uppsida
    """
    st.header("📈 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Säkerställ schema och aliasa kolumner för scoring
    base = ensure_schema(df.copy(), FINAL_COLS)
    base = _alias_columns_for_scoring(base)

    # Risklabel om saknas
    if "Risklabel" not in base.columns or base["Risklabel"].isna().all():
        if "Market Cap" in base.columns:
            base["Risklabel"] = base["Market Cap"].apply(risk_label_from_mcap)
        else:
            base["Risklabel"] = "Unknown"

    # Stil-val
    style = st.radio(
        "Strategi",
        options=["Tillväxt", "Utdelning"],
        horizontal=True,
        key="inv_style",
    )
    style_key = "dividend" if style == "Utdelning" else "growth"

    # Scora
    scored = score_dataframe(base, style=style_key)

    # Filtrering
    cols_fil = st.columns([1, 1, 1])
    # sektor
    sektorer: List[str] = ["(Alla)"]
    if "Sektor" in scored.columns:
        sektorer += sorted(
            [s for s in scored["Sektor"].dropna().astype(str).unique() if s and s.lower() != "nan"]
        )
    elif "Sector" in scored.columns:
        sektorer += sorted(
            [s for s in scored["Sector"].dropna().astype(str).unique() if s and s.lower() != "nan"]
        )
    val_sektor = cols_fil[0].selectbox("Sektor", sektorer, key="inv_sektor")

    # risk
    risk_opts = ["(Alla)", "Mega", "Large", "Mid", "Small", "Micro", "Unknown"]
    val_risk = cols_fil[1].selectbox("Risklabel", risk_opts, key="inv_risk")

    # min-täckning (gynnar bolag med fler datapunkter)
    min_cov = cols_fil[2].slider("Min. täckning (%)", min_value=0, max_value=100, value=0, step=5, key="inv_cov_min")

    work = scored.copy()

    if val_sektor != "(Alla)":
        if "Sektor" in work.columns:
            work = work[work["Sektor"].astype(str) == val_sektor]
        elif "Sector" in work.columns:
            work = work[work["Sector"].astype(str) == val_sektor]

    if val_risk != "(Alla)":
        work = work[work["Risklabel"].astype(str) == val_risk]

    if "Coverage" in work.columns:
        work = work[pd.to_numeric(work["Coverage"], errors="coerce").fillna(0) >= float(min_cov)]

    # sortera – högst TotalScore först
    if "TotalScore" in work.columns:
        work = work.sort_values(by="TotalScore", ascending=False, na_position="last")

    total = len(work)
    if total == 0:
        st.info("Inga bolag matchar filtret ännu.")
        return

    # Bläddring
    page = _page_controls(total, key_prefix="inv")
    page_size = int(st.session_state.get("inv_page_size", PROPOSALS_PAGE_SIZE))
    start = (page - 1) * page_size
    end = start + page_size
    page_df = work.iloc[start:end].reset_index(drop=True)

    # Rendera rader
    for idx, row in page_df.iterrows():
        global_rank = start + idx + 1
        tkr = str(row.get("Ticker", "") or "")
        namn = str(row.get("Bolagsnamn", "") or row.get("Name", "") or tkr)
        valuta = str(row.get("Valuta", "") or "USD")

        col_head1, col_head2 = st.columns([3, 1])
        with col_head1:
            st.subheader(f"#{global_rank} – {namn} ({tkr})")
        with col_head2:
            # Badge för rekommendation
            rec = str(row.get("Recommendation", "") or "")
            if rec:
                st.markdown(f"<div style='text-align:right; font-weight:700;'>{rec}</div>", unsafe_allow_html=True)

        # Top-metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TotalScore", f"{_format_metric(row.get('TotalScore'))}")
        c2.metric("Täckning", f"{_format_metric(row.get('Coverage'))}%")
        # Market cap
        mcap = to_float(row.get("Market Cap", np.nan))
        c3.metric("Market Cap (nu)", format_large_number(mcap, valuta))
        # Uppsida / riktkurs
        calc = compute_ps_target(row)
        ups = row.get("Uppsida (%)", None)
        if ups is None and calc.get("Uppsida (%)") is not None:
            ups = calc.get("Uppsida (%)")
        rik = row.get("Riktkurs (valuta)", None)
        if rik is None and calc.get("Riktkurs (valuta)") is not None:
            rik = calc.get("Riktkurs (valuta)")
        c4.metric("Uppsida (PS)", f"{_format_metric(ups)}%")

        with st.expander("Visa nyckeltal / historik"):
            # Visa ett komprimerat urval av nyckeltal
            left, right = st.columns(2)

            # Vänsterkolumn – värdering & lönsamhet
            left.write("**Värdering & lönsamhet**")
            left.write(f"- P/S (TTM): **{_format_metric(row.get('P/S'))}**")
            if "P/S-snitt (Q1..Q4)" in row.index:
                left.write(f"- P/S-snitt (4Q): **{_format_metric(row.get('P/S-snitt (Q1..Q4)'))}**")
            left.write(f"- EV/EBITDA (ttm): **{_format_metric(row.get('EV/EBITDA (ttm)'))}**")
            left.write(f"- P/B: **{_format_metric(row.get('P/B'))}**")
            left.write(f"- ROE (%): **{_format_metric(row.get('ROE (%)'))}**")
            left.write(f"- Gross margin (%): **{_format_metric(row.get('Gross margin (%)'))}**")
            left.write(f"- Operating margin (%): **{_format_metric(row.get('Operating margin (%)'))}**")
            left.write(f"- Net margin (%): **{_format_metric(row.get('Net margin (%)'))}**")
            left.write(f"- FCF Yield (%): **{_format_metric(row.get('FCF Yield (%)'))}**")

            # Högerkolumn – balans/utdelning/övrigt
            right.write("**Balans / utdelning**")
            right.write(f"- Debt/Equity: **{_format_metric(row.get('Debt/Equity'))}**")
            right.write(f"- Net debt / EBITDA: **{_format_metric(row.get('Net debt / EBITDA'))}**")
            right.write(f"- Dividend yield (%): **{_format_metric(row.get('Dividend yield (%)'))}**")
            right.write(f"- Payout (FCF) (%): **{_format_metric(row.get('Dividend payout (FCF) (%)'))}**")
            right.write(f"- Utestående aktier (milj.): **{_format_metric(row.get('Utestående aktier (milj.)'))}**")
            right.write(f"- Sektor: **{row.get('Sektor', row.get('Sector', '–'))}**")
            # Riktkurs presenteras här också
            if rik is not None:
                right.write(f"- Riktkurs ({valuta}): **{_format_metric(rik)}**")

        st.divider()
