# stockapp/investment.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from .utils import safe_float, format_large_number
from .rates import hamta_valutakurs
from .scoring import (
    growth_score,
    dividend_score,
    assign_action_label,
    risk_label_from_mcap,
)

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    """
    Investeringsförslag (Tillväxt/Utdelning) med:
      - sektor- & riskfilter
      - score (0..100)
      - uppsida mot vald riktkurs
      - handlingsetikett: Köp / Håll / Trimma / Sälj inkl. motivering
      - robust mot saknade kolumner
    """
    st.header("💡 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Val: fokus & riktkursfält
    mode = st.radio("Fokus", ["Tillväxt", "Utdelning"], horizontal=True, index=0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas i potential-beräkningen?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    # Filter: sektor & risketikett
    sector_vals = ["Alla"] + sorted(
        [s for s in df.get("Sektor", pd.Series(dtype=str)).astype(str).unique() if s]
    )
    sektor = st.selectbox("Filtrera på sektor", sector_vals, index=0)

    risk_vals = ["Alla", "Micro", "Small", "Mid", "Large", "Mega"]
    risk_choice = st.selectbox("Filtrera på risklabel (Market Cap)", risk_vals, index=0)

    # Basurval – kräver pris & riktkurs > 0
    work = df.copy()
    work["Aktuell kurs"] = pd.to_numeric(work.get("Aktuell kurs", 0), errors="coerce").fillna(0.0)
    work[riktkurs_val] = pd.to_numeric(work.get(riktkurs_val, 0), errors="coerce").fillna(0.0)

    base = work[(work["Aktuell kurs"] > 0) & (work[riktkurs_val] > 0)].copy()

    if sektor != "Alla":
        base = base[base.get("Sektor", "").astype(str) == sektor]

    if risk_choice != "Alla":
        base["_RiskLabel"] = base.get("Market Cap", 0).apply(risk_label_from_mcap)
        base = base[base["_RiskLabel"] == risk_choice]

    if base.empty:
        st.info("Inga bolag matchar filtren just nu.")
        return

    # Potential & score
    def _potential(row):
        px = safe_float(row.get("Aktuell kurs"))
        tgt = safe_float(row.get(riktkurs_val))
        if px <= 0 or tgt <= 0:
            return 0.0
        return (tgt - px) / px * 100.0

    base["Potential (%)"] = base.apply(_potential, axis=1)

    if mode == "Utdelning":
        base["_Score"] = base.apply(lambda r: dividend_score(r), axis=1)
    else:
        base["_Score"] = base.apply(lambda r: growth_score(r, riktkurs_col=riktkurs_val), axis=1)

    # Sortera: högst score först, sedan potential
    base = base.sort_values(by=["_Score", "Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # Navigering
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base) - 1)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ Föregående"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with c2:
        st.write(f"Förslag {st.session_state.forslags_index + 1}/{len(base)}")
    with c3:
        if st.button("➡️ Nästa"):
            st.session_state.forslags_index = min(len(base) - 1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljandel (om möjligt) + GAV i SEK
    vx = hamta_valutakurs(rad.get("Valuta", "SEK"), user_rates)

    if "Antal aktier" in df.columns:
        pos_value_sek = safe_float(rad.get("Antal aktier")) * safe_float(rad.get("Aktuell kurs")) * vx
        port = df[pd.to_numeric(df.get("Antal aktier", 0), errors="coerce").fillna(0.0) > 0].copy()
        if not port.empty:
            port["Växelkurs"] = port.get("Valuta", "SEK").apply(lambda v: hamta_valutakurs(v, user_rates))
            port["Värde (SEK)"] = (
                pd.to_numeric(port.get("Antal aktier", 0), errors="coerce").fillna(0.0)
                * pd.to_numeric(port.get("Aktuell kurs", 0), errors="coerce").fillna(0.0)
                * port["Växelkurs"]
            )
            tot_val = float(port["Värde (SEK)"].sum())
        else:
            tot_val = 0.0
        pos_weight = (pos_value_sek / tot_val * 100.0) if tot_val > 0 else None
    else:
        pos_weight = None

    gav_sek = safe_float(rad.get("GAV (SEK)")) if "GAV (SEK)" in df.columns else None

    # Handlingsetikett + motivering
    label, reason, metrics = assign_action_label(
        rad,
        mode=mode,
        riktkurs_col=riktkurs_val,
        pos_weight_pct=pos_weight,
        gav_sek=gav_sek,
        fx_to_sek=vx,
    )

    # Presentation
    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad.get('Ticker','')})")

    ps_now = safe_float(rad.get("P/S"))
    ps_avg = safe_float(rad.get("P/S-snitt"))
    mcap = safe_float(rad.get("Market Cap"))
    uts_m = safe_float(rad.get("Utestående aktier"))
    yld = metrics.get("yield_pct", 0.0)

    lines = [
        f"- **Aktuell kurs:** {safe_float(rad.get('Aktuell kurs')):.2f} {rad.get('Valuta','')}",
        f"- **{riktkurs_val}:** {safe_float(rad.get(riktkurs_val)):.2f} {rad.get('Valuta','')}",
        f"- **Uppsida:** {metrics.get('potential_pct', 0.0):.2f} %",
        f"- **P/S (nu):** {ps_now:.2f}  | **P/S-snitt:** {ps_avg:.2f}",
        f"- **Utestående aktier (M):** {uts_m:.2f}",
        f"- **Market Cap:** {format_large_number(mcap)}",
    ]
    if mode == "Utdelning":
        lines.append(f"- **Direktavkastning:** {yld:.2f} %")

    label_emoji = {"Köp": "🟢", "Håll": "🟡", "Trimma": "🟠", "Sälj": "🔴"}.get(label, "🔷")
    st.markdown(f"### {label_emoji} Rekommendation: **{label}**")
    st.markdown("**Motivering:** " + (reason or "—"))
    st.markdown("\n".join(lines))

    with st.expander("Visa fler detaljer"):
        extra = {}
        for k in [
            "Sektor", "CAGR 5 år (%)", "Bruttomarginal (%)", "Netto-marginal (%)",
            "Debt/Equity", "Free Cash Flow (M)", "Kassa (M)",
            "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
            "MCAP Q1 (M)", "MCAP Q2 (M)", "MCAP Q3 (M)", "MCAP Q4 (M)"
        ]:
            if k in rad.index:
                extra[k] = rad.get(k)
        if extra:
            df_extra = pd.DataFrame({"Nyckel": list(extra.keys()), "Värde": [extra[k] for k in extra.keys()]})
            st.dataframe(df_extra, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("**Toppkandidater (översikt):**")
    top = base.head(10).copy()
    out_cols = [
        "Ticker", "Bolagsnamn", "_Score", "Potential (%)", "Aktuell kurs",
        riktkurs_val, "P/S", "P/S-snitt", "Market Cap"
    ]
    show = [c for c in out_cols if c in top.columns]
    st.dataframe(
        top[show].rename(columns={"_Score": "Score"}).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
