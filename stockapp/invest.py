# stockapp/invest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .rates import hamta_valutakurs
from .utils import safe_float, format_large_number
from .scoring import (
    growth_score,
    dividend_score,
    assign_action_label,
    risk_label_from_mcap,
)

# ---------------------------
# Hjälpare: datatäckning
# ---------------------------
def _is_present(val) -> bool:
    try:
        v = float(val)
        if np.isnan(v):
            return False
        return True
    except Exception:
        return str(val).strip() != ""

def _coverage_fields(mode: str) -> List[str]:
    if mode == "Utdelning":
        return [
            "Årlig utdelning",
            "Aktuell kurs",
            "Market Cap",
            "Free Cash Flow (M)",
            "Kassa (M)",
            "Debt/Equity",
            "Bruttomarginal (%)",
            "Netto-marginal (%)",
            "Utdelningskvot FCF (%)",
            "Utdelningskvot Vinst (%)",
        ]
    return [
        "P/S",
        "P/S-snitt",
        "CAGR 5 år (%)",
        "Market Cap",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "MCAP Q1 (M)", "MCAP Q2 (M)", "MCAP Q3 (M)", "MCAP Q4 (M)",
        "Bruttomarginal (%)",
        "Netto-marginal (%)",
        "Debt/Equity",
        "Free Cash Flow (M)",
        "Kassa (M)",
    ]

def _coverage_ratio_row(row: pd.Series, mode: str) -> Tuple[float, int, int]:
    fields = _coverage_fields(mode)
    present = 0
    for k in fields:
        if k in row.index and _is_present(row.get(k)):
            present += 1
    total = len(fields)
    ratio = (present / total) if total > 0 else 0.0
    return ratio, present, total

def _potential_pct(row: pd.Series, riktkurs_col: str) -> float:
    px = safe_float(row.get("Aktuell kurs"))
    tgt = safe_float(row.get(riktkurs_col))
    if px <= 0 or tgt <= 0:
        return 0.0
    return (tgt - px) / max(px, 1e-9) * 100.0

def _normalize_potential(pct: float) -> float:
    lo, hi = -50.0, 150.0
    x = max(lo, min(hi, pct))
    return (x - lo) / (hi - lo)

# ---------------------------
# Huvudvy
# ---------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    mode = st.radio("Fokus", ["Tillväxt", "Utdelning"], horizontal=True, index=0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas i potential-beräkningen?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    sector_vals = ["Alla"] + sorted([s for s in df.get("Sektor", pd.Series(dtype=str)).astype(str).unique() if s])
    sektor = st.selectbox("Filtrera på sektor", sector_vals, index=0)

    risk_vals = ["Alla", "Micro", "Small", "Mid", "Large", "Mega", "Unknown"]
    risk_choice = st.selectbox("Filtrera på risklabel (Market Cap)", risk_vals, index=0)

    # Basurval – kräver pris; riktkurs används om finns, annars fallback
    work = df.copy()
    work["Aktuell kurs"] = pd.to_numeric(work.get("Aktuell kurs", 0), errors="coerce").fillna(0.0)
    work[riktkurs_val] = pd.to_numeric(work.get(riktkurs_val, 0), errors="coerce").fillna(0.0)
    base = work[work["Aktuell kurs"] > 0].copy()

    if sektor != "Alla":
        base = base[base.get("Sektor", "").astype(str) == sektor]

    # Risklabel robust
    if "Market Cap" in base.columns:
        mcap_series = pd.to_numeric(base["Market Cap"], errors="coerce")
    else:
        mcap_series = pd.Series([np.nan] * len(base), index=base.index, dtype="float64")
    base["_RiskLabel"] = mcap_series.apply(risk_label_from_mcap)
    if risk_choice != "Alla":
        if risk_choice == "Unknown":
            base = base[base["_RiskLabel"] == "Unknown"]
        else:
            base = base[base["_RiskLabel"] == risk_choice]

    if base.empty:
        st.info("Inga bolag matchar filtren just nu.")
        return

    # Potential + täckning + score
    base["Potential (%)"] = base.apply(lambda r: _potential_pct(r, riktkurs_val), axis=1)
    base["_Coverage"], base["_Present"], base["_Total"] = zip(*base.apply(lambda r: _coverage_ratio_row(r, mode), axis=1))

    has_any_target = (base[riktkurs_val] > 0).any()

    if mode == "Utdelning":
        base["_BaseScore"] = base.apply(lambda r: dividend_score(r), axis=1)
    else:
        # Growth-score tar hänsyn till riktkurs om den finns
        base["_BaseScore"] = base.apply(lambda r: growth_score(r, riktkurs_col=riktkurs_val), axis=1)

    base["_FinalScore"] = (
        base["_BaseScore"] * (base["_Coverage"] ** 1.25)
        + 15.0 * base["_Coverage"]
        + 0.2 * base["Potential (%)"].apply(_normalize_potential) * 100.0
    )

    # Om riktkurser saknas rakt av (has_any_target == False), gör FALLBACK:
    if not has_any_target:
        # sortera på P/S-snitt om det finns, annars P/S
        psn = pd.to_numeric(base.get("P/S-snitt", 0), errors="coerce").fillna(0.0)
        ps0 = pd.to_numeric(base.get("P/S", 0), errors="coerce").fillna(0.0)
        base["_Sorterare"] = np.where(psn > 0, -psn, np.where(ps0 > 0, -ps0, 0.0))
        base = base.sort_values(by=["_Sorterare", "_Coverage"], ascending=[True, False]).reset_index(drop=True)
    else:
        base = base.sort_values(by=["_FinalScore", "_Coverage", "Potential (%)"], ascending=[False, False, False]).reset_index(drop=True)

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

    # Portföljandel & GAV SEK
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

    label, reason, metrics = assign_action_label(
        rad,
        mode=mode,
        riktkurs_col=riktkurs_val,
        pos_weight_pct=pos_weight,
        gav_sek=gav_sek,
        fx_to_sek=vx,
    )

    # Header
    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad.get('Ticker','')})")

    # Nyckeltal i kortform
    ps_now = safe_float(rad.get("P/S"))
    ps_avg = safe_float(rad.get("P/S-snitt"))
    mcap = safe_float(rad.get("Market Cap"))
    uts_m = safe_float(rad.get("Utestående aktier"))

    label_emoji = {"Köp": "🟢", "Håll": "🟡", "Trimma": "🟠", "Sälj": "🔴"}.get(label, "🔷")
    st.markdown(f"### {label_emoji} Rekommendation: **{label}**")
    st.markdown("**Motivering:** " + (reason or "—"))

    lines = [
        f"- **Aktuell kurs:** {safe_float(rad.get('Aktuell kurs')):.2f} {rad.get('Valuta','')}",
        f"- **{riktkurs_val}:** {safe_float(rad.get(riktkurs_val)):.2f} {rad.get('Valuta','')}",
        f"- **Uppsida:** {metrics.get('potential_pct', 0.0):.2f} %",
        f"- **P/S (nu):** {ps_now:.2f}  | **P/S-snitt:** {ps_avg:.2f}",
        f"- **Utestående aktier (M):** {uts_m:.2f}",
        f"- **Market Cap:** {format_large_number(mcap)}",
        f"- **Datatäckning:** {rad.get('_Coverage', 0.0)*100:.0f}%  ({int(rad.get('_Present',0))}/{int(rad.get('_Total',0))} nyckeltal)",
        f"- **Score:** {rad.get('_FinalScore', 0.0):.1f}",
    ]
    if mode == "Utdelning":
        yld = metrics.get("yield_pct", 0.0)
        lines.append(f"- **Direktavkastning:** {yld:.2f} %")

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
        "Ticker", "Bolagsnamn", "_FinalScore", "_Coverage", "Potential (%)",
        "Aktuell kurs", riktkurs_val, "P/S", "P/S-snitt", "Market Cap"
    ]
    show = [c for c in out_cols if c in top.columns]
    if "_Coverage" in show:
        top["_Coverage (%)"] = (top["_Coverage"] * 100.0).round(0)
        show = [c for c in show if c != "_Coverage"] + ["_Coverage (%)"]
    if "_FinalScore" in show:
        top.rename(columns={"_FinalScore": "Score"}, inplace=True)
    st.dataframe(
        top[show].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
