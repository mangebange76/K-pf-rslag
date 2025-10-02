# stockapp/invest.py
# -*- coding: utf-8 -*-
"""
Investeringsförslag – sektors-/stilmedveten scoring + bläddringsvy.

Publikt API:
    visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st

# Utils & konfig
from .utils import (
    safe_float,
    format_large_number,
    risk_label_from_mcap,
)
from .rates import hamta_valutakurs

# ---------------------------------------
# Kolumnalias / helpers
# ---------------------------------------
PRICE_COLS = ["Kurs", "Aktuell kurs"]
MCAP_COLS  = ["Market Cap (SEK)", "Market Cap (valuta)", "Market Cap"]
PSQ_COLS   = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
PS_AVG_COL = "P/S-snitt"              # om saknas beräknas fallback "P/S-snitt (Q1..Q4)"
REV_NOW    = "Omsättning i år (est.)"
REV_NEXT   = "Omsättning nästa år (est.)"
SHARES_COL = "Utestående aktier"      # för riktkurs = impl_mcap / shares
CURRENCY   = "Valuta"

# Nyckeltal som väger in "coverage"
COVERAGE_CANDIDATES = [
    "Debt/Equity", "Bruttomarginal (%)", "Nettomarginal (%)", "FCF (M)",
    "Runway (kvartal)", "EV/EBITDA", "Dividend Yield (%)", "Payout Ratio CF (%)",
    "P/S", "P/S-snitt", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"
]

# ---------------------------------------
# Viktning – sektorsvis och per stil
# ---------------------------------------
def _sector_base_weights(sektor: str) -> Dict[str, float]:
    """
    Basvikter per sektor för delbetyg:
      - value: värdering (riktkurs/uppsida)
      - margins: bruttomarginal/nettomarginal
      - debt: skuld (lägre = bättre)
      - efficiency: FCF, EV/EBITDA, runway
      - dividend: yield & payout (hållbarhet)
    Summerar till ~1.0 (normaliseras senare).
    """
    s = (sektor or "").lower()
    # default
    w = dict(value=0.40, margins=0.20, debt=0.10, efficiency=0.20, dividend=0.10)

    if "information" in s or "tech" in s or "it" in s:
        w = dict(value=0.45, margins=0.25, debt=0.05, efficiency=0.20, dividend=0.05)
    elif "health" in s or "sjukvård" in s:
        w = dict(value=0.40, margins=0.25, debt=0.10, efficiency=0.15, dividend=0.10)
    elif "financial" in s or "finans" in s:
        w = dict(value=0.35, margins=0.15, debt=0.20, efficiency=0.15, dividend=0.15)
    elif "energy" in s or "energi" in s:
        w = dict(value=0.35, margins=0.15, debt=0.15, efficiency=0.20, dividend=0.15)
    elif "utilities" in s or "kraft" in s:
        w = dict(value=0.30, margins=0.15, debt=0.20, efficiency=0.15, dividend=0.20)
    elif "consumer staples" in s or "dagligvaror" in s:
        w = dict(value=0.35, margins=0.20, debt=0.15, efficiency=0.15, dividend=0.15)
    elif "consumer discretionary" in s or "konsument" in s:
        w = dict(value=0.40, margins=0.20, debt=0.10, efficiency=0.20, dividend=0.10)
    elif "industrial" in s or "industri" in s:
        w = dict(value=0.40, margins=0.20, debt=0.10, efficiency=0.20, dividend=0.10)
    elif "real estate" in s or "fastighet" in s:
        w = dict(value=0.30, margins=0.10, debt=0.30, efficiency=0.15, dividend=0.15)
    elif "communication" in s or "kommunikation" in s:
        w = dict(value=0.35, margins=0.20, debt=0.15, efficiency=0.20, dividend=0.10)

    # normalisera:
    tot = sum(w.values())
    if tot <= 0:
        return dict(value=0.4, margins=0.2, debt=0.1, efficiency=0.2, dividend=0.1)
    return {k: v / tot for k, v in w.items()}

def _style_adjust(weights: Dict[str, float], style: str) -> Dict[str, float]:
    """Justera vikter för Growth/Dividend-stil."""
    w = dict(weights)
    sty = (style or "").lower()
    if "growth" in sty or "tillväxt" in sty:
        # mer värdering/marginaler/effektivitet
        w["value"] *= 1.15
        w["margins"] *= 1.10
        w["efficiency"] *= 1.10
        w["dividend"] *= 0.70
        w["debt"] *= 0.90
    elif "dividend" in sty or "utdel" in sty:
        # mer utdelning/skuld, mindre ren värderingsuppsida
        w["dividend"] *= 1.40
        w["debt"] *= 1.10
        w["value"] *= 0.80
    # normalisera igen
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else weights

# ---------------------------------------
# Scoring helpers
# ---------------------------------------
def _pick_first(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None

def _ps_avg(row: pd.Series) -> float:
    # använd given P/S-snitt annars medel av Q1..Q4
    v = safe_float(row.get("P/S-snitt"), np.nan)
    if not math.isnan(v):
        return v
    vals = [safe_float(row.get(c), np.nan) for c in PSQ_COLS]
    arr = [x for x in vals if not math.isnan(x)]
    return float(np.mean(arr)) if arr else np.nan

def _coverage_factor(row: pd.Series) -> float:
    # 0.5–1.0 beroende på hur många nyckeltal som faktiskt finns
    present = 0
    possible = 0
    for c in COVERAGE_CANDIDATES:
        if c in row.index:
            possible += 1
            v = row.get(c)
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                present += 1
            elif isinstance(v, str) and v.strip() != "":
                present += 1
    if possible == 0:
        return 0.6
    ratio = present / possible
    return 0.5 + 0.5 * ratio

def _clip01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))

def _z_score(val: float, lo: float, hi: float, invert: bool=False) -> float:
    """Skala till 0..1 inom [lo, hi]; invert=True vänder (lägre är bättre)."""
    if math.isnan(val):
        return 0.0
    if hi == lo:
        return 0.5
    pos = (val - lo) / (hi - lo)
    pos = 1.0 - pos if invert else pos
    return _clip01(pos)

def _score_row(row: pd.Series, style: str) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Returnerar total_score, delbetyg, diagnoser (uppsida %, riktkurs, etc.)
    """
    sektor = str(row.get("Sektor", "") or "")
    base = _sector_base_weights(sektor)
    weights = _style_adjust(base, style)

    # 1) Value – uppsida från riktkurs vs kurs
    price = safe_float(row.get(_pick_first(pd.DataFrame([row]), PRICE_COLS) or PRICE_COLS[0]), np.nan)
    shares = safe_float(row.get(SHARES_COL), np.nan)
    ps_avg = _ps_avg(row)
    rev = safe_float(row.get(REV_NOW), np.nan)

    implied_mcap = np.nan
    target_price = np.nan
    upside = np.nan
    if not math.isnan(ps_avg) and not math.isnan(rev) and not math.isnan(shares) and shares > 0:
        implied_mcap = ps_avg * rev  # i bolagets valuta
        target_price = implied_mcap / shares
        if not math.isnan(price) and price > 0:
            upside = (target_price / price) - 1.0

    # Skala uppsida: -50%..+100% -> 0..1
    s_value = 0.0 if math.isnan(upside) else _z_score(upside, -0.5, 1.0, invert=False)

    # 2) Margins – bruttomarginal, nettomarginal (0..100)
    gm = safe_float(row.get("Bruttomarginal (%)"), np.nan)
    nm = safe_float(row.get("Nettomarginal (%)"), np.nan)
    s_margins = 0.5 * _z_score(gm, 0.0, 70.0) + 0.5 * _z_score(nm, -20.0, 40.0)

    # 3) Debt – Debt/Equity lägre är bättre
    de = safe_float(row.get("Debt/Equity"), np.nan)
    s_debt = _z_score(de, 0.0, 2.0, invert=True)

    # 4) Efficiency – FCF, EV/EBITDA, runway
    fcf = safe_float(row.get("FCF (M)"), np.nan)
    ev_ebitda = safe_float(row.get("EV/EBITDA"), np.nan)
    runway = safe_float(row.get("Runway (kvartal)"), np.nan)
    # FCF: positivt bättre, klipp från -500..+500 M
    s_fcf = _z_score(fcf, -500.0, 500.0)
    # EV/EBITDA: 5..25 -> 1..0 (lägre bättre)
    s_ev = _z_score(ev_ebitda, 5.0, 25.0, invert=True)
    # runway: 0..16+ kvartal
    s_run = _z_score(runway, 0.0, 16.0)
    s_eff = 0.4 * s_fcf + 0.4 * s_ev + 0.2 * s_run

    # 5) Dividend – yield högre bättre, payout (CF) lägre bättre (t.ex. 0.3~0.7 optimalt)
    dy = safe_float(row.get("Dividend Yield (%)"), np.nan)
    pr_cf = safe_float(row.get("Payout Ratio CF (%)"), np.nan)
    s_yield = _z_score(dy, 0.0, 8.0)  # 0..8%+
    # payout bästa kring 40–60%; mappa som "klockform": närmast 50% ger 1.0
    if math.isnan(pr_cf):
        s_payout = 0.0
    else:
        pr = pr_cf / 100.0
        s_payout = max(0.0, 1.0 - abs(pr - 0.5) / 0.5)  # 0 vid 0%/100%, 1 vid 50%
    s_div = 0.6 * s_yield + 0.4 * s_payout

    # Coverage-penalty
    cov = _coverage_factor(row)

    # Total
    parts = {
        "value": s_value,
        "margins": s_margins,
        "debt": s_debt,
        "efficiency": s_eff,
        "dividend": s_div,
    }
    total = cov * sum(parts[k] * weights.get(k, 0.0) for k in parts.keys())

    diags = {
        "Implied MCAP": implied_mcap,
        "Target Price": target_price,
        "Upside %": (upside * 100.0) if not math.isnan(upside) else np.nan,
        "Coverage": cov * 100.0,
    }
    return float(total * 100.0), parts, diags  # skala till 0..100


def _label_from_score(sc: float) -> str:
    if sc >= 85:
        return "✅ Mycket bra"
    if sc >= 70:
        return "👍 Bra"
    if sc >= 55:
        return "🙂 Okej"
    if sc >= 40:
        return "⚠️ Något övervärderad"
    return "🛑 Övervärderad / Sälj"


# ---------------------------------------
# Huvudvy
# ---------------------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📈 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    work = df.copy()

    # Robust pris/mcap-kolumner
    price_col = _pick_first(work, PRICE_COLS) or PRICE_COLS[0]
    mcap_col  = _pick_first(work, MCAP_COLS) or MCAP_COLS[-1]

    # Sätt P/S-snitt (fallback) om saknas
    if "P/S-snitt" not in work.columns:
        for c in PSQ_COLS:
            if c not in work.columns:
                work[c] = np.nan
        work["P/S-snitt"] = pd.to_numeric(work[PSQ_COLS].mean(axis=1), errors="coerce")

    # Lägg Risklabel om saknas
    if "Risklabel" not in work.columns:
        work["Risklabel"] = work[mcap_col].apply(risk_label_from_mcap) if mcap_col in work.columns else "Unknown"

    # Filterrad
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    st_style = c1.radio("Köpstil", ["Tillväxt", "Utdelning"], horizontal=True, index=0)
    sektorer = ["Alla"]
    if "Sektor" in work.columns:
        sektorer += sorted([s for s in work["Sektor"].dropna().astype(str).unique() if s and s != "nan"])
    val_sektor = c2.selectbox("Sektor", sektorer)
    risk_opts = ["Alla", "Mega", "Large", "Mid", "Small", "Micro", "Unknown"]
    val_risk = c3.selectbox("Risklabel", risk_opts)
    page_size = c4.number_input("Poster per sida", 1, 20, value=int(st.session_state.get("page_size", 5)))
    st.session_state["page_size"] = int(page_size)

    # Filtrera
    if val_sektor != "Alla" and "Sektor" in work.columns:
        work = work[work["Sektor"].astype(str) == val_sektor]
    if val_risk != "Alla":
        work = work[work["Risklabel"].astype(str) == val_risk]

    # Beräkna score per rad
    scores = []
    parts_list = []
    diags_list = []
    for _, row in work.iterrows():
        sc, parts, diags = _score_row(row, st_style)
        scores.append(sc)
        parts_list.append(parts)
        diags_list.append(diags)
    work = work.assign(Score=scores, _parts=parts_list, _diags=diags_list)

    # Sortera fallande på Score
    work = work.sort_values(by="Score", ascending=False, na_position="last")

    # Paginering 1/X
    total = len(work)
    if total == 0:
        st.info("Inga träffar efter filter.")
        return
    pages = max(1, math.ceil(total / st.session_state["page_size"]))
    st.session_state["page"] = max(1, min(st.session_state.get("page", 1), pages))

    colp1, colp2, colp3 = st.columns([1, 2, 1])
    if colp1.button("◀ Föregående", disabled=st.session_state["page"] <= 1, key="inv_prev"):
        st.session_state["page"] -= 1
        st.rerun()
    colp2.markdown(f"<div style='text-align:center'>**{st.session_state['page']} / {pages}**</div>", unsafe_allow_html=True)
    if colp3.button("Nästa ▶", disabled=st.session_state["page"] >= pages, key="inv_next"):
        st.session_state["page"] += 1
        st.rerun()

    start = (st.session_state["page"] - 1) * st.session_state["page_size"]
    end = start + st.session_state["page_size"]
    page_df = work.iloc[start:end].reset_index(drop=True)

    # Rendera kort
    for _, row in page_df.iterrows():
        with st.container(border=True):
            namn = str(row.get("Bolagsnamn", "") or "")
            tkr  = str(row.get("Ticker", "") or "")
            sektor = str(row.get("Sektor", "") or "")
            risk = str(row.get("Risklabel", "") or "Unknown")
            st.subheader(f"{namn} ({tkr})")

            # head-metrics
            cA, cB, cC, cD = st.columns(4)
            # P/S (TTM) – om finns
            ps_ttm = safe_float(row.get("P/S"), np.nan)
            cA.metric("P/S (TTM)", f"{ps_ttm:.2f}" if not math.isnan(ps_ttm) else "–")
            # P/S-snitt
            ps_avg = _ps_avg(row)
            cB.metric("P/S-snitt (4Q)", f"{ps_avg:.2f}" if not math.isnan(ps_avg) else "–")
            # MCAP
            mcap = safe_float(row.get(_pick_first(pd.DataFrame([row]), MCAP_COLS) or MCAP_COLS[-1]), np.nan)
            cur = str(row.get(CURRENCY, "USD")).upper()
            cB2 = format_large_number(mcap, cur) if not math.isnan(mcap) else "–"
            cC.metric("Market Cap (nu)", cB2)
            cD.write(f"**Sektor:** {sektor}  \n**Risklabel:** {risk}")

            # Betyg
            sc = safe_float(row.get("Score"), np.nan)
            tag = _label_from_score(sc) if not math.isnan(sc) else "–"
            st.markdown(f"**Betyg:** {sc:.1f} – {tag}" if not math.isnan(sc) else "**Betyg:** –")

            # Expander med detaljer/nyckeltal & diagnoser
            with st.expander("Visa nyckeltal / beräkningar"):
                diag = row.get("_diags", {}) or {}
                parts = row.get("_parts", {}) or {}

                # Visa diag (riktkurs, uppsida, coverage)
                tp = safe_float(diag.get("Target Price"), np.nan)
                up = safe_float(diag.get("Upside %"), np.nan)
                cov = safe_float(diag.get("Coverage"), np.nan)
                st.write("**Riktkurs/uppsida**")
                st.write(f"- Riktkurs (i bolagets valuta): {tp:.2f}" if not math.isnan(tp) else "- Riktkurs: –")
                st.write(f"- Uppsida: {up:.1f} %" if not math.isnan(up) else "- Uppsida: –")
                st.write(f"- Täckning (coverage): {cov:.0f} %" if not math.isnan(cov) else "- Täckning: –")

                # Poängdelar
                st.write("**Delbetyg (0–1)**")
                st.write(f"- Value: {parts.get('value', 0):.2f}")
                st.write(f"- Margins: {parts.get('margins', 0):.2f}")
                st.write(f"- Debt: {parts.get('debt', 0):.2f}")
                st.write(f"- Efficiency: {parts.get('efficiency', 0):.2f}")
                st.write(f"- Dividend: {parts.get('dividend', 0):.2f}")

                # Nyckeltal
                st.write("**Nyckeltal**")
                # P/S-kvartar
                ps_line = []
                for c in PSQ_COLS:
                    v = row.get(c)
                    if isinstance(v, (int, float)) and not math.isnan(float(v)):
                        ps_line.append(f"{c}: {float(v):.2f}")
                st.write("- " + (", ".join(ps_line) if ps_line else "P/S Q1–Q4: –"))

                # Marginaler, skuld, FCF, EV/EBITDA, runway, utdelning
                kvs = [
                    ("Bruttomarginal (%)", row.get("Bruttomarginal (%)")),
                    ("Nettomarginal (%)", row.get("Nettomarginal (%)")),
                    ("Debt/Equity", row.get("Debt/Equity")),
                    ("FCF (M)", row.get("FCF (M)")),
                    ("EV/EBITDA", row.get("EV/EBITDA")),
                    ("Runway (kvartal)", row.get("Runway (kvartal)")),
                    ("Dividend Yield (%)", row.get("Dividend Yield (%)")),
                    ("Payout Ratio CF (%)", row.get("Payout Ratio CF (%)")),
                    ("Utestående aktier", row.get(SHARES_COL)),
                ]
                for k, v in kvs:
                    if isinstance(v, (int, float)) and not math.isnan(float(v)):
                        st.write(f"- **{k}:** {float(v):.2f}")
                    elif isinstance(v, str) and v.strip():
                        st.write(f"- **{k}:** {v}")
                    else:
                        st.write(f"- **{k}:** –")
