# stockapp/invest.py
# -*- coding: utf-8 -*-
"""
Investeringsförslag – beräkning & vy.

Funktioner:
- compute_scores(df, focus="Balanserad") -> pd.DataFrame  (adderar/uppdaterar kolumnen 'Score')
- visa_investeringsforslag(df, user_rates=None, page_size=5) -> None  (renderar UI)

Principer:
- Sektorspecifik viktning: olika nyckeltal väger olika beroende på Sektor.
- Täckt-data-boost: färre tillgängliga nyckeltal => sänkt totalpoäng (coverage-justering).
- Värdering via P/S-upside: använder 'Omsättning i år (est.)' och 'P/S-snitt' (eller medel av Q1..Q4)
  för att beräkna 'implied' market cap och uppsida mot nuvarande 'Market Cap'.
- Robust mot saknade kolumner – allt guardas.
- Expander per bolag visar de centrala nyckeltalen.

OBS:
- Alla monetära värden antas ligga i bolagets basvaluta (kolumn 'Valuta').
- Vi gör inga valutaomräkningar här (visningen är informativ); portföljvyn hanterar SEK.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import math
import numpy as np
import pandas as pd
import streamlit as st

# Utils & config
from .utils import (
    safe_float,
    format_large_number,
    risk_label_from_mcap,
)

# ------------------------------------------------------------
# Hjälpfunktioner: läsa säkert från rad
# ------------------------------------------------------------
def _f(row, col, default=np.nan) -> float:
    return safe_float(row.get(col), default)

def _s(row, col, default="") -> str:
    v = row.get(col)
    return "" if v is None else str(v)

def _ps_avg_from_row(row: pd.Series) -> Optional[float]:
    # 1) använd 'P/S-snitt' om finns
    v = _f(row, "P/S-snitt", np.nan)
    if not math.isnan(v):
        return float(v)
    # 2) räkna medel av Q1..Q4
    qs = [_f(row, "P/S Q1", np.nan), _f(row, "P/S Q2", np.nan),
          _f(row, "P/S Q3", np.nan), _f(row, "P/S Q4", np.nan)]
    qs = [x for x in qs if not math.isnan(x)]
    if qs:
        return float(np.mean(qs))
    # 3) fallback: P/S (TTM)
    v2 = _f(row, "P/S", np.nan)
    if not math.isnan(v2):
        return float(v2)
    return None

def _implied_mcap(row: pd.Series) -> Optional[float]:
    """
    implied = (Omsättning i år (est.)) * (P/S-snitt)
    """
    sales = _f(row, "Omsättning i år (est.)", np.nan)
    if math.isnan(sales):
        return None
    ps_avg = _ps_avg_from_row(row)
    if ps_avg is None:
        return None
    return float(sales) * float(ps_avg)

def _ps_upside(row: pd.Series) -> Optional[float]:
    """
    Uppsida i % från P/S-snitt och estimerad omsättning.
    (implied / current_mcap - 1) * 100
    """
    implied = _implied_mcap(row)
    mcap = _f(row, "Market Cap", np.nan)
    if implied is None or math.isnan(mcap) or mcap <= 0:
        return None
    return (implied / mcap - 1.0) * 100.0


# ------------------------------------------------------------
# Normalisering av nyckeltal till 0..100 (högre = bättre)
# ------------------------------------------------------------
def _norm_higher_better(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None or math.isnan(x):
        return None
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 100.0)

def _norm_lower_better(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    """
    lo = bäst (lägst), hi = sämst (högst)
    """
    if x is None or math.isnan(x):
        return None
    return float(np.clip((hi - x) / (hi - lo), 0.0, 1.0) * 100.0)

def _score_ps_upside(row: pd.Series) -> Optional[float]:
    """
    Kartlägg uppsida -50% .. +100% till 0..100
    """
    u = _ps_upside(row)
    if u is None:
        return None
    # -50% => 0, 0% => ~50, +100% => 100 (klipp utanför)
    return float(np.clip((u + 50.0) / 150.0, 0.0, 1.0) * 100.0)

def _score_ev_ebitda(row: pd.Series) -> Optional[float]:
    # Lägre bättre, 4..25 mappas 100..0
    x = _f(row, "EV/EBITDA", np.nan)
    return _norm_lower_better(x, 4.0, 25.0)

def _score_dte(row: pd.Series) -> Optional[float]:
    # lägre bättre, 0..2
    x = _f(row, "Debt/Equity", np.nan)
    return _norm_lower_better(x, 0.0, 2.0)

def _score_gm(row: pd.Series) -> Optional[float]:
    x = _f(row, "Bruttomarginal (%)", np.nan)
    return _norm_higher_better(x, 20.0, 80.0)

def _score_nm(row: pd.Series) -> Optional[float]:
    x = _f(row, "Nettomarginal (%)", np.nan)
    return _norm_higher_better(x, 0.0, 40.0)

def _score_div_yield(row: pd.Series) -> Optional[float]:
    # högre bättre men plateau kring ~8%
    x = _f(row, "Dividend Yield (%)", np.nan)
    return _norm_higher_better(x, 0.0, 8.0)

def _score_payout_cf(row: pd.Series) -> Optional[float]:
    # 0..60% bäst => 100..; 100% => 50; 150% => 0
    x = _f(row, "Payout Ratio CF (%)", np.nan)
    if math.isnan(x):
        return None
    if x <= 60:
        return 100.0
    if x >= 150:
        return 0.0
    # 60..150 => 100..0 linjärt
    return float(np.clip((150.0 - x) / (150.0 - 60.0), 0.0, 1.0) * 100.0)

def _score_ps_ttm(row: pd.Series) -> Optional[float]:
    # lägre bättre, typ 2..30 mappas 100..0
    x = _f(row, "P/S", np.nan)
    return _norm_lower_better(x, 2.0, 30.0)

# ------------------------------------------------------------
# Viktning per sektor + fokus
# ------------------------------------------------------------
_BASE_WEIGHTS: Dict[str, float] = {
    "ps_upside": 3.0,
    "ps_ttm": 1.5,
    "ev_ebitda": 2.0,
    "dte": 1.5,
    "gm": 1.5,
    "nm": 1.5,
    "div_yield": 1.5,
    "payout_cf": 1.0,
}

# per sektor justering (multiplikatorer)
_SECTOR_MULTS: Dict[str, Dict[str, float]] = {
    "Technology":      {"ps_upside": 1.4, "gm": 1.3, "nm": 1.2, "ev_ebitda": 1.0, "div_yield": 0.7},
    "Industrials":     {"ev_ebitda": 1.3, "dte": 1.2, "ps_upside": 1.0},
    "Healthcare":      {"ps_upside": 1.2, "gm": 1.3, "nm": 1.2},
    "Energy":          {"ev_ebitda": 1.4, "dte": 1.2, "ps_ttm": 1.1},
    "Financials":      {"dte": 1.3, "nm": 1.2, "ev_ebitda": 1.1},
    "Consumer Staples":{"nm": 1.2, "gm": 1.2, "div_yield": 1.2, "ps_ttm": 1.1},
    "Consumer Discretionary":{"ps_upside": 1.2, "gm": 1.2},
    "Communication Services":{"ps_upside": 1.2, "nm": 1.2},
    "Utilities":       {"div_yield": 1.4, "payout_cf": 1.2, "dte": 1.2},
    "Real Estate":     {"div_yield": 1.4, "payout_cf": 1.3, "dte": 1.2, "ev_ebitda": 1.2},
    "Materials":       {"ev_ebitda": 1.3, "dte": 1.2},
    "Unknown":         {},
}

# fokus: Balanserad/Tillväxt/Utdelning (multiplikatorer ovanpå sektor)
_FOCUS_MULTS: Dict[str, Dict[str, float]] = {
    "Balanserad": {},
    "Tillväxt":   {"ps_upside": 1.3, "gm": 1.2, "nm": 1.1, "div_yield": 0.6, "payout_cf": 0.8},
    "Utdelning":  {"div_yield": 1.5, "payout_cf": 1.3, "ps_upside": 0.8, "ps_ttm": 0.9},
}

def _weights_for(sector: str, focus: str) -> Dict[str, float]:
    base = dict(_BASE_WEIGHTS)
    sec = _SECTOR_MULTS.get(sector or "Unknown", {})
    foc = _FOCUS_MULTS.get(focus or "Balanserad", {})
    for k, v in sec.items():
        base[k] = base.get(k, 0.0) * float(v)
    for k, v in foc.items():
        base[k] = base.get(k, 0.0) * float(v)
    return base

# ------------------------------------------------------------
# Poäng per rad
# ------------------------------------------------------------
def _row_score(row: pd.Series, focus: str = "Balanserad") -> Tuple[float, float]:
    """
    Returnerar (score, coverage) där score ∈ [0..100], coverage ∈ [0..1]
    """
    sector = _s(row, "Sektor", "Unknown")
    W = _weights_for(sector, focus)

    parts: List[Tuple[float, float]] = []  # (score_component, weight)

    # komponenter
    m = _score_ps_upside(row)
    if m is not None:
        parts.append((m, W["ps_upside"]))
    m = _score_ps_ttm(row)
    if m is not None:
        parts.append((m, W["ps_ttm"]))
    m = _score_ev_ebitda(row)
    if m is not None:
        parts.append((m, W["ev_ebitda"]))
    m = _score_dte(row)
    if m is not None:
        parts.append((m, W["dte"]))
    m = _score_gm(row)
    if m is not None:
        parts.append((m, W["gm"]))
    m = _score_nm(row)
    if m is not None:
        parts.append((m, W["nm"]))
    m = _score_div_yield(row)
    if m is not None:
        parts.append((m, W["div_yield"]))
    m = _score_payout_cf(row)
    if m is not None:
        parts.append((m, W["payout_cf"]))

    if not parts:
        return 0.0, 0.0

    num = sum(s * w for s, w in parts)
    den = sum(w for _, w in parts)
    base = num / den if den > 0 else 0.0

    # coverage = andel av totalvikt som var tillgänglig
    total_w = sum(_weights_for(sector, focus).values())
    avail_w = den
    coverage = float(np.clip(avail_w / total_w if total_w > 0 else 0.0, 0.0, 1.0))

    # Justera: fler datapunkter => högre score
    # skala 0.5..1.0 (min 50% av base om extremt låg täckning)
    adjusted = base * (0.5 + 0.5 * coverage)
    return float(np.clip(adjusted, 0.0, 100.0)), coverage


# ------------------------------------------------------------
# Publika API:n
# ------------------------------------------------------------
def compute_scores(df: pd.DataFrame, focus: str = "Balanserad") -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    scores = []
    covs = []
    for _, r in work.iterrows():
        s, c = _row_score(r, focus=focus)
        scores.append(s)
        covs.append(c)
    work["Score"] = scores
    work["_coverage"] = covs
    # Risklabel om saknas
    if "Risklabel" not in work.columns:
        mc = work.get("Market Cap", pd.Series([np.nan] * len(work)))
        work["Risklabel"] = [risk_label_from_mcap(safe_float(v, np.nan)) if not math.isnan(safe_float(v, np.nan)) else "Unknown" for v in mc]
    return work


def _grade_from_score(sc: float) -> str:
    if sc >= 85:
        return "✅ Mycket bra"
    if sc >= 70:
        return "👍 Bra"
    if sc >= 55:
        return "🙂 Okej"
    if sc >= 40:
        return "⚠️ Något övervärderad"
    return "🛑 Övervärderad / Sälj"


def visa_investeringsforslag(df: pd.DataFrame, user_rates: Optional[Dict[str, float]] = None, page_size: int = 5) -> None:
    st.header("📈 Investeringsförslag")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Fokus & filter
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    focus = c1.selectbox("Fokus", ["Balanserad", "Tillväxt", "Utdelning"])
    sektorer = ["Alla"]
    if "Sektor" in df.columns:
        sektorer += sorted([s for s in df["Sektor"].dropna().astype(str).unique() if s and s != "nan"])
    val_sektor = c2.selectbox("Sektor", sektorer)
    risk_opts = ["Alla", "Mega", "Large", "Mid", "Small", "Micro", "Unknown"]
    val_risk = c3.selectbox("Risklabel", risk_opts)
    page_size = int(c4.number_input("Poster per sida", min_value=1, max_value=20, value=int(page_size)))

    # Beräkna score (inkl coverage)
    work = compute_scores(df, focus=focus)

    # Filter
    if val_sektor != "Alla" and "Sektor" in work.columns:
        work = work[work["Sektor"].astype(str) == val_sektor]
    if val_risk != "Alla":
        work = work[work["Risklabel"].astype(str) == val_risk]

    # Sortera: Score desc, därefter högst coverage, därefter störst uppsida
    work["_ps_upside"] = work.apply(lambda r: _ps_upside(r), axis=1)
    work = work.sort_values(
        by=["Score", "_coverage", "_ps_upside"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    # Paging
    total = len(work)
    pages = max(1, math.ceil(total / page_size))
    if "page" not in st.session_state:
        st.session_state["page"] = 1
    st.session_state["page"] = max(1, min(st.session_state["page"], pages))

    colp1, colp2, colp3 = st.columns([1, 2, 1])
    if colp1.button("◀ Föregående", disabled=st.session_state["page"] <= 1):
        st.session_state["page"] -= 1
        st.rerun()
    colp2.markdown(f"<div style='text-align:center'>**{st.session_state['page']} / {pages}**</div>", unsafe_allow_html=True)
    if colp3.button("Nästa ▶", disabled=st.session_state["page"] >= pages):
        st.session_state["page"] += 1
        st.rerun()

    start = (st.session_state["page"] - 1) * page_size
    end = start + page_size
    page_df = work.iloc[start:end].reset_index(drop=True)

    # Rendera varje bolag
    for _, row in page_df.iterrows():
        with st.container(border=True):
            name = _s(row, "Bolagsnamn") or _s(row, "Ticker")
            tkr = _s(row, "Ticker")
            st.subheader(f"{name} ({tkr})")

            # Topptegel
            cA, cB, cC, cD = st.columns(4)
            cA.metric("Score", f"{float(row.get('Score', 0.0)):.1f}", help="Sektorspecifik poäng 0..100. Justerad för datatäckning.")
            mcap_disp = format_large_number(_f(row, "Market Cap", np.nan), _s(row, "Valuta", ""))
            cB.metric("Market Cap (nu)", mcap_disp)
            psavg = _ps_avg_from_row(row)
            cC.metric("P/S-snitt (4Q/TTM)", f"{psavg:.2f}" if psavg is not None else "–")
            up = _ps_upside(row)
            cD.metric("Uppsida (P/S)", f"{up:.1f}%" if up is not None else "–")

            # Betyg/etikett
            tag = _grade_from_score(float(row.get("Score", 0.0)))
            st.markdown(f"**Betyg:** {tag}")

            # Expander med nyckeltal
            with st.expander("Visa nyckeltal / historik"):
                bullets: List[Tuple[str, str]] = []

                # Värdering & intäkter
                bullets.append(("Valuta", _s(row, "Valuta", "–")))
                bullets.append(("Risklabel", _s(row, "Risklabel", "Unknown")))

                # Market cap & implied
                implied = _implied_mcap(row)
                if implied is not None:
                    bullets.append(("Implied MC (P/S*Sales est.)", format_large_number(implied, _s(row, "Valuta", ""))))
                bullets.append(("Market Cap (nu)", mcap_disp))

                # P/S detaljer
                for lab in ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
                    v = _f(row, lab, np.nan)
                    bullets.append((lab, f"{v:.2f}" if not math.isnan(v) else "–"))

                # Omsättning
                bullets.append(("Omsättning i år (est.)", f"{_f(row, 'Omsättning i år (est.)', np.nan):,.0f}" if not math.isnan(_f(row, "Omsättning i år (est.)", np.nan)) else "–"))

                # Lönsamhet & finansiellt
                bullets.append(("EV/EBITDA", f"{_f(row,'EV/EBITDA',np.nan):.2f}" if not math.isnan(_f(row,"EV/EBITDA",np.nan)) else "–"))
                bullets.append(("Debt/Equity", f"{_f(row,'Debt/Equity',np.nan):.2f}" if not math.isnan(_f(row,"Debt/Equity",np.nan)) else "–"))
                bullets.append(("Bruttomarginal (%)", f"{_f(row,'Bruttomarginal (%)',np.nan):.1f}%" if not math.isnan(_f(row,"Bruttomarginal (%)",np.nan)) else "–"))
                bullets.append(("Nettomarginal (%)", f"{_f(row,'Nettomarginal (%)',np.nan):.1f}%" if not math.isnan(_f(row,"Nettomarginal (%)",np.nan)) else "–"))

                # Utdelning
                bullets.append(("Dividend Yield (%)", f"{_f(row,'Dividend Yield (%)',np.nan):.2f}%" if not math.isnan(_f(row,"Dividend Yield (%)",np.nan)) else "–"))
                bullets.append(("Årlig utdelning", f"{_f(row,'Årlig utdelning',np.nan):.2f}" if not math.isnan(_f(row,"Årlig utdelning",np.nan)) else "–"))
                bullets.append(("Payout Ratio CF (%)", f"{_f(row,'Payout Ratio CF (%)',np.nan):.0f}%" if not math.isnan(_f(row,"Payout Ratio CF (%)",np.nan)) else "–"))

                # Kassaflöde & kassa
                bullets.append(("FCF (M)", f"{_f(row,'FCF (M)',np.nan):,.0f}" if not math.isnan(_f(row,"FCF (M)",np.nan)) else "–"))
                bullets.append(("Kassa (M)", f"{_f(row,'Kassa (M)',np.nan):,.0f}" if not math.isnan(_f(row,"Kassa (M)",np.nan)) else "–"))
                bullets.append(("Runway (kvartal)", f"{_f(row,'Runway (kvartal)',np.nan):.1f}" if not math.isnan(_f(row,"Runway (kvartal)",np.nan)) else "–"))

                # Struktur
                bullets.append(("Utestående aktier (milj.)", f"{_f(row,'Utestående aktier',np.nan)/1e6:,.2f}" if not math.isnan(_f(row,"Utestående aktier",np.nan)) else "–"))
                bullets.append(("Sektor", _s(row, "Sektor", "–")))
                bullets.append(("Industri", _s(row, "Industri", "–")))

                for k, v in bullets:
                    st.write(f"- **{k}:** {v}")p
