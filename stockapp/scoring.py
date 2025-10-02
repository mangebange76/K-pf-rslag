# -*- coding: utf-8 -*-
"""
stockapp.scoring

Sektors- och stilberoende poängsättning av bolag.
- Uppsida & riktkurs via P/S-snitt × prognos-omsättning (manuell, i miljoner).
- Normaliserar nyckeltal och väger dem olika per sektor och stil (growth/dividend).
- Returnerar TotalScore [0..100], Coverage [%], Recommendation samt Uppsida/Riktkurs.

Publika:
- compute_ps_target(row) -> dict
- score_row(row, style="growth") -> dict
- score_dataframe(df, style="growth") -> df
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import math

import numpy as np
import pandas as pd

# -------------------------------------------------------------
# Hjälpare
# -------------------------------------------------------------
def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _get_first(row: pd.Series, keys: List[str], default=None):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _norm_higher_better(x: float, lo: float, hi: float) -> float:
    x = _safe_float(x, lo)
    if hi <= lo:
        return 0.0
    return _clip01((x - lo) / (hi - lo))

def _norm_lower_better(x: float, lo: float, hi: float) -> float:
    x = _safe_float(x, hi)
    if hi <= lo:
        return 0.0
    return _clip01((hi - x) / (hi - lo))

def _norm_ratio_pos(x: float, hi: float) -> float:
    """Kvoter som gärna ska vara <= hi. x<=0 (nettokassa) => toppscore."""
    x = _safe_float(x, hi)
    if x <= 0:
        return 1.0
    return _norm_lower_better(x, 0.0, hi)

def _triangular_score(x: float, center: float, left: float, right: float) -> float:
    x = _safe_float(x, center)
    if x <= left or x >= right:
        return 0.0
    if x == center:
        return 1.0
    if x < center:
        return (x - left) / (center - left)
    return (right - x) / (right - center)

def _guess_is_millions(x: float) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return 0 < v < 1e9

# -------------------------------------------------------------
# Uppsida & riktkurs (P/S-baserad)
# -------------------------------------------------------------
def compute_ps_target(row: pd.Series) -> Dict[str, float]:
    """
    target_mcap = (P/S-snitt eller P/S) * prognos-omsättning (i bolagets valuta)
    - Prognos-omsättning antas manuellt inmatad i **miljoner** (din process).
    Returnerar ev. {"Uppsida (%)", "Riktkurs (valuta)", "_TargetMktCap"}.
    """
    out: Dict[str, float] = {}

    ps_avg = _safe_float(_get_first(row, ["P/S-snitt (Q1..Q4)", "P/S-snitt", "P/S"]), 0.0)
    if ps_avg <= 0:
        return out

    rev_now = _get_first(row, ["Omsättning i år (est.)", "Omsättning idag", "Revenue (current FY est.)"])
    rev_now = _safe_float(rev_now, 0.0)
    if rev_now <= 0:
        return out

    # Antag miljoner-inmatning:
    if _guess_is_millions(rev_now):
        rev_now *= 1e6

    mcap = _safe_float(_get_first(row, ["Market Cap", "MarketCap"]), 0.0)
    shares_mil = _safe_float(_get_first(row, ["Utestående aktier (milj.)", "Utestående aktier"]), 0.0)
    shares = shares_mil * 1e6 if shares_mil > 0 else 0.0

    target_mcap = ps_avg * rev_now
    out["_TargetMktCap"] = float(target_mcap)

    if target_mcap > 0 and mcap > 0:
        out["Uppsida (%)"] = round((target_mcap / mcap - 1.0) * 100.0, 2)
        if shares > 0:
            out["Riktkurs (valuta)"] = round(target_mcap / shares, 2)

    return out

# -------------------------------------------------------------
# Normalisering & vikter
# -------------------------------------------------------------
def _norm_ps_upside(ups: float) -> float:
    return _norm_higher_better(_safe_float(ups, -50.0), -50.0, 150.0)

BASE_WEIGHTS_GROWTH = {
    "UpsidePS": 2.0,
    "EV/EBITDA": 1.2,
    "Net debt / EBITDA": 1.0,
    "Bruttomarginal (%)": 1.0,
    "Rörelsemarginal (%)": 0.8,
    "Nettomarginal (%)": 0.8,
    "ROE (%)": 1.0,
    "P/B": 0.5,
    "FCF Yield (%)": 0.8,
    "Debt/Equity": 0.6,
    "Dividend Yield (%)": 0.2,
    "P/S": 0.4,
    "Payout Ratio CF (%)": 0.3,
}

BASE_WEIGHTS_DIVIDEND = {
    "UpsidePS": 0.6,
    "EV/EBITDA": 1.2,
    "Net debt / EBITDA": 1.4,
    "Bruttomarginal (%)": 0.6,
    "Rörelsemarginal (%)": 0.6,
    "Nettomarginal (%)": 0.8,
    "ROE (%)": 0.8,
    "P/B": 0.6,
    "FCF Yield (%)": 1.6,
    "Debt/Equity": 1.0,
    "Dividend Yield (%)": 2.0,
    "P/S": 0.3,
    "Payout Ratio CF (%)": 1.2,
}

# Multiplikatorer per sektor (engelska sektornamn är vanligast från API)
SECTOR_MULTIPLIERS = {
    "Technology": {"UpsidePS": 1.2, "Bruttomarginal (%)": 1.2, "ROE (%)": 1.2},
    "Communication Services": {"UpsidePS": 1.1, "Rörelsemarginal (%)": 1.1},
    "Consumer Cyclical": {"Rörelsemarginal (%)": 1.1, "ROE (%)": 1.1, "FCF Yield (%)": 1.1},
    "Consumer Defensive": {"Dividend Yield (%)": 1.3, "Payout Ratio CF (%)": 1.2, "Net debt / EBITDA": 1.1},
    "Healthcare": {"Bruttomarginal (%)": 1.2, "Nettomarginal (%)": 1.2},
    "Financial Services": {"ROE (%)": 1.3, "P/B": 1.2, "Net debt / EBITDA": 0.7},
    "Energy": {"EV/EBITDA": 1.3, "Net debt / EBITDA": 1.2, "FCF Yield (%)": 1.2},
    "Industrials": {"EV/EBITDA": 1.2, "ROE (%)": 1.1, "FCF Yield (%)": 1.1},
    "Basic Materials": {"EV/EBITDA": 1.2, "Net debt / EBITDA": 1.2},
    "Utilities": {"Dividend Yield (%)": 1.4, "Net debt / EBITDA": 1.2, "Debt/Equity": 1.2},
    "Real Estate": {"Dividend Yield (%)": 1.3, "P/B": 1.2, "Net debt / EBITDA": 1.2},
}

def _apply_sector_multipliers(weights: Dict[str, float], sector: str) -> Dict[str, float]:
    w = weights.copy()
    mult = SECTOR_MULTIPLIERS.get(sector or "", {})
    for k, m in mult.items():
        if k in w:
            w[k] *= float(m)
    return w

def _coverage_factor(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    needed = sum(1 for k in weights.keys())
    have = sum(1 for k in scores.keys() if k in weights and scores[k] is not None)
    if needed <= 0:
        return 1.0
    cov = have / float(needed)
    return 0.6 + 0.4 * cov  # 0.6..1.0

def _aggregate(scores: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    total = 0.0
    breakdown = {}
    for k, w in weights.items():
        s = scores.get(k, None)
        if s is None:
            continue
        part = float(w) * float(s)
        total += part
        breakdown[k] = round(float(s) * 100.0, 1)  # i %
    return total, breakdown

# -------------------------------------------------------------
# Poäng per rad
# -------------------------------------------------------------
def _metric_scores(row: pd.Series, style: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    base = BASE_WEIGHTS_DIVIDEND if str(style).lower().startswith("div") else BASE_WEIGHTS_GROWTH
    weights = base.copy()
    scores: Dict[str, float] = {}

    # Läsa nyckeltal (tål lite olika namn)
    ev_ebitda = _safe_float(_get_first(row, ["EV/EBITDA", "EV/EBITDA (ttm)"]))
    nd_ebitda = _safe_float(_get_first(row, ["Net debt / EBITDA", "Nettoskuld/EBITDA"]))
    gm = _safe_float(_get_first(row, ["Bruttomarginal (%)", "Gross margin (%)"]))
    om = _safe_float(_get_first(row, ["Rörelsemarginal (%)", "Operating margin (%)"]))
    nm = _safe_float(_get_first(row, ["Nettomarginal (%)", "Net margin (%)"]))
    roe = _safe_float(row.get("ROE (%)"))
    pb = _safe_float(_get_first(row, ["P/B", "PB"]))
    fcfy = _safe_float(_get_first(row, ["FCF Yield (%)"]))
    de = _safe_float(row.get("Debt/Equity"))
    dy = _safe_float(_get_first(row, ["Dividend Yield (%)", "Utdelningsyield (%)"]))
    ps = _safe_float(row.get("P/S"))
    payout = _safe_float(_get_first(row, ["Payout Ratio CF (%)", "Dividend payout (FCF) (%)"]))

    ups = _safe_float(row.get("Uppsida (%)"))
    if not row.get("Uppsida (%)", None):
        ups_calc = compute_ps_target(row).get("Uppsida (%)")
        if ups_calc is not None:
            ups = ups_calc

    # Normalisera
    if ev_ebitda > 0:
        scores["EV/EBITDA"] = _norm_lower_better(ev_ebitda, 3.0, 25.0)
    scores["Net debt / EBITDA"] = _norm_ratio_pos(nd_ebitda, 4.0)
    scores["Bruttomarginal (%)"] = _norm_higher_better(gm, 0.0, 80.0)
    scores["Rörelsemarginal (%)"] = _norm_higher_better(om, 0.0, 30.0)
    scores["Nettomarginal (%)"] = _norm_higher_better(nm, -10.0, 25.0)
    scores["ROE (%)"] = _norm_higher_better(roe, 0.0, 30.0)
    scores["P/B"] = _norm_lower_better(pb, 0.5, 6.0)
    scores["FCF Yield (%)"] = _norm_higher_better(fcfy, 0.0, 10.0)
    scores["Debt/Equity"] = _norm_lower_better(de, 0.0, 2.0)
    # dividend stil har högre tak men vi kapslar inte in det här – vikterna sköter preferensen
    scores["Dividend Yield (%)"] = _norm_higher_better(dy, 0.0, 12.0)
    scores["P/S"] = _norm_lower_better(ps, 1.0, 25.0)
    scores["Payout Ratio CF (%)"] = _triangular_score(payout, center=60.0, left=0.0, right=150.0)
    scores["UpsidePS"] = _norm_ps_upside(ups)

    return scores, weights

def _recommendation(row: pd.Series, total_score_0_100: float) -> str:
    ups = _safe_float(row.get("Uppsida (%)"))
    ev_eb = _safe_float(_get_first(row, ["EV/EBITDA", "EV/EBITDA (ttm)"]))
    nd_eb = _safe_float(_get_first(row, ["Net debt / EBITDA", "Nettoskuld/EBITDA"]))
    payout = _safe_float(_get_first(row, ["Payout Ratio CF (%)", "Dividend payout (FCF) (%)"]))

    # Riskflaggor
    if nd_eb > 4.0 or payout > 130.0 or ev_eb > 30.0:
        if ups < -10 or total_score_0_100 < 40:
            return "Sälj"
        return "Trimma"

    if ups >= 40 and total_score_0_100 >= 70:
        return "Köp"
    if ups >= 10 and total_score_0_100 >= 55:
        return "Behåll / Köp på dip"
    if -10 <= ups <= 10:
        return "Fair"
    if ups < -10 and total_score_0_100 < 55:
        return "Trimma"
    return "Behåll"

# -------------------------------------------------------------
# Publika API
# -------------------------------------------------------------
def score_row(row: pd.Series, style: str = "growth") -> Dict[str, object]:
    # Säkerställ uppsida/riktkurs i lokalt exemplar
    add = compute_ps_target(row)
    row_local = row.copy()
    for k, v in add.items():
        row_local[k] = v

    scores, weights = _metric_scores(row_local, style)

    # Sektormultiplikatorer – sektorn kan vara engelska (vanligast) eller svenska
    sector = _get_first(row_local, ["Sektor", "Sector"], "")
    weights2 = _apply_sector_multipliers(weights, str(sector))

    raw_sum, breakdown = _aggregate(scores, weights2)
    max_possible = sum(float(w) for w in weights2.values() if w is not None)
    base_pct = 0.0 if max_possible <= 0 else (raw_sum / max_possible) * 100.0

    cov = _coverage_factor(scores, weights2)
    total_pct = max(0.0, min(100.0, base_pct * cov))

    rec = _recommendation(row_local, total_pct)

    out = {
        "TotalScore": round(total_pct, 1),
        "Coverage": round(cov * 100.0, 1),
        "Recommendation": rec,
        "ScoreBreakdown": breakdown,
    }
    if "Uppsida (%)" in add:
        out["Uppsida (%)"] = add["Uppsida (%)"]
    if "Riktkurs (valuta)" in add:
        out["Riktkurs (valuta)"] = add["Riktkurs (valuta)"]
    if "_TargetMktCap" in add:
        out["_TargetMktCap"] = add["_TargetMktCap"]
    return out


def score_dataframe(df: pd.DataFrame, style: str = "growth") -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        rows.append(score_row(row, style=style))

    extra = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in extra.columns:
        out[col] = extra[col]

    if "TotalScore" in out.columns:
        out = out.sort_values(by="TotalScore", ascending=False, na_position="last")
    return out
