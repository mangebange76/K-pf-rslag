# -*- coding: utf-8 -*-
"""
stockapp/scoring.py

Sektors- och stilberoende poängsättning av bolag.
- Beräknar uppsida & riktkurs från P/S-snitt × prognos-omsättning.
- Normaliserar nyckeltal och väger dem olika per sektor och "stil"
  (growth vs dividend).
- Returnerar TotalScore [0..100], Rekommendation och ScoreBreakdown.

Publika funktioner:
- compute_ps_target(row) -> dict  (Uppsida (%), Riktkurs (valuta), Target MktCap)
- score_row(row, style="growth")  -> dict
- score_dataframe(df, style="growth") -> df med extra kolumner
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Hjälpare
# ---------------------------------------------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _pos(x: float) -> bool:
    try:
        return float(x) > 0
    except Exception:
        return False

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _triangular_score(x: float, center: float, left: float, right: float) -> float:
    """
    Triangel-formad scoring med topp=1 vid center.
    Faller linjärt till 0 vid 'left' respektive 'right'.
    """
    x = float(x)
    if x <= left or x >= right:
        return 0.0
    if x == center:
        return 1.0
    if x < center:
        return (x - left) / (center - left)
    return (right - x) / (right - center)

def _guess_monetary_unit_is_millions(x: float) -> bool:
    """
    Gissar om värdet är angivet i miljoner (t.ex. 206.45 för 206,45 miljarder -> 206450 M).
    Heuristik: < 1e9 => troligen miljoner.
    """
    try:
        v = float(x)
    except Exception:
        return False
    return v > 0 and v < 1e9

def _get_first(row: pd.Series, keys: List[str], default=None):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default

# ---------------------------------------------------------------------
# Uppsida & riktkurs (P/S-baserad)
# ---------------------------------------------------------------------

def compute_ps_target(row: pd.Series) -> Dict[str, float]:
    """
    Beräknar target market cap och riktkurs från:
      target_mcap = (P/S-snitt eller P/S) * prognos-omsättning (i basvaluta)
    Prognos-omsättning läses från några tänkbara kolumnnamn (SWE/ENG), i miljoner eller full valuta.
    Returnerar:
      {"Uppsida (%)": float, "Riktkurs (valuta)": float, "_TargetMktCap": float}
    Saknas byggstenar returneras tom dict.
    """
    out = {}

    ps_avg = _safe_float(_get_first(row, ["P/S-snitt", "P/S snitt", "PS-snitt", "PS Avg", "P/S"]), 0.0)
    if ps_avg <= 0:
        return out

    # Försök hitta prognos-omsättning (i bolagets valuta). Vi prioriterar "i år".
    rev_now = _get_first(row, [
        "Omsättning i år (förv.)",
        "Omsättning i år",
        "Revenue This Year (Est.)",
        "Revenue (Current FY Est.)",
        "Revenue (Forecast)"
    ])
    if rev_now is None:
        return out

    rev_now = _safe_float(rev_now, 0.0)
    if rev_now <= 0:
        return out

    # Gissa om det är "miljoner". Användaren sa att de matar in i miljoner i bolagets valuta.
    if _guess_monetary_unit_is_millions(rev_now):
        rev_now = rev_now * 1e6  # från "miljoner" till valuta-enheter

    mcap = _safe_float(row.get("Market Cap"), 0.0)
    shares_mil = _safe_float(row.get("Utestående aktier"), 0.0)  # i miljoner
    shares = shares_mil * 1e6 if shares_mil > 0 else 0.0

    target_mcap = ps_avg * rev_now
    if target_mcap <= 0 or not _pos(mcap):
        return {"_TargetMktCap": target_mcap}

    upside = (target_mcap / mcap - 1.0) * 100.0
    out["Uppsida (%)"] = round(float(upside), 2)
    out["_TargetMktCap"] = float(target_mcap)

    if _pos(shares):
        out["Riktkurs (valuta)"] = round(float(target_mcap / shares), 2)

    return out

# ---------------------------------------------------------------------
# Normalisering av nyckeltal → [0..1]
# ---------------------------------------------------------------------

def _norm_higher_better(x: float, lo: float, hi: float) -> float:
    """
    Skalar x till [0..1] där lo->0, hi->1. Klipper utanför.
    """
    x = _safe_float(x, lo)
    if hi <= lo:
        return 0.0
    return _clip01((x - lo) / (hi - lo))

def _norm_lower_better(x: float, lo: float, hi: float) -> float:
    """
    Låg siffra = bra. Returnerar 1 vid x<=lo och 0 vid x>=hi.
    """
    x = _safe_float(x, hi)
    if hi <= lo:
        return 0.0
    return _clip01((hi - x) / (hi - lo))

def _norm_ratio_pos(x: float, hi: float) -> float:
    """
    För kvoter som gärna ska vara <= hi. x<=0 (nettokassa) => toppbetyg 1.0.
    """
    x = _safe_float(x, hi)
    if x <= 0:
        return 1.0
    return _norm_lower_better(x, 0.0, hi)

def _norm_ps_upside(upside_pct: float) -> float:
    """
    Uppsida från P/S-snitt i %: mappa -50%..+150% → 0..1
    """
    u = _safe_float(upside_pct, -50.0)
    return _norm_higher_better(u, -50.0, 150.0)

# ---------------------------------------------------------------------
# Vikter per sektor + stil
# ---------------------------------------------------------------------

# Basvikter för metrics (innan sektormultiplikatorer)
BASE_WEIGHTS_GROWTH = {
    "UpsidePS": 2.0,
    "EV/EBITDA (ttm)": 1.2,
    "Net debt / EBITDA": 1.0,
    "Gross margin (%)": 1.0,
    "Operating margin (%)": 0.8,
    "Net margin (%)": 0.8,
    "ROE (%)": 1.0,
    "P/B": 0.5,
    "FCF Yield (%)": 0.8,
    "Debt/Equity": 0.6,
    "Dividend yield (%)": 0.2,
    "P/S": 0.4,
    "Dividend payout (FCF) (%)": 0.3,
}

BASE_WEIGHTS_DIVIDEND = {
    "UpsidePS": 0.6,
    "EV/EBITDA (ttm)": 1.2,
    "Net debt / EBITDA": 1.4,
    "Gross margin (%)": 0.6,
    "Operating margin (%)": 0.6,
    "Net margin (%)": 0.8,
    "ROE (%)": 0.8,
    "P/B": 0.6,
    "FCF Yield (%)": 1.6,
    "Debt/Equity": 1.0,
    "Dividend yield (%)": 2.0,
    "P/S": 0.3,
    "Dividend payout (FCF) (%)": 1.2,
}

# Sektormultiplikatorer (lättviktiga, 0.7..1.3)
SECTOR_MULTIPLIERS = {
    "Technology": {
        "UpsidePS": 1.2, "Gross margin (%)": 1.2, "ROE (%)": 1.2, "EV/EBITDA (ttm)": 1.0, "Net debt / EBITDA": 1.0,
    },
    "Communication Services": {
        "UpsidePS": 1.1, "Operating margin (%)": 1.1, "EV/EBITDA (ttm)": 1.1, "Net debt / EBITDA": 1.0
    },
    "Consumer Cyclical": {
        "Operating margin (%)": 1.1, "ROE (%)": 1.1, "FCF Yield (%)": 1.1
    },
    "Consumer Defensive": {
        "Dividend yield (%)": 1.3, "Dividend payout (FCF) (%)": 1.2, "Net debt / EBITDA": 1.1
    },
    "Healthcare": {
        "Gross margin (%)": 1.2, "Net margin (%)": 1.2, "EV/EBITDA (ttm)": 1.1
    },
    "Financial Services": {
        "ROE (%)": 1.3, "P/B": 1.2, "Net debt / EBITDA": 0.7  # mindre relevant
    },
    "Energy": {
        "EV/EBITDA (ttm)": 1.3, "Net debt / EBITDA": 1.2, "FCF Yield (%)": 1.2
    },
    "Industrials": {
        "EV/EBITDA (ttm)": 1.2, "ROE (%)": 1.1, "FCF Yield (%)": 1.1
    },
    "Basic Materials": {
        "EV/EBITDA (ttm)": 1.2, "Net debt / EBITDA": 1.2, "FCF Yield (%)": 1.1
    },
    "Utilities": {
        "Dividend yield (%)": 1.4, "Net debt / EBITDA": 1.2, "Debt/Equity": 1.2
    },
    "Real Estate": {
        "Dividend yield (%)": 1.3, "P/B": 1.2, "Net debt / EBITDA": 1.2
    },
}

# ---------------------------------------------------------------------
# Enskild rad → normaliserade delpoäng
# ---------------------------------------------------------------------

def _metric_scores(row: pd.Series, style: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returnerar (scores, weights) för alla metrics på 0..1 (innan sektorjustering & täckning).
    """
    # välj basvikter
    base = BASE_WEIGHTS_DIVIDEND if str(style).lower().startswith("div") else BASE_WEIGHTS_GROWTH
    weights = base.copy()

    # Samla källvärden
    ev_ebitda = _safe_float(row.get("EV/EBITDA (ttm)"))
    netdebt_ebitda = _safe_float(row.get("Net debt / EBITDA"))
    gm = _safe_float(row.get("Gross margin (%)"))
    om = _safe_float(row.get("Operating margin (%)"))
    nm = _safe_float(row.get("Net margin (%)"))
    roe = _safe_float(row.get("ROE (%)"))
    pb = _safe_float(row.get("P/B"))
    fcfy = _safe_float(row.get("FCF Yield (%)"))
    de = _safe_float(row.get("Debt/Equity"))
    dy = _safe_float(row.get("Dividend yield (%)"))
    ps = _safe_float(row.get("P/S"))
    payout_fcf = _safe_float(row.get("Dividend payout (FCF) (%)"))

    # Uppsida via PS-target (räkna on-the-fly om saknas)
    ups = _safe_float(row.get("Uppsida (%)"))
    if not row.get("Uppsida (%)", None):
        ups_calc = compute_ps_target(row).get("Uppsida (%)")
        if ups_calc is not None:
            ups = ups_calc

    scores: Dict[str, float] = {}

    # EV/EBITDA lägre bättre: 3..25
    if ev_ebitda > 0:
        scores["EV/EBITDA (ttm)"] = _norm_lower_better(ev_ebitda, 3.0, 25.0)

    # Net debt / EBITDA: 0..4 (<=0 topp)
    if netdebt_ebitda or netdebt_ebitda == 0:
        scores["Net debt / EBITDA"] = _norm_ratio_pos(netdebt_ebitda, 4.0)

    # Margins
    if gm or gm == 0:
        scores["Gross margin (%)"] = _norm_higher_better(gm, 0.0, 80.0)
    if om or om == 0:
        scores["Operating margin (%)"] = _norm_higher_better(om, 0.0, 30.0)
    if nm or nm == 0:
        scores["Net margin (%)"] = _norm_higher_better(nm, -10.0, 25.0)

    # ROE: 0..30
    if roe or roe == 0:
        scores["ROE (%)"] = _norm_higher_better(roe, 0.0, 30.0)

    # P/B lägre bättre: 0.5..6
    if pb or pb == 0:
        scores["P/B"] = _norm_lower_better(pb, 0.5, 6.0)

    # FCF Yield: 0..10% (högre än 10% klipps)
    if fcfy or fcfy == 0:
        scores["FCF Yield (%)"] = _norm_higher_better(fcfy, 0.0, 10.0)

    # Debt/Equity lägre bättre: 0..2
    if de or de == 0:
        scores["Debt/Equity"] = _norm_lower_better(de, 0.0, 2.0)

    # Dividend yield: 0..8% (Dividend-stil: upp till 12%)
    if dy or dy == 0:
        hi = 12.0 if weights is BASE_WEIGHTS_DIVIDEND else 8.0
        scores["Dividend yield (%)"] = _norm_higher_better(dy, 0.0, hi)

    # P/S lägre bättre: 1..25
    if ps or ps == 0:
        scores["P/S"] = _norm_lower_better(ps, 1.0, 25.0)

    # Payout (FCF): triangel med optimum ~60%, 0 vid 0% och 150%
    if payout_fcf or payout_fcf == 0:
        scores["Dividend payout (FCF) (%)"] = _triangular_score(payout_fcf, center=60.0, left=0.0, right=150.0)

    # Uppsida (PS)
    if ups or ups == 0:
        scores["UpsidePS"] = _norm_ps_upside(ups)

    return scores, weights


def _apply_sector_multipliers(weights: Dict[str, float], sector: str) -> Dict[str, float]:
    w = weights.copy()
    mult = SECTOR_MULTIPLIERS.get(str(sector) or "", {})
    for k, m in mult.items():
        if k in w:
            w[k] *= float(m)
    return w


def _coverage_factor(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Gynnar bolag där fler metrics finns.
    60% bas + 40% * (täckning).
    """
    needed = sum(1 for k in weights.keys())
    have = sum(1 for k in scores.keys() if k in weights and scores[k] is not None)
    if needed <= 0:
        return 1.0
    cov = have / float(needed)
    return 0.6 + 0.4 * cov  # 0.6..1.0


def _aggregate(scores: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Summerar viktade poäng; returnerar (sum, breakdown_per_metric).
    """
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

# ---------------------------------------------------------------------
# Rekommendationer
# ---------------------------------------------------------------------

def _recommendation(row: pd.Series, total_score_0_100: float) -> str:
    """
    Enkel regelbaserat label, kombinerar uppsida & risk.
    """
    ups = _safe_float(row.get("Uppsida (%)"))
    ev_eb = _safe_float(row.get("EV/EBITDA (ttm)"))
    nd_eb = _safe_float(row.get("Net debt / EBITDA"))
    payout = _safe_float(row.get("Dividend payout (FCF) (%)"))

    # Hårda risk-flaggor
    if nd_eb > 4.0 or payout > 130.0 or ev_eb > 30.0:
        if ups < -10 or total_score_0_100 < 40:
            return "Sälj"
        return "Trimma"

    # Primärt på uppsida och total score
    if ups >= 40 and total_score_0_100 >= 70:
        return "Köp"
    if ups >= 10 and total_score_0_100 >= 55:
        return "Behåll / Köp på dip"
    if -10 <= ups <= 10:
        return "Fair"
    if ups < -10 and total_score_0_100 < 55:
        return "Trimma"
    return "Behåll"

# ---------------------------------------------------------------------
# Publika API
# ---------------------------------------------------------------------

def score_row(row: pd.Series, style: str = "growth") -> Dict[str, object]:
    """
    Poängsätter en enskild rad.
    Returnerar dict med:
      - TotalScore (0..100)
      - Coverage (0..100)
      - Recommendation (text)
      - Uppsida (%) / Riktkurs (valuta) (om beräkningsbart)
      - ScoreBreakdown (dict metric -> procent)
    """
    # Säkerställ uppsida/riktkurs i row (utan att mutera originalet)
    add = compute_ps_target(row)
    row_local = row.copy()
    for k, v in add.items():
        row_local[k] = v

    # Delpoäng + basvikter
    scores, weights = _metric_scores(row_local, style)

    # Sektor-multiplikatorer
    sector = str(row_local.get("Sector", "") or "")
    weights2 = _apply_sector_multipliers(weights, sector)

    # Aggregat
    raw_sum, breakdown = _aggregate(scores, weights2)

    # Skala råsumma till 0..100 via relativ normalisering:
    # Vi antar maxviktade poäng som sum(weights2.values()).
    max_possible = sum(float(w) for w in weights2.values() if w is not None)
    base_pct = 0.0 if max_possible <= 0 else (raw_sum / max_possible) * 100.0

    # Täckningsfaktor (gynnar bolag med fler datapunkter)
    cov = _coverage_factor(scores, weights2)
    total_pct = base_pct * cov
    total_pct = max(0.0, min(100.0, total_pct))

    # Rekommendation
    rec = _recommendation(row_local, total_pct)

    out = {
        "TotalScore": round(total_pct, 1),
        "Coverage": round(cov * 100.0, 1),
        "Recommendation": rec,
        "ScoreBreakdown": breakdown,
    }
    # Lägg med uppsida/riktkurs om de finns
    if "Uppsida (%)" in add:
        out["Uppsida (%)"] = add["Uppsida (%)"]
    if "Riktkurs (valuta)" in add:
        out["Riktkurs (valuta)"] = add["Riktkurs (valuta)"]
    if "_TargetMktCap" in add:
        out["_TargetMktCap"] = add["_TargetMktCap"]

    return out


def score_dataframe(df: pd.DataFrame, style: str = "growth") -> pd.DataFrame:
    """
    Poängsätter hela dataframe och lägger till kolumner:
      ["TotalScore", "Coverage", "Recommendation", "Uppsida (%)", "Riktkurs (valuta)"]
    Modifierar INTE inkommande df (returnerar kopia).
    """
    if df is None or df.empty:
        return df

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        res = score_row(row, style=style)
        rows.append(res)

    extra = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for col in extra.columns:
        out[col] = extra[col]

    # Sortera default på TotalScore desc
    if "TotalScore" in out.columns:
        out = out.sort_values(by="TotalScore", ascending=False, na_position="last")
    return out
