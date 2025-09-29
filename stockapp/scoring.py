# stockapp/scoring.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# ============================================================
# Grundkonstanter (trösklar och vikter)
# ============================================================

# Risketikett efter market cap (i samma valuta som pris).
# Vi räknar MCAP ≈ Aktuell kurs * Utestående aktier (miljoner) * 1e6
MCAP_BUCKETS = [
    (2.0e10, "Mega"),   # ≥ 20e9
    (1.0e10, "Large"),  # ≥ 10e9
    (2.0e9,  "Mid"),    # ≥ 2e9
    (3.0e8,  "Small"),  # ≥ 300e6
    (0.0,    "Micro"),  # < 300e6
]

# Valuation label – gränser baserade på kombination av uppsida (%) och fundamentalscore [0..100]
# Du kan tweaka vid behov.
VAL_LABEL_RULES = [
    # (min_score, min_upside_pct, label)
    (80,  25, "Mycket bra (Köp)"),
    (65,  10, "Bra (Köp)"),
    (50,  -5, "Fair / Behåll"),
    (35, -10, "Trimma"),
    (0,  -20, "Sälj / Övervärderad"),
]

# Normaliseringsmål (rimliga intervall för nyckeltal):
NORMS = {
    "ps": (1.0, 20.0),                # lägre är bättre; 1→bäst, 20→sämst
    "cagr": (0.0, 40.0),              # högre är bättre; 0..40%
    "gross_margin": (20.0, 70.0),     # högre är bättre
    "net_margin": (0.0, 30.0),        # högre är bättre
    "debt_to_equity": (0.0, 200.0),   # lägre är bättre
    "ev_ebitda": (5.0, 30.0),         # lägre är bättre
    "fcf_margin": (0.0, 25.0),        # högre är bättre
    "div_yield": (1.5, 8.0),          # högre är bättre (utdelningsläge)
    # payout FCF (%) ideal ~ 40..70 → belöna "nära sweet spot"
}

# Sektor-/branschviktning per läge
DEFAULT_WEIGHTS_GROWTH = {
    "ps": 0.20, "cagr": 0.30, "gross_margin": 0.15, "net_margin": 0.10,
    "debt_to_equity": 0.10, "ev_ebitda": 0.10, "fcf_margin": 0.05
}
DEFAULT_WEIGHTS_DIV = {
    "div_yield": 0.35, "fcf_margin": 0.20, "net_margin": 0.10, "gross_margin": 0.10,
    "debt_to_equity": 0.15, "payout_fcf": 0.10
}

# Exempel på sektorspecifika justeringar (lägg till fler efter behov)
SECTOR_OVERRIDES_GROWTH: Dict[str, Dict[str, float]] = {
    # Tech: värdera tillväxt/marginaler lite högre
    "Technology": {"cagr": 0.35, "ps": 0.18, "gross_margin": 0.17, "ev_ebitda": 0.10},
    # Utilities: något lägre vikt på cagr, högre på ev/ebitda och debt_to_equity
    "Utilities": {"cagr": 0.20, "ev_ebitda": 0.20, "debt_to_equity": 0.15, "ps": 0.15},
}

SECTOR_OVERRIDES_DIV: Dict[str, Dict[str, float]] = {
    # Utilities: utdelning & balans viktigare
    "Utilities": {"div_yield": 0.40, "debt_to_equity": 0.20, "payout_fcf": 0.15, "fcf_margin": 0.15},
    # REITs (om sektor heter t.ex. Real Estate)
    "Real Estate": {"div_yield": 0.45, "payout_fcf": 0.20, "debt_to_equity": 0.15},
}

# ============================================================
# Hjälpare
# ============================================================

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None

def _avg_positive(vals: List[float]) -> Optional[float]:
    clean = [float(v) for v in vals if _safe_float(v) is not None and float(v) > 0]
    if not clean:
        return None
    return float(np.mean(clean))

def compute_ps_snitt(row: pd.Series) -> Optional[float]:
    qs = []
    for q in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
        if q in row:
            qs.append(_safe_float(row[q]))
    ps_avg = _avg_positive(qs)
    if ps_avg is not None:
        return round(ps_avg, 2)
    # fallback: P/S
    return _safe_float(row.get("P/S", None))

def compute_mcap(row: pd.Series) -> Optional[float]:
    """Beräkna MCAP ~ Pris * (Utestående aktier * 1e6). Samma valuta som priset."""
    px = _safe_float(row.get("Aktuell kurs"))
    shares_m = _safe_float(row.get("Utestående aktier"))
    if px is None or shares_m is None or shares_m <= 0 or px <= 0:
        return None
    return float(px * shares_m * 1e6)

def mcap_bucket(row: pd.Series) -> str:
    mcap = compute_mcap(row)
    if mcap is None:
        return "Okänd"
    for thr, name in MCAP_BUCKETS:
        if mcap >= thr:
            return name
    return "Okänd"

# Normaliseringsfunktioner → [0..1]
def _norm_higher_better(x: Optional[float], lo: float, hi: float) -> Optional[float]:
    if x is None:
        return None
    # cap till [lo, hi]
    v = max(lo, min(hi, float(x)))
    return (v - lo) / (hi - lo) if hi > lo else None

def _norm_lower_better(x: Optional[float], lo_good: float, hi_bad: float) -> Optional[float]:
    """lo_good = bra (->1), hi_bad = dåligt (->0)."""
    if x is None:
        return None
    v = max(lo_good, min(hi_bad, float(x)))
    return 1.0 - ((v - lo_good) / (hi_bad - lo_good)) if hi_bad > lo_good else None

def _norm_sweet_spot(x: Optional[float], sweet_lo: float, sweet_hi: float, range_lo: float, range_hi: float) -> Optional[float]:
    """
    Belöna värden i [sweet_lo, sweet_hi] med 1.0, fallande mot range_lo/range_hi.
    Returnerar 0 om utanför [range_lo, range_hi].
    """
    if x is None:
        return None
    v = float(x)
    if v < range_lo or v > range_hi:
        return 0.0
    if sweet_lo <= v <= sweet_hi:
        return 1.0
    # linjär avtrappning
    if v < sweet_lo:
        return (v - range_lo) / (sweet_lo - range_lo) if sweet_lo > range_lo else 0.0
    # v > sweet_hi
    return (range_hi - v) / (range_hi - sweet_hi) if range_hi > sweet_hi else 0.0

# ============================================================
# Poängberäkning – rå score per rad
# ============================================================

def _collect_metrics(row: pd.Series) -> Dict[str, Optional[float]]:
    """Plocka ut kända nyckeltal från raden om de finns."""
    out: Dict[str, Optional[float]] = {}

    out["ps"] = compute_ps_snitt(row)  # P/S-snitt om möjligt, annars P/S
    out["cagr"] = _safe_float(row.get("CAGR 5 år (%)"))

    # Marginaler (om vi har lagt in dessa fält i databasen)
    out["gross_margin"] = _safe_float(row.get("Bruttomarginal (%)"))
    out["net_margin"]   = _safe_float(row.get("Nettomarginal (%)"))

    # Skuldsättning och värdering
    out["debt_to_equity"] = _safe_float(row.get("Debt/Equity"))
    out["ev_ebitda"]      = _safe_float(row.get("EV/EBITDA"))

    # Kassaflöde
    out["fcf_margin"]     = _safe_float(row.get("FCF-marginal (%)"))

    # Utdelning
    # Om "Årlig utdelning" och "Aktuell kurs" finns kan vi härleda yield
    div_per_share = _safe_float(row.get("Årlig utdelning"))
    price = _safe_float(row.get("Aktuell kurs"))
    if div_per_share is not None and price and price > 0:
        out["div_yield"] = (div_per_share / price) * 100.0
    else:
        out["div_yield"] = _safe_float(row.get("Direktavkastning (%)"))

    # Payout (FCF) – om vi senare beräknar detta i appen, annars None
    out["payout_fcf"] = _safe_float(row.get("Payout FCF (%)"))

    return out

def _pick_weights(sector: Optional[str], mode: str) -> Dict[str, float]:
    if mode == "dividend":
        base = DEFAULT_WEIGHTS_DIV.copy()
        if sector and sector in SECTOR_OVERRIDES_DIV:
            base.update(SECTOR_OVERRIDES_DIV[sector])
        # normalisera till summa 1.0
        s = sum(base.values())
        if s > 0:
            for k in base:
                base[k] = base[k] / s
        return base
    else:
        base = DEFAULT_WEIGHTS_GROWTH.copy()
        if sector and sector in SECTOR_OVERRIDES_GROWTH:
            base.update(SECTOR_OVERRIDES_GROWTH[sector])
        s = sum(base.values())
        if s > 0:
            for k in base:
                base[k] = base[k] / s
        return base

def _metric_score(name: str, x: Optional[float]) -> Optional[float]:
    """
    Returnerar normaliserad score [0..1] för ett givet nyckeltal.
    """
    if name == "ps":
        lo, hi = NORMS["ps"]
        return _norm_lower_better(x, lo, hi)
    if name == "cagr":
        lo, hi = NORMS["cagr"]
        return _norm_higher_better(x, lo, hi)
    if name == "gross_margin":
        lo, hi = NORMS["gross_margin"]
        return _norm_higher_better(x, lo, hi)
    if name == "net_margin":
        lo, hi = NORMS["net_margin"]
        return _norm_higher_better(x, lo, hi)
    if name == "debt_to_equity":
        lo, hi = NORMS["debt_to_equity"]
        return _norm_lower_better(x, lo, hi)
    if name == "ev_ebitda":
        lo, hi = NORMS["ev_ebitda"]
        return _norm_lower_better(x, lo, hi)
    if name == "fcf_margin":
        lo, hi = NORMS["fcf_margin"]
        return _norm_higher_better(x, lo, hi)
    if name == "div_yield":
        lo, hi = NORMS["div_yield"]
        return _norm_higher_better(x, lo, hi)
    if name == "payout_fcf":
        # sweet-spot 40..70%, range 10..100%
        return _norm_sweet_spot(x, 40.0, 70.0, 10.0, 100.0)
    return None

def score_row(row: pd.Series, mode: str = "growth") -> Tuple[float, Dict[str, float], Dict[str, Optional[float]]]:
    """
    Beräknar total score [0..100] för en rad och returnerar:
      (score, scores_per_metric, raw_metrics)
    - scores_per_metric: normaliserade delpoäng [0..1] per nyckeltal som användes
    - raw_metrics: råa värden som låg till grund
    """
    sector = str(row.get("Sektor") or row.get("Sector") or "").strip() or None
    weights = _pick_weights(sector, mode=mode)
    metrics = _collect_metrics(row)

    used_scores: Dict[str, float] = {}
    total = 0.0
    wsum = 0.0

    for mname, w in weights.items():
        if w <= 0:
            continue
        val = metrics.get(mname)
        s = _metric_score(mname, val)
        if s is None:
            # saknat nyckeltal: räkna som neutral (0.5) men med reducerad vikt
            # eller ignorera? Här väljer vi att ignorera (inte räkna med vikten)
            # för att inte belöna/straffa okänd data.
            continue
        used_scores[mname] = float(s)
        total += float(s) * float(w)
        wsum += float(w)

    if wsum <= 0:
        final_score = 50.0  # helt okänt → neutral
    else:
        final_score = float(total / wsum) * 100.0

    return round(final_score, 2), used_scores, metrics

# ============================================================
# Värderingssignal
# ============================================================

def valuation_signal(
    row: pd.Series,
    mode: str,
    score: float,
    riktkurs_col: str = "Riktkurs om 1 år",
) -> Tuple[str, str]:
    """
    Returnerar (label, förklaring) baserat på uppsida mot vald riktkurs och fundamentalscore.
    """
    price = _safe_float(row.get("Aktuell kurs"))
    target = _safe_float(row.get(riktkurs_col))
    upside = None
    if price and price > 0 and target and target > 0:
        upside = (target - price) / price * 100.0

    # Välj etikett enligt reglerna (kombination av min_score & min_upside)
    label = "Fair / Behåll"
    for min_score, min_up, lb in VAL_LABEL_RULES:
        if (score >= min_score) and (upside is None or upside >= min_up):
            label = lb
            break

    # Förklaring (kort)
    expl_bits = []
    expl_bits.append(f"score={score:.1f}")
    if upside is not None:
        expl_bits.append(f"uppsida={upside:.1f}%")
    sector = row.get("Sektor") or row.get("Sector") or ""
    if sector:
        expl_bits.append(f"sektor={sector}")

    return label, " | ".join(expl_bits)

# ============================================================
# Rankning av kandidater
# ============================================================

def rank_candidates(
    df: pd.DataFrame,
    mode: str = "growth",
    riktkurs_col: str = "Riktkurs om 1 år",
    sector_filter: Optional[str] = None,
    mcap_buckets: Optional[List[str]] = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Filtrerar/poängsätter/rankar och returnerar ny DataFrame med extra kolumner:
      - Score
      - Valuationsignal
      - Förklaring (kort)
      - P/S-snitt
      - MCAP Bucket
      - Potential (%)  (mot riktkurs_col)
    """
    work = df.copy()

    # Filtrera bort rader utan pris eller riktkurs
    def _has_data(r: pd.Series) -> bool:
        p = _safe_float(r.get("Aktuell kurs"))
        t = _safe_float(r.get(riktkurs_col))
        return bool(p and p > 0 and t and t > 0)

    work = work[work.apply(_has_data, axis=1)].copy()
    if work.empty:
        return work

    # Sektorfilter
    if sector_filter and sector_filter.strip():
        lab = sector_filter.strip().lower()
        work = work[
            work.get("Sektor", "").astype(str).str.lower().str.contains(lab, na=False)
            | work.get("Sector", "").astype(str).str.lower().str.contains(lab, na=False)
        ].copy()

    # MCAP bucket preliminärt
    work["MCAP Bucket"] = work.apply(mcap_bucket, axis=1)
    if mcap_buckets:
        wh = set([s.lower() for s in mcap_buckets])
        work = work[work["MCAP Bucket"].astype(str).str.lower().isin(wh)].copy()

    # P/S-snitt & Potential
    work["P/S-snitt"] = work.apply(compute_ps_snitt, axis=1).astype(float).fillna(0.0)
    def _potential(r: pd.Series) -> float:
        p = _safe_float(r.get("Aktuell kurs"))
        t = _safe_float(r.get(riktkurs_col))
        if p and t and p > 0:
            return (t - p) / p * 100.0
        return 0.0
    work["Potential (%)"] = work.apply(_potential, axis=1)

    # Score + signal
    scores: List[float] = []
    labels: List[str] = []
    reasons: List[str] = []
    for _, r in work.iterrows():
        s, _, _ = score_row(r, mode=mode)
        lb, ex = valuation_signal(r, mode=mode, score=s, riktkurs_col=riktkurs_col)
        scores.append(s); labels.append(lb); reasons.append(ex)
    work["Score"] = [round(x, 2) for x in scores]
    work["Valuationsignal"] = labels
    work["Förklaring"] = reasons

    # Sortering: 1) högst Score, 2) störst Potential (%)
    work = work.sort_values(by=["Score", "Potential (%)"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    return work
