# stockapp/scoring.py
# -*- coding: utf-8 -*-
"""
Poängmodell / ranking för investeringsförslag.

Huvud-API:
- compute_scoring_inputs(df, user_rates) -> df  (beräknar komponentkolumner)
- score_growth(df) -> pd.Series (0..100)
- score_dividend(df) -> pd.Series (0..100)
- grade_from_score(score) -> str   (Utmärkt/Bra/OK/Svag/Riskabel)
- reco_from_score_and_valuation(score, valuation_label) -> str  (Köp/Överväg/Avvakta/Trimma/Sälj)
- rank_for_investing(df, mode="growth", sector=None, cap=None, top=50) -> df (filtrerad + sorterad)

Modellen använder:
- P/S, P/S-snitt (via ps_rel = P/S / P/S-snitt; lägre är bättre)
- Estimerad tillväxt: (Omsättning nästa år / Omsättning idag - 1)
- Valuation gap: (Riktkurs om 1 år / Aktuell kurs - 1)
- Utdelningsyield: Årlig utdelning / Aktuell kurs
- Storlek: Market Cap i USD (via finance.market_cap_usd eller befintlig _MC_USD)
- Risklabel: Micro/Small/Mid/Large/Mega → siffra för penalty
- Sektorjustering: små viktjusteringar per sektor (om kolumn 'Sektor'/'Sector' finns)

Kräver:
- Kolumner som appen redan hanterar (Aktuell kurs, Årlig utdelning, P/S, P/S Q1..Q4, Omsättning idag/ nästa år,
  Riktkurs om 1 år, Valuta, Utestående aktier, ev. _MC_USD/Risk/Valuation).
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

from .finance import (
    ps_snitt_from_row,
    market_cap_usd,
    valuation_label,
)

# --------------------------------------------------------------------
# Normalisering & utils
# --------------------------------------------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _has_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def _sector_col(df: pd.DataFrame) -> str:
    # Tillåt både 'Sektor' och 'Sector'
    if "Sektor" in df.columns: return "Sektor"
    if "Sector" in df.columns: return "Sector"
    return ""

def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)

def _robust_minmax(series: pd.Series, higher_better: bool = True) -> pd.Series:
    """
    Skalar till [0,1] via 5:e och 95:e percentilen (robust mot outliers).
    Om alla lika → 0.5.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(np.nan)
    if s.dropna().empty:
        return pd.Series([0.5] * len(s), index=s.index)

    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)
    if hi <= lo:
        out = pd.Series([0.5] * len(s), index=s.index)
    else:
        out = (s - lo) / (hi - lo)
        out = out.clip(0, 1)

    if higher_better:
        return out.fillna(0.5)
    return (1.0 - out).fillna(0.5)

# --------------------------------------------------------------------
# Risk-mappning + sektorjusteringar
# --------------------------------------------------------------------

_RISK_MAP = {
    "Microcap": 1.00,  # störst risk → störst penalty
    "Smallcap": 0.75,
    "Midcap":   0.50,
    "Largecap": 0.25,
    "Megacap":  0.10,
}

# Lätta sektorjusteringar på vikter (multiplikatorer).
# Nycklarna matchar df[_sector_col] case-insensitivt.
_SECTOR_WEIGHT_ADJ = {
    # Tech: tillväxt, valuation gap får lite extra vikt
    "information technology": {"growth": {"growth": 1.10, "valgap": 1.10, "psrel": 1.00, "yield": 0.90},
                               "div":    {"yield": 1.00, "safety": 1.00}},
    "technology":             {"growth": {"growth": 1.10, "valgap": 1.10, "psrel": 1.00, "yield": 0.90},
                               "div":    {"yield": 1.00, "safety": 1.00}},
    # Utilities/Telecom/Staples: dividend-vikt upp
    "utilities":              {"growth": {"yield": 1.05, "psrel": 1.00},
                               "div":    {"yield": 1.15, "safety": 1.10}},
    "telecommunication":      {"growth": {"yield": 1.05},
                               "div":    {"yield": 1.15, "safety": 1.10}},
    "communication services": {"growth": {"yield": 1.05},
                               "div":    {"yield": 1.12, "safety": 1.08}},
    "consumer staples":       {"growth": {"yield": 1.05},
                               "div":    {"yield": 1.12, "safety": 1.08}},
    # Industrials: psrel/valgap neutral, tillväxt marginellt upp
    "industrials":            {"growth": {"growth": 1.05}, "div": {"safety": 1.05}},
    # Financials: safety (storlek/risk) viktigare i dividend-läge
    "financials":             {"growth": {"valgap": 1.05}, "div": {"safety": 1.12}},
    # Energy: dividend ofta viktig, safety upp
    "energy":                 {"growth": {"yield": 1.05}, "div": {"yield": 1.10, "safety": 1.10}},
    # Health Care: growth+valgap upp lite
    "health care":            {"growth": {"growth": 1.08, "valgap": 1.08}, "div": {"safety": 1.05}},
    # Consumer Discretionary: growth prioriteras
    "consumer discretionary": {"growth": {"growth": 1.10, "valgap": 1.05}, "div": {"yield": 1.00}},
    # Materials: neutral
    "materials":              {"growth": {}, "div": {}},
    # Real Estate: dividend & safety
    "real estate":            {"growth": {"yield": 1.05}, "div": {"yield": 1.15, "safety": 1.10}},
}

def _sector_adjust(weights: Dict[str, float], sector: str, mode: str) -> Dict[str, float]:
    """
    Multiplicera base-vikter med sektorjusteringar.
    keys i weights: 'growth','valgap','psrel','yield','safety'
    """
    if not sector:
        return weights
    key = sector.strip().lower()
    for s_name, adj in _SECTOR_WEIGHT_ADJ.items():
        if s_name in key:
            m = adj.get("growth" if mode == "growth" else "div", {})
            out = weights.copy()
            for k, mult in m.items():
                if k in out:
                    out[k] *= float(mult)
            return out
    return weights

# --------------------------------------------------------------------
# Beräkning av komponentkolumner
# --------------------------------------------------------------------

def compute_scoring_inputs(df: pd.DataFrame, user_rates: Optional[Dict[str, float]]) -> pd.DataFrame:
    """
    Lägger till/uppdaterar följande kolumner:
      - P/S-snitt
      - est_growth: (Omsättning nästa år / Omsättning idag) - 1
      - val_gap: (Riktkurs om 1 år / Aktuell kurs) - 1
      - ps_rel: P/S / P/S-snitt   (clip 0.1..10 och inverteras senare i normalisering)
      - div_yield: Årlig utdelning / Aktuell kurs
      - _MC_USD: Market cap i USD (om saknas)
      - risk_penalty: mappar Risk → tal (0.1..1.0)
    """
    w = df.copy()

    # P/S-snitt (om saknas)
    if not _has_col(w, "P/S-snitt"):
        w["P/S-snitt"] = w.apply(ps_snitt_from_row, axis=1)

    # estimerad tillväxt
    rev_now = pd.to_numeric(w.get("Omsättning idag", pd.Series([np.nan] * len(w))), errors="coerce")
    rev_next = pd.to_numeric(w.get("Omsättning nästa år", pd.Series([np.nan] * len(w))), errors="coerce")
    w["est_growth"] = np.where((rev_now > 0) & (rev_next > 0), (rev_next / rev_now) - 1.0, np.nan)

    # valuation gap
    price = pd.to_numeric(w.get("Aktuell kurs", pd.Series([np.nan] * len(w))), errors="coerce")
    rk1y  = pd.to_numeric(w.get("Riktkurs om 1 år", pd.Series([np.nan] * len(w))), errors="coerce")
    w["val_gap"] = np.where((price > 0) & (rk1y > 0), (rk1y / price) - 1.0, np.nan)

    # ps_rel
    ps = pd.to_numeric(w.get("P/S", pd.Series([np.nan] * len(w))), errors="coerce")
    psavg = pd.to_numeric(w.get("P/S-snitt", pd.Series([np.nan] * len(w))), errors="coerce")
    w["ps_rel"] = np.where((ps > 0) & (psavg > 0), ps / psavg, np.nan)
    w["ps_rel"] = w["ps_rel"].clip(lower=0.1, upper=10.0)

    # dividend yield
    div = pd.to_numeric(w.get("Årlig utdelning", pd.Series([0.0] * len(w))), errors="coerce").fillna(0.0)
    px  = pd.to_numeric(w.get("Aktuell kurs", pd.Series([np.nan] * len(w))), errors="coerce")
    w["div_yield"] = np.where(px > 0, div / px, 0.0)

    # _MC_USD (om saknas)
    if not _has_col(w, "_MC_USD"):
        w["_MC_USD"] = w.apply(lambda r: market_cap_usd(r, user_rates), axis=1)

    # risk_penalty (låg bättre)
    risk = w.get("Risk", pd.Series(["-"] * len(w)))
    w["risk_penalty"] = [ _RISK_MAP.get(str(x), 0.50) for x in risk ]

    return w

# --------------------------------------------------------------------
# Poängberäkning
# --------------------------------------------------------------------

# Basvikter (innan sektor-justeringar).
# Growth-mode prioriterar est_growth, val_gap och ps_rel (inverterat).
_BASE_WEIGHTS_GROWTH = {
    "growth": 0.35,   # est_growth
    "valgap": 0.30,   # val_gap
    "psrel":  0.20,   # ps_rel (lower better → inverteras via normalisering)
    "yield":  0.05,   # div_yield
    "safety": 0.10,   # (1 - risk_penalty_norm)
}

# Dividend-mode prioriterar utdelningsyield och säkerhet/storlek.
_BASE_WEIGHTS_DIV = {
    "growth": 0.10,
    "valgap": 0.20,
    "psrel":  0.15,
    "yield":  0.35,
    "safety": 0.20,
}

def _norm_components(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Producerar normaliserade komponenter 0..1:
      - growth_s (higher better)
      - valgap_s (higher better)
      - psrel_s (lower better → vi matar ps_rel och sätter higher_better=False)
      - yield_s (higher better)
      - safety_s (här normaliserar vi risk_penalty och inverterar → högre är bättre)
    """
    comps = {}
    comps["growth_s"] = _robust_minmax(df["est_growth"], higher_better=True) if "est_growth" in df else pd.Series([0.5]*len(df), index=df.index)
    comps["valgap_s"] = _robust_minmax(df["val_gap"],    higher_better=True) if "val_gap"    in df else pd.Series([0.5]*len(df), index=df.index)
    comps["psrel_s"]  = _robust_minmax(df["ps_rel"],     higher_better=False) if "ps_rel"     in df else pd.Series([0.5]*len(df), index=df.index)
    comps["yield_s"]  = _robust_minmax(df["div_yield"],  higher_better=True) if "div_yield"  in df else pd.Series([0.5]*len(df), index=df.index)
    # risk → lower better, så vi skickar higher_better=False
    comps["safety_s"] = _robust_minmax(df["risk_penalty"], higher_better=False) if "risk_penalty" in df else pd.Series([0.5]*len(df), index=df.index)
    return comps

def _apply_weights(comps: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    # Se till att alla komponenter finns
    growth = comps.get("growth_s"); valgap = comps.get("valgap_s")
    psrel  = comps.get("psrel_s");  yld    = comps.get("yield_s")
    safe   = comps.get("safety_s")

    # Defaults
    if growth is None: growth = pd.Series([0.5]*len(next(iter(comps.values()))), index=next(iter(comps.values())).index)
    if valgap is None: valgap = pd.Series([0.5]*len(growth), index=growth.index)
    if psrel  is None: psrel  = pd.Series([0.5]*len(growth), index=growth.index)
    if yld    is None: yld    = pd.Series([0.5]*len(growth), index=growth.index)
    if safe   is None: safe   = pd.Series([0.5]*len(growth), index=growth.index)

    w_growth = float(weights.get("growth", 0.0))
    w_val    = float(weights.get("valgap", 0.0))
    w_ps     = float(weights.get("psrel",  0.0))
    w_y      = float(weights.get("yield",  0.0))
    w_s      = float(weights.get("safety", 0.0))

    total_w = max(1e-9, w_growth + w_val + w_ps + w_y + w_s)
    score01 = (w_growth * growth + w_val * valgap + w_ps * psrel + w_y * yld + w_s * safe) / total_w
    return _clip01(score01) * 100.0  # 0..100

def _weights_for_row(base_weights: Dict[str, float], sector: str, mode: str) -> Dict[str, float]:
    return _sector_adjust(base_weights, sector, mode)

def score_growth(df: pd.DataFrame) -> pd.Series:
    comps = _norm_components(df)
    sec_col = _sector_col(df)
    if sec_col:
        weights_each_row = [
            _weights_for_row(_BASE_WEIGHTS_GROWTH, str(sector), mode="growth")
            for sector in df[sec_col].fillna("").astype(str).tolist()
        ]
        # radvis vikter → gör en vektoriserad approx (gemensam baseline + radvisa korr)
        # förenkling: ta radvis dot
        scores = []
        for i, (idx, _) in enumerate(df.iterrows()):
            w = weights_each_row[i]
            score_i = _apply_weights({k: v.iloc[[i]].rename(index={idx: 0}) for k, v in comps.items()}, w).iloc[0]
            scores.append(score_i)
        return pd.Series(scores, index=df.index, name="ScoreGrowth")
    else:
        # inga sektorer → basvikter
        return _apply_weights(comps, _BASE_WEIGHTS_GROWTH).rename("ScoreGrowth")

def score_dividend(df: pd.DataFrame) -> pd.Series:
    comps = _norm_components(df)
    sec_col = _sector_col(df)
    if sec_col:
        weights_each_row = [
            _weights_for_row(_BASE_WEIGHTS_DIV, str(sector), mode="div")
            for sector in df[sec_col].fillna("").astype(str).tolist()
        ]
        scores = []
        for i, (idx, _) in enumerate(df.iterrows()):
            w = weights_each_row[i]
            score_i = _apply_weights({k: v.iloc[[i]].rename(index={idx: 0}) for k, v in comps.items()}, w).iloc[0]
            scores.append(score_i)
        return pd.Series(scores, index=df.index, name="ScoreDividend")
    else:
        return _apply_weights(comps, _BASE_WEIGHTS_DIV).rename("ScoreDividend")

# --------------------------------------------------------------------
# Betyg & rekommendation
# --------------------------------------------------------------------

def grade_from_score(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "-"
    if s >= 80: return "Utmärkt"
    if s >= 65: return "Bra"
    if s >= 50: return "OK"
    if s >= 35: return "Svag"
    return "Riskabel"

def reco_from_score_and_valuation(score: float, valuation: str) -> str:
    """
    Kombinerar modellpoäng med värderingsindikator:
      - Om valuation ∈ {"Sälj"} → "Sälj"
      - Om valuation ∈ {"Trimma"} och score < 60 → "Trimma", annars "Avvakta"
      - Annars baserat på score:
           >=75 → "Köp"
           60–75 → "Överväg"
           45–60 → "Avvakta"
           <45   → "Avstå"
    """
    val = str(valuation or "-")
    try:
        s = float(score)
    except Exception:
        s = 50.0

    if val == "Sälj":
        return "Sälj"
    if val == "Trimma":
        return "Trimma" if s < 60 else "Avvakta"

    if s >= 75: return "Köp"
    if s >= 60: return "Överväg"
    if s >= 45: return "Avvakta"
    return "Avstå"

# --------------------------------------------------------------------
# Ranking-hjälpare
# --------------------------------------------------------------------

def rank_for_investing(
    df: pd.DataFrame,
    mode: str = "growth",
    sector: Optional[str] = None,
    cap: Optional[str] = None,
    top: int = 50,
) -> pd.DataFrame:
    """
    Filtrerar och rangordnar efter vald modell.
    - mode: "growth" eller "dividend"
    - sector: om satt, matchar (case-insensitive substring) mot Sektor/Sector
    - cap: en av {"Microcap","Smallcap","Midcap","Largecap","Megacap"} för att filtrera
    - top: antal rader i resultatet
    Returnerar kopia med kolumner: Score, Grade, Valuation, Reco + inputs.
    """
    work = df.copy()

    # Säkra scoring-inputs
    work = compute_scoring_inputs(work, user_rates=None)

    # Filtrera på sektor
    sec_col = _sector_col(work)
    if sector and sec_col:
        key = sector.strip().lower()
        mask = work[sec_col].fillna("").str.lower().str.contains(key, na=False)
        work = work.loc[mask].copy()

    # Filtrera på cap
    if cap and "Risk" in work.columns:
        work = work.loc[work["Risk"].astype(str).str.lower() == str(cap).lower()].copy()

    # Score
    if mode.lower().startswith("div"):
        scr = score_dividend(work)
        work["Score"] = scr
    else:
        scr = score_growth(work)
        work["Score"] = scr

    # Grade/Valuation/Reco
    work["Grade"] = work["Score"].apply(grade_from_score)
    # Faller tillbaka till att räkna valuation_label om kolumn saknas
    if "Valuation" not in work.columns:
        work["Valuation"] = work.apply(lambda r: valuation_label(r, "Riktkurs om 1 år"), axis=1)
    work["Reco"] = [reco_from_score_and_valuation(s, v) for s, v in zip(work["Score"], work["Valuation"])]

    # Sortera
    work = work.sort_values(by=["Score", "Valuation"], ascending=[False, True])

    # Returnera topp
    cols_pref = [
        "Ticker", "Bolagsnamn", "Sektor" if "Sektor" in work.columns else ("Sector" if "Sector" in work.columns else None),
        "Aktuell kurs", "Valuta", "Årlig utdelning", "div_yield",
        "P/S", "P/S-snitt", "ps_rel",
        "Omsättning idag", "Omsättning nästa år", "est_growth",
        "Riktkurs om 1 år", "val_gap",
        "_MC_USD", "Risk",
        "Score", "Grade", "Valuation", "Reco"
    ]
    cols_pref = [c for c in cols_pref if c and c in work.columns]
    rest = [c for c in work.columns if c not in cols_pref]
    out = work[cols_pref + rest].head(int(top)).copy()
    return out
