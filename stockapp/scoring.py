# stockapp/scoring.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Tuple, Optional
import math
import numpy as np
import pandas as pd

# mjuka imports
try:
    import streamlit as st
except Exception:
    st = None

# Lokala helpers
from .utils import safe_float, normalize_ticker

# ------------------------------------------------------------
# Normaliseringshjälpare
# ------------------------------------------------------------
def _nz(v, d=0.0) -> float:
    try:
        return d if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    except Exception:
        return d

def _cap_between(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ------------------------------------------------------------
# Risketikett utifrån Market Cap (i lok. valuta – används endast som label)
# ------------------------------------------------------------
def risk_label_from_mcap(mcap: float) -> str:
    mc = _nz(mcap)
    if mc >= 2e11:    return "Mega"
    if mc >= 1e10:    return "Large"
    if mc >= 2e9:     return "Mid"
    if mc >= 3e8:     return "Small"
    return "Micro"

# ------------------------------------------------------------
# Potentials & värderingsgap
# ------------------------------------------------------------
def potential_pct(current_px: float, target_px: float) -> float:
    c = _nz(current_px)
    t = _nz(target_px)
    if c <= 0 or t <= 0:
        return 0.0
    return (t - c) / c * 100.0

# ------------------------------------------------------------
# Growth-score & Dividend-score (robusta – saknade fält = neutralt)
# Skalar till 0..100
# ------------------------------------------------------------
def growth_score(row: pd.Series, riktkurs_col: str = "Riktkurs om 1 år") -> float:
    px = _nz(row.get("Aktuell kurs"))
    tgt = _nz(row.get(riktkurs_col))
    pot = potential_pct(px, tgt)  # kan vara neg
    ps_now = _nz(row.get("P/S"))
    ps_avg = _nz(row.get("P/S-snitt"))
    cagr5 = _nz(row.get("CAGR 5 år (%)"))

    # Potentials (40p)
    pot_norm = _cap_between((pot + 20) / 60, 0, 1)  # -20% → 0, +40% → 1
    w_pot = 40 * pot_norm

    # P/S-rimlighet (20p): om ps_now <= 1.25*ps_avg ger poäng, annars avdrag
    if ps_avg > 0 and ps_now > 0:
        ratio = ps_now / max(ps_avg, 1e-9)
        if ratio <= 1.25:
            ps_norm = _cap_between((1.25 - ratio) / 0.25, 0, 1)  # 1.25→0, 1.0→1
        else:
            ps_norm = _cap_between(1 - (ratio - 1.25) / 0.75, 0, 1)  # 2.0→~0
    else:
        ps_norm = 0.5  # neutralt
    w_ps = 20 * ps_norm

    # Tillväxt (25p): cagr 0..50% → 0..1 (cap vid 50%)
    cg_norm = _cap_between(cagr5 / 50.0, 0, 1)
    w_cg = 25 * cg_norm

    # Kvalitet (15p): enkla proxies om finns (brutto/netto-marginal)
    gm = _nz(row.get("Bruttomarginal (%)"))
    nm = _nz(row.get("Netto-marginal (%)"))
    gm_norm = _cap_between(gm / 60.0, 0, 1)   # 60%+ bra
    nm_norm = _cap_between((nm + 10) / 30.0, 0, 1)  # -10..+20% → 0..1
    qual = 0.65 * gm_norm + 0.35 * nm_norm
    w_ql = 15 * qual

    score = w_pot + w_ps + w_cg + w_ql
    return _cap_between(score, 0, 100)

def dividend_score(row: pd.Series) -> float:
    # Direktavkastning
    px = _nz(row.get("Aktuell kurs"))
    div_ps = _nz(row.get("Årlig utdelning"))  # per aktie, bolagsvaluta
    yld = div_ps / px if px > 0 and div_ps >= 0 else 0.0  # ex: 0.05 = 5%

    # FCF & kapitalstyrka (om finns)
    fcf_m = _nz(row.get("Free Cash Flow (M)"))
    cash_m = _nz(row.get("Kassa (M)"))
    debt_to_equity = _nz(row.get("Debt/Equity"))

    # Payout approx via FCF: payout_fcf = utdelning_summa / FCF
    shares_m = _nz(row.get("Utestående aktier"))  # i miljoner
    payout_fcf = None
    if shares_m > 0 and div_ps > 0 and fcf_m > 0:
        # gissa utdelning i miljoner (per-aktie * antal aktier)
        payout_fcf = (div_ps * shares_m) / max(fcf_m, 1e-9)
    # Normaliseringar
    y_norm = _cap_between(yld / 0.08, 0, 1)          # 8%+ ⇒ max
    fcf_norm = 1.0 if fcf_m > 0 else 0.2             # negativ FCF = lågt
    cash_norm = _cap_between(cash_m / 1000.0, 0, 1)  # 1000M = topp (heuristik)
    if debt_to_equity > 0:
        de_norm = _cap_between(1.5 / debt_to_equity, 0, 1)  # <=1.5 bra
    else:
        de_norm = 0.6
    if payout_fcf is None:
        pr_norm = 0.5
    else:
        # 0.4..0.7 bra ⇒ hög poäng, >1 dåligt
        if payout_fcf <= 0:
            pr_norm = 0.8
        elif payout_fcf <= 0.7:
            pr_norm = 1.0
        elif payout_fcf <= 1.0:
            pr_norm = 0.6
        else:
            pr_norm = _cap_between(1.5 / payout_fcf, 0, 0.5)

    # Vikter
    score = (
        35 * y_norm +
        20 * pr_norm +
        20 * fcf_norm +
        15 * cash_norm +
        10 * de_norm
    )
    return _cap_between(score, 0, 100)

# ------------------------------------------------------------
# Handlingsetikett (Köp/Håll/Trimma/Sälj) + motivering
# ------------------------------------------------------------
def assign_action_label(
    row: pd.Series,
    mode: str = "Tillväxt",
    riktkurs_col: str = "Riktkurs om 1 år",
    pos_weight_pct: Optional[float] = None,
    gav_sek: Optional[float] = None,
    fx_to_sek: Optional[float] = None,
) -> Tuple[str, str, Dict[str, float]]:
    """
    Returnerar (label, reason, metrics) där:
      label ∈ {"Köp","Håll","Trimma","Sälj"}
      reason = kort text
      metrics = {"score":..., "potential":..., "yield":..., ...}
    """
    px = _nz(row.get("Aktuell kurs"))
    tgt = _nz(row.get(riktkurs_col))
    pot = potential_pct(px, tgt)

    # yield (om utdelning)
    div_ps = _nz(row.get("Årlig utdelning"))
    yld = div_ps / px if px > 0 and div_ps >= 0 else 0.0

    # score per läge
    if (mode or "").lower().startswith("utdel"):
        score = dividend_score(row)
    else:
        score = growth_score(row, riktkurs_col=riktkurs_col)

    # positionens vikt i portföljen (kan vara None)
    pos_w = _nz(pos_weight_pct, None)
    overweight = (pos_w is not None and pos_w >= 15.0)

    # GAV-kopplad context (SEK)
    gav_ok = False
    if gav_sek and fx_to_sek and px > 0:
        px_sek = px * fx_to_sek
        gain_pct = (px_sek - gav_sek) / max(gav_sek, 1e-9) * 100.0
        gav_ok = True
    else:
        gain_pct = None

    # Basregler – robusta, enkla
    label: str
    reasons = []

    # Justera thresholds lite mellan lägen
    if (mode or "").lower().startswith("utdel"):
        # Utdelning – mer vikt på yield och hållbarhet
        if score >= 75 and pot >= -5 and yld >= 0.035:
            label = "Köp"
            reasons.append("Hög poäng för utdelning & hållbarhet.")
        elif score >= 55 and pot >= -15:
            label = "Håll"
            reasons.append("Stabil utdelningsprofil.")
        else:
            # nedre fallet – fundera på trim/sälj
            if pot <= -15 or score < 40:
                label = "Sälj" if (yld < 0.02 or score < 35) else "Trimma"
            else:
                label = "Trimma"
            reasons.append("Svag utdelningsprofil eller övervärderad.")
    else:
        # Tillväxt
        if score >= 70 and pot >= 10:
            label = "Köp"
            reasons.append("Hög kvalitet + attraktiv uppsida.")
        elif score >= 55 and pot >= -5:
            label = "Håll"
            reasons.append("OK kvalitet och rimlig värdering.")
        else:
            if pot <= -20 and score < 45:
                label = "Sälj"
                reasons.append("Tydlig övervärdering i förhållande till kvalitet.")
            else:
                label = "Trimma"
                reasons.append("Begränsad uppsida eller tveksam kvalitet.")

    # Övervikt → tryck ned mot Trimma/Sälj
    if overweight:
        if label == "Köp":
            label = "Håll"
            reasons.append("Övervikt i portföljen – undvik att öka.")
        elif label == "Håll":
            label = "Trimma"
            reasons.append("Övervikt i portföljen – överväg att trimma.")
        elif label == "Trimma":
            reasons.append("Övervikt förstärker trim-signal.")

    # GAV & vinstskydd (bara förstärkning – inte motsägelse)
    if gav_ok and gain_pct is not None:
        if gain_pct > 50 and (label in ("Köp", "Håll")) and pot <= 0:
            label = "Trimma"
            reasons.append("Stor orealiserad vinst & begränsad uppsida – säkra del av vinsten.")
        if gain_pct < -30 and label == "Sälj":
            # fallande kniv – låt säljsignal stå kvar
            reasons.append("Betydande nedgång vs GAV – var disciplinerad.")

    metrics = {
        "score": round(score, 1),
        "potential_pct": round(pot, 2),
        "yield_pct": round(yld * 100.0, 2),
    }
    return label, " ".join(reasons), metrics
