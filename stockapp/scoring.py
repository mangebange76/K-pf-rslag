# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .config import SCORING_WEIGHTS_GROWTH, SCORING_WEIGHTS_DIVIDEND

def _nz(x):
    try: return float(x) if x is not None else 0.0
    except: return 0.0

def _minmax(x, lo, hi, invert=False):
    if hi==lo: return 0.0
    v = (x - lo) / (hi - lo)
    v = max(0.0, min(1.0, v))
    return 1.0 - v if invert else v

def score_growth(df: pd.DataFrame) -> pd.DataFrame:
    """
    En enkel komposit: valuation(PS-gap + riktkursgap) + growth(CAGR) + quality(margins) + safety(runway, D/E).
    Skalar 0..100.
    """
    if df is None or df.empty: return df
    w = SCORING_WEIGHTS_GROWTH
    d = df.copy()

    # features
    ps_now = d["P/S"].astype(float).fillna(0.0)
    ps_avg = d["P/S-snitt"].astype(float).fillna(0.0)
    ps_gap = (ps_avg - ps_now) / ps_avg.replace(0, np.nan)
    ps_gap = ps_gap.replace([np.inf,-np.inf], 0.0).fillna(0.0)

    rk = d["Riktkurs om 1 år"].astype(float).fillna(0.0)
    px = d["Aktuell kurs"].astype(float).fillna(0.0)
    rk_gap = (rk - px) / px.replace(0, np.nan)
    rk_gap = rk_gap.replace([np.inf,-np.inf], 0.0).fillna(0.0)

    cagr = d["CAGR 5 år (%)"].astype(float).fillna(0.0)
    gm = d["Gross Margin (%)"].astype(float).fillna(0.0)
    nm = d["Net Margin (%)"].astype(float).fillna(0.0)
    runway = d["Runway (quarters)"].astype(float).fillna(0.0)
    de = d["Debt/Equity"].astype(float).fillna(0.0)

    # normalize
    v1 = _minmax(ps_gap, -1.0, 1.0)   # -100%..+100% rabatt mot snitt
    v2 = _minmax(rk_gap, 0.0, 1.0)    # 0..100% uppsida
    valuation = 0.6*v1 + 0.4*v2

    growth = _minmax(cagr, 0.0, 60.0)

    quality = 0.6*_minmax(gm, 0.0, 70.0) + 0.4*_minmax(nm, 0.0, 40.0)

    safety = 0.6*_minmax(runway, 0.0, 12.0) + 0.4*_minmax(de, 0.0, 2.0, invert=True)

    score = 100.0*(w["valuation"]*valuation + w["growth"]*growth + w["quality"]*quality + w["safety"]*safety)
    d["_growth_score"] = score.round(2)
    return d

def score_dividend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dividend score: yield, safety (payout ~ FCF, D/E), quality (FCF margin), valuation (ps-gap).
    """
    if df is None or df.empty: return df
    w = SCORING_WEIGHTS_DIVIDEND
    d = df.copy()

    # yield
    div = d["Årlig utdelning"].astype(float).fillna(0.0)
    px = d["Aktuell kurs"].astype(float).fillna(0.0)
    dy = (div / px.replace(0, np.nan)) * 100.0
    dy = dy.replace([np.inf,-np.inf], 0.0).fillna(0.0)

    # payout via FCF (enkel approx: utdelning / max(FCF, liten))
    fcf = d["Free Cash Flow"].astype(float).fillna(0.0)
    payout = 0.0*dy
    denom = fcf.copy()
    denom[denom==0] = np.nan
    payout = (div * d["Utestående aktier"].astype(float).fillna(0.0) * 1e6) / denom
    payout = payout.replace([np.inf,-np.inf], np.nan).fillna(0.0) * 100.0  # %

    de = d["Debt/Equity"].astype(float).fillna(0.0)
    fcf_margin = d["FCF Margin (%)"].astype(float).fillna(0.0)
    ps_now = d["P/S"].astype(float).fillna(0.0)
    ps_avg = d["P/S-snitt"].astype(float).fillna(0.0)
    ps_gap = (ps_avg - ps_now) / ps_avg.replace(0, np.nan)
    ps_gap = ps_gap.replace([np.inf,-np.inf], 0.0).fillna(0.0)

    comp = 100.0*(
        w["yield"]    * _minmax(dy, 0.0, 12.0) +
        w["safety"]   * (0.6*_minmax(payout, 20.0, 80.0, invert=True) + 0.4*_minmax(de, 0.0, 2.0, invert=True)) +
        w["quality"]  * _minmax(fcf_margin, 0.0, 20.0) +
        w["valuation"]* _minmax(ps_gap, -1.0, 1.0)
    )
    d["_div_score"] = comp.round(2)
    return d

def valuation_label(row: pd.Series) -> str:
    """
    Grov etikett baserad på uppsida och PS-gap.
    """
    ps_now = _nz(row.get("P/S"))
    ps_avg = _nz(row.get("P/S-snitt"))
    px = _nz(row.get("Aktuell kurs"))
    rk = _nz(row.get("Riktkurs om 1 år"))
    gap = (rk - px)/px if px>0 else 0.0
    ps_rel = (ps_avg - ps_now)/ps_avg if ps_avg>0 else 0.0

    score = 0.6*ps_rel + 0.4*gap
    if score >= 0.35: return "Mycket billig"
    if score >= 0.15: return "Billig"
    if score >= -0.05: return "Fair"
    if score >= -0.20: return "Övervärderad (trim)"
    return "Övervärderad (sälj)"
