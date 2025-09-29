# stockapp/scoring.py
# -*- coding: utf-8 -*-
"""
Poängsättning, värdering och ranking:
- compute_fair_valuation_fields(row) -> dict med 'Fair price', 'Fair mcap', 'Upside (%)'
- valuation_grade_from_upside(up) -> etikett ('Mycket bra', 'Bra', 'Fair/Behåll', 'Trimma', 'Säljvarning')
- growth_score_for_row(row) -> (score, breakdown)
- dividend_score_for_row(row) -> (score, breakdown)
- rank_candidates(df, mode='growth'|'dividend', sector_filter=None, cap_filter=None, top_n=None)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Små hjälpare
# ------------------------------------------------------------

def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _clamp01(x: float) -> float:
    if x is None or np.isnan(x):
        return 0.0
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)

def _scale_pos(x: float, lo: float, hi: float) -> float:
    """Mappar [lo..hi] till [0..1] med klämning. Högre är bättre."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    if hi == lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))

def _scale_inv(x: float, lo: float, hi: float) -> float:
    """Lägre är bättre. Mappar [lo..hi] till [1..0]."""
    return 1.0 - _scale_pos(x, lo, hi)

def _safe_div(a: float, b: float) -> float:
    a = _f(a); b = _f(b)
    return a / b if b != 0 else 0.0

def _risk_from_mcap_usd(mcap_usd: float) -> str:
    mc = _f(mcap_usd)
    if mc >= 200_000_000_000:  # >= 200B
        return "Megacap"
    if mc >= 10_000_000_000:
        return "Largecap"
    if mc >= 2_000_000_000:
        return "Midcap"
    if mc >= 300_000_000:
        return "Smallcap"
    return "Microcap"

# ------------------------------------------------------------
# Fair value / Upside via P/S-snitt (din modell)
# ------------------------------------------------------------

def compute_fair_valuation_fields(row: pd.Series) -> Dict[str, float]:
    """
    Beräknar 'Fair price' & 'Upside (%)' enligt:
      fair_mcap = (Omsättning nästa år * P/S-snitt)
      fair_price = fair_mcap / Utestående aktier
      upside% = (fair_price / Aktuell kurs - 1) * 100
    Alla belopp i bolagets prisvaluta (SEK/FX hanteras i visning/portfölj).
    """
    rev_next_m = _f(row.get("Omsättning nästa år", 0.0))  # miljoner
    ps_snitt   = _f(row.get("P/S-snitt", 0.0))
    shares_m   = _f(row.get("Utestående aktier", 0.0))    # milj aktier
    price_now  = _f(row.get("Aktuell kurs", 0.0))

    fair_mcap = 0.0
    fair_price = 0.0
    upside_pct = 0.0

    if rev_next_m > 0 and ps_snitt > 0 and shares_m > 0:
        fair_mcap = rev_next_m * ps_snitt  # "miljoner i prisvaluta"
        fair_price = fair_mcap / shares_m  # prisvaluta per aktie
        if price_now > 0:
            upside_pct = (fair_price / price_now - 1.0) * 100.0

    return {
        "Fair mcap (M)": round(fair_mcap, 2),
        "Fair price": round(fair_price, 4),
        "Upside (%)": round(u pside_pct, 2)
    }

def valuation_grade_from_upside(up: float) -> str:
    """
    Etikett enligt överenskomna trösklar:
      >= +40%:  Mycket bra (billig)
      +15..40:  Bra
      -10..+15: Fair/Behåll
      -25..-10: Trimma
      < -25:    Säljvarning
    """
    u = _f(up)
    if u >= 40.0:
        return "Mycket bra (billig)"
    if u >= 15.0:
        return "Bra"
    if u >= -10.0:
        return "Fair/Behåll"
    if u >= -25.0:
        return "Trimma"
    return "Säljvarning"

# ------------------------------------------------------------
# Viktning per sektor (enkla defaultar)
# ------------------------------------------------------------

def _sector_weights(sector: str, mode: str) -> Dict[str, float]:
    """
    Returnerar vikt-nycklar:
      valuation, growth, margins, leverage, size, stability, yield, safety, quality
    Summan normaliseras till 1.0.
    """
    s = (sector or "").lower()
    m = (mode or "growth").lower()

    # default
    w = dict(
        valuation=0.30,
        growth=0.25,
        margins=0.15,
        leverage=0.10,
        size=0.05,
        stability=0.05,
        yield_=0.05,
        safety=0.05,
        quality=0.00,
    )

    if m == "dividend":
        w = dict(
            valuation=0.15,
            growth=0.10,
            margins=0.10,
            leverage=0.05,
            size=0.10,
            stability=0.10,
            yield_=0.35,
            safety=0.15,
            quality=0.00,
        )

    # Lätta sektorjusteringar
    if "financial" in s or "bank" in s:
        w["leverage"] += 0.05
        w["stability"] += 0.05
        w["valuation"] -= 0.05

    if "real estate" in s:
        w["yield_"] += 0.05
        w["safety"] += 0.05
        w["growth"] -= 0.05

    if "technology" in s:
        w["growth"] += 0.05
        w["valuation"] += 0.05

    # normalisera
    tot = sum(w.values())
    if tot <= 0:
        return w
    return {k: v / tot for k, v in w.items()}

# ------------------------------------------------------------
# Poängfunktioner
# ------------------------------------------------------------

def growth_score_for_row(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    """
    Poäng för tillväxtcase. Returnerar (score 0..100, breakdown).
    Vi använder värden om de finns – annars ignoreras de tyst.
    """
    sector = str(row.get("Sektor") or row.get("Sector") or "")
    w = _sector_weights(sector, mode="growth")
    bd: Dict[str, float] = {}

    # Valuation: lägre P/S bättre, men även "Upside (%)" via fair price
    ps_now = _f(row.get("P/S", 0.0))
    ps_avg = _f(row.get("P/S-snitt", 0.0))
    # Om P/S-snitt saknas – approximera med P/S själv → neutral (0.5)
    if ps_now > 0 and ps_avg > 0:
        val_rel = _scale_inv(ps_now / ps_avg, lo=0.3, hi=3.0)  # <1 bättre
    else:
        val_rel = 0.5

    up = _f(row.get("Upside (%)", 0.0))
    up_norm = _scale_pos(up, lo=-50.0, hi=100.0)

    valuation = 0.6 * val_rel + 0.4 * up_norm
    bd["valuation"] = valuation * 100

    # Growth: CAGR 5 år
    cagr = _f(row.get("CAGR 5 år (%)", 0.0))
    growth = _scale_pos(cagr, lo=0.0, hi=50.0)
    bd["growth"] = growth * 100

    # Margins: Gross/Net margin om finns
    gm = _f(row.get("Gross Margin (%)", row.get("Bruttomarginal (%)", 0.0)))
    nm = _f(row.get("Net Margin (%)", row.get("Nettomarginal (%)", 0.0)))
    margins = 0.5 * _scale_pos(gm, lo=20.0, hi=70.0) + 0.5 * _scale_pos(nm, lo=5.0, hi=30.0)
    bd["margins"] = margins * 100

    # Leverage: Debt/Equity lägre = bättre
    de = _f(row.get("Debt/Equity", 0.0))
    leverage = _scale_inv(de, lo=0.0, hi=2.0)
    bd["leverage"] = leverage * 100

    # Size (risk): större = bättre stabilitet
    risk = str(row.get("Risk") or "")
    risk_idx = dict(Microcap=0.0, Smallcap=0.25, Midcap=0.6, Largecap=0.85, Megacap=1.0).get(risk, None)
    if risk_idx is None:
        mc = _f(row.get("_MC_USD", 0.0))
        risk_idx = dict(Microcap=0.0, Smallcap=0.25, Midcap=0.6, Largecap=0.85, Megacap=1.0)[_risk_from_mcap_usd(mc)]
    size = risk_idx
    bd["size"] = size * 100

    # Stability: lägre beta bättre (om finns)
    beta = _f(row.get("Beta", 1.0))
    stability = _scale_inv(beta, lo=0.6, hi=2.0)
    bd["stability"] = stability * 100

    # Quality (om FCF-margin/EBITDA-margin finns)
    fcf_margin = _f(row.get("FCF Margin (%)", 0.0))
    ebitda_margin = _f(row.get("EBITDA Margin (%)", 0.0))
    if fcf_margin != 0.0 or ebitda_margin != 0.0:
        quality = 0.5 * _scale_pos(fcf_margin, lo=0.0, hi=25.0) + 0.5 * _scale_pos(ebitda_margin, lo=10.0, hi=40.0)
    else:
        quality = 0.0
    bd["quality"] = quality * 100

    # Total score (0..100)
    score01 = (
        w["valuation"] * valuation +
        w["growth"] * growth +
        w["margins"] * margins +
        w["leverage"] * leverage +
        w["size"] * size +
        w["stability"] * stability +
        w["quality"] * quality
    )
    return round(score01 * 100.0, 2), bd

def dividend_score_for_row(row: pd.Series) -> Tuple[float, Dict[str, float]]:
    """
    Poäng för utdelningscase. Returnerar (score 0..100, breakdown).
    """
    sector = str(row.get("Sektor") or row.get("Sector") or "")
    w = _sector_weights(sector, mode="dividend")
    bd: Dict[str, float] = {}

    # Yield
    div_rate = _f(row.get("Årlig utdelning", 0.0))
    price = _f(row.get("Aktuell kurs", 0.0))
    dy = _safe_div(div_rate, price) * 100.0  # i %
    dy_norm = _scale_pos(dy, lo=2.0, hi=10.0)
    bd["yield"] = dy_norm * 100

    # Safety: payout (om finns, EPS/FCF), annars proxys via net margin, leverage
    payout_eps = _f(row.get("Payout EPS (%)", 0.0))
    payout_fcf = _f(row.get("Payout FCF (%)", 0.0))

    if payout_eps > 0 or payout_fcf > 0:
        # lägre payout bättre; över 100% dåligt
        pe = 1.0 - _scale_pos(payout_eps, lo=40.0, hi=110.0) if payout_eps > 0 else 0.5
        pf = 1.0 - _scale_pos(payout_fcf, lo=40.0, hi=110.0) if payout_fcf > 0 else 0.5
        safety = 0.5 * pe + 0.5 * pf
    else:
        nm = _f(row.get("Net Margin (%)", 0.0))
        de = _f(row.get("Debt/Equity", 0.0))
        safety = 0.6 * _scale_pos(nm, lo=5.0, hi=25.0) + 0.4 * _scale_inv(de, lo=0.0, hi=2.0)
    bd["safety"] = _clamp01(safety) * 100

    # Valuation via upside från fair price + relativ P/S
    ps_now = _f(row.get("P/S", 0.0))
    ps_avg = _f(row.get("P/S-snitt", 0.0))
    val_rel = _scale_inv(ps_now / ps_avg, lo=0.5, hi=3.0) if (ps_now > 0 and ps_avg > 0) else 0.5
    up = _f(row.get("Upside (%)", 0.0))
    up_norm = _scale_pos(up, lo=-30.0, hi=80.0)
    valuation = 0.5 * val_rel + 0.5 * up_norm
    bd["valuation"] = valuation * 100

    # Stability & Size
    beta = _f(row.get("Beta", 1.0))
    stability = _scale_inv(beta, lo=0.6, hi=2.0)
    bd["stability"] = stability * 100

    risk = str(row.get("Risk") or "")
    risk_idx = dict(Microcap=0.0, Smallcap=0.25, Midcap=0.6, Largecap=0.85, Megacap=1.0).get(risk, None)
    if risk_idx is None:
        mc = _f(row.get("_MC_USD", 0.0))
        risk_idx = dict(Microcap=0.0, Smallcap=0.25, Midcap=0.6, Largecap=0.85, Megacap=1.0)[_risk_from_mcap_usd(mc)]
    size = risk_idx
    bd["size"] = size * 100

    # Margins & Growth
    gm = _f(row.get("Gross Margin (%)", 0.0))
    nm = _f(row.get("Net Margin (%)", 0.0))
    margins = 0.5 * _scale_pos(gm, lo=20.0, hi=70.0) + 0.5 * _scale_pos(nm, lo=5.0, hi=25.0)
    bd["margins"] = margins * 100

    cagr = _f(row.get("CAGR 5 år (%)", 0.0))
    growth = _scale_pos(cagr, lo=0.0, hi=15.0)
    bd["growth"] = growth * 100

    # Total score
    score01 = (
        w["yield_"] * dy_norm +
        w["safety"] * safety +
        w["valuation"] * valuation +
        w["stability"] * stability +
        w["size"] * size +
        w["margins"] * margins +
        w["growth"] * growth
    )
    return round(score01 * 100.0, 2), bd

# ------------------------------------------------------------
# Ranking-API för vyer
# ------------------------------------------------------------

def _ensure_fair_fields(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Fair price", "Upside (%)"}.issubset(df.columns):
        extra = df.apply(lambda r: compute_fair_valuation_fields(r), axis=1, result_type="expand")
        for c in extra.columns:
            df[c] = extra[c]
    return df

def _apply_mode_score(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    scores = []
    breakdowns: List[Dict[str, float]] = []
    for _, r in df.iterrows():
        if mode == "dividend":
            sc, br = dividend_score_for_row(r)
        else:
            sc, br = growth_score_for_row(r)
        scores.append(sc)
        breakdowns.append(br)
    df = df.copy()
    df["Score"] = scores
    df["_score_breakdown"] = breakdowns
    return df

def rank_candidates(
    df: pd.DataFrame,
    mode: str = "growth",
    sector_filter: Optional[str] = None,
    cap_filter: Optional[List[str]] = None,
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Returnerar nytt DataFrame med Score, Upside, Grade, sorterad på Score desc.
    - mode: 'growth' eller 'dividend'
    - sector_filter: om satt, behåll endast den sektorn (case-insensitive contains)
    - cap_filter: lista av risklabels att behålla (t.ex. ['Largecap','Midcap'])
    - top_n: om satt, truncerar listan
    """
    work = df.copy()

    # Säkerställ fair-fields & grade
    work = _ensure_fair_fields(work)
    work["Grade"] = work["Upside (%)"].apply(valuation_grade_from_upside)

    # Filtrering
    if sector_filter:
        s = sector_filter.lower()
        work = work[work["Sektor"].astype(str).str.lower().str.contains(s, na=False) |
                    work["Sector"].astype(str).str.lower().str.contains(s, na=False)]

    if cap_filter:
        capset = {c.lower() for c in cap_filter}
        risk_col = work.get("Risk")
        if risk_col is None:
            # härleder från _MC_USD
            work["Risk"] = work["_MC_USD"].apply(_risk_from_mcap_usd)
        work = work[work["Risk"].astype(str).str.lower().isin(capset)]

    # Score
    work = _apply_mode_score(work, mode=mode)

    # Sortera & topp
    work = work.sort_values(by=["Score", "Upside (%)"], ascending=[False, False])
    if top_n is not None and top_n > 0:
        work = work.head(top_n)

    # Runda snyggt
    for c in ["Score", "Upside (%)", "Fair price"]:
        if c in work.columns:
            work[c] = work[c].apply(lambda x: round(_f(x), 2))
    return work.reset_index(drop=True)
