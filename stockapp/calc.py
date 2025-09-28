# stockapp/calc.py
# -*- coding: utf-8 -*-
"""
Beräkningar och etiketter:
- P/S-snitt (om saknas)
- Omsättning om 2 & 3 år (från manuella prognoser + CAGR-clamp)
- Riktkurser (idag, 1, 2, 3 år)
- Direktavkastning (%)
- Risklabel (Micro/Small/Mid/Large/Mega) baserat på Market Cap (nu)
- Sektorviktade poäng: GrowthScore, DividendScore
- Övergripande värdering ("Köp", "Bra", "Fair/Behåll", "Trimma", "Sälj")

OBS:
- Rör inte manuella fält "Omsättning idag" & "Omsättning nästa år".
- Utestående aktier lagras i MILJONER (st) – och omsättningar i MILJONER (i bolagets valuta).
  Därför blir riktkurs: (Omsättning * P/S-snitt) / (Utestående aktier).
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .config import säkerställ_kolumner, konvertera_typer

# -----------------------------------------------------------------------------
# Små hjälpare
# -----------------------------------------------------------------------------

def clamp_cagr(raw_cagr_pct: float) -> float:
    """Clamp logik: >100% -> 50%, <0 -> 2%."""
    if raw_cagr_pct is None:
        return 2.0
    try:
        c = float(raw_cagr_pct)
    except Exception:
        return 2.0
    if c > 100.0:
        return 50.0
    if c < 0.0:
        return 2.0
    return c

def safe_float(x, default=0.0) -> float:
    try:
        f = float(x)
        if np.isnan(f):
            return default
        return f
    except Exception:
        return default

def ps_average_from_quarters(row: pd.Series) -> float:
    vals = [safe_float(row.get(f"P/S Q{i}", 0.0)) for i in range(1,5)]
    vals = [v for v in vals if v > 0]
    return round(float(np.mean(vals)), 2) if vals else 0.0

def human_mcap(n: float) -> str:
    """Formatera market cap med T/B/M (engelsk skala)."""
    x = safe_float(n, 0.0)
    if x >= 1e12:
        return f"{x/1e12:.2f} T"
    if x >= 1e9:
        return f"{x/1e9:.2f} B"
    if x >= 1e6:
        return f"{x/1e6:.2f} M"
    return f"{x:.0f}"

def marketcap_risk_label(mcap: float) -> str:
    """
    Indelning (engelsk skala):
    Micro < 300M, Small < 2B, Mid < 10B, Large < 200B, Mega 200B+
    """
    v = safe_float(mcap, 0.0)
    if v < 3e8:      return "Microcap"
    if v < 2e9:      return "Smallcap"
    if v < 1e10:     return "Midcap"
    if v < 2e11:     return "Largecap"
    return "Megacap"

# -----------------------------------------------------------------------------
# Poängmodeller (sektorviktade)
# -----------------------------------------------------------------------------

def _sector_weights(sector: str) -> Dict[str, float]:
    """
    Returnerar vikter för Growth vs Dividend (summerar till 1.0).
    Justerar lite beroende på sektor.
    """
    s = (sector or "").lower()
    if any(k in s for k in ["technology", "communication", "consumer discretionary", "health care", "industrial", "industrials"]):
        return {"growth": 0.7, "dividend": 0.3}
    if any(k in s for k in ["utilities", "real estate", "financial", "financials", "energy", "consumer staples"]):
        return {"growth": 0.4, "dividend": 0.6}
    # default
    return {"growth": 0.55, "dividend": 0.45}

def _score_growth(row: pd.Series) -> float:
    """
    GrowthScore 0–100. Högre bättre.
    Faktorer:
      + CAGR 5 år (upp till 40% ger maxpoäng för den komponenten)
      + Bruttomarginal, Nettomarginal (positiva bra)
      + P/S-snitt (lägre bättre → invers-skala med cutoff)
      + FCF (TTM) positivt en liten bonus
      + Debt/Equity lägre bättre
    """
    cagr = safe_float(row.get("CAGR 5 år (%)", 0.0))
    gm = safe_float(row.get("Bruttomarginal (%)", 0.0))
    nm = safe_float(row.get("Nettomarginal (%)", 0.0))
    ps = safe_float(row.get("P/S-snitt", 0.0))
    fcf = safe_float(row.get("FCF (TTM)", 0.0))
    de = safe_float(row.get("Debt/Equity", 0.0))

    # skala
    cagr_score = min(max(cagr, 0.0), 40.0) / 40.0  # 0..1
    gm_score   = min(max(gm, 0.0), 60.0) / 60.0
    nm_score   = min(max(nm, 0.0), 30.0) / 30.0

    # P/S: invers – under 10 anses ok, 5 bättre, under 2 top
    if ps <= 0:
        ps_score = 0.2
    elif ps <= 2:
        ps_score = 1.0
    elif ps <= 5:
        ps_score = 0.8
    elif ps <= 10:
        ps_score = 0.6
    elif ps <= 20:
        ps_score = 0.4
    else:
        ps_score = 0.2

    fcf_bonus = 0.1 if fcf > 0 else 0.0
    # Debt/Equity: 0–0.5 →1.0, 0.5–1 →0.8, 1–2 →0.5, >2 →0.2
    if de <= 0:
        de_score = 0.8
    elif de <= 0.5:
        de_score = 1.0
    elif de <= 1.0:
        de_score = 0.8
    elif de <= 2.0:
        de_score = 0.5
    else:
        de_score = 0.2

    # viktning inom growth
    # cagr 35%, ps 25%, gm 15%, nm 15%, de 10% + fcf bonus
    base = (0.35*cagr_score + 0.25*ps_score + 0.15*gm_score + 0.15*nm_score + 0.10*de_score)
    score = min(1.0, base + fcf_bonus)
    return round(score * 100.0, 1)

def _score_dividend(row: pd.Series) -> float:
    """
    DividendScore 0–100.
    Faktorer:
      + Direktavkastning (optimalt 3–7%)
      + FCF (TTM) positivt (säkerhet)
      + Debt/Equity lågt bättre
      + Nettomarginal positiv
      + Payout-proxy: om utdelning >0 men FCF <=0 → kraftig sänkning
    """
    dy = safe_float(row.get("Direktavkastning (%)", 0.0))
    fcf = safe_float(row.get("FCF (TTM)", 0.0))
    de = safe_float(row.get("Debt/Equity", 0.0))
    nm = safe_float(row.get("Nettomarginal (%)", 0.0))

    # yield sweet spot ~ 3–7%
    if dy <= 0:
        y_score = 0.0
    elif dy < 2:
        y_score = 0.4
    elif dy <= 4:
        y_score = 0.9
    elif dy <= 7:
        y_score = 1.0
    elif dy <= 10:
        y_score = 0.7
    else:
        y_score = 0.3  # väldigt hög yield = risk

    fcf_score = 1.0 if fcf > 0 else 0.2
    # D/E lägre bättre
    if de <= 0.5:
        de_score = 1.0
    elif de <= 1.0:
        de_score = 0.8
    elif de <= 2.0:
        de_score = 0.5
    else:
        de_score = 0.2

    nm_score = 1.0 if nm > 0 else 0.3

    # proxy penalty om utdelning men negativ FCF
    penalty = 0.25 if (dy > 0 and fcf <= 0) else 0.0

    base = (0.45*y_score + 0.25*fcf_score + 0.20*de_score + 0.10*nm_score)
    score = max(0.0, min(1.0, base - penalty))
    return round(score * 100.0, 1)

def compute_scores_and_label(row: pd.Series) -> Tuple[float, float, str]:
    """
    Returnerar (GrowthScore, DividendScore, Värdering-label).
    Värdering baseras på uppsida mot valda riktkursen (standard: 1 år) och score.
    """
    sector = str(row.get("Sektor", "") or "")
    weights = _sector_weights(sector)

    g = _score_growth(row)
    d = _score_dividend(row)

    # Värderingsetikett – basera på uppsida mot 1-års riktkurs (om finns)
    px = safe_float(row.get("Aktuell kurs", 0.0))
    tgt = safe_float(row.get("Riktkurs om 1 år", 0.0))
    upside_pct = 0.0
    if px > 0 and tgt > 0:
        upside_pct = (tgt - px) / px * 100.0

    # Sammansatt besluts-score (sektorviktning)
    composite = weights["growth"] * g + weights["dividend"] * d  # 0..100

    # Trösklar: kombinerar uppsida + composite
    # - Sälj: uppsida < -20% ELLER (composite<40 och uppsida<0)
    # - Trimma: uppsida < -5% och composite < 60
    # - Köp: uppsida > 15% och composite >= 65
    # - Bra: uppsida > 5% och composite >= 55
    # - Annars: Fair/Behåll
    label = "Fair/Behåll"
    if upside_pct < -20 or (composite < 40 and upside_pct < 0):
        label = "Sälj"
    elif upside_pct < -5 and composite < 60:
        label = "Trimma"
    elif upside_pct > 15 and composite >= 65:
        label = "Köp"
    elif upside_pct > 5 and composite >= 55:
        label = "Bra"

    return round(g,1), round(d,1), label

# -----------------------------------------------------------------------------
# Huvuduppdatering över DataFrame
# -----------------------------------------------------------------------------

def update_calculations(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Uppdaterar härledda fält utan att röra manuella nyckelfält.
    - P/S-snitt (om saknas)
    - Omsättning om 2 & 3 år (från Omsättning nästa år + clampad CAGR)
    - Riktkurser (idag, 1, 2, 3) från P/S-snitt & Utestående aktier
    - Direktavkastning (%) från Årlig utdelning / Aktuell kurs
    - Market Cap (nu) human och Risklabel (endast features – risklabel lagras i 'Värdering' ej; men vi använder i vy)
    - GrowthScore, DividendScore, Värdering
    """
    if df is None or df.empty:
        return df

    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Precompute P/S-snitt
    maybe_ps_avg = df.get("P/S-snitt", pd.Series([0.0]*len(df)))
    if "P/S-snitt" not in df.columns or (maybe_ps_avg <= 0).all():
        df["P/S-snitt"] = df.apply(ps_average_from_quarters, axis=1)

    # Direktavkastning
    def _div_yield(row):
        price = safe_float(row.get("Aktuell kurs", 0.0))
        div   = safe_float(row.get("Årlig utdelning", 0.0))
        if price > 0 and div > 0:
            return round(div / price * 100.0, 2)
        return 0.0
    df["Direktavkastning (%)"] = df.apply(_div_yield, axis=1)

    # Omsättning om 2 & 3 år från "Omsättning nästa år" med CAGR clamp
    def _future_sales(row):
        cagr = clamp_cagr(safe_float(row.get("CAGR 5 år (%)", 0.0)))
        g = cagr / 100.0
        nxt = safe_float(row.get("Omsättning nästa år", 0.0))
        if nxt > 0:
            y2 = round(nxt * (1.0 + g), 2)
            y3 = round(nxt * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev. befintliga om fyllda
            y2 = safe_float(row.get("Omsättning om 2 år", 0.0))
            y3 = safe_float(row.get("Omsättning om 3 år", 0.0))
        return pd.Series({"Omsättning om 2 år": y2, "Omsättning om 3 år": y3})
    fut = df.apply(_future_sales, axis=1)
    for c in ["Omsättning om 2 år", "Omsättning om 3 år"]:
        df[c] = fut[c]

    # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
    def _targets(row):
        ps_avg = safe_float(row.get("P/S-snitt", 0.0))
        shares_m = safe_float(row.get("Utestående aktier", 0.0))  # miljoner
        if ps_avg <= 0 or shares_m <= 0:
            return pd.Series({
                "Riktkurs idag": 0.0, "Riktkurs om 1 år": 0.0,
                "Riktkurs om 2 år": 0.0, "Riktkurs om 3 år": 0.0
            })
        s0 = safe_float(row.get("Omsättning idag", 0.0))
        s1 = safe_float(row.get("Omsättning nästa år", 0.0))
        s2 = safe_float(row.get("Omsättning om 2 år", 0.0))
        s3 = safe_float(row.get("Omsättning om 3 år", 0.0))
        return pd.Series({
            "Riktkurs idag":    round((s0 * ps_avg) / shares_m, 2) if s0 > 0 else 0.0,
            "Riktkurs om 1 år": round((s1 * ps_avg) / shares_m, 2) if s1 > 0 else 0.0,
            "Riktkurs om 2 år": round((s2 * ps_avg) / shares_m, 2) if s2 > 0 else 0.0,
            "Riktkurs om 3 år": round((s3 * ps_avg) / shares_m, 2) if s3 > 0 else 0.0,
        })
    tg = df.apply(_targets, axis=1)
    for c in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
        df[c] = tg[c]

    # Scores & etiketter
    def _scores_label(row):
        g, d, lab = compute_scores_and_label(row)
        return pd.Series({"GrowthScore": g, "DividendScore": d, "Värdering": lab})
    sc = df.apply(_scores_label, axis=1)
    for c in ["GrowthScore","DividendScore","Värdering"]:
        df[c] = sc[c]

    return df
