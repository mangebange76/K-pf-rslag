# stockapp/finance.py
# -*- coding: utf-8 -*-
"""
Finans-/beräkningshjälp för appen.

Publika funktioner (stabila API:n för övriga moduler):
- ps_snitt_from_row(row) -> float
- compute_future_revenues(row) -> (rev_y2, rev_y3)
- compute_price_targets(row, ps_snitt) -> dict med "Riktkurs ..." i radens valuta
- compute_market_cap(row) -> float (i radens valuta)
- marketcap_to_str(x) -> str (t.ex. '4.25 T', '512.3 B', '23.1 M')
- market_cap_usd(row, user_rates) -> float (för risklabel)
- risk_label_from_mcap_usd(mcap_usd) -> str
- valuation_label(row, prefer_target='Riktkurs om 1 år') -> str
- sell_watch_signal(row, prefer_target='Riktkurs om 1 år') -> Optional[str]
- enrich_for_investing(df, user_rates) -> DataFrame (adderar: P/S-snitt, MCAP, MCAP (str), Risk, Valuation, SellWatch)
- uppdatera_berakningar(df, user_rates) -> DataFrame (replikerar din tidigare “uppdatera_berakningar” med små förbättringar)

Antaganden:
- Kolumnnamn följer appens svenska schema.
- “Utestående aktier” är i MILJONER (som i din befintliga app).
- “Aktuell kurs” och P/S-relaterat är i bolagets prisvaluta.
- user_rates: dict med växelkurs -> SEK (ex. {'USD': 10.5, 'EUR': 11.2, 'SEK': 1.0, ...})
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Bas-helpers
# ------------------------------------------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _pos(x: float) -> bool:
    try:
        return float(x) > 0.0
    except Exception:
        return False

# ------------------------------------------------------------
# P/S-snitt och framtidsomsättning
# ------------------------------------------------------------

def ps_snitt_from_row(row: pd.Series) -> float:
    """Snitt av positiva P/S Q1–Q4. 0 om inga positiva."""
    q = []
    for k in ("P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"):
        if k in row:
            v = _safe_float(row.get(k, 0.0))
            if v > 0:
                q.append(v)
    return round(float(np.mean(q)), 2) if q else 0.0

def compute_future_revenues(row: pd.Series) -> Tuple[float, float]:
    """
    Beräknar 'Omsättning om 2 år' och 'Omsättning om 3 år' från 'Omsättning nästa år'
    med 'CAGR 5 år (%)' clamp:
      >100% -> 50%, <0% -> 2%.
    """
    cagr = _safe_float(row.get("CAGR 5 år (%)", 0.0))
    if cagr > 100.0:
        cagr = 50.0
    if cagr < 0.0:
        cagr = 2.0
    g = cagr / 100.0
    next_rev = _safe_float(row.get("Omsättning nästa år", 0.0))
    if next_rev <= 0:
        # behåll ev. befintliga värden
        y2 = _safe_float(row.get("Omsättning om 2 år", 0.0))
        y3 = _safe_float(row.get("Omsättning om 3 år", 0.0))
        return (y2, y3)
    y2 = round(next_rev * (1.0 + g), 2)
    y3 = round(next_rev * ((1.0 + g) ** 2), 2)
    return (y2, y3)

# ------------------------------------------------------------
# Riktkurser
# ------------------------------------------------------------

def compute_price_targets(row: pd.Series, ps_snitt: float) -> Dict[str, float]:
    """
    Räknar riktkurser = (Omsättning * P/S-snitt) / Utestående aktier.
    Omsättning är i MILJONER i radens valuta; Utestående aktier är i MILJONER.
    Resultatet blir i samma valuta som 'Aktuell kurs'.
    """
    out = {"Riktkurs idag": 0.0, "Riktkurs om 1 år": 0.0, "Riktkurs om 2 år": 0.0, "Riktkurs om 3 år": 0.0}
    shares_mn = _safe_float(row.get("Utestående aktier", 0.0))  # miljoner
    if shares_mn <= 0 or ps_snitt <= 0:
        return out

    def _rk(rev_mn: float) -> float:
        if rev_mn <= 0:
            return 0.0
        # (miljoner * P/S) / miljoner = prisenhet
        return round((rev_mn * ps_snitt) / shares_mn, 2)

    out["Riktkurs idag"]    = _rk(_safe_float(row.get("Omsättning idag", 0.0)))
    out["Riktkurs om 1 år"] = _rk(_safe_float(row.get("Omsättning nästa år", 0.0)))
    out["Riktkurs om 2 år"] = _rk(_safe_float(row.get("Omsättning om 2 år", 0.0)))
    out["Riktkurs om 3 år"] = _rk(_safe_float(row.get("Omsättning om 3 år", 0.0)))
    return out

# ------------------------------------------------------------
# Market cap & formatering
# ------------------------------------------------------------

def compute_market_cap(row: pd.Series) -> float:
    """
    Market cap i radens prisvaluta. Kräver Aktuell kurs > 0 och Utestående aktier (miljoner) > 0.
    """
    px = _safe_float(row.get("Aktuell kurs", 0.0))
    sh_mn = _safe_float(row.get("Utestående aktier", 0.0))
    if px > 0 and sh_mn > 0:
        return float(px * sh_mn * 1e6)
    return 0.0

def marketcap_to_str(x: float) -> str:
    """Snygg text för market cap (T, B, M, K)."""
    try:
        n = float(x)
    except Exception:
        return "-"
    if n <= 0:
        return "-"
    T = 1_000_000_000_000.0
    B = 1_000_000_000.0
    M = 1_000_000.0
    K = 1_000.0
    if n >= T:
        return f"{n / T:.2f} T"
    if n >= B:
        return f"{n / B:.2f} B"
    if n >= M:
        return f"{n / M:.2f} M"
    if n >= K:
        return f"{n / K:.2f} K"
    return f"{n:.0f}"

def _cross_ccy_amount_to_usd(amount_in_ccy: float, ccy: str, user_rates: Optional[Dict[str, float]]) -> float:
    """
    Korsa via SEK: ccy->SEK, USD->SEK → amount(USD) = amount(ccy) * (ccy/SEK) / (USD/SEK) = amount * (ccy_SEK / USD_SEK)
    user_rates förväntas vara SEK-quotes (t.ex. {"USD": 10.5, "EUR": 11.2, "SEK": 1.0, ...})
    """
    try:
        if not user_rates or not ccy:
            return amount_in_ccy  # bästa gissning
        c = ccy.upper()
        usd_sek = float(user_rates.get("USD", 0.0))
        ccy_sek = float(user_rates.get(c, 0.0))
        if usd_sek > 0 and ccy_sek > 0:
            return float(amount_in_ccy) * (ccy_sek / usd_sek)
        return amount_in_ccy
    except Exception:
        return amount_in_ccy

def market_cap_usd(row: pd.Series, user_rates: Optional[Dict[str, float]]) -> float:
    """
    Market cap i USD, korsad via SEK-kvoter.
    """
    mc_local = compute_market_cap(row)
    ccy = str(row.get("Valuta", "USD") or "USD").upper()
    return _cross_ccy_amount_to_usd(mc_local, ccy, user_rates)

# ------------------------------------------------------------
# Risklabel, värderingsetikett & säljvakt
# ------------------------------------------------------------

def risk_label_from_mcap_usd(mcap_usd: float) -> str:
    """
    Grov indelning enligt vanliga USA-cutoffs.
    """
    if mcap_usd <= 0:
        return "-"
    if mcap_usd < 300_000_000:                 # < $0.3B
        return "Microcap"
    if mcap_usd < 2_000_000_000:               # $0.3–2B
        return "Smallcap"
    if mcap_usd < 10_000_000_000:              # $2–10B
        return "Midcap"
    if mcap_usd < 200_000_000_000:             # $10–200B
        return "Largecap"
    return "Megacap"

def valuation_label(row: pd.Series, prefer_target: str = "Riktkurs om 1 år") -> str:
    """
    Enkel värderingsindikator:
    - Pris vs riktkurs (valfri kolumn).
    - P/S vs P/S-snitt som sekundär check.

    Returnerar en av: "Billig", "Fair", "Dyr", "Trimma", "Sälj" eller "-"
    """
    price = _safe_float(row.get("Aktuell kurs", 0.0))
    target = _safe_float(row.get(prefer_target, 0.0))
    ps = _safe_float(row.get("P/S", 0.0))
    ps_avg = ps_snitt_from_row(row)

    if price <= 0 or target <= 0:
        return "-"

    # över/under riktkurs
    prem = (price - target) / target  # +20% över = 0.20
    # relativ P/S
    ps_rel = (ps / ps_avg) if (ps > 0 and ps_avg > 0) else 1.0

    # trösklar (konservativt)
    if prem <= -0.20 and ps_rel <= 0.9:
        return "Billig"
    if -0.20 < prem < 0.20 and 0.9 <= ps_rel <= 1.1:
        return "Fair"
    if prem >= 0.50 and ps_rel >= 1.5:
        return "Sälj"
    if prem >= 0.25 and ps_rel >= 1.2:
        return "Trimma"
    return "Dyr" if prem > 0.0 else "Fair"

def sell_watch_signal(row: pd.Series, prefer_target: str = "Riktkurs om 1 år") -> Optional[str]:
    """
    Returnerar None eller en kort signal: "Trimma" / "Sälj".
    """
    label = valuation_label(row, prefer_target=prefer_target)
    if label in ("Trimma", "Sälj"):
        return label
    return None

# ------------------------------------------------------------
# Enricher och full re-calc
# ------------------------------------------------------------

def enrich_for_investing(df: pd.DataFrame, user_rates: Optional[Dict[str, float]]) -> pd.DataFrame:
    """
    Lägger till:
      - P/S-snitt
      - Market cap (valuta)
      - Market cap (str)
      - Risk (USD-baserad)
      - Valuation label
      - SellWatch
    Ändrar inte befintliga kolumner, lägger bara nya/fyller om de finns.
    """
    work = df.copy()
    # P/S-snitt
    work["P/S-snitt"] = work.apply(ps_snitt_from_row, axis=1)
    # Market cap (valuta)
    work["_MarketCapLocal"] = work.apply(compute_market_cap, axis=1)
    work["MarketCap (str)"] = work["_MarketCapLocal"].apply(marketcap_to_str)
    # Risk
    work["_MC_USD"] = work.apply(lambda r: market_cap_usd(r, user_rates), axis=1)
    work["Risk"] = work["_MC_USD"].apply(risk_label_from_mcap_usd)
    # Valuation & säljvakt
    work["Valuation"] = work.apply(lambda r: valuation_label(r, "Riktkurs om 1 år"), axis=1)
    work["SellWatch"] = work.apply(lambda r: sell_watch_signal(r, "Riktkurs om 1 år"), axis=1)
    return work

def uppdatera_berakningar(df: pd.DataFrame, user_rates: Optional[Dict[str, float]]) -> pd.DataFrame:
    """
    Replikerar din tidigare “uppdatera_berakningar” med några små förbättringar:
      - Räknar P/S-snitt
      - Räknar 'Omsättning om 2 år'/'Omsättning om 3 år'
      - Räknar riktkurser (idag/1/2/3)
      - Marknadsdata (MarketCap, Risklabel, Valuation) för analys/förslag
    """
    work = df.copy()
    # P/S-snitt
    work["P/S-snitt"] = work.apply(ps_snitt_from_row, axis=1)
    # Framtida omsättning
    y2_list = []
    y3_list = []
    for _, r in work.iterrows():
        y2, y3 = compute_future_revenues(r)
        y2_list.append(y2); y3_list.append(y3)
    work["Omsättning om 2 år"] = y2_list
    work["Omsättning om 3 år"] = y3_list
    # Riktkurser
    rk_today, rk_1y, rk_2y, rk_3y = [], [], [], []
    for _, r in work.iterrows():
        rk = compute_price_targets(r, _safe_float(r.get("P/S-snitt", 0.0)))
        rk_today.append(rk["Riktkurs idag"])
        rk_1y.append(rk["Riktkurs om 1 år"])
        rk_2y.append(rk["Riktkurs om 2 år"])
        rk_3y.append(rk["Riktkurs om 3 år"])
    work["Riktkurs idag"]    = rk_today
    work["Riktkurs om 1 år"] = rk_1y
    work["Riktkurs om 2 år"] = rk_2y
    work["Riktkurs om 3 år"] = rk_3y
    # Marknad & risk
    work["_MarketCapLocal"] = work.apply(compute_market_cap, axis=1)
    work["MarketCap (str)"] = work["_MarketCapLocal"].apply(marketcap_to_str)
    work["_MC_USD"] = work.apply(lambda r: market_cap_usd(r, user_rates), axis=1)
    work["Risk"] = work["_MC_USD"].apply(risk_label_from_mcap_usd)
    # Valuation & säljvakt
    work["Valuation"] = work.apply(lambda r: valuation_label(r, "Riktkurs om 1 år"), axis=1)
    work["SellWatch"] = work.apply(lambda r: sell_watch_signal(r, "Riktkurs om 1 år"), axis=1)
    return work
