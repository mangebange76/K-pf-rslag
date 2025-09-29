# stockapp/scoring.py
# -*- coding: utf-8 -*-
"""
Poängsättning, risklabel och säljvakt.

Publika funktioner:
- compute_scores(df, user_rates=None) -> pd.DataFrame
- label_valuation(row) -> str
- sell_watch(df, user_rates=None, target_portfolio_risk=None) -> pd.DataFrame

Kräver inga Streamlit-importer. Robust mot saknade kolumner.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import math

import numpy as np
import pandas as pd

# Vi använder växelkurs-funktionen om den finns
try:
    from .rates import hamta_valutakurs
except Exception:
    def hamta_valutakurs(v: str, rates: Dict[str, float]) -> float:
        return 1.0


# ----------------------------- Hjälpmetoder ----------------------------------

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _norm_higher_better(x: float, good: float, bad: float, cap: Tuple[float, float] = None) -> float:
    """
    Normalisera till 0..1 där högre är bättre.
    'good' och 'bad' är ungefärliga riktmärken (inte hårda max/min).
    """
    if cap:
        x = _clamp(x, cap[0], cap[1])
    if good == bad:
        return 0.5
    # linjärt: bad -> 0, good -> 1
    return _clamp((x - bad) / (good - bad), 0.0, 1.0)

def _norm_lower_better(x: float, good: float, bad: float, cap: Tuple[float, float] = None) -> float:
    """
    Normalisera till 0..1 där lägre är bättre (t.ex. skuldsättning, P/S).
    """
    if cap:
        x = _clamp(x, cap[0], cap[1])
    if good == bad:
        return 0.5
    return _clamp((bad - x) / (bad - good), 0.0, 1.0)

def _mk_mcap_sek(row: pd.Series, user_rates: Optional[Dict[str, float]]) -> float:
    px = _safe_float(row.get("Aktuell kurs", 0.0))
    shares_m = _safe_float(row.get("Utestående aktier", 0.0))  # i miljoner
    if px <= 0 or shares_m <= 0:
        return 0.0
    ccy = str(row.get("Valuta", "SEK")).upper()
    fx = hamta_valutakurs(ccy, user_rates or {"SEK": 1.0})
    try:
        return float(px * shares_m * 1_000_000.0 * fx)
    except Exception:
        return 0.0

def _risk_bucket(mcap_sek: float) -> str:
    # Grova nivåer i SEK (kan justeras)
    if mcap_sek <= 0:
        return "Okänd"
    if mcap_sek < 3e9:
        return "Microcap"
    if mcap_sek < 20e9:
        return "Smallcap"
    if mcap_sek < 100e9:
        return "Midcap"
    if mcap_sek < 500e9:
        return "Largecap"
    return "Megacap"

def label_valuation(row: pd.Series) -> str:
    """
    Värderingsetikett:
    1) Finns 'Uppsida (%)' → basera på den
    2) Annars jämför P/S mot P/S-snitt (TTM/kvartalssnitt)
    """
    up = row.get("Uppsida (%)", None)
    try:
        if up is not None and not pd.isna(up):
            up = float(up)
            if up >= 40:
                return "Billig"
            if up >= 10:
                return "Rimlig"
            if up >= -10:
                return "Dyr"
            return "Övervärderad"
    except Exception:
        pass

    ps = _safe_float(row.get("P/S", 0.0))
    ps_avg = _safe_float(row.get("P/S-snitt", 0.0))
    if ps <= 0 or ps_avg <= 0:
        return "Okänt"
    ratio = ps / ps_avg
    if ratio <= 0.7:
        return "Billig"
    if ratio <= 1.2:
        return "Rimlig"
    if ratio <= 1.6:
        return "Dyr"
    return "Övervärderad"


# ----------------------------- Poängsättning ---------------------------------

def _score_valuation(row: pd.Series) -> float:
    """Lägre multiplar → högre score. Använder det som råkar finnas."""
    parts = []
    # P/S vs P/S-snitt (ratio <1 är bra)
    ps = _safe_float(row.get("P/S", 0.0))
    ps_avg = _safe_float(row.get("P/S-snitt", 0.0))
    if ps > 0 and ps_avg > 0:
        ratio = ps / ps_avg
        parts.append(_norm_lower_better(ratio, good=0.8, bad=1.6, cap=(0.2, 3.0)))  # 0.8~billigare än snitt; 1.6~dyrt

    # EV/EBITDA (lägre är bättre) om finns
    ev_ebitda = row.get("EV/EBITDA", None)
    if ev_ebitda is not None:
        parts.append(_norm_lower_better(_safe_float(ev_ebitda), good=8, bad=20, cap=(1, 60)))

    # P/E (om finns)
    pe = row.get("P/E", None)
    if pe is not None and _safe_float(pe) > 0:
        parts.append(_norm_lower_better(_safe_float(pe), good=15, bad=40, cap=(1, 120)))

    # Pris/FCF (om finns)
    p_fcf = row.get("P/FCF", None)
    if p_fcf is not None and _safe_float(p_fcf) > 0:
        parts.append(_norm_lower_better(_safe_float(p_fcf), good=15, bad=40, cap=(1, 120)))

    if not parts:
        return 0.5
    return float(np.mean(parts))


def _score_growth(row: pd.Series) -> float:
    """Tillväxt: CAGR, och/eller 'Omsättning nästa år' vs 'Omsättning idag'."""
    parts = []

    # CAGR 5 år (%)
    cagr = _safe_float(row.get("CAGR 5 år (%)", 0.0))
    if cagr != 0.0:
        parts.append(_norm_higher_better(cagr, good=40, bad=0, cap=(-20, 120)))

    # Nästa års omsättning / dagens
    now_rev = _safe_float(row.get("Omsättning idag", 0.0))
    nxt_rev = _safe_float(row.get("Omsättning nästa år", 0.0))
    if now_rev > 0 and nxt_rev > 0:
        growth = (nxt_rev / now_rev - 1.0) * 100.0
        parts.append(_norm_higher_better(growth, good=30, bad=0, cap=(-20, 100)))

    if not parts:
        return 0.5
    return float(np.mean(parts))


def _score_quality(row: pd.Series) -> float:
    """Lönsamhet/finansiell kvalitet: marginaler, ROIC/ROE, räntetäckning, D/E (lägre)."""
    parts = []

    # Marginaler
    gm = row.get("Bruttomarginal (%)", None)
    if gm is not None:
        parts.append(_norm_higher_better(_safe_float(gm), good=60, bad=20, cap=(0, 90)))

    nm = row.get("Netto-marginal (%)", None)
    if nm is not None:
        parts.append(_norm_higher_better(_safe_float(nm), good=20, bad=0, cap=(-50, 50)))

    # Avkastning
    roic = row.get("ROIC (%)", None)
    if roic is not None:
        parts.append(_norm_higher_better(_safe_float(roic), good=15, bad=3, cap=(0, 40)))

    roe = row.get("ROE (%)", None)
    if roe is not None:
        parts.append(_norm_higher_better(_safe_float(roe), good=18, bad=5, cap=(0, 60)))

    # Räntetäckningsgrad
    ic = row.get("Räntetäckningsgrad", None)
    if ic is not None:
        parts.append(_norm_higher_better(_safe_float(ic), good=8, bad=1, cap=(0, 40)))

    # Skuldsättning (lägre bättre): Debt/Equity
    de = row.get("Debt/Equity", None)
    if de is not None and _safe_float(de) > 0:
        parts.append(_norm_lower_better(_safe_float(de), good=0.5, bad=2.5, cap=(0, 6)))

    if not parts:
        return 0.5
    return float(np.mean(parts))


def _score_income(row: pd.Series) -> float:
    """Utdelningsprofil: yield och hållbarhet (payout, FCF, historik)."""
    parts = []

    # Direktavkastning (%)
    dy = row.get("Dividend Yield (%)", None)
    if dy is not None:
        # Högre är bättre (till en gräns), men över 10% kan vara risk
        parts.append(_norm_higher_better(_safe_float(dy), good=6, bad=1, cap=(0, 12)))

    # Payout på FCF (%), lägre är bättre
    pfcf = row.get("Payout FCF (%)", None)
    if pfcf is not None and _safe_float(pfcf) > 0:
        # 30% bra, 80% dåligt
        parts.append(_norm_lower_better(_safe_float(pfcf), good=30, bad=80, cap=(0, 150)))

    # Utdelningshistorik (år)
    yrs = row.get("Utdelningsår (streak)", None)
    if yrs is not None:
        parts.append(_norm_higher_better(_safe_float(yrs), good=15, bad=0, cap=(0, 50)))

    if not parts:
        return 0.5
    return float(np.mean(parts))


def compute_scores(df: pd.DataFrame, user_rates: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Beräknar Risklabel, Mcap (SEK) och samtliga del-/totalscore.
    Lämnar befintliga kolumner orörda och returnerar nytt df (copy).
    """
    work = df.copy()

    # Beräkna Mcap (SEK) och Risklabel
    mcap_sek = []
    risklabel = []
    for _, r in work.iterrows():
        mc = _mk_mcap_sek(r, user_rates)
        mcap_sek.append(mc)
        risklabel.append(_risk_bucket(mc))
    work["Mcap (SEK)"] = mcap_sek
    work["Risklabel"] = risklabel

    # Delpoäng
    work["Score_Valuation"] = work.apply(_score_valuation, axis=1)
    work["Score_Growth"]    = work.apply(_score_growth, axis=1)
    work["Score_Quality"]   = work.apply(_score_quality, axis=1)
    work["Score_Income"]    = work.apply(_score_income, axis=1)

    # Totalpoäng: viktad mix. Här default = lika vikt, men du kan anpassa i vy.
    # Om något saknas (0 bidrag) normaliseras vikten genom att snittet tas på icke-NaN.
    totals = []
    for _, r in work.iterrows():
        parts = [
            _safe_float(r.get("Score_Valuation", np.nan), np.nan),
            _safe_float(r.get("Score_Growth",    np.nan), np.nan),
            _safe_float(r.get("Score_Quality",   np.nan), np.nan),
            _safe_float(r.get("Score_Income",    np.nan), np.nan),
        ]
        arr = [p for p in parts if not (p is None or (isinstance(p, float) and math.isnan(p)))]
        totals.append(float(np.mean(arr)) if arr else 0.5)
    work["Score_Total"] = totals

    # Värderingsetikett
    work["Värderingslabel"] = work.apply(label_valuation, axis=1)

    return work


# ----------------------------- Säljvakt --------------------------------------

def _position_share_and_value(row: pd.Series, user_rates: Optional[Dict[str, float]]) -> Tuple[float, float]:
    """
    Returnerar (position_vikt_procent, position_värde_SEK).
    """
    antal = _safe_float(row.get("Antal aktier", 0.0))
    if antal <= 0:
        return 0.0, 0.0
    px = _safe_float(row.get("Aktuell kurs", 0.0))
    ccy = str(row.get("Valuta", "SEK")).upper()
    fx = hamta_valutakurs(ccy, user_rates or {"SEK": 1.0})
    value = antal * px * fx
    # Portfolio total beräknas i vy vanligtvis; här gör vi lokalt om kolumn finns
    # Alternativt får vy skicka in total och vi normaliserar där.
    return 0.0, value  # vikt sätts i vy; här bara SEK-värde


def sell_watch(df: pd.DataFrame,
               user_rates: Optional[Dict[str, float]] = None,
               target_portfolio_risk: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Bygger en tabell med rekommendationer: Behåll / Trimma / Sälj.
    Heuristik:
      - Om Värderingslabel = 'Övervärderad' ELLER Score_Valuation < 0.35 → kandidat
      - Stärk signal om Score_Total < 0.45
      - Storleksvikt (portföljandel) förstärker trim/sälj
    """
    port = df.copy()
    port = port[port.get("Antal aktier", 0) > 0].copy()
    if port.empty:
        return pd.DataFrame(columns=[
            "Ticker","Bolagsnamn","Värderingslabel","Score_Total","Score_Valuation",
            "Rekommendation","Motivering","Trim-förslag (SEK)","Trim-antal"
        ])

    # Beräkna portföljvärde SEK
    def _row_value_sek(r):
        px = _safe_float(r.get("Aktuell kurs", 0.0))
        n = _safe_float(r.get("Antal aktier", 0.0))
        c = str(r.get("Valuta", "SEK")).upper()
        fx = hamta_valutakurs(c, user_rates or {"SEK": 1.0})
        return n * px * fx

    port["Värde (SEK)"] = port.apply(_row_value_sek, axis=1)
    total = float(port["Värde (SEK)"].sum()) or 1.0
    port["Andel (%)"] = port["Värde (SEK)"] / total * 100.0

    # Baslinje för storlek: risklabeln styr hur hög typisk max-andel bör vara (enkel heuristik)
    default_risk_caps = {
        "Microcap": 2.0,
        "Smallcap": 4.0,
        "Midcap": 7.0,
        "Largecap": 12.0,
        "Megacap": 18.0,
        "Okänd": 4.0,
    }
    risk_cap = target_portfolio_risk or default_risk_caps

    recs = []
    for _, r in port.iterrows():
        label = str(r.get("Värderingslabel") or label_valuation(r))
        s_val = _safe_float(r.get("Score_Valuation", 0.5))
        s_tot = _safe_float(r.get("Score_Total", 0.5))
        share = _safe_float(r.get("Andel (%)", 0.0))
        bucket = str(r.get("Risklabel") or "Okänd")
        cap_pct = float(risk_cap.get(bucket, 4.0))

        # Heuristik
        over_size = share > cap_pct
        very_over = share > cap_pct * 1.5

        action = "Behåll"
        reason = []

        if label == "Övervärderad" or s_val < 0.35:
            if very_over or s_tot < 0.40:
                action = "Sälj"
            else:
                action = "Trimma"
            if label == "Övervärderad":
                reason.append("övervärderad")
            if s_val < 0.35:
                reason.append("svag värderingsscore")
            if s_tot < 0.40:
                reason.append("svag totalpoäng")
            if over_size:
                reason.append("för stor portföljandel")

        # Trimförslag: sikta mot cap_pct (eller cap_pct*0.8 om Sälj)
        trim_sek = 0.0
        trim_shares = 0.0
        if action in ("Trimma", "Sälj") and share > cap_pct:
            px = _safe_float(r.get("Aktuell kurs", 0.0))
            fx = hamta_valutakurs(str(r.get("Valuta","SEK")).upper(), user_rates or {"SEK": 1.0})
            target_pct = cap_pct * (0.8 if action == "Sälj" else 1.0)
            target_value = total * (target_pct / 100.0)
            excess = r["Värde (SEK)"] - target_value
            if excess > 0 and px * fx > 0:
                trim_sek = excess
                trim_shares = excess / (px * fx)

        recs.append({
            "Ticker": r.get("Ticker",""),
            "Bolagsnamn": r.get("Bolagsnamn",""),
            "Värderingslabel": label,
            "Score_Total": round(s_tot, 3),
            "Score_Valuation": round(s_val, 3),
            "Andel (%)": round(share, 2),
            "Risklabel": bucket,
            "Rekommendation": action,
            "Motivering": ", ".join(reason) if reason else "",
            "Trim-förslag (SEK)": round(trim_sek, 0),
            "Trim-antal": int(round(trim_shares, 0)),
        })

    out = pd.DataFrame(recs)
    # Prioritera starkast åtgärd och största övervikt
    order = {"Sälj": 0, "Trimma": 1, "Behåll": 2}
    out["rank_action"] = out["Rekommendation"].map(order).fillna(3)
    out = out.sort_values(by=["rank_action","Andel (%)","Score_Total"], ascending=[True, False, True]).drop(columns=["rank_action"])
    return out
