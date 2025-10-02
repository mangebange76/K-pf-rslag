# stockapp/invest.py
# -*- coding: utf-8 -*-
"""
Investeringsförslag – sektorviktad scoring, bläddring 1/X, expander med nyckeltal.
Kräver inga fetch-anrop; läser bara från df och visar resultat.

Publikt API:
    visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st

from .utils import (
    safe_float,
    format_large_number,
    risk_label_from_mcap,
)

# ---------------------------
# Hjälpare: kolumnalias
# ---------------------------
ALIAS = {
    "price": ["Kurs", "Aktuell kurs"],
    "ps_ttm": ["P/S", "PS", "P/S (TTM)"],
    "ps_q1": ["P/S Q1"],
    "ps_q2": ["P/S Q2"],
    "ps_q3": ["P/S Q3"],
    "ps_q4": ["P/S Q4"],
    "ps_avg": ["P/S-snitt", "P/S-snitt (Q1..Q4)"],
    "revenue_now": ["Omsättning i år (est.)", "Omsättning idag"],
    "revenue_next": ["Omsättning nästa år (est.)", "Omsättning nästa år"],
    "mcap": ["Market Cap", "Market Cap (valuta)", "Market Cap (SEK)"],
    "dy": ["Dividend Yield (%)", "Direct Yield (%)", "Utdelning Yield (%)"],
    "payout_cf": ["Payout Ratio CF (%)"],
    "de_ratio": ["Debt/Equity"],
    "gross_margin": ["Bruttomarginal (%)"],
    "net_margin": ["Nettomarginal (%)"],
    "fcf": ["FCF (M)", "Free Cash Flow (M)"],
    "cash": ["Kassa (M)", "Cash (M)"],
    "runway": ["Runway (kvartal)"],
    "ev_ebitda": ["EV/EBITDA"],
    "sector": ["Sektor", "Sector"],
    "industry": ["Industri", "Industry"],
    "target_now": ["Riktkurs idag", "Riktkurs (nu)"],
    "target_1y": ["Riktkurs om 1 år"],
    "target_2y": ["Riktkurs om 2 år"],
    "target_3y": ["Riktkurs om 3 år"],
    "shares": ["Antal aktier"],
    "ticker": ["Ticker"],
    "name": ["Bolagsnamn", "Namn"],
    "currency": ["Valuta"],
    "risklabel": ["Risklabel"],
}

def _get(df_or_row, keys: List[str], default=np.nan):
    """Hämtar första förekomsten av kolumn bland alias."""
    for k in keys:
        if isinstance(df_or_row, pd.Series):
            if k in df_or_row.index:
                return df_or_row.get(k)
        else:
            if k in df_or_row.columns:
                return df_or_row[k]
    return default

def _get_num(row: pd.Series, keys: List[str]) -> float:
    return safe_float(_get(row, keys, np.nan), np.nan)

def _sector_of(row: pd.Series) -> str:
    s = _get(row, ALIAS["sector"], "")
    s = str(s).strip()
    return s if s not in ("", "nan", "None") else "Unknown"

def _mcap_of(row: pd.Series) -> float:
    # Prova Market Cap (valuta) först, annars SEK, annars generisk
    mc = _get_num(row, ["Market Cap (valuta)", "Market Cap (SEK)", "Market Cap"])
    return mc

def _price_of(row: pd.Series) -> float:
    return _get_num(row, ALIAS["price"])

def _ps_avg(row: pd.Series) -> float:
    v = _get_num(row, ALIAS["ps_avg"])
    if math.isnan(v):
        # fallback: snitta Q1..Q4
        q = [_get_num(row, ALIAS["ps_q1"]), _get_num(row, ALIAS["ps_q2"]),
             _get_num(row, ALIAS["ps_q3"]), _get_num(row, ALIAS["ps_q4"])]
        q = [x for x in q if not math.isnan(x)]
        if q:
            return float(np.mean(q))
    return v

def _upside(row: pd.Series) -> float:
    """Uppsida mot Riktkurs om 1 år om finns, annars 'Riktkurs idag'. Returnerar i %."""
    price = _price_of(row)
    if not price or math.isnan(price) or price <= 0:
        return np.nan
    tgt = _get_num(row, ALIAS["target_1y"])
    if math.isnan(tgt):
        tgt = _get_num(row, ALIAS["target_now"])
    if math.isnan(tgt) or tgt <= 0:
        return np.nan
    return (tgt / price - 1.0) * 100.0

def _is_dividend_stock(row: pd.Series) -> bool:
    dy = _get_num(row, ALIAS["dy"])
    return (not math.isnan(dy)) and (dy >= 2.0)  # enkel heuristik

# ---------------------------
# Sektorviktade vikter
# ---------------------------
DEFAULT_WEIGHTS = {
    # positivt om högre är bättre, negativt om lägre är bättre
    "ps": -0.25,           # lägre P/S bättre
    "gross_margin": +0.15,
    "net_margin": +0.15,
    "ev_ebitda": -0.2,     # lägre bättre
    "de_ratio": -0.1,      # lägre bättre
    "dy": +0.2,
    "payout_cf": -0.1,     # lägre bättre (över ~80% dåligt)
    "runway": +0.1,        # fler kvartal runway bättre
    "fcf": +0.15,          # större (positiv) FCF bättre
    "upside": +0.2,        # uppsida mot riktkurs
}

SECTOR_OVERRIDES: Dict[str, Dict[str, float]] = {
    # Tech/growth: betona marginaler, ev/ebitda, runway; mindre vikt på utdelning
    "Technology": {
        "ps": -0.25, "gross_margin": +0.2, "net_margin": +0.2, "ev_ebitda": -0.25, "de_ratio": -0.1, "runway": +0.15,
        "dy": +0.05, "payout_cf": -0.05, "fcf": +0.15, "upside": +0.2
    },
    "Communication Services": {
        "ps": -0.2, "gross_margin": +0.15, "net_margin": +0.2, "ev_ebitda": -0.2, "de_ratio": -0.1,
        "dy": +0.1, "payout_cf": -0.1, "fcf": +0.1, "upside": +0.2
    },
    # Defensiva utdelare: betona DY, payout_cf, skuldsättning
    "Utilities": {
        "ps": -0.1, "gross_margin": +0.05, "net_margin": +0.1, "ev_ebitda": -0.1, "de_ratio": -0.15,
        "dy": +0.3, "payout_cf": -0.2, "fcf": +0.1, "upside": +0.15
    },
    "Real Estate": {
        "ps": -0.1, "gross_margin": +0.05, "net_margin": +0.1, "ev_ebitda": -0.15, "de_ratio": -0.15,
        "dy": +0.3, "payout_cf": -0.2, "fcf": +0.1, "upside": +0.15
    },
    # Cykliska: ev/ebitda, nettomarginal, skuldsättning, DY viss vikt
    "Energy": {
        "ps": -0.15, "gross_margin": +0.05, "net_margin": +0.2, "ev_ebitda": -0.25, "de_ratio": -0.15,
        "dy": +0.15, "payout_cf": -0.1, "fcf": +0.2, "upside": +0.15
    },
    "Materials": {
        "ps": -0.15, "gross_margin": +0.1, "net_margin": +0.15, "ev_ebitda": -0.2, "de_ratio": -0.15,
        "dy": +0.15, "payout_cf": -0.1, "fcf": +0.2, "upside": +0.15
    },
    "Industrials": {
        "ps": -0.2, "gross_margin": +0.1, "net_margin": +0.15, "ev_ebitda": -0.2, "de_ratio": -0.15,
        "dy": +0.1, "payout_cf": -0.1, "fcf": +0.15, "upside": +0.15
    },
    # Consumer defensives: utdelning & marginaler
    "Consumer Staples": {
        "ps": -0.15, "gross_margin": +0.15, "net_margin": +0.15, "ev_ebitda": -0.15, "de_ratio": -0.1,
        "dy": +0.2, "payout_cf": -0.15, "fcf": +0.15, "upside": +0.15
    },
    # Financials: ev/ebitda mindre relevant, fokus lönsamhet, utdelningshållbarhet, D/E
    "Financials": {
        "ps": -0.1, "gross_margin": +0.05, "net_margin": +0.25, "ev_ebitda": -0.05, "de_ratio": -0.2,
        "dy": +0.2, "payout_cf": -0.15, "fcf": +0.1, "upside": +0.15
    },
    # Consumer Discretionary: mellanläge
    "Consumer Discretionary": {
        "ps": -0.2, "gross_margin": +0.15, "net_margin": +0.15, "ev_ebitda": -0.2, "de_ratio": -0.1,
        "dy": +0.1, "payout_cf": -0.1, "fcf": +0.15, "upside": +0.2
    },
    # Health Care: marginaler och skuldsättning, mindre utdelningsfokus
    "Health Care": {
        "ps": -0.2, "gross_margin": +0.2, "net_margin": +0.2, "ev_ebitda": -0.2, "de_ratio": -0.1,
        "dy": +0.1, "payout_cf": -0.1, "fcf": +0.15, "upside": +0.2
    },
}

def _weights_for_sector(sector: str) -> Dict[str, float]:
    return SECTOR_OVERRIDES.get(sector, DEFAULT_WEIGHTS)

# ---------------------------
# Normalisering / scoring
# ---------------------------
def _bounded_score(value: float, *, higher_is_better: bool, lo: float, hi: float) -> float:
    """
    Skalar value till [0..100] mellan lo..hi. Klipper utanför. Vänder skala vid lower-is-better.
    """
    if math.isnan(value):
        return np.nan
    x = float(value)
    if higher_is_better:
        s = 100.0 * (x - lo) / max(1e-9, (hi - lo))
    else:
        # lägre är bättre
        s = 100.0 * (hi - x) / max(1e-9, (hi - lo))
    return float(min(100.0, max(0.0, s)))

def _metric_scores(row: pd.Series, sector: str) -> Dict[str, float]:
    """
    Beräknar delpoäng 0..100 för varje nyckeltal (om data finns).
    Använder breda intervall (lo, hi) som är "sunda" för att undvika outliers.
    """
    scores: Dict[str, float] = {}
    w = _weights_for_sector(sector)

    ps = _ps_avg(row)
    if not math.isnan(ps):
        scores["ps"] = _bounded_score(ps, higher_is_better=False, lo=1.0, hi=30.0)

    gm = _get_num(row, ALIAS["gross_margin"])
    if not math.isnan(gm):
        scores["gross_margin"] = _bounded_score(gm, higher_is_better=True, lo=10.0, hi=80.0)

    nm = _get_num(row, ALIAS["net_margin"])
    if not math.isnan(nm):
        scores["net_margin"] = _bounded_score(nm, higher_is_better=True, lo=0.0, hi=40.0)

    ev = _get_num(row, ALIAS["ev_ebitda"])
    if not math.isnan(ev):
        scores["ev_ebitda"] = _bounded_score(ev, higher_is_better=False, lo=4.0, hi=25.0)

    de = _get_num(row, ALIAS["de_ratio"])
    if not math.isnan(de):
        scores["de_ratio"] = _bounded_score(de, higher_is_better=False, lo=0.0, hi=2.0)

    dy = _get_num(row, ALIAS["dy"])
    if not math.isnan(dy):
        scores["dy"] = _bounded_score(dy, higher_is_better=True, lo=0.5, hi=8.0)

    pr = _get_num(row, ALIAS["payout_cf"])
    if not math.isnan(pr):
        # payout över 80 blir lågt
        scores["payout_cf"] = _bounded_score(pr, higher_is_better=False, lo=0.0, hi=80.0)

    rw = _get_num(row, ALIAS["runway"])
    if not math.isnan(rw):
        scores["runway"] = _bounded_score(rw, higher_is_better=True, lo=2.0, hi=12.0)

    fcf = _get_num(row, ALIAS["fcf"])
    if not math.isnan(fcf):
        # FCF (M) – vi antar att > 0 upp till stora tal är bra; klipp mellan 0..50000
        scores["fcf"] = _bounded_score(fcf, higher_is_better=True, lo=0.0, hi=50000.0)

    up = _upside(row)
    if not math.isnan(up):
        # -20..+80% -> 0..100
        scores["upside"] = _bounded_score(up, higher_is_better=True, lo=-20.0, hi=80.0)

    return scores

def _coverage_weight(scores: Dict[str, float]) -> float:
    """Ju fler delpoäng (ej NaN), desto större vikt (0..1)."""
    if not scores:
        return 0.0
    present = [v for v in scores.values() if not math.isnan(v)]
    if not present:
        return 0.0
    ratio = len(present) / max(1, len(scores))
    # gör den lite mjuk: ^0.7
    return float(ratio ** 0.7)

def _weighted_total(scores: Dict[str, float], sector: str) -> Tuple[float, float]:
    """Returnerar (base_score, final_score) där final_score vägs med coverage."""
    w = _weights_for_sector(sector)
    base = 0.0
    wsum = 0.0
    used = 0
    for k, sc in scores.items():
        if math.isnan(sc):
            continue
        weight = w.get(k, 0.0)
        if abs(weight) < 1e-9:
            continue
        # normalisera vikt till positiv skala genom att teckna delscore vid lower-is-better
        # I _metric_scores är alla 0..100 "högre är bättre" redan, så vi kan
        # använda signen för att trycka upp eller ned
        base += sc * (1.0 if weight >= 0 else 1.0) * abs(weight)
        wsum += abs(weight)
        used += 1

    if wsum <= 0 or used == 0:
        return 0.0, 0.0

    base_score = 100.0 * (base / wsum) / 100.0  # skalar tillbaka till 0..100
    cov = _coverage_weight(scores)
    final = float(base_score * cov)
    return float(min(100.0, max(0.0, base_score))), float(min(100.0, max(0.0, final)))

def _label_for_score(sc: float) -> str:
    if sc >= 85:
        return "✅ Mycket bra (Köp)"
    if sc >= 70:
        return "👍 Bra (Köp / Bevaka)"
    if sc >= 55:
        return "🙂 Okej (Behåll)"
    if sc >= 40:
        return "⚠️ Något övervärderad (Trimma)"
    return "🛑 Övervärderad (Sälj)"

# ---------------------------
# Publikt UI
# ---------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float], *, page_size: int = 5) -> None:
    st.header("📈 Investeringsförslag")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    work = df.copy()

    # Risklabel (om saknas – beräkna grovt från Market Cap)
    if "Risklabel" not in work.columns:
        work["Risklabel"] = work.apply(lambda r: risk_label_from_mcap(_mcap_of(r)), axis=1)

    # Filtrering
    cols = st.columns([1, 1, 1, 1])
    # Sektor
    sektorer = ["Alla"]
    if "Sektor" in work.columns or "Sector" in work.columns:
        sektorer += sorted([s for s in _get(work, ALIAS["sector"]).dropna().astype(str).unique() if s and s != "nan"])
    val_sektor = cols[0].selectbox("Sektor", sektorer)

    # Risk
    risk_opts = ["Alla", "Mega", "Large", "Mid", "Small", "Micro", "Unknown"]
    val_risk = cols[1].selectbox("Risklabel", risk_opts)

    # Fokus
    val_fokus = cols[2].selectbox("Fokus", ["Alla", "Tillväxt", "Utdelare"])

    # Sida
    page_size = int(cols[3].number_input("Poster / sida", min_value=1, max_value=20, value=page_size))

    # Applicera filter
    if val_sektor != "Alla":
        work = work[_get(work, ALIAS["sector"]).astype(str) == val_sektor]
    if val_risk != "Alla":
        work = work["Risklabel"].astype(str) == val_risk if isinstance(work["Risklabel"], pd.Series) else work
        work = work[work["Risklabel"].astype(str) == val_risk]
    if val_fokus == "Tillväxt":
        # heuristik: privilegiera bolag med låg PS & hög uppsida
        w2 = []
        for _, r in work.iterrows():
            ps = _ps_avg(r)
            up = _upside(r)
            ok = (not math.isnan(ps) and ps <= 20) or (not math.isnan(up) and up >= 5)
            if ok:
                w2.append(True)
            else:
                w2.append(False)
        work = work[w2]
    elif val_fokus == "Utdelare":
        work = work[work.apply(_is_dividend_stock, axis=1)]

    if work.empty:
        st.info("Inga träffar efter filter.")
        return

    # Poängberäkning
    rows = []
    for _, r in work.iterrows():
        sector = _sector_of(r)
        scores = _metric_scores(r, sector)
        base, final = _weighted_total(scores, sector)
        cov = _coverage_weight(scores)
        rows.append({
            "Ticker": r.get(_first_present(ALIAS["ticker"], r), r.get("Ticker")),
            "Bolagsnamn": r.get(_first_present(ALIAS["name"], r), r.get("Bolagsnamn")),
            "Sector": sector,
            "Risklabel": r.get("Risklabel", "Unknown"),
            "Score_base": base,
            "Score": final,
            "Coverage": cov,
            "Uppsida (%)": _upside(r),
        })
    score_df = pd.DataFrame(rows)

    # sortera – högst Score, sedan Coverage, sedan Uppsida
    score_df = score_df.sort_values(by=["Score", "Coverage", "Uppsida (%)"], ascending=[False, False, False], na_position="last")

    # Bläddring 1/X
    total = len(score_df)
    pages = max(1, math.ceil(total / page_size))
    st.session_state.setdefault("inv_page", 1)
    st.session_state["inv_page"] = max(1, min(st.session_state["inv_page"], pages))

    cnav1, cnav2, cnav3 = st.columns([1, 2, 1])
    if cnav1.button("◀ Föregående", disabled=st.session_state["inv_page"] <= 1):
        st.session_state["inv_page"] -= 1
        st.rerun()
    cnav2.markdown(f"<div style='text-align:center'>**{st.session_state['inv_page']} / {pages}**</div>", unsafe_allow_html=True)
    if cnav3.button("Nästa ▶", disabled=st.session_state["inv_page"] >= pages):
        st.session_state["inv_page"] += 1
        st.rerun()

    start = (st.session_state["inv_page"] - 1) * page_size
    end = start + page_size
    page = score_df.iloc[start:end].reset_index(drop=True)

    # Visa kort per bolag
    for _, row in page.iterrows():
        tkr = row.get("Ticker", "")
        name = row.get("Bolagsnamn", "")
        sector = row.get("Sector", "Unknown")
        risk = row.get("Risklabel", "Unknown")
        base = safe_float(row.get("Score_base"), np.nan)
        sc = safe_float(row.get("Score"), np.nan)
        cov = safe_float(row.get("Coverage"), np.nan)
        up = safe_float(row.get("Uppsida (%)"), np.nan)
        label = _label_for_score(sc) if not math.isnan(sc) else "–"

        with st.container(border=True):
            st.subheader(f"{name} ({tkr})")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", f"{sc:.1f}" if not math.isnan(sc) else "–")
            c2.metric("Coverage", f"{100.0*cov:.0f}%" if not math.isnan(cov) else "–")
            c3.metric("Uppsida (1y)", f"{up:.1f}%" if not math.isnan(up) else "–")
            mcap_txt = format_large_number(_mcap_of(work.loc[work['Ticker']==tkr].iloc[0]) if (work['Ticker']==tkr).any() else np.nan, "USD")
            c4.metric("Market Cap", mcap_txt if mcap_txt else "–")
            st.write(f"**Sektor:** {sector} · **Risklabel:** {risk}")
            st.markdown(f"**Betyg:** {label}")

            with st.expander("Visa nyckeltal / detaljer"):
                _render_details_block(work, tkr)

def _render_details_block(df: pd.DataFrame, ticker: str) -> None:
    """Renderar expander-innehållet för ett givet ticker."""
    try:
        r = df[df["Ticker"].astype(str) == str(ticker)].iloc[0]
    except Exception:
        st.write("—")
        return

    # Nyckeltal vi visar om de finns:
    fields = [
        ("Valuta", _get(r, ALIAS["currency"], "—")),
        ("P/S (TTM)", _ps_avg(r)),
        ("P/S Q1", _get_num(r, ALIAS["ps_q1"])),
        ("P/S Q2", _get_num(r, ALIAS["ps_q2"])),
        ("P/S Q3", _get_num(r, ALIAS["ps_q3"])),
        ("P/S Q4", _get_num(r, ALIAS["ps_q4"])),
        ("Bruttomarginal (%)", _get_num(r, ALIAS["gross_margin"])),
        ("Nettomarginal (%)", _get_num(r, ALIAS["net_margin"])),
        ("EV/EBITDA", _get_num(r, ALIAS["ev_ebitda"])),
        ("Debt/Equity", _get_num(r, ALIAS["de_ratio"])),
        ("Dividend Yield (%)", _get_num(r, ALIAS["dy"])),
        ("Payout Ratio CF (%)", _get_num(r, ALIAS["payout_cf"])),
        ("FCF (M)", _get_num(r, ALIAS["fcf"])),
        ("Kassa (M)", _get_num(r, ALIAS["cash"])),
        ("Runway (kvartal)", _get_num(r, ALIAS["runway"])),
        ("Riktkurs (nu)", _get_num(r, ALIAS["target_now"])),
        ("Riktkurs (1y)", _get_num(r, ALIAS["target_1y"])),
        ("Riktkurs (2y)", _get_num(r, ALIAS["target_2y"])),
        ("Riktkurs (3y)", _get_num(r, ALIAS["target_3y"])),
    ]
    # Market Cap
    mcap_f = _mcap_of(r)
    fields.insert(0, ("Market Cap", format_large_number(mcap_f, "USD")))

    # Render
    for label, val in fields:
        if isinstance(val, (int, float)) and not math.isnan(float(val)):
            st.write(f"- **{label}:** {val}")
        else:
            st.write(f"- **{label}:** {val if isinstance(val, str) else '–'}")

def _first_present(keys: List[str], row: pd.Series) -> str:
    for k in keys:
        if k in row.index:
            return k
    return keys[0]
