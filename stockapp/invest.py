# -*- coding: utf-8 -*-
"""
stockapp.invest
----------------
Investeringsförslag med bläddringsvy, sektor- & riskfilter,
sektorviktad scoring och robust hantering av saknade nyckeltal.

Publik:
- visa_investeringsforslag(df, user_rates, page_size=1, page_key="inv_pg")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Lokala hjälpfunktioner
# -------------------------
def _coalesce(row: pd.Series, keys: List[str], default=None):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


def _format_large_number(value: float, unit: Optional[str] = None) -> str:
    """
    1_234 -> 1.23k
    12_345_678 -> 12.35M etc
    """
    try:
        v = float(value)
    except Exception:
        return "-"
    abs_v = abs(v)
    suffix = ""
    if abs_v >= 1_000_000_000_000:
        v = v / 1_000_000_000_000.0
        suffix = "T"
    elif abs_v >= 1_000_000_000:
        v = v / 1_000_000_000.0
        suffix = "B"
    elif abs_v >= 1_000_000:
        v = v / 1_000_000.0
        suffix = "M"
    elif abs_v >= 1_000:
        v = v / 1_000.0
        suffix = "k"
    out = f"{v:.2f}{suffix}"
    if unit:
        out = f"{out} {unit}"
    return out


def _risk_label_from_mcap(mcap: float) -> str:
    v = _safe_float(mcap, 0.0)
    if v <= 0:
        return "Okänd"
    if v < 300e6:
        return "Microcap"
    if v < 2e9:
        return "Smallcap"
    if v < 10e9:
        return "Midcap"
    if v < 200e9:
        return "Largecap"
    return "Megacap"


def _normalize(val: Optional[float], lo: float, hi: float, invert: bool = False) -> float:
    """Skalar val till [0,1] utifrån intervall. Klipper utanför. invert=True => 1-x."""
    v = _safe_float(val, None)
    if v is None:
        return 0.0
    if hi == lo:
        return 0.0
    x = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    return 1.0 - x if invert else x


# -------------------------
# Sektorvikter & score
# -------------------------
_DEFAULT_SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Nycklar: valu, growth, quality, income, balance
    "Technology":    {"valu": 0.25, "growth": 0.35, "quality": 0.25, "income": 0.05, "balance": 0.10},
    "Consumer":      {"valu": 0.30, "growth": 0.25, "quality": 0.25, "income": 0.10, "balance": 0.10},
    "Healthcare":    {"valu": 0.25, "growth": 0.30, "quality": 0.30, "income": 0.05, "balance": 0.10},
    "Financial Services": {"valu": 0.30, "growth": 0.10, "quality": 0.35, "income": 0.10, "balance": 0.15},
    "Energy":        {"valu": 0.30, "growth": 0.20, "quality": 0.20, "income": 0.15, "balance": 0.15},
    "Industrials":   {"valu": 0.30, "growth": 0.20, "quality": 0.25, "income": 0.10, "balance": 0.15},
    "Utilities":     {"valu": 0.25, "growth": 0.10, "quality": 0.20, "income": 0.30, "balance": 0.15},
    "Real Estate":   {"valu": 0.30, "growth": 0.10, "quality": 0.20, "income": 0.25, "balance": 0.15},
    "Communication Services": {"valu": 0.30, "growth": 0.25, "quality": 0.25, "income": 0.10, "balance": 0.10},
    # fallback:
    "_default":      {"valu": 0.30, "growth": 0.25, "quality": 0.25, "income": 0.10, "balance": 0.10},
}


def _sector_weights(sector: str) -> Dict[str, float]:
    if not sector:
        return _DEFAULT_SECTOR_WEIGHTS["_default"]
    for k in _DEFAULT_SECTOR_WEIGHTS:
        if k != "_default" and k.lower() in sector.lower():
            return _DEFAULT_SECTOR_WEIGHTS[k]
    return _DEFAULT_SECTOR_WEIGHTS["_default"]


def _compute_score(row: pd.Series) -> Tuple[float, Dict[str, float], float]:
    """
    Beräknar en sektorviktad score.
    Returnerar (score, komponenter, coverage)
      - coverage = andel nyckeltal som fanns (0..1), används även för nedviktning.
    Komponenter (0..1): valu, growth, quality, income, balance
    """
    # --- Inputs (tål saknade) ---
    sector = str(_coalesce(row, ["Sektor", "Sector"], ""))
    price = _safe_float(_coalesce(row, ["Senaste Kurs (USD)", "Senaste Kurs", "Price"], None), None)
    target = _safe_float(_coalesce(row, ["Riktkurs (USD)", "Riktkurs", "Target Price"], None), None)
    ps_now = _safe_float(_coalesce(row, ["P/S (nu)", "P/S"], None), None)
    ps_avg = _safe_float(_coalesce(row, ["P/S snitt 4q", "P/S (4q-snitt)"], None), None)
    rev_fwd = _safe_float(_coalesce(row, ["Omsättning i år (förv.)", "Revenue (FWD)"], None), None)
    gm = _safe_float(_coalesce(row, ["Bruttomarginal (%)", "Gross Margin (%)"], None), None)
    nm = _safe_float(_coalesce(row, ["Nettomarginal (%)", "Net Margin (%)"], None), None)
    de = _safe_float(_coalesce(row, ["Debt/Equity", "Skuldsättning"], None), None)
    ev_ebitda = _safe_float(_coalesce(row, ["EV/EBITDA"], None), None)
    fcf_margin = _safe_float(_coalesce(row, ["FCF-marginal (%)", "FCF Margin (%)"], None), None)
    yield_pct = _safe_float(_coalesce(row, ["Utdelning Yield (%)", "Dividend Yield (%)"], None), None)
    payout_fcf = _safe_float(_coalesce(row, ["Payout (FCF) (%)"], None), None)
    runway_q = _safe_float(_coalesce(row, ["Runway (kvartal)"], None), None)

    # --- Coverage (hur mycket data finns) ---
    raw_vals = [price, target, ps_now, ps_avg, rev_fwd, gm, nm, de, ev_ebitda, fcf_margin, yield_pct, payout_fcf, runway_q]
    have = sum(1 for v in raw_vals if v is not None and not (isinstance(v, float) and math.isnan(v)))
    coverage = have / max(1, len(raw_vals))

    # --- Komponenter ---
    # Valuation: uppsida + PS-normalisering (lägre bättre)
    comp_valu = 0.0
    if price and target and price > 0:
        upside = (target - price) / price  # t.ex. 0.2 = +20%
        comp_valu += _normalize(upside, -0.3, 0.6, invert=False) * 0.6  # vikt i komponenten
    if ps_now and ps_avg and ps_now > 0:
        rel_ps = ps_avg / ps_now  # >1 bättre (nu billigare än snitt)
        comp_valu += _normalize(rel_ps, 0.6, 1.6, invert=False) * 0.4
    comp_valu = min(1.0, comp_valu)

    # Growth: FWD revenue (skala mot typ 0..50% yoy proxy) + bruttomarginal som proxy för skala
    comp_growth = 0.0
    if rev_fwd:
        # utan historik: approximera *bara* att det finns prognos
        comp_growth += 0.6
    if gm is not None:
        comp_growth += _normalize(gm, 20.0, 70.0) * 0.4
    comp_growth = min(1.0, comp_growth)

    # Quality: nettomarginal + EV/EBITDA (lägre bättre)
    comp_quality = 0.0
    if nm is not None:
        comp_quality += _normalize(nm, 0.0, 25.0) * 0.6
    if ev_ebitda is not None:
        comp_quality += _normalize(ev_ebitda, 4.0, 20.0, invert=True) * 0.4
    comp_quality = min(1.0, comp_quality)

    # Income: utdelning & payout (FCF)
    comp_income = 0.0
    if yield_pct is not None:
        comp_income += _normalize(yield_pct, 0.0, 8.0) * 0.7
    if payout_fcf is not None:
        # 0–70% bäst, över 100% dåligt
        good = _normalize(payout_fcf, 0.0, 70.0)
        bad = _normalize(payout_fcf, 100.0, 200.0)  # högre sämre
        comp_income += max(0.0, good - 0.5 * bad) * 0.3
    comp_income = min(1.0, comp_income)

    # Balance: skuldsättning (lägre bättre) + runway
    comp_balance = 0.0
    if de is not None:
        comp_balance += _normalize(de, 0.0, 2.0, invert=True) * 0.6
    if runway_q is not None:
        comp_balance += _normalize(runway_q, 2.0, 12.0) * 0.4
    comp_balance = min(1.0, comp_balance)

    comps = {
        "valu": comp_valu,
        "growth": comp_growth,
        "quality": comp_quality,
        "income": comp_income,
        "balance": comp_balance,
    }

    # Sektorvikter
    w = _sector_weights(sector)
    base = (
        comps["valu"] * w["valu"]
        + comps["growth"] * w["growth"]
        + comps["quality"] * w["quality"]
        + comps["income"] * w["income"]
        + comps["balance"] * w["balance"]
    )

    # Nedviktning med coverage (mer data => högre förtroende)
    score = base * (0.5 + 0.5 * coverage)

    return float(score), comps, float(coverage)


def _label_from_score(score: float, upside: Optional[float]) -> str:
    """
    Returnerar etikett: 'Köp', 'Behåll', 'Trimma', 'Sälj' med försiktiga trösklar.
    Vi nyttjar även uppsida (om finns).
    """
    if upside is None:
        upside = 0.0
    # konservativa trösklar
    if score >= 0.70 and upside >= 0.15:
        return "Köp"
    if score >= 0.55 and upside >= 0.05:
        return "Behåll"
    if score <= 0.40 and upside <= -0.05:
        return "Sälj"
    if score <= 0.50 and upside <= 0.00:
        return "Trimma"
    return "Behåll"


# -------------------------
# Publik vy
# -------------------------
def visa_investeringsforslag(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    page_size: int = 1,
    page_key: str = "inv_pg",
) -> None:
    """
    Visar investeringsförslag i bläddringsvy (page_size poster per sida).
    - Robust mot saknade kolumner
    - Sektor- & riskfilter
    - Sektorviktad scoring
    - Expander med nyckeltal

    Parametrar:
      df: DataFrame med din databas
      user_rates: valutakurser (ej strikt nödvändigt här men kvar för kompatibilitet)
      page_size: antal kort per sida (default 1 för ”bläddringskänsla”)
      page_key: nyckel för session_state (om du vill ha separat paging per vy)
    """
    st.markdown("## Investeringsförslag")

    work = df.copy()
    if work.empty:
        st.info("Ingen data i databasen ännu.")
        return

    # Basfält som kan visas / hämtas
    # Hantera saknader utan att krascha
    if "Ticker" not in work.columns:
        work["Ticker"] = ""
    if "Namn" not in work.columns and "Name" in work.columns:
        work["Namn"] = work["Name"]
    elif "Namn" not in work.columns:
        work["Namn"] = ""

    # Market Cap & Sektor
    if "Market Cap" not in work.columns:
        work["Market Cap"] = np.nan
    if "Sektor" not in work.columns and "Sector" in work.columns:
        work["Sektor"] = work["Sector"]
    elif "Sektor" not in work.columns:
        work["Sektor"] = ""

    # Pris & riktkurs
    if "Senaste Kurs (USD)" not in work.columns and "Price" in work.columns:
        work["Senaste Kurs (USD)"] = work["Price"]
    elif "Senaste Kurs (USD)" not in work.columns:
        work["Senaste Kurs (USD)"] = np.nan

    if "Riktkurs (USD)" not in work.columns and "Target Price" in work.columns:
        work["Riktkurs (USD)"] = work["Target Price"]
    elif "Riktkurs (USD)" not in work.columns:
        work["Riktkurs (USD)"] = np.nan

    # P/S nu & snitt
    if "P/S (nu)" not in work.columns and "P/S" in work.columns:
        work["P/S (nu)"] = work["P/S"]
    elif "P/S (nu)" not in work.columns:
        work["P/S (nu)"] = np.nan

    if "P/S snitt 4q" not in work.columns:
        work["P/S snitt 4q"] = np.nan

    # Uträknat: Uppsida (%)
    def _calc_upside(row):
        p = _safe_float(row.get("Senaste Kurs (USD)"), None)
        t = _safe_float(row.get("Riktkurs (USD)"), None)
        if p and t and p > 0:
            return (t - p) / p * 100.0
        return np.nan

    work["_Uppsida (%)"] = work.apply(_calc_upside, axis=1)

    # Scoring
    scores = []
    covs = []
    comps_list = []
    for _, r in work.iterrows():
        s, comps, cov = _compute_score(r)
        scores.append(s)
        covs.append(cov)
        comps_list.append(comps)
    work["_Score"] = scores
    work["_Coverage"] = covs

    # Risk label
    work["_RiskLabel"] = work["Market Cap"].apply(_risk_label_from_mcap)

    # Filtrering – sektor & risk
    sectors = sorted([s for s in work["Sektor"].dropna().astype(str).unique() if s])
    picked_sector = st.multiselect("Filtrera sektor", options=sectors, default=sectors)
    risk_opts = ["Microcap", "Smallcap", "Midcap", "Largecap", "Megacap", "Okänd"]
    picked_risk = st.multiselect("Filtrera risklabel", options=risk_opts, default=risk_opts)

    base = work[
        work["Sektor"].astype(str).isin(picked_sector)
        & work["_RiskLabel"].astype(str).isin(picked_risk)
    ].copy()

    if base.empty:
        st.warning("Inga kandidater matchade filtren.")
        return

    # Rangordning: Score DESC, Coverage DESC, Uppsida DESC
    base = base.sort_values(by=["_Score", "_Coverage", "_Uppsida (%)"], ascending=[False, False, False])

    # Paging
    total = len(base)
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    page_size = max(1, int(page_size))
    max_page = max(0, math.ceil(total / page_size) - 1)
    idx = min(max(0, int(st.session_state[page_key])), max_page)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("◀︎ Föregående", disabled=(idx == 0)):
            idx = max(0, idx - 1)
    with c2:
        st.markdown(f"<div style='text-align:center'>**{idx+1}/{max_page+1}**</div>", unsafe_allow_html=True)
    with c3:
        if st.button("Nästa ▶︎", disabled=(idx >= max_page)):
            idx = min(max_page, idx + 1)
    st.session_state[page_key] = idx

    start = idx * page_size
    stop = min(total, start + page_size)
    view = base.iloc[start:stop].copy()

    # Rendera kort
    for _, row in view.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        name = str(row.get("Namn", "")).strip() or ticker
        sector = str(row.get("Sektor", ""))
        price = _safe_float(row.get("Senaste Kurs (USD)"), None)
        target = _safe_float(row.get("Riktkurs (USD)"), None)
        mcap = _safe_float(row.get("Market Cap"), None)
        ps_now = _safe_float(row.get("P/S (nu)"), None)
        ps_avg = _safe_float(row.get("P/S snitt 4q"), None)
        ups = row.get("_Uppsida (%)")
        score = float(row.get("_Score", 0.0))
        cov = float(row.get("_Coverage", 0.0))
        risk = str(row.get("_RiskLabel", ""))

        # Label
        up_val = None if (ups is None or (isinstance(ups, float) and math.isnan(ups))) else float(ups) / 100.0
        label = _label_from_score(score, up_val)

        st.markdown(f"### {ticker} — {name}")
        st.caption(f"Sektor: {sector} • Risk: {risk}")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Score", f"{score:.2f}")
        with cols[1]:
            st.metric("Coverage", f"{cov*100:.0f}%")
        with cols[2]:
            st.metric("Uppsida", ("-" if ups is None or (isinstance(ups, float) and math.isnan(ups)) else f"{ups:.1f}%"))
        with cols[3]:
            st.metric("Bedömning", label)

        cols2 = st.columns(4)
        with cols2[0]:
            st.write("**Kurs (USD)**")
            st.write("-" if price is None else f"{price:.2f}")
        with cols2[1]:
            st.write("**Riktkurs (USD)**")
            st.write("-" if target is None else f"{target:.2f}")
        with cols2[2]:
            st.write("**Market Cap**")
            st.write("-" if mcap is None else _format_large_number(mcap))
        with cols2[3]:
            st.write("**P/S (nu) vs snitt**")
            a = "-" if ps_now is None else f"{ps_now:.2f}"
            b = "-" if ps_avg is None else f"{ps_avg:.2f}"
            st.write(f"{a} / {b}")

        with st.expander("Nyckeltal & detaljer", expanded=False):
            # Lista över intressanta fält att visa om de finns
            show_fields = [
                "Sektor", "Industri",
                "P/S (nu)", "P/S snitt 4q",
                "EV/EBITDA",
                "Bruttomarginal (%)", "Nettomarginal (%)",
                "Debt/Equity",
                "Omsättning i år (förv.)", "Omsättning nästa år (förv.)",
                "Kassa (valuta)", "Kassa (SEK)", "Runway (kvartal)",
                "Utdelning Yield (%)", "Payout (FCF) (%)",
                "Antal aktier",
            ]
            lines = []
            for f in show_fields:
                if f in row and pd.notna(row[f]):
                    v = row[f]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if "marginal" in f.lower() or "yield" in f.lower() or "payout" in f.lower():
                            lines.append(f"- **{f}:** {float(v):.2f}%")
                        elif "P/S" in f or "EV/EBITDA" in f:
                            lines.append(f"- **{f}:** {float(v):.2f}")
                        elif "Omsättning" in f:
                            lines.append(f"- **{f}:** {_format_large_number(float(v))}")
                        elif "Kassa" in f:
                            lines.append(f"- **{f}:** {_format_large_number(float(v))}")
                        elif "Antal aktier" in f:
                            lines.append(f"- **{f}:** {float(v):,.0f}".replace(",", " "))
                        else:
                            lines.append(f"- **{f}:** {float(v):.2f}")
                    else:
                        lines.append(f"- **{f}:** {v}")
            if lines:
                st.markdown("\n".join(lines))
            else:
                st.caption("Inga detaljer tillgängliga.")

        st.divider()

    # Fotnot
    st.caption(
        "Poängen väger in värdering, tillväxt, kvalitet, utdelning och balansräkning med sektorspecifika vikter. "
        "Bolag med färre datapunkter nedviktas automatiskt."
    )
