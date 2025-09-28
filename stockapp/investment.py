# stockapp/investment.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Vi återanvänder källhjälpare (yfinance mm) från sources
from .sources import _y_ticker, _y_hist_close_on_or_before

# ------------------------------------------------------------
# Små utils
# ------------------------------------------------------------

CAP_BUCKETS = [
    ("Nano", 0, 50e6),
    ("Micro", 50e6, 300e6),
    ("Small", 300e6, 2e9),
    ("Mid", 2e9, 10e9),
    ("Large", 10e9, 200e9),
    ("Mega", 200e9, float("inf")),
]

def fmt_compact(v: float) -> str:
    """Kompakt formattering för Market Cap m.m."""
    try:
        n = float(v)
    except Exception:
        return "-"
    absv = abs(n)
    if absv >= 1e12:
        return f"{n/1e12:.2f} T"
    if absv >= 1e9:
        return f"{n/1e9:.2f} B"
    if absv >= 1e6:
        return f"{n/1e6:.2f} M"
    if absv >= 1e3:
        return f"{n/1e3:.0f} k"
    return f"{n:.0f}"

def risk_label_from_mcap(mcap: float) -> str:
    for name, lo, hi in CAP_BUCKETS:
        if mcap >= lo and mcap < hi:
            return name
    return "Okänd"

def pick_cap_buckets(mcap: float) -> List[str]:
    lab = risk_label_from_mcap(mcap)
    return [lab] if lab != "Okänd" else []

def _safe_float(x, d=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return d
        return float(x)
    except Exception:
        return d

def _ensure_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0 if any(k in str(c).lower() for k in ["p/s","omsättning","riktkurs","kurs","utdelning","marginal","debt","cash","fcf","runway","market cap","antal"]) else ""

# ------------------------------------------------------------
# MCap-historik (Q1..Q4) on-the-fly om möjligt
# ------------------------------------------------------------

def _mcap_q_history_row(row: pd.Series) -> Dict[str, float]:
    """
    Försöker beräkna MCap Q1..Q4 med hjälp av shares * historiska priser
    vid de kvartals-slut som motsvarar P/S Q1..Q4-beräkningen.
    Vi använder prisdatum som approx: utgår från att P/S Qn följer senaste 4 TTM-fönster.
    Om vi inte får fram någon historia → returnerar tom dict.
    """
    tkr = str(row.get("Ticker","")).strip()
    shares_m = _safe_float(row.get("Utestående aktier"), 0.0)  # i miljoner
    if not tkr or shares_m <= 0:
        return {}

    # Försök hitta TTM-datum vi använt tidigare: saknas i DF → approximera senaste 4 kvartals slut
    # Vi tar de 4 senaste kvartalen som kalendermånader: 3/6/9/12 med 'nära slutet av månaden'
    # och hämtar närmaste pris på/innan dessa datum. Det är en rimlig proxy.
    approx_dates = []
    try:
        # Hämta 5 kvartal tillbaka så vi kan ta 4 senaste
        t = _y_ticker(tkr)
        hist = t.history(period="2y", interval="1d")
        if hist is None or hist.empty:
            return {}
        idx = list(hist.index.date)
        # hitta sista handelsdag i månader: mar/jun/sep/dec/ (samt ev jan för brutna räkenskapsår)
        candidates = []
        seen = set()
        for d in reversed(idx):
            if d.month in (3,6,9,12):
                key = (d.year, d.month)
                if key not in seen:
                    seen.add(key)
                    candidates.append(d)
            if len(candidates) >= 6:
                break
        approx_dates = candidates[:4]
        px_map = _y_hist_close_on_or_before(tkr, approx_dates)
        shares = shares_m * 1e6
        out = {}
        # Nyast→äldst
        for j, d in enumerate(approx_dates, start=1):
            px = _safe_float(px_map.get(d), 0.0)
            if px > 0 and shares > 0:
                out[f"MCap Q{j}"] = float(shares * px)
        return out
    except Exception:
        return {}

# ------------------------------------------------------------
# Poängmodeller (Tillväxt vs Utdelning)
# ------------------------------------------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def score_growth(row: pd.Series, target_col: str) -> Tuple[float, Dict[str, float]]:
    """Poäng för tillväxt-case."""
    price = _safe_float(row.get("Aktuell kurs"), 0.0)
    target = _safe_float(row.get(target_col), 0.0)
    ps_now = _safe_float(row.get("P/S"), 0.0)
    ps_avg = _safe_float(row.get("P/S-snitt"), 0.0)
    cagr = _safe_float(row.get("CAGR 5 år (%)"), 0.0)
    gm = _safe_float(row.get("Bruttomarginal (%)"), 0.0)
    nm = _safe_float(row.get("Nettomarginal (%)"), 0.0)
    de = _safe_float(row.get("Debt/Equity"), 0.0)
    runway = _safe_float(row.get("Runway (mån)"), 0.0)

    # Undervärdering mot egen riktkurs
    underv = 0.0
    if price > 0 and target > 0:
        underv = _clip01((target/price - 1.0) / 1.0)  # 0 uppsida=0, 100% uppsida ~1

    # Låg P/S bättre (om ps_avg saknas använd ps_now)
    ps_eff = ps_avg if ps_avg > 0 else ps_now
    ps_score = _clip01((20.0 / max(1.0, ps_eff)))  # ps=20 → ~1.0; ps=40 → 0.5; ps=10 → >1 clampas

    # Tillväxt & lönsamhet
    cagr_s = _clip01(cagr / 50.0)       # 50% CAGR → 1.0
    gm_s   = _clip01(gm / 70.0)         # 70% GM → 1.0
    nm_s   = _clip01((nm + 20.0) / 40)  # -20%..+20% → 0..1

    # Finansiell risk (lägre D/E bättre, längre runway bättre)
    de_s = _clip01(1.0 - min(de, 2.0) / 2.0)  # D/E=0 →1, D/E=2 →0
    rw_s = _clip01(runway / 24.0)             # 24 mån →1.0

    # Vikter (justerbart)
    w = {
        "underv": 0.28,
        "ps": 0.18,
        "cagr": 0.16,
        "gm": 0.10,
        "nm": 0.10,
        "de": 0.10,
        "rw": 0.08,
    }
    total = 100.0 * (
        w["underv"] * underv +
        w["ps"]    * ps_score +
        w["cagr"]  * cagr_s +
        w["gm"]    * gm_s +
        w["nm"]    * nm_s +
        w["de"]    * de_s +
        w["rw"]    * rw_s
    )

    breakdown = {
        "underv": round(100*underv*w["underv"], 1),
        "ps": round(100*ps_score*w["ps"], 1),
        "cagr": round(100*cagr_s*w["cagr"], 1),
        "gm": round(100*gm_s*w["gm"], 1),
        "nm": round(100*nm_s*w["nm"], 1),
        "de": round(100*de_s*w["de"], 1),
        "rw": round(100*rw_s*w["rw"], 1),
    }
    return float(round(total, 1)), breakdown

def score_dividend(row: pd.Series, target_col: str) -> Tuple[float, Dict[str, float]]:
    """Poäng för utdelnings-case: hög yield, rimlig risk, kassaflödesstöd."""
    price = _safe_float(row.get("Aktuell kurs"), 0.0)
    div_ps = _safe_float(row.get("Årlig utdelning"), 0.0)  # per aktie i bolagsvaluta
    shares_m = _safe_float(row.get("Utestående aktier"), 0.0)  # miljoner
    fcf = _safe_float(row.get("FCF TTM (valuta)"), 0.0)
    cash = _safe_float(row.get("Kassa (valuta)"), 0.0)
    de = _safe_float(row.get("Debt/Equity"), 0.0)
    gm = _safe_float(row.get("Bruttomarginal (%)"), 0.0)
    nm = _safe_float(row.get("Nettomarginal (%)"), 0.0)
    ps_now = _safe_float(row.get("P/S"), 0.0)

    # Yield
    dividend_yield = (div_ps / price) if (price > 0 and div_ps > 0) else 0.0
    y_s = _clip01(dividend_yield / 0.08)  # 8% → 1.0

    # Utdelningsbelopp total ≈ div_ps * shares
    total_div = div_ps * (shares_m * 1e6) if shares_m > 0 else 0.0

    # FCF-säkerhet: FCF / total_div (>=1 tryggt)
    fcf_cov = (fcf / total_div) if (total_div > 0) else 0.0
    fcf_s = _clip01(fcf_cov / 1.0)  # 1.0 → 1.0

    # Kassa-säkerhet relativt utdelning (1 års utdelning)
    cash_cov = (cash / total_div) if (total_div > 0) else 0.0
    cash_s = _clip01(cash_cov / 1.0)

    # Låg D/E bra
    de_s = _clip01(1.0 - min(de, 2.0)/2.0)

    # Lönsamhet (positiv marginal)
    gm_s = _clip01(gm / 60.0)
    nm_s = _clip01((nm + 10.0)/20.0)  # -10..+10 => 0..1

    # Billighet via P/S (låg bättre)
    ps_s = _clip01(15.0 / max(1.0, ps_now))

    w = {
        "yield": 0.30,
        "fcf":   0.20,
        "cash":  0.15,
        "de":    0.15,
        "gm":    0.10,
        "nm":    0.05,
        "ps":    0.05,
    }
    total = 100.0 * (
        w["yield"] * y_s +
        w["fcf"]   * fcf_s +
        w["cash"]  * cash_s +
        w["de"]    * de_s +
        w["gm"]    * gm_s +
        w["nm"]    * nm_s +
        w["ps"]    * ps_s
    )
    breakdown = {
        "yield": round(100*y_s*w["yield"],1),
        "fcf":   round(100*fcf_s*w["fcf"],1),
        "cash":  round(100*cash_s*w["cash"],1),
        "de":    round(100*de_s*w["de"],1),
        "gm":    round(100*gm_s*w["gm"],1),
        "nm":    round(100*nm_s*w["nm"],1),
        "ps":    round(100*ps_s*w["ps"],1),
        "yield_raw_%": round(dividend_yield*100.0,2),
        "fcf_cov": round(fcf_cov,2),
        "cash_cov": round(cash_cov,2),
    }
    return float(round(total, 1)), breakdown

def label_from_score(s: float) -> str:
    if s >= 80: return "Mycket bra (Köp)"
    if s >= 65: return "Bra"
    if s >= 50: return "Fair/Behåll"
    if s >= 35: return "Övervärderad (Trimma)"
    return "Sälj/Undvik"

# ------------------------------------------------------------
# Huvudvy
# ------------------------------------------------------------

def investment_view(df: pd.DataFrame, user_rates: dict):
    st.header("💡 Investeringsförslag")

    # säkerställ kolumner som används
    _ensure_cols(df, [
        "Ticker","Bolagsnamn","Sektor","Market Cap (nu)",
        "Aktuell kurs","Valuta",
        "Utestående aktier","Årlig utdelning",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity",
        "Kassa (valuta)","FCF TTM (valuta)","Runway (mån)"
    ])

    # Strategi
    strat = st.radio("Strategi", ["Tillväxt","Utdelning"], horizontal=True, index=0)

    # Riktkurskolumn
    target_col = st.selectbox(
        "Vilken riktkurs?", ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"], index=0
    )

    # Sektorfilter
    sectors = sorted([s for s in df["Sektor"].astype(str).unique() if s and s != "nan"])
    chosen_sectors = st.multiselect("Filtrera sektor (valfritt)", sectors, default=[])

    # Cap buckets
    cap_cats = [b[0] for b in CAP_BUCKETS]
    chosen_caps = st.multiselect("Filtrera värdekategori (valfritt)", cap_cats, default=[])

    # Basurval
    base = df.copy()
    # krav: vi måste ha pris och target
    base = base[(base["Aktuell kurs"] > 0) & (base[target_col] > 0)]

    # sektorfilter
    if chosen_sectors:
        base = base[base["Sektor"].astype(str).isin(chosen_sectors)]

    # cap-kategorier
    if chosen_caps:
        cats = []
        for _, r in base.iterrows():
            lab = risk_label_from_mcap(_safe_float(r.get("Market Cap (nu)"), 0.0))
            cats.append(lab)
        base = base.assign(_cap_bucket=cats)
        base = base[base["_cap_bucket"].isin(chosen_caps)]

    if base.empty:
        st.info("Inga bolag matchar urvalet just nu.")
        return

    # beräkna P/S-snitt om saknas
    def _ps_avg_row(r: pd.Series) -> float:
        vals = []
        for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
            v = _safe_float(r.get(k), 0.0)
            if v > 0: vals.append(v)
        return round(float(np.mean(vals)), 2) if vals else _safe_float(r.get("P/S-snitt"), 0.0)

    base = base.copy()
    base["P/S-snitt"] = base.apply(_ps_avg_row, axis=1)

    # Poäng
    scores = []
    breakdowns = []
    for _, r in base.iterrows():
        if strat == "Tillväxt":
            s, br = score_growth(r, target_col)
        else:
            s, br = score_dividend(r, target_col)
        scores.append(s); breakdowns.append(br)

    base["_score"] = scores
    base["_label"] = [label_from_score(s) for s in scores]
    base["_potential_%"] = (base[target_col] / base["Aktuell kurs"] - 1.0) * 100.0

    # sortera på score, därefter största uppsida
    base = base.sort_values(by=["_score","_potential_%"], ascending=[False,False]).reset_index(drop=True)

    # Bläddring
    if "inv_idx" not in st.session_state: st.session_state.inv_idx = 0
    st.session_state.inv_idx = min(st.session_state.inv_idx, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.inv_idx = max(0, st.session_state.inv_idx - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.inv_idx+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.inv_idx = min(len(base)-1, st.session_state.inv_idx + 1)

    r = base.iloc[st.session_state.inv_idx]
    ridx = df.index[df["Ticker"].astype(str).str.upper() == str(r["Ticker"]).upper()]
    ridx = ridx[0] if len(ridx) else None

    # Huvudkort
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    top_cols = st.columns([2,1,1,1])
    with top_cols[0]:
        st.metric("Poäng", f"{r['_score']:.1f}", help=f"Etikett: {r['_label']}")
        st.caption(r['_label'])
    with top_cols[1]:
        st.metric("Uppsida mot mål", f"{r['_potential_%']:.1f}%")
    with top_cols[2]:
        st.metric("P/S (nu)", f"{_safe_float(r.get('P/S'),0.0):.2f}")
    with top_cols[3]:
        st.metric("P/S-snitt (Q1–Q4)", f"{_safe_float(r.get('P/S-snitt'),0.0):.2f}")

    # Info-rad
    mc = _safe_float(r.get("Market Cap (nu)"), 0.0)
    st.markdown(
        f"- **Aktuell kurs:** {r['Aktuell kurs']:.2f} {r['Valuta']}  \n"
        f"- **Riktkurs ({target_col}):** {r[target_col]:.2f} {r['Valuta']}  \n"
        f"- **Market Cap (nu):** {fmt_compact(mc)}  \n"
        f"- **Risklabel:** {risk_label_from_mcap(mc)}  \n"
        f"- **Sektor:** {str(r.get('Sektor') or '-')}"
    )

    # Expander med detaljer
    with st.expander("Visa fler nyckeltal & historik"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Debt/Equity:** { _safe_float(r.get('Debt/Equity'),0.0):.2f}")
            st.write(f"**Bruttomarginal:** { _safe_float(r.get('Bruttomarginal (%)'),0.0):.1f}%")
            st.write(f"**Nettomarginal:** { _safe_float(r.get('Nettomarginal (%)'),0.0):.1f}%")
        with c2:
            st.write(f"**Kassa:** {fmt_compact(_safe_float(r.get('Kassa (valuta)'),0.0))}")
            st.write(f"**FCF TTM:** {fmt_compact(_safe_float(r.get('FCF TTM (valuta)'),0.0))}")
            st.write(f"**Runway:** { _safe_float(r.get('Runway (mån)'),0.0):.1f} mån")
        with c3:
            st.write(f"**Utestående aktier:** { _safe_float(r.get('Utestående aktier'),0.0):.2f} M")
            # P/S Q1..Q4
            ps_hist_txt = ", ".join([f"{k}:{_safe_float(r.get(k),0.0):.2f}" for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"] if k in r.index])
            st.write(f"**P/S Q1–Q4:** {ps_hist_txt if ps_hist_txt else '-'}")

        # Försök räkna fram MCap Q1..Q4 on-the-fly
        try:
            m_hist = _mcap_q_history_row(r)
            if m_hist:
                st.write("**MCap Q1–Q4 (approx):** " + ", ".join([f"{k}:{fmt_compact(v)}" for k, v in m_hist.items()]))
        except Exception:
            pass

        # Breakdown av score
        idx = base.index[st.session_state.inv_idx]
        br = breakdowns[idx]
        br_lines = [f"- **{k}**: {v}" for k, v in br.items() if not k.endswith("_raw_%") and not k.endswith("_cov")]
        st.markdown("**Score-breakdown:**  \n" + "\n".join(br_lines))
        # Extra råvärden om utdelningscase
        if strat == "Utdelning":
            extra = []
            if "yield_raw_%" in br:
                extra.append(f"Utdelningsyield: {br['yield_raw_%']:.2f}%")
            if "fcf_cov" in br:
                extra.append(f"FCF-täckning: {br['fcf_cov']:.2f}×")
            if "cash_cov" in br:
                extra.append(f"Kassa-täckning: {br['cash_cov']:.2f}×")
            if extra:
                st.caption(" / ".join(extra))

    # Liten tabell över topp 10 för snabb översikt (utan lagg)
    st.divider()
    show_cols = ["Ticker","Bolagsnamn","Sektor","_score","_label","Aktuell kurs","Valuta",target_col,"_potential_%","P/S","P/S-snitt","Market Cap (nu)"]
    show_cols = [c for c in show_cols if c in base.columns]
    st.dataframe(
        base[show_cols].head(10).assign(**{
            "_potential_%": base["_potential_%"].head(10).round(1),
            "_score": base["_score"].head(10).round(1),
            "Market Cap (nu)": base["Market Cap (nu)"].head(10).apply(fmt_compact)
        }),
        use_container_width=True,
        hide_index=True
    )
