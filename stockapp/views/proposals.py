# -*- coding: utf-8 -*-
"""
stockapp/views/proposals.py
Investeringsförslag: rankar bolag för tillväxt/utdelning
- Filter: sektor, börsvärdesklass
- Beräkningar: P/S-snitt, MCap, uppsida, riktkurs (om data finns)
- Fallback-score om externa scorer saknas
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import streamlit as st
import pandas as pd
import numpy as np

# ---------- Små hjälpare ----------

def _fmt_large(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    neg = v < 0
    v = abs(v)
    unit = ""
    if v >= 1e12:
        v /= 1e12; unit = " tn"
    elif v >= 1e9:
        v /= 1e9; unit = " mdr"
    elif v >= 1e6:
        v /= 1e6; unit = " md"
    elif v >= 1e3:
        v /= 1e3; unit = " k"
    out = f"{'-' if neg else ''}{v:,.2f}{unit}".replace(",", " ")
    return out

def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x)*100.0:.2f} %"
    except Exception:
        return "-"

def _safe_float(v, default=np.nan):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default

def _classify_mcap(usd_value: float) -> str:
    # grova nivåer (USD)
    v = _safe_float(usd_value, np.nan)
    if np.isnan(v):
        return "Okänt"
    if v < 300e6:   return "Microcap"
    if v < 2e9:     return "Smallcap"
    if v < 10e9:    return "Midcap"
    if v < 200e9:   return "Largecap"
    return "Megacap"

def _ps_avg(row: pd.Series) -> float:
    vals = []
    for k in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
        if k in row and _safe_float(row[k], 0) > 0:
            vals.append(float(row[k]))
    if vals:
        return float(np.mean(vals))
    # fallback: använd P/S om Q-uppdelning saknas
    if _safe_float(row.get("P/S"), 0) > 0:
        return float(row.get("P/S"))
    return np.nan

def _current_mcap(row: pd.Series) -> float:
    px = _safe_float(row.get("Aktuell kurs"), np.nan)
    sh_m = _safe_float(row.get("Utestående aktier"), np.nan)  # i miljoner
    if np.isnan(px) or np.isnan(sh_m) or sh_m <= 0 or px <= 0:
        return np.nan
    return px * sh_m * 1e6  # i bolagets valuta

def _target_from_ps(row: pd.Series) -> Tuple[float, float, float]:
    """
    Returnerar (target_mcap, upside, target_price). Om något saknas -> (nan, nan, nan).
    - revenue_m: 'Omsättning idag' (miljoner, i bolagsvaluta)
    - ps_avg: medel av P/S Q1..Q4 (eller P/S)
    """
    rev_m = _safe_float(row.get("Omsättning idag"), np.nan)  # miljoner
    ps = _ps_avg(row)
    mcap_now = _current_mcap(row)
    px_now = _safe_float(row.get("Aktuell kurs"), np.nan)
    if np.isnan(rev_m) or rev_m <= 0 or np.isnan(ps) or ps <= 0 or np.isnan(mcap_now) or mcap_now <= 0 or np.isnan(px_now) or px_now <= 0:
        return (np.nan, np.nan, np.nan)
    target_mcap = rev_m * 1e6 * ps
    upside = target_mcap / mcap_now - 1.0
    target_price = px_now * (1.0 + upside)
    return (target_mcap, upside, target_price)

def _rating_from_score(score: float) -> str:
    if score >= 85: return "Mycket bra läge (Köp)"
    if score >= 70: return "Bra läge (Överväg köp)"
    if score >= 55: return "Fair/Neutral (Behåll)"
    if score >= 40: return "Något dyr (Trimma)"
    return "Övervärderad (Sälj/undvik)"

# ---------- Fallback-scorer ----------

def _fallback_growth_score(row: pd.Series) -> Tuple[float, Dict[str, Any]]:
    """
    Enkel, transparent tillväxt-score (0–100).
    Högre P/S sänker, starka marginaler höjer, lägre D/E höjer,
    positiv uppsida mot P/S-snitt höjer.
    """
    ps = _ps_avg(row)
    de = _safe_float(row.get("Debt/Equity"), np.nan)
    gm = _safe_float(row.get("Bruttomarginal"), np.nan)
    nm = _safe_float(row.get("Nettomarginal"), np.nan)
    target_mcap, upside, _ = _target_from_ps(row)

    # normalisering
    ps_comp = 0.0 if np.isnan(ps) else max(0.0, 100.0 - min(ps, 50.0) * 2.0)  # ps 0..50 -> 100..0
    de_comp = 0.0 if np.isnan(de) else max(0.0, 100.0 - min(de, 5.0) * 15.0)  # D/E 0..5 -> 100..25
    gm_comp = 0.0 if np.isnan(gm) else min(max(gm, 0.0), 80.0) * 1.0         # 0..80 -> 0..80
    nm_comp = 0.0 if np.isnan(nm) else min(max(nm, 0.0), 60.0) * 1.2         # 0..60 -> 0..72
    up_comp = 0.0 if np.isnan(upside) else min(max(upside*100.0, -50.0), 100.0) * 0.6

    # vikter
    score = 0.30*ps_comp + 0.15*de_comp + 0.25*gm_comp + 0.15*nm_comp + 0.15*up_comp
    score = float(np.clip(score, 0, 100))
    diag = {
        "ps_comp": ps_comp, "de_comp": de_comp, "gm_comp": gm_comp,
        "nm_comp": nm_comp, "upside_comp": up_comp
    }
    return score, diag

def _fallback_dividend_score(row: pd.Series) -> Tuple[float, Dict[str, Any]]:
    """
    Enkel utdelnings-score – fokus på hög och hållbar yield.
    Ser på: dividend_yield, payout_cashflow (om finns), nettoskuld/EBITDA (om finns), D/E.
    """
    dy = _safe_float(row.get("Utdelningsyield"), np.nan)  # i %
    pr_cf = _safe_float(row.get("Payout (CF)"), np.nan)   # i %
    nde = _safe_float(row.get("NetDebt/EBITDA"), np.nan)
    de = _safe_float(row.get("Debt/Equity"), np.nan)

    dy_comp = 0.0 if np.isnan(dy) else min(max(dy, 0.0), 12.0) * 8.0  # 0..12% -> 0..96
    pr_comp = 0.0 if np.isnan(pr_cf) else max(0.0, 100.0 - min(pr_cf, 120.0))  # bäst nära 50–70
    nde_comp = 0.0 if np.isnan(nde) else max(0.0, 100.0 - min(nde, 6.0)*15.0)
    de_comp  = 0.0 if np.isnan(de) else max(0.0, 100.0 - min(de, 5.0)*12.0)

    score = 0.45*dy_comp + 0.25*pr_comp + 0.15*nde_comp + 0.15*de_comp
    score = float(np.clip(score, 0, 100))
    diag = {"yield_comp": dy_comp, "payout_cf_comp": pr_comp, "ndebitda_comp": nde_comp, "de_comp": de_comp}
    return score, diag

# ---------- Huvudvy ----------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    if df.empty:
        st.info("Ingen data i tabellen ännu.")
        return

    # Förbered härledda kolumner
    work = df.copy()
    if "P/S-snitt" not in work.columns:
        work["P/S-snitt"] = work.apply(_ps_avg, axis=1)

    # Klassificera börsvärde (för filter och etikett)
    if "Market Cap (nu) USD" in work.columns:
        work["Risklabel"] = work["Market Cap (nu) USD"].apply(_classify_mcap)
    else:
        # försök härleda MCap i bolagsvaluta – men risklabel kräver USD; då blir "Okänt"
        work["Risklabel"] = "Okänt"

    # Filter: sektor + cap-klass
    sectors = []
    if "Sektor" in work.columns:
        sectors = sorted([s for s in work["Sektor"].dropna().unique().tolist() if str(s).strip()])

    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        mode = st.radio("Typ av förslag", ["Tillväxt", "Utdelning"], horizontal=True)
    with c2:
        sector_filter = st.multiselect("Filtrera sektor", sectors, default=[])
    with c3:
        cap_filter = st.multiselect("Storleksklass", ["Microcap","Smallcap","Midcap","Largecap","Megacap"], default=[])

    if sector_filter and "Sektor" in work.columns:
        work = work[work["Sektor"].isin(sector_filter)]
    if cap_filter:
        work = work[work["Risklabel"].isin(cap_filter)]

    # Scorer (extern → state; annars fallback)
    scorer = None
    if mode == "Tillväxt":
        scorer = st.session_state.get("scorer_growth")
        fallback = _fallback_growth_score
    else:
        scorer = st.session_state.get("scorer_dividend")
        fallback = _fallback_dividend_score

    scores: List[float] = []
    diags: List[dict] = []
    for _, r in work.iterrows():
        try:
            if callable(scorer):
                s, d = scorer(r)
            else:
                s, d = fallback(r)
        except Exception:
            s, d = 0.0, {}
        scores.append(float(np.clip(s, 0, 100)))
        diags.append(d)
    work["Score"] = scores
    work["_diag"] = diags

    # Beräkna riktkurs/uppsida
    tgt, up, tpx = [], [], []
    for _, r in work.iterrows():
        a,b,c = _target_from_ps(r)
        tgt.append(a); up.append(b); tpx.append(c)
    work["Target MCap"] = tgt
    work["Uppsida"] = up
    work["Riktkurs"] = tpx
    work["Rating"] = work["Score"].apply(_rating_from_score)

    # Visa topp N
    top_n = st.slider("Visa topp N", 5, 50, 20, step=5)
    show_cols = ["Bolagsnamn","Ticker","Sektor","Risklabel","Score","Rating","P/S-snitt","Uppsida","Riktkurs"]
    show_cols = [c for c in show_cols if c in work.columns]
    show = work.sort_values(by="Score", ascending=False).head(top_n)

    # Infotext om fallback
    if not callable(scorer):
        st.warning("Vyn 'Investeringsförslag' kör **fallback-score** (inga externa scorer registrerade). "
                   "Du kan sätta `st.session_state['scorer_growth']` / `['scorer_dividend']` för att ta över poängsättning.")

    st.dataframe(
        show[show_cols],
        use_container_width=True,
        hide_index=True
    )

    # Detalj-expander per rad
    st.markdown("#### Detaljer & historik (för de visade)")
    for _, r in show.iterrows():
        with st.expander(f"📊 {r.get('Bolagsnamn','?')} ({r.get('Ticker','?')})"):
            mc_now = _current_mcap(r)
            valuta = r.get("Valuta","")
            st.markdown(f"- **Market Cap (nu):** {_fmt_large(mc_now)} ({valuta})")
            if _safe_float(r.get("P/S"), np.nan) > 0:
                st.markdown(f"- **P/S (nu):** {float(r.get('P/S')):.3f}")
            if _safe_float(r.get("P/S (Yahoo)"), np.nan) > 0:
                st.markdown(f"- **P/S (Yahoo):** {float(r.get('P/S (Yahoo)')):.3f}")
            if _safe_float(r.get("P/S-snitt"), np.nan) > 0:
                st.markdown(f"- **P/S-snitt (Q1..Q4):** {float(r.get('P/S-snitt')):.2f}")

            # Q-tabell om finns
            rows = []
            for i, q in enumerate(["Q1","Q2","Q3","Q4"], start=1):
                psq = r.get(f"P/S {q}")
                mcapq = r.get(f"MCap {q}")  # om du sparar detta
                dts = r.get(f"Periodslut {q}") or r.get(f"TS_P/S {q}")
                if _safe_float(psq, np.nan) > 0 or _safe_float(mcapq, np.nan) > 0 or (dts and str(dts).strip()):
                    rows.append({
                        "Period": q,
                        "P/S": (f"{float(psq):.3f}" if _safe_float(psq, np.nan) > 0 else "-"),
                        "MCap": (_fmt_large(mcapq) if _safe_float(mcapq, np.nan) > 0 else "-"),
                        "Datum (TTM-slut)": str(dts) if dts else "-"
                    })
            if rows:
                st.table(pd.DataFrame(rows))

            # Övriga nyckeltal
            d_e = r.get("Debt/Equity", None)
            gm  = r.get("Bruttomarginal", None)
            nm  = r.get("Nettomarginal", None)
            cash = r.get("Kassa", None)
            if d_e is not None: st.markdown(f"- **Debt/Equity:** {d_e}")
            if gm  is not None: st.markdown(f"- **Bruttomarginal:** {gm} %")
            if nm  is not None: st.markdown(f"- **Nettomarginal:** {nm} %")
            if cash is not None: st.markdown(f"- **Kassa:** {_fmt_large(_safe_float(cash, 0.0))} ({valuta})")

            # Riktkurs/uppsida
            if not np.isnan(r.get("Riktkurs", np.nan)):
                st.markdown(f"- **Riktkurs:** {float(r['Riktkurs']):.2f} {valuta}")
            if not np.isnan(r.get("Uppsida", np.nan)):
                st.markdown(f"- **Uppsida:** {_fmt_pct(float(r['Uppsida']))}")

            # Diagnostik från score
            diag = r.get("_diag") or {}
            if diag:
                with st.expander("Visa diagnos för score"):
                    st.json(diag)
