# stockapp/invest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .rates import hamta_valutakurs
from .utils import format_large_number

# ---------------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------------
def _risk_label_from_mcap(mcap: float) -> str:
    """
    Enkelt risklabel baserat på market cap i lokal valuta.
    Trösklarna är ungefärliga och kan justeras.
    """
    if mcap is None or not np.isfinite(mcap) or mcap <= 0:
        return "Okänd"
    # Trösklar (miljoner i lokal valuta): <300 => Micro, <2 000 => Small, <10 000 => Mid, <200 000 => Large, annars Mega
    if mcap < 300:           # < 0.3 B
        return "Micro"
    if mcap < 2_000:         # < 2 B
        return "Small"
    if mcap < 10_000:        # < 10 B
        return "Mid"
    if mcap < 200_000:       # < 200 B
        return "Large"
    return "Mega"


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    Procentuell skillnad (a/b - 1). Returnerar None om omöjligt.
    """
    if a is None or b is None:
        return None
    if b == 0:
        return None
    return a / b - 1.0


def _winsorize(series: pd.Series, p: float = 0.02) -> pd.Series:
    """
    Skär av extrema outliers (för robustare normalisering).
    """
    if series.empty:
        return series
    lo = series.quantile(p)
    hi = series.quantile(1 - p)
    return series.clip(lower=lo, upper=hi)


def _normalize_higher_better(s: pd.Series) -> pd.Series:
    """
    0–1 normalisering där högre är bättre.
    """
    s2 = _winsorize(s.astype(float))
    mn, mx = s2.min(), s2.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn == 0:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s2 - mn) / (mx - mn)


def _normalize_lower_better(s: pd.Series) -> pd.Series:
    """
    0–1 normalisering där lägre är bättre.
    """
    return 1.0 - _normalize_higher_better(s)


def _calc_target_price_row(row: pd.Series) -> Optional[float]:
    """
    Räkna riktkurs om den inte finns:
    pris = (P/S-snitt * Omsättning nästa år [miljoner]) / Utestående aktier [miljoner]
    Antas vara i samma valuta som aktien.
    """
    # Om det redan finns en riktkurs för 1 år – använd den
    for col in ["Riktkurs om 1 år", "Riktkurs idag"]:
        if col in row.index:
            v = _safe_float(row.get(col))
            if v and v > 0:
                return float(v)

    ps_s = _safe_float(row.get("P/S-snitt"))
    rev_next = _safe_float(row.get("Omsättning nästa år"))
    shares_m = _safe_float(row.get("Utestående aktier"))
    if ps_s and rev_next and shares_m and shares_m > 0:
        return float((ps_s * rev_next) / shares_m)
    return None


def _completeness_score(row: pd.Series, expected_cols: List[str]) -> float:
    """
    Hur komplett radens data är (0..1). Bara kolumner som faktiskt existerar i df räknas.
    """
    have = 0
    total = 0
    for c in expected_cols:
        if c in row.index:
            total += 1
            if row.get(c) not in (None, "", np.nan):
                try:
                    # För numeriska: kräver finite
                    if isinstance(row.get(c), (int, float, np.floating)):
                        if np.isfinite(float(row.get(c))):
                            have += 1
                    else:
                        have += 1
                except Exception:
                    pass
    if total == 0:
        return 0.3  # minimistraff om vi inte vet något alls om schemat
    return have / total


def _valuation_gap(row: pd.Series) -> Optional[float]:
    """
    Gap = target_price / current_price - 1.
    """
    cur = _safe_float(row.get("Aktuell kurs"))
    tgt = _calc_target_price_row(row)
    if cur and tgt:
        return tgt / cur - 1.0
    return None


def _label_from_gap_and_score(gap: Optional[float], score: Optional[float]) -> str:
    """
    Köp/Håll/Trimma/Sälj baserat på värderingsgap och totalpoäng.
    Tydliga trösklar som kan finjusteras.
    """
    if gap is None or score is None:
        return "Osäker"

    # Gap i %
    g = gap * 100.0

    # Grov tröskellogik:
    if g >= 30 and score >= 0.65:
        return "Köp"
    if g >= 10 and score >= 0.55:
        return "Överväg köp"
    if -10 <= g < 10:
        return "Behåll"
    if g < -10 and score < 0.55:
        return "Trimma"
    if g < -25 and score < 0.50:
        return "Sälj"
    return "Behåll"


# ---------------------------------------------------------------------
# Viktning per fokus och per sektor
# ---------------------------------------------------------------------
DEFAULT_EXPECTED_COLS = [
    "Aktuell kurs", "P/S", "P/S-snitt", "Omsättning idag", "Omsättning nästa år",
    "Utestående aktier", "Market Cap", "Bruttomarginal (%)", "Nettomarginal (%)",
    "Debt/Equity", "EV/EBITDA", "FCF (ttm)", "Kassa & ekvivalenter", "Utdelningsyield (%)"
]

# Basvikter (Mix)
BASE_WEIGHTS = {
    "P/S-snitt_rel": 0.18,        # lägre än sektor-median är bra
    "GrossMargin": 0.12,          # högre bättre
    "NetMargin": 0.12,            # högre bättre
    "DebtToEquity": 0.14,         # lägre bättre
    "EVEBITDA_rel": 0.12,         # lägre bättre (om finns)
    "FCF_Positive": 0.10,         # positivt FCF premieras
    "DividendYield": 0.07,        # högre bättre (men inte allt)
    "ValuationGap": 0.15          # högre bättre
}

# Justering per fokus
FOCUS_ADJUST = {
    "Mix":   {"DividendYield": 1.0, "ValuationGap": 1.0, "P/S-snitt_rel": 1.0, "EVEBITDA_rel": 1.0},
    "Tillväxt": {"DividendYield": 0.4, "ValuationGap": 1.1, "P/S-snitt_rel": 1.2, "EVEBITDA_rel": 0.9},
    "Utdelning": {"DividendYield": 1.6, "ValuationGap": 0.9, "P/S-snitt_rel": 0.8, "EVEBITDA_rel": 1.1},
}

# Sektorspecifika nyanser (lätta, så det funkar även när data saknas)
SECTOR_TILT = {
    "Technology":   {"P/S-snitt_rel": 1.15, "GrossMargin": 1.1, "EVEBITDA_rel": 0.9},
    "Consumer Cyclical": {"P/S-snitt_rel": 1.05, "NetMargin": 1.05},
    "Consumer Defensive": {"DividendYield": 1.2, "DebtToEquity": 1.05},
    "Healthcare":   {"GrossMargin": 1.1, "NetMargin": 1.05},
    "Energy":       {"EVEBITDA_rel": 1.2, "DebtToEquity": 1.1, "DividendYield": 1.1},
    "Financial Services": {"EVEBITDA_rel": 1.15, "DebtToEquity": 1.15},
    "Industrials":  {"EVEBITDA_rel": 1.1, "NetMargin": 1.05},
    "Materials":    {"EVEBITDA_rel": 1.1, "DebtToEquity": 1.05},
    "Utilities":    {"DividendYield": 1.4, "DebtToEquity": 1.1},
    "Real Estate":  {"DividendYield": 1.5, "DebtToEquity": 1.15},
}


def _sector_weighted_score(df: pd.DataFrame, focus: str) -> pd.Series:
    """
    Räknar ut en totalpoäng 0..1 per rad, med sektorvikter och fokusjustering.
    Använder robust normalisering mot sektormedianer / winsorize.
    """
    work = df.copy()

    # Sektormedianer för relativa mått
    def sector_median(col: str) -> pd.Series:
        if col not in work.columns:
            return pd.Series([np.nan] * len(work), index=work.index)
        return work.groupby("Sektor")[col].transform("median")

    # Bygg features
    # Relativ P/S mot sektor (lägre bättre)
    if "P/S-snitt" in work.columns:
        ps_rel = work["P/S-snitt"] / sector_median("P/S-snitt")
        work["_feat_ps_rel"] = _normalize_lower_better(ps_rel.replace([np.inf, -np.inf], np.nan).fillna(ps_rel.median()))
    else:
        work["_feat_ps_rel"] = 0.5

    # EV/EBITDA (lägre bättre)
    if "EV/EBITDA" in work.columns:
        work["_feat_ebitda_rel"] = _normalize_lower_better(_winsorize(work["EV/EBITDA"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(work["EV/EBITDA"].median())))
    else:
        work["_feat_ebitda_rel"] = 0.5

    # Debt/Equity (lägre bättre)
    if "Debt/Equity" in work.columns:
        work["_feat_de"] = _normalize_lower_better(_winsorize(work["Debt/Equity"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(work["Debt/Equity"].median())))
    else:
        work["_feat_de"] = 0.5

    # Marginaler
    if "Bruttomarginal (%)" in work.columns:
        work["_feat_gm"] = _normalize_higher_better(_winsorize(work["Bruttomarginal (%)"].astype(float).fillna(work["Bruttomarginal (%)"].median())))
    else:
        work["_feat_gm"] = 0.5
    if "Nettomarginal (%)" in work.columns:
        work["_feat_nm"] = _normalize_higher_better(_winsorize(work["Nettomarginal (%)"].astype(float).fillna(work["Nettomarginal (%)"].median())))
    else:
        work["_feat_nm"] = 0.5

    # FCF positivt?
    if "FCF (ttm)" in work.columns:
        fcf = work["FCF (ttm)"].astype(float)
        work["_feat_fcfpos"] = (fcf > 0).astype(float)  # 1 om positivt, annars 0
    else:
        work["_feat_fcfpos"] = 0.5

    # Utdelningsyield (högre bättre)
    if "Utdelningsyield (%)" in work.columns:
        work["_feat_div"] = _normalize_higher_better(_winsorize(work["Utdelningsyield (%)"].astype(float).fillna(0.0)))
    else:
        work["_feat_div"] = 0.5

    # Värderingsgap (högre bättre)
    # Beräkna per rad
    gaps = []
    for _, r in work.iterrows():
        gaps.append(_safe_float(_valuation_gap(r)))
    gaps = pd.Series([g if g is not None else np.nan for g in gaps], index=work.index)
    work["_feat_gap"] = _normalize_higher_better(_winsorize(gaps.fillna(gaps.median() if np.isfinite(gaps.median()) else 0.0)))

    # Basvikter + fokusjustering + sektortilt
    w = BASE_WEIGHTS.copy()
    adj = FOCUS_ADJUST.get(focus, FOCUS_ADJUST["Mix"])
    for k in w:
        if k in adj:
            w[k] *= adj[k]

    # Sektortilt per rad
    sector_tilt = []
    for _, r in work.iterrows():
        s = r.get("Sektor", "Other")
        tilt = SECTOR_TILT.get(s, {})
        sector_tilt.append(tilt)
    # Beräkna totalpoäng
    score = (
        work["_feat_ps_rel"]   * (w["P/S-snitt_rel"]  * np.array([t.get("P/S-snitt_rel", 1.0) for t in sector_tilt])) +
        work["_feat_gm"]       * (w["GrossMargin"]    * np.array([t.get("GrossMargin", 1.0) for t in sector_tilt])) +
        work["_feat_nm"]       * (w["NetMargin"]      * np.array([t.get("NetMargin", 1.0) for t in sector_tilt])) +
        work["_feat_de"]       * (w["DebtToEquity"]   * np.array([t.get("DebtToEquity", 1.0) for t in sector_tilt])) +
        work["_feat_ebitda_rel"] * (w["EVEBITDA_rel"] * np.array([t.get("EVEBITDA_rel", 1.0) for t in sector_tilt])) +
        work["_feat_fcfpos"]   * (w["FCF_Positive"]   * np.array([t.get("FCF_Positive", 1.0) for t in sector_tilt])) +
        work["_feat_div"]      * (w["DividendYield"]  * np.array([t.get("DividendYield", 1.0) for t in sector_tilt])) +
        work["_feat_gap"]      * (w["ValuationGap"]   * np.array([t.get("ValuationGap", 1.0) for t in sector_tilt]))
    )

    # Normalisera viktsumman så att summan ~1.0
    wsum = (
        (w["P/S-snitt_rel"]) + (w["GrossMargin"]) + (w["NetMargin"]) + (w["DebtToEquity"]) +
        (w["EVEBITDA_rel"]) + (w["FCF_Positive"]) + (w["DividendYield"]) + (w["ValuationGap"])
    )
    score = score / max(wsum, 1e-9)

    # Datakompletthets-penalty per rad
    expected = DEFAULT_EXPECTED_COLS[:]
    comp = []
    for _, r in work.iterrows():
        comp.append(_completeness_score(r, expected))
    comp = np.array(comp)

    # Liten exponent för att straffa ofullständighet men inte döma ut helt
    final_score = np.clip(score * (0.6 + 0.4 * comp), 0.0, 1.0)
    return pd.Series(final_score, index=work.index)


# ---------------------------------------------------------------------
# Huvudvy
# ---------------------------------------------------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("🧭 Investeringsförslag")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    base = df.copy()

    # Risklabel om saknas
    if "Risklabel" not in base.columns:
        base["Risklabel"] = base.get("Market Cap", 0).apply(lambda x: _risk_label_from_mcap(_safe_float(x)))

    # Filterpanel
    cols_f = st.columns([1, 1, 1, 2])
    with cols_f[0]:
        fokus = st.selectbox("Fokus", ["Mix", "Tillväxt", "Utdelning"], index=0)
    with cols_f[1]:
        sektorer = sorted([s for s in base.get("Sektor", pd.Series(["Okänd"])).dropna().unique().tolist() if s])
        valda_sektorer = st.multiselect("Sektorfilter", options=sektorer, default=sektorer)
    with cols_f[2]:
        risk_opts = ["Micro", "Small", "Mid", "Large", "Mega", "Okänd"]
        valda_risk = st.multiselect("Risklabel", options=risk_opts, default=risk_opts)
    with cols_f[3]:
        cap_filter_tip = "Filtrera bort bolag utan riktkurs-beräkning?"
        must_have_target = st.checkbox("Kräv beräknad riktkurs", value=False, help=cap_filter_tip)

    # Filtrera
    if "Sektor" in base.columns:
        base = base[base["Sektor"].isin(valda_sektorer)]
    base = base[base["Risklabel"].isin(valda_risk)]

    # Beräkna riktkurs och gap
    tprices = []
    gaps = []
    for _, r in base.iterrows():
        tp = _calc_target_price_row(r)
        tprices.append(tp)
        gaps.append(_safe_float(_valuation_gap(r)))
    base["_TargetPrice"] = tprices
    base["_ValGap"] = gaps

    if must_have_target:
        base = base[base["_TargetPrice"].notna()]

    if base.empty:
        st.warning("Inga bolag matchar filtren.")
        return

    # Totalpoäng sektor-viktad
    score = _sector_weighted_score(base, fokus)
    base["_Score"] = score

    # Köp/Håll/Trimma/Sälj
    labels = []
    for _, r in base.iterrows():
        labels.append(_label_from_gap_and_score(r["_ValGap"], r["_Score"]))
    base["_Betyg"] = labels

    # Sortera på totalpoäng (högst först)
    base = base.sort_values(by="_Score", ascending=False).reset_index(drop=True)

    # Bläddringsfunktion
    st.session_state.setdefault("invest_idx", 0)
    st.session_state["invest_idx"] = int(
        np.clip(st.session_state.get("invest_idx", 0), 0, max(0, len(base) - 1))
    )

    col_prev, col_mid, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("⬅️ Föregående", key="inv_prev"):
            st.session_state["invest_idx"] = max(0, st.session_state["invest_idx"] - 1)
    with col_next:
        if st.button("➡️ Nästa", key="inv_next"):
            st.session_state["invest_idx"] = min(len(base) - 1, st.session_state["invest_idx"] + 1)
    with col_mid:
        st.write(f"Post {st.session_state['invest_idx'] + 1}/{len(base)}")

    r = base.iloc[st.session_state["invest_idx"]]

    # Kort toppsammanfattning
    top_cols = [c for c in [
        "Ticker", "Bolagsnamn", "Sektor", "Risklabel",
        "Aktuell kurs", "Valuta", "Market Cap", "P/S-snitt",
        "_TargetPrice", "_ValGap", "_Score", "_Betyg"
    ] if c in base.columns]
    head_df = pd.DataFrame([r[top_cols].to_dict()])

    # Formatera lite
    if "Market Cap" in head_df.columns:
        head_df["Market Cap"] = head_df["Market Cap"].apply(lambda x: format_large_number(_safe_float(x) or 0.0, r.get("Valuta", "SEK")))
    if "_ValGap" in head_df.columns:
        head_df["_ValGap"] = head_df["_ValGap"].apply(lambda x: f"{x*100:.1f}%" if x is not None else "–")
    if "_Score" in head_df.columns:
        head_df["_Score"] = head_df["_Score"].apply(lambda s: f"{s:.2f}")

    st.subheader(f"{r.get('Bolagsnamn','?')} ({r.get('Ticker','?')})")
    st.dataframe(head_df, use_container_width=True, hide_index=True)

    # Expander: Nyckeltal
    with st.expander("📊 Nyckeltal & detaljer", expanded=False):
        show_keys = [
            # Värdering/omsättning
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
            "Omsättning idag", "Omsättning nästa år",
            # Multiplar
            "EV/EBITDA", "Debt/Equity",
            # Marginaler
            "Bruttomarginal (%)", "Nettomarginal (%)",
            # Kassaflöde/kassa
            "FCF (ttm)", "Kassa & ekvivalenter", "Kassaburn (ttm)", "Runway (kvartal)",
            # Utdelning
            "Utdelningsyield (%)", "Årlig utdelning",
            # Antal aktier
            "Utestående aktier", "Antal aktier",
            # Riktkurser
            "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
            # Tidsstämplar
            "Senast manuellt uppdaterad", "Senast auto-uppdaterad", "Senast uppdaterad källa",
        ]

        rows = []
        for k in show_keys:
            if k in base.columns:
                val = r.get(k)
                # lätt formattering
                if k in ("FCF (ttm)", "Kassa & ekvivalenter", "Kassaburn (ttm)"):
                    rows.append((k, format_large_number(_safe_float(val) or 0.0, r.get("Valuta", "SEK"))))
                elif k in ("Omsättning idag", "Omsättning nästa år"):
                    rows.append((k, format_large_number(_safe_float(val) or 0.0, r.get("Valuta", "SEK"))))
                elif k == "Utdelningsyield (%)":
                    rows.append((k, f"{_safe_float(val) or 0:.2f}%"))
                elif k.startswith("Riktkurs"):
                    rows.append((k, f"{_safe_float(val) or 0:.2f} {r.get('Valuta','')}"))
                else:
                    # default
                    rows.append((k, val if (val is not None and val != "") else "–"))
        if "_TargetPrice" in r.index:
            rows.append(("Beräknad riktkurs (fallback)", f"{_safe_float(r['_TargetPrice']) or 0:.2f} {r.get('Valuta','')}"))
        if "_ValGap" in r.index:
            rows.append(("Värderingsgap mot mål", f"{(r['_ValGap']*100.0):.1f}%" if r["_ValGap"] is not None else "–"))
        rows.append(("Totalpoäng (0–1)", f"{r.get('_Score',0):.2f}"))
        rows.append(("Betyg", r.get("_Betyg", "Osäker")))
        det_df = pd.DataFrame(rows, columns=["Nyckeltal", "Värde"])
        st.dataframe(det_df, use_container_width=True, hide_index=True)

    # Kort motivering
    st.markdown("#### 🧠 Motivering")
    bullets = []
    if r.get("_ValGap") is not None:
        g = r["_ValGap"] * 100.0
        if g >= 20:
            bullets.append("Stor uppsida relativt mål (värderingsgap ≥ 20%).")
        elif g <= -10:
            bullets.append("Negativt gap mot mål (risk för övervärdering).")
    if r.get("Debt/Equity") is not None:
        de = _safe_float(r["Debt/Equity"]) or 0.0
        if de < 0.8:
            bullets.append("Låg skuldsättning.")
        elif de > 2.0:
            bullets.append("Hög skuldsättning – bevaka.")
    if r.get("FCF (ttm)") is not None:
        fcf = _safe_float(r["FCF (ttm)"]) or 0.0
        bullets.append("Positivt fritt kassaflöde." if fcf > 0 else "Negativt fritt kassaflöde.")
    if r.get("Utdelningsyield (%)") is not None:
        y = _safe_float(r["Utdelningsyield (%)"]) or 0.0
        if y >= 4:
            bullets.append("Attraktiv direktavkastning.")
    if not bullets:
        bullets.append("Underlag delvis ofullständigt – bedömning med förbehåll.")
    st.write("\n".join([f"- {b}" for b in bullets]))

    # Liten topplista (för sammanhang)
    st.divider()
    st.subheader("🏆 Topp 10 enligt din filtrering")
    show = ["Ticker", "Bolagsnamn", "Sektor", "Risklabel", "_Score", "_Betyg", "_ValGap", "_TargetPrice", "P/S-snitt"]
    show = [c for c in show if c in base.columns]
    top = base[show].head(10).copy()
    if "_ValGap" in top.columns:
        top["_ValGap"] = top["_ValGap"].apply(lambda x: f"{x*100:.1f}%" if x is not None else "–")
    if "_Score" in top.columns:
        top["_Score"] = top["_Score"].apply(lambda s: f"{s:.2f}")
    if "_TargetPrice" in top.columns:
        # visa med valuta om möjligt (blandad valuta -> lämna utan)
        top["_TargetPrice"] = top["_TargetPrice"].apply(lambda x: f"{x:.2f}" if x is not None else "–")
    st.dataframe(top.reset_index(drop=True), use_container_width=True, hide_index=True)
