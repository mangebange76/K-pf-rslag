# stockapp/views/investment.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import streamlit as st
except Exception:
    st = None

# ------------------------------------------------------------
# Små helpers
# ------------------------------------------------------------

def _safe_float(v, default=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default

def _ps_avg_from_row(row: pd.Series) -> float:
    # använd existerande kolumn om finns, annars räkna
    if "P/S-snitt" in row and _safe_float(row.get("P/S-snitt"), 0.0) > 0:
        return float(row.get("P/S-snitt"))
    vals = []
    for k in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
        x = _safe_float(row.get(k, 0.0), 0.0)
        if x > 0:
            vals.append(x)
    return float(round(np.mean(vals), 2)) if vals else 0.0

def _format_large(v: float) -> str:
    # Human läsbar (t, bn, mn)
    try:
        n = float(v)
    except Exception:
        return "-"
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12:
        return f"{sign}{n/1e12:.2f} T"
    if n >= 1e9:
        return f"{sign}{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{sign}{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{sign}{n/1e3:.2f} K"
    return f"{sign}{n:.0f}"

def _cap_bucket(mcap: float) -> str:
    x = _safe_float(mcap, 0.0)
    # Gränser i lokal valuta (oftast USD för US). Det duger som heuristik.
    if x >= 2e11:   # >= 200B
        return "Mega"
    if x >= 1e10:
        return "Large"
    if x >= 2e9:
        return "Mid"
    if x >= 3e8:
        return "Small"
    return "Micro"

def _risk_label(mcap: float) -> str:
    m = _cap_bucket(mcap)
    return {
        "Mega": "Låg",
        "Large": "Låg–Medel",
        "Mid": "Medel",
        "Small": "Medel–Hög",
        "Micro": "Hög",
    }.get(m, "Okänd")

def _dividend_yield(row: pd.Series) -> float:
    div_ps = _safe_float(row.get("Årlig utdelning"), 0.0)
    px = _safe_float(row.get("Aktuell kurs"), 0.0)
    if div_ps > 0 and px > 0:
        return 100.0 * (div_ps / px)
    return 0.0

def _payout_ratio_fcf(row: pd.Series) -> float:
    # approx per share: FCF per aktie = FCF TTM / shares
    fcf = _safe_float(row.get("FCF TTM (valuta)"), 0.0)
    sh_m = _safe_float(row.get("Utestående aktier"), 0.0)  # i miljoner
    sh = sh_m * 1e6 if sh_m > 0 else 0.0
    div_ps = _safe_float(row.get("Årlig utdelning"), 0.0)
    if fcf > 0 and sh > 0 and div_ps > 0:
        fcf_ps = fcf / sh
        if fcf_ps > 0:
            return 100.0 * (div_ps / fcf_ps)
    return 0.0

def _score_clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

# ------------------------------------------------------------
# Scoring – Tillväxt
# ------------------------------------------------------------

def _growth_score(row: pd.Series) -> Tuple[float, Dict[str,float]]:
    ps_now = _safe_float(row.get("P/S"), 0.0)
    ps_avg = _ps_avg_from_row(row)
    cagr = _safe_float(row.get("CAGR 5 år (%)"), 0.0)
    gm   = _safe_float(row.get("Bruttomarginal (%)"), 0.0)
    nm   = _safe_float(row.get("Nettomarginal (%)"), 0.0)
    de   = _safe_float(row.get("Debt/Equity"), 0.0)
    runway = _safe_float(row.get("Runway (mån)"), 0.0)
    mcap = _safe_float(row.get("Market Cap (nu)"), 0.0)
    sektor = str(row.get("Sektor") or "")

    # undervaluation via P/S (lägre än snitt → bra)
    if ps_avg > 0 and ps_now > 0:
        underv = (ps_avg - ps_now) / ps_avg  # 0.2 betyder 20% under
        underv_s = _score_clip(0.5 + underv)  # ~[0..1]
    else:
        underv_s = 0.5

    # CAGR 0–40+% -> 0..1
    cagr_s = _score_clip(cagr / 40.0)

    # Marginaler – mix av gross och net
    gm_s = _score_clip(gm / 70.0)   # 70%+ ≈ topp
    nm_s = _score_clip((nm + 10.0) / 30.0)  # -10..20% -> 0..1
    margin_s = 0.6*gm_s + 0.4*nm_s

    # D/E – lägre bättre. approx score = 1/(1+D/E), clamp
    de_s = _score_clip(1.0 / (1.0 + max(0.0, de)))

    # Runway – skala ~0..24m
    runway_s = _score_clip(runway / 24.0)

    # Cap risk
    cap = _cap_bucket(mcap)
    cap_s = {"Mega": 1.0, "Large": 0.9, "Mid": 0.75, "Small": 0.55, "Micro": 0.35}.get(cap, 0.6)

    # Sektorjustering – enkelt: Tech & Health får lite mer vikt på tillväxt/marginal, mindre på cap
    w_underv, w_cagr, w_margin, w_de, w_runway, w_cap = 0.35, 0.2, 0.15, 0.1, 0.1, 0.1
    if sektor in ("Information Technology","Technology","Health Care","Healthcare"):
        w_cagr += 0.05; w_margin += 0.05; w_cap -= 0.05

    detail = {
        "underv": underv_s, "cagr": cagr_s, "margin": margin_s, "de": de_s, "runway": runway_s, "cap": cap_s
    }
    score = (
        w_underv * underv_s +
        w_cagr   * cagr_s   +
        w_margin * margin_s +
        w_de     * de_s     +
        w_runway * runway_s +
        w_cap    * cap_s
    )
    return float(round(score, 4)), detail

# ------------------------------------------------------------
# Scoring – Utdelning
# ------------------------------------------------------------

def _dividend_score(row: pd.Series) -> Tuple[float, Dict[str,float]]:
    dy = _dividend_yield(row)  # %
    payout_fcf = _payout_ratio_fcf(row)  # %
    de   = _safe_float(row.get("Debt/Equity"), 0.0)
    nm   = _safe_float(row.get("Nettomarginal (%)"), 0.0)
    mcap = _safe_float(row.get("Market Cap (nu)"), 0.0)
    ps_now = _safe_float(row.get("P/S"), 0.0)
    ps_avg = _ps_avg_from_row(row)

    # yield 0..8% → 0..1
    dy_s = _score_clip(dy / 8.0)

    # payout FCF – stegvis
    if payout_fcf <= 0:
        payout_s = 0.4 if dy > 0 else 0.5
    elif payout_fcf <= 60:
        payout_s = 1.0
    elif payout_fcf <= 100:
        payout_s = 0.6
    elif payout_fcf <= 150:
        payout_s = 0.35
    else:
        payout_s = 0.2

    # D/E
    de_s = _score_clip(1.0 / (1.0 + max(0.0, de)))

    # Net margin 0..20% → 0..1 (negativ → sänker)
    nm_s = _score_clip((nm + 5.0) / 25.0)

    # Cap stabilitet
    cap = _cap_bucket(mcap)
    cap_s = {"Mega": 1.0, "Large": 0.9, "Mid": 0.75, "Small": 0.55, "Micro": 0.35}.get(cap, 0.6)

    # Liten underviktning av undervaluation
    if ps_avg > 0 and ps_now > 0:
        underv = (ps_avg - ps_now) / ps_avg
        underv_s = _score_clip(0.5 + underv)
    else:
        underv_s = 0.5

    detail = {
        "yield": dy_s, "payout": payout_s, "de": de_s, "margin": nm_s, "cap": cap_s, "underv": underv_s
    }

    w_y, w_p, w_de, w_nm, w_cap, w_und = 0.35, 0.25, 0.15, 0.1, 0.1, 0.05
    score = (
        w_y   * dy_s +
        w_p   * payout_s +
        w_de  * de_s +
        w_nm  * nm_s +
        w_cap * cap_s +
        w_und * underv_s
    )
    return float(round(score, 4)), detail

# ------------------------------------------------------------
# Etikett utifrån score & ägarskap
# ------------------------------------------------------------

def _score_label(score: float, owned: bool = False) -> str:
    s = float(score)
    if s >= 0.80:
        return "Mycket bra (Köp)"
    if s >= 0.70:
        return "Bra (Köp)"
    if s >= 0.60:
        return "Fair (Behåll)"
    if s >= 0.50:
        return "Svag"
    # under 0.50
    return "Övervärderad / Undvik" if not owned else "Trimma / Sälj-varning"

# ------------------------------------------------------------
# Huvudvy
# ------------------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    if df is None or df.empty:
        st.info("Ingen data.")
        return

    # Robust kolumn-säkring (utan att skriva tillbaka)
    for c in ["Ticker","Bolagsnamn","Aktuell kurs","Valuta","Market Cap (nu)","Årlig utdelning",
              "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
              "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)","Sektor","Antal aktier"]:
        if c not in df.columns:
            df[c] = 0.0 if c not in ("Ticker","Bolagsnamn","Valuta","Sektor") else ""

    # Filterrad
    cols = st.columns([1,1,1,1.2])
    with cols[0]:
        mode = st.selectbox("Typ", ["Tillväxt", "Utdelning"])
    with cols[1]:
        sector_opts = sorted([s for s in df["Sektor"].dropna().astype(str).unique() if s.strip()])
        sectors = st.multiselect("Sektorer", sector_opts, default=[])
    with cols[2]:
        cap_choice = st.selectbox("Cap-bucket", ["Alla","Micro","Small","Mid","Large","Mega"], index=0)
    with cols[3]:
        riktkurs_val = st.selectbox("Riktkursfält", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)

    base = df.copy()

    # Deriverade fält
    base["P/S-snitt (calc)"] = base.apply(_ps_avg_from_row, axis=1)
    base["Dividend yield (%)"] = base.apply(_dividend_yield, axis=1)
    base["Payout FCF (%)"]    = base.apply(_payout_ratio_fcf, axis=1)
    base["CapBucket"]         = base["Market Cap (nu)"].apply(_cap_bucket)
    base["Risklabel"]         = base["Market Cap (nu)"].apply(_risk_label)

    # Filter
    if sectors:
        base = base[base["Sektor"].astype(str).isin(sectors)]
    if cap_choice != "Alla":
        base = base[base["CapBucket"] == cap_choice]

    # Rensa bort rader som saknar pris
    base = base[base["Aktuell kurs"] > 0].copy()
    if base.empty:
        st.info("Inget bolag matchar filtren.")
        return

    # Score
    if mode == "Tillväxt":
        scores = base.apply(lambda r: _growth_score(r), axis=1)
    else:
        scores = base.apply(lambda r: _dividend_score(r), axis=1)
    base["Score"] = [s for (s, _) in scores]
    base["_ScoreDetail"] = [d for (_, d) in scores]

    # Potential mot vald riktkurs
    rv = base.get(riktkurs_val)
    if rv is not None:
        base["Potential (%)"] = np.where(
            (base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0),
            (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0,
            0.0
        )
    else:
        base["Potential (%)"] = 0.0

    # Sortera – primärt på Score, sekundärt potential
    base = base.sort_values(by=["Score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # Nav/bläddrare (robust)
    N = len(base)
    if "prop_idx" not in st.session_state:
        st.session_state.prop_idx = 0
    st.session_state.prop_idx = int(max(0, min(st.session_state.prop_idx, N-1)))

    # Lista över toppkandidater (tabell)
    show_tbl = base[["Bolagsnamn","Ticker","Sektor","CapBucket","Risklabel","Score","Potential (%)","Aktuell kurs","P/S","P/S-snitt (calc)","Market Cap (nu)"]].copy()
    show_tbl["Market Cap (nu)"] = show_tbl["Market Cap (nu)"].apply(_format_large)
    show_tbl["Score"] = show_tbl["Score"].apply(lambda x: round(float(x), 3))
    show_tbl["Potential (%)"] = show_tbl["Potential (%)"].apply(lambda x: round(float(x), 1))
    st.dataframe(show_tbl.head(25), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Detaljkort för valt index
    if N > 0:
        colp, colmid, coln = st.columns([1,2,1])
        with colp:
            if st.button("⬅️ Föregående", key="prop_prev"):
                st.session_state.prop_idx = max(0, st.session_state.prop_idx - 1)
        with colmid:
            st.write(f"Förslag {st.session_state.prop_idx+1}/{N}")
        with coln:
            if st.button("➡️ Nästa", key="prop_next"):
                st.session_state.prop_idx = min(N-1, st.session_state.prop_idx + 1)

        r = base.iloc[st.session_state.prop_idx]
        score = float(r["Score"])
        owned = _safe_float(r.get("Antal aktier", 0.0), 0.0) > 0
        label = _score_label(score, owned=owned)

        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']}) — {label}")
        # huvudrad
        st.markdown(
            f"- **Aktuell kurs:** {round(_safe_float(r['Aktuell kurs']), 2)} {str(r.get('Valuta') or '')}  \n"
            f"- **Sektor:** {str(r.get('Sektor') or '-')}, **Cap:** {r['CapBucket']} ({r['Risklabel']})  \n"
            f"- **Score ({mode}):** {score:.3f}  \n"
            f"- **P/S (nu):** {round(_safe_float(r['P/S']),2)}  \n"
            f"- **P/S-snitt (4q):** {round(_safe_float(r['P/S-snitt (calc)']),2)}  \n"
            f"- **Market cap:** { _format_large(_safe_float(r['Market Cap (nu)'])) }"
        )

        # potentiell riktkurs & uppsida
        if _safe_float(r[riktkurs_val], 0.0) > 0:
            pot = _safe_float(r["Potential (%)"], 0.0)
            st.markdown(
                f"- **{riktkurs_val}:** {round(_safe_float(r[riktkurs_val]),2)} {str(r.get('Valuta') or '')} "
                f"(**Uppsida:** {pot:.1f}%)"
            )

        # Expander med fler nyckeltal
        with st.expander("Visa detaljerade nyckeltal"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Lönsamhet**")
                st.write(f"Bruttomarginal: {round(_safe_float(r['Bruttomarginal (%)']),1)}%")
                st.write(f"Nettomarginal: {round(_safe_float(r['Nettomarginal (%)']),1)}%")
                st.write(f"CAGR 5 år: {round(_safe_float(r.get('CAGR 5 år (%)',0.0)),1)}%")
            with c2:
                st.markdown("**Finansiell ställning**")
                st.write(f"Debt/Equity: {round(_safe_float(r['Debt/Equity']),2)}")
                st.write(f"Kassa: { _format_large(_safe_float(r['Kassa (valuta)'])) }")
                st.write(f"FCF TTM: { _format_large(_safe_float(r['FCF TTM (valuta)'])) }")
                st.write(f"Runway: { round(_safe_float(r['Runway (mån)']),1) } mån")
            with c3:
                st.markdown("**Utdelning**")
                st.write(f"Årlig utdelning/aktie: {round(_safe_float(r['Årlig utdelning']),2)} {str(r.get('Valuta') or '')}")
                st.write(f"Direktavkastning: {round(_dividend_yield(r),2)}%")
                pr = _payout_ratio_fcf(r)
                st.write(f"Payout (FCF, approx): {round(pr,1)}%")

            # Om du sparar historisk P/S Q1..Q4 i df – visa
            cols = []
            for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                if k in r.index and _safe_float(r[k], 0.0) > 0:
                    cols.append(f"{k}: {round(_safe_float(r[k]),2)}")
            if cols:
                st.markdown("**P/S historik:** " + " · ".join(cols))

    # Info-ruta om hur score funkar
    with st.expander("🔎 Hur räknas scoren?"):
        if mode == "Tillväxt":
            st.markdown(
                "- **Undervärdering (P/S vs snitt)** ~35%  \n"
                "- **CAGR 5 år** ~20%  \n"
                "- **Marginaler** ~15%  \n"
                "- **Debt/Equity** ~10%  \n"
                "- **Runway** ~10%  \n"
                "- **Cap-bucket** ~10%  \n"
                "_Sektorjustering_ ökar vikten för tillväxt/marginal i Tech/Healthcare."
            )
        else:
            st.markdown(
                "- **Direktavkastning** ~35%  \n"
                "- **Payout (FCF-baserad, approx)** ~25%  \n"
                "- **Debt/Equity** ~15%  \n"
                "- **Nettomarginal** ~10%  \n"
                "- **Cap-stabilitet** ~10%  \n"
                "- **Undervärdering** ~5%"
            )
