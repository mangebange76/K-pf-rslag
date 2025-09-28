# stockapp/views/invest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

# -----------------------------
# Små helpers
# -----------------------------

def _human_mc(val: float, suffix_ccy: str = "") -> str:
    try:
        x = float(val)
    except Exception:
        return "-"
    units = [("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)]
    for u, m in units:
        if abs(x) >= m:
            return f"{x/m:,.2f} {u}{(' ' + suffix_ccy) if suffix_ccy else ''}"
    return f"{x:,.0f}{(' ' + suffix_ccy) if suffix_ccy else ''}"

def _ps_snitt(row: pd.Series) -> float:
    vals = []
    for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
        if k in row and pd.notna(row[k]) and float(row[k]) > 0:
            vals.append(float(row[k]))
    return round(float(np.mean(vals)), 2) if vals else 0.0

def _risk_label(mcap: float) -> str:
    # enkla bucket-gränser (USD-liknande; vi visar bara label)
    if mcap >= 200e9: return "Mega"
    if mcap >= 10e9:  return "Large"
    if mcap >= 2e9:   return "Mid"
    if mcap >= 0.3e9: return "Small"
    return "Micro"

def _div_yield(annual_div: float, price: float) -> float:
    try:
        if price > 0:
            return float(annual_div) / float(price) * 100.0
    except Exception:
        pass
    return 0.0

def _grade(value: float, good_thr: float, ok_thr: float, reverse: bool = False) -> str:
    """
    quick grader:
      reverse=False:  >=good -> Bra, >=ok -> Ok, else Svagt
      reverse=True:   <=good -> Bra, <=ok -> Ok, else Svagt  (för nyckeltal där lägre är bättre)
    """
    try:
        v = float(value)
    except Exception:
        return "Svagt"
    if not reverse:
        if v >= good_thr: return "Bra"
        if v >= ok_thr:   return "Ok"
        return "Svagt"
    else:
        if v <= good_thr: return "Bra"
        if v <= ok_thr:   return "Ok"
        return "Svagt"

def _get(df: pd.DataFrame, row: pd.Series, key: str, default=0.0):
    try:
        v = row.get(key, default)
        # robust float när det förväntas
        if isinstance(default, (int, float)):
            return float(v) if pd.notna(v) else float(default)
        return v if v is not None else default
    except Exception:
        return default

# -----------------------------
# Scoring
# -----------------------------

def _score_row(row: pd.Series, mode: str, riktkurs_col: str) -> Dict[str, float]:
    """
    Beräknar delpoäng och totalpoäng:
      - valuation: uppsida mot valgt riktkurs
      - quality: marginaler + D/E (lägre är bättre)
      - risk: bucket-penalty för Micro/Small
      - mode "Tillväxt": lägger vikt på CAGR 5 år och P/S-snitt (lägre bättre)
      - mode "Utdelning": viktar utdelningsyield + payout-sustain proxys (FCF > 0, D/E låg)
    """
    price = _get(None, row, "Aktuell kurs", 0.0)
    target = _get(None, row, riktkurs_col, 0.0)
    mcap = _get(None, row, "Market Cap (nu)", 0.0)
    ps_now = _get(None, row, "P/S", 0.0)
    ps_avg = _ps_snitt(row)

    de = _get(None, row, "Debt/Equity", 0.0)
    gm = _get(None, row, "Bruttomarginal (%)", 0.0)
    nm = _get(None, row, "Nettomarginal (%)", 0.0)
    fcf = _get(None, row, "FCF TTM (valuta)", 0.0)
    cash = _get(None, row, "Kassa (valuta)", 0.0)
    cagr = _get(None, row, "CAGR 5 år (%)", 0.0)
    annual_div = _get(None, row, "Årlig utdelning", 0.0)
    dy = _div_yield(annual_div, price)

    # valuation: uppsida %
    upside = 0.0
    if price > 0 and target > 0:
        upside = (target - price) / price * 100.0
    # clamp
    upside_c = max(min(upside, 150.0), -90.0)
    valuation_score = (upside_c + 90.0) / 240.0  # normalisera ca 0..1

    # quality
    # D/E (lägre bättre), GM/NM (högre bättre), FCF positivt
    de_score = 1.0 if de <= 0.5 else (0.7 if de <= 1.0 else (0.4 if de <= 2.0 else 0.2))
    gm_score = 0.2 + max(0.0, min(gm / 80.0, 0.8))  # 80% ~ topp
    nm_score = 0.2 + max(0.0, min(nm / 40.0, 0.8))  # 40% ~ topp
    fcf_score = 0.8 if fcf > 0 else 0.3
    quality_score = (de_score*0.3 + gm_score*0.35 + nm_score*0.2 + fcf_score*0.15)

    # risk (cap-bucket penalty)
    risk_pen = 0.0
    if mcap < 0.3e9:          risk_pen = 0.25   # micro
    elif mcap < 2e9:          risk_pen = 0.15   # small
    elif mcap < 10e9:         risk_pen = 0.08   # mid
    else:                     risk_pen = 0.03   # large/mega
    risk_score = 1.0 - risk_pen

    # mode-specific
    if mode == "Tillväxt":
        # lägre P/S-snitt är bättre, högre CAGR bättre
        ps_score = 0.8 if ps_avg <= 5 else (0.6 if ps_avg <= 10 else (0.4 if ps_avg <= 20 else 0.2))
        cagr_score = 0.2 + max(0.0, min(cagr/50.0, 0.8))  # 50% CAGR ~ topp
        mode_score = ps_score*0.45 + cagr_score*0.55
    else:  # Utdelning
        # högre yield bättre men cap även för risk
        dy_score = 0.2 + max(0.0, min(dy/10.0, 0.8))  # 10% ~ topp
        # hållbarhet: positiv FCF + låg D/E
        sustain = (1.0 if fcf > 0 else 0.4) * (1.0 if de <= 1.0 else 0.6)
        mode_score = dy_score*0.6 + sustain*0.4

    total = valuation_score*0.35 + quality_score*0.35 + risk_score*0.1 + mode_score*0.2
    return {
        "Upside (%)": upside,
        "ValuationScore": valuation_score,
        "QualityScore": quality_score,
        "RiskScore": risk_score,
        "ModeScore": mode_score,
        "TotalScore": total,
    }

# -----------------------------
# Huvudvy
# -----------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Val av läge och riktkurs
    mode = st.radio("Köp-läge", ["Tillväxt","Utdelning"], horizontal=True, index=0)
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )
    subset = st.radio("Urval", ["Alla bolag","Endast portfölj"], horizontal=True)

    # Filter: sektor & cap-buckets
    sectors = sorted([s for s in df.get("Sektor", pd.Series([])).astype(str).unique() if s and s != "nan"])
    pick_sectors = st.multiselect("Filtrera på sektor", options=sectors, default=[])
    cap_opts = ["Micro","Small","Mid","Large","Mega"]
    pick_caps = st.multiselect("Filtrera på börsvärde-bucket", options=cap_opts, default=[])

    # Basdata för urval
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base.get("Antal aktier", 0) > 0].copy()

    # behov: pris + riktkurs + mcap
    need_cols = ["Aktuell kurs", riktkurs_val, "Market Cap (nu)"]
    for c in need_cols:
        if c not in base.columns:
            base[c] = 0.0
    base = base[(base["Aktuell kurs"] > 0) & (base[riktkurs_val] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar uppsatta kriterier (saknar pris/riktkurs).")
        return

    # sektorfilter
    if pick_sectors:
        base = base[base["Sektor"].astype(str).isin(pick_sectors)].copy()

    # cap bucket beräkning + filter
    if "Market Cap (nu)" not in base.columns:
        base["Market Cap (nu)"] = 0.0
    base["_Risklabel"] = base["Market Cap (nu)"].apply(_risk_label)
    if pick_caps:
        base = base[base["_Risklabel"].isin(pick_caps)].copy()

    if base.empty:
        st.info("Inga bolag kvar efter filtrering.")
        return

    # beräkna P/S-snitt + score
    base["P/S-snitt"] = base.apply(_ps_snitt, axis=1)
    scores = []
    for i, r in base.iterrows():
        sc = _score_row(r, mode=mode, riktkurs_col=riktkurs_val)
        for k, v in sc.items():
            base.at[i, k] = v
        scores.append(sc["TotalScore"])

    # sortera på högst poäng
    base = base.sort_values(by="TotalScore", ascending=False).reset_index(drop=True)

    # stabil bläddring
    key_idx = "inv_idx"
    st.session_state.setdefault(key_idx, 0)
    st.session_state[key_idx] = min(st.session_state[key_idx], max(0, len(base)-1))

    st.write(f"Förslag {st.session_state[key_idx]+1}/{len(base)}")

    col_prev, col_next = st.columns([1,1])
    with col_prev:
        if st.button("⬅️ Föregående", use_container_width=True):
            st.session_state[key_idx] = max(0, st.session_state[key_idx]-1)
    with col_next:
        if st.button("➡️ Nästa", use_container_width=True):
            st.session_state[key_idx] = min(len(base)-1, st.session_state[key_idx]+1)

    r = base.iloc[st.session_state[key_idx]]

    # presentation
    st.subheader(f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})")

    # huvudrader
    price = float(r.get("Aktuell kurs", 0.0) or 0.0)
    target = float(r.get(riktkurs_val, 0.0) or 0.0)
    mcap_now = float(r.get("Market Cap (nu)", 0.0) or 0.0)
    valuta = str(r.get("Valuta",""))
    ps_now = float(r.get("P/S", 0.0) or 0.0)
    ps_avg = float(r.get("P/S-snitt", 0.0) or 0.0)
    shares_m = float(r.get("Utestående aktier", 0.0) or 0.0)

    cols = st.columns(3)
    with cols[0]:
        st.metric("Aktuell kurs", f"{price:.2f} {valuta}")
        st.metric("Riktkurs (vald)", f"{target:.2f} {valuta}")
        st.metric("Uppsida", f"{float(r.get('Upside (%)',0.0)):.1f} %")
    with cols[1]:
        st.metric("Market Cap (nu)", _human_mc(mcap_now, valuta))
        st.metric("P/S (nu)", f"{ps_now:.2f}" if ps_now>0 else "-")
        st.metric("P/S-snitt (Q1–Q4)", f"{ps_avg:.2f}" if ps_avg>0 else "-")
    with cols[2]:
        st.metric("Utestående aktier", f"{shares_m:,.2f} M" if shares_m>0 else "-")
        st.metric("Risklabel", str(r.get("_Risklabel","-")))
        st.metric("Sektor", str(r.get("Sektor","-")))

    # expanders
    with st.expander("Nyckeltal & diagnos", expanded=False):
        de = float(r.get("Debt/Equity", 0.0) or 0.0)
        gm = float(r.get("Bruttomarginal (%)", 0.0) or 0.0)
        nm = float(r.get("Nettomarginal (%)", 0.0) or 0.0)
        fcf = float(r.get("FCF TTM (valuta)", 0.0) or 0.0)
        cash = float(r.get("Kassa (valuta)", 0.0) or 0.0)
        runway = float(r.get("Runway (kvartal)", 0.0) or 0.0)
        cagr = float(r.get("CAGR 5 år (%)", 0.0) or 0.0)
        dy = _div_yield(float(r.get("Årlig utdelning",0.0) or 0.0), price)

        diag = {
            "Debt/Equity": _grade(de, 0.5, 1.0, reverse=True),
            "Bruttomarginal": _grade(gm, 50, 30, reverse=False),
            "Nettomarginal": _grade(nm, 20, 10, reverse=False),
            "FCF TTM": "Bra" if fcf > 0 else "Svagt",
            "Kassa": "Bra" if cash > 0 else "Svagt",
            "Runway (kvartal)": "Bra" if runway >= 8 else ("Ok" if runway >= 4 else "Svagt"),
            "CAGR 5 år": _grade(cagr, 25, 10, reverse=False),
            "Direktavkastning": _grade(dy, 5, 3, reverse=False) if mode=="Utdelning" else ("Info"),
        }

        df_kpis = pd.DataFrame([
            ["Debt/Equity", f"{de:.2f}", diag["Debt/Equity"]],
            ["Bruttomarginal (%)", f"{gm:.1f}", diag["Bruttomarginal"]],
            ["Nettomarginal (%)", f"{nm:.1f}", diag["Nettomarginal"]],
            ["FCF TTM", _human_mc(fcf, valuta), diag["FCF TTM"]],
            ["Kassa", _human_mc(cash, valuta), diag["Kassa"]],
            ["Runway (kvartal)", f"{runway:.1f}", diag["Runway (kvartal)"]],
            ["CAGR 5 år (%)", f"{cagr:.1f}", diag["CAGR 5 år"]],
            ["Direktavkastning (%)", f"{dy:.2f}", diag["Direktavkastning"]],
        ], columns=["Nyckeltal","Värde","Bedömning"])
        st.dataframe(df_kpis, hide_index=True, use_container_width=True)

    with st.expander("Historik: P/S & MCAP (om tillgängligt)", expanded=False):
        cols_ps = ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]
        data = []
        for k in cols_ps:
            v = r.get(k, None)
            data.append(f"{float(v):.2f}" if v and float(v)>0 else "-")
        mcaps = []
        for k in ["MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4"]:
            if k in r and pd.notna(r[k]) and float(r[k])>0:
                mcaps.append(_human_mc(float(r[k]), valuta))
            else:
                mcaps.append("-")
        hist = pd.DataFrame([data, mcaps], index=["P/S","MCAP"], columns=["Q1","Q2","Q3","Q4"])
        st.dataframe(hist, use_container_width=True)

    # visning av poäng
    with st.expander("Poängförklaring", expanded=False):
        st.write(f"**TotalScore:** {r.get('TotalScore',0):.3f}")
        st.write(f"- ValuationScore: {r.get('ValuationScore',0):.3f}")
        st.write(f"- QualityScore: {r.get('QualityScore',0):.3f}")
        st.write(f"- RiskScore: {r.get('RiskScore',0):.3f}")
        st.write(f"- ModeScore ({mode}): {r.get('ModeScore',0):.3f}")

    # lista / tabell (topp 15) för snabb överblick
    st.markdown("### Toppval just nu")
    show_cols = [
        "Ticker","Bolagsnamn","Sektor","_Risklabel",
        "Aktuell kurs", riktkurs_val, "Upside (%)",
        "Market Cap (nu)","P/S","P/S-snitt",
        "TotalScore"
    ]
    for c in show_cols:
        if c not in base.columns:
            base[c] = ""
    grid = base[show_cols].copy()
    grid["Market Cap (nu)"] = grid["Market Cap (nu)"].apply(lambda x: _human_mc(float(x or 0.0), valuta))
    grid = grid.head(15)
    st.dataframe(grid, use_container_width=True, hide_index=True)
