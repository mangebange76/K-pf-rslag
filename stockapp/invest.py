# stockapp/invest.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .rates import hamta_valutakurs
from .config import STANDARD_VALUTAKURSER

# -----------------------------------------------------------
# Hjälpare
# -----------------------------------------------------------

def _safe_num(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _format_mcap(n: float) -> str:
    """
    Formatera market cap med T/B/M/K-suffix (svenskt mellanrum).
    """
    try:
        n = float(n)
    except Exception:
        return "-"
    abn = abs(n)
    if abn >= 1e12:
        return f"{n/1e12:,.2f} T".replace(",", " ").replace(".00", "")
    if abn >= 1e9:
        return f"{n/1e9:,.2f} B".replace(",", " ").replace(".00", "")
    if abn >= 1e6:
        return f"{n/1e6:,.2f} M".replace(",", " ").replace(".00", "")
    if abn >= 1e3:
        return f"{n/1e3:,.2f} K".replace(",", " ").replace(".00", "")
    return f"{n:,.0f}".replace(",", " ")

def _risk_label_from_mcap_sek(mcap_sek: float, usd_to_sek: float) -> str:
    """
    Risklabel utifrån market cap i SEK. Trösklar baserat på USD-trösklar konverterade till SEK.
    USD thresholds: Micro <300M, Small <2B, Mid <10B, Large <200B, annars Mega
    """
    if usd_to_sek <= 0:
        usd_to_sek = STANDARD_VALUTAKURSER.get("USD", 10.0)
    micro = 300e6 * usd_to_sek
    small = 2e9 * usd_to_sek
    mid   = 10e9 * usd_to_sek
    large = 200e9 * usd_to_sek

    n = _safe_num(mcap_sek)
    if n < micro:  return "Micro"
    if n < small:  return "Small"
    if n < mid:    return "Mid"
    if n < large:  return "Large"
    return "Mega"

def _calc_ps_snitt(row: pd.Series) -> float:
    vals = []
    for k in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
        if k in row and _safe_num(row[k]) > 0:
            vals.append(_safe_num(row[k]))
    if vals:
        return round(float(np.mean(vals)), 2)
    return _safe_num(row.get("P/S-snitt", 0.0))

def _compute_market_caps(row: pd.Series, user_rates: Dict[str, float]) -> Tuple[float, float]:
    """
    Returnerar (mcap_local, mcap_sek) baserat på Aktuell kurs * Utestående aktier.
    Utestående aktier förvaras i miljoner i databasen.
    """
    price = _safe_num(row.get("Aktuell kurs"))
    shares_m = _safe_num(row.get("Utestående aktier"))  # miljoner
    local = price * shares_m * 1_000_000.0 if (price > 0 and shares_m > 0) else 0.0
    ccy = str(row.get("Valuta", "USD")).upper()
    fx = hamta_valutakurs(ccy, user_rates)
    sek = local * float(fx)
    return local, sek

def _calc_potential_pct(row: pd.Series, target_col: str) -> float:
    px = _safe_num(row.get("Aktuell kurs"))
    tgt = _safe_num(row.get(target_col))
    if px > 0 and tgt > 0:
        return (tgt - px) / px * 100.0
    return 0.0

def _completeness_bonus(metrics_used: List[str], row: pd.Series]) -> float:
    """
    Bonus som gynnar bolag med fler datapunkter.
    Skala 0.7–1.0: 0.7 + 0.3 * (antal_fyllda / antal_tot).
    """
    if not metrics_used:
        return 1.0
    tot = len(metrics_used)
    got = 0
    for m in metrics_used:
        v = row.get(m, None)
        try:
            ok = (v is not None) and (str(v).strip() != "") and (not (isinstance(v, (int, float)) and float(v) == 0.0))
        except Exception:
            ok = False
        if ok:
            got += 1
    return 0.7 + 0.3 * (got / tot)


# -----------------------------------------------------------
# Scoring
# -----------------------------------------------------------

def _score_tillvaxt(row: pd.Series, target_col: str) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Returnerar (score, breakdown, metrics_used) för tillväxt-inriktning.
    """
    breakdown = {}
    used = []

    # 1) Potential mot vald riktkurs (viktigast)
    pot = _calc_potential_pct(row, target_col)  # %
    # Skala: 0 vid <=0% uppsida, 1.0 vid >=50% uppsida (clamp)
    pot_score = max(0.0, min(1.0, pot / 50.0))
    breakdown["Potential"] = pot_score; used.append(target_col)

    # 2) CAGR 5 år (%): högre bättre, 0–40% → 0..1
    cagr = _safe_num(row.get("CAGR 5 år (%)"))
    cagr_score = max(0.0, min(1.0, cagr / 40.0))
    breakdown["CAGR"] = cagr_score; used.append("CAGR 5 år (%)")

    # 3) P/S relativt P/S-snitt (lägre är bättre om under snittet)
    ps_now = _safe_num(row.get("P/S"))
    ps_avg = _calc_ps_snitt(row)
    ps_rel_score = 0.5  # neutral
    if ps_now > 0 and ps_avg > 0:
        rel = ps_now / ps_avg
        if rel <= 1.0:
            # 1.0 → 0.8 score, 0.5 → 1.0 score (clamp)
            ps_rel_score = min(1.0, 0.8 + (1.0 - rel) * 0.4 / 0.5)
        else:
            # 1.0–2.0 → 0.8→0.0
            over = min(rel, 2.0)
            ps_rel_score = max(0.0, 0.8 - (over - 1.0) * 0.8)
    breakdown["P/S vs snitt"] = ps_rel_score; used.extend(["P/S", "P/S-snitt"])

    # Vikter
    w_pot = 0.55
    w_cagr = 0.25
    w_psr = 0.20
    base = pot_score * w_pot + cagr_score * w_cagr + ps_rel_score * w_psr

    return base, breakdown, used

def _score_utdelning(row: pd.Series) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Returnerar (score, breakdown, metrics_used) för utdelnings-inriktning.
    """
    breakdown = {}
    used = []

    price = _safe_num(row.get("Aktuell kurs"))
    div = _safe_num(row.get("Årlig utdelning"))
    yld = (div / price) * 100.0 if (price > 0 and div > 0) else 0.0
    # 0–8% → 0..1 (clamp 12% till 1.0 också)
    yld_score = min(1.0, yld / 8.0)
    breakdown["Direktavkastning"] = yld_score; used.extend(["Årlig utdelning", "Aktuell kurs"])

    # Uthållighet (om finns): FCF Yield %, Payout FCF %, Net Debt/EBITDA
    fcf_yld = _safe_num(row.get("FCF Yield (%)"))
    fcf_score = min(1.0, fcf_yld / 8.0) if fcf_yld > 0 else 0.0
    if "FCF Yield (%)" in row: used.append("FCF Yield (%)")
    breakdown["FCF-yield"] = fcf_score

    payout_fcf = _safe_num(row.get("Dividend Payout FCF (%)"))
    payout_score = 0.0
    if payout_fcf > 0:
        # <=60% → bra; 60–100% → sjunkande; >100% → 0
        if payout_fcf <= 60:
            payout_score = 1.0
        elif payout_fcf <= 100:
            payout_score = max(0.0, 1.0 - (payout_fcf - 60) / 40.0)
        else:
            payout_score = 0.0
        used.append("Dividend Payout FCF (%)")
    breakdown["Payout(FCF)"] = payout_score

    nde = _safe_num(row.get("Net Debt/EBITDA"))
    nde_score = 0.0
    if nde > 0:
        # <=1.0 → 1.0; 1–3 → 1→0.3; >3 → 0
        if nde <= 1.0:
            nde_score = 1.0
        elif nde <= 3.0:
            nde_score = max(0.3, 1.0 - (nde - 1.0) * 0.35)
        else:
            nde_score = 0.0
        used.append("Net Debt/EBITDA")
    breakdown["Skuld"] = nde_score

    # Vikter
    w_y = 0.55
    w_f = 0.25
    w_p = 0.10
    w_d = 0.10
    base = yld_score * w_y + fcf_score * w_f + payout_score * w_p + nde_score * w_d
    return base, breakdown, used


# -----------------------------------------------------------
# Huvudvy
# -----------------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Se till att grundfält finns
    need = ["Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Utestående aktier",
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
            "P/S-snitt",
            "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
            "Årlig utdelning"]
    for c in need:
        if c not in df.columns:
            df[c] = 0.0 if c not in ["Ticker","Bolagsnamn","Valuta"] else ""

    work = df.copy()

    # Beräkna P/S-snitt om saknas
    if "P/S-snitt" not in work.columns:
        work["P/S-snitt"] = 0.0
    work["P/S-snitt"] = work.apply(_calc_ps_snitt, axis=1)

    # Market Cap & risklabel
    usdsek = float(user_rates.get("USD", STANDARD_VALUTAKURSER["USD"]))
    mcap_local_list = []
    mcap_sek_list = []
    risklabel_list = []
    for _, r in work.iterrows():
        local, sek = _compute_market_caps(r, user_rates)
        mcap_local_list.append(local)
        mcap_sek_list.append(sek)
        risklabel_list.append(_risk_label_from_mcap_sek(sek, usdsek))
    work["Market Cap"] = mcap_local_list
    work["Market Cap (SEK)"] = mcap_sek_list
    work["_RiskLabel"] = risklabel_list

    # UI: val
    target_col = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )
    focus = st.radio("Inriktning", ["Båda", "Tillväxt", "Utdelning"], horizontal=True, index=0)

    # Sektorfiltret (om Sector finns)
    sector_col = None
    for cand in ["Sektor", "Sector", "Bransch", "Industry"]:
        if cand in work.columns:
            sector_col = cand
            break
    sectors = ["Alla"]
    if sector_col:
        sectors += sorted(list({str(x) for x in work[sector_col].fillna("").astype(str) if str(x).strip()}))
    chosen_sector = st.selectbox("Sektorfilter", sectors, index=0)

    # Risklabel-filter
    risk_opts = ["Alla", "Mega", "Large", "Mid", "Small", "Micro"]
    chosen_risk = st.selectbox("Risklabel", risk_opts, index=0)

    # Filtrera
    base = work.copy()
    if sector_col and chosen_sector != "Alla":
        base = base[base[sector_col].astype(str) == chosen_sector]
    if chosen_risk != "Alla":
        base = base[base["_RiskLabel"] == chosen_risk]

    # Poängberäkning
    scores = []
    brks = []
    compl = []
    for _, r in base.iterrows():
        if focus == "Tillväxt":
            s, b, used = _score_tillvaxt(r, target_col)
        elif focus == "Utdelning":
            s, b, used = _score_utdelning(r)
        else:
            s1, b1, u1 = _score_tillvaxt(r, target_col)
            s2, b2, u2 = _score_utdelning(r)
            s = 0.5 * s1 + 0.5 * s2
            b = {**{f"TG {k}": v for k, v in b1.items()},
                 **{f"UD {k}": v for k, v in b2.items()}}
            used = list(set(u1 + u2 + [target_col]))
        bonus = _completeness_bonus(used, r)
        scores.append(s * bonus)
        brks.append(b)
        compl.append(bonus)

    base = base.copy()
    base["_Score"] = scores
    base["_Score completeness"] = compl
    base["_Potential (%)"] = base.apply(lambda row: _calc_potential_pct(row, target_col), axis=1)

    # Sortering: score desc, potential desc, mcap_sek desc
    base = base.sort_values(by=["_Score", "_Potential (%)", "Market Cap (SEK)"], ascending=[False, False, False])

    st.caption("Bolag rankas efter **TotalScore** (justerat för datapunkternas täckning). Högst = bäst.")

    # Visa topplista
    top = base.head(30).reset_index(drop=True)

    # Säker kolumnlista (visa endast de som finns)
    desired_show = [
        "Ticker", "Bolagsnamn", "Valuta",
        "Aktuell kurs", "Utestående aktier",
        target_col, "_Potential (%)",
        "P/S", "P/S-snitt",
        "Market Cap", "Market Cap (SEK)", "_RiskLabel"
    ]
    show = [c for c in desired_show if c in top.columns]

    # Formatera
    if "Market Cap" in top.columns:
        top["Market Cap (txt)"] = top["Market Cap"].apply(_format_mcap)
    if "Market Cap (SEK)" in top.columns:
        top["Market Cap (SEK) (txt)"] = top["Market Cap (SEK)"].apply(_format_mcap)
    if "_Potential (%)" in top.columns:
        top["_Potential (%)"] = top["_Potential (%)"].round(2)
    if "P/S" in top.columns:
        top["P/S"] = top["P/S"].replace([np.inf, -np.inf], np.nan).fillna(0.0).round(2)
    if "P/S-snitt" in top.columns:
        top["P/S-snitt"] = top["P/S-snitt"].replace([np.inf, -np.inf], np.nan).fillna(0.0).round(2)
    if "Aktuell kurs" in top.columns:
        top["Aktuell kurs"] = top["Aktuell kurs"].round(2)
    if target_col in top.columns:
        top[target_col] = top[target_col].round(2)
    if "Utestående aktier" in top.columns:
        top["Utestående aktier"] = top["Utestående aktier"].round(2)

    # Lägg till textkolumner om de finns
    desired_plus = []
    if "Market Cap (txt)" in top.columns:
        desired_plus.append("Market Cap (txt)")
    if "Market Cap (SEK) (txt)" in top.columns:
        desired_plus.append("Market Cap (SEK) (txt)")

    show_final = [c for c in (show + desired_plus + ["_Score"]) if c in top.columns]

    st.dataframe(
        top[show_final],
        use_container_width=True,
        hide_index=True
    )

    # Expander: visa scoring för topp 5
    st.markdown("### Detalj för topp 5")
    for i, (_, rr) in enumerate(top.head(5).iterrows(), start=1):
        with st.expander(f"{i}. {rr.get('Bolagsnamn','')} ({rr.get('Ticker','')}) – Score: {rr.get('_Score',0):.3f}"):
            cols = st.columns(3)
            with cols[0]:
                st.write("**Översikt**")
                st.write(f"- Valuta: {rr.get('Valuta','')}")
                st.write(f"- Aktuell kurs: {rr.get('Aktuell kurs',0):.2f}")
                st.write(f"- Utestående aktier (milj): {_safe_num(rr.get('Utestående aktier')):.2f}")
                st.write(f"- P/S nu: {_safe_num(rr.get('P/S')):.2f}")
                st.write(f"- P/S-snitt (4Q): {_safe_num(rr.get('P/S-snitt')):.2f}")
            with cols[1]:
                st.write("**Kapital**")
                st.write(f"- Market Cap: {_format_mcap(rr.get('Market Cap',0))}")
                st.write(f"- Market Cap (SEK): {_format_mcap(rr.get('Market Cap (SEK)',0))}")
                st.write(f"- Risklabel: {rr.get('_RiskLabel','')}")
            with cols[2]:
                st.write("**Potential**")
                st.write(f"- {target_col}: {_safe_num(rr.get(target_col)):.2f}")
                st.write(f"- Uppsida: {_safe_num(rr.get('_Potential (%)')):.2f}%")

            # Visa ev. utdelningsdata
            if _safe_num(rr.get("Årlig utdelning")) > 0 and _safe_num(rr.get("Aktuell kurs")) > 0:
                y = _safe_num(rr.get("Årlig utdelning")) / max(1e-9, _safe_num(rr.get("Aktuell kurs"))) * 100.0
                st.write(f"**Direktavkastning:** {y:.2f}%")

            # Visa nyckel-P/S kvartal om finns
            psq = []
            for k in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
                if k in rr and _safe_num(rr.get(k)) > 0:
                    psq.append(f"{k}: {_safe_num(rr.get(k)):.2f}")
            if psq:
                st.write("**P/S (4 senaste kvartal):** " + " | ".join(psq))
