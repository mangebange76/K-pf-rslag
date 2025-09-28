# stockapp/views/portfolio.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple
import streamlit as st
import pandas as pd
import numpy as np

from ..calc import safe_float, human_mcap, marketcap_risk_label

# -----------------------------
# Hjälp
# -----------------------------

def _rate(user_rates: Dict[str, float], ccy: str) -> float:
    if not ccy:
        return 1.0
    try:
        return float(user_rates.get(ccy.upper(), 1.0))
    except Exception:
        return 1.0

def _ps_now(row: pd.Series) -> float:
    return safe_float(row.get("P/S", 0.0))

def _ps_avg(row: pd.Series) -> float:
    return safe_float(row.get("P/S-snitt", 0.0))

def _potential_vs_target(row: pd.Series, target_col: str) -> float:
    px = safe_float(row.get("Aktuell kurs", 0.0))
    tgt = safe_float(row.get(target_col, 0.0))
    if px > 0 and tgt > 0:
        return (tgt - px) / px * 100.0
    return 0.0

def _overvaluation_factor(row: pd.Series, target_col: str) -> float:
    """Kombinerad övervärderingsindikator (större = mer övervärderad)."""
    # 1) Kurs över riktkurs
    pot = _potential_vs_target(row, target_col)  # negativ => över riktkurs
    over_tgt = max(0.0, -pot)  # % över riktkurs

    # 2) P/S relativt snitt
    ps = _ps_now(row); psavg = _ps_avg(row)
    over_ps = 0.0
    if psavg > 0 and ps > psavg:
        over_ps = (ps / psavg - 1.0) * 100.0  # % över P/S-snitt

    # 3) Värderingslabel
    label = (row.get("Värdering") or "").lower()
    label_bump = 0.0
    if "dyr" in label or "över" in label or "sell" in label:
        label_bump = 15.0
    elif "fair" in label:
        label_bump = 0.0
    elif "billig" in label or "undervärderad" in label:
        label_bump = -10.0

    # Vägd summa
    return 0.6 * over_tgt + 0.4 * over_ps + label_bump

def _profit_vs_gav(row: pd.Series, rate: float) -> float:
    """Vinst i % relativt GAV i SEK."""
    px = safe_float(row.get("Aktuell kurs", 0.0))
    gav = safe_float(row.get("GAV i SEK", 0.0))
    if gav <= 0:
        return 0.0
    return (px * rate - gav) / gav * 100.0

def _position_value_sek(row: pd.Series, user_rates: Dict[str, float]) -> float:
    px = safe_float(row.get("Aktuell kurs", 0.0))
    qty = safe_float(row.get("Antal aktier", 0.0))
    rate = _rate(user_rates, str(row.get("Valuta","") or "SEK"))
    return qty * px * rate

def _annual_dividend_sek(row: pd.Series, user_rates: Dict[str, float]) -> float:
    div = safe_float(row.get("Årlig utdelning", 0.0))
    qty = safe_float(row.get("Antal aktier", 0.0))
    rate = _rate(user_rates, str(row.get("Valuta","") or "SEK"))
    return qty * div * rate

def _cap_label_from_row(row: pd.Series) -> str:
    # Market cap nu (px*shares) – shares i miljoner
    px = safe_float(row.get("Aktuell kurs", 0.0))
    sh_m = safe_float(row.get("Utestående aktier", 0.0))
    mcap = px * sh_m * 1e6 if (px > 0 and sh_m > 0) else safe_float(row.get("Market Cap (nu)", 0.0))
    return marketcap_risk_label(mcap)

def _sell_score(row: pd.Series, tgt_col: str, port_total_sek: float, user_rates: Dict[str, float]) -> float:
    """Mix av övervärdering, övervikt och vinst mot GAV."""
    # Övervärdering
    ov = _overvaluation_factor(row, tgt_col)

    # Övervikt (vikt i portfölj)
    val = _position_value_sek(row, user_rates)
    weight = (val / port_total_sek * 100.0) if port_total_sek > 0 else 0.0
    # Normalisera övervikt över 1.0x targetweight i UI – tas in separat i trim-förslag
    over_w = weight  # enkel proxy

    # Vinst mot GAV
    pr = _profit_vs_gav(row, _rate(user_rates, str(row.get("Valuta","") or "SEK")))

    # Låga fundament (lågt Growth/DividendScore drar upp säljsignal)
    g = safe_float(row.get("GrowthScore", 0.0)); d = safe_float(row.get("DividendScore", 0.0))
    low_fund = max(0.0, 50.0 - 0.5*(g + d))  # om snittscore <100 => plus

    # Viktning
    return 0.5*ov + 0.3*over_w + 0.2*pr + 0.2*low_fund

def _trim_to_target(weight: float, target_weight: float, port_total_sek: float, px: float, rate: float) -> int:
    """Hur många aktier sälja för att ta ned vikten till target_weight."""
    if port_total_sek <= 0 or px <= 0 or rate <= 0 or weight <= target_weight:
        return 0
    curr_value = weight/100.0 * port_total_sek
    target_value = target_weight/100.0 * port_total_sek
    delta_sek = curr_value - target_value
    # antal ≈ delta_sek / (pris_i_SEK)
    shares = int(np.floor(delta_sek / (px * rate)))
    return max(0, shares)

# -----------------------------
# Portfölj-vy
# -----------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📦 Min portfölj")

    if df is None or df.empty:
        st.info("Ingen data.")
        return

    base = df.copy()
    base = base[base.get("Antal aktier", pd.Series([0]*len(base))) > 0].copy()
    if base.empty:
        st.info("Du äger inga aktier.")
        return

    # Beräkna SEK-värden & vikt
    base["Värde (SEK)"] = base.apply(lambda r: _position_value_sek(r, user_rates), axis=1)
    total_value = float(base["Värde (SEK)"].sum())
    base["Andel (%)"] = np.where(total_value > 0, base["Värde (SEK)"]/total_value*100.0, 0.0).round(2)

    base["Utdelning/år (SEK)"] = base.apply(lambda r: _annual_dividend_sek(r, user_rates), axis=1)
    total_div = float(base["Utdelning/år (SEK)"].sum())
    base["Direktavkastning (%)"] = np.where(base["Värde (SEK)"] > 0, base["Utdelning/år (SEK)"]/base["Värde (SEK)"]*100.0, 0.0).round(2)

    st.markdown(f"**Totalt portföljvärde:** {total_value:,.0f} SEK".replace(",", " "))
    st.markdown(f"**Total kommande utdelning:** {total_div:,.0f} SEK / år".replace(",", " "))
    st.markdown(f"**Ungefärlig månadsutdelning:** {total_div/12.0:,.0f} SEK / månad".replace(",", " "))

    # Visa tabell per innehav
    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Utdelning/år (SEK)","Direktavkastning (%)","GAV i SEK"]
    show_cols = [c for c in show_cols if c in base.columns]
    st.dataframe(base[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    # Sektoröversikt + varning
    st.markdown("---")
    st.subheader("🏷️ Sektoröversikt")
    base["Sektor"] = base.get("Sektor", pd.Series(["Okänd"]*len(base))).fillna("Okänd").astype(str)
    sector = base.groupby("Sektor", as_index=False).agg({"Värde (SEK)":"sum"})
    sector["Andel (%)"] = np.where(total_value>0, sector["Värde (SEK)"]/total_value*100.0, 0.0).round(2)
    sector = sector.sort_values(by="Andel (%)", ascending=False)
    st.dataframe(sector, use_container_width=True, hide_index=True)

    warn_thr = st.slider("Varning för sektorövervikt över (%)", min_value=10, max_value=80, value=35, step=5)
    heavy = sector[sector["Andel (%)"] >= warn_thr]
    if not heavy.empty:
        st.warning("Övervikt i följande sektorer:\n\n" + "\n".join([f"- {r['Sektor']}: {r['Andel (%)']:.1f}%" for _, r in heavy.iterrows()]))

    # --------------------------
    # Säljvakt / Trim-förslag
    # --------------------------
    st.markdown("---")
    st.subheader("🚨 Säljvakt / Trim-förslag")

    colcfg1, colcfg2, colcfg3 = st.columns([1,1,1])
    with colcfg1:
        target_col = st.selectbox("Jämför mot riktkurs", ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    with colcfg2:
        max_names = st.number_input("Visa topp N förslag", min_value=3, max_value=30, value=10, step=1)
    with colcfg3:
        target_weight = st.number_input("Målvikt per innehav (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)

    # Beräkna säljsignal
    work = base.copy()
    work["SellScore"] = work.apply(lambda r: _sell_score(r, target_col, total_value, user_rates), axis=1)
    work["Övervärdering (%)"] = work.apply(lambda r: _overvaluation_factor(r, target_col), axis=1).round(2)
    work["Vinst vs GAV (%)"] = work.apply(lambda r: _profit_vs_gav(r, _rate(user_rates, str(r.get("Valuta","") or "SEK"))), axis=1).round(2)

    # Grov etikett (för UI)
    def _label_row(r: pd.Series) -> str:
        ov = r["Övervärdering (%)"]; pr = r["Vinst vs GAV (%)"]; w = safe_float(r.get("Andel (%)",0.0))
        if ov > 30 and pr > 30 and w > target_weight*1.2:
            return "⚠️ Övervärderad – Överväg SÄLJ"
        if ov > 15 and pr > 15:
            return "🔶 Överväg TRIMMA"
        if ov > 5 and w > target_weight*1.1:
            return "🔸 Lätt trim pga övervikt"
        return "—"

    work["Rekommendation"] = work.apply(_label_row, axis=1)

    # Trim-antal för att nå target_weight
    def _trim_shares(r: pd.Series) -> int:
        w = safe_float(r.get("Andel (%)",0.0))
        px = safe_float(r.get("Aktuell kurs", 0.0))
        rate = _rate(user_rates, str(r.get("Valuta","") or "SEK"))
        return _trim_to_target(w, target_weight, total_value, px, rate)

    work["Föreslagen försäljning (st)"] = work.apply(_trim_shares, axis=1)

    # Sortera på SellScore
    work = work.sort_values(by=["SellScore","Andel (%)","Övervärdering (%)"], ascending=[False, False, False]).reset_index(drop=True)

    # Visa topp N
    top = work.head(int(max_names)).copy()
    show_cols2 = ["Ticker","Bolagsnamn","Andel (%)","Övervärdering (%)","Vinst vs GAV (%)","Direktavkastning (%)","Rekommendation","Föreslagen försäljning (st)"]
    show_cols2 = [c for c in show_cols2 if c in top.columns]

    st.dataframe(top[show_cols2], use_container_width=True, hide_index=True)

    # Detaljerad panel för vald rad i listan
    if not top.empty:
        st.markdown("#### Detaljer för högst rankade")
        r0 = top.iloc[0]
        cc = r0.get("Valuta","") or ""
        rate0 = _rate(user_rates, str(cc))
        px0 = safe_float(r0.get("Aktuell kurs",0.0))
        qty0 = safe_float(r0.get("Antal aktier",0.0))
        val0 = qty0 * px0 * rate0
        pot = _potential_vs_target(r0, target_col)
        psn = _ps_now(r0); psa = _ps_avg(r0)
        st.write(f"- **{r0.get('Bolagsnamn','')} ({r0.get('Ticker','')})**")
        st.write(f"- Kurs: {px0:.2f} {cc}   |   Innehavsvärde: {val0:,.0f} SEK".replace(",", " "))
        st.write(f"- Vikt: {safe_float(r0.get('Andel (%)',0.0)):.2f}%   |   Pot. mot mål: {pot:.2f}%")
        st.write(f"- P/S (nu): {psn:.2f}   |   P/S-snitt (Q1–Q4): {psa:.2f}")
        if "Värdering" in r0:
            st.write(f"- Värdering: {r0.get('Värdering','—')}")
        if "GrowthScore" in r0 or "DividendScore" in r0:
            st.write(f"- GrowthScore / DividendScore: {safe_float(r0.get('GrowthScore',0.0)):.1f} / {safe_float(r0.get('DividendScore',0.0)):.1f}")
        st.write(f"- Föreslagen försäljning: **{int(safe_float(r0.get('Föreslagen försäljning (st)',0)))} st** (för att nå {target_weight:.1f}% vikt)")
