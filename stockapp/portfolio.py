# stockapp/portfolio.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Hjälp-funktioner
# ------------------------------------------------------------

def _f(x, d=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return d
        return float(x)
    except Exception:
        return d

def _rate(ccy: str, rates: dict) -> float:
    if not ccy:
        return 1.0
    return float(rates.get(str(ccy).upper(), 1.0))

def _fmt_money_sek(v: float) -> str:
    try:
        n = float(v)
    except Exception:
        return "-"
    if abs(n) >= 1e12: return f"{n/1e12:.2f} TSEK"
    if abs(n) >= 1e9:  return f"{n/1e9:.2f} BSEK"
    if abs(n) >= 1e6:  return f"{n/1e6:.2f} MSEK"
    if abs(n) >= 1e3:  return f"{n/1e3:.0f} kSEK"
    return f"{n:.0f} SEK"

def _fmt_pct(x: float) -> str:
    return f"{x:.1f}%"

def _ensure_port_cols(df: pd.DataFrame) -> pd.DataFrame:
    need = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Antal aktier","Årlig utdelning",
        "GAV SEK","Sektor","FCF TTM (valuta)","Kassa (valuta)","Debt/Equity",
        "Bruttomarginal (%)","Nettomarginal (%)","Runway (mån)",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Market Cap (nu)"
    ]
    for c in need:
        if c not in df.columns:
            df[c] = 0.0 if any(k in c.lower() for k in ["kurs","antal","utdelning","gav","fcf","kassa","debt","marginal","runway","riktkurs","market cap"]) else ""
    return df

def _pick_target_col() -> str:
    return st.selectbox(
        "Riktkurs att jämföra mot",
        ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=0
    )

# ------------------------------------------------------------
# Säljvakt – kärnlogik för ett innehav
# ------------------------------------------------------------

def _sell_guard_for_row(
    row: pd.Series,
    *,
    total_value_sek: float,
    sector_weight_map: Dict[str, float],
    target_col: str,
    rates: dict,
    settings: dict
) -> Optional[Dict]:
    """
    Returnerar dict med trim/sälj-förslag för en rad, eller None om inga flaggor.
    """

    tkr   = str(row.get("Ticker","")).upper()
    name  = str(row.get("Bolagsnamn",""))
    ccy   = str(row.get("Valuta","") or "USD").upper()
    px    = _f(row.get("Aktuell kurs"), 0.0)
    n_sh  = _f(row.get("Antal aktier"), 0.0)
    gav   = _f(row.get("GAV SEK"), 0.0)           # per aktie i SEK (användarens snitt)
    div_ps= _f(row.get("Årlig utdelning"), 0.0)   # per aktie i bolagsvaluta
    sector= str(row.get("Sektor") or "-")

    if n_sh <= 0 or px <= 0:
        return None

    r = _rate(ccy, rates)
    val_sek = n_sh * px * r
    weight_pct = (val_sek / total_value_sek * 100.0) if total_value_sek > 0 else 0.0

    # Target & övervärdering
    target = _f(row.get(target_col), 0.0)
    overval = (px > target * (1.0 + settings["overval_margin"])) if (target > 0) else False
    overfrac = (px/target - 1.0) if (target > 0) else 0.0

    # GAV-baserad uppgång/förlust
    px_sek = px * r
    gain_pct = ((px_sek - gav) / gav * 100.0) if gav > 0 else 0.0

    # Dividend/coverage
    dy = (div_ps / px) if px > 0 else 0.0
    shares_total = n_sh
    total_div_valuta = div_ps * shares_total
    fcf = _f(row.get("FCF TTM (valuta)"), 0.0)
    cash = _f(row.get("Kassa (valuta)"), 0.0)
    fcf_cov  = (fcf / total_div_valuta)  if total_div_valuta > 0 else 0.0
    cash_cov = (cash / total_div_valuta) if total_div_valuta > 0 else 0.0

    is_dividend = (div_ps > 0 and dy >= settings["min_div_yield_for_div_case"])

    # Sektorvikt
    sec_w = sector_weight_map.get(sector, 0.0)

    reasons = []
    suggested_trim_shares = 0
    severity = 0  # 0..3

    # 1) Position för stor?
    if weight_pct > settings["max_pos_pct"]:
        reasons.append(f"Vikt {weight_pct:.1f}% över max {settings['max_pos_pct']}%")
        # trimma ned till max-pos
        target_value = settings["max_pos_pct"]/100.0 * total_value_sek
        trim_value = max(0.0, val_sek - target_value)
        trim_shares_weight = math.floor(trim_value / max(px_sek, 1e-9))
        suggested_trim_shares = max(suggested_trim_shares, trim_shares_weight)
        severity = max(severity, 2)

    # 2) Sektor för tung?
    if sec_w > settings["max_sector_pct"]:
        reasons.append(f"Sektorn '{sector}' väger {sec_w:.1f}% (> {settings['max_sector_pct']}%)")
        # mild trim (hälften av överskottet för innehav)
        over_sector_val = (sec_w - settings["max_sector_pct"])/100.0 * total_value_sek
        part = 0.5 * over_sector_val
        trim_shares_sector = math.floor(part / max(px_sek, 1e-9))
        suggested_trim_shares = max(suggested_trim_shares, trim_shares_sector)
        severity = max(severity, 2)

    # 3) Övervärderad mot mål + stor vinst mot GAV?
    if overval and gain_pct >= settings["gain_trigger_pct"]:
        reasons.append(f"Över mål ({(overfrac*100):.0f}%) och upp {gain_pct:.0f}% från GAV")
        # trim-ratio beroende på övervärdering
        trim_ratio = min(0.30, max(0.10, overfrac))  # 10–30%
        trim_overval = math.floor(n_sh * trim_ratio)
        suggested_trim_shares = max(suggested_trim_shares, trim_overval)
        severity = max(severity, 3)

    # 4) Utdelningscase – skona om stark täckning
    if is_dividend and (dy >= settings["keep_if_yield_ge"]) and (fcf_cov >= 1.0 or cash_cov >= 1.0):
        # mildra trim-förslag (halvera) om orsaken inte är position/sector över max
        if suggested_trim_shares > 0 and not (weight_pct > settings["max_pos_pct"] or sec_w > settings["max_sector_pct"]):
            reasons.append("Starkt utdelningsstöd (hög yield & FCF/kassa-täckning) – mildrar trim")
            suggested_trim_shares = max(0, math.floor(suggested_trim_shares * 0.5))

    # Minsta trimvärde
    if suggested_trim_shares > 0:
        trim_value_sek = suggested_trim_shares * px_sek
        if trim_value_sek < settings["min_trim_value_sek"]:
            # Runda upp till nå min-värdet (om rimligt)
            need = math.ceil(settings["min_trim_value_sek"] / max(px_sek, 1e-9))
            suggested_trim_shares = max(suggested_trim_shares, need)

    if suggested_trim_shares <= 0:
        return None

    new_value_sek = max(0.0, val_sek - suggested_trim_shares * px_sek)
    new_weight_pct = (new_value_sek / total_value_sek * 100.0) if total_value_sek > 0 else 0.0

    return {
        "Ticker": tkr,
        "Bolagsnamn": name,
        "Sektor": sector,
        "Kurs": px,
        "Valuta": ccy,
        "Antal": n_sh,
        "GAV SEK": gav,
        "Värde (SEK)": val_sek,
        "Vikt (%)": weight_pct,
        "Riktkurs": target,
        "Övervärderad?": overval,
        "Upp från GAV (%)": gain_pct,
        "Utdelningsyield (%)": dy*100.0 if dy>0 else 0.0,
        "FCF-täckning (x)": fcf_cov,
        "Kassa-täckning (x)": cash_cov,
        "Föreslagen trim (st)": suggested_trim_shares,
        "Trimvärde (SEK)": suggested_trim_shares * px_sek,
        "Ny vikt (%)": new_weight_pct,
        "Skäl": "; ".join(reasons),
        "Allvar": severity
    }

# ------------------------------------------------------------
# Publik vy
# ------------------------------------------------------------

def portfolio_view(df: pd.DataFrame, user_rates: dict):
    st.header("📦 Min portfölj & Säljvakt")

    df = _ensure_port_cols(df.copy())
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Värden i SEK
    port["Kurs SEK"] = port.apply(lambda r: _f(r.get("Aktuell kurs"),0.0) * _rate(r.get("Valuta"), user_rates), axis=1)
    port["Värde (SEK)"] = port["Antal aktier"] * port["Kurs SEK"]
    total_value = float(port["Värde (SEK)"].sum())

    # Utdelning i SEK
    port["Utdelning (SEK)"] = port.apply(
        lambda r: _f(r.get("Årlig utdelning"),0.0) * _f(r.get("Antal aktier"),0.0) * _rate(r.get("Valuta"), user_rates), axis=1
    )
    total_div = float(port["Utdelning (SEK)"].sum())

    # Vikt
    port["Vikt (%)"] = np.where(total_value>0, port["Värde (SEK)"]/total_value*100.0, 0.0)

    # Headline
    tcols = st.columns(3)
    with tcols[0]:
        st.metric("Portföljvärde", _fmt_money_sek(total_value))
    with tcols[1]:
        st.metric("Årlig utdelning", _fmt_money_sek(total_div))
    with tcols[2]:
        st.metric("Månadsutdelning", _fmt_money_sek(total_div/12.0 if total_div>0 else 0.0))

    # Sektorallokering
    sector_map = port.groupby(port["Sektor"].replace("", "Okänd"))["Värde (SEK)"].sum()
    sector_weights = (sector_map / max(total_value,1e-9) * 100.0).sort_values(ascending=False)
    st.subheader("Sektorallokering")
    st.dataframe(
        pd.DataFrame({"Värde (SEK)": sector_map.apply(lambda x: _fmt_money_sek(float(x))),
                      "Vikt (%)": sector_weights.round(1)}),
        use_container_width=True
    )

    # Inställningar Säljvakt
    st.subheader("🛡️ Säljvakt")
    with st.expander("Inställningar", expanded=False):
        target_col = _pick_target_col()
        c1, c2, c3 = st.columns(3)
        with c1:
            overval_margin = st.number_input("Över mål – marginal", min_value=0.0, max_value=1.0, value=0.10, step=0.01, help="Ex: 0.10 = 10% över riktkurs räknas som övervärderad.")
            gain_trigger   = st.number_input("Vinst mot GAV (tröskel %)", min_value=0.0, max_value=500.0, value=40.0, step=5.0)
        with c2:
            max_pos_pct    = st.number_input("Max vikt per innehav (%)", min_value=1.0, max_value=100.0, value=12.0, step=1.0)
            max_sector_pct = st.number_input("Max vikt per sektor (%)",  min_value=5.0, max_value=100.0, value=25.0, step=1.0)
        with c3:
            min_trim_value = st.number_input("Minsta trimvärde (SEK)", min_value=0.0, value=1000.0, step=100.0)
            keep_if_yield  = st.number_input("Skona utdelare om yield ≥", min_value=0.0, max_value=0.20, value=0.05, step=0.005)
        min_div_case = st.number_input("Min. yield för att klassas som utdelningscase", min_value=0.0, max_value=0.20, value=0.02, step=0.005)

    settings = {
        "overval_margin": float(overval_margin),
        "gain_trigger_pct": float(gain_trigger),
        "max_pos_pct": float(max_pos_pct),
        "max_sector_pct": float(max_sector_pct),
        "min_trim_value_sek": float(min_trim_value),
        "keep_if_yield_ge": float(keep_if_yield),
        "min_div_yield_for_div_case": float(min_div_case),
    }

    # Sektorvikter map
    sector_w_map = sector_weights.to_dict()

    # Kör säljvakt per rad
    suggestions: List[Dict] = []
    for _, r in port.iterrows():
        sug = _sell_guard_for_row(
            r,
            total_value_sek=total_value,
            sector_weight_map=sector_w_map,
            target_col=target_col,
            rates=user_rates,
            settings=settings
        )
        if sug:
            suggestions.append(sug)

    # Sortera förslag: allvar → trimvärde → vikt
    if suggestions:
        s_df = pd.DataFrame(suggestions)
        s_df = s_df.sort_values(by=["Allvar","Trimvärde (SEK)","Vikt (%)"], ascending=[False, False, False])
        st.markdown("### Förslag (trim/sälj)")
        show_cols = [
            "Ticker","Bolagsnamn","Sektor","Vikt (%)","Kurs","Valuta","Föreslagen trim (st)",
            "Trimvärde (SEK)","Ny vikt (%)","Övervärderad?","Upp från GAV (%)","Utdelningsyield (%)","FCF-täckning (x)","Kassa-täckning (x)","Skäl"
        ]
        s_view = s_df.copy()
        for col in ["Vikt (%)","Ny vikt (%)","Upp från GAV (%)","Utdelningsyield (%)"]:
            if col in s_view.columns:
                s_view[col] = s_view[col].astype(float).round(1)
        if "Trimvärde (SEK)" in s_view.columns:
            s_view["Trimvärde (SEK)"] = s_view["Trimvärde (SEK)"].apply(lambda x: _fmt_money_sek(_f(x)))
        st.dataframe(s_view[show_cols], use_container_width=True, hide_index=True)
        st.caption("Tips: Bekräfta manuellt i 'Lägg till/uppdatera bolag' om du väljer att genomföra en trim/sälj.")
    else:
        st.success("Inga tydliga trim/sälj-signaler enligt dina inställningar just nu.")

    # Full portföljtabell
    st.divider()
    st.subheader("Innehav (översikt)")
    base_cols = ["Ticker","Bolagsnamn","Sektor","Antal aktier","Aktuell kurs","Valuta","Kurs SEK","GAV SEK","Värde (SEK)","Vikt (%)","Årlig utdelning"]
    base_cols = [c for c in base_cols if c in port.columns]
    view = port[base_cols].copy()
    view["Värde (SEK)"] = view["Värde (SEK)"].apply(lambda x: _fmt_money_sek(_f(x)))
    view["Vikt (%)"] = view["Vikt (%)"].round(1)
    st.dataframe(view.sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)
