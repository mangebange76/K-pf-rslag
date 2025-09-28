# -*- coding: utf-8 -*-
"""
stockapp/views/ideas.py

Investeringsförslag:
- Filtrera på Growth / Dividend-läge
- Filtrera på sektor och riskklass (Micro/Small/Mid/Large/Mega)
- Ranka bolag med en enkel poängmodell (olika vikter för Growth vs Dividend)
- Visa uppsida till vald riktkurs, aktuellt market cap, P/S nu & snitt, samt nyckeltal
- Räkna antal att köpa givet kapital i SEK och växelkurs-tabell
- Expander för historik (P/S Q1–Q4, Mcap Q1–Q4 om finns) och kassaflöde/runway om data finns
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import streamlit as st
import pandas as pd
import numpy as np
import math

# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------

def _num(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _safe_div(a, b, default=0.0) -> float:
    a = _num(a, 0.0); b = _num(b, 0.0)
    if b == 0:
        return float(default)
    return float(a) / float(b)

def _fmt_money_short(v: float) -> str:
    try:
        x = float(v)
    except Exception:
        return "-"
    ax = abs(x)
    if ax >= 1_000_000_000_000:
        return f"{x/1_000_000_000_000:.2f} T"
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f} B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    return f"{x:.2f}"

def _risk_label_from_mcap(mcap: float) -> str:
    """
    Grov storleksklass baserat på market cap (lokal valuta).
    Trösklar: Micro < 300 M, Small < 2 B, Mid < 10 B, Large < 200 B, annars Mega.
    """
    v = _num(mcap, 0.0)
    if v < 300_000_000:
        return "Micro"
    if v < 2_000_000_000:
        return "Small"
    if v < 10_000_000_000:
        return "Mid"
    if v < 200_000_000_000:
        return "Large"
    return "Mega"

def _sector_choices(df: pd.DataFrame) -> List[str]:
    vals = sorted(set([str(x) for x in df.get("Sektor", pd.Series([],dtype=object)).fillna("").tolist() if str(x).strip()]))
    return vals

def _calc_market_cap_now(row: pd.Series) -> float:
    price = _num(row.get("Aktuell kurs", 0.0))
    sh_m = _num(row.get("Utestående aktier", 0.0))  # miljoner
    if price > 0 and sh_m > 0:
        return price * sh_m * 1_000_000.0
    return 0.0

def _ps_avg(row: pd.Series) -> float:
    # Använd "P/S-snitt" om finns, annars medel av Q1–Q4 (positiva)
    val = row.get("P/S-snitt", None)
    if val is not None and _num(val) > 0:
        return _num(val)
    qs = [_num(row.get("P/S Q1",0.0)), _num(row.get("P/S Q2",0.0)), _num(row.get("P/S Q3",0.0)), _num(row.get("P/S Q4",0.0))]
    qs = [x for x in qs if x > 0]
    return float(np.mean(qs)) if qs else 0.0

def _upside_pct(price: float, target: float) -> float:
    p = _num(price, 0.0); t = _num(target, 0.0)
    if p <= 0 or t <= 0:
        return 0.0
    return (t - p) / p * 100.0

def _shares_affordable(capital_sek: float, price_local: float, ccy: str, rates: Dict[str, float]) -> Tuple[int, float]:
    """
    Returnerar (antal_aktier, investerat_SEK).
    """
    r = float(rates.get(str(ccy).upper(), 1.0))
    px_sek = _num(price_local) * r
    if px_sek <= 0:
        return 0, 0.0
    n = int(capital_sek // px_sek)
    return n, n * px_sek

def _bound(x, lo, hi):
    return max(lo, min(hi, x))

# ------------------------------------------------------------
# Poängmodeller
# ------------------------------------------------------------

def _score_growth(row: pd.Series, target_col: str) -> float:
    """
    Growth-score (0–100) med enkla heuristikvikter:
      - Value/Upside (40%)
      - Kvalitet: CAGR 5y (15%), bruttomarginal (10%), nettomarginal (10%)
      - Rimlig värdering: P/S relativt snitt (15%)
      - Finansiell risk: Debt/Equity (5%) + Cash runway (5%, om negativ FCF & kassa finns)
    """
    price = _num(row.get("Aktuell kurs",0.0))
    target = _num(row.get(target_col,0.0))
    upside = _bound(_upside_pct(price, target)/100.0, -1.0, 3.0)  # -100%..+300% → -1..+3
    part_value = (upside + 1.0) / 4.0 * 100.0  # normalisera ~0..100

    cagr = _num(row.get("CAGR 5 år (%)",0.0))
    part_cagr = _bound((cagr/40.0)*100.0, 0.0, 100.0)  # 40%+ ≈ 100

    gm = _num(row.get("Bruttomarginal",0.0))  # antas %
    part_gm = _bound((gm/60.0)*100.0, 0.0, 100.0)      # 60% ≈ 100

    nm = _num(row.get("Nettomarginal",0.0))
    part_nm = _bound(((nm+20.0)/30.0)*100.0, 0.0, 100.0)  # -20..+10 → 0..100 ungefär

    ps_now = _num(row.get("P/S",0.0))
    ps_avg = _ps_avg(row)
    rel = _safe_div(ps_avg, ps_now, default=0.0) if ps_now>0 else 0.0  # >1 är bättre (nuvarande <= snitt)
    part_ps = _bound(rel*100.0, 0.0, 130.0)
    part_ps = min(part_ps, 100.0)

    de = _num(row.get("Debt/Equity",0.0))
    # 0–0.5 bäst ~100 → sveper ned mot 0 poäng vid D/E > 3
    part_de = _bound((0.5/max(0.5, de))*100.0 if de>0 else 100.0, 0.0, 100.0)

    # runway: om FCF < 0 och Kassa > 0: runway kvartal = Kassa / (|FCF|/4)
    fcf = _num(row.get("FCF",0.0))
    cash = _num(row.get("Kassa",0.0))
    runway_q = (cash / (_num(abs(fcf))/4.0)) if (fcf < 0 and cash > 0) else None
    if runway_q is None:
        part_runway = 60.0  # neutral
    else:
        # 0q → 0p, 4q → 60p, 8q+ → 100p
        part_runway = _bound((runway_q/8.0)*100.0, 0.0, 100.0)

    score = (
        0.40*part_value +
        0.15*part_cagr +
        0.10*part_gm +
        0.10*part_nm +
        0.15*part_ps +
        0.05*part_de +
        0.05*part_runway
    )
    return float(_bound(score, 0.0, 100.0))

def _score_dividend(row: pd.Series, target_col: str) -> float:
    """
    Dividend-score (0–100) med vikter:
      - Direktyield (35%) (diminishing över ~8–10%)
      - Payout (FCF-baserad om möjligt, annars EPS/udl per aktie) (20%) sweetspot ~30–60%
      - Stabilitet/profitabilitet: nettomarginal (10%) + bruttomarginal (5%) + D/E (10%)
      - Värdering via P/S relativt snitt (10%)
      - Uppsida till riktkurs (10%)
    """
    # Direktyield
    yld = _num(row.get("Utdelningsyield",0.0))  # %
    # 0–10% → 0..100, därefter avtagande (cap ~120 men klipp till 100)
    y_norm = _bound((yld/10.0)*100.0, 0.0, 100.0)

    # Payout (FCF)
    # approx: tot utd / FCF. tot utd ≈ årlig utdelning per aktie * antal aktier
    udl_pa = _num(row.get("Årlig utdelning",0.0))
    shares = _num(row.get("Utestående aktier",0.0)) * 1_000_000.0
    fcf = _num(row.get("FCF",0.0))
    total_div_cash = udl_pa * shares if (udl_pa>0 and shares>0) else None
    payout = None
    if total_div_cash is not None and fcf > 0:
        payout = (total_div_cash / fcf) * 100.0  # %
    # Scora payout: 30–60% ≈ 100p, 0% → 30p, >100% → ner snabbt
    if payout is None:
        part_payout = 60.0
    else:
        if payout <= 0:
            part_payout = 30.0
        elif payout <= 30:
            part_payout = 70.0 * (payout/30.0) + 30.0  # 0→30%: 30→100
        elif payout <= 60:
            part_payout = 100.0
        elif payout <= 90:
            part_payout = 100.0 - (payout-60.0) * 1.5  # 60→90: ned 45p
        else:
            part_payout = max(10.0, 55.0 - (payout-90.0)*1.5)

    # Profitabilitet & balans
    nm = _num(row.get("Nettomarginal",0.0))
    part_nm = _bound(((nm+15.0)/25.0)*100.0, 0.0, 100.0)  # -15..+10 ~ 0..100
    gm = _num(row.get("Bruttomarginal",0.0))
    part_gm = _bound((gm/60.0)*100.0, 0.0, 100.0)
    de = _num(row.get("Debt/Equity",0.0))
    part_de = _bound((0.6/max(0.6,de))*100.0 if de>0 else 100.0, 0.0, 100.0)

    # Värdering via P/S relativt snitt
    ps_now = _num(row.get("P/S",0.0))
    ps_avg = _ps_avg(row)
    rel = _safe_div(ps_avg, ps_now, default=0.0) if ps_now>0 else 0.0
    part_ps = min(_bound(rel*100.0, 0.0, 130.0), 100.0)

    # Uppsida
    price = _num(row.get("Aktuell kurs",0.0))
    target = _num(row.get(target_col,0.0))
    part_up = _bound((_upside_pct(price, target)/100.0)*100.0 + 50.0, 0.0, 100.0)  # center ~50

    score = (
        0.35*y_norm +
        0.20*part_payout +
        0.10*part_nm +
        0.05*part_gm +
        0.10*part_de +
        0.10*part_ps +
        0.10*part_up
    )
    return float(_bound(score, 0.0, 100.0))

def _label_from_score(score: float) -> str:
    """
    Översätt totalpoäng → etikett.
    """
    s = _num(score, 0.0)
    if s >= 85: return "Mycket bra"
    if s >= 70: return "Bra"
    if s >= 55: return "OK / Fair"
    if s >= 40: return "Något övervärderad"
    return "Övervärderad / Försiktig"

# ------------------------------------------------------------
# Huvudvy
# ------------------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    if df.empty or "Aktuell kurs" not in df.columns:
        st.info("Inga bolag i databasen ännu.")
        return

    # Val: läge, riktkurs, filter
    mode = st.radio("Läge", ["Growth", "Dividend"], horizontal=True, index=0)
    target_col = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs om 1 år", "Riktkurs idag", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=0
    )

    # Filtrering
    cols = st.columns(3)
    with cols[0]:
        sektorer = _sector_choices(df)
        sel_sectors = st.multiselect("Filtrera på sektor", options=sektorer, default=[])
    with cols[1]:
        risk_opt = ["Alla","Micro","Small","Mid","Large","Mega"]
        sel_risk = st.selectbox("Riskklass (market cap)", risk_opt, index=0)
    with cols[2]:
        max_n = st.slider("Max antal förslag", min_value=5, max_value=50, value=20, step=5)

    # Kapital och valuta
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", min_value=0.0, value=500.0, step=100.0)

    # Basurval: måste ha pris och mål
    base = df.copy()
    base["Aktuell kurs"] = pd.to_numeric(base.get("Aktuell kurs",0.0), errors="coerce").fillna(0.0)
    base[target_col] = pd.to_numeric(base.get(target_col,0.0), errors="coerce").fillna(0.0)
    base = base[(base["Aktuell kurs"] > 0) & (base[target_col] > 0)]

    if sel_sectors:
        base = base[base["Sektor"].astype(str).isin(sel_sectors)]

    # Market cap now (lokal valuta)
    base["_mcap_now"] = base.apply(_calc_market_cap_now, axis=1)
    base["_risk"] = base["_mcap_now"].apply(_risk_label_from_mcap)
    if sel_risk != "Alla":
        base = base[base["_risk"] == sel_risk]

    if base.empty:
        st.info("Inga bolag matchar dina filter.")
        return

    # Score per rad
    scores = []
    for _, r in base.iterrows():
        if mode == "Growth":
            sc = _score_growth(r, target_col)
        else:
            sc = _score_dividend(r, target_col)
        scores.append(sc)
    base["_score"] = scores
    base["_label"] = base["_score"].apply(_label_from_score)

    # Sortera på score
    base = base.sort_values(by=["_score","Bolagsnamn","Ticker"], ascending=[False, True, True]).head(int(max_n)).reset_index(drop=True)

    # Visa lista
    for i, row in base.iterrows():
        name = f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})"
        st.subheader(f"{i+1}. {name}")

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.metric("Score", f"{row['_score']:.1f}", help="0–100; högre är bättre enligt valt läge.")
            st.caption(row["_label"])
        with c2:
            mcap_now = row["_mcap_now"]
            st.metric("Market cap (nu)", f"{_fmt_money_short(mcap_now)} {row.get('Valuta','')}")
            st.caption(f"Risk: {row['_risk']}")
        with c3:
            ps_now = _num(row.get("P/S",0.0))
            ps_avg = _ps_avg(row)
            st.metric("P/S (nu / snitt)", f"{ps_now:.2f} / {ps_avg:.2f}")

        # Uppsida & köp
        price = _num(row.get("Aktuell kurs",0.0))
        target = _num(row.get(target_col,0.0))
        upside = _upside_pct(price, target)
        n_shares, invest_sek = _shares_affordable(kapital_sek, price, row.get("Valuta","SEK"), user_rates)

        st.markdown(
            f"- **Aktuell kurs:** {price:.2f} {row.get('Valuta','')}\n"
            f"- **{target_col}:** {target:.2f} {row.get('Valuta','')}\n"
            f"- **Uppsida:** {upside:.1f} %\n"
            f"- **Antal att köpa för {int(kapital_sek)} SEK:** {n_shares} st "
            f"(~{invest_sek:.0f} SEK investerat)"
        )

        # Nyckeltal-kort
        key_cols = {
            "CAGR 5 år (%)": "CAGR 5y",
            "Utdelningsyield": "Yield",
            "Debt/Equity": "D/E",
            "Bruttomarginal": "Brutto%",
            "Nettomarginal": "Netto%",
            "EV/EBITDA": "EV/EBITDA",
        }
        line = []
        for col, label in key_cols.items():
            if col in row.index:
                val = row.get(col, "")
                try:
                    v = float(val)
                    line.append(f"- **{label}:** {v:.2f}")
                except Exception:
                    if str(val).strip():
                        line.append(f"- **{label}:** {val}")
        if line:
            st.markdown("\n".join(line))

        # Expander: Historik
        with st.expander("Historik (P/S & Mcap)", expanded=False):
            ps_hist = []
            for q in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                if q in row.index and _num(row.get(q,0.0))>0:
                    ps_hist.append(f"{q}: {float(row[q]):.2f}")
            st.write("**P/S (senaste 4 TTM-fönster):** " + (", ".join(ps_hist) if ps_hist else "–"))

            mcap_hist_vals = []
            for q in ["Mcap Q1","Mcap Q2","Mcap Q3","Mcap Q4"]:
                if q in row.index and _num(row.get(q,0.0))>0:
                    mcap_hist_vals.append(f"{q}: {_fmt_money_short(_num(row[q]))} {row.get('Valuta','')}")
            st.write("**Market cap (historia):** " + (", ".join(mcap_hist_vals) if mcap_hist_vals else "–"))

        # Expander: Kassaflöde & runway
        with st.expander("Kassaflöde & runway", expanded=False):
            cash = _num(row.get("Kassa",0.0))
            fcf = _num(row.get("FCF",0.0))
            burn_per_q = abs(fcf)/4.0 if fcf<0 else 0.0
            runway_q = (cash / burn_per_q) if (cash>0 and burn_per_q>0) else None
            st.write(f"- **Kassa:** {_fmt_money_short(cash)} {row.get('Valuta','')}")
            st.write(f"- **FCF (årligen):** {_fmt_money_short(fcf)} {row.get('Valuta','')}")
            if runway_q is None:
                st.write("- **Runway:** –")
            else:
                st.write(f"- **Runway (kvartal):** {runway_q:.1f}")

        st.divider()

    # Liten påminnelse
    st.caption("Tips: Filtrera på sektor och riskklass för att smalna av listan. Både poäng och uppsida vägs in beroende på valt läge.")
