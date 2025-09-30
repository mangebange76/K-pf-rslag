# stockapp/invest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .rates import hamta_valutakurs
from .utils import safe_float

# ------------------------------------------------------------
# Hjälpare
# ------------------------------------------------------------
def risk_label_from_mcap(mcap: float) -> str:
    if mcap is None or np.isnan(mcap) or mcap <= 0:
        return "Okänd"
    # USD kap-segment (kan justeras vid behov)
    # Micro <0.3B, Small <2B, Mid <10B, Large <200B, Mega >=200B
    if mcap < 3e8:
        return "Micro"
    if mcap < 2e9:
        return "Small"
    if mcap < 1e10:
        return "Mid"
    if mcap < 2e11:
        return "Large"
    return "Mega"

def _get_shares_outstanding(row: pd.Series) -> Optional[float]:
    for col in ["Utestående aktier", "Shares Outstanding", "Antal utestående aktier"]:
        if col in row.index:
            v = safe_float(row[col], None)
            if v and v > 0:
                return v
    return None

def _ps_avg_from_row(row: pd.Series) -> Optional[float]:
    # Primärt snitt av Q1..Q4 om de finns, annars fallback på "P/S snitt (4)", därefter "P/S"
    qcols = [c for c in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"] if c in row.index]
    vals = []
    for c in qcols:
        v = safe_float(row[c], None)
        if v is not None and v > 0 and v < 1e5:
            vals.append(v)
    if len(vals) >= 2:
        return float(np.mean(vals))
    if "P/S snitt (4)" in row.index:
        v = safe_float(row["P/S snitt (4)"], None)
        if v and v > 0 and v < 1e5:
            return float(v)
    if "P/S" in row.index:
        v = safe_float(row["P/S"], None)
        if v and v > 0 and v < 1e5:
            return float(v)
    return None

def _currency(row: pd.Series) -> str:
    v = str(row.get("Valuta", "USD") or "USD").upper().strip()
    return v if v else "USD"

def _sector(row: pd.Series) -> str:
    return str(row.get("Sector", "Unknown") or "Unknown")

def _industry(row: pd.Series) -> str:
    return str(row.get("Industry", "") or "")

def _fmt_money(v: Optional[float]) -> str:
    if v is None or np.isnan(v):
        return "-"
    # skriv i tusentals/miljoner/miljarder/triljoner
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v/1e12:.2f} T"
    if abs_v >= 1e9:
        return f"{v/1e9:.2f} B"
    if abs_v >= 1e6:
        return f"{v/1e6:.2f} M"
    if abs_v >= 1e3:
        return f"{v/1e3:.2f} k"
    return f"{v:.0f}"

def _fmt_pct(v: Optional[float]) -> str:
    if v is None or np.isnan(v):
        return "-"
    return f"{v:.2f}%"

def _present_cols(row: pd.Series, cols: List[str]) -> int:
    n = 0
    for c in cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
            n += 1
    return n

# ------------------------------------------------------------
# Sektorsvikter (förenklat – fallback om du inte har i config)
# ------------------------------------------------------------
# Poängsättning bygger på 0..1-normalisering lokalt per kolumn (robust z/percentil)
# och sektorspecifika vikter. Saknas en kolumn → vikt ignoreras och omfördelas.
_DEFAULT_SECTOR_WEIGHTS: Dict[str, Dict[str, float]] = {
    # generella nycklar: 'ps', 'gm', 'nm', 'ev_ebitda', 'de', 'fcy', 'runway', 'dy', 'payout_cfo'
    "Technology": {"ps": 0.35, "gm": 0.15, "nm": 0.10, "ev_ebitda": 0.15, "de": 0.05, "fcy": 0.10, "runway": 0.10},
    "Communication Services": {"ps": 0.30, "gm": 0.10, "nm": 0.10, "ev_ebitda": 0.20, "de": 0.05, "fcy": 0.10, "runway": 0.15},
    "Consumer Discretionary": {"ps": 0.30, "gm": 0.10, "nm": 0.10, "ev_ebitda": 0.20, "de": 0.05, "fcy": 0.10, "runway": 0.15},
    "Health Care": {"ps": 0.30, "gm": 0.10, "nm": 0.10, "ev_ebitda": 0.20, "de": 0.05, "fcy": 0.10, "runway": 0.15},
    "Financials": {"ps": 0.15, "gm": 0.05, "nm": 0.20, "ev_ebitda": 0.25, "de": 0.10, "fcy": 0.10, "runway": 0.15},
    "Industrials": {"ps": 0.25, "gm": 0.10, "nm": 0.10, "ev_ebitda": 0.25, "de": 0.10, "fcy": 0.10, "runway": 0.10},
    "Materials": {"ps": 0.20, "gm": 0.10, "nm": 0.10, "ev_ebitda": 0.25, "de": 0.10, "fcy": 0.10, "runway": 0.15},
    "Energy": {"ps": 0.15, "gm": 0.05, "nm": 0.10, "ev_ebitda": 0.30, "de": 0.10, "fcy": 0.15, "runway": 0.15, "dy": 0.05, "payout_cfo": 0.05},
    "Utilities": {"ps": 0.10, "gm": 0.05, "nm": 0.15, "ev_ebitda": 0.25, "de": 0.20, "fcy": 0.10, "runway": 0.10, "dy": 0.05, "payout_cfo": 0.05},
    "Real Estate": {"ps": 0.10, "gm": 0.05, "nm": 0.10, "ev_ebitda": 0.35, "de": 0.15, "fcy": 0.15, "runway": 0.10},
    "Consumer Staples": {"ps": 0.15, "gm": 0.15, "nm": 0.15, "ev_ebitda": 0.25, "de": 0.10, "fcy": 0.10, "runway": 0.10},
    "Unknown": {"ps": 0.25, "gm": 0.10, "nm": 0.10, "ev_ebitda": 0.25, "de": 0.05, "fcy": 0.10, "runway": 0.15},
}

# nycklar → kolumnnamn i df
_COLMAP = {
    "ps": None,  # hanteras via _ps_avg_from_row
    "gm": "Bruttomarginal (%)",
    "nm": "Netto-marginal (%)",
    "ev_ebitda": "EV/EBITDA",
    "de": "Skuldsättning (Debt/Equity)",
    "fcy": "FCF (milj)",         # free cash flow i miljoner (bolagets valuta)
    "runway": "Runway (kvartal)",
    "dy": "Direktavkastning (%)",
    "payout_cfo": "Payout ratio (CFO) (%)",
}

# ------------------------------------------------------------
# Normalisering och score
# ------------------------------------------------------------
def _winsorize_series(s: pd.Series, low_q=0.05, high_q=0.95) -> pd.Series:
    s = s.astype(float)
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    s = s.clip(lower=lo, upper=hi)
    return s

def _normalize_positive_good(s: pd.Series) -> pd.Series:
    s = _winsorize_series(s.dropna())
    if s.empty or s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def _normalize_negative_good(s: pd.Series) -> pd.Series:
    # lägre är bättre (t.ex. EV/EBITDA, Debt/Equity) → invertera
    pos = _normalize_positive_good(s)
    return 1.0 - pos

def _build_scoring_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    # extrahera ps_avg
    work["_ps_avg"] = work.apply(_ps_avg_from_row, axis=1)

    # kolumner med normalisering
    norm_cols = {}
    # ps → lägre P/S bättre i värderingsperspektiv (negativt bra)
    if work["_ps_avg"].notna().sum() > 0:
        s = work["_ps_avg"].astype(float)
        norm_cols["ps"] = _normalize_negative_good(s[s.notna()])
    # gm, nm, fcy, runway, dy → högre bättre
    for key in ["gm", "nm", "fcy", "runway", "dy"]:
        col = _COLMAP[key]
        if col in work.columns:
            s = pd.to_numeric(work[col], errors="coerce")
            if s.notna().sum() > 0:
                norm_cols[key] = _normalize_positive_good(s[s.notna()])
    # ev/ebitda, de, payout_cfo → lägre bättre
    for key in ["ev_ebitda", "de", "payout_cfo"]:
        col = _COLMAP[key]
        if col in work.columns:
            s = pd.to_numeric(work[col], errors="coerce")
            if s.notna().sum() > 0:
                norm_cols[key] = _normalize_negative_good(s[s.notna()])

    # init score=0
    work["_score_raw"] = 0.0
    work["_score_terms"] = 0.0  # antal faktiskt vägda termer för omfördelning

    # räkna ut sektorspecifik viktning per rad
    def _row_score(ix, row):
        sector = _sector(row)
        weights = _DEFAULT_SECTOR_WEIGHTS.get(sector, _DEFAULT_SECTOR_WEIGHTS["Unknown"]).copy()

        # håll bara vikter med data
        use_weights = {}
        for k, w in weights.items():
            if k == "ps" and row["_ps_avg"] is not None:
                use_weights[k] = w
            elif k != "ps":
                col = _COLMAP.get(k)
                if col and col in df.columns and pd.notna(row.get(col)):
                    use_weights[k] = w

        if not use_weights:
            return 0.0, 0

        # normalisera vikter till 1.0
        s = sum(use_weights.values())
        if s <= 0:
            return 0.0, 0
        for k in list(use_weights.keys()):
            use_weights[k] = use_weights[k] / s

        # samla ihop normaliserade poäng
        score = 0.0
        terms = 0
        for k, w in use_weights.items():
            if k == "ps":
                # hämta normaliserat värde från norm_cols["ps"]
                if "ps" in norm_cols and ix in norm_cols["ps"].index:
                    score += w * float(norm_cols["ps"].loc[ix])
                    terms += 1
            else:
                col = _COLMAP[k]
                if k in norm_cols and ix in norm_cols[k].index:
                    score += w * float(norm_cols[k].loc[ix])
                    terms += 1
                elif col in df.columns and pd.notna(row.get(col)):
                    # om inte normaliseringsserie innehåller detta index (borde inte hända), räkna 0
                    terms += 1

        return float(score), int(terms)

    for ix, row in work.iterrows():
        sc, tm = _row_score(ix, row)
        work.at[ix, "_score_raw"] = sc
        work.at[ix, "_score_terms"] = tm

    # datapålitlighet/komplettering – boost om fler nyckeltal finns
    key_cols = [
        "Bruttomarginal (%)", "Netto-marginal (%)", "EV/EBITDA", "Skuldsättning (Debt/Equity)",
        "FCF (milj)", "Kassa (milj)", "Runway (kvartal)",
        "Direktavkastning (%)", "Payout ratio (CFO) (%)", "Payout ratio (EPS) (%)",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt (4)"
    ]
    present_cnt = work.apply(lambda r: _present_cols(r, key_cols), axis=1)
    present_max = max(1, int(np.nanmax(present_cnt.values)))  # undvik division med 0
    completeness = present_cnt / present_max
    # 60% bas + 40% * completeness
    work["_score"] = work["_score_raw"] * (0.6 + 0.4 * completeness)

    return work


# ------------------------------------------------------------
# Riktkurs & etiketter
# ------------------------------------------------------------
def _compute_targets(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    w = df.copy()
    w["_ps_avg"] = w["_ps_avg"].astype(float)
    w["_target_price"] = np.nan
    w["_upside_pct"] = np.nan

    for ix, row in w.iterrows():
        px = safe_float(row.get("Aktuell kurs"), None)
        ps_avg = row.get("_ps_avg", None)
        rev = safe_float(row.get("Omsättning idag"), None)  # i miljoner, bolagets valuta
        shares = _get_shares_outstanding(row)
        cur = _currency(row)

        if ps_avg and rev and rev > 0 and shares and shares > 0:
            # Target mcap (i bolagets valuta) = rev(milj) * 1e6 * ps_avg
            target_mcap = float(rev) * 1e6 * float(ps_avg)
            target_price = target_mcap / float(shares)  # i bolagets valuta
            w.at[ix, "_target_price"] = target_price
            if px and px > 0:
                w.at[ix, "_upside_pct"] = (target_price / float(px) - 1.0) * 100.0

    # risklabel
    if "Market Cap" in w.columns:
        w["_RiskLabel"] = w["Market Cap"].apply(lambda x: risk_label_from_mcap(safe_float(x, np.nan)))
    else:
        w["_RiskLabel"] = "Okänd"

    return w

def _valuation_label(upside_pct: Optional[float], score: Optional[float]) -> str:
    """
    Kombinerar uppsida och totalpoäng till en etikett.
    Justera trösklar vid behov.
    """
    s = 0.0 if score is None or np.isnan(score) else float(score)
    u = -999.0 if upside_pct is None or np.isnan(upside_pct) else float(upside_pct)

    # primär bas på uppsida
    if u >= 30 and s >= 0.65:
        return "KÖP (stark)"
    if u >= 15 and s >= 0.55:
        return "Köp"
    if u >= 5 and s >= 0.45:
        return "Håll / Lägg bevakning"
    if -10 <= u < 5:
        return "Fair värderad"
    if -25 <= u < -10:
        return "Trimma / försiktig"
    if u < -25 and s <= 0.45:
        return "Övervärderad – säljvakt"
    return "Ok"

# ------------------------------------------------------------
# UI – Investeringsförslag (bläddringsläge som standard)
# ------------------------------------------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.subheader("Investeringsförslag")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    base = df.copy()

    # Risklabel (cap-segment)
    if "Market Cap" in base.columns:
        base["_RiskLabel"] = base["Market Cap"].apply(lambda x: risk_label_from_mcap(safe_float(x, np.nan)))
    else:
        base["_RiskLabel"] = "Okänd"

    sectors = ["Alla"] + sorted(list({str(x) for x in base.get("Sector", pd.Series(dtype=str)).fillna("Unknown")}))

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        sector_f = st.selectbox("Filtrera sektor", sectors, index=0)
    with c2:
        cap_f = st.selectbox("Filtrera cap-segment", ["Alla", "Micro", "Small", "Mid", "Large", "Mega"], index=0)
    with c3:
        view_mode = st.selectbox("Visningsläge", ["Bläddra", "Tabell"], index=0)

    # Filtrera
    if sector_f != "Alla":
        base = base.loc[base["Sector"].fillna("Unknown") == sector_f].copy()
    if cap_f != "Alla":
        base = base.loc[base["_RiskLabel"] == cap_f].copy()

    if base.empty:
        st.info("Inga bolag matchade filtret.")
        return

    # Scoring
    scored = _build_scoring_table(base)

    # Targets & uppsida
    scored = _compute_targets(scored, user_rates)
    scored["_ValuationLabel"] = scored.apply(lambda r: _valuation_label(r.get("_upside_pct"), r.get("_score")), axis=1)

    # sortering (högst score överst)
    scored = scored.sort_values(by=["_score", "_upside_pct"], ascending=[False, False])

    # Visning
    if view_mode == "Tabell":
        # visa kompakt tabell
        show = ["Ticker", "Bolagsnamn", "Sector", "Industry", "_RiskLabel",
                "_score", "_upside_pct", "Aktuell kurs", "_target_price",
                "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt (4)",
                "Market Cap"]
        show = [c for c in show if c in scored.columns]
        tbl = scored[show].rename(columns={
            "_score": "Score",
            "_upside_pct": "Uppsida (%)",
            "_target_price": "Riktkurs",
        })
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, hide_index=True)
        return

    # Bläddringsläge
    st.session_state.setdefault("_inv_idx", 0)
    n = len(scored)
    cL, cM, cR = st.columns([1, 2, 1])
    with cL:
        if st.button("◀ Föregående", key="inv_prev"):
            st.session_state["_inv_idx"] = max(0, int(st.session_state["_inv_idx"]) - 1)
    with cR:
        if st.button("Nästa ▶", key="inv_next"):
            st.session_state["_inv_idx"] = min(n - 1, int(st.session_state["_inv_idx"]) + 1)
    with cM:
        st.markdown(f"<div style='text-align:center; font-weight:600;'>Post {int(st.session_state['_inv_idx'])+1} av {n}</div>", unsafe_allow_html=True)

    idx = int(st.session_state["_inv_idx"])
    row = scored.iloc[idx]

    # Rubrik
    tkr = str(row.get("Ticker", ""))
    name = str(row.get("Bolagsnamn", "") or "")
    sector = _sector(row)
    industry = _industry(row)
    st.markdown(f"### {tkr} — {name}")
    st.caption(f"{sector} / {industry} • Cap: **{row.get('_RiskLabel','Okänd')}**")

    # Huvudmetrikrad
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Score", f"{float(row.get('_score',0))*100:.0f}/100")
    with c2:
        up = row.get("_upside_pct")
        st.metric("Uppsida (%)", "-" if up is None or np.isnan(up) else f"{up:.1f}%")
    with c3:
        cur = safe_float(row.get("Aktuell kurs"), None)
        st.metric("Aktuell kurs", "-" if cur is None else f"{cur:.2f} { _currency(row)}")
    with c4:
        tgt = row.get("_target_price")
        st.metric("Riktkurs", "-" if tgt is None or np.isnan(tgt) else f"{float(tgt):.2f} { _currency(row)}")
    with c5:
        st.metric("Etikett", row.get("_ValuationLabel", "Ok"))

    # Expander – alla nyckeltal
    with st.expander("Visa nyckeltal & detaljer", expanded=False):
        left, right = st.columns(2)

        # Vänster: värdering/struktur
        with left:
            st.write("**Värdering & struktur**")
            st.write(f"- **Market Cap:** { _fmt_money(safe_float(row.get('Market Cap'), np.nan)) }")
            st.write(f"- **Utestående aktier:** { _fmt_money(_get_shares_outstanding(row) or np.nan) }")
            st.write(f"- **Valuta:** { _currency(row) }")
            pa = _ps_avg_from_row(row)
            st.write(f"- **P/S snitt (4):** { '-' if pa is None else f'{pa:.2f}' }")
            for c in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S"]:
                if c in row.index and pd.notna(row[c]):
                    st.write(f"- **{c}:** {safe_float(row[c], None)}")
            if "EV/EBITDA" in row.index and pd.notna(row["EV/EBITDA"]):
                st.write(f"- **EV/EBITDA:** {safe_float(row['EV/EBITDA'], None)}")
            if "Skuldsättning (Debt/Equity)" in row.index and pd.notna(row["Skuldsättning (Debt/Equity)"]):
                st.write(f"- **Debt/Equity:** {safe_float(row['Skuldsättning (Debt/Equity)'], None)}")

            if "Omsättning idag" in row.index and pd.notna(row["Omsättning idag"]):
                st.write(f"- **Omsättning (i år, manuell):** { _fmt_money(safe_float(row['Omsättning idag'], None) * 1e6 if safe_float(row['Omsättning idag'], None) else None) }")

        # Höger: lönsamhet/utdelning/kassa
        with right:
            st.write("**Lönsamhet & kassaflöde**")
            if "Bruttomarginal (%)" in row.index and pd.notna(row["Bruttomarginal (%)"]):
                st.write(f"- **Bruttomarginal:** {_fmt_pct(safe_float(row['Bruttomarginal (%)'], None))}")
            if "Netto-marginal (%)" in row.index and pd.notna(row["Netto-marginal (%)"]):
                st.write(f"- **Netto-marginal:** {_fmt_pct(safe_float(row['Netto-marginal (%)'], None))}")
            if "FCF (milj)" in row.index and pd.notna(row["FCF (milj)"]):
                st.write(f"- **Free Cash Flow (milj):** {safe_float(row['FCF (milj)'], None)}")
            if "Kassa (milj)" in row.index and pd.notna(row["Kassa (milj)"]):
                st.write(f"- **Kassa (milj):** {safe_float(row['Kassa (milj)'], None)}")
            if "Runway (kvartal)" in row.index and pd.notna(row["Runway (kvartal)"]):
                st.write(f"- **Runway (kvartal):** {safe_float(row['Runway (kvartal)'], None)}")

            st.write("**Utdelning**")
            if "Direktavkastning (%)" in row.index and pd.notna(row["Direktavkastning (%)"]):
                st.write(f"- **Direktavkastning:** {_fmt_pct(safe_float(row['Direktavkastning (%)'], None))}")
            if "Payout ratio (CFO) (%)" in row.index and pd.notna(row["Payout ratio (CFO) (%)"]):
                st.write(f"- **Payout (CFO):** {_fmt_pct(safe_float(row['Payout ratio (CFO) (%)'], None))}")
            if "Payout ratio (EPS) (%)" in row.index and pd.notna(row["Payout ratio (EPS) (%)"]):
                st.write(f"- **Payout (EPS):** {_fmt_pct(safe_float(row['Payout ratio (EPS) (%)'], None))}")
            if "Utdelningstillväxt 5y (%)" in row.index and pd.notna(row["Utdelningstillväxt 5y (%)"]):
                st.write(f"- **Utdeln.tillväxt 5y:** {_fmt_pct(safe_float(row['Utdelningstillväxt 5y (%)'], None))}")
            if "Utdelningstillväxt 3y (%)" in row.index and pd.notna(row["Utdelningstillväxt 3y (%)"]):
                st.write(f"- **Utdeln.tillväxt 3y:** {_fmt_pct(safe_float(row['Utdelningstillväxt 3y (%)'], None))}")
            if "Utdelning år i rad" in row.index and pd.notna(row["Utdelning år i rad"]):
                st.write(f"- **År av utdelning i rad:** {int(safe_float(row['Utdelning år i rad'], 0))}")

    # Mini-topplista för kontext
    st.markdown("---")
    st.caption("Topp 10 (score-baserad):")
    mini = scored[["Ticker", "Bolagsnamn", "_score", "_upside_pct", "_ValuationLabel"]].head(10).reset_index(drop=True)
    mini = mini.rename(columns={"_score": "Score", "_upside_pct": "Uppsida (%)", "_ValuationLabel": "Etikett"})
    st.dataframe(mini, use_container_width=True, hide_index=True)
