# -*- coding: utf-8 -*-
"""
stockapp/invest.py

Vyn "Investeringsförslag":
- Rankar bolag via sektor- och stilberoende scoring (Growth/Dividend).
- Filter: sektor + börsvärdesklass (Micro/Small/Mid/Large/Mega).
- Sök på ticker/namn.
- Paginering.
- Kort per bolag + expander med nyckeltal (alla tillgängliga fält).

Publik funktion:
- visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float], default_style: str = "growth") -> None
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st

from .scoring import score_dataframe

# -----------------------------------------------------------
# Hjälpfunktioner (självförsörjande – inga hårda beroenden)
# -----------------------------------------------------------

def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str):
            if not x.strip():
                return float(default)
            x = x.replace(" ", "").replace(",", ".")
        return float(x)
    except Exception:
        return float(default)

def _fmt_money(v: float, currency: Optional[str] = None) -> str:
    """
    Snygg formattering av stora tal med T/B/M/K-suffix.
    """
    try:
        n = float(v)
    except Exception:
        return "-"
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12:
        s = f"{n/1e12:.2f}T"
    elif n >= 1e9:
        s = f"{n/1e9:.2f}B"
    elif n >= 1e6:
        s = f"{n/1e6:.2f}M"
    elif n >= 1e3:
        s = f"{n/1e3:.0f}K"
    else:
        s = f"{n:.0f}"
    return f"{sign}{s}{(' ' + currency) if currency else ''}"

def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1f}%"
    except Exception:
        return "-"

def _risk_label_from_mcap(mcap: float) -> str:
    """
    Enkelt labelsystem:
      Micro: < 0.3B
      Small: 0.3–2B
      Mid:   2–10B
      Large: 10–200B
      Mega:  >= 200B
    """
    x = _safe_float(mcap, 0.0)
    if x < 3e8:
        return "Micro"
    if x < 2e9:
        return "Small"
    if x < 1e10:
        return "Mid"
    if x < 2e11:
        return "Large"
    return "Mega"

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def _collect_metrics_for_expander(row: pd.Series) -> List[Tuple[str, str]]:
    """
    Samlar ihop alla relevanta fält som kan finnas och returnerar (etikett, värde_str).
    Visar bara de som faktiskt finns i raden.
    """
    # Ordning/gruppering – lägg gärna till vid behov.
    blocks: List[Tuple[str, str, str]] = []

    # Övergripande/identitet
    blocks.append(("Ticker", str(row.get("Ticker", "-")), "text"))
    blocks.append(("Namn", str(row.get("Bolag", row.get("Name", "")) or "-"), "text"))
    blocks.append(("Sektor", str(row.get("Sector", "-")), "text"))
    blocks.append(("Risklabel", _risk_label_from_mcap(_safe_float(row.get("Market Cap"))), "text"))

    # Börsvärde / riktkurs / uppsida
    blocks.append(("Market Cap", _fmt_money(_safe_float(row.get("Market Cap")), None), "text"))
    if pd.notna(row.get("Uppsida (%)", None)):
        blocks.append(("Uppsida (P/S)", _fmt_pct(row.get("Uppsida (%)")), "text"))
    if pd.notna(row.get("Riktkurs (valuta)", None)):
        blocks.append(("Riktkurs (valuta)", f"{_safe_float(row.get('Riktkurs (valuta)')):.2f}", "text"))

    # Intäkter & prognos
    for k in [
        "Omsättning i år (förv.)", "Omsättning i år",
        "Revenue This Year (Est.)", "Revenue (Current FY Est.)"
    ]:
        if k in row and pd.notna(row[k]):
            blocks.append((k, _fmt_money(_safe_float(row[k])), "text"))
            break

    # P/S, historik
    for k in ["P/S", "P/S-snitt", "P/S snitt", "PS Avg"]:
        if k in row and pd.notna(row[k]):
            blocks.append((k, f"{_safe_float(row[k]):.2f}", "text"))
            break
    for q in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]:
        if q in row and pd.notna(row[q]):
            blocks.append((q, f"{_safe_float(row[q]):.2f}", "text"))

    # Marginaler
    for k in ["Gross margin (%)", "Operating margin (%)", "Net margin (%)"]:
        if k in row and pd.notna(row[k]):
            blocks.append((k, _fmt_pct(row[k]), "text"))

    # Lönsamhet & värdering
    for k in ["ROE (%)", "P/B", "EV/EBITDA (ttm)", "P/S"]:
        if k in row and pd.notna(row[k]):
            if k.endswith("(ttm)") or k in ("P/B", "P/S"):
                blocks.append((k, f"{_safe_float(row[k]):.2f}", "text"))
            else:
                blocks.append((k, _fmt_pct(row[k]), "text"))

    # Skuld & utdelning
    for k in ["Net debt / EBITDA", "Debt/Equity"]:
        if k in row and pd.notna(row[k]):
            blocks.append((k, f"{_safe_float(row[k]):.2f}", "text"))
    for k in ["Dividend yield (%)", "Dividend payout (FCF) (%)"]:
        if k in row and pd.notna(row[k]):
            blocks.append((k, _fmt_pct(row[k]), "text"))

    # Kassaflöde & kassa
    for k in ["FCF Yield (%)", "Operating Cash Flow", "CapEx", "Free Cash Flow",
              "Kassa & Kortfristiga placeringar", "Netto-kassa/skuld"]:
        if k in row and pd.notna(row[k]):
            if k.endswith("Yield (%)"):
                blocks.append((k, _fmt_pct(row[k]), "text"))
            else:
                blocks.append((k, _fmt_money(_safe_float(row[k])), "text"))

    # Runway
    for k in ["Cash Runway (kvartal)"]:
        if k in row and pd.notna(row[k]):
            blocks.append((k, f"{_safe_float(row[k]):.1f}", "text"))

    # Antal aktier
    if pd.notna(row.get("Utestående aktier", None)):
        blocks.append(("Utestående aktier (milj.)", f"{_safe_float(row.get('Utestående aktier')):.2f}", "text"))

    # Score-detaljer (kan ha kommit från scoring; annars ignoreras)
    if isinstance(row.get("ScoreBreakdown"), dict):
        # Lägg sist
        sb = row["ScoreBreakdown"]
        # sortera efter högst delprocent
        for k, v in sorted(sb.items(), key=lambda kv: kv[1], reverse=True):
            blocks.append((f"Score: {k}", f"{float(v):.1f}%", "text"))

    # Ta med tidsstämplar om de finns (för kontroll)
    for k in [
        "TS Pris", "TS P/S", "TS Market Cap", "TS FCF", "TS Marginaler", "TS Skuld", "TS Utdelning",
        "TS Prognos i år", "TS Prognos nästa år"
    ]:
        if k in row and pd.notna(row[k]):
            blocks.append((k, str(row[k]), "text"))

    # Omvandla till [(label, value_str)]
    out_pairs: List[Tuple[str, str]] = []
    for (label, val, _kind) in blocks:
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        out_pairs.append((label, str(val)))
    return out_pairs

def _init_pagination_state(key_page: str) -> None:
    if key_page not in st.session_state:
        st.session_state[key_page] = 0

# -----------------------------------------------------------
# Huvudvy
# -----------------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float], default_style: str = "growth") -> None:
    """
    Renderar investeringsförslagsvyn.
    - df: portfölj-DF (alla bolag)
    - user_rates: valutakurser (används i andra vyer, ej direkt här)
    - default_style: "growth" eller "dividend"
    """
    st.header("📊 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Säkerställ kolumner vi använder
    need_cols = ["Ticker", "Bolag", "Name", "Sector", "Market Cap"]
    df = _ensure_cols(df, need_cols)

    # UI – filterrad
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.2, 1.2])

    with c1:
        style = st.radio(
            "Strategi",
            options=["Growth", "Dividend"],
            horizontal=True,
            index=(0 if str(default_style).lower().startswith("g") else 1),
            help="Välj om poängsättningen ska viktas som 'Growth' eller 'Dividend'.",
        )

    sector_vals = sorted([s for s in df["Sector"].dropna().unique() if str(s).strip()])
    with c2:
        sectors = st.multiselect(
            "Sektorfilter", sector_vals, default=[],
            help="Lämna tomt för alla sektorer."
        )

    with c3:
        size = st.selectbox(
            "Börsvärdesklass",
            options=["Alla", "Micro", "Small", "Mid", "Large", "Mega"],
            index=0,
            help="Filtrera på risklabel baserat på market cap."
        )

    with c4:
        query = st.text_input("Sök (ticker/namn)", value="", placeholder="t.ex. NVDA eller 'Nvidia'")

    # Poängsättning (style → growth/dividend)
    style_key = "dividend" if style.lower().startswith("d") else "growth"
    scored = score_dataframe(df, style=style_key)

    # Risklabel in i DF (för filter/visning)
    if "Market Cap" not in scored.columns:
        scored["Market Cap"] = np.nan
    scored["_RiskLabel"] = scored["Market Cap"].apply(_risk_label_from_mcap)

    # Filtrera
    filt = scored.copy()
    if sectors:
        filt = filt[filt["Sector"].isin(sectors)]
    if size != "Alla":
        filt = filt[filt["_RiskLabel"] == size]
    if query.strip():
        q = query.strip().lower()
        mask = (
            filt.get("Ticker", pd.Series([""]*len(filt))).astype(str).str.lower().str.contains(q, na=False)
            | filt.get("Bolag", pd.Series([""]*len(filt))).astype(str).str.lower().str.contains(q, na=False)
            | filt.get("Name", pd.Series([""]*len(filt))).astype(str).str.lower().str.contains(q, na=False)
        )
        filt = filt[mask]

    # Sortera – TotalScore desc, Coverage desc
    if "TotalScore" in filt.columns:
        filt = filt.sort_values(by=["TotalScore", "Coverage"], ascending=[False, False])

    # Tomt efter filter?
    if filt.empty:
        st.warning("Inga träffar för dina filter/sök.")
        return

    # Paginering
    _init_pagination_state("_inv_page")
    page_size = st.selectbox("Poster per sida", options=[5, 10, 20, 50], index=1)
    total = len(filt)
    pages = int(math.ceil(total / page_size))
    page = st.session_state["_inv_page"]
    if page >= pages:
        page = pages - 1
        page = max(0, page)
        st.session_state["_inv_page"] = page

    cpg1, cpg2, cpg3 = st.columns([1, 2, 1])
    with cpg1:
        if st.button("⬅️ Föregående", use_container_width=True, disabled=(page <= 0)):
            st.session_state["_inv_page"] = max(0, page - 1)
            st.experimental_rerun()
    with cpg2:
        st.markdown(f"<div style='text-align:center;'>Sida {page+1} / {pages} &nbsp;&middot;&nbsp; "
                    f"{total} träffar</div>", unsafe_allow_html=True)
    with cpg3:
        if st.button("Nästa ➡️", use_container_width=True, disabled=(page >= pages - 1)):
            st.session_state["_inv_page"] = min(pages - 1, page + 1)
            st.experimental_rerun()

    start = page * page_size
    stop = min(start + page_size, total)
    view = filt.iloc[start:stop].copy()

    # Lista/kort
    for _, row in view.iterrows():
        ticker = str(row.get("Ticker", "—"))
        name = str(row.get("Bolag", row.get("Name", "")) or "")
        sector = str(row.get("Sector", "—"))
        mcap = _fmt_money(_safe_float(row.get("Market Cap")))
        rec = str(row.get("Recommendation", "—"))
        score = f"{_safe_float(row.get('TotalScore')):.1f}"
        cov = f"{_safe_float(row.get('Coverage')):.1f}%"

        ups_str = "-"
        if pd.notna(row.get("Uppsida (%)", None)):
            ups_str = _fmt_pct(row.get("Uppsida (%)"))
        tgt_str = ""
        if pd.notna(row.get("Riktkurs (valuta)", None)):
            tgt_str = f" • Riktkurs: {float(row.get('Riktkurs (valuta)')):.2f}"

        with st.container(border=True):
            st.markdown(
                f"### {ticker} — {name}"
            )
            st.caption(f"Sektor: **{sector}** • Börsvärde: **{mcap}** • Risklabel: **{_risk_label_from_mcap(_safe_float(row.get('Market Cap')))}**")
            st.markdown(
                f"**Rekommendation:** {rec}  &nbsp;|&nbsp;  "
                f"**Score:** {score}  &nbsp;|&nbsp;  **Täckning:** {cov}  &nbsp;|&nbsp;  "
                f"**Uppsida (P/S):** {ups_str}{tgt_str}"
            )

            with st.expander("Visa nyckeltal & detaljer"):
                pairs = _collect_metrics_for_expander(row)
                if not pairs:
                    st.info("Inga extra nyckeltal tillgängliga.")
                else:
                    # Visa i två kolumner för bättre läsbarhet
                    left, right = st.columns(2)
                    half = (len(pairs) + 1) // 2
                    for i, (label, val) in enumerate(pairs):
                        target_col = left if i < half else right
                        with target_col:
                            st.write(f"**{label}:** {val}")

    st.divider()
    # Liten sammanfattning
    avg_score = float(view["TotalScore"].dropna().mean()) if "TotalScore" in view.columns else np.nan
    avg_cov = float(view["Coverage"].dropna().mean()) if "Coverage" in view.columns else np.nan
    st.caption(
        f"Snittscore (sidan): {avg_score:.1f} • Snitt täckning: {avg_cov:.1f}% • Strategi: {style}"
    )
