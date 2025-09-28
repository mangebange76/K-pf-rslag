# stockapp/views/invest.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from ..calc import human_mcap, marketcap_risk_label, safe_float

CAP_BUCKETS = ["Microcap", "Smallcap", "Midcap", "Largecap", "Megacap"]

def _cap_label_from_row(row: pd.Series) -> str:
    """Försök bestämma cap-klass. Använd befintlig Market Cap-kolumn, annars räkna px*shares."""
    mcap = 0.0
    # försök hämta ev. Market Cap (nu)
    for k in ["Market Cap (nu)", "MarketCap", "MarketCapNow"]:
        if k in row and pd.notna(row[k]):
            mcap = safe_float(row[k], 0.0)
            break
    if mcap <= 0:
        # fallback: implied via kurs * shares (shares i styck; kolumnen är i miljoner)
        px = safe_float(row.get("Aktuell kurs", 0.0), 0.0)
        sh_m = safe_float(row.get("Utestående aktier", 0.0), 0.0)  # miljoner
        if px > 0 and sh_m > 0:
            mcap = px * sh_m * 1e6
    return marketcap_risk_label(mcap)

def _compute_mcap_now(row: pd.Series) -> float:
    # samma som ovan, men numeriskt värde
    for k in ["Market Cap (nu)", "MarketCap", "MarketCapNow"]:
        if k in row and pd.notna(row[k]):
            v = safe_float(row[k], 0.0)
            if v > 0:
                return v
    px = safe_float(row.get("Aktuell kurs", 0.0), 0.0)
    sh_m = safe_float(row.get("Utestående aktier", 0.0), 0.0)
    if px > 0 and sh_m > 0:
        return px * sh_m * 1e6
    return 0.0

def _composite_score(row: pd.Series, focus: str) -> float:
    g = safe_float(row.get("GrowthScore", 0.0), 0.0)
    d = safe_float(row.get("DividendScore", 0.0), 0.0)
    if focus == "Tillväxt":
        return 0.85 * g + 0.15 * d
    elif focus == "Utdelning":
        return 0.25 * g + 0.75 * d
    return 0.5 * g + 0.5 * d  # Balanserad

def _row_potential(row: pd.Series, target_col: str) -> float:
    px = safe_float(row.get("Aktuell kurs", 0.0), 0.0)
    tgt = safe_float(row.get(target_col, 0.0), 0.0)
    if px > 0 and tgt > 0:
        return (tgt - px) / px * 100.0
    return 0.0

def _ps_now(row: pd.Series) -> float:
    """P/S (TTM) nu om kolumn finns, annars använd P/S som proxy."""
    cand = safe_float(row.get("P/S", 0.0), 0.0)
    return round(cand, 2) if cand > 0 else 0.0

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    if df is None or df.empty:
        st.info("Ingen data att visa ännu.")
        return

    # Välj fokus & filter
    colf1, colf2 = st.columns([1,1])
    with colf1:
        fokus = st.radio("Fokus", ["Balanserad", "Tillväxt", "Utdelning"], horizontal=True, index=0, key="inv_focus")
    with colf2:
        target_col = st.selectbox("Riktkurs att jämföra mot", ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"], index=0, key="inv_tgt")

    colf3, colf4 = st.columns([1,1])
    with colf3:
        sektorer = sorted([s for s in df.get("Sektor", pd.Series(dtype=str)).dropna().unique() if str(s).strip()])
        val_sekt = st.multiselect("Filtrera på sektor (valfritt)", sektorer, default=[], key="inv_sectors")
    with colf4:
        val_caps = st.multiselect("Cap-klass (valfritt)", CAP_BUCKETS, default=[], key="inv_caps")

    only_with_target = st.checkbox("Visa endast bolag med vald riktkurs > 0", value=True, key="inv_only_tgt")

    # Bygg arbetskopia
    work = df.copy()
    # lägg till risklabel/cap
    work["CapLabel"] = work.apply(_cap_label_from_row, axis=1)
    # filtrera sektor
    if val_sekt:
        work = work[work["Sektor"].astype(str).isin(val_sekt)]
    # filtrera caps
    if val_caps:
        work = work[work["CapLabel"].astype(str).isin(val_caps)]
    # filtrera riktkurs
    if only_with_target and target_col in work.columns:
        work = work[safe_float_series(work[target_col]) > 0]

    if work.empty:
        st.warning("Inget matchade filtren.")
        return

    # Beräkna ranking-score per rad
    work["CompositeScore"] = work.apply(lambda r: _composite_score(r, fokus), axis=1)
    work["Potential (%)"] = work.apply(lambda r: round(_row_potential(r, target_col), 2), axis=1)
    work["Direktavkastning (%)"] = work.get("Direktavkastning (%)", pd.Series([0.0]*len(work)))

    # Sorteras: primärt score, sekundärt potential (fallande)
    work = work.sort_values(by=["CompositeScore","Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # Navigering (robust)
    n = len(work)
    st.session_state.setdefault("forslags_idx", 0)
    st.session_state.forslags_idx = int(np.clip(st.session_state.forslags_idx, 0, max(0, n-1)))

    nav1, nav2, nav3 = st.columns([1,2,1])
    with nav1:
        if st.button("⬅️ Föregående", use_container_width=True):
            st.session_state.forslags_idx = max(0, st.session_state.forslags_idx - 1)
    with nav2:
        st.markdown(f"<div style='text-align:center;'>Förslag {st.session_state.forslags_idx+1} / {n}</div>", unsafe_allow_html=True)
    with nav3:
        if st.button("➡️ Nästa", use_container_width=True):
            st.session_state.forslags_idx = min(n-1, st.session_state.forslags_idx + 1)

    row = work.iloc[st.session_state.forslags_idx]

    # Visa kort info + expander
    st.subheader(f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})")
    k1, k2, k3, k4 = st.columns([1,1,1,1])
    with k1:
        st.metric("Kurs", f"{safe_float(row.get('Aktuell kurs',0)):.2f} {row.get('Valuta','')}")
    with k2:
        st.metric("Vald riktkurs", f"{safe_float(row.get(target_col,0)):.2f} {row.get('Valuta','')}")
    with k3:
        st.metric("Uppsida", f"{safe_float(row.get('Potential (%)',0)):.2f}%")
    with k4:
        st.metric("Fokus-score", f"{safe_float(row.get('CompositeScore',0)):.1f}")

    with st.expander("📦 Detaljer", expanded=True):
        mcap_now = _compute_mcap_now(row)
        ps_now = _ps_now(row)
        ps_avg = safe_float(row.get("P/S-snitt", 0.0), 0.0)
        shares_m = safe_float(row.get("Utestående aktier", 0.0), 0.0)
        sektor = str(row.get("Sektor","") or "")
        caplab = _cap_label_from_row(row)
        vard = str(row.get("Värdering","") or "")

        left, right = st.columns([1,1])
        with left:
            st.write(f"- **Market Cap (nu):** {human_mcap(mcap_now)}")
            st.write(f"- **P/S (nu):** {ps_now:.2f}")
            st.write(f"- **P/S-snitt (Q1–Q4):** {ps_avg:.2f}")
            st.write(f"- **Utestående aktier:** {shares_m:.2f} M")
            if "Årlig utdelning" in row:
                st.write(f"- **Direktavkastning:** {safe_float(row.get('Direktavkastning (%)',0.0)):.2f}%")
        with right:
            st.write(f"- **Sektor:** {sektor or '—'}")
            st.write(f"- **Cap-klass:** {caplab}")
            st.write(f"- **GrowthScore:** {safe_float(row.get('GrowthScore',0.0)):.1f}")
            st.write(f"- **DividendScore:** {safe_float(row.get('DividendScore',0.0)):.1f}")
            st.write(f"- **Värdering:** {vard or '—'}")

        # Visa ev. historiska mcap-kolumner om de finns
        mcols = [c for c in row.index if c.lower().startswith("mcap q") or c.lower().startswith("market cap q")]
        if mcols:
            st.write("**Historisk Market Cap (senaste Q):**")
            lines = []
            for c in sorted(mcols):
                lines.append(f"- {c}: {human_mcap(safe_float(row.get(c,0.0)))}")
            st.markdown("\n".join(lines))

    # Tabell med topp 15 för snabb överblick
    st.markdown("### 📋 Snabböversikt (topp 15 efter filter)")
    show = work.head(15).copy()
    show_cols = ["Ticker","Bolagsnamn","Sektor","CapLabel","Aktuell kurs",target_col,"Potential (%)","P/S","P/S-snitt","GrowthScore","DividendScore","Värdering"]
    show_cols = [c for c in show_cols if c in show.columns]
    if "Aktuell kurs" in show.columns:
        show.rename(columns={"Aktuell kurs": "Kurs"}, inplace=True)
    st.dataframe(
        show[show_cols],
        use_container_width=True,
        hide_index=True
    )


# ------------------------------
# Hjälp: safe_float på Series
# ------------------------------
def safe_float_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    except Exception:
        return pd.Series([0.0]*len(s))
