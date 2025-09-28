# -*- coding: utf-8 -*-
"""
stockapp/views/analysis.py

Analys-vy:
- Välj bolag att granska
- Snabb översikt av nyckelfält
- Visar TS-etiketter (senaste Auto/Manuell + TS per spårat fält)
"""

from __future__ import annotations
from typing import Dict, List, Optional
import streamlit as st
import pandas as pd
import numpy as np

# Håll i synk med övriga moduler
TS_FIELDS: Dict[str, str] = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

def _fmt_money(v) -> str:
    try:
        x = float(v)
    except Exception:
        return "-"
    # kort formattering
    absx = abs(x)
    if absx >= 1_000_000_000_000:
        return f"{x/1_000_000_000_000:.2f} T"
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f} B"
    if absx >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    return f"{x:.2f}"

def _badge(ts: str, label: str) -> str:
    if not ts:
        return ""
    # liten grön/grå etikett
    return f"<span style='background:#eef7ee;border:1px solid #b7e1b7;border-radius:4px;padding:2px 6px;margin-left:6px;font-size:11px;color:#256029;'>{label}: {ts}</span>"

def _field_row(df_row: pd.Series, field: str, fmt: str = "num") -> str:
    """
    Returnerar html-rad med värde + ev TS-badge för field.
    fmt: "num" | "text" | "money"
    """
    if field not in df_row.index:
        return ""
    val = df_row.get(field, "")
    if fmt == "money":
        s = _fmt_money(val)
    elif fmt == "num":
        try:
            s = f"{float(val):.2f}"
        except Exception:
            s = str(val)
    else:
        s = str(val)

    ts_col = TS_FIELDS.get(field, "")
    ts_val = df_row.get(ts_col, "") if ts_col in df_row.index else ""
    badge = _badge(str(ts_val), "TS")
    return f"<div style='margin-bottom:6px;'><b>{field}:</b> {s} {badge}</div>"

def analysvy(df: pd.DataFrame) -> None:
    st.header("📈 Analys")

    if df.empty or "Ticker" not in df.columns:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = df.copy().sort_values(by=[c for c in ["Bolagsnamn","Ticker"] if c in df.columns]).reset_index(drop=True)
    etiketter = [f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})" for _, r in vis_df.iterrows()]

    # Index-hantering
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = max(0, min(st.session_state.analys_idx, len(vis_df)-1))

    # Val via selectbox
    choice = st.selectbox("Välj bolag", options=list(range(len(etiketter))), format_func=lambda i: etiketter[i], index=st.session_state.analys_idx)
    st.session_state.analys_idx = int(choice)

    # Bläddring
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_a:
        if st.button("⬅️ Föregående", use_container_width=True):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_c:
        if st.button("➡️ Nästa", use_container_width=True):
            st.session_state.analys_idx = min(len(vis_df)-1, st.session_state.analys_idx+1)
    st.caption(f"Post {st.session_state.analys_idx+1}/{len(vis_df)}")

    # Data för vald
    r = vis_df.iloc[st.session_state.analys_idx]
    title = f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})"
    st.subheader(title)

    # Toppband: uppdateringsetiketter
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Senast auto-uppdaterad", str(r.get("Senast auto-uppdaterad","")) or "–")
    with col2:
        st.metric("Senast manuellt uppdaterad", str(r.get("Senast manuellt uppdaterad","")) or "–")
    with col3:
        st.metric("Uppdaterad källa", str(r.get("Senast uppdaterad källa","")) or "–")

    st.divider()

    # Nyckelblock
    colL, colR = st.columns(2)
    with colL:
        # Pris & bas
        html = []
        html.append(_field_row(r, "Aktuell kurs", "num") + (f" <span style='color:#555;'>{r.get('Valuta','')}</span>" if 'Valuta' in r.index else ""))
        html.append(_field_row(r, "Utestående aktier", "num"))
        html.append(_field_row(r, "Årlig utdelning", "num"))
        html.append(_field_row(r, "CAGR 5 år (%)", "num"))
        st.markdown("\n".join(html), unsafe_allow_html=True)

        # P/S och historik
        html2 = []
        html2.append(_field_row(r, "P/S", "num"))
        for q in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
            if q in r.index:
                html2.append(_field_row(r, q, "num"))
        if "P/S-snitt" in r.index:
            try:
                psn = float(r.get("P/S-snitt", 0.0) or 0.0)
                html2.append(f"<div><b>P/S-snitt:</b> {psn:.2f}</div>")
            except Exception:
                html2.append(f"<div><b>P/S-snitt:</b> {r.get('P/S-snitt')}</div>")
        st.markdown("\n".join(html2), unsafe_allow_html=True)

    with colR:
        # Omsättningar
        html3 = []
        html3.append(_field_row(r, "Omsättning idag", "money"))
        html3.append(_field_row(r, "Omsättning nästa år", "money"))
        if "Omsättning om 2 år" in r.index:
            html3.append(_field_row(r, "Omsättning om 2 år", "money"))
        if "Omsättning om 3 år" in r.index:
            html3.append(_field_row(r, "Omsättning om 3 år", "money"))
        st.markdown("\n".join(html3), unsafe_allow_html=True)

        # Riktkurser
        html4 = []
        for fld in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
            if fld in r.index:
                html4.append(_field_row(r, fld, "num") + (f" <span style='color:#555;'>{r.get('Valuta','')}</span>" if 'Valuta' in r.index else ""))
        st.markdown("\n".join(html4), unsafe_allow_html=True)

    st.divider()

    # Full rad dump (valfritt)
    with st.expander("Visa alla fält (debug)", expanded=False):
        show_cols = [c for c in df.columns if c not in ["_oldest_any_ts","_oldest_any_ts_fill"]]
        st.dataframe(pd.DataFrame([r[show_cols].to_dict()]), use_container_width=True, hide_index=True)
