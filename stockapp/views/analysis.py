# stockapp/views/analysis.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from ..config import (
    säkerställ_kolumner,
    konvertera_typer,
    TS_FIELDS,
    add_oldest_ts_col,
)
from ..calc import human_mcap, marketcap_risk_label, safe_float


# -----------------------------
# Små UI-hjälpare
# -----------------------------

def _chip(text: str, color: str = "#EEF2FF", fg: str = "#111827") -> str:
    return (
        f"<span style='display:inline-block;"
        f"padding:2px 8px;border-radius:999px;"
        f"background:{color};color:{fg};font-size:12px;"
        f"margin-right:6px;margin-bottom:6px;'>{text}</span>"
    )

def _kv(label: str, value: str) -> str:
    return f"<div style='margin-bottom:6px;'><b>{label}:</b> {value}</div>"

def _fmt_num(x, dec=2):
    try:
        return f"{float(x):.{dec}f}"
    except Exception:
        return "0.00"


# -----------------------------
# Analys-vy
# -----------------------------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    if df is None or df.empty:
        st.info("Ingen data inläst ännu.")
        return

    # Säkerställ schema/typer
    df = säkerställ_kolumner(df.copy())
    df = konvertera_typer(df)

    # Sortera visningslista
    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if not etiketter:
        st.info("Inga bolag i databasen ännu.")
        return

    # Robust index i sessionen
    st.session_state.setdefault("analys_idx", 0)
    st.session_state.analys_idx = int(np.clip(st.session_state.analys_idx, 0, len(etiketter)-1))

    # Välj bolag
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        chosen = st.selectbox("Välj bolag", etiketter, index=st.session_state.analys_idx, key="analys_selectbox")
        # Synka index när selectbox ändras
        if chosen:
            st.session_state.analys_idx = etiketter.index(chosen)
    with col_sel2:
        st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    # Bläddring
    nav1, nav2 = st.columns([1,1])
    with nav1:
        if st.button("⬅️ Föregående", use_container_width=True):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with nav2:
        if st.button("➡️ Nästa", use_container_width=True):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx + 1)

    # Rad att visa
    r = vis_df.iloc[st.session_state.analys_idx]

    # Titelrad
    st.subheader(f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})")

    # Badges: senaste uppdateringar
    man_dt = str(r.get("Senast manuellt uppdaterad","") or "").strip()
    auto_dt = str(r.get("Senast auto-uppdaterad","") or "").strip()
    source  = str(r.get("Senast uppdaterad källa","") or "").strip()

    badge_html = ""
    if auto_dt:
        badge_html += _chip(f"Auto: {auto_dt}", color="#DCFCE7", fg="#065F46")
    if man_dt:
        badge_html += _chip(f"Manuellt: {man_dt}", color="#E0E7FF", fg="#1E3A8A")
    if source:
        badge_html += _chip(f"Källa: {source}", color="#FFE4E6", fg="#9F1239")
    if badge_html:
        st.markdown(badge_html, unsafe_allow_html=True)

    # Snabbkort med nyckeltal
    top1, top2, top3, top4 = st.columns([1,1,1,1])
    with top1:
        st.metric("Kurs", f"{_fmt_num(r.get('Aktuell kurs',0.0))} {r.get('Valuta','')}")
        st.metric("Utest. aktier (M)", f"{_fmt_num(r.get('Utestående aktier',0.0),2)}")
    with top2:
        ps_now = safe_float(r.get("P/S", 0.0))
        st.metric("P/S (nu)", f"{ps_now:.2f}")
        ps_avg = safe_float(r.get("P/S-snitt", 0.0))
        st.metric("P/S-snitt (Q1–Q4)", f"{ps_avg:.2f}")
    with top3:
        # Market cap nu via kurs*aktier om inte kolumn finns
        mcap = safe_float(r.get("Market Cap (nu)", 0.0))
        if mcap <= 0:
            px = safe_float(r.get("Aktuell kurs", 0.0))
            sh_m = safe_float(r.get("Utestående aktier", 0.0))
            mcap = px * sh_m * 1e6 if (px > 0 and sh_m > 0) else 0.0
        st.metric("Market Cap", human_mcap(mcap))
        st.metric("Cap-klass", marketcap_risk_label(mcap))
    with top4:
        st.metric("GrowthScore", f"{safe_float(r.get('GrowthScore',0.0)):.1f}")
        st.metric("DividendScore", f"{safe_float(r.get('DividendScore',0.0)):.1f}")

    # Riktkurser & omsättning
    block1, block2 = st.columns([1,1])
    with block1:
        st.markdown("#### Riktkurser")
        lines = [
            _kv("Riktkurs idag", f"{_fmt_num(r.get('Riktkurs idag',0.0))} {r.get('Valuta','')}"),
            _kv("Riktkurs om 1 år", f"{_fmt_num(r.get('Riktkurs om 1 år',0.0))} {r.get('Valuta','')}"),
            _kv("Riktkurs om 2 år", f"{_fmt_num(r.get('Riktkurs om 2 år',0.0))} {r.get('Valuta','')}"),
            _kv("Riktkurs om 3 år", f"{_fmt_num(r.get('Riktkurs om 3 år',0.0))} {r.get('Valuta','')}"),
        ]
        st.markdown("".join(lines), unsafe_allow_html=True)

    with block2:
        st.markdown("#### Omsättning (miljoner)")
        lines = [
            _kv("Idag (manuell)", _fmt_num(r.get("Omsättning idag",0.0))),
            _kv("Nästa år (manuell)", _fmt_num(r.get("Omsättning nästa år",0.0))),
            _kv("Om 2 år", _fmt_num(r.get("Omsättning om 2 år",0.0))),
            _kv("Om 3 år", _fmt_num(r.get("Omsättning om 3 år",0.0))),
        ]
        st.markdown("".join(lines), unsafe_allow_html=True)

    # Expander: övrigt + P/S Q1–Q4 + TS
    with st.expander("🔎 Detaljer & tidsstämplar", expanded=False):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.markdown("**P/S kvartal**")
            for q in range(1,5):
                st.write(f"- P/S Q{q}: {_fmt_num(r.get(f'P/S Q{q}',0.0))}")
        with c2:
            st.markdown("**Övrigt**")
            st.write(f"- Årlig utdelning: {_fmt_num(r.get('Årlig utdelning',0.0))} {r.get('Valuta','')}")
            st.write(f"- CAGR 5 år: {_fmt_num(r.get('CAGR 5 år (%)',0.0))}%")
            st.write(f"- Sektor: {str(r.get('Sektor','') or '—')}")
            st.write(f"- Värdering: {str(r.get('Värdering','') or '—')}")
        with c3:
            st.markdown("**Rad-uppdateringar**")
            st.write(f"- Senast auto: {auto_dt or '—'}")
            st.write(f"- Senast manuellt: {man_dt or '—'}")
            st.write(f"- Källa: {source or '—'}")

        st.markdown("---")
        st.markdown("**TS per fält (senaste ändring)**")
        # Lista alla TS_-kolumner i rad
        ts_cols = [c for c in r.index if str(c).startswith("TS_")]
        if not ts_cols:
            st.write("—")
        else:
            ts_df = pd.DataFrame(
                [{"Fält": c.replace("TS_", ""), "Datum": str(r.get(c,''))}] for c in ts_cols
            )
            st.dataframe(ts_df, use_container_width=True, hide_index=True)

    # Valfritt: rådata-tabell för raden
    with st.expander("🧾 Radrådata"):
        show_cols = [
            "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
            "Utestående aktier","Årlig utdelning","CAGR 5 år (%)",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "GrowthScore","DividendScore","Värdering",
            "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"
        ]
        show_cols = [c for c in show_cols if c in r.index]
        st.dataframe(
            pd.DataFrame([r[show_cols].to_dict()]),
            use_container_width=True,
            hide_index=True
        )
