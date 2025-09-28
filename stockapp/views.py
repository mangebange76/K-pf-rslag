# stockapp/views.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import streamlit as st

from .editor import editor_view
from .batch import sidebar_batch_controls
from .sources import _safe_float, _now_stamp

# -------------------------------------------------------------------
# Små utils
# -------------------------------------------------------------------

def _ensure_col(df: pd.DataFrame, col: str, default=0.0):
    if col not in df.columns:
        df[col] = default

def _fmt_big(n: float) -> str:
    """Enkel formattering: T, B, M, K."""
    try:
        x = float(n)
    except Exception:
        return str(n)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12:
        return f"{sign}{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{sign}{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{sign}{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}{x/1e3:.0f}K"
    return f"{sign}{x:.0f}"

def _rate_for(ccy: str, user_rates: Dict[str, float]) -> float:
    if not ccy:
        return 1.0
    return float(user_rates.get(str(ccy).upper(), 1.0))

def _risk_label(mcap: float) -> str:
    # Grov klassning (USD-trösklar). Används mest som snabb "storleks-signal".
    v = _safe_float(mcap, 0.0)
    if v >= 2e11: return "Mega"
    if v >= 1e10: return "Large"
    if v >= 2e9:  return "Mid"
    if v >= 3e8:  return "Small"
    return "Micro"

# -------------------------------------------------------------------
# 1) Kontroll-vy
# -------------------------------------------------------------------

def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for c in row.index:
        if str(c).startswith("TS_") and str(row[c]).strip():
            d = pd.to_datetime(str(row[c]), errors="coerce")
            if pd.notna(d):
                dates.append(d)
    if not dates:
        return None
    return min(dates)

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # Äldst uppdaterade spårade fält (topp 20)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    if df.empty:
        st.info("Ingen data än.")
    else:
        work = df.copy()
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("1900-01-01"))
        vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
        cols = ["Ticker","Bolagsnamn"]
        ts_cols = [c for c in df.columns if str(c).startswith("TS_")]
        cols.extend(ts_cols[:10])  # visa inte extremt brett
        cols.append("_oldest_any_ts")
        st.dataframe(vis[cols], use_container_width=True, hide_index=True)

    st.divider()

    # Batch-panel i sidopanelen kan redan vara renderad via main, men vi visar ev. logg:
    st.subheader("📒 Senaste körlogg (Batch)")
    log = st.session_state.get("_batch_log", {})
    if not log or (not log.get("ok") and not log.get("fail")):
        st.info("Ingen batch-körning i denna session ännu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**OK**")
            oks = log.get("ok", [])
            st.write(", ".join(oks[:50]) + ("…" if len(oks) > 50 else ""))
        with col2:
            st.markdown("**Fail**")
            fls = log.get("fail", [])
            st.write(", ".join(fls[:50]) + ("…" if len(fls) > 50 else ""))
        with st.expander("Detaljer"):
            st.json(log.get("details", []))

# -------------------------------------------------------------------
# 2) Analys-vy
# -------------------------------------------------------------------

def analysvy(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    work = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in work.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(labels)-1), value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", labels, index=st.session_state.analys_idx if labels else 0, key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(labels)-1, st.session_state.analys_idx+1)

    st.write(f"Post {st.session_state.analys_idx+1}/{len(labels)}")
    r = work.iloc[st.session_state.analys_idx]

    cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","Utestående aktier","Market Cap (nu)",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)",
        "Antal aktier","Årlig utdelning","GAV SEK",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    ]
    for c in cols:
        _ensure_col(df, c, "")

    row_df = pd.DataFrame([r[cols].to_dict()])
    # berikning för snygg mcap
    if "Market Cap (nu)" in row_df.columns:
        row_df["Market Cap (nu)"] = row_df["Market Cap (nu)"].apply(_fmt_big)
    st.dataframe(row_df, use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# 3) Investeringsförslag
# -------------------------------------------------------------------

def _score_growth(row: pd.Series) -> float:
    """Enkel tillväxt-score av några nyckeltal."""
    ps = _safe_float(row.get("P/S-snitt"), 0.0)
    gm = _safe_float(row.get("Bruttomarginal (%)"), 0.0)
    nm = _safe_float(row.get("Nettomarginal (%)"), 0.0)
    de = _safe_float(row.get("Debt/Equity"), 0.0)
    fcf = _safe_float(row.get("FCF TTM (valuta)"), 0.0)
    # Penalize extrem P/S, debt, negativa marginaler
    score = 0.0
    if ps > 0:
        score += max(0.0, 20.0 - min(ps, 40.0)) * 1.0   # lägre P/S bättre upp till 20
    score += max(0.0, min(gm, 80.0)) * 0.1             # bruttomarginal upp till 80%
    score += max(0.0, min(nm, 40.0)) * 0.2             # nettomarginal upp till 40%
    score += 5.0 if fcf > 0 else 0.0                   # positivt fcf = plus
    if de > 0:
        score -= min(de*5.0, 15.0)
    return round(score, 2)

def _score_dividend(row: pd.Series) -> float:
    """Enkel utdelnings-score: yield, payout (prox), skuld, fcf, runway."""
    # Proxy yield = Årlig utdelning / Aktuell kurs
    div = _safe_float(row.get("Årlig utdelning"), 0.0)
    px  = _safe_float(row.get("Aktuell kurs"), 0.0)
    yield_pct = (div/px*100.0) if (div>0 and px>0) else 0.0
    de = _safe_float(row.get("Debt/Equity"), 0.0)
    fcf = _safe_float(row.get("FCF TTM (valuta)"), 0.0)
    runway = _safe_float(row.get("Runway (mån)"), 0.0)
    score = 0.0
    score += min(yield_pct, 12.0) * 2.0          # upp till 12% yield belönas
    score += 10.0 if fcf > 0 else 0.0            # positivt fcf = starkt
    score += min(runway, 60.0) * 0.2             # längre runway är bra
    if de > 0:
        score -= min(de*5.0, 20.0)               # straffa hög skuld
    return round(score, 2)

def _label_value(row: pd.Series) -> str:
    """
    Värderingsetikett baserat på P/S nu vs P/S-snitt + enkel buffert.
    """
    ps_now = _safe_float(row.get("P/S"), 0.0)
    ps_avg = _safe_float(row.get("P/S-snitt"), 0.0)
    if ps_now <= 0 or ps_avg <= 0:
        return "—"
    ratio = ps_now/ps_avg
    if ratio <= 0.7:  return "Billig"
    if ratio <= 1.1:  return "Fair"
    if ratio <= 1.4:  return "Dyr"
    if ratio <= 1.8:  return "Övervärderad"
    return "Mycket övervärderad"

def _potential_pct(row: pd.Series, target_col: str) -> float:
    px = _safe_float(row.get("Aktuell kurs"), 0.0)
    tgt = _safe_float(row.get(target_col), 0.0)
    if px <= 0 or tgt <= 0:
        return 0.0
    return (tgt - px)/px * 100.0

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    # val: tillväxt eller utdelning
    mode = st.radio("Fokus", ["Tillväxt", "Utdelning"], horizontal=True, index=0)

    # Vilken riktkurs mäts potential mot (för visning)
    target_col = st.selectbox("Riktkursfält för potential", ["Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år","Riktkurs idag"], index=0)

    # Filter: universum, storlek, sektor
    subset = st.radio("Urval", ["Alla bolag","Endast portfölj"], horizontal=True)
    size_filter = st.multiselect("Storlek (risklabel)", ["Mega","Large","Mid","Small","Micro"], default=["Mega","Large","Mid","Small","Micro"])
    sektorer = sorted(list(set([str(x) for x in df.get("Sektor", pd.Series([])).dropna().unique()])))
    sector_filter = st.multiselect("Sektorer (valfritt)", sektorer, default=sektorer[:10] if sektorer else [])

    base = df.copy()
    if subset == "Endast portfölj":
        _ensure_col(base, "Antal aktier")
        base = base[base["Antal aktier"] > 0].copy()

    # kolumner som behövs
    for c in ["Aktuell kurs","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt","Årlig utdelning",
              "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","FCF TTM (valuta)","Runway (mån)",
              "Market Cap (nu)","Sektor","Utestående aktier","Ticker","Bolagsnamn", target_col]:
        _ensure_col(base, c, 0.0 if c not in ["Ticker","Bolagsnamn","Sektor"] else "")

    # Risklabel + filter
    base["Risklabel"] = base["Market Cap (nu)"].apply(_risk_label)
    if size_filter:
        base = base[base["Risklabel"].isin(size_filter)]
    if sector_filter:
        base = base[base["Sektor"].isin(sector_filter)]

    if base.empty:
        st.info("Inga bolag matchar filtren.")
        return

    # Score
    if mode == "Tillväxt":
        base["Score"] = base.apply(_score_growth, axis=1)
    else:
        base["Score"] = base.apply(_score_dividend, axis=1)

    base["Potential (%)"] = base.apply(lambda r: _potential_pct(r, target_col), axis=1)
    # sortera: högst score först; tie-breaker: potential
    base = base.sort_values(by=["Score","Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # Visning – en i taget med bläddring
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    r = base.iloc[st.session_state.forslags_index]
    vx = _rate_for(r.get("Valuta","USD"), user_rates)
    kurs_sek = _safe_float(r.get("Aktuell kurs"),0.0) * vx

    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']}) – {mode}")
    top_cols = st.columns(3)
    with top_cols[0]:
        st.metric(label="Aktuell kurs", value=f"{_safe_float(r['Aktuell kurs'],0.0):.2f} {r.get('Valuta','')}")
        st.metric(label="Kurs i SEK", value=f"{kurs_sek:.2f} SEK")
    with top_cols[1]:
        st.metric(label="P/S (nu)", value=f"{_safe_float(r['P/S'],0.0):.2f}")
        st.metric(label="P/S-snitt (4q)", value=f"{_safe_float(r['P/S-snitt'],0.0):.2f}")
    with top_cols[2]:
        st.metric(label="Market Cap (nu)", value=_fmt_big(_safe_float(r.get("Market Cap (nu)"),0.0)))
        st.metric(label="Risklabel", value=_risk_label(_safe_float(r.get("Market Cap (nu)"),0.0)))

    mid_cols = st.columns(3)
    with mid_cols[0]:
        st.metric(label="Riktkurs (val)", value=f"{_safe_float(r.get(target_col),0.0):.2f} {r.get('Valuta','')}")
    with mid_cols[1]:
        st.metric(label="Potential", value=f"{_safe_float(r.get('Potential (%)'),0.0):.1f}%")
    with mid_cols[2]:
        st.metric(label="Score", value=f"{_safe_float(r.get('Score'),0.0):.1f}")

    st.markdown(f"**Värdering:** { _label_value(r) }")

    with st.expander("🔎 Mer nyckeltal & detaljer"):
        # visa fler värden, inklusive utestående aktier
        more = {
            "Sektor": r.get("Sektor",""),
            "Utestående aktier (milj)": _safe_float(r.get("Utestående aktier"),0.0),
            "P/S Q1": _safe_float(r.get("P/S Q1"),0.0),
            "P/S Q2": _safe_float(r.get("P/S Q2"),0.0),
            "P/S Q3": _safe_float(r.get("P/S Q3"),0.0),
            "P/S Q4": _safe_float(r.get("P/S Q4"),0.0),
            "Bruttomarginal (%)": _safe_float(r.get("Bruttomarginal (%)"),0.0),
            "Nettomarginal (%)": _safe_float(r.get("Nettomarginal (%)"),0.0),
            "Debt/Equity": _safe_float(r.get("Debt/Equity"),0.0),
            "Kassa (valuta)": _fmt_big(_safe_float(r.get("Kassa (valuta)"),0.0)),
            "FCF TTM (valuta)": _fmt_big(_safe_float(r.get("FCF TTM (valuta)"),0.0)),
            "Runway (mån)": _safe_float(r.get("Runway (mån)"),0.0),
            "Årlig utdelning": _safe_float(r.get("Årlig utdelning"),0.0),
            "GAV SEK": _safe_float(r.get("GAV SEK"),0.0),
        }
        for k, v in more.items():
            st.write(f"- **{k}:** {v}")

# -------------------------------------------------------------------
# 4) Lägg till/uppdatera bolag (delegat)
# -------------------------------------------------------------------

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str,float], save_cb=None) -> pd.DataFrame:
    return editor_view(df, user_rates, save_cb=save_cb)

# -------------------------------------------------------------------
# 5) Portfölj
# -------------------------------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📦 Min portfölj")
    if df.empty or "Antal aktier" not in df.columns:
        st.info("Du äger inga aktier.")
        return

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # SEKurser
    if "Valuta" not in port.columns:
        port["Valuta"] = "USD"
    if "Aktuell kurs" not in port.columns:
        port["Aktuell kurs"] = 0.0

    port["Växelkurs"] = port["Valuta"].apply(lambda v: _rate_for(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)
    _ensure_col(port, "Årlig utdelning", 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = [
        "Ticker","Bolagsnamn","Sektor","Antal aktier","Aktuell kurs","Valuta",
        "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)","GAV SEK"
    ]
    for c in show_cols:
        _ensure_col(port, c, 0.0 if c not in ["Ticker","Bolagsnamn","Sektor","Valuta"] else "")

    st.dataframe(
        port[show_cols].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True, hide_index=True
    )
