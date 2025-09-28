# -*- coding: utf-8 -*-
"""
Investeringsförslag-vy
Robust mot saknade kolumner. Visar potential, köpantal, MCap/P-S-detaljer
och valfri sektorfiltrering.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st


# ---------- Hjälpare ----------

def _fx(ccy: str, rates: Dict[str, float]) -> float:
    """Hämta SEK-kurs för valuta (default 1.0)."""
    try:
        if not ccy:
            return 1.0
        return float(rates.get(str(ccy).upper(), 1.0))
    except Exception:
        return 1.0


def _fmt_money_short(x: float, ccy: str) -> str:
    """4.34 tn / 2.75 mdr / 680 mn osv."""
    try:
        n = float(x)
    except Exception:
        return f"- {ccy}"
    units = [
        (1_000_000_000_000, "tn"),
        (1_000_000_000, "mdr"),
        (1_000_000, "mn"),
        (1_000, "t"),
    ]
    for base, tag in units:
        if abs(n) >= base:
            return f"{n/base:,.2f} {tag} {ccy}".replace(",", " ")
    return f"{n:,.2f} {ccy}".replace(",", " ")


def _market_cap_now(row: pd.Series) -> float:
    """Beräkna MCap = pris * utestående aktier (miljoner→styck). Fallback: kolumn 'MCap (nu)' om finns."""
    try:
        px = float(row.get("Aktuell kurs", 0.0) or 0.0)
        sh_m = float(row.get("Utestående aktier", 0.0) or 0.0)  # i miljoner
        if px > 0 and sh_m > 0:
            return px * (sh_m * 1e6)
    except Exception:
        pass
    try:
        m = float(row.get("MCap (nu)", 0.0) or 0.0)
        if m > 0:
            return m
    except Exception:
        pass
    return 0.0


def _risklabel(mcap_usd: float) -> str:
    if mcap_usd >= 200e9: return "Megacap"
    if mcap_usd >= 10e9:  return "Largecap"
    if mcap_usd >= 2e9:   return "Midcap"
    if mcap_usd >= 300e6: return "Smallcap"
    return "Microcap"


# ---------- Själva vyn ----------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    # Inputs
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    rikt_alternativ = [c for c in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"] if c in df.columns]
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        rikt_alternativ,
        index=1 if "Riktkurs om 1 år" in rikt_alternativ else (0 if rikt_alternativ else 0)
    ) if rikt_alternativ else None

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    sort_lage = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    # Sektorfilter (om kolumn finns)
    work = df.copy()
    if "Sektor" in work.columns:
        sektorer = sorted([s for s in work["Sektor"].dropna().astype(str).unique() if s.strip()])
        valda = st.multiselect("Filtrera på sektor (valfritt)", sektorer, default=[])
        if valda:
            work = work[work["Sektor"].astype(str).isin(valda)].copy()

    # Subset
    if subset == "Endast portfölj":
        work = work[(work.get("Antal aktier", 0) > 0)].copy()

    # Säkerställ nödvändiga kolumner
    for c in ["Aktuell kurs","Valuta","Utestående aktier"]:
        if c not in work.columns:
            work[c] = 0.0
    if riktkurs_val and riktkurs_val not in work.columns:
        work[riktkurs_val] = 0.0

    # Filtrera
    if riktkurs_val:
        work = work[(work["Aktuell kurs"] > 0) & (work[riktkurs_val] > 0)].copy()
    else:
        work = work[(work["Aktuell kurs"] > 0)].copy()

    # Tomt → fallback
    if work.empty:
        if "P/S-snitt" in df.columns and not df["P/S-snitt"].dropna().empty:
            st.info("Vyn 'Investeringsförslag' saknas. Fallback visar top-20 på P/S-snitt om tillgängligt.")
            fb = df[["Bolagsnamn","Ticker","P/S-snitt"]].copy()
            fb = fb.replace([np.inf, -np.inf], np.nan).dropna(subset=["P/S-snitt"])
            st.dataframe(fb.sort_values(by="P/S-snitt").head(20), use_container_width=True)
        else:
            st.info("Inget att visa ännu.")
        return

    # Potential & sortering
    if riktkurs_val:
        work["Potential (%)"] = (work[riktkurs_val] - work["Aktuell kurs"]) / work["Aktuell kurs"] * 100.0
        work["Diff till mål (%)"] = (work["Aktuell kurs"] - work[riktkurs_val]) / work[riktkurs_val] * 100.0
        if sort_lage == "Störst potential":
            work = work.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
        else:
            work["absdiff"] = work["Diff till mål (%)"].abs()
            work = work.sort_values(by="absdiff", ascending=True).reset_index(drop=True)
    else:
        if "P/S-snitt" in work.columns:
            work = work.replace([np.inf, -np.inf], np.nan)
            work = work.dropna(subset=["P/S-snitt"]).sort_values(by="P/S-snitt", ascending=True).reset_index(drop=True)
        else:
            work = work.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)

    # Bläddring
    key_idx = "forslags_idx"
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0
    st.session_state[key_idx] = min(st.session_state[key_idx], max(0, len(work)-1))

    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("⬅️ Föregående förslag"):
            st.session_state[key_idx] = max(0, st.session_state[key_idx]-1)
    with c2:
        st.write(f"Förslag {st.session_state[key_idx]+1}/{len(work)}")
    with c3:
        if st.button("➡️ Nästa förslag"):
            st.session_state[key_idx] = min(len(work)-1, st.session_state[key_idx]+1)

    row = work.iloc[st.session_state[key_idx]]

    # Köpberäkning (SEK)
    vx = _fx(row.get("Valuta","SEK"), user_rates)
    kurs_sek = float(row.get("Aktuell kurs", 0.0)) * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9)) if kurs_sek > 0 else 0
    investering = antal_köp * kurs_sek

    # Portföljandel
    port = df[df.get("Antal aktier", 0) > 0].copy()
    port_värde = 0.0
    nuv_innehav = 0.0
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: _fx(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
        match = port[port["Ticker"].astype(str).str.upper() == str(row.get("Ticker","")).upper()]
        if not match.empty:
            nuv_innehav = float(match["Värde (SEK)"].sum())

    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round(((nuv_innehav + investering) / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Presentera
    st.subheader(f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')})")

    lines = [f"- **Aktuell kurs:** {round(float(row.get('Aktuell kurs',0.0)),2)} {row.get('Valuta','')}"]
    if riktkurs_val:
        for rk in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
            if rk in work.columns:
                tag = " **⬅ vald**" if rk == riktkurs_val else ""
                lines.append(f"- **{rk}:** {round(float(row.get(rk,0.0)),2)} {row.get('Valuta','')}{tag}")
        lines.append(f"- **Uppsida (valda riktkursen):** {round(float(row.get('Potential (%)',0.0)),2)} %")
    lines.extend([
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
    ])
    st.markdown("\n".join(lines))

    # Detaljer
    with st.expander("📊 Detaljer & historik (MCap, P/S, nyckeltal)"):
        mcap = _market_cap_now(row)
        st.markdown(f"- **Market Cap (nu):** {_fmt_money_short(mcap, row.get('Valuta',''))}")

        ps_now = row.get("P/S", np.nan)
        if not pd.isna(ps_now) and float(ps_now) > 0:
            st.markdown(f"- **P/S (TTM – modell):** {float(ps_now):.3f}")
        if "P/S (Yahoo)" in row and pd.notna(row["P/S (Yahoo)"]) and float(row["P/S (Yahoo)"]) > 0:
            st.markdown(f"- **P/S (Yahoo):** {float(row['P/S (Yahoo)']):.3f}")
        if "P/S-snitt" in row and pd.notna(row["P/S-snitt"]) and float(row["P/S-snitt"]) > 0:
            st.markdown(f"- **P/S-snitt (Q1..Q4):** {float(row['P/S-snitt']):.2f}")

        # visa senaste 4 TTM-fönster om data finns
        rows_hist = []
        for q in ["Q1","Q2","Q3","Q4"]:
            psq = row.get(f"P/S {q}", np.nan)
            mcapq = row.get(f"MCap {q}", np.nan)
            dateq = row.get(f"TTM-slut {q}", "")
            if (not pd.isna(psq)) or (not pd.isna(mcapq)) or (dateq not in ("", None)):
                rows_hist.append({"Period": q, "P/S": psq, "MCap": mcapq, "Datum (TTM-slut)": dateq})
        if rows_hist:
            vis = pd.DataFrame(rows_hist)
            ccy = row.get("Valuta","")
            def _fmt(x): 
                try:
                    xv = float(x)
                    return _fmt_money_short(xv, ccy) if xv > 0 else ""
                except Exception:
                    return ""
            if "MCap" in vis.columns:
                vis["MCap"] = vis["MCap"].apply(_fmt)
            st.dataframe(vis, use_container_width=True, hide_index=True)

        if mcap > 0:
            st.markdown(f"- **Klass:** {_risklabel(mcap)}")

        for label, col in [("Debt/Equity","Debt/Equity"),
                           ("Bruttomarginal","Bruttomarginal"),
                           ("Nettomarginal","Nettomarginal"),
                           ("Kassa","Kassa")]:
            if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                val = row[col]
                if label in ("Bruttomarginal","Nettomarginal"):
                    try:
                        st.markdown(f"- **{label}:** {float(val):.2f} %")
                    except Exception:
                        st.markdown(f"- **{label}:** {val}")
                elif label == "Kassa":
                    try:
                        st.markdown(f"- **{label}:** {_fmt_money_short(float(val), row.get('Valuta',''))}")
                    except Exception:
                        st.markdown(f"- **{label}:** {val}")
                else:
                    st.markdown(f"- **{label}:** {val}")
