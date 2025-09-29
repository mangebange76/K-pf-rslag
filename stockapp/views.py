# stockapp/views.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from .config import TS_FIELDS
from .rates import hamta_valutakurs
from .utils import (
    now_stamp,
    add_oldest_ts_col,
    safe_float,
    format_large_number,
)

# ------------------------------------------------------------
# Hjälpfunktioner för vyerna
# ------------------------------------------------------------
def _ensure_ps_snitt(df: pd.DataFrame) -> pd.DataFrame:
    if "P/S-snitt" in df.columns:
        return df
    ps_cols = [c for c in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"] if c in df.columns]
    if not ps_cols:
        df["P/S-snitt"] = 0.0
        return df
    def _avg(row):
        vals = []
        for c in ps_cols:
            v = safe_float(row.get(c, 0.0))
            if v and v > 0:
                vals.append(v)
        return round(float(np.mean(vals)), 2) if vals else 0.0
    df["P/S-snitt"] = df.apply(_avg, axis=1)
    return df

def _risk_label(mcap: float) -> str:
    try:
        mc = float(mcap or 0.0)
    except Exception:
        mc = 0.0
    if mc < 300e6: return "Micro"
    if mc < 2e9:   return "Small"
    if mc < 10e9:  return "Mid"
    if mc < 200e9: return "Large"
    return "Mega"

# ------------------------------------------------------------
# ANALYSVY (enkel, bibehållen)
# ------------------------------------------------------------
def analysvy(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    if len(etiketter) == 0:
        st.info("Inga bolag i databasen ännu.")
        return

    st.session_state.analys_idx = st.number_input(
        "Visa bolag #",
        min_value=0,
        max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx, step=1
    )
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        TS_FIELDS.get("Utestående aktier", ""),
        TS_FIELDS.get("P/S", ""), TS_FIELDS.get("P/S Q1", ""), TS_FIELDS.get("P/S Q2", ""),
        TS_FIELDS.get("P/S Q3", ""), TS_FIELDS.get("P/S Q4", ""),
        TS_FIELDS.get("Omsättning idag", ""), TS_FIELDS.get("Omsättning nästa år", "")
    ]
    cols = [c for c in cols if c and c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# PORTFÖLJ (oförändrad logik, robust ifall kolumner saknas)
# ------------------------------------------------------------
def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📦 Min portfölj")
    if "Antal aktier" not in df.columns or "Aktuell kurs" not in df.columns:
        st.info("Saknar nödvändiga kolumner (Antal aktier/Aktuell kurs).")
        return
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    if "Valuta" not in port.columns:
        port["Valuta"] = "USD"
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0
    port["Andel (%)"] = np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)

    if "Årlig utdelning" in port.columns:
        port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
        tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    else:
        port["Total årlig utdelning (SEK)"] = 0.0
        tot_utd = 0.0

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = [c for c in ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"] if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# LÄGG TILL / UPPDATERA (kort, men med “Manuell prognoslista”-sektion)
# ------------------------------------------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]):
    st.header("➕ Lägg till / uppdatera bolag")

    # Enkel form: bara kärnfält (rest auto-hämtas via dina fetchers)
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            tkr = st.text_input("Ticker (Yahoo-format)", "").upper().strip()
            ut_m = st.number_input("Utestående aktier (miljoner)", value=0.0, step=1.0)
            antal = st.number_input("Antal aktier du äger", value=0.0, step=1.0)
        with c2:
            ps  = st.number_input("P/S", value=0.0, step=0.1)
            ps1 = st.number_input("P/S Q1", value=0.0, step=0.1)
            ps2 = st.number_input("P/S Q2", value=0.0, step=0.1)
            ps3 = st.number_input("P/S Q3", value=0.0, step=0.1)
            ps4 = st.number_input("P/S Q4", value=0.0, step=0.1)

        st.markdown("**Manuella prognosfält (måste uppdateras manuellt):**")
        c3, c4 = st.columns(2)
        with c3:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=0.0, step=1.0)
            valuta   = st.text_input("Valuta (t.ex. USD)", "USD").upper().strip()
        with c4:
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=0.0, step=1.0)
            bolagsnamn = st.text_input("Bolagsnamn (valfritt)", "")

        save = st.form_submit_button("💾 Spara/uppdatera")

    if save and tkr:
        # Dublettskydd
        if "Ticker" in df.columns and (df["Ticker"].str.upper().str.strip() == tkr).any():
            # Uppdatera befintlig
            ridx = df.index[df["Ticker"].str.upper().str.strip() == tkr][0]
            df.at[ridx, "Utestående aktier"] = ut_m
            df.at[ridx, "Antal aktier"] = antal
            for k, v in [("P/S", ps), ("P/S Q1", ps1), ("P/S Q2", ps2), ("P/S Q3", ps3), ("P/S Q4", ps4),
                         ("Omsättning idag", oms_idag), ("Omsättning nästa år", oms_next)]:
                if k in df.columns:
                    df.at[ridx, k] = v
            if bolagsnamn:
                df.at[ridx, "Bolagsnamn"] = bolagsnamn
            if valuta:
                df.at[ridx, "Valuta"] = valuta
            if "Senast manuellt uppdaterad" in df.columns:
                df.at[ridx, "Senast manuellt uppdaterad"] = now_stamp()
            st.success(f"Uppdaterade {tkr}.")
        else:
            # Ny rad
            row = {c: 0.0 for c in df.columns if c not in ("Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa")}
            row.update({
                "Ticker": tkr,
                "Bolagsnamn": bolagsnamn,
                "Valuta": valuta,
                "Utestående aktier": ut_m,
                "Antal aktier": antal,
                "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
                "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
                "Senast manuellt uppdaterad": now_stamp()
            })
            # Säkerställ att kolumner finns
            for need in ["Ticker","Bolagsnamn","Valuta","Utestående aktier","Antal aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år","Senast manuellt uppdaterad"]:
                row.setdefault(need, "" if need in ("Ticker","Bolagsnamn","Valuta") else 0.0)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            st.success(f"La till {tkr}.")

    # --- Manuell prognoslista (äldre TS för just omsättningsfälten) ---
    st.subheader("📝 Manuell prognoslista (äldst först)")
    need_cols = ["Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS.get(c) for c in need_cols if TS_FIELDS.get(c)]
    if ts_cols:
        work = add_oldest_ts_col(df.copy(), ts_cols)
        vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
        cols_show = ["Ticker","Bolagsnamn"] + [c for c in ts_cols if c in vis.columns] + ["_oldest_any_ts"]
        st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)
    else:
        st.info("TS-kolumner för prognos saknas i konfigurationen.")

    return df

# ------------------------------------------------------------
# KONTROLLVY (lite förenklad + prognoslistan)
# ------------------------------------------------------------
def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    all_ts_cols = [ts for ts in TS_FIELDS.values() if ts in df.columns]
    if all_ts_cols:
        work = add_oldest_ts_col(df.copy(), all_ts_cols)
        vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
        cols_show = ["Ticker","Bolagsnamn"] + [c for c in all_ts_cols if c in vis.columns] + ["_oldest_any_ts"]
        st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)
    else:
        st.info("Inga TS_-kolumner funna i data.")

    # 2) Manuell prognoslista
    st.subheader("📝 Manuell prognoslista (äldst först)")
    need_cols = ["Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS.get(c) for c in need_cols if TS_FIELDS.get(c)]
    if ts_cols:
        work = add_oldest_ts_col(df.copy(), ts_cols)
        vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
        cols_show = ["Ticker","Bolagsnamn"] + [c for c in ts_cols if c in vis.columns] + ["_oldest_any_ts"]
        st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)
    else:
        st.info("TS-kolumner för prognos saknas i konfigurationen.")

# ------------------------------------------------------------
# INVESTERINGSFÖRSLAG (NY – riktkurs, potential, poäng + filter)
# ------------------------------------------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # UI-val
    colm = st.columns([1, 1, 1, 1])
    with colm[0]:
        mode = st.radio("Strategi", ["Tillväxt", "Utdelning"], horizontal=True, key="inv_mode")
    with colm[1]:
        riktkurs_hor = st.selectbox(
            "Riktkurs-horisont",
            ["Riktkurs om 1 år", "Riktkurs idag", "Riktkurs om 2 år", "Riktkurs om 3 år"],
            index=0, key="inv_rk"
        )

    # Filtrering: sektor & risk
    sectors = []
    if "Sektor" in df.columns:
        sectors = sorted([s for s in df["Sektor"].dropna().unique() if str(s).strip()])
    risks = ["Micro", "Small", "Mid", "Large", "Mega"]

    with colm[2]:
        chosen_sector = st.selectbox("Filtrera sektor", ["(Alla)"] + sectors, index=0, key="inv_sector")
    with colm[3]:
        chosen_risk = st.selectbox("Filtrera risk (MCAP)", ["(Alla)"] + risks, index=0, key="inv_risk")

    # Förbered arbetskopia
    work = df.copy()
    work = _ensure_ps_snitt(work)

    # Market Cap (nu) och riskklass
    if "Market Cap" not in work.columns:
        work["Market Cap"] = 0.0
    for i, r in work.iterrows():
        mc = safe_float(r.get("Market Cap", 0))
        px = safe_float(r.get("Aktuell kurs", 0))
        sh_m = safe_float(r.get("Utestående aktier", 0.0)) * 1e6
        if mc <= 0 and px > 0 and sh_m > 0:
            mc = px * sh_m
            work.at[i, "Market Cap"] = mc
        work.at[i, "_risk"] = _risk_label(mc)

    # Filtrera på sektor & risk
    if chosen_sector != "(Alla)" and "Sektor" in work.columns:
        work = work[work["Sektor"] == chosen_sector]
    if chosen_risk != "(Alla)":
        work = work[work["_risk"] == chosen_risk]

    # Kräver kurs & aktier för härledning av riktkurs
    work = work[(work.get("Aktuell kurs", 0) > 0) & (work.get("Utestående aktier", 0) > 0)]
    if work.empty:
        st.info("Inga bolag matchar filtren/kriterierna just nu.")
        return

    # Beräkna riktkurs om saknas
    for i, r in work.iterrows():
        hk = riktkurs_hor
        rk = safe_float(r.get(hk, 0))
        if rk <= 0:
            ps = safe_float(r.get("P/S-snitt", 0))
            ut_m = safe_float(r.get("Utestående aktier", 0))
            if "1 år" in hk:
                rev = safe_float(r.get("Omsättning nästa år", 0.0))
            elif "2 år" in hk:
                rev = safe_float(r.get("Omsättning om 2 år", 0.0))
            elif "3 år" in hk:
                rev = safe_float(r.get("Omsättning om 3 år", 0.0))
            else:
                rev = safe_float(r.get("Omsättning idag", 0.0))
            if ps > 0 and ut_m > 0 and rev > 0:
                rk = (rev * ps) / ut_m
                work.at[i, hk] = round(rk, 2)

    # Potential & P/S nu
    work["Potential (%)"] = (work[riktkurs_hor] - work["Aktuell kurs"]) / work["Aktuell kurs"] * 100.0
    work["_ps_now"] = work.get("P/S", 0.0).astype(float)
    work["_ps_avg"] = work.get("P/S-snitt", 0.0).astype(float)

    def clip01(x): 
        try:
            return float(np.clip(x, 0.0, 1.0))
        except Exception:
            return 0.0

    # Scorer
    def score_growth(row):
        pot = float(row.get("Potential (%)", 0.0))
        cagr = float(row.get("CAGR 5 år (%)", 0.0))
        psn  = float(row.get("_ps_now", 0.0))
        psa  = float(row.get("_ps_avg", 0.0))
        gm   = float(row.get("Bruttomarginal (%)", 0.0))
        nm   = float(row.get("Netto-marginal (%)", 0.0))
        de   = float(row.get("Debt/Equity", 0.0))

        pot_s = clip01(pot/100.0)
        cagr_s= clip01(cagr/50.0)
        val_s = clip01((max(psa-psn, 0)/max(psa, 1e-6)) if psa>0 and psn>0 else 0)
        gm_s  = clip01(gm/70.0)
        nm_s  = clip01(nm/40.0)
        de_s  = clip01(max(0.0, 1.0 - de/2.0))

        return 0.35*pot_s + 0.25*cagr_s + 0.20*val_s + 0.10*gm_s + 0.08*nm_s + 0.02*de_s

    def score_dividend(row):
        yld = float(row.get("Utdelningsyield (%)", 0.0))
        fcf_cov = float(row.get("Utdelning/FCF coverage", 0.0))   # <1 bättre
        eps_cov = float(row.get("Payout ratio (%)", 0.0))         # lägre bättre
        de   = float(row.get("Debt/Equity", 0.0))
        pot  = float(row.get("Potential (%)", 0.0))

        yld_s = clip01(yld/10.0)
        cov_s = clip01(max(0.0, 1.0 - max(0.0, fcf_cov-1.0)))
        pr_s  = clip01(max(0.0, 1.0 - eps_cov/100.0))
        de_s  = clip01(max(0.0, 1.0 - de/2.0))
        pot_s = clip01(pot/50.0)

        return 0.40*yld_s + 0.25*cov_s + 0.15*pr_s + 0.10*de_s + 0.10*pot_s

    work["_score"] = work.apply(score_growth if mode=="Tillväxt" else score_dividend, axis=1)

    # Sortering
    base = work.sort_values(by=["_score", "Potential (%)"], ascending=[False, False]).reset_index(drop=True)

    # Navigering
    if "inv_idx" not in st.session_state:
        st.session_state.inv_idx = 0
    st.session_state.inv_idx = min(st.session_state.inv_idx, len(base)-1)

    nav = st.columns([1, 2, 1])
    with nav[0]:
        if st.button("⬅️ Föregående", key="inv_prev"):
            st.session_state.inv_idx = max(0, st.session_state.inv_idx-1)
    with nav[1]:
        st.write(f"Förslag {st.session_state.inv_idx+1}/{len(base)}")
    with nav[2]:
        if st.button("➡️ Nästa", key="inv_next"):
            st.session_state.inv_idx = min(len(base)-1, st.session_state.inv_idx+1)

    r = base.iloc[st.session_state.inv_idx]

    # Presentera
    st.subheader(f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})")
    mcap = float(r.get("Market Cap", 0.0))
    ps_now = float(r.get("_ps_now", 0.0))
    ps_avg = float(r.get("_ps_avg", 0.0))
    uts_m = float(r.get("Utestående aktier", 0.0))

    lines = [
        f"- **Aktuell kurs:** {float(r.get('Aktuell kurs',0.0)):.2f} {r.get('Valuta','')}",
        f"- **Riktkurs (val):** {float(r.get(riktkurs_hor,0.0)):.2f} {r.get('Valuta','')}",
        f"- **Uppsida:** {float(r.get('Potential (%)',0.0)):.1f} %",
        f"- **Market Cap (nu):** {format_large_number(mcap)}",
        f"- **P/S (nu):** {ps_now:.2f}",
        f"- **P/S-snitt (Q1–Q4):** {ps_avg:.2f}",
        f"- **Utestående aktier:** {format_large_number(uts_m*1e6, no_currency=True)} st",
        f"- **Riskklass:** {r.get('_risk','')}",
        f"- **Score ({mode}):** {float(r.get('_score',0.0)):.3f}",
    ]
    st.markdown("\n".join(lines))

    with st.expander("Nyckeltal & underlag"):
        show = {
            "Ticker": r.get("Ticker",""),
            "Bolagsnamn": r.get("Bolagsnamn",""),
            "Sektor": r.get("Sektor",""),
            "Aktuell kurs": r.get("Aktuell kurs",0.0),
            "Valuta": r.get("Valuta",""),
            "Omsättning idag": r.get("Omsättning idag",0.0),
            "Omsättning nästa år": r.get("Omsättning nästa år",0.0),
            "Omsättning om 2 år": r.get("Omsättning om 2 år",0.0),
            "Omsättning om 3 år": r.get("Omsättning om 3 år",0.0),
            "P/S": r.get("P/S",0.0),
            "P/S Q1": r.get("P/S Q1",0.0),
            "P/S Q2": r.get("P/S Q2",0.0),
            "P/S Q3": r.get("P/S Q3",0.0),
            "P/S Q4": r.get("P/S Q4",0.0),
            "P/S-snitt": r.get("P/S-snitt",0.0),
            "CAGR 5 år (%)": r.get("CAGR 5 år (%)",0.0),
            "Bruttomarginal (%)": r.get("Bruttomarginal (%)",0.0),
            "Netto-marginal (%)": r.get("Netto-marginal (%)",0.0),
            "Debt/Equity": r.get("Debt/Equity",0.0),
            "Utdelningsyield (%)": r.get("Utdelningsyield (%)",0.0),
            "Utdelning/FCF coverage": r.get("Utdelning/FCF coverage",0.0),
            "Payout ratio (%)": r.get("Payout ratio (%)",0.0),
            "Market Cap": r.get("Market Cap",0.0),
            riktkurs_hor: r.get(riktkurs_hor,0.0),
            "Potential (%)": r.get("Potential (%)",0.0),
            "Score": r.get("_score",0.0),
        }
        st.dataframe(
            (pd.DataFrame([show]))
            .T.rename(columns={0: "Värde"})
            .reset_index()
            .rename(columns={"index": "Nyckeltal"}),
            use_container_width=True, hide_index=True
        )
