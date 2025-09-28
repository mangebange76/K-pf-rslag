# stockapp/views.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from .utils import recompute_derived, add_oldest_ts_col

# ---------------------------
# Små hjälpfunktioner
# ---------------------------

def _rate_for(ccy: str, user_rates: Dict[str, float]) -> float:
    if not ccy:
        return 1.0
    return float(user_rates.get(str(ccy).upper(), 1.0))

def _fmt_money_short(x: float, ccy: str = "") -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    suffix = ""
    if abs(v) >= 1_000_000_000_000:
        v = v / 1_000_000_000_000.0
        suffix = " tn"
    elif abs(v) >= 1_000_000_000:
        v = v / 1_000_000_000.0
        suffix = " md"
    elif abs(v) >= 1_000_000:
        v = v / 1_000_000.0
        suffix = " m"
    return f"{v:,.2f}{suffix} {ccy}".replace(",", " ")

def _fmt_num(x: float, decimals: int = 2) -> str:
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "-"

def _risk_label_from_mcap(mcap: float, ccy: str = "") -> str:
    # Buckets i basvalutan (bolagets prisvaluta)
    # Micro < 0.3B, Small < 2B, Mid < 10B, Large < 200B, Mega >= 200B
    b = float(mcap or 0.0)
    if b <= 0:
        return "Okänd"
    if b < 300_000_000: return "Microcap"
    if b < 2_000_000_000: return "Smallcap"
    if b < 10_000_000_000: return "Midcap"
    if b < 200_000_000_000: return "Largecap"
    return "Megacap"

def _current_market_cap(row: pd.Series) -> float:
    price = float(row.get("Aktuell kurs", 0.0) or 0.0)
    shares_m = float(row.get("Utestående aktier", 0.0) or 0.0)  # i miljoner
    if price > 0 and shares_m > 0:
        return price * shares_m * 1_000_000.0
    return float(row.get("Market Cap (nu)", 0.0) or 0.0)

def _ps_snitt(row: pd.Series) -> float:
    # Om P/S-snitt finns i df använd det, annars räkna snittet av Q1..Q4 > 0
    v = float(row.get("P/S-snitt", 0.0) or 0.0)
    if v > 0:
        return v
    arr = []
    for k in ("P/S Q1","P/S Q2","P/S Q3","P/S Q4"):
        x = float(row.get(k, 0.0) or 0.0)
        if x > 0:
            arr.append(x)
    return float(np.mean(arr)) if arr else 0.0

def _target_price(row: pd.Series, field: str) -> float:
    return float(row.get(field, 0.0) or 0.0)

def _upside_pct(price: float, target: float) -> float:
    if price <= 0 or target <= 0:
        return 0.0
    return (target - price) / price * 100.0

def _dividend_yield(row: pd.Series) -> float:
    price = float(row.get("Aktuell kurs", 0.0) or 0.0)
    div_ps = float(row.get("Årlig utdelning", 0.0) or 0.0)  # per share
    if price > 0 and div_ps > 0:
        return div_ps / price * 100.0
    return 0.0

def _fcf_coverage(row: pd.Series) -> float:
    """
    Approx: FCF_TTM / (Dividend per share * shares)
    Enhets-agnostiskt om div per share och FCF i samma valuta (normalt).
    """
    fcf = float(row.get("FCF TTM (valuta)", 0.0) or 0.0)
    dps = float(row.get("Årlig utdelning", 0.0) or 0.0)
    shares_m = float(row.get("Utestående aktier", 0.0) or 0.0)
    if fcf <= 0 or dps <= 0 or shares_m <= 0:
        return 0.0
    total_div = dps * shares_m * 1_000_000.0
    if total_div <= 0:
        return 0.0
    return fcf / total_div

def _label_from_score(score: float, mode: str) -> str:
    if mode == "Utdelning":
        if score >= 80: return "KÖP"
        if score >= 65: return "Bra"
        if score >= 50: return "Fair/Behåll"
        if score >= 40: return "Trimma"
        return "Sälj"
    # Tillväxt
    if score >= 80: return "KÖP"
    if score >= 65: return "Bra"
    if score >= 50: return "Fair/Behåll"
    if score >= 35: return "Trimma"
    return "Sälj"

# --- Scoringfunktioner -------------------------------------------------------

def _score_growth(row: pd.Series, target_field: str) -> Tuple[float, Dict[str,float]]:
    price = float(row.get("Aktuell kurs", 0.0) or 0.0)
    tgt = float(row.get(target_field, 0.0) or 0.0)
    up = _upside_pct(price, tgt)
    up_score = max(0.0, min(100.0, up))  # direkt 1:1 capped

    cagr = max(0.0, float(row.get("CAGR 5 år (%)", 0.0) or 0.0))
    cagr_score = min(100.0, cagr * 2.0)  # 50% CAGR => 100p

    psn = _ps_snitt(row)
    if psn <= 0:
        ps_score = 50.0
    elif psn <= 5: ps_score = 90.0
    elif psn <= 10: ps_score = 70.0
    elif psn <= 20: ps_score = 50.0
    else: ps_score = 30.0

    de = float(row.get("Debt/Equity", 0.0) or 0.0)
    if de <= 0.0: de_score = 80.0
    elif de <= 0.5: de_score = 90.0
    elif de <= 1.0: de_score = 75.0
    elif de <= 2.0: de_score = 60.0
    else: de_score = 40.0

    gm = float(row.get("Bruttomarginal (%)", 0.0) or 0.0)
    gm_score = max(0.0, min(100.0, gm))  # 50% → 50p, 60% → 60p

    nm = float(row.get("Nettomarginal (%)", 0.0) or 0.0)
    nm_score = max(0.0, min(100.0, nm + 50.0))  # -50% →0p, 0% →50p, 50% →100p

    runway = float(row.get("Runway (mån)", 0.0) or 0.0)
    if runway >= 24: rw_score = 100.0
    elif runway >= 12: rw_score = 70.0
    elif runway >= 6: rw_score = 40.0
    else: rw_score = 20.0

    # vikter (summa=1.0)
    weights = {
        "Uppsida": 0.45,
        "CAGR": 0.15,
        "P/S": 0.10,
        "D/E": 0.10,
        "Br.marg": 0.08,
        "Net.marg": 0.07,
        "Runway": 0.05,
    }
    parts = {
        "Uppsida": up_score,
        "CAGR": cagr_score,
        "P/S": ps_score,
        "D/E": de_score,
        "Br.marg": gm_score,
        "Net.marg": nm_score,
        "Runway": rw_score,
    }
    total = sum(parts[k] * weights[k] for k in weights)
    return float(total), parts


def _score_dividend(row: pd.Series, target_field: str) -> Tuple[float, Dict[str,float]]:
    y = _dividend_yield(row)
    if y <= 1: y_score = 20.0
    elif y <= 2: y_score = 45.0
    elif y <= 4: y_score = 70.0
    elif y <= 8: y_score = 90.0
    elif y <= 12: y_score = 70.0
    else: y_score = 40.0

    cov = _fcf_coverage(row)
    if cov >= 2.0: cov_score = 95.0
    elif cov >= 1.5: cov_score = 85.0
    elif cov >= 1.0: cov_score = 70.0
    elif cov >= 0.6: cov_score = 50.0
    else: cov_score = 20.0

    de = float(row.get("Debt/Equity", 0.0) or 0.0)
    if de <= 0.3: de_score = 95.0
    elif de <= 0.8: de_score = 85.0
    elif de <= 1.2: de_score = 70.0
    elif de <= 2.0: de_score = 55.0
    else: de_score = 35.0

    # liten vikt på uppsida mot vald riktkurs
    price = float(row.get("Aktuell kurs", 0.0) or 0.0)
    tgt = float(row.get(target_field, 0.0) or 0.0)
    up = _upside_pct(price, tgt)
    up_score = max(0.0, min(100.0, up))

    # vikter
    weights = {"Yield": 0.35, "FCF-coverage": 0.35, "D/E": 0.15, "Uppsida": 0.15}
    parts = {"Yield": y_score, "FCF-coverage": cov_score, "D/E": de_score, "Uppsida": up_score}
    total = sum(parts[k] * weights[k] for k in weights)
    return float(total), parts


def _is_dividend_stock(row: pd.Series) -> bool:
    y = _dividend_yield(row)
    if y >= 1.5:
        return True
    # eller om användaren äger aktier och bolaget har någon utdelning alls
    if float(row.get("Antal aktier", 0.0) or 0.0) > 0 and float(row.get("Årlig utdelning", 0.0) or 0.0) > 0:
        return True
    return False


# ---------------------------
# Kontroll-vy
# ---------------------------
def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"]).head(20)
    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns: cols_show.append(k)
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Senaste batchlogg
    st.subheader("📒 Senaste batchlogg")
    log = st.session_state.get("_batch_log", {})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Ändringar** (ticker → fält)")
        st.json(log.get("changed", {}))
    with c2:
        st.markdown("**Missar** (ticker → fält/orsak)")
        st.json(log.get("misses", {}))
    with c3:
        st.markdown("**Fel**")
        st.json(log.get("errors", {}))

# ---------------------------
# Analys-vy
# ---------------------------
def analysvy(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📈 Analys")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Sortera och bygg etiketter
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    # Navigeringstillstånd
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(st.session_state.analys_idx, max(0, len(etiketter)-1))

    # Välj bolag
    col_top_a, col_top_b = st.columns([2,3])
    with col_top_a:
        st.session_state.analys_idx = st.number_input(
            "Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
            value=st.session_state.analys_idx, step=1
        )
    with col_top_b:
        st.selectbox(
            "Eller välj i lista",
            etiketter,
            index=st.session_state.analys_idx if etiketter else 0,
            key="analys_select",
        )
        # Håll index i synk med selectbox
        if etiketter:
            try:
                st.session_state.analys_idx = etiketter.index(st.session_state.analys_select)
            except Exception:
                pass

    # Bläddringsknappar
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("⬅️ Föregående", key="analys_prev_btn"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_nav2:
        if st.button("➡️ Nästa", key="analys_next_btn"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx + 1)

    if not etiketter:
        return

    r = vis_df.iloc[st.session_state.analys_idx]
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")
    st.subheader(f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})")

    # Datum-/källa-etikett
    man_dt = str(r.get("Senast manuellt uppdaterad","")).strip()
    auto_dt = str(r.get("Senast auto-uppdaterad","")).strip()
    src     = str(r.get("Senast uppdaterad källa","")).strip()
    badges = []
    if man_dt:
        badges.append(f"🖐️ Manuell: {man_dt}")
    if auto_dt:
        badges.append(f"⚙️ Auto: {auto_dt}")
    if src:
        badges.append(f"🔗 Källa: {src}")
    if badges:
        st.info(" • ".join(badges))

    # Nyckeltal & beräknat
    price = float(r.get("Aktuell kurs", 0.0) or 0.0)
    ccy   = str(r.get("Valuta","") or "")
    shares_m = float(r.get("Utestående aktier", 0.0) or 0.0)
    mcap_now = _current_market_cap(r)
    psn = _ps_snitt(r)
    yld = _dividend_yield(r)
    risk = _risk_label_from_mcap(mcap_now, ccy)

    # P/S historik
    ps_hist = {k: float(r.get(k, 0.0) or 0.0) for k in ("P/S Q1","P/S Q2","P/S Q3","P/S Q4")}

    # Tidsstämplar per spårat fält
    ts_cols = {
        "Utestående aktier": r.get("TS_Utestående aktier",""),
        "P/S": r.get("TS_P/S",""),
        "P/S Q1": r.get("TS_P/S Q1",""),
        "P/S Q2": r.get("TS_P/S Q2",""),
        "P/S Q3": r.get("TS_P/S Q3",""),
        "P/S Q4": r.get("TS_P/S Q4",""),
        "Omsättning idag": r.get("TS_Omsättning idag",""),
        "Omsättning nästa år": r.get("TS_Omsättning nästa år",""),
    }

    # Visa tabell
    rows = []
    rows.append(("Aktuell kurs", f"{_fmt_num(price,2)} {ccy}"))
    rows.append(("Valuta", ccy or "-"))
    rows.append(("Utestående aktier (milj.)", _fmt_num(shares_m, 2)))
    rows.append(("Market cap (nu)", _fmt_money_short(mcap_now, ccy)))
    rows.append(("P/S-snitt", _fmt_num(psn, 2)))
    rows.append(("P/S Q1", _fmt_num(ps_hist["P/S Q1"],2)))
    rows.append(("P/S Q2", _fmt_num(ps_hist["P/S Q2"],2)))
    rows.append(("P/S Q3", _fmt_num(ps_hist["P/S Q3"],2)))
    rows.append(("P/S Q4", _fmt_num(ps_hist["P/S Q4"],2)))
    rows.append(("Omsättning idag (M)", _fmt_num(float(r.get("Omsättning idag",0.0) or 0.0),2)))
    rows.append(("Omsättning nästa år (M)", _fmt_num(float(r.get("Omsättning nästa år",0.0) or 0.0),2)))
    rows.append(("Riktkurs idag", f"{_fmt_num(float(r.get('Riktkurs idag',0.0) or 0.0),2)} {ccy}"))
    rows.append(("Riktkurs om 1 år", f"{_fmt_num(float(r.get('Riktkurs om 1 år',0.0) or 0.0),2)} {ccy}"))
    rows.append(("Riktkurs om 2 år", f"{_fmt_num(float(r.get('Riktkurs om 2 år',0.0) or 0.0),2)} {ccy}"))
    rows.append(("Riktkurs om 3 år", f"{_fmt_num(float(r.get('Riktkurs om 3 år',0.0) or 0.0),2)} {ccy}"))
    rows.append(("Årlig utdelning (per aktie)", f"{_fmt_num(float(r.get('Årlig utdelning',0.0) or 0.0),2)} {ccy}"))
    rows.append(("Direktavkastning (%)", _fmt_num(yld,2)))
    rows.append(("CAGR 5 år (%)", _fmt_num(float(r.get("CAGR 5 år (%)",0.0) or 0.0),2)))
    rows.append(("Risklabel", risk))

    # ev fundamenta om finns
    for k, lab in [
        ("Debt/Equity","Debt/Equity"),
        ("Bruttomarginal (%)","Bruttomarginal (%)"),
        ("Nettomarginal (%)","Nettomarginal (%)"),
        ("FCF TTM (valuta)","FCF TTM"),
        ("Kassa (valuta)","Kassa"),
        ("Runway (mån)","Runway (mån)"),
        ("Sektor","Sektor"),
        ("Industri","Industri"),
    ]:
        if k in r:
            v = r.get(k)
            if "valuta" in k.lower():
                rows.append((lab, f"{_fmt_money_short(float(v or 0.0), ccy)}"))
            else:
                rows.append((lab, _fmt_num(float(v or 0.0),2) if isinstance(v,(int,float,np.floating)) else str(v)))

    df_show = pd.DataFrame(rows, columns=["Nyckel","Värde"])
    st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Tidsstämplar per fält
    with st.expander("Tidsstämplar per fält"):
        ts_rows = [(k, str(v or "")) for k, v in ts_cols.items()]
        st.dataframe(pd.DataFrame(ts_rows, columns=["Fält","Senast uppdaterad"]), use_container_width=True, hide_index=True)

# ---------------------------
# Investeringsförslag
# ---------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")

    if df is None or df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Grundfiltrering: kräver pris och något mål
    candidates = df.copy()
    candidates = candidates[(candidates["Aktuell kurs"] > 0)]
    if candidates.empty:
        st.warning("Inga kandidater med giltig aktuell kurs.")
        return

    # Val: läge, riktkurs, filter
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        mode = st.selectbox("Läge", ["Tillväxt","Utdelning"], index=0)
    with c2:
        target_field = st.selectbox("Riktkursfält", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)
    with c3:
        subset = st.radio("Urval", ["Alla bolag","Endast portfölj"], horizontal=True)

    if subset == "Endast portfölj":
        candidates = candidates[candidates["Antal aktier"] > 0].copy()

    # Riskfilter
    # Beräkna risklabel per rad
    candidates["_mcap_now"] = candidates.apply(_current_market_cap, axis=1)
    candidates["_risk"] = candidates["_mcap_now"].apply(_risk_label_from_mcap)
    risk_opts = ["Microcap","Smallcap","Midcap","Largecap","Megacap","Okänd"]
    risk_sel = st.multiselect("Riskfilter (marketcap)", risk_opts, default=risk_opts)

    # Sektorfiler
    sectors = sorted(list(set([str(x) for x in candidates.get("Sektor", pd.Series()).fillna("").tolist()])))
    if sectors and len(sectors) > 1:
        sec_sel = st.multiselect("Filtrera sektor (valfritt)", sectors, default=sectors)
    else:
        sec_sel = sectors

    # Applicera filter
    if risk_sel:
        candidates = candidates[candidates["_risk"].isin(risk_sel)]
    if sec_sel:
        candidates = candidates[candidates["Sektor"].astype(str).isin(sec_sel)]

    if candidates.empty:
        st.warning("Inga kandidater kvar efter filtren.")
        return

    # Scoring per rad
    scores = []
    for idx, row in candidates.iterrows():
        if mode == "Utdelning":
            total, parts = _score_dividend(row, target_field)
        else:
            total, parts = _score_growth(row, target_field)
        scores.append((idx, float(total), parts))
    if not scores:
        st.info("Inga poäng kunde beräknas.")
        return

    # Lägg in i DF
    sc_map = {i: s for (i, s, _) in scores}
    candidates["_Score"] = candidates.index.map(sc_map)
    # etikett
    candidates["_Label"] = candidates["_Score"].apply(lambda s: _label_from_score(float(s or 0.0), mode))

    # Sortera
    candidates = candidates.sort_values(by="_Score", ascending=False).reset_index(drop=False).rename(columns={"index":"_orig_index"})

    # Topplista tabell
    with st.expander("Visa topplista (topp 20)", expanded=False):
        show = candidates.head(20).copy()
        show["_Uppsida (%)"] = (show[target_field] - show["Aktuell kurs"]) / show["Aktuell kurs"] * 100.0
        show_cols = [
            "Ticker","Bolagsnamn","Sektor","_risk","Aktuell kurs",target_field,
            "_Uppsida (%)","P/S-snitt","CAGR 5 år (%)","Årlig utdelning","_Score","_Label"
        ]
        show_cols = [c for c in show_cols if c in show.columns]
        st.dataframe(show[show_cols], use_container_width=True, hide_index=True)

    # Robust bläddrare
    if "_inv_idx" not in st.session_state:
        st.session_state._inv_idx = 0
    max_idx = len(candidates) - 1
    st.session_state._inv_idx = min(st.session_state._inv_idx, max_idx)

    nav1, mid, nav2 = st.columns([1,2,1])
    with nav1:
        if st.button("⬅️ Föregående förslag", key="inv_prev"):
            st.session_state._inv_idx = max(0, st.session_state._inv_idx - 1)
    with mid:
        st.write(f"Förslag {st.session_state._inv_idx+1}/{len(candidates)}")
    with nav2:
        if st.button("➡️ Nästa förslag", key="inv_next"):
            st.session_state._inv_idx = min(max_idx, st.session_state._inv_idx + 1)

    row = candidates.iloc[st.session_state._inv_idx]
    # Score-delar för valt bolag
    if mode == "Utdelning":
        total, parts = _score_dividend(row, target_field)
    else:
        total, parts = _score_growth(row, target_field)

    # Presentera
    st.subheader(f"{row.get('Bolagsnamn','')} ({row.get('Ticker','')}) — {mode}")
    ccy = str(row.get("Valuta","") or "")
    price = float(row.get("Aktuell kurs", 0.0) or 0.0)
    tgt = float(row.get(target_field, 0.0) or 0.0)
    up = _upside_pct(price, tgt)
    psn = _ps_snitt(row)
    yld = _dividend_yield(row)
    mcap_now = _current_market_cap(row)

    # P/S historik
    ps_hist = [float(row.get(k, 0.0) or 0.0) for k in ("P/S Q1","P/S Q2","P/S Q3","P/S Q4")]
    # Market cap historik (om finns)
    mcap_hist_cols = [c for c in row.index if c.upper().startswith("MCAP Q")]
    mcap_hist = [float(row.get(c, 0.0) or 0.0) for c in sorted(mcap_hist_cols)][:4]

    lines = []
    lines.append(f"- **Aktuell kurs:** {_fmt_num(price,2)} {ccy}")
    lines.append(f"- **{target_field}:** {_fmt_num(tgt,2)} {ccy}")
    lines.append(f"- **Uppsida:** {_fmt_num(up,2)} %")
    lines.append(f"- **P/S-snitt:** {_fmt_num(psn,2)}")
    lines.append(f"- **Direktavkastning:** {_fmt_num(yld,2)} %")
    lines.append(f"- **Market cap (nu):** {_fmt_money_short(mcap_now, ccy)}")
    if ps_hist:
        lines.append(f"- **P/S Q1–Q4:** " + ", ".join(_fmt_num(x,2) for x in ps_hist if x>0))
    if any(mcap_hist):
        lines.append(f"- **MCAP Q1–Q4:** " + ", ".join(_fmt_money_short(x, ccy) for x in mcap_hist if x>0))
    lines.append(f"- **Risklabel:** {row.get('_risk','Okänd')}")
    lines.append(f"- **Score:** {_fmt_num(total,1)} → **{_label_from_score(total, mode)}**")
    st.markdown("\n".join(lines))

    with st.expander("Score-detaljer"):
        det_rows = [(k, _fmt_num(v,1)) for k, v in parts.items()]
        st.dataframe(pd.DataFrame(det_rows, columns=["Komponent","Poäng"]), use_container_width=True, hide_index=True)

    # Fundamenta – valfritt
    with st.expander("Fundamenta (om tillgängligt)"):
        fund = []
        for k, lab in [
            ("Debt/Equity","Debt/Equity"),
            ("Bruttomarginal (%)","Bruttomarginal (%)"),
            ("Nettomarginal (%)","Nettomarginal (%)"),
            ("FCF TTM (valuta)","FCF TTM"),
            ("Kassa (valuta)","Kassa"),
            ("Runway (mån)","Runway (mån)"),
            ("Sektor","Sektor"),
            ("Industri","Industri"),
            ("Utestående aktier","Utestående aktier (milj.)"),
        ]:
            if k in row.index:
                v = row.get(k)
                if "valuta" in k.lower():
                    fund.append((lab, _fmt_money_short(float(v or 0.0), ccy)))
                elif k == "Utestående aktier":
                    fund.append((lab, _fmt_num(float(v or 0.0), 2)))
                else:
                    fund.append((lab, _fmt_num(float(v or 0.0),2) if isinstance(v,(int,float,np.floating)) else str(v)))
        st.dataframe(pd.DataFrame(fund, columns=["Nyckel","Värde"]), use_container_width=True, hide_index=True)

# ---------------------------
# Portfölj-vy
# ---------------------------
def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📦 Min portfölj")

    if df is None or df.empty or "Antal aktier" not in df.columns:
        st.info("Du äger inga aktier (finns inga rader med 'Antal aktier' > 0).")
        return

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Säkerställ kolumner som kan saknas
    for col in ["Valuta","Aktuell kurs","Årlig utdelning","GAV (SEK)"]:
        if col not in port.columns:
            port[col] = 0.0 if col != "Valuta" else "SEK"

    # Beräkningar
    port["Växelkurs"] = port["Valuta"].apply(lambda v: _rate_for(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(
        total_värde > 0,
        np.round(port["Värde (SEK)"] / total_värde * 100.0, 2),
        0.0
    )
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]

    # Aggregerat
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    st.markdown(f"**Totalt portföljvärde:** {_fmt_money_short(total_värde, 'SEK')}")
    st.markdown(f"**Total förväntad utdelning/år:** {_fmt_money_short(tot_utd, 'SEK')}")
    st.markdown(f"**Ungefärlig månadsutdelning:** {_fmt_money_short(tot_utd/12.0, 'SEK')}")

    # Visa tabell
    show_cols = [
        "Ticker","Bolagsnamn","Antal aktier",
        "Aktuell kurs","Valuta","Växelkurs",
        "Värde (SEK)","Andel (%)",
        "Årlig utdelning","Total årlig utdelning (SEK)",
    ]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(
        port[show_cols].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True, hide_index=True
    )


# ---------------------------
# Lägg till/Uppdatera-vy
# ---------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: Dict[str, float]) -> pd.DataFrame:
    """
    Enkel redigeringsvy:
    - Välj befintligt bolag (eller nytt)
    - Uppdatera nyckelfält (inkl. manuell prognos: 'Omsättning idag' & 'Omsättning nästa år')
    - Sätt tidsstämplar för ändrade fält där TS_-kolumner finns
    - Visa "Manuell prognoslista" sorterad på äldsta TS för just omsättningsfälten

    Returnerar en *modifierad* DataFrame (uppdateringar sker i minnet; persistens hanteras i app.py).
    """
    st.header("➕ Lägg till / uppdatera bolag")

    if df is None:
        df = pd.DataFrame()

    # Säkerställ några kolumner som ofta efterfrågas här
    ensure_cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
        "Utestående aktier","Antal aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år",
        "Årlig utdelning","GAV (SEK)",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
        "TS_Omsättning idag","TS_Omsättning nästa år",
    ]
    for c in ensure_cols:
        if c not in df.columns:
            # numeriska default
            if any(k in c.lower() for k in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","gav"]):
                df[c] = 0.0
            else:
                df[c] = ""

    # Sorteringsval
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], index=0)
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # Välj bolag
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(val_lista)-1))

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=st.session_state.edit_index)

    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef_mask = df["Ticker"] == namn_map[valt_label]
        bef = df.loc[bef_mask].iloc[0] if bef_mask.any() else pd.Series({}, dtype=object)
    else:
        bef = pd.Series({}, dtype=object)

    # Form
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=str(bef.get("Ticker","")) if not bef.empty else "").upper().strip()
            bol_namn = st.text_input("Bolagsnamn", value=str(bef.get("Bolagsnamn","")) if not bef.empty else "")
            valuta = st.text_input("Valuta (t.ex. USD, SEK)", value=str(bef.get("Valuta","USD") or "USD"))

            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0) or 0.0))
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0) or 0.0))
            gav_sek = st.number_input("GAV (SEK) per aktie", value=float(bef.get("GAV (SEK)",0.0) or 0.0))

            ps  = st.number_input("P/S", value=float(bef.get("P/S",0.0) or 0.0))
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0) or 0.0))
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0) or 0.0))
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0) or 0.0))
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0) or 0.0))
        with c2:
            akt_kurs = st.number_input("Aktuell kurs", value=float(bef.get("Aktuell kurs",0.0) or 0.0))
            arlig_utd = st.number_input("Årlig utdelning (per aktie)", value=float(bef.get("Årlig utdelning",0.0) or 0.0))

            oms_idag  = st.number_input("Omsättning idag (miljoner) — MANUELL",  value=float(bef.get("Omsättning idag",0.0) or 0.0))
            oms_next  = st.number_input("Omsättning nästa år (miljoner) — MANUELL", value=float(bef.get("Omsättning nästa år",0.0) or 0.0))

            # Info om att dessa är manuella
            st.caption("Fälten ovan för omsättning är **manuella** och ska inte hämtas automatiskt.")

        spar = st.form_submit_button("💾 Spara")

    # Spara
    if spar:
        if not ticker:
            st.error("Ticker krävs.")
            return df

        # Dublett-kontroll vid nytt bolag
        if bef.empty and str(ticker).upper() in df["Ticker"].astype(str).str.upper().tolist():
            st.error(f"Ticker '{ticker}' finns redan. Dubbletter tillåts inte.")
            return df

        ny = {
            "Ticker": ticker, "Bolagsnamn": bol_namn, "Valuta": valuta,
            "Utestående aktier": utest, "Antal aktier": antal, "GAV (SEK)": gav_sek,
            "Aktuell kurs": akt_kurs, "Årlig utdelning": arlig_utd,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
        }

        # Upptäck ändringar för tidsstämpling
        track_fields = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år","Utestående aktier"]
        changed = []
        if not bef.empty:
            for f in track_fields:
                try:
                    before = float(bef.get(f, 0.0) or 0.0)
                    after  = float(ny.get(f, 0.0) or 0.0)
                    if before != after:
                        changed.append(f)
                except Exception:
                    pass
        else:
            # ny rad – stämpla allt som anges
            changed = [f for f in track_fields if float(ny.get(f, 0.0) or 0.0) != 0.0]

        # Skriv in nya fält
        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"] == ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in df.columns}
            for k, v in ny.items():
                tom[k] = v
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Tidsstämplar (sätt där TS_-kolumner finns)
        from datetime import datetime
        try:
            import pytz
            dt_str = datetime.now(pytz.timezone("Europe/Stockholm")).strftime("%Y-%m-%d")
        except Exception:
            dt_str = datetime.now().strftime("%Y-%m-%d")

        ridx_list = df.index[df["Ticker"] == ticker].tolist()
        if ridx_list:
            ridx = ridx_list[0]
            # markera manuell
            if "Senast manuellt uppdaterad" in df.columns:
                df.at[ridx, "Senast manuellt uppdaterad"] = dt_str
            # stämpla spårade fält
            ts_map = {
                "Utestående aktier": "TS_Utestående aktier",
                "P/S": "TS_P/S",
                "P/S Q1": "TS_P/S Q1",
                "P/S Q2": "TS_P/S Q2",
                "P/S Q3": "TS_P/S Q3",
                "P/S Q4": "TS_P/S Q4",
                "Omsättning idag": "TS_Omsättning idag",
                "Omsättning nästa år": "TS_Omsättning nästa år",
            }
            for f in changed:
                ts_col = ts_map.get(f)
                if ts_col and ts_col in df.columns:
                    df.at[ridx, ts_col] = dt_str

        # Räkna om derived
        df = recompute_derived(df, user_rates)
        st.success("Sparat.")

    st.divider()
    # ---------------------------
    # Manuell prognoslista (endast de två prognos-fälten)
    # ---------------------------
    st.subheader("📝 Manuell prognoslista (Omsättning idag / nästa år)")
    # Sätt upp “äldre av de två” tidsstämplar per rad
    tmp = df.copy()
    def _oldest_of_two(row):
        a = str(row.get("TS_Omsättning idag","")).strip()
        b = str(row.get("TS_Omsättning nästa år","")).strip()
        if not a and not b:
            return ""
        if a and b:
            return min(a, b)
        return a or b

    tmp["_oldest_forecast_ts"] = tmp.apply(_oldest_of_two, axis=1)
    # Sortera: tomma först (saknar TS), sedan äldsta datum
    # För robusthet – hantera tomma som högsta sort (dvs överst)
    def _key(v):
        s = str(v or "")
        if not s:
            return ("", "")
        return (s, s)

    tmp = tmp.sort_values(
        by=["_oldest_forecast_ts","Bolagsnamn","Ticker"],
        key=lambda col: col.map(_key)
    )

    cols_show = ["Ticker","Bolagsnamn","Omsättning idag","TS_Omsättning idag","Omsättning nästa år","TS_Omsättning nästa år"]
    cols_show = [c for c in cols_show if c in tmp.columns]
    st.dataframe(tmp[cols_show].head(30), use_container_width=True, hide_index=True)

    return df
