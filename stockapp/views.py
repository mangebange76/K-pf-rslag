# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from .rates import hamta_valutakurs
from .utils import now_stamp, add_oldest_ts_col
from .config import TS_FIELDS

# -----------------------------------------------------------------------------------
# Hjälpare
# -----------------------------------------------------------------------------------
def _format_mcap(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    if v >= 1e12:  return f"{v/1e12:.2f} T"
    if v >= 1e9:   return f"{v/1e9:.2f} B"
    if v >= 1e6:   return f"{v/1e6:.2f} M"
    if v >= 1e3:   return f"{v/1e3:.2f} k"
    return f"{v:.0f}"

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# -----------------------------------------------------------------------------------
# Kontroll-vy
# -----------------------------------------------------------------------------------
def kontrollvy(df: pd.DataFrame):
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns: cols_show.append(k)
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Senaste batchlogg
    log = st.session_state.get("_batch_log", [])
    st.subheader("📒 Senaste batchlogg")
    if not log:
        st.info("Ingen batchkörning körd i denna session ännu.")
    else:
        for row in log[-15:]:
            emoji = "✅" if row.get("ok") else "⚠️"
            st.write(f"{emoji} {row.get('ticker')}: {row.get('msg')}  ({row.get('ts')})")

# -----------------------------------------------------------------------------------
# Analys-vy
# -----------------------------------------------------------------------------------
def analysvy(df: pd.DataFrame, user_rates: dict):
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    if vis_df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1)
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
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Sektor","Industri","Risklabel",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år",
        "Mcap Q1","Mcap Q2","Mcap Q3","Mcap Q4"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------------
# Lägg till / uppdatera
# -----------------------------------------------------------------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))

    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    from .sources import run_update_price_only, run_update_full

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav_sek = st.number_input("GAV (SEK)", value=float(bef.get("GAV (SEK)",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner) – MANUELL",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner) – MANUELL", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Tips:** Dessa två fält uppdateras inte automatiskt.")
            st.write("- Vid spara räknas riktkurser/beräkningar om.")

            colu = st.columns(2)
            with colu[0]:
                upd_price = st.form_submit_button("🔼 Uppdatera kurs (denna)")
            with colu[1]:
                upd_full = st.form_submit_button("🔄 Full uppdatering (denna)")

        spar = st.form_submit_button("💾 Spara")

    # Hantering
    if upd_price and (bef.empty and not ticker or not bef.empty):
        # hitta index
        if not bef.empty:
            ridx = df.index[df["Ticker"] == bef["Ticker"]][0]
        else:
            st.warning("Lägg till sparat bolag först, kör sedan uppdatering.")
            ridx = None
        if ridx is not None:
            from .sources import run_update_price_only
            df, changed, msg = run_update_price_only(df, ridx, user_rates)
            st.success(msg if msg else "Klar")
            return df

    if upd_full and (bef.empty and not ticker or not bef.empty):
        if not bef.empty:
            ridx = df.index[df["Ticker"] == bef["Ticker"]][0]
        else:
            st.warning("Lägg till sparat bolag först, kör sedan uppdatering.")
            ridx = None
        if ridx is not None:
            from .sources import run_update_full
            df, changed, msg = run_update_full(df, ridx, user_rates)
            st.success(msg if msg else "Klar")
            return df

    if spar and ticker:
        # Dublettskydd
        if bef.empty and ticker in set(df["Ticker"].astype(str)):
            st.error("Ticker finns redan – dubbletter tillåts inte.")
            return df

        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV (SEK)": gav_sek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Skriv in nya fält
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            # skapa tom rad
            row = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa","Sektor","Industri","Risklabel","Värderingslabel"] and not str(c).startswith("TS_") else "") for c in df.columns}
            row.update(ny)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        st.success("Sparat.")
        return df

    # Manuell prognoslista (äldst TS för Omsättning idag/nästa år)
    st.markdown("### 📝 Manuell prognoslista (äldst först)")
    cols_watch = []
    if "TS_Omsättning idag" in df.columns: cols_watch.append("TS_Omsättning idag")
    if "TS_Omsättning nästa år" in df.columns: cols_watch.append("TS_Omsättning nästa år")
    if cols_watch:
        tmp = df.copy()
        # beräkna äldsta av dessa två TS per rad
        def _oldest_ts(row):
            cand = []
            for c in cols_watch:
                s = str(row.get(c,"")).strip()
                if s:
                    try:
                        d = pd.to_datetime(s, errors="coerce")
                        if pd.notna(d): cand.append(d)
                    except Exception:
                        pass
            return min(cand) if cand else pd.NaT
        tmp["_oldest_manual_ts"] = tmp.apply(_oldest_ts, axis=1)
        tmp = tmp.sort_values(by=["_oldest_manual_ts","Bolagsnamn"], na_position="last")
        show_cols = ["Ticker","Bolagsnamn"] + cols_watch + ["_oldest_manual_ts"]
        st.dataframe(tmp[show_cols].head(20), use_container_width=True, hide_index=True)

    return df

# -----------------------------------------------------------------------------------
# Investeringsförslag
# -----------------------------------------------------------------------------------
def _label_value(val, good_high=True, th_ok=0.0):
    """Snabb etikettning av nyckeltal: '👍', '⚠️', '👎'."""
    try:
        v = float(val)
    except Exception:
        return "–"
    if good_high:
        return "👍" if v > th_ok else "👎"
    else:
        return "👍" if v < th_ok else "👎"

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict):
    st.header("💡 Investeringsförslag")

    # Filter – typ och sektor och riskklass
    typ = st.radio("Fokus", ["Tillväxt","Utdelning"], horizontal=True)
    sector = st.selectbox("Filtrera sektor (valfritt)", ["(alla)"] + sorted(list(set(df.get("Sektor", pd.Series(dtype=str)).astype(str)))) )
    risk = st.selectbox("Risklabel (valfritt)", ["(alla)","Nano","Micro","Small","Mid","Large"], index=0)

    # Riktkursval
    riktkurs_val = st.selectbox("Riktkurs att jämföra mot", ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)

    base = df.copy()
    if sector != "(alla)":
        base = base[base["Sektor"].astype(str) == sector]
    if risk != "(alla)":
        base = base[base["Risklabel"].astype(str) == risk]

    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar filtren just nu.")
        return

    # Scoring (enkel fallback). Högre score = bättre.
    # Exempel: Tillväxt => viktar upp CAGR, ned P/S. Utdelning => viktar upp yield, ned payout-risk (om fanns).
    def _score_row(r):
        ps  = _safe_float(r.get("P/S",0))
        psn = _safe_float(r.get("P/S-snitt",0))
        cagr = _safe_float(r.get("CAGR 5 år (%)",0))
        yield_ = 0.0
        if _safe_float(r.get("Aktuell kurs",0)) > 0:
            yield_ = _safe_float(r.get("Årlig utdelning",0)) / _safe_float(r.get("Aktuell kurs",1e-9)) * 100.0

        upside = ( _safe_float(r.get(riktkurs_val,0)) - _safe_float(r.get("Aktuell kurs",0)) ) / max(_safe_float(r.get("Aktuell kurs",0)),1e-9) * 100.0

        if typ == "Tillväxt":
            score = 0.40*(cagr) + 0.35*(upside/2) - 0.25*(ps if ps>0 else 0)
        else:
            score = 0.45*(yield_) + 0.35*(upside/2) - 0.20*(ps if ps>0 else 0)
        return float(score)

    base["Score"] = base.apply(_score_row, axis=1)

    # Sortera på score
    base = base.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # Navigering
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

    rad = base.iloc[st.session_state.forslags_index]
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = _safe_float(rad["Aktuell kurs"]) * vx

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']}) — Score: {rad['Score']:.1f}")

    # Snabbpanel
    mcap_now = 0.0
    for mc_col in ["Mcap Q1","Mcap Q2","Mcap Q3","Mcap Q4"]:
        if mc_col in df.columns and _safe_float(rad.get(mc_col,0)) > mcap_now:
            mcap_now = _safe_float(rad.get(mc_col,0))  # ungefärlig (senaste historiska om nu ej finns)

    psn = _safe_float(rad.get("P/S-snitt",0))
    lines = [
        f"- **Aktuell kurs:** {rad['Aktuell kurs']:.2f} {rad['Valuta']}  (~{kurs_sek:.2f} SEK)",
        f"- **Utestående aktier:** {_safe_float(rad.get('Utestående aktier',0)):.2f} M",
        f"- **P/S (nu):** {_safe_float(rad.get('P/S',0)):.2f}",
        f"- **P/S-snitt (Q1–Q4):** {psn:.2f}",
        f"- **Riktkurs (val):** {_safe_float(rad.get(riktkurs_val,0)):.2f} {rad['Valuta']}",
    ]
    st.markdown("\n".join(lines))

    with st.expander("Nyckeltal & historik"):
        cols = st.columns(2)
        with cols[0]:
            st.write("**Market cap historik**")
            mc_table = {
                "Mcap Q1": _format_mcap(rad.get("Mcap Q1",0)),
                "Mcap Q2": _format_mcap(rad.get("Mcap Q2",0)),
                "Mcap Q3": _format_mcap(rad.get("Mcap Q3",0)),
                "Mcap Q4": _format_mcap(rad.get("Mcap Q4",0)),
            }
            st.json(mc_table)
            st.write("**P/S historik**")
            ps_table = {
                "P/S Q1": f"{_safe_float(rad.get('P/S Q1',0)):.2f}",
                "P/S Q2": f"{_safe_float(rad.get('P/S Q2',0)):.2f}",
                "P/S Q3": f"{_safe_float(rad.get('P/S Q3',0)):.2f}",
                "P/S Q4": f"{_safe_float(rad.get('P/S Q4',0)):.2f}",
            }
            st.json(ps_table)
        with cols[1]:
            st.write("**Bolagsprofil**")
            st.write(f"- Sektor: {rad.get('Sektor','')}")
            st.write(f"- Industri: {rad.get('Industri','')}")
            st.write(f"- Risklabel: {rad.get('Risklabel','')}")

# -----------------------------------------------------------------------------------
# Portfölj
# -----------------------------------------------------------------------------------
def visa_portfolj(df: pd.DataFrame, user_rates: dict):
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)
    port["Utdelningsyield (%)"] = np.where(port["Aktuell kurs"]>0, port["Årlig utdelning"] / port["Aktuell kurs"] * 100.0, 0.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Utdelningsyield (%)","Total årlig utdelning (SEK)","GAV (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    # Sektor-exponering
    if "Sektor" in port.columns:
        st.subheader("📊 Sektor-exponering")
        sec = port.groupby("Sektor", dropna=False)["Värde (SEK)"].sum().sort_values(ascending=False)
        if not sec.empty:
            st.dataframe(sec.to_frame("Värde (SEK)").assign(**{"Andel (%)": (sec/sec.sum()*100).round(2)}), use_container_width=True)
