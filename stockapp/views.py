# stockapp/views.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from .utils import (
    now_stamp,
    ensure_schema,
    to_float,
    pretty_money,
    pretty_number,
    safe_select_index,
    stamp_fields_ts,
)
from .config import FINAL_COLS, TS_FIELDS
from .scoring import rank_candidates, score_row, valuation_signal, compute_mcap, compute_ps_snitt, mcap_bucket
from .sources import run_update_price_only, run_update_full
from .rates import hamta_valutakurs


# ============================================================
# Små UI-hjälpare
# ============================================================

def _ts_label(row: pd.Series) -> str:
    auto = str(row.get("Senast auto-uppdaterad", "") or "").strip()
    src  = str(row.get("Senast uppdaterad källa", "") or "").strip()
    manu = str(row.get("Senast manuellt uppdaterad","") or "").strip()
    bits = []
    if auto:
        bits.append(f"**Auto:** {auto}" + (f" • *{src}*" if src else ""))
    if manu:
        bits.append(f"**Manuell:** {manu}")
    return " | ".join(bits) if bits else "–"

def _maybe(val, default="—"):
    return default if val in (None, "", 0, 0.0, np.nan) else val

def _ps_now(row: pd.Series) -> Optional[float]:
    """P/S nu ≈ (MCAP / LTM-omsättning). Om ej LTM i DF, fall tillbaka till 'P/S'."""
    ps = to_float(row.get("P/S"))
    if ps and ps > 0:
        return float(ps)
    # annars saknas robust LTM i DF, lämna None så vi kan visa '—'
    return None

def _read_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def _warn(text: str):
    st.markdown(f":red[**{text}**]")

def _ok(text: str):
    st.markdown(f":green[**{text}**]")


# ============================================================
# KONTROLLVY
# ============================================================

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # 1) Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    # bygga kolumn "_oldest_any_ts"
    def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
        dates = []
        for c in TS_FIELDS.values():
            if c in row and str(row[c]).strip():
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
        return min(dates) if dates else pd.NaT

    work = df.copy()
    work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
    # sortera äldst → nyast
    vis = work.sort_values(by=["_oldest_any_ts", "Bolagsnamn"], ascending=[True, True]).head(20)

    cols_show = ["Ticker", "Bolagsnamn"]
    for k in [
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4",
        "TS_Omsättning idag","TS_Omsättning nästa år"
    ]:
        if _read_col(df, k):
            cols_show.append(k)
    cols_show.append("_oldest_any_ts")

    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Snabbstatus för TS-fält (hur många saknas?)
    st.subheader("📋 Fältstatus")
    need_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    miss_val = df[need_cols].applymap(lambda x: float(to_float(x) or 0.0) <= 0.0).sum().to_dict()
    miss_ts = {}
    for base in ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]:
        tsc = TS_FIELDS.get(base, None)
        if tsc and tsc in df.columns:
            miss_ts[base] = int(df[tsc].astype(str).str.strip().eq("").sum())
    st.write("**Saknade värden (antal rader):**")
    st.json(miss_val)
    st.write("**Saknade TS-stämplar (antal rader):**")
    st.json(miss_ts)


# ============================================================
# ANALYSVY
# ============================================================

def analysvy(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #",
        min_value=0, max_value=max(0, len(etiketter)-1),
        value=int(st.session_state.analys_idx), step=1
    )
    st.selectbox("Eller välj i lista", etiketter, index=safe_select_index(st.session_state.analys_idx, len(etiketter)))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬅️ Föregående", key="ana_prev"):
            st.session_state.analys_idx = max(0, int(st.session_state.analys_idx)-1)
    with col_b:
        if st.button("➡️ Nästa", key="ana_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, int(st.session_state.analys_idx)+1)
    st.write(f"Post {int(st.session_state.analys_idx)+1}/{len(etiketter)}")

    r = vis_df.iloc[int(st.session_state.analys_idx)]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    show_cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[show_cols].to_dict()]), use_container_width=True, hide_index=True)
    st.caption(_ts_label(r))


# ============================================================
# LÄGG TILL / UPPDATERA
# ============================================================

def lagg_till_eller_uppdatera(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    allow_price_only_btn: bool = True,
    allow_full_auto_btn: bool = True,
) -> pd.DataFrame:
    """
    Returnerar ev. uppdaterad df (skriv i app.py om något ändrats).
    """
    st.header("➕ Lägg till / uppdatera bolag")
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        df = ensure_schema(df)

    # Sorteringsval
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        # Äldst TS
        def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
            dates = []
            for c in TS_FIELDS.values():
                if c in row and str(row[c]).strip():
                    d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                    if pd.notna(d):
                        dates.append(d)
            return min(dates) if dates else pd.NaT
        work = df.copy()
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        vis_df = work.sort_values(by=["_oldest_any_ts","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    labels = [""] + list(namn_map.keys())

    if "edit_idx" not in st.session_state:
        st.session_state.edit_idx = 0
    selected = st.selectbox("Välj bolag (lämna tomt för nytt)", labels, index=safe_select_index(st.session_state.edit_idx, len(labels)))
    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_idx = max(0, int(st.session_state.edit_idx) - 1)
    with col_mid:
        st.write(f"Post {int(st.session_state.edit_idx)}/{max(1, len(labels)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_idx = min(len(labels)-1, int(st.session_state.edit_idx) + 1)

    if selected and selected in namn_map:
        bef = df[df["Ticker"] == namn_map[selected]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest  = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal  = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gavsek = st.number_input("GAV (SEK)", value=float(bef.get("GAV (SEK)",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)

        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            valuta   = st.text_input("Valuta (t.ex. USD/EUR/SEK/NOK/CAD)", value=str(bef.get("Valuta","")) if not bef.empty else "")

            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över dina manuella 0-värden):**")
            st.write("- Beräkningar/riktkurser räknas om i appen efter spara")

        saved = st.form_submit_button("💾 Spara")

    if saved and ticker:
        # skapa/upd rad
        if (df["Ticker"].astype(str).str.upper() == ticker).any():
            # uppdatera befintlig
            ridx = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]
        else:
            # ny rad med defaults
            empty = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            empty["Ticker"] = ticker
            df = pd.concat([df, pd.DataFrame([empty])], ignore_index=True)
            ridx = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]

        before = {k: float(df.at[ridx, k]) if k in df.columns else 0.0 for k in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]}
        after  = {"P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4, "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next}

        # skriv värden
        df.at[ridx, "Ticker"] = ticker
        df.at[ridx, "Utestående aktier"] = float(utest or 0.0)
        df.at[ridx, "Antal aktier"] = float(antal or 0.0)
        df.at[ridx, "GAV (SEK)"] = float(gavsek or 0.0)
        df.at[ridx, "P/S"] = float(ps or 0.0)
        df.at[ridx, "P/S Q1"] = float(ps1 or 0.0)
        df.at[ridx, "P/S Q2"] = float(ps2 or 0.0)
        df.at[ridx, "P/S Q3"] = float(ps3 or 0.0)
        df.at[ridx, "P/S Q4"] = float(ps4 or 0.0)
        df.at[ridx, "Omsättning idag"] = float(oms_idag or 0.0)
        df.at[ridx, "Omsättning nästa år"] = float(oms_next or 0.0)
        if valuta:
            df.at[ridx, "Valuta"] = valuta.upper()

        # TS för manuella fält som ändrats
        changed_manual = [k for k in after if float(before.get(k, 0.0)) != float(after[k])]
        if changed_manual:
            df.at[ridx, "Senast manuellt uppdaterad"] = now_stamp()
            stamp_fields_ts(df, ridx, changed_manual)

        _ok("Sparat (inte skrivet till Google Sheet ännu – det gör app.py när vyn returnerat).")

    # Snabb-åtgärder för valt bolag
    if selected and selected in namn_map:
        tkr = namn_map[selected]
        ridx = df.index[df["Ticker"] == tkr][0]
        st.markdown("---")
        st.subheader("⚙️ Uppdatera det här bolaget")

        c1, c2 = st.columns(2)
        with c1:
            if allow_price_only_btn and st.button("🔁 Uppdatera **kurs** (pris)"):
                try:
                    df2, changed, err = run_update_price_only(tkr, df, user_rates)
                    if err:
                        _warn(f"Fel: {err}")
                    else:
                        df = df2
                        if changed:
                            df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
                            df.at[ridx, "Senast uppdaterad källa"] = "Yahoo pris"
                        _ok(f"Klar. Fält ändrade: {changed if changed else '—'}")
                except Exception as e:
                    _warn(f"Fel vid pris-uppdatering: {e}")

        with c2:
            if allow_full_auto_btn and st.button("🚀 Full auto (1 bolag)"):
                try:
                    df2, changed, err = run_update_full(tkr, df, user_rates)
                    if err:
                        _warn(f"Fel: {err}")
                    else:
                        df = df2
                        if changed:
                            df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
                            df.at[ridx, "Senast uppdaterad källa"] = "Auto (SEC/Yahoo→Yahoo→FMP/Finnhub)"
                        _ok(f"Klar. Fält ändrade: {changed if changed else '—'}")
                except Exception as e:
                    _warn(f"Fel vid full auto: {e}")

        st.caption(_ts_label(df.loc[ridx]))

    st.markdown("---")
    # Manuell prognoslista (äldre TS på Omsättning idag / nästa år) – hitflyttad
    st.subheader("📝 Manuell prognoslista (omsättning i år & nästa år)")
    # välj äldst av just de två TS_kolumnerna
    def _oldest_two(row: pd.Series) -> Optional[pd.Timestamp]:
        ds = []
        for base in ["Omsättning idag","Omsättning nästa år"]:
            ts_col = TS_FIELDS.get(base)
            if ts_col and ts_col in row and str(row[ts_col]).strip():
                d = pd.to_datetime(str(row[ts_col]).strip(), errors="coerce")
                if pd.notna(d):
                    ds.append(d)
        return min(ds) if ds else pd.NaT

    w2 = df.copy()
    w2["_oldest_forecast_ts"] = w2.apply(_oldest_two, axis=1)
    w2 = w2.sort_values(by=["_oldest_forecast_ts","Bolagsnamn"], ascending=[True, True]).head(25)

    cols2 = ["Ticker","Bolagsnamn","Omsättning idag","Omsättning nästa år"]
    for base in ["Omsättning idag","Omsättning nästa år"]:
        tsc = TS_FIELDS.get(base)
        if tsc and tsc in df.columns:
            cols2.append(tsc)
    cols2.append("_oldest_forecast_ts")
    st.dataframe(w2[cols2], use_container_width=True, hide_index=True)

    return df


# ============================================================
# INVESTERINGSFÖRSLAG
# ============================================================

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("💡 Investeringsförslag")
    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    # Filter- & lägespanel
    col_top = st.columns(4)
    with col_top[0]:
        mode = st.radio("Läge", ["Tillväxt","Utdelning"], horizontal=True)
    with col_top[1]:
        riktkurs_val = st.selectbox("Riktkurs", ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"], index=0)
    with col_top[2]:
        sektor = st.text_input("Filter: Sektor (valfritt)", "")
    with col_top[3]:
        mcap_sel = st.multiselect("MCAP-bucket", ["Micro","Small","Mid","Large","Mega"], default=[])

    # ranka kandidater
    rk = rank_candidates(
        df, 
        mode=("dividend" if mode=="Utdelning" else "growth"),
        riktkurs_col=riktkurs_val,
        sector_filter=sektor or None,
        mcap_buckets=mcap_sel or None,
        top_n=50,
    )
    if rk.empty:
        st.warning("Inga kandidater efter filter/krav.")
        return

    st.markdown(f"**Visar {len(rk)} kandidater** (sorterade på *Score* → *Potential %*).")

    # Kortlistning
    for _, row in rk.iterrows():
        with st.expander(f"{row.get('Bolagsnamn','?')} ({row.get('Ticker','?')}): Score {row['Score']:.1f} • {row['Valuationsignal']} • Potential {row['Potential (%)']:.1f}%"):
            # Bas
            px = to_float(row.get("Aktuell kurs")) or 0.0
            psn = compute_ps_snitt(row)
            psn_str = f"{psn:.2f}" if psn is not None else "—"

            mcap = compute_mcap(row)
            mcap_str = pretty_money(mcap, suffix=row.get("Valuta","")) if mcap else "—"

            st.write(
                f"- **Aktuell kurs:** {px:.2f} {row.get('Valuta','')}\n"
                f"- **P/S nu:** {(_ps_now(row) if _ps_now(row) is not None else '—')}\n"
                f"- **P/S-snitt (Q1–Q4):** {psn_str}\n"
                f"- **MCAP (nu):** {mcap_str}\n"
                f"- **MCAP bucket:** {mcap_bucket(row)}\n"
                f"- **Riktkurs (val):** {to_float(row.get(riktkurs_val)) or 0.0:.2f} {row.get('Valuta','')}\n"
                f"- **Potential:** {row['Potential (%)']:.2f} %\n"
                f"- **Sektor:** {row.get('Sektor') or row.get('Sector') or '—'}"
            )

            # Visa ev. historik om finns i DF
            hist_bits = []
            for k in ["MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4"]:
                if k in df.columns and pd.notna(row.get(k)):
                    hist_bits.append(f"{k}: {pretty_money(to_float(row.get(k)))}")
            if hist_bits:
                st.write("**Historik:** " + " | ".join(hist_bits))

            # Förklaring från scoring
            st.caption(row.get("Förklaring",""))

    st.success("Klar – välj gärna annan sektor, MCAP-bucket eller riktkurs för att se andra förslag.")


# ============================================================
# PORTFÖLJ
# ============================================================

def _sell_watch_flag(row: pd.Series, mode: str) -> Optional[str]:
    """
    Enkel säljvakt: flaggar "Trimma" eller "Sälj" om:
      - lågt fundamentalscore och/eller negativ uppsida mot riktkurs
      - övervikt i portföljen (hanteras i portföljvyn via andel)
    """
    riktkurs_col = "Riktkurs om 1 år"
    s, _, _ = score_row(row, mode=("dividend" if mode=="Utdelning" else "growth"))
    label, _ = valuation_signal(row, mode=("dividend" if mode=="Utdelning" else "growth"), score=s, riktkurs_col=riktkurs_col)
    if "Sälj" in label:
        return "Sälj/Övervärderad"
    if "Trimma" in label:
        return "Trimma"
    return None

def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str, float]) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Värden i SEK
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())

    # Andel, utdelning
    port["Andel (%)"] = np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)
    port["Total årlig utdelning (SEK)"] = (port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]).fillna(0.0)
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {pretty_money(total_värde, 'SEK')}")
    st.markdown(f"**Total kommande utdelning:** {pretty_money(tot_utd, 'SEK')}")
    st.markdown(f"**Ungefärlig månadsutdelning:** {pretty_money(tot_utd/12.0, 'SEK')}")

    # Risketikett & säljvakt
    mode = st.radio("Säljvakt – utvärdera som", ["Tillväxt","Utdelning"], index=0, horizontal=True)
    port["Säljvakt"] = port.apply(lambda r: _sell_watch_flag(r, mode), axis=1)

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta"]
    if "GAV (SEK)" in port.columns:
        show_cols.append("GAV (SEK)")
    for c in ["Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)","Sektor","Säljvakt"]:
        if c in port.columns or c in ["Säljvakt"]:
            show_cols.append(c)

    st.dataframe(
        port[show_cols].sort_values(by="Värde (SEK)", ascending=False),
        use_container_width=True, hide_index=True
    )

    # Sektorfördelning
    st.subheader("📊 Sektorfördelning")
    sec = port.groupby(port["Sektor"].fillna("Okänd"))["Värde (SEK)"].sum().sort_values(ascending=False)
    if not sec.empty:
        st.bar_chart(sec)

    # Lista säljvakt-förslag
    st.subheader("🚨 Säljvakt – förslag att se över")
    cand = port[port["Säljvakt"].notna()].copy()
    if cand.empty:
        st.info("Inga tydliga säljsignaler just nu.")
    else:
        st.dataframe(
            cand[["Ticker","Bolagsnamn","Säljvakt","Andel (%)","Aktuell kurs","Valuta","GAV (SEK)"]].fillna("—"),
            use_container_width=True, hide_index=True
        )
