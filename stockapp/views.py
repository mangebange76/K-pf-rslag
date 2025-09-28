# stockapp/views.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Dict, Optional, List
import streamlit as st
import pandas as pd
import numpy as np

from .utils import (
    human_money,
    recompute_derived,
    add_oldest_ts_col,
    top_missing_by_ts,
    ps_consistency_flag,
)

# Egen now_stamp (lokal Stockholm om pytz finns)
from datetime import datetime
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp(): return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp(): return datetime.now().strftime("%Y-%m-%d")


# ------------------------------------------------------------
# Hjälpare (TS-stämpling & robust index-bläddrare)
# ------------------------------------------------------------

MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"
]

def _stamp_ts_for(df: pd.DataFrame, ticker: str, fields: List[str]):
    if "Ticker" not in df.columns or not (df["Ticker"] == ticker).any():
        return
    ridx = df.index[df["Ticker"] == ticker][0]
    for f in fields:
        ts_col = f"TS_{f}"
        if ts_col in df.columns:
            df.loc[ridx, ts_col] = now_stamp()
    if "Senast manuellt uppdaterad" in df.columns:
        df.loc[ridx, "Senast manuellt uppdaterad"] = now_stamp()

def _robust_nav_state(key: str, max_len: int) -> int:
    """Säkerställ att en sessionsindex alltid ligger 0..max_len-1."""
    if key not in st.session_state:
        st.session_state[key] = 0
    try:
        idx = int(st.session_state[key])
    except Exception:
        idx = 0
    if max_len <= 0:
        st.session_state[key] = 0
        return 0
    if idx < 0: idx = 0
    if idx > max_len - 1: idx = max_len - 1
    st.session_state[key] = idx
    return idx


# ------------------------------------------------------------
# Kontroll-vy
# ------------------------------------------------------------
def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # 1) Äldst uppdaterade (alla TS_-fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"]).head(20)
    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns:
            cols_show.append(k)
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Sanity P/S
    st.subheader("🧪 P/S sanity-check (extrema värden)")
    tmp = df.copy()
    tmp["PS-flagga"] = tmp.apply(ps_consistency_flag, axis=1)
    problem = tmp[tmp["PS-flagga"] != "ok"][["Ticker","Bolagsnamn","P/S","P/S-snitt","PS-flagga"]].sort_values(by="PS-flagga", ascending=False)
    if problem.empty:
        st.success("Inga uppenbara extrema P/S just nu.")
    else:
        st.warning(f"{len(problem)} bolag flaggade:")
        st.dataframe(problem, use_container_width=True, hide_index=True)

    st.divider()

    # 3) Senaste körlogg (om batchpanelen lagt något)
    if "_batch_log" in st.session_state:
        st.subheader("📒 Senaste körlogg (Batch)")
        log = st.session_state["_batch_log"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Ändringar**")
            st.json(log.get("changed") or {})
        with col2:
            st.markdown("**Missar**")
            st.json(log.get("misses") or {})
        with col3:
            st.markdown("**Fel**")
            st.json(log.get("errors") or {})
    else:
        st.info("Ingen batchkörning i denna session ännu.")

    st.caption("Tip: Använd sidopanelens batch för att skapa en kö (med 1/X-visning) och köra nästa/5/helt.")


# ------------------------------------------------------------
# Analys-vy
# ------------------------------------------------------------
def analysvy(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("📈 Analys")

    vis_df = df.copy()
    if vis_df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = vis_df.sort_values(by=[c for c in ["Bolagsnamn","Ticker"] if c in vis_df.columns]).reset_index(drop=True)
    etiketter = [f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})" for _, r in vis_df.iterrows()]

    idx = _robust_nav_state("analys_idx", len(etiketter))

    cA, cB, cC = st.columns([1,2,1])
    with cA:
        if st.button("⬅️ Föregående", key="ana_prev"):
            st.session_state["analys_idx"] = max(0, idx-1)
    with cB:
        st.selectbox("Välj bolag", etiketter, index=idx, key="analys_idx")
    with cC:
        if st.button("➡️ Nästa", key="ana_next"):
            st.session_state["analys_idx"] = min(len(etiketter)-1, idx+1)

    r = vis_df.iloc[_robust_nav_state("analys_idx", len(etiketter))]

    st.subheader(f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})")
    badge = []
    if str(r.get("Senast manuellt uppdaterad","")).strip():
        badge.append(f"✍️ Manuell: {r.get('Senast manuellt uppdaterad')}")
    if str(r.get("Senast auto-uppdaterad","")).strip():
        badge.append(f"⚙️ Auto: {r.get('Senast auto-uppdaterad')}")
    if badge:
        st.caption(" • ".join(badge))

    show_cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta",
        "Aktuell kurs","Utestående aktier","Market Cap (nu)",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Årlig utdelning","GAV SEK",
        "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[show_cols].to_dict()]), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Lägg till / uppdatera
# ------------------------------------------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame,
                              user_rates: Dict[str,float],
                              save_cb: Optional[Callable[[pd.DataFrame], None]] = None) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=[c for c in ["Bolagsnamn","Ticker"] if c in df.columns])

    namn_map = {f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})": r.get('Ticker','') for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    idx = _robust_nav_state("edit_idx", len(val_lista))
    st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=idx, key="edit_idx")

    col_nav_l, col_nav_m, col_nav_r = st.columns([1,2,1])
    with col_nav_l:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state["edit_idx"] = max(0, idx-1)
    with col_nav_m:
        st.caption(f"Post {idx}/{max(1, len(val_lista)-1)}")
    with col_nav_r:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state["edit_idx"] = min(len(val_lista)-1, idx+1)

    ticker_selected = ""
    if val_lista and idx < len(val_lista):
        label = val_lista[idx]
        if label and label in namn_map:
            ticker_selected = namn_map[label]

    if ticker_selected:
        mask = (df["Ticker"].astype(str).str.upper() == str(ticker_selected).upper())
        bef = df[mask].iloc[0] if mask.any() else pd.Series({}, dtype=object)
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_edit_company"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            namn   = st.text_input("Bolagsnamn", value=bef.get("Bolagsnamn","") if not bef.empty else "")
            sektor = st.text_input("Sektor", value=bef.get("Sektor","") if not bef.empty else "")
            valuta = st.text_input("Valuta", value=bef.get("Valuta","") if not bef.empty else "USD")

            utest  = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal  = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav    = st.number_input("GAV SEK (genomsnittligt anskaffningsvärde i SEK)", value=float(bef.get("GAV SEK",0.0)) if not bef.empty else 0.0)

            ps   = st.number_input("P/S", value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1  = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2  = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3  = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4  = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            oms_2y   = st.number_input("Omsättning om 2 år (miljoner)", value=float(bef.get("Omsättning om 2 år",0.0)) if not bef.empty else 0.0)
            oms_3y   = st.number_input("Omsättning om 3 år (miljoner)", value=float(bef.get("Omsättning om 3 år",0.0)) if not bef.empty else 0.0)

            utd      = st.number_input("Årlig utdelning (per aktie i bolagets valuta)", value=float(bef.get("Årlig utdelning",0.0)) if not bef.empty else 0.0)
            kassa    = st.number_input("Kassa (valuta)", value=float(bef.get("Kassa (valuta)",0.0)) if not bef.empty else 0.0)
            fcf      = st.number_input("FCF TTM (valuta)", value=float(bef.get("FCF TTM (valuta)",0.0)) if not bef.empty else 0.0)
            de       = st.number_input("Debt/Equity", value=float(bef.get("Debt/Equity",0.0)) if not bef.empty else 0.0)
            gm       = st.number_input("Bruttomarginal (%)", value=float(bef.get("Bruttomarginal (%)",0.0)) if not bef.empty else 0.0)
            nm       = st.number_input("Nettomarginal (%)", value=float(bef.get("Nettomarginal (%)",0.0)) if not bef.empty else 0.0)

            st.caption("Vid spara: Riktkurser och härledda fält räknas om.")

        col_s1, col_s2, col_s3 = st.columns([1,1,1])
        with col_s1:
            spar = st.form_submit_button("💾 Spara")
        with col_s2:
            run_price = st.form_submit_button("🔄 Uppdatera endast kurs (runner)")
        with col_s3:
            run_full  = st.form_submit_button("🧰 Full auto (runner)")

    if spar and ticker:
        # Dublettskydd lokalt
        if (ticker_selected == "" or ticker_selected != ticker) and (df["Ticker"].astype(str).str.upper() == ticker).any():
            st.error(f"Ticker {ticker} finns redan. Ändra ticker eller välj den posten i listan.")
            return df

        ny = {
            "Ticker": ticker, "Bolagsnamn": namn, "Sektor": sektor, "Valuta": valuta,
            "Utestående aktier": utest, "Antal aktier": antal, "GAV SEK": gav,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
            "Omsättning om 2 år": oms_2y, "Omsättning om 3 år": oms_3y,
            "Årlig utdelning": utd, "Kassa (valuta)": kassa, "FCF TTM (valuta)": fcf,
            "Debt/Equity": de, "Bruttomarginal (%)": gm, "Nettomarginal (%)": nm,
        }

        if ticker_selected:
            ridx = df.index[df["Ticker"].astype(str).str.upper() == ticker_selected.upper()][0]
            bef_before = df.loc[ridx].copy()
            for k, v in ny.items():
                df.loc[ridx, k] = v

            changed_manual_fields = []
            for f in MANUELL_FALT_FOR_DATUM:
                try:
                    before = float(bef_before.get(f, 0.0))
                    after  = float(ny.get(f, 0.0))
                    if before != after:
                        changed_manual_fields.append(f)
                except Exception:
                    pass
            if changed_manual_fields:
                _stamp_ts_for(df, ticker, changed_manual_fields)
        else:
            for col in df.columns:
                ny.setdefault(col, "" if (col in ["Ticker","Bolagsnamn","Sektor","Valuta"] or str(col).startswith("TS_")) else 0.0)
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)

            changed_manual_fields = [f for f in MANUELL_FALT_FOR_DATUM if float(ny.get(f,0.0)) != 0.0]
            if changed_manual_fields:
                _stamp_ts_for(df, ticker, changed_manual_fields)

        df = recompute_derived(df)

        if save_cb:
            try:
                save_cb(df)
                st.success("Sparat.")
            except Exception as e:
                st.warning(f"Kunde inte spara: {e}")
        else:
            st.info("Ingen spara-callback bunden (ändringen ligger kvar i minnet denna session).")

    # Runner-knappar (om du satt runners i sessionen)
    if ticker_selected and (run_price or run_full):
        runner = None
        if run_price:
            runner = st.session_state.get("_runner_price_only")
            if runner is None:
                st.info("Ingen 'kurs-only' runner bunden (_runner_price_only).")
        if run_full and runner is None:
            runner = st.session_state.get("_runner")
            if runner is None:
                st.info("Ingen 'full auto' runner bunden (_runner).")

        if runner is not None:
            try:
                df2, changed, status_str = runner(df, ticker_selected, user_rates)
                if isinstance(df2, pd.DataFrame):
                    df[:] = df2
                df = recompute_derived(df)
                if save_cb:
                    try:
                        save_cb(df)
                        st.success(f"Runner körd ({status_str}). Ändringar: {', '.join(changed) if changed else '–'}")
                    except Exception as e:
                        st.warning(f"Kunde inte spara efter runner: {e}")
            except Exception as e:
                st.error(f"Runner-fel: {e}")

    st.markdown("### 📝 Manuell prognoslista (äldsta TS för *Omsättning idag* & *Omsättning nästa år*)")
    need = top_missing_by_ts(df, ["Omsättning idag","Omsättning nästa år"], limit=30)
    if need.empty:
        st.success("Alla ser uppdaterade ut just nu.")
    else:
        st.dataframe(need, use_container_width=True, hide_index=True)

    return df


# ------------------------------------------------------------
# Investeringsförslag (med robust fallback)
# ------------------------------------------------------------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("💡 Investeringsförslag")

    if df.empty:
        st.info("Inga bolag i databasen.")
        return

    df = recompute_derived(df)

    # ------- Filter: sektor & risklabel -------
    sektorer = ["(Alla)"] + sorted([s for s in df.get("Sektor", pd.Series()).dropna().unique().tolist() if str(s).strip()])
    sektor_val = st.selectbox("Filtrera sektor", sektorer, index=0)

    def _risklabel(mcap):
        try:
            v = float(mcap)
        except Exception:
            return "Unknown"
        if v >= 1_000_000_000_000:   # >= 1T
            return "Mega"
        if v >= 200_000_000_000:
            return "Large"
        if v >= 10_000_000_000:
            return "Mid"
        if v >= 2_000_000_000:
            return "Small"
        return "Micro"

    base = df.copy()
    if "Market Cap (nu)" not in base.columns:
        base["Market Cap (nu)"] = 0.0

    if sektor_val != "(Alla)":
        base = base[base.get("Sektor","") == sektor_val]

    base["Risklabel"] = base["Market Cap (nu)"].apply(_risklabel)

    risk_opts = ["(Alla)","Micro","Small","Mid","Large","Mega"]
    risk_val = st.selectbox("Filtrera risklabel", risk_opts, index=0)
    if risk_val != "(Alla)":
        base = base[base["Risklabel"] == risk_val]

    # ------- Mode: Tillväxt / Utdelning -------
    mode = st.radio("Typ av förslag", ["Tillväxt","Utdelning"], horizontal=True, index=0)

    has_psn = "P/S-snitt" in base.columns
    has_ps  = "P/S" in base.columns
    has_price = "Aktuell kurs" in base.columns
    has_target = "Riktkurs om 1 år" in base.columns

    # ------- Huvudurval/score -------
    main = base.copy()
    if has_price: main = main[main["Aktuell kurs"] > 0]
    if has_psn:   main = main[main["P/S-snitt"] > 0]
    if mode == "Tillväxt" and has_target and has_price:
        main["Potential (%)"] = (main["Riktkurs om 1 år"] - main["Aktuell kurs"]) / main["Aktuell kurs"] * 100.0
    else:
        main["Potential (%)"] = np.nan

    if not main.empty and has_psn:
        if mode == "Tillväxt":
            main["Score"] = (
                0.6 * main["Potential (%)"].clip(lower=-100, upper=300).fillna(0.0) -
                0.3 * main["P/S-snitt"].clip(upper=2000).fillna(0.0) +
                0.1 * (main.get("Runway (mån)", 0.0) >= 24).astype(float) * 20.0
            )
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                dy = (main.get("Årlig utdelning",0.0) / main["Aktuell kurs"]).replace([np.inf, -np.inf], 0.0).fillna(0.0) * 100.0
            main["Score"] = (
                0.6 * dy.clip(0, 20) -
                0.2 * main.get("Debt/Equity", 0.0).clip(0, 400) +
                0.2 * (main.get("FCF TTM (valuta)", 0.0) > 0).astype(float) * 20.0
            )
        main = main.sort_values(by="Score", ascending=False).reset_index(drop=True)

    # ======= FALLBACK: Top-20 lägst P/S-snitt =======
    def _show_fallback_psn(df_f):
        st.info("⚠️ Fallback aktiv: visar **Top-20 (lägst) P/S-snitt**.")
        tmp = df_f.copy()
        if "P/S-snitt" not in tmp.columns and {"P/S Q1","P/S Q2","P/S Q3","P/S Q4"}.issubset(tmp.columns):
            ps_cols = ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]
            tmp["P/S-snitt"] = tmp[ps_cols].replace(0, np.nan).mean(axis=1).fillna(0.0)

        if "P/S-snitt" not in tmp.columns:
            st.warning("P/S-snitt saknas — kan inte visa P/S-fallback.")
            return

        tmp = tmp[tmp["P/S-snitt"] > 0].sort_values(by="P/S-snitt", ascending=True).head(20)
        if tmp.empty:
            st.warning("Inget att visa i fallback heller.")
            return

        show_cols = [c for c in ["Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","P/S-snitt","P/S","Market Cap (nu)"] if c in tmp.columns]
        st.dataframe(tmp[show_cols], use_container_width=True, hide_index=True)

    if main.empty:
        _show_fallback_psn(base)
        return

    # ------- UI för enskilt förslag -------
    idx = _robust_nav_state("forslag_idx", len(main))
    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag", key="sug_prev"):
            st.session_state["forslag_idx"] = max(0, idx - 1)
    with col_mid:
        st.write(f"Förslag {idx+1}/{len(main)}")
    with col_next:
        if st.button("➡️ Nästa förslag", key="sug_next"):
            st.session_state["forslag_idx"] = min(len(main)-1, idx + 1)

    rad = main.iloc[_robust_nav_state("forslag_idx", len(main))]

    st.subheader(f"{rad.get('Bolagsnamn','')} ({rad.get('Ticker','')}) — **Score: {round(float(rad['Score']),2)}**")
    vx_label = f"{rad.get('Valuta','')}"
    lines = [
        f"- **Aktuell kurs:** {round(float(rad['Aktuell kurs']),2)} {vx_label}" if has_price else "- **Aktuell kurs:** –",
        f"- **Market cap (nu):** {human_money(rad.get('Market Cap (nu)',0.0), unit=vx_label)}",
        f"- **Utestående aktier:** {round(float(rad.get('Utestående aktier',0.0)),2)} M",
        f"- **P/S (nu):** {round(float(rad.get('P/S',0.0)),2)}" if has_ps else "- **P/S (nu):** –",
        f"- **P/S-snitt (4 kv):** {round(float(rad.get('P/S-snitt',0.0)),2)}" if has_psn else "- **P/S-snitt:** –",
    ]
    if has_target and has_price:
        pot = (float(rad["Riktkurs om 1 år"]) - float(rad["Aktuell kurs"])) / float(rad["Aktuell kurs"]) * 100.0
        lines.append(f"- **Riktkurs om 1 år:** {round(float(rad['Riktkurs om 1 år']),2)} {vx_label}")
        lines.append(f"- **Uppsida (mot 1 år):** {round(pot,2)} %")
    st.markdown("\n".join(lines))

    with st.expander("Visa mer nyckeltal", expanded=False):
        cols = [
            "Ticker","Bolagsnamn","Sektor","Valuta",
            "Market Cap (nu)","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Årlig utdelning","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år",
        ]
        cols = [c for c in cols if c in main.columns]
        st.dataframe(pd.DataFrame([rad[cols].to_dict()]), use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Portfölj
# ------------------------------------------------------------
def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("📦 Min portfölj")

    port = df[df.get("Antal aktier", 0) > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    def _vx(v):
        cur = str(v or "SEK").upper()
        return float(user_rates.get(cur, 1.0))

    port["Växelkurs"] = port["Valuta"].apply(_vx)
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    if total_värde <= 0:
        st.info("Portföljen har inget värde (kontrollera kurser/antal).")
        return

    port["Andel (%)"] = (port["Värde (SEK)"] / total_värde * 100.0).round(2)
    port["Total årlig utdelning (SEK)"] = (port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]).fillna(0.0)
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning (grovt):** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = [
        "Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)",
        "Årlig utdelning","Total årlig utdelning (SEK)","GAV SEK"
    ]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    if "Sektor" in port.columns:
        dist = port.groupby("Sektor", dropna=False)["Värde (SEK)"].sum().sort_values(ascending=False)
        st.subheader("🏷️ Sektorfördelning")
        st.dataframe(dist.rename("Värde (SEK)").to_frame(), use_container_width=True)
        if not dist.empty:
            top_sec, top_val = dist.index[0], float(dist.iloc[0])
            pct = (top_val / total_värde) * 100.0
            if pct > 40:
                st.warning(f"Stor övervikt mot **{top_sec}** ({pct:.1f}%). Överväg diversifiering.")
            elif pct > 30:
                st.info(f"Lätt övervikt mot **{top_sec}** ({pct:.1f}%). Håll koll.")
