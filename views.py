# views.py
from __future__ import annotations
import time
import json
import streamlit as st
import pandas as pd
import numpy as np

from calc_and_cache import uppdatera_berakningar
from data_sources import (
    hamta_yahoo_fält,
    hamta_live_valutakurser,
    hamta_sec_filing_lankar,
)
from sheets_utils import (
    las_sparade_valutakurser,
    spara_valutakurser,
    spara_data,
    hamta_valutakurs,
    now_stamp,
    save_logs_to_sheet,
    save_proposals_to_sheet,
)

# Fält som triggar "Senast manuellt uppdaterad"
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]


# -------------------------- SIDOPANEL: VALUTOR + SEC --------------------------
def hamta_valutakurser_sidebar() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")
    saved = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved.get("USD", 9.75)), step=0.01, format="%.4f")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved.get("NOK", 0.95)), step=0.01, format="%.4f")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved.get("CAD", 7.05)), step=0.01, format="%.4f")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved.get("EUR", 11.18)), step=0.01, format="%.4f")
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("💾 Spara kurser"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with c2:
        if st.button("🌐 Live-kurser (Yahoo)"):
            live = hamta_live_valutakurser()
            spara_valutakurser(live)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Live-kurser hämtade & sparade.")
            st.rerun()

    with st.sidebar.expander("⚙️ SEC-inställningar"):
        default_cutoff = int(st.secrets.get("SEC_CUTOFF_YEARS", 6))
        default_backfill = str(st.secrets.get("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF", "false")).lower() == "true"
        cutoff = st.number_input("Cutoff (år)", min_value=3, max_value=10,
                                 value=int(st.session_state.get("SEC_CUTOFF_YEARS", default_cutoff)), step=1)
        backfill = st.checkbox("Tillåt backfill äldre kvartal",
                               value=bool(st.session_state.get("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF", default_backfill)))
        st.session_state["SEC_CUTOFF_YEARS"] = int(cutoff)
        st.session_state["SEC_ALLOW_BACKFILL_BEYOND_CUTOFF"] = bool(backfill)
        st.caption("Slå på backfill om Q3–Q4 saknas direkt efter rapportsäsong.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return user_rates


# -------------------------- HÄMTLOGG --------------------------
def _fmt_log_row(r: dict) -> str:
    ps = r.get("ps", {})
    extra = f"ps_source={ps.get('ps_source','')}, q_cols={ps.get('q_cols','?')}, price_hits={ps.get('price_hits','?')}, sec_cik={ps.get('sec_cik','')}, sec_shares_pts={ps.get('sec_shares_pts','')}, sec_rev_pts={ps.get('sec_rev_pts','')}/{ps.get('sec_ixbrl_pts','0')} (cutoff {ps.get('cutoff_years','?')}, backfill={ps.get('backfill_used', False)})"
    return f"[{r.get('ts','')}] {r.get('ticker','?')}: {r.get('summary','')} | {extra}"

def visa_hamtlogg_panel():
    logs = st.session_state.get("fetch_logs", [])
    with st.sidebar.expander("🧾 Hämtlogg (senaste 15)"):
        if not logs:
            st.caption("Inga hämtningar ännu.")
        else:
            for row in reversed(logs[-15:]):
                st.caption(_fmt_log_row(row))

def spara_logg_till_sheets():
    logs = st.session_state.get("fetch_logs", [])
    if not logs:
        st.info("Ingen logg att spara ännu.")
        return
    n = save_logs_to_sheet(logs)
    st.success(f"Sparade {n} loggrader till LOGS.")


# -------------------------- MASSUPPDATERA (SIDOPANEL-KNAPP) -------------------
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo/SEC", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        miss = []
        total = len(df)
        for i, row in df.iterrows():
            tkr = str(row.get("Ticker","")).strip()
            if not tkr:
                continue
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)

            # Applicera fält
            for k in ["Bolagsnamn","Aktuell kurs","Valuta","Årlig utdelning","CAGR 5 år (%)",
                      "Utestående aktier","Källa Aktuell kurs","Källa Utestående aktier","Källa P/S",
                      "Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
                      "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum","P/S"]:
                if k in data:
                    df.at[i, k] = data[k]

            for q in (1,2,3,4):
                col = f"P/S Q{q}"
                if col in data:
                    df.at[i, col] = data[col]

            # tidsstämplar
            ts = now_stamp()
            df.at[i, "TS P/S"] = ts
            df.at[i, "TS Utestående aktier"] = ts
            df.at[i, "Senast auto uppdaterad"] = ts

            bar.progress((i+1)/max(1,total))
            time.sleep(0.2)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Uppdatering slutförd.")
        if miss:
            st.sidebar.warning("Vissa fält saknades:\n" + "\n".join(miss))
    return df


# -------------------------- LÄGG TILL / UPPDATERA -----------------------------
def _label_ts(base: str, ts: str | None, src: str | None) -> str:
    t = ts.strip() if ts else "–"
    s = src.strip() if src else ""
    return f"{base} [{t}]{(' ('+s+')') if s else ''}"

def _get_last_log_for_ticker(ticker: str) -> str:
    logs = st.session_state.get("fetch_logs", [])
    for r in reversed(logs):
        if r.get("ticker","").upper() == ticker.upper():
            return _fmt_log_row(r)
    return "(ingen logg för denna ticker)"

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # sortering för redigering
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum","Bolagsnamn"])
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

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()

            utest_label = _label_ts("Utestående aktier (miljoner)", bef.get("TS Utestående aktier",""), bef.get("Källa Utestående aktier",""))
            utest = st.number_input(utest_label, value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)

            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps_label = _label_ts("P/S (TTM)", bef.get("TS P/S",""), bef.get("Källa P/S",""))
            ps  = st.number_input(ps_label,   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)

            # Kvartal – visa datum + källa bredvid fältnamnet
            q1_lab = f"P/S Q1 (senaste) — {bef.get('P/S Q1 datum','–')} ({bef.get('Källa P/S Q1','') or '–'})"
            ps1 = st.number_input(q1_lab, value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            q2_lab = f"P/S Q2 — {bef.get('P/S Q2 datum','–')} ({bef.get('Källa P/S Q2','') or '–'})"
            ps2 = st.number_input(q2_lab, value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            q3_lab = f"P/S Q3 — {bef.get('P/S Q3 datum','–')} ({bef.get('Källa P/S Q3','') or '–'})"
            ps3 = st.number_input(q3_lab, value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            q4_lab = f"P/S Q4 — {bef.get('P/S Q4 datum','–')} ({bef.get('Källa P/S Q4','') or '–'})"
            ps4 = st.number_input(q4_lab, value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)

        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner) [0.0]",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner) [0.0]", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.caption(f"Aktuell kurskälla: {bef.get('Källa Aktuell kurs','') or '–'}")
            st.caption(f"Senast manuellt uppdaterad: {bef.get('Senast manuellt uppdaterad','') or '–'}")
            st.caption(f"Senast auto uppdaterad: {bef.get('Senast auto uppdaterad','') or '–'}")

            st.markdown("**Uppdateras automatiskt vid spara:** Namn, Valuta, Kurs, Utdelning, CAGR, Utestående aktier, P/S (TTM & Q1..Q4).")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo/SEC")

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            # skapa tomrad med standardkolumner om ny ticker
            tom = {}
            for c in df.columns:
                tom[c] = 0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad",
                                          "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S",
                                          "Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
                                          "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum","TS P/S","TS Utestående aktier","TS Omsättning"] else ""
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta automatik från Yahoo/SEC
        data = hamta_yahoo_fält(ticker)
        for k in data:
            df.loc[df["Ticker"]==ticker, k] = data[k]
        ts = now_stamp()
        df.loc[df["Ticker"]==ticker, "TS P/S"] = ts
        df.loc[df["Ticker"]==ticker, "TS Utestående aktier"] = ts
        df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = ts

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo/SEC.")
        st.rerun()

    # Snabbhämta/refresh-knapp
    if not bef.empty and st.button("🔁 Hämta igen denna ticker (Yahoo/SEC)"):
        data = hamta_yahoo_fält(bef["Ticker"])
        for k in data:
            df.loc[df["Ticker"]==bef["Ticker"], k] = data[k]
        ts = now_stamp()
        df.loc[df["Ticker"]==bef["Ticker"], "TS P/S"] = ts
        df.loc[df["Ticker"]==bef["Ticker"], "TS Utestående aktier"] = ts
        df.loc[df["Ticker"]==bef["Ticker"], "Senast auto uppdaterad"] = ts
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Ticker uppdaterad.")
        st.rerun()

    # Debugrad + SEC-filingar
    if not bef.empty:
        with st.expander("Debug"):
            st.code(_get_last_log_for_ticker(str(bef["Ticker"])))
        with st.expander("📄 Senaste SEC-filingar (öppna i ny flik)"):
            links = hamta_sec_filing_lankar(str(bef["Ticker"]))
            if not links:
                st.caption("Inga länkar hittades.")
            else:
                for L in links:
                    st.markdown(f"- **{L['form']} {L['date']}** — [iXBRL-viewer]({L['viewer']}) · [Arkiv]({L['url']}) · CIK `{L['cik']}`")

    # Lista: Äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
                 use_container_width=True)

    return df


# -------------------------- ANALYSVY ------------------------------------------
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
                                                  value=st.session_state.analys_idx, step=1)
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")

    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
                "TS P/S","TS Utestående aktier","Senast manuellt uppdaterad","Senast auto uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)


# -------------------------- INVESTERINGSFÖRSLAG ------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def _compute_proposals(base: pd.DataFrame, riktkurs_val: str, kapital_sek: float, user_rates: dict) -> pd.DataFrame:
    df = base.copy()
    df["Potential (%)"] = (df[riktkurs_val] - df["Aktuell kurs"]) / df["Aktuell kurs"] * 100.0
    return df

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
                                index=1)

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Cacha beräkningen
    base = _compute_proposals(base, riktkurs_val, kapital_sek, user_rates)

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

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

    # Portföljandelar
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 3 år" else ""}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

    # Spara förslagslistan (hela base) till Google Sheets
    if st.button("💾 Spara nuvarande förslagslista till Google Sheets"):
        # Begränsa kolumner lite
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "Potential (%)"]
        saved = base[cols].copy()
        save_proposals_to_sheet(saved)
        st.success("Förslagslistan sparad till fliken FÖRSLAG.")
