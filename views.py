# views.py
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import streamlit as st

from data_sources import (
    hamta_yahoo_fält,
    hamta_live_valutakurser,
    hamta_sec_filing_lankar,
)
from sheets_utils import (
    hamta_data,
    spara_data,
    las_sparade_valutakurser,
    spara_valutakurser,
    säkerställ_kolumner,
    migrera_gamla_riktkurskolumner,
    konvertera_typer,
    uppdatera_berakningar,
    hamta_valutakurs,
)

# ------------------------------------------------------------
# Hjälpare för etiketter
# ------------------------------------------------------------
def _psq_label(row: pd.Series, q: int) -> str:
    """Bygg label för P/S Q1..Q4 med FY-kvartal + datum + källa."""
    d = str(row.get(f"P/S Q{q} datum", "") or "–")
    src = str(row.get(f"Källa P/S Q{q}", "") or "n/a")
    # Försök plocka ut 'FYxx Qy' ur källsträngen, ex: 'Computed/FY25 Q2/price@2025-08-28'
    fy = ""
    if src.startswith("Computed/"):
        parts = src.split("/")
        if len(parts) >= 2 and parts[1].startswith("FY"):
            fy = parts[1]
    if fy:
        return f"P/S Q{q} ({fy}) — {d}"
    return f"P/S Q{q} — {d} ({src})"

def _ps_ttm_label(row: pd.Series) -> str:
    ts = str(row.get("TS P/S", "") or "")
    src = str(row.get("Källa P/S", "") or "Yahoo/ps_ttm")
    if ts:
        return f"P/S (TTM) [{ts}] ({src})"
    return f"P/S (TTM) ({src})"

def _shares_label(row: pd.Series) -> str:
    ts = str(row.get("TS Utestående aktier", "") or "")
    src = str(row.get("Källa Utestående aktier", "") or "Yahoo/info")
    base = "Utestående aktier (miljoner)"
    if ts and src:
        return f"{base} [{ts}] ({src})"
    if ts:
        return f"{base} [{ts}]"
    if src:
        return f"{base} ({src})"
    return base

# ------------------------------------------------------------
# Massuppdatera – oförändrad logik, men låt Yahoo/SEC hämta P/S TTMs & Q1..Q4
# ------------------------------------------------------------
def massuppdatera(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo/SEC", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        miss = []
        total = len(df)
        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)

            # Bas
            for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)"]:
                if k in data:
                    df.at[i, k] = data[k] or df.at[i, k]

            # Aktier (lagras i miljoner)
            if float(data.get("Utestående aktier", 0) or 0) > 0:
                df.at[i, "Utestående aktier"] = float(data["Utestående aktier"])

            # P/S (TTM) + Q1..Q4 och metadata
            if "P/S" in data:
                df.at[i, "P/S"] = float(data["P/S"])
            for q in (1,2,3,4):
                df.at[i, f"P/S Q{q}"] = float(data.get(f"P/S Q{q}", 0.0) or 0.0)
                df.at[i, f"P/S Q{q} datum"] = data.get(f"P/S Q{q} datum", "")
                df.at[i, f"Källa P/S Q{q}"] = data.get(f"Källa P/S Q{q}", "")

            # Källor + tidsstämplar
            for k in ["Källa P/S","Källa Utestående aktier","TS P/S","TS Utestående aktier"]:
                if k in data:
                    df.at[i, k] = data.get(k, "")

            # Felbokning
            if not data.get("Bolagsnamn"):
                miss.append(f"{tkr}: Bolagsnamn")
            if not data.get("Valuta"):
                miss.append(f"{tkr}: Valuta")

            time.sleep(0.4)
            bar.progress((i+1)/total)

        # Räkna om och spara
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if miss:
            st.sidebar.warning("Vissa fält saknades. Kopiera nedan:")
            st.sidebar.text_area("Saknade fält", "\n".join(miss), height=160, key=f"{key_prefix}_miss")

    return df

# ------------------------------------------------------------
# Lägg till / uppdatera bolag – UI med FY-etiketter
# ------------------------------------------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sortval
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0

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
            utest = st.number_input(_shares_label(bef), value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps_ttm = st.number_input(_ps_ttm_label(bef), value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, key="ps_ttm")
            ps1 = st.number_input(_psq_label(bef, 1), value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, key="ps_q1")
            ps2 = st.number_input(_psq_label(bef, 2), value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, key="ps_q2")
        with c2:
            ps3 = st.number_input(_psq_label(bef, 3), value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, key="ps_q3")
            ps4 = st.number_input(_psq_label(bef, 4), value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, key="ps_q4")

            oms_idag  = st.number_input("Omsättning idag (miljoner) [0.0]",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner) [0.0]", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.caption("Aktuell kurskälla: " + (bef.get("Källa Aktuell kurs","Yahoo/info") or "Yahoo/info"))
            st.caption("Senast manuellt uppdaterad: " + (bef.get("Senast manuellt uppdaterad","") or "—"))
            st.caption("Senast auto uppdaterad: " + (bef.get("Senast auto uppdaterad","") or "—"))

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo/SEC")

    # Spara + hämta
    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps_ttm, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Sätt datum för manuell uppdatering om omsättnings/P-S-fält ändrats
        MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
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
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad","Källa Aktuell kurs"] else "") for c in df.columns}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

        # Hämta från källor (Yahoo/SEC)
        data = hamta_yahoo_fält(ticker)
        # Bas
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)"]:
            if k in data:
                df.loc[df["Ticker"]==ticker, k] = data[k]
        # Aktier (mn)
        if float(data.get("Utestående aktier", 0) or 0) > 0:
            df.loc[df["Ticker"]==ticker, "Utestående aktier"] = float(data["Utestående aktier"])
        # P/S
        df.loc[df["Ticker"]==ticker, "P/S"] = float(data.get("P/S", 0.0) or 0.0)
        for q in (1,2,3,4):
            df.loc[df["Ticker"]==ticker, f"P/S Q{q}"] = float(data.get(f"P/S Q{q}", 0.0) or 0.0)
            df.loc[df["Ticker"]==ticker, f"P/S Q{q} datum"] = data.get(f"P/S Q{q} datum", "")
            df.loc[df["Ticker"]==ticker, f"Källa P/S Q{q}"] = data.get(f"Källa P/S Q{q}", "")
        # källor + tidsstämplar
        for k in ["Källa P/S","Källa Utestående aktier","TS P/S","TS Utestående aktier"]:
            if k in data:
                df.loc[df["Ticker"]==ticker, k] = data.get(k, "")

        # uppdatera & spara
        df = uppdatera_berakningar(df)
        df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo/SEC.")

    # “Hämta igen” för vald ticker
    if not bef.empty and st.button("↻ Hämta igen denna ticker (Yahoo/SEC)"):
        tkr = bef.get("Ticker","")
        data = hamta_yahoo_fält(tkr)
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)"]:
            if k in data:
                df.loc[df["Ticker"]==tkr, k] = data[k]
        if float(data.get("Utestående aktier",0) or 0) > 0:
            df.loc[df["Ticker"]==tkr, "Utestående aktier"] = float(data["Utestående aktier"])
        df.loc[df["Ticker"]==tkr, "P/S"] = float(data.get("P/S", 0.0) or 0.0)
        for q in (1,2,3,4):
            df.loc[df["Ticker"]==tkr, f"P/S Q{q}"] = float(data.get(f"P/S Q{q}", 0.0) or 0.0)
            df.loc[df["Ticker"]==tkr, f"P/S Q{q} datum"] = data.get(f"P/S Q{q} datum", "")
            df.loc[df["Ticker"]==tkr, f"Källa P/S Q{q}"] = data.get(f"Källa P/S Q{q}", "")
        for k in ["Källa P/S","Källa Utestående aktier","TS P/S","TS Utestående aktier"]:
            if k in data:
                df.loc[df["Ticker"]==tkr, k] = data.get(k, "")
        df = uppdatera_berakningar(df)
        df.loc[df["Ticker"]==tkr, "Senast auto uppdaterad"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        spara_data(df)
        st.success("Hämtat igen.")

    # Debug / logg
    if "fetch_logs" in st.session_state and not bef.empty:
        tkr = bef.get("Ticker","").upper()
        logs = [x for x in st.session_state["fetch_logs"] if x.get("ticker","") == tkr]
        if logs:
            last = logs[-1]
            meta = last.get("ps", {})
            q_cols = meta.get("q_cols","?")
            sec_pts = meta.get("sec_shares_pts","?")
            sec_cik = meta.get("sec_cik","")
            st.caption(f"Debug: ps_source={meta.get('ps_source','?')}, qrev_pts={q_cols}, sec_cik={sec_cik}, sec_shares_pts={sec_pts}")

    # SEC-länkar (om finns i secrets)
    if not bef.empty:
        links = hamta_sec_filing_lankar(bef.get("Ticker",""))
        if links:
            with st.expander("📄 Senaste SEC-filingar (öppna i ny flik)"):
                for it in links:
                    st.markdown(f"- **{it.get('form','')}** {it.get('date','')} — [iXBRL-viewer]({it.get('viewer','')}) · [Arkiv]({it.get('url','')}) · CIK `{it.get('cik','')}`")

    # Topp: äldst manuella omsättningar
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]], use_container_width=True)

    return df

# ------------------------------------------------------------
# Övriga vyer (kortade, oförändrad kärnlogik)
# ------------------------------------------------------------
def analysvy(df: pd.DataFrame) -> None:
    st.header("📈 Analys")
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state: st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1)
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
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "Omsättning idag","Omsättning nästa år","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "Senast manuellt uppdaterad","Senast auto uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

def visa_portfolj(df: pd.DataFrame) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Valuta"].map(lambda v: hamta_valutakurs(v))
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Valuta"].map(lambda v: hamta_valutakurs(v))
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")
    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

def visa_investeringsforslag(df: pd.DataFrame) -> None:
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

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
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
    vx = 1.0  # om du har valutakonverter här, anropa hamta_valutakurs(rad["Valuta"])
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs (vald):** {round(rad[riktkurs_val],2)} {rad['Valuta']}
- **Uppsida:** {round((rad[riktkurs_val]-rad['Aktuell kurs'])/rad['Aktuell kurs']*100.0,2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
"""
    )
