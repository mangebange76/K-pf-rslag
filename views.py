# views.py
from __future__ import annotations
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

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
)

# Försök använda din riktiga beräkningsmodul; fallback om den saknas
try:
    from calc_and_cache import uppdatera_berakningar  # pragma: no cover
    _UPD_USES_RATES = True
except Exception:
    _UPD_USES_RATES = False
    def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict | None = None) -> pd.DataFrame:
        """Minimal fallback om calc_and_cache saknas (ignorerar user_rates)."""
        df = df.copy()
        for i, rad in df.iterrows():
            # P/S-snitt
            ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
            ps_clean = [float(x) for x in ps_vals if float(x) > 0]
            ps_snitt = round(float(np.mean(ps_clean)) if ps_clean else 0.0, 2)
            df.at[i, "P/S-snitt"] = ps_snitt

            # CAGR clamp
            cagr = float(rad.get("CAGR 5 år (%)", 0.0))
            just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
            g = just_cagr / 100.0

            # Omsättning om 2 & 3 år
            oms_next = float(rad.get("Omsättning nästa år", 0.0))
            if oms_next > 0:
                df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
                df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)

            # Riktkurser – OBS: både omsättning och utest. aktier är i “miljoner”
            ps_use = ps_snitt if ps_snitt > 0 else float(rad.get("P/S", 0.0))
            aktier_ut_mn = float(rad.get("Utestående aktier", 0.0))
            if aktier_ut_mn > 0 and ps_use > 0:
                df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))     * ps_use) / aktier_ut_mn, 2)
                df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0)) * ps_use) / aktier_ut_mn, 2)
                df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])      * ps_use) / aktier_ut_mn, 2)
                df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])      * ps_use) / aktier_ut_mn, 2)
            else:
                for k in ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]:
                    df.at[i, k] = 0.0
        return df


# ---------- Etikett-hjälpare (visar FY-kvartal, datum och källa) ----------
def _psq_label(row: pd.Series, q: int) -> str:
    d = str(row.get(f"P/S Q{q} datum", "") or "–")
    src = str(row.get(f"Källa P/S Q{q}", "") or "n/a")
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
    return f"P/S (TTM) [{ts}] ({src})" if ts else f"P/S (TTM) ({src})"

def _shares_label(row: pd.Series) -> str:
    ts = str(row.get("TS Utestående aktier", "") or "")
    src = str(row.get("Källa Utestående aktier", "") or "Yahoo/info")
    base = "Utestående aktier (miljoner)"
    if ts and src: return f"{base} [{ts}] ({src})"
    if ts:         return f"{base} [{ts}]"
    if src:        return f"{base} ({src})"
    return base


# ========================== Publika funktioner (som app.py importerar) ==========================

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

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return user_rates


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

            # Basfält
            for k in ["Bolagsnamn","Aktuell kurs","Valuta","Årlig utdelning","CAGR 5 år (%)"]:
                if k in data:
                    df.at[i, k] = data[k]

            # Aktier (miljoner)
            if float(data.get("Utestående aktier", 0) or 0) > 0:
                df.at[i, "Utestående aktier"] = float(data["Utestående aktier"])

            # P/S (TTM) + Q1..Q4 och metadata
            if "P/S" in data:
                df.at[i, "P/S"] = float(data["P/S"])
            for q in (1,2,3,4):
                df.at[i, f"P/S Q{q}"] = float(data.get(f"P/S Q{q}", 0.0) or 0.0)
                df.at[i, f"P/S Q{q} datum"] = data.get(f"P/S Q{q} datum", "")
                df.at[i, f"Källa P/S Q{q}"] = data.get(f"Källa P/S Q{q}", "")
            for k in ["Källa P/S","Källa Utestående aktier","TS P/S","TS Utestående aktier"]:
                if k in data:
                    df.at[i, k] = data.get(k, "")

            if not data.get("Bolagsnamn"):
                miss.append(f"{tkr}: Bolagsnamn")
            if not data.get("Valuta"):
                miss.append(f"{tkr}: Valuta")

            time.sleep(0.3)
            bar.progress((i+1)/max(1,total))

        df = uppdatera_berakningar(df, user_rates) if _UPD_USES_RATES else uppdatera_berakningar(df)
        spara_data(df)
        st.sidebar.success("Klart! Uppdatering slutförd.")
        if miss:
            st.sidebar.warning("Vissa fält saknades:\n" + "\n".join(miss))
    return df


def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

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

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps_ttm, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

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

        data = hamta_yahoo_fält(ticker)
        for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)"]:
            if k in data:
                df.loc[df["Ticker"]==ticker, k] = data[k]
        if float(data.get("Utestående aktier", 0) or 0) > 0:
            df.loc[df["Ticker"]==ticker, "Utestående aktier"] = float(data["Utestående aktier"])
        df.loc[df["Ticker"]==ticker, "P/S"] = float(data.get("P/S", 0.0) or 0.0)
        for q in (1,2,3,4):
            df.loc[df["Ticker"]==ticker, f"P/S Q{q}"] = float(data.get(f"P/S Q{q}", 0.0) or 0.0)
            df.loc[df["Ticker"]==ticker, f"P/S Q{q} datum"] = data.get(f"P/S Q{q} datum", "")
            df.loc[df["Ticker"]==ticker, f"Källa P/S Q{q}"] = data.get(f"Källa P/S Q{q}", "")
        for k in ["Källa P/S","Källa Utestående aktier","TS P/S","TS Utestående aktier"]:
            if k in data:
                df.loc[df["Ticker"]==ticker, k] = data.get(k, "")

        df = uppdatera_berakningar(df, user_rates) if _UPD_USES_RATES else uppdatera_berakningar(df)
        df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo/SEC.")
        st.rerun()

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
        df = uppdatera_berakningar(df, user_rates) if _UPD_USES_RATES else uppdatera_berakningar(df)
        df.loc[df["Ticker"]==tkr, "Senast auto uppdaterad"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        spara_data(df)
        st.success("Hämtat igen.")
        st.rerun()

    # Debug / SEC-länkar
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

    if not bef.empty:
        links = hamta_sec_filing_lankar(bef.get("Ticker",""))
        if links:
            with st.expander("📄 Senaste SEC-filingar (öppna i ny flik)"):
                for it in links:
                    st.markdown(f"- **{it.get('form','')}** {it.get('date','')} — [iXBRL-viewer]({it.get('viewer','')}) · [Arkiv]({it.get('url','')}) · CIK `{it.get('cik','')}`")

    # Lista: äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
                 use_container_width=True)

    return df


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
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "Senast manuellt uppdaterad","Senast auto uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)


def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )


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


# ========================== Hämtlogg (som app.py importerar) ==========================

def visa_hamtlogg_panel() -> None:
    """Liten loggpanel i sidopanelen. Visar rader som data_sources lägger i st.session_state['fetch_logs']."""
    st.sidebar.markdown("### 🧪 Hämtlogg")
    logs = st.session_state.get("fetch_logs", [])
    if not logs:
        st.sidebar.caption("Tom logg.")
        return

    tkr_filter = st.sidebar.text_input("Filter (ticker, valfritt)").strip().upper()
    show = logs
    if tkr_filter:
        show = [x for x in logs if str(x.get("ticker","")).upper().find(tkr_filter) >= 0]

    # visa max 30 senaste
    for item in list(show)[-30:][::-1]:
        ts = item.get("ts","")
        tkr = item.get("ticker","")
        summary = item.get("summary","")
        st.sidebar.caption(f"{ts} · {tkr} — {summary}")


def spara_logg_till_sheets(sheet_name: str = "Hamtlogg") -> None:
    """Skriver st.session_state['fetch_logs'] till ett blad 'Hamtlogg' i samma Google Sheet."""
    logs = st.session_state.get("fetch_logs", [])
    if not logs:
        st.warning("Ingen hämtlogg att spara.")
        return

    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_info = st.secrets.get("GOOGLE_CREDENTIALS", {})
        if not creds_info:
            raise RuntimeError("Saknar GOOGLE_CREDENTIALS i secrets.")
        credentials = Credentials.from_service_account_info(creds_info, scopes=scope)
        client = gspread.authorize(credentials)

        sheet_url = st.secrets.get("SHEET_URL", "")
        if not sheet_url:
            raise RuntimeError("Saknar SHEET_URL i secrets.")
        ss = client.open_by_url(sheet_url)

        try:
            ws = ss.worksheet(sheet_name)
        except Exception:
            ws = ss.add_worksheet(title=sheet_name, rows=2000, cols=10)

        df = pd.DataFrame(logs)
        # Packa meta (ps) som JSON-sträng om det finns
        if "ps" in df.columns:
            df["meta_ps"] = df["ps"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else ("" if x is None else str(x)))
        else:
            df["meta_ps"] = ""

        cols = ["ts", "ticker", "summary", "meta_ps"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        values = [cols] + df[cols].astype(str).values.tolist()

        ws.clear()
        ws.update(values)
        st.success(f"Hämtlogg sparad till bladet '{sheet_name}'.")
    except Exception as e:
        st.error(f"Kunde inte spara hämtlogg: {e}")
