# views.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from sheets_utils import (
    las_sparade_valutakurser, spara_valutakurser, hamta_valutakurs,
    spara_data, spara_forslag_till_sheet, spara_logs_till_sheet, now_stamp
)
from data_sources import (
    hamta_yahoo_fält, hamta_live_valutakurser, hamta_sec_filing_lankar
)
from calc_and_cache import uppdatera_berakningar

# --------- UI: Sidopanel valutakurser + SEC ----------
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

    # SEC-inställningar (UI-override av secrets)
    with st.sidebar.expander("⚙️ SEC-inställningar"):
        default_cutoff = int(st.secrets.get("SEC_CUTOFF_YEARS", 6))
        default_backfill = str(st.secrets.get("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF", "false")).lower() == "true"
        cutoff = st.number_input("Cutoff (år)", min_value=3, max_value=10,
                                 value=int(st.session_state.get("SEC_CUTOFF_YEARS", default_cutoff)), step=1)
        backfill = st.checkbox("Tillåt backfill äldre kvartal",
                               value=bool(st.session_state.get("SEC_ALLOW_BACKFILL_BEYOND_CUTOFF", default_backfill)))
        st.session_state["SEC_CUTOFF_YEARS"] = int(cutoff)
        st.session_state["SEC_ALLOW_BACKFILL_BEYOND_CUTOFF"] = bool(backfill)
        st.caption("Tips: Slå på backfill om Q3/Q4 saknas direkt efter rapportsäsong.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets"):
        st.cache_data.clear()
        st.rerun()

    return user_rates

# --------- Hämtlogg i sidopanel ----------
def visa_hamtlogg_panel(max_rows: int = 8):
    logs = st.session_state.get("fetch_logs", [])
    if not logs:
        return
    with st.sidebar.expander("🧾 Hämtlogg (senast)"):
        df = pd.DataFrame(logs[-max_rows:])
        st.dataframe(df, use_container_width=True, height=200)

def spara_logg_till_sheets():
    logs = st.session_state.get("fetch_logs", [])
    if not logs:
        st.sidebar.info("Ingen hämtlogg att spara ännu.")
        return
    spara_logs_till_sheet(logs)
    st.sidebar.success("Hämtlogg sparad till fliken LOGS.")

# --------- Massuppdatera ----------
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []
        total = len(df)
        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            try:
                data = hamta_yahoo_fält(tkr)

                # Skriv värden
                for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","Utestående aktier","P/S",
                          "P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                          "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"]:
                    if k in data:
                        df.at[i, k] = data[k]

                df.at[i, "TS P/S"] = now_stamp()
                df.at[i, "TS Utestående aktier"] = now_stamp()
                df.at[i, "Senast auto uppdaterad"] = now_stamp()
            except Exception as e:
                misslyckade.append(f"{tkr}: {e}")

            bar.progress((i+1)/total)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa rader misslyckades. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# --------- Form: Lägg till / uppdatera ----------
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"
]

def _etikett_dt(base: str, ts: str, src: str | None = None) -> str:
    ts_txt = f"[{ts}]" if ts else "[–]"
    src_txt = f" ({src})" if src else ""
    return f"{base} {ts_txt}{src_txt}"

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # sort för redigering
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
            utest_lbl = _etikett_dt(
                "Utestående aktier (miljoner)",
                bef.get("TS Utestående aktier","") if not bef.empty else "",
                bef.get("Källa Utestående aktier","") if not bef.empty else None
            )
            utest = st.number_input(utest_lbl, value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)

            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps_lbl = _etikett_dt("P/S (TTM)",
                                 bef.get("TS P/S","") if not bef.empty else "",
                                 bef.get("Källa P/S","") if not bef.empty else None)
            ps  = st.number_input(ps_lbl, value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)

            # Q1..Q4 etiketter + datum/källa
            def qlabel(q):
                d = bef.get(f"P/S Q{q} datum","") if not bef.empty else ""
                src = bef.get(f"Källa P/S Q{q}","") if not bef.empty else ""
                if d and src:
                    return f"P/S Q{q} — {d} ({src})"
                elif d:
                    return f"P/S Q{q} — {d}"
                else:
                    return f"P/S Q{q} — (–)"
            ps1 = st.number_input(qlabel(1), value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input(qlabel(2), value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
        with c2:
            ps3 = st.number_input(qlabel(3), value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input(qlabel(4), value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)

            oms_lbl1 = _etikett_dt("Omsättning idag (miljoner)", bef.get("TS Omsättning","") if not bef.empty else "")
            oms_lbl2 = _etikett_dt("Omsättning nästa år (miljoner)", bef.get("TS Omsättning","") if not bef.empty else "")
            oms_idag  = st.number_input(oms_lbl1,  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input(oms_lbl2, value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.caption(f"Aktuell kurskälla: {bef.get('Källa Aktuell kurs','') if not bef.empty else '–'}")
            st.caption(f"Senast manuellt uppdaterad: {bef.get('Senast manuellt uppdaterad','') if not bef.empty else '–'}")
            st.caption(f"Senast auto uppdaterad: {bef.get('Senast auto uppdaterad','') if not bef.empty else '–'}")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    # Spara
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
            # skapa tom rad med alla kolumner ifyllda
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad",
                                        "TS P/S","TS Utestående aktier","TS Omsättning",
                                        "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
                                        "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum"] else "")
                   for c in df.columns}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = now_stamp()
            df.loc[df["Ticker"]==ticker, "TS Omsättning"] = now_stamp()

        # Auto-hämtning
        data = hamta_yahoo_fält(ticker)
        for k in data:
            if k in df.columns:
                df.loc[df["Ticker"]==ticker, k] = data[k]

        # Tidsstämplar
        df.loc[df["Ticker"]==ticker, "TS P/S"] = now_stamp()
        df.loc[df["Ticker"]==ticker, "TS Utestående aktier"] = now_stamp()
        df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = now_stamp()

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo/SEC.")
        st.rerun()

    # SEC-filingar som snabb-länkar
    if not bef.empty:
        with st.expander("📄 Senaste SEC-filingar (öppna i ny flik)"):
            links = hamta_sec_filing_lankar(bef.get("Ticker",""))
            if not links:
                st.caption("Inga filer hittades.")
            else:
                for L in links:
                    st.markdown(f"- **{L['form']} {L['date']}** — [iXBRL-viewer]({L['viewer']}) · [Arkiv]({L['url']}) · CIK `{L['cik']}`")

    # Topp 10 äldst manuellt uppdaterade (Omsättning)
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","Omsättning idag","Omsättning nästa år",
              "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]],
        use_container_width=True
    )

    return df

# --------- Analysvy ----------
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
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
                "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om
