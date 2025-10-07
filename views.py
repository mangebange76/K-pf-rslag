# views.py
import streamlit as st
import pandas as pd
from sheets_utils import (
    las_sparade_valutakurser, spara_valutakurser,
    hamta_valutakurs, get_spreadsheet, now_stamp,
    skapa_snapshot_om_saknas, spara_data
)
from data_sources import hamta_yahoo_fält, hamta_live_valutakurser
from calc_and_cache import uppdatera_berakningar, bygg_forslag_cache

MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

def _fmt_date_label(val: str) -> str:
    bad = {"", "0", "0.0", "nan", "NaT", "None"}
    if val is None: return "–"
    s = str(val).strip()
    return "–" if s in bad else s

def _fmt_src(val: str) -> str:
    s = str(val or "").strip()
    return "–" if s in {"", "0", "0.0", "nan", "None"} else s

# --------- Hämtlogg i sidopanelen ----------
def visa_hamtlogg_panel():
    with st.sidebar.expander("🔎 Hämtlogg (senaste körning)"):
        last = (st.session_state.get("fetch_logs") or [])[-1:]
        if not last:
            st.write("Ingen logg ännu.")
        else:
            import json
            e = last[0]
            st.markdown(f"**Ticker:** `{e.get('ticker','?')}`  \n**Tid:** {e.get('ts','?')}")
            st.code(json.dumps(e, indent=2, ensure_ascii=False), language="json")

def spara_logg_till_sheets():
    logs = st.session_state.get("fetch_logs", [])
    if not logs:
        st.sidebar.info("Ingen logg att spara."); return
    try:
        ss = get_spreadsheet()
        try:
            ws = ss.worksheet("LOGS")
        except Exception:
            ss.add_worksheet(title="LOGS", rows=2000, cols=10)
            ws = ss.worksheet("LOGS")
            ws.update("A1", [["ts","ticker","summary","raw_json"]])
        import json
        rows = [[e.get("ts",""), e.get("ticker",""), e.get("summary",""), json.dumps(e, ensure_ascii=False)] for e in logs]
        ws.append_rows(rows)
        st.sidebar.success(f"Sparade {len(rows)} loggrader till LOGS.")
    except Exception as ex:
        st.sidebar.warning(f"Kunde inte spara logg: {ex}")

# --------- Valutakurser i sidopanelen ----------
def hamta_valutakurser_sidebar():
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

# --------- Massuppdatering & snapshot ----------
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    colA, colB = st.sidebar.columns(2)
    with colA:
        create_snap = st.button("📸 Skapa snapshot nu")
    with colB:
        do_mass = st.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn")

    if create_snap:
        ok, msg = skapa_snapshot_om_saknas(df)
        (st.sidebar.success if ok else st.sidebar.warning)(msg)

    if do_mass:
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []
        total = len(df)
        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")

            data = hamta_yahoo_fält(tkr)
            failed = []

            if "Bolagsnamn" in data: df.at[i, "Bolagsnamn"] = data.get("Bolagsnamn") or ""
            if "Valuta" in data:     df.at[i, "Valuta"] = data.get("Valuta") or ""
            if "Aktuell kurs" in data: df.at[i, "Aktuell kurs"] = float(data.get("Aktuell kurs") or 0.0)
            if "Årlig utdelning" in data: df.at[i, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            if "CAGR 5 år (%)" in data:   df.at[i, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

            # Nollställ P/S-fält innan ny skrivning
            for k in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                df.at[i, k] = 0.0
            for k in ["P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                      "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"]:
                df.at[i, k] = ""

            for k in ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                if k in data:
                    df.at[i, k] = float(data.get(k) or 0.0)
            for k in ["P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                      "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"]:
                if k in data:
                    df.at[i, k] = str(data.get(k) or "")

            df.at[i, "TS P/S"] = now_stamp()
            df.at[i, "TS Utestående aktier"] = now_stamp()
            df.at[i, "Senast auto uppdaterad"] = now_stamp()

            import time; time.sleep(1.0)
            bar.progress((i+1)/total)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
    return df

# --------- Lägg till / uppdatera ----------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00 00:00")
        vis_df = df.sort_values(by=["_sort_datum","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"): st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"): st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    ts_ps  = str(bef.get("TS P/S","–")) if not bef.empty else "–"
    ts_ut  = str(bef.get("TS Utestående aktier","–")) if not bef.empty else "–"
    ts_oms = str(bef.get("TS Omsättning","–")) if not bef.empty else "–"

    src_px  = bef.get("Källa Aktuell kurs","") if not bef.empty else ""
    src_ut  = bef.get("Källa Utestående aktier","") if not bef.empty else ""
    src_ps  = bef.get("Källa P/S","") if not bef.empty else ""
    src_ps1 = bef.get("Källa P/S Q1","") if not bef.empty else ""
    src_ps2 = bef.get("Källa P/S Q2","") if not bef.empty else ""
    src_ps3 = bef.get("Källa P/S Q3","") if not bef.empty else ""
    src_ps4 = bef.get("Källa P/S Q4","") if not bef.empty else ""

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input(f"Utestående aktier (miljoner) [{ts_ut}] ({_fmt_src(src_ut)})",
                                    value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger",
                                    value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            ps  = st.number_input(f"P/S (TTM) [{ts_ps}] ({_fmt_src(src_ps)})",
                                  value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input(f"P/S Q1 (senaste) — {_fmt_date_label(bef.get('P/S Q1 datum',''))} ({_fmt_src(src_ps1)})",
                                  value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input(f"P/S Q2 — {_fmt_date_label(bef.get('P/S Q2 datum',''))} ({_fmt_src(src_ps2)})",
                                  value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input(f"P/S Q3 — {_fmt_date_label(bef.get('P/S Q3 datum',''))} ({_fmt_src(src_ps3)})",
                                  value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input(f"P/S Q4 — {_fmt_date_label(bef.get('P/S Q4 datum',''))} ({_fmt_src(src_ps4)})",
                                  value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input(f"Omsättning idag (miljoner) [{ts_oms}]",
                                        value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input(f"Omsättning nästa år (miljoner) [{ts_oms}]",
                                        value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            st.caption(f"Aktuell kurskälla: {_fmt_src(src_px)}")
            st.caption(f"Senast manuellt uppdaterad: {bef.get('Senast manuellt uppdaterad','–') if not bef.empty else '–'}")
            st.caption(f"Senast auto uppdaterad: {bef.get('Senast auto uppdaterad','–') if not bef.empty else '–'}")
            st.markdown("**Uppdateras automatiskt vid spara:** Namn, Valuta, Kurs, Utdelning, CAGR, Utestående aktier, P/S (TTM & Q1..Q4).")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    # 🔁 Hämta igen (rensar cache + nollställer P/S)
    if valt_label and (bef.get("Ticker") or "").strip() and st.button("🔁 Hämta igen denna ticker (Yahoo/SEC)"):
        st.cache_data.clear()
        ticker = bef.get("Ticker")

        for k in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
            df.loc[df["Ticker"]==ticker, k] = 0.0
        for k in ["P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                  "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"]:
            df.loc[df["Ticker"]==ticker, k] = ""

        data = hamta_yahoo_fält(ticker)

        for k in ["Bolagsnamn","Valuta","Källa Aktuell kurs","Källa Utestående aktier",
                  "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
                  "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum"]:
            if k in data: df.loc[df["Ticker"]==ticker, k] = str(data.get(k) or "")

        for k in ["Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","Utestående aktier",
                  "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
            if k in data: df.loc[df["Ticker"]==ticker, k] = float(data.get(k) or 0.0)

        now = now_stamp()
        df.loc[df["Ticker"]==ticker, "TS P/S"] = now
        df.loc[df["Ticker"]==ticker, "TS Utestående aktier"] = now
        df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = now

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.rerun()

    if spar:
        ticker = (ticker or "").upper().strip()
        if ticker:
            ny = {
                "Ticker": ticker, "Utestående aktier": float(utest), "Antal aktier": float(antal),
                "P/S": float(ps), "P/S Q1": float(ps1), "P/S Q2": float(ps2), "P/S Q3": float(ps3), "P/S Q4": float(ps4),
                "Omsättning idag": float(oms_idag), "Omsättning nästa år": float(oms_next)
            }
            datum_sätt = False
            if not bef.empty:
                before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
                after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
                if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM): datum_sätt = True
            else:
                if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM): datum_sätt = True

            if not bef.empty:
                for k,v in ny.items(): df.loc[df["Ticker"]==ticker, k] = v
            else:
                tom_init_cols = [
                    "Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto uppdaterad",
                    "TS P/S","TS Omsättning","TS Utestående aktier",
                    "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                    "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"
                ]
                tom = {c: (0.0 if c not in tom_init_cols else "") for c in df.columns}
                tom.update(ny)
                df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

            if datum_sätt:
                ts = now_stamp()
                df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = ts
                df.loc[df["Ticker"]==ticker, "TS Omsättning"] = ts

            # nollställ P/S relaterat först
            for k in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                df.loc[df["Ticker"]==ticker, k] = 0.0
            for k in ["P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                      "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4"]:
                df.loc[df["Ticker"]==ticker, k] = ""

            data = hamta_yahoo_fält(ticker)

            for k in ["Bolagsnamn","Valuta","Källa Aktuell kurs","Källa Utestående aktier",
                      "Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
                      "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum"]:
                if k in data: df.loc[df["Ticker"]==ticker, k] = str(data.get(k) or "")

            for k in ["Aktuell kurs","Årlig utdelning","CAGR 5 år (%)","Utestående aktier",
                      "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                if k in data: df.loc[df["Ticker"]==ticker, k] = float(data.get(k) or 0.0)

            now = now_stamp()
            df.loc[df["Ticker"]==ticker, "TS Utestående aktier"] = now
            df.loc[df["Ticker"]==ticker, "TS P/S"] = now
            df.loc[df["Ticker"]==ticker, "Senast auto uppdaterad"] = now

            df = uppdatera_berakningar(df, user_rates)
            spara_data(df)
            st.success("Sparat och uppdaterat från Yahoo.")
            st.rerun()

    # Debugrad från logg
    if not bef.empty and bef.get("Ticker"):
        logs = st.session_state.get("fetch_logs") or []
        last_for_tk = next((e for e in reversed(logs) if e.get("ticker","").upper()==str(bef.get("Ticker")).upper()), None)
        if last_for_tk:
            psdbg = last_for_tk.get("ps", {})
            st.caption(
                f"Debug: ps_source={psdbg.get('ps_source','-')}, q_cols={psdbg.get('q_cols',0)}, "
                f"price_hits={psdbg.get('price_hits',0)}, sec_cik={psdbg.get('sec_cik') or '–'}, "
                f"sec_shares_pts={psdbg.get('sec_shares_pts',0)}, "
                f"sec_rev_pts={psdbg.get('sec_rev_pts',0)}/{psdbg.get('sec_rev_pts_after_cutoff',0)} "
                f"(cutoff {psdbg.get('cutoff_years',6)}y)"
            )

    st.markdown("### ⏱️ Äldst manuellt uppdaterade (Omsättning)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00 00:00")
    filt = (df["Omsättning idag"] > 0) | (df["Omsättning nästa år"] > 0)
    tips = df[filt].sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","Omsättning idag","Omsättning nästa år"]], use_container_width=True)
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
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
                "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "P/S Q1 datum","P/S Q2 datum","P/S Q3 datum","P/S Q4 datum",
                "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
                "TS Utestående aktier","TS P/S","TS Omsättning",
                "Källa Aktuell kurs","Källa Utestående aktier","Källa P/S","Källa P/S Q1","Källa P/S Q2","Källa P/S Q3","Källa P/S Q4",
                "Senast manuellt uppdaterad","Senast auto uppdaterad"]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

# --------- Investeringsförslag ----------
def spara_forslag_till_sheets(base: pd.DataFrame, meta: dict):
    try:
        ss = get_spreadsheet()
        try:
            ws = ss.worksheet("FÖRSLAG")
        except Exception:
            ss.add_worksheet(title="FÖRSLAG", rows=5000, cols=20)
            ws = ss.worksheet("FÖRSLAG")
            ws.update("A1", [[
                "ts","riktkurs","subset","kapital_sek",
                "rank","Ticker","Bolagsnamn","Aktuell kurs","Valuta",
                "Riktkurs","Potential (%)","Diff till mål (%)"
            ]])
        rows = []
        rk_col = meta["riktkurs_val"]
        for i, r in base.reset_index(drop=True).iterrows():
            rows.append([
                now_stamp(), meta["riktkurs_val"], meta["subset"], meta["kapital_sek"],
                i+1, r["Ticker"], r["Bolagsnamn"], float(r["Aktuell kurs"]), r["Valuta"],
                float(r[rk_col]), float(r["Potential (%)"]), float(r["Diff till mål (%)"])
            ])
        ws.append_rows(rows)
        st.success(f"Sparade {len(rows)} förslag till 'FÖRSLAG'.")
    except Exception as e:
        st.warning(f"Kunde inte spara förslag: {e}")

def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?", ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"], index=1)
    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)

    df_ser = df.to_json(orient="split")
    base = bygg_forslag_cache(df_ser, riktkurs_val, subset, kapital_sek)

    if st.button("↻ Uppdatera förslag (rebuild)"):
        bygg_forslag_cache.clear()
        base = bygg_forslag_cache(df_ser, riktkurs_val, subset, kapital_sek)

    if base.empty:
        st.info("Inga bolag matchar just nu."); return

    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)
    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    if "forslags_index" not in st.session_state: st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"): st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"): st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

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
        if not r.empty: nuv_innehav = float(r["Värde (SEK)"].sum())
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

    if st.button("🧾 Spara dessa förslag till Sheets"):
        spara_forslag_till_sheets(base, {"riktkurs_val": riktkurs_val, "subset": subset, "kapital_sek": kapital_sek})
