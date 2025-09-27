# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np

from .config import FINAL_COLS, TS_FIELDS
from .utils import (säkerställ_kolumner, konvertera_typer, add_oldest_ts_col,
                    build_requires_manual_df, hamta_valutakurs, make_pretty_money, risklabel_from_mcap_sek, now_stamp)
from .calc import recompute_all, apply_auto_updates_to_row
from .fetchers import auto_fetch_for_ticker
from .scoring import score_growth, score_dividend, valuation_label

# ---------- KONTROLL ----------
def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")
    # Äldst uppdaterade (alla spårade fält)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy(), TS_FIELDS)
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år","_oldest_any_ts"]:
        if k in vis.columns: cols_show.append(k)
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

# ---------- ANALYS ----------
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Inga bolag i databasen ännu."); return
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    options = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0,len(options)-1), value=0, step=1)
    st.selectbox("Eller välj i lista", options, index=idx if options else 0, key="analys_select")
    r = vis_df.iloc[idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Årlig utdelning","Sector","Industry","Debt/Equity","Gross Margin (%)","Net Margin (%)",
        "Market Cap (valuta)","Market Cap (SEK)","MCAP Q1","MCAP Q2","MCAP Q3","MCAP Q4","Cash & Equivalents","Free Cash Flow","Runway (quarters)","GAV SEK",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# ---------- LÄGG TILL / UPPDATERA ----------
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict, save_cb) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Manuell prognoslista (efterfrågad under denna vy)
    with st.expander("📋 Manuell prognoslista (äldre/obefintliga 'Omsättning idag/ nästa år')", expanded=False):
        older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30, key="manu_older_days")
        need = build_requires_manual_df(df, older_than_days=int(older_days), ts_map=TS_FIELDS)
        if need.empty:
            st.success("Inga uppenbara kandidater för manuell uppdatering just nu.")
        else:
            st.warning(f"{len(need)} bolag kan behöva manuell uppdatering:")
            st.dataframe(need, use_container_width=True, hide_index=True)

    # Sortering för edit-lista
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], key="edit_sort")
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy(), TS_FIELDS)
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    sel = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=0, key="edit_select")
    bef = df[df["Ticker"] == namn_map.get(sel,"")].iloc[0] if sel and sel in namn_map else pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav   = st.number_input("GAV i SEK", value=float(bef.get("GAV SEK",0.0)) if not bef.empty else 0.0)
            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            st.caption("Vid spara uppdateras även: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning (via Yahoo). Riktkurser räknas om.")

        spar = st.form_submit_button("💾 Spara")

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV SEK": gav,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }
        # skriv
        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # sätt manuell TS om relevanta fält ändrats/angivits
        ridx = df.index[df["Ticker"]==ticker][0]
        df.loc[ridx, "Senast manuellt uppdaterad"] = now_stamp()
        for f in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]:
            ts = TS_FIELDS.get(f)
            if ts: df.loc[ridx, ts] = now_stamp()

        # hämta basfält via auto-fetch med soft mode (bara basics)
        vals, dbg = auto_fetch_for_ticker(ticker)
        apply_auto_updates_to_row(df, ridx, {k: vals[k] for k in ["Bolagsnamn","Valuta","Aktuell kurs","Årlig utdelning"] if k in vals}, source="Form Yahoo-basics", force_stamp_ts=False)

        # räkna om lokalt och spara
        df = recompute_all(df, user_rates)
        if save_cb: save_cb(df)
        st.success("Sparat.")

    # Singel-uppdatering
    st.subheader("⚙️ Enskild uppdatering")
    col1, col2 = st.columns(2)
    tick = st.text_input("Ticker att uppdatera", value="", key="single_update_tkr").upper()
    with col1:
        if st.button("🔁 Uppdatera kurs (endast pris)", key="btn_px_only") and tick:
            vals, dbg = auto_fetch_for_ticker(tick)
            mask = (df["Ticker"].astype(str).str.upper()==tick)
            if not mask.any():
                st.error(f"{tick} hittades inte i tabellen.")
            else:
                ridx = df.index[mask][0]
                fields = {}
                if "Aktuell kurs" in vals: fields["Aktuell kurs"] = vals["Aktuell kurs"]
                apply_auto_updates_to_row(df, ridx, fields, source="Kurs (Yahoo)", force_stamp_ts=True)
                df = recompute_all(df, user_rates)
                if save_cb: save_cb(df)
                st.success("Kurs uppdaterad. (1/1)")

    with col2:
        if st.button("🚀 Full auto för denna", key="btn_auto_one") and tick:
            vals, dbg = auto_fetch_for_ticker(tick)
            mask = (df["Ticker"].astype(str).str.upper()==tick)
            if not mask.any():
                st.error(f"{tick} hittades inte i tabellen.")
            else:
                ridx = df.index[mask][0]
                apply_auto_updates_to_row(df, ridx, vals, source="Auto (SEC/Yahoo→Yahoo→FMP)", force_stamp_ts=True)
                df = recompute_all(df, user_rates)
                if save_cb: save_cb(df)
                st.success("Full auto uppdaterad. (1/1)")

    return df

# ---------- INVESTERINGSFÖRSLAG ----------
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    if df.empty:
        st.info("Inga bolag."); return

    # Val: Growth / Dividend
    mode = st.radio("Fokus", ["Tillväxt (Growth)","Utdelning (Dividend)"], horizontal=True)
    # Filter sektor + cap
    sektorer = ["Alla"] + sorted([s for s in df["Sector"].dropna().unique() if str(s).strip()])
    sektor = st.selectbox("Filtrera sektor", sektorer)
    caps = ["Alla","Nano","Micro","Small","Mid","Large","Mega"]
    capf = st.selectbox("Filtrera cap-klass", caps)

    # Räkna om score
    base = df.copy()
    base = base[(base["Aktuell kurs"] > 0) & (base["Riktkurs om 1 år"] >= 0)]
    # Risklabel från mcap SEK
    if "Market Cap (SEK)" not in base.columns:
        base["Market Cap (SEK)"] = 0.0
    base["_cap_label"] = base["Market Cap (SEK)"].apply(risklabel_from_mcap_sek)

    if sektor != "Alla":
        base = base[base["Sector"] == sektor]
    if capf != "Alla":
        base = base[base["_cap_label"] == capf]

    if base.empty:
        st.info("Inga bolag matchar filter."); return

    if mode.startswith("Tillväxt"):
        base = score_growth(base)
        base["_rank_score"] = base["_growth_score"]
        sort_desc = True
    else:
        base = score_dividend(base)
        base["_rank_score"] = base["_div_score"]
        sort_desc = True

    base = base.sort_values(by="_rank_score", ascending=not sort_desc).reset_index(drop=True)

    # Förslags-navigering (stabil)
    if "forslags_idx" not in st.session_state: st.session_state.forslags_idx = 0
    st.session_state.forslags_idx = max(0, min(st.session_state.forslags_idx, len(base)-1))
    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_idx = max(0, st.session_state.forslags_idx - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_idx+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_idx = min(len(base)-1, st.session_state.forslags_idx + 1)

    r = base.iloc[st.session_state.forslags_idx]

    # Beräkna extra visningar
    vx = hamta_valutakurs(r["Valuta"], user_rates)
    ps_now = float(r.get("P/S",0.0)); ps_avg = float(r.get("P/S-snitt",0.0))
    mcap_now = float(r.get("Market Cap (valuta)",0.0))
    shares_m = float(r.get("Utestående aktier",0.0))
    uppsida = ((float(r.get("Riktkurs om 1 år",0.0)) - float(r.get("Aktuell kurs",0.0))) / max(float(r.get("Aktuell kurs",0.0)), 1e-9)) * 100.0
    label = valuation_label(r)

    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    lines = [
        f"- **Aktuell kurs:** {round(float(r['Aktuell kurs']),2)} {r['Valuta']}",
        f"- **Riktkurs om 1 år:** {round(float(r['Riktkurs om 1 år']),2)} {r['Valuta']}",
        f"- **Uppsida:** {uppsida:.1f} %",
        f"- **Utestående aktier:** {shares_m:.2f} M",
        f"- **Marketcap nu:** {make_pretty_money(mcap_now, r['Valuta'])}",
        f"- **P/S nu:** {ps_now:.2f}",
        f"- **P/S-snitt (Q1–Q4):** {ps_avg:.2f}",
        f"- **Etikett:** {label}",
    ]
    st.markdown("\n".join(lines))

    with st.expander("🔎 Detaljer & historik", expanded=False):
        mcaps = [("MCAP Q1", r.get("MCAP Q1",0.0)), ("MCAP Q2", r.get("MCAP Q2",0.0)),
                 ("MCAP Q3", r.get("MCAP Q3",0.0)), ("MCAP Q4", r.get("MCAP Q4",0.0))]
        cols = st.columns(4)
        for i,(k,v) in enumerate(mcaps):
            cols[i].metric(k, make_pretty_money(v, r["Valuta"]))
        st.write(f"**Sector/Industry:** {r.get('Sector','-')} / {r.get('Industry','-')}")
        st.write(f"**D/E:** {r.get('Debt/Equity',0.0):.2f} • **Gross margin:** {r.get('Gross Margin (%)',0.0):.1f}% • **Net margin:** {r.get('Net Margin (%)',0.0):.1f}%")
        st.write(f"**Cash:** {make_pretty_money(r.get('Cash & Equivalents',0.0), r['Valuta'])} • **FCF:** {make_pretty_money(r.get('Free Cash Flow',0.0), r['Valuta'])} • **Runway:** {r.get('Runway (quarters)',0.0):.1f} kvartal")

# ---------- PORTFÖLJ + SÄLJVAKT ----------
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier."); return
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)","GAV SEK"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)

    # Säljvakt
    st.subheader("🚨 Säljvakt (trim/sälj-kandidater)")
    # En enkel heuristik: etikett från valuation_label samt GAV & uppsida
    def _flag(row):
        lab = valuation_label(row)
        if lab.startswith("Övervärderad"):
            # om pris långt över riktkurs och stor andel
            if row.get("Andel (%)",0) >= 8.0: return True, lab
        return False, lab
    flags = []
    for _, r in port.iterrows():
        f, lab = _flag(r)
        if f:
            flags.append({
                "Ticker": r["Ticker"], "Bolagsnamn": r["Bolagsnamn"], "Etikett": lab,
                "Andel (%)": r["Andel (%)"], "Aktuell kurs": r["Aktuell kurs"], "GAV SEK": r.get("GAV SEK",0.0)
            })
    if flags:
        st.warning("Följande innehav kan vara värda att trimma/sälja:")
        st.dataframe(pd.DataFrame(flags).sort_values(by="Andel (%)", ascending=False), hide_index=True, use_container_width=True)
    else:
        st.success("Inga uppenbara trim/sälj-kandidater just nu.")
