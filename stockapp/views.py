# stockapp/views.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

# Utils för ticker-normalisering & dubblettkontroll
try:
    from stockapp.utils import normalize_ticker, ensure_ticker_col, find_duplicate_tickers
except Exception:
    def normalize_ticker(x: str) -> str:
        return str(x or "").strip().upper()
    def ensure_ticker_col(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Ticker" not in df.columns:
            df["Ticker"] = ""
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
        return df
    def find_duplicate_tickers(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

# Runner för enkeltticker-uppdateringar (om du har stockapp.sources)
try:
    from stockapp.sources import get_runner_by_name
except Exception:
    get_runner_by_name = None

# TS-fältmap & vilka fält som räknas som manuellt tidsstämplade
try:
    from stockapp.config import TS_FIELDS, MANUELL_FALT_FOR_DATUM
except Exception:
    TS_FIELDS = {
        "Utestående aktier":"TS_Utestående aktier",
        "P/S":"TS_P/S", "P/S Q1":"TS_P/S Q1", "P/S Q2":"TS_P/S Q2", "P/S Q3":"TS_P/S Q3", "P/S Q4":"TS_P/S Q4",
        "Omsättning idag":"TS_Omsättning idag","Omsättning nästa år":"TS_Omsättning nästa år",
    }
    MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Kontroll-vy
# ---------------------------------------------------------------------------

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")
    if df is None or df.empty:
        st.info("Inga data att visa.")
        return

    df = ensure_ticker_col(df)

    # 1) Äldst uppdaterade (bland TS-fälten)
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    def _oldest_any_ts(row: pd.Series):
        dates = []
        for c in TS_FIELDS.values():
            if c in row and str(row[c]).strip():
                try:
                    d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                    if pd.notna(d): dates.append(d)
                except: pass
        return min(dates) if dates else pd.NaT

    work = df.copy()
    work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
    work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"]).head(20)

    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns: cols_show.append(k)
    cols_show.append("_oldest_any_ts")

    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # 2) Batchresultat (om nyligen kört)
    st.subheader("📒 Senaste batchrapport")
    log = st.session_state.get("_last_batch_log")
    if not log:
        st.info("Ingen batchkörning loggad i denna session.")
    else:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("Ladda ner som CSV", data=csv, file_name="batchrapport.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Analys-vy (enkel)
# ---------------------------------------------------------------------------

def analysvy(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("📈 Analys")

    df = ensure_ticker_col(df)
    if df.empty:
        st.info("Inga bolag i databasen.")
        return

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
    st.caption(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    cols = [
        "Ticker","Bolagsnamn","Sektor","Valuta","Aktuell kurs","Utestående aktier","Market Cap (nu)",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Bruttomarginal (%)","Nettomarginal (%)","Debt/Equity","Kassa (valuta)","FCF TTM (valuta)","Runway (mån)",
        "Årlig utdelning","GAV SEK",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Lägg till / uppdatera bolag
# ---------------------------------------------------------------------------

def _stamp_manual(df: pd.DataFrame, row_mask: pd.Series, changed_fields: List[str]) -> None:
    """Sätt 'Senast manuellt uppdaterad' och TS_ för ändrade fält om kolumnerna finns."""
    try:
        df.loc[row_mask, "Senast manuellt uppdaterad"] = _today_str()
    except Exception:
        pass
    for f in changed_fields:
        ts_col = TS_FIELDS.get(f)
        if ts_col and ts_col in df.columns:
            try:
                df.loc[row_mask, ts_col] = _today_str()
            except Exception:
                pass

def lagg_till_eller_uppdatera(
    df: pd.DataFrame,
    user_rates: Dict[str,float],
    save_cb=None,
    single_update_runner: str = "Full auto",
    price_only_runner: str = "Endast kurs",
) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    df = ensure_ticker_col(df)

    # Sortering
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        work = df.copy()
        def _oldest_any_ts(row: pd.Series):
            dates = []
            for c in TS_FIELDS.values():
                if c in row and str(row[c]).strip():
                    try:
                        d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                        if pd.notna(d): dates.append(d)
                    except: pass
            return min(dates) if dates else pd.NaT
        work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
        work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state: st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.caption(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    # --- Formulär ---
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker_input = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "")
            ticker = normalize_ticker(ticker_input)

            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)

            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav_sek = st.number_input("GAV (SEK)", value=float(bef.get("GAV SEK",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning i år (miljoner, MANUELL)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner, MANUELL)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Vid spara uppdateras inte automatiskt dessa fält:** Omsättning i år/ nästa år (manuella).")

        spar = st.form_submit_button("💾 Spara")

    # --- Spara ---
    if spar and ticker:
        # Dubblett-kontroller
        existing_mask = df["Ticker"].astype(str).str.upper().str.strip() == ticker
        if bef.empty:
            if existing_mask.any():
                st.error(f"Tickern {ticker} finns redan i databasen. Välj den i listan för att uppdatera istället.")
                return df
        else:
            original_ticker = normalize_ticker(str(bef.get("Ticker","")))
            if ticker != original_ticker and existing_mask.any():
                st.error(f"Tickern {ticker} används redan av en annan rad. Välj en unik ticker.")
                return df

        # Förbered uppdatering
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV SEK": gav_sek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Tracka vilka manuella fält som faktiskt ändrades
        changed_manual_fields = []
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            for k in MANUELL_FALT_FOR_DATUM:
                if before.get(k,0.0) != after.get(k,0.0):
                    changed_manual_fields.append(k)
        else:
            for k in MANUELL_FALT_FOR_DATUM:
                if float(ny.get(k,0.0)) != 0.0:
                    changed_manual_fields.append(k)

        # Skriv in nya fält i DF
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==bef["Ticker"], k] = v
            row_mask = (df["Ticker"] == ny["Ticker"])
        else:
            # skapa tom rad med alla kolumner bevarade
            tom = {c: df[c].dtype.type(0) if pd.api.types.is_numeric_dtype(df[c]) else "" for c in df.columns}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            row_mask = (df["Ticker"] == ny["Ticker"])

        # TS-stämpla manuellt ändrade
        if changed_manual_fields:
            _stamp_manual(df, row_mask, changed_manual_fields)

        st.success("Sparat i tabellen (lokalt).")
        if save_cb:
            try:
                save_cb(df)
                st.success("Sparat till Google Sheet.")
            except Exception as e:
                st.warning(f"Kunde inte spara till Sheet: {e}")

    st.divider()

    # --- Enkeltticker-uppdatering (via runner) ---
    runner = None
    if get_runner_by_name is not None:
        runner = get_runner_by_name(single_update_runner)

    if not bef.empty:
        colu, colv = st.columns([1,1])
        with colu:
            if st.button("⚡ Full uppdatering (runner)", key="btn_run_full_one"):
                if runner is None:
                    st.warning("Ingen runner registrerad (sources).")
                else:
                    tkr = normalize_ticker(str(bef.get("Ticker","")))
                    try:
                        res = runner(df, tkr, user_rates=user_rates)
                        st.info(f"Körning klar: {res.get('status','ok')}")
                        if res.get("saved") and save_cb:
                            try:
                                save_cb(df)
                                st.success("Ändringar sparade.")
                            except Exception as e:
                                st.warning(f"Kunde inte spara: {e}")
                    except Exception as e:
                        st.error(f"Fel vid uppdatering: {e}")
        with colv:
            runner_px = get_runner_by_name(price_only_runner) if get_runner_by_name else None
            if st.button("💱 Uppdatera endast kurs", key="btn_run_price_one"):
                if runner_px is None:
                    st.warning("Ingen kurs-runner registrerad (sources).")
                else:
                    tkr = normalize_ticker(str(bef.get("Ticker","")))
                    try:
                        res = runner_px(df, tkr, user_rates=user_rates)
                        st.info(f"Körning klar: {res.get('status','ok')}")
                        if res.get("saved") and save_cb:
                            try:
                                save_cb(df)
                                st.success("Ändringar sparade.")
                            except Exception as e:
                                st.warning(f"Kunde inte spara: {e}")
                    except Exception as e:
                        st.error(f"Fel vid uppdatering: {e}")

    st.divider()

    # --- Manuell prognoslista (flyttad hit) ---
    st.subheader("📝 Manuell prognoslista (äldsta stämpling för Omsättning i år / nästa år)")
    def _pair_oldest(row: pd.Series) -> Optional[pd.Timestamp]:
        cand = []
        for k in ["TS_Omsättning idag","TS_Omsättning nästa år"]:
            if k in row and str(row[k]).strip():
                try:
                    d = pd.to_datetime(str(row[k]).strip(), errors="coerce")
                    if pd.notna(d):
                        cand.append(d)
                except: pass
        if cand:
            return min(cand)
        return pd.NaT

    w = df.copy()
    w["_oldest_forecast"] = w.apply(_pair_oldest, axis=1)
    w["_oldest_forecast_fill"] = w["_oldest_forecast"].fillna(pd.Timestamp("2099-12-31"))
    out = w.sort_values(by=["_oldest_forecast_fill","Bolagsnamn","Ticker"])[["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år","_oldest_forecast"]].head(30)
    st.dataframe(out, use_container_width=True, hide_index=True)

    # Dubblettvarning i denna vy
    dups = find_duplicate_tickers(df)
    if not dups.empty:
        with st.expander("⚠️ Dubbletter upptäckta (klicka för att visa)"):
            st.dataframe(dups[["Ticker","Bolagsnamn"]], use_container_width=True, hide_index=True)

    return df

# ---------------------------------------------------------------------------
# Investeringsförslag (förenklad, behåll din mer avancerade om du redan har)
# ---------------------------------------------------------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("💡 Investeringsförslag")

    # Filtrera bort rader utan riktkurs/kurs
    candidates = df.copy()
    candidates = candidates[(candidates.get("Aktuell kurs",0) > 0)]
    # fallback: använd "Riktkurs om 1 år" om finns annars "Riktkurs idag"
    target_col = "Riktkurs om 1 år" if "Riktkurs om 1 år" in candidates.columns else "Riktkurs idag"
    if target_col not in candidates.columns:
        st.info("Saknar riktkurskolumner. Lägg till först.")
        return
    candidates = candidates[candidates[target_col] > 0]
    if candidates.empty:
        st.info("Inga kandidater med riktkurs.")
        return

    # Potential
    candidates["Potential (%)"] = (candidates[target_col] - candidates["Aktuell kurs"]) / candidates["Aktuell kurs"] * 100.0

    # Visa topp 20
    show_cols = ["Ticker","Bolagsnamn","Aktuell kurs",target_col,"P/S-snitt","Potential (%)","Market Cap (nu)","Utestående aktier"]
    show_cols = [c for c in show_cols if c in candidates.columns]
    st.dataframe(candidates.sort_values(by="Potential (%)", ascending=False).head(20)[show_cols],
                 use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Portfölj
# ---------------------------------------------------------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("📦 Min portfölj")
    df = ensure_ticker_col(df)
    port = df[df.get("Antal aktier",0) > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs
    def _rate(v):
        try:
            return float(user_rates.get(str(v).upper(), 1.0))
        except Exception:
            return 1.0
    port["Växelkurs"] = port["Valuta"].apply(_rate)
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, port["Värde (SEK)"] / total_värde * 100.0, 0.0).round(2)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port.get("Årlig utdelning",0) * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)","GAV SEK"]
    show_cols = [c for c in show_cols if c in port.columns]

    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False),
                 use_container_width=True, hide_index=True)
