# stockapp/views.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import streamlit as st

from .config import FINAL_COLS, TS_FIELDS
from .rates import hamta_valutakurs
from .sources import run_update_price_only, run_update_full

# För rankning (med fallback om modulen saknas)
try:
    from .scoring import rank_candidates, format_investment_badges
except Exception:
    def rank_candidates(df: pd.DataFrame, mode: str = "growth") -> pd.DataFrame:
        out = df.copy()
        out["_score"] = 0.0
        # Enkel fallback: högst uppsida via "Potential (%)" om finns
        if "Potential (%)" in out.columns:
            out["_score"] = out["Potential (%)"].fillna(0.0)
        else:
            out["_score"] = out.get("P/S-snitt", 0.0).rsub(out["P/S-snitt"].max()).fillna(0.0)
        return out.sort_values(by="_score", ascending=False)

    def format_investment_badges(row: pd.Series) -> str:
        return ""

# ---------------------------
# Hjälpare (beräkningar/UI)
# ---------------------------

def _now_stamp() -> str:
    try:
        import pytz
        tz = pytz.timezone("Europe/Stockholm")
        return datetime.now(tz).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")

def _format_mcap(x: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    neg = v < 0
    v = abs(v)
    if v >= 1_000_000_000_000:
        s = f"{v/1_000_000_000_000:.2f} T"
    elif v >= 1_000_000_000:
        s = f"{v/1_000_000_000:.2f} B"
    elif v >= 1_000_000:
        s = f"{v/1_000_000:.2f} M"
    else:
        s = f"{v:.0f}"
    return f"-{s}" if neg else s

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in FINAL_COLS:
        if col not in df.columns:
            if any(x in col.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","mcap"]):
                df[col] = 0.0
            elif col.startswith("TS_"):
                df[col] = ""
            elif col in ("Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"):
                df[col] = ""
            else:
                df[col] = ""
    # unika kolumner
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def _cagr_clamp(val: float) -> float:
    if val is None:
        return 0.0
    v = float(val)
    if v > 100.0: return 50.0
    if v < 0.0:   return 2.0
    return v

def uppdatera_berakningar(df: pd.DataFrame, user_rates: Dict[str,float]) -> pd.DataFrame:
    """Replikerar dina tidigare härledda fält."""
    df = _ensure_schema(df.copy())
    for i, r in df.iterrows():
        # P/S-snitt
        ps_vals = [r.get("P/S Q1",0), r.get("P/S Q2",0), r.get("P/S Q3",0), r.get("P/S Q4",0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        df.at[i, "P/S-snitt"] = round(float(np.mean(ps_clean)) if ps_clean else 0.0, 2)

        # Omsättning 2 & 3 år från 'Omsättning nästa år'
        cagr = _cagr_clamp(r.get("CAGR 5 år (%)", 0.0)) / 100.0
        oms_next = float(r.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + cagr), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + cagr) ** 2), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = float(r.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(r.get("Omsättning om 3 år", 0.0))

        # Riktkurser
        psn = float(df.at[i, "P/S-snitt"] or 0.0)
        shares_m = float(r.get("Utestående aktier", 0.0))
        if psn > 0 and shares_m > 0:
            act = float(r.get("Omsättning idag", 0.0))
            nxt = float(r.get("Omsättning nästa år", 0.0))
            y2  = float(df.at[i, "Omsättning om 2 år"] or 0.0)
            y3  = float(df.at[i, "Omsättning om 3 år"] or 0.0)
            den = shares_m  # miljoner aktier -> omsättning m i samma valuta
            df.at[i, "Riktkurs idag"]    = round((act * psn) / den, 2) if act>0 else 0.0
            df.at[i, "Riktkurs om 1 år"] = round((nxt * psn) / den, 2) if nxt>0 else 0.0
            df.at[i, "Riktkurs om 2 år"] = round((y2  * psn) / den, 2) if y2>0 else 0.0
            df.at[i, "Riktkurs om 3 år"] = round((y3  * psn) / den, 2) if y3>0 else 0.0
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for c in TS_FIELDS.values():
        if c in row and str(row[c]).strip():
            try:
                d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None

def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
    work["_oldest_any_ts_fill"] = pd.to_datetime(work["_oldest_any_ts"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    return work

def build_requires_manual_df(df: pd.DataFrame, older_than_days: int = 365) -> pd.DataFrame:
    """
    Flaggar bolag där något kärnfält saknas/TS saknas, eller äldsta TS är för gammal.
    """
    need_cols = ["Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]
    ts_cols = [TS_FIELDS[c] for c in TS_FIELDS if c in need_cols]
    out_rows = []
    cutoff = datetime.now() - timedelta(days=older_than_days)
    for _, r in df.iterrows():
        missing_val = any((float(r.get(c, 0.0)) <= 0.0) for c in need_cols)
        missing_ts  = any((not str(r.get(ts, "")).strip()) for ts in ts_cols if ts in r)
        oldest = _oldest_any_ts(r)
        oldest_dt = pd.to_datetime(oldest).to_pydatetime() if pd.notna(oldest) else None
        too_old = (oldest_dt is not None and oldest_dt < cutoff)
        if missing_val or missing_ts or too_old:
            out_rows.append({
                "Ticker": r.get("Ticker",""),
                "Bolagsnamn": r.get("Bolagsnamn",""),
                "Äldsta TS": oldest.strftime("%Y-%m-%d") if pd.notna(oldest) else "",
                "Saknar värde?": "Ja" if missing_val else "Nej",
                "Saknar TS?": "Ja" if missing_ts else "Nej",
            })
    return pd.DataFrame(out_rows)

def _manual_forecast_list(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Manuell prognoslista: sortera på äldsta TS av *just* Omsättning idag / nästa år.
    """
    df2 = df.copy()
    def _pick_oldest_oms_ts(row: pd.Series) -> Optional[pd.Timestamp]:
        keys = [TS_FIELDS.get("Omsättning idag"), TS_FIELDS.get("Omsättning nästa år")]
        dates = []
        for k in keys:
            if k and k in row and str(row[k]).strip():
                d = pd.to_datetime(str(row[k]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
        return min(dates) if dates else pd.NaT
    df2["_oldest_oms_ts"] = df2.apply(_pick_oldest_oms_ts, axis=1)
    df2["_oldest_oms_ts_fill"] = pd.to_datetime(df2["_oldest_oms_ts"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    vis = df2.sort_values(by=["_oldest_oms_ts_fill","Bolagsnamn"]).head(top_n)
    show_cols = ["Ticker","Bolagsnamn", TS_FIELDS.get("Omsättning idag",""), TS_FIELDS.get("Omsättning nästa år",""), "_oldest_oms_ts"]
    show_cols = [c for c in show_cols if c]
    return vis[show_cols]

def _risk_label(mcap: float) -> str:
    try:
        v = float(mcap)
    except Exception:
        return "Unknown"
    if v < 300e6: return "Microcap"
    if v <   2e9: return "Smallcap"
    if v <  10e9: return "Midcap"
    return "Largecap"

# ---------------------------
# Vy: Kontroll
# ---------------------------

def kontrollvy(df: pd.DataFrame) -> None:
    st.header("🧭 Kontroll")

    # Äldst uppdaterade – topp 20
    st.subheader("⏱️ Äldst uppdaterade (alla spårade fält)")
    work = add_oldest_ts_col(df.copy())
    vis = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"]).head(20)
    cols_show = ["Ticker","Bolagsnamn"]
    for k in ["TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"]:
        if k in vis.columns: cols_show.append(k)
    cols_show.append("_oldest_any_ts")
    st.dataframe(vis[cols_show], use_container_width=True, hide_index=True)

    st.divider()

    # Kräver manuell hantering?
    st.subheader("🛠️ Kräver manuell hantering")
    older_days = st.number_input("Flagga om äldsta TS är äldre än (dagar)", min_value=30, max_value=2000, value=365, step=30, key="kontroll_older_days")
    need = build_requires_manual_df(df, older_than_days=int(older_days))
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell hantering just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell hantering:")
        st.dataframe(need, use_container_width=True, hide_index=True)

    st.divider()

    # Batch-logg (om satt av sidopanelen)
    st.subheader("📒 Senaste batch-körlogg")
    log = st.session_state.get("batch_log")
    if not log:
        st.info("Ingen batch-körning i denna session ännu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ändringar** (ticker → fält)")
            if log.get("changed"):
                st.json(log["changed"])
            else:
                st.write("–")
        with col2:
            st.markdown("**Missar** (ticker → orsak)")
            if log.get("misses"):
                st.json(log["misses"])
            else:
                st.write("–")

# ---------------------------
# Vy: Analys
# ---------------------------

def analysvy(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Inga bolag i databasen ännu.")
        return

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = st.number_input("Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1), value=st.session_state.analys_idx, step=1, key="analys_idx_num")
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev_btn"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next_btn"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)

    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")

    r = vis_df.iloc[st.session_state.analys_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    # Etikett om senaste uppdateringstyp
    auto = str(r.get("Senast auto-uppdaterad","")).strip()
    manu = str(r.get("Senast manuellt uppdaterad","")).strip()
    src  = str(r.get("Senast uppdaterad källa","")).strip()
    if auto or manu:
        lab = "Auto" if auto and (auto >= manu) else "Manuellt"
        tid = auto if lab=="Auto" else manu
        st.caption(f"Senast uppdaterad: **{lab}** ({tid}){f' – {src}' if src else ''}")

    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
        "TS_Utestående aktier","TS_P/S","TS_P/S Q1","TS_P/S Q2","TS_P/S Q3","TS_P/S Q4","TS_Omsättning idag","TS_Omsättning nästa år"
    ]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True, hide_index=True)

# ---------------------------
# Vy: Lägg till / uppdatera
# ---------------------------

def lagg_till_eller_uppdatera(
    df: pd.DataFrame,
    user_rates: Dict[str,float],
    save_cb = None,        # valfri callback(df) -> None för att spara
) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsval
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], key="edit_sort")
    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn"])
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
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef_mask = (df["Ticker"].astype(str).str.upper() == namn_map[valt_label].upper())
        bef = df[bef_mask].iloc[0] if bef_mask.any() else pd.Series({}, dtype=object)
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav_sek = st.number_input("GAV (SEK)", value=float(bef.get("GAV (SEK)",0.0)) if "GAV (SEK)" in df.columns and not bef.empty else 0.0)
            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över manuella 0-värden):**")
            st.write("- Riktkurser/beräkningar räknas om")

        spar = st.form_submit_button("💾 Spara")

    # Spara
    if spar and ticker:
        # Dublettskydd (fall 1: ny)
        if ticker.upper() not in df["Ticker"].astype(str).str.upper().values:
            # ok
            pass
        else:
            # Fall 2: redigering av befintlig (ok), men blockera om användaren av misstag byter till en ticker
            # som redan finns i en annan rad.
            exists_mask = (df["Ticker"].astype(str).str.upper() == ticker.upper())
            if bef.empty and exists_mask.any():
                st.error("Ticker finns redan – inga dubbletter tillåtna.")
                return df

        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV (SEK)": gav_sek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        df2 = df.copy()
        if not bef.empty:
            for k,v in ny.items():
                if k in df2.columns:
                    df2.loc[df2["Ticker"].astype(str).str.upper()==ticker.upper(), k] = v
        else:
            # skapa tom rad
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in FINAL_COLS}
            for k,v in ny.items():
                tom[k] = v
            df2 = pd.concat([df2, pd.DataFrame([tom])], ignore_index=True)

        # stämpla manuell TS för fält som ändrats
        ridx = df2.index[df2["Ticker"].astype(str).str.upper()==ticker.upper()][0]
        df2.at[ridx, "Senast manuellt uppdaterad"] = _now_stamp()
        for fld in ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]:
            if fld in TS_FIELDS and fld in ny:
                df2.at[ridx, TS_FIELDS[fld]] = _now_stamp()

        df2 = uppdatera_berakningar(df2, user_rates)
        if callable(save_cb):
            save_cb(df2)
            st.success("Sparat.")
        return df2

    # Enskilda uppdateringsknappar (för valt bolag)
    if not bef.empty:
        colu1, colu2 = st.columns([1,1])
        tick = bef.get("Ticker","")
        with colu1:
            if st.button("💹 Uppdatera pris (Yahoo)", key="upd_price_btn"):
                try:
                    df2, changed, msg = run_update_price_only(df, tick, user_rates)
                    df2 = uppdatera_berakningar(df2, user_rates)
                    if callable(save_cb): save_cb(df2)
                    st.success(f"{tick}: {msg}. Ändrade fält: {changed or 'Inga'}")
                    return df2
                except Exception as e:
                    st.error(f"{tick}: Fel: {e}")
        with colu2:
            if st.button("🔄 Full uppdatering (Yahoo)", key="upd_full_btn"):
                try:
                    df2, changed, msg = run_update_full(df, tick, user_rates)
                    df2 = uppdatera_berakningar(df2, user_rates)
                    if callable(save_cb): save_cb(df2)
                    st.success(f"{tick}: {msg}. Ändrade fält: {changed or 'Inga'}")
                    return df2
                except Exception as e:
                    st.error(f"{tick}: Fel: {e}")

    st.divider()
    st.subheader("📝 Manuell prognoslista (äldre Omsättning TS först)")
    vis = _manual_forecast_list(df, top_n=25)
    if vis.empty:
        st.info("Alla 'Omsättning idag/nästa år' har aktuella datumstämplar.")
    else:
        st.dataframe(vis, use_container_width=True, hide_index=True)

    return df

# ---------------------------
# Vy: Investeringsförslag
# ---------------------------

def visa_investeringsforslag(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("💡 Investeringsförslag")

    # Val för typ: tillväxt / utdelning
    mode = st.radio("Fokus", ["Tillväxt", "Utdelning"], horizontal=True, key="inv_mode")
    mode_key = "dividend" if mode == "Utdelning" else "growth"

    # Filtrera på kap-segment & sektor
    cap_filter = st.selectbox("Filter: börsvärde", ["Alla","Microcap","Smallcap","Midcap","Largecap"], index=0, key="inv_cap")
    sector_filter = st.text_input("Filtrera på sektor (frivilligt, case-insensitive delsträng)", value="", key="inv_sector")

    # Förbered data
    base = df.copy()
    # Market cap (om ej kolumn finns – försök härleda via kurs*aktier)
    if "MarketCap" not in base.columns:
        base["MarketCap"] = 0.0
    px = base.get("Aktuell kurs", 0.0).astype(float)
    sh_m = base.get("Utestående aktier", 0.0).astype(float) # i miljoner
    # approximera mcap i prisets valuta
    base["MarketCap"] = np.where((px>0) & (sh_m>0), px * (sh_m*1e6), base["MarketCap"].astype(float))

    # Risklabel
    base["Risklabel"] = base["MarketCap"].apply(_risk_label)

    # P/S-snitt säkras
    if "P/S-snitt" not in base.columns:
        base["P/S-snitt"] = 0.0
    # Potential (%) mot "Riktkurs om 1 år" om finns
    if "Riktkurs om 1 år" in base.columns:
        base["Potential (%)"] = np.where(
            (base["Aktuell kurs"]>0) & (base["Riktkurs om 1 år"]>0),
            (base["Riktkurs om 1 år"] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0,
            0.0
        )
    else:
        base["Potential (%)"] = 0.0

    # Filter kap
    if cap_filter != "Alla":
        base = base[base["Risklabel"] == cap_filter].copy()

    # Filter sektor
    if sector_filter.strip():
        col = "Sektor" if "Sektor" in base.columns else "Sector"
        if col in base.columns:
            base = base[base[col].astype(str).str.lower().str.contains(sector_filter.strip().lower())].copy()

    # Growth vs Dividend logik (enkel selektion)
    if mode_key == "dividend":
        # behåll bolag med positiv årlig utdelning
        base = base[base.get("Årlig utdelning", 0.0).astype(float) > 0.0].copy()
    # Rankning
    if base.empty:
        st.info("Inga bolag matchar filtren.")
        return

    ranked = rank_candidates(base, mode=mode_key).reset_index(drop=True)

    # UI: välj riktkurs att jämföra
    riktkurs_val = st.selectbox(
        "Jämför mot riktkurs",
        ["Riktkurs om 1 år","Riktkurs idag","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=0,
        key="inv_rk"
    )

    # Visa topp 20 med expander
    show = ranked.head(20).copy()
    for i, r in show.iterrows():
        lbl = f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})"
        with st.expander(f"{i+1}. {lbl}  —  Score: {round(r.get('_score',0.0),2)}"):
            valuta = r.get("Valuta","")
            st.markdown(
                f"- **Aktuell kurs:** {round(float(r.get('Aktuell kurs',0.0)),2)} {valuta}\n"
                f"- **Market Cap:** { _format_mcap(r.get('MarketCap',0.0)) } {valuta}\n"
                f"- **P/S (TTM):** {round(float(r.get('P/S',0.0)),2)}\n"
                f"- **P/S-snitt (Q1–Q4):** {round(float(r.get('P/S-snitt',0.0)),2)}\n"
                f"- **Riktkurs (val):** {round(float(r.get(riktkurs_val,0.0)),2)} {valuta}\n"
                f"- **Uppsida mot riktkurs:** {round(float(r.get('Potential (%)',0.0)),2)} %\n"
                f"- **Risklabel:** {r.get('Risklabel','')}\n"
            )
            # historisk P/S
            ps_hist = []
            for q in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
                if q in r and float(r.get(q,0.0))>0:
                    ps_hist.append(f"{q.split()[-1]}: {round(float(r[q]),2)}")
            if ps_hist:
                st.caption("Historik P/S: " + " | ".join(ps_hist))

            # ev. badges/kommentar
            try:
                badges = format_investment_badges(r)
                if badges:
                    st.markdown(badges)
            except Exception:
                pass

# ---------------------------
# Vy: Portfölj
# ---------------------------

def visa_portfolj(df: pd.DataFrame, user_rates: Dict[str,float]) -> None:
    st.header("📦 Min portfölj")
    port = df[df.get("Antal aktier",0) > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs -> SEK
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"].astype(float) * port["Aktuell kurs"].astype(float) * port["Växelkurs"].astype(float)
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde>0, (port["Värde (SEK)"] / total_värde * 100.0), 0.0).round(2)
    port["Total årlig utdelning (SEK)"] = (port["Antal aktier"].astype(float) * port["Årlig utdelning"].astype(float) * port["Växelkurs"].astype(float))
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total årlig utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    show_cols = ["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]
    show_cols = [c for c in show_cols if c in port.columns]
    st.dataframe(port[show_cols].sort_values(by="Värde (SEK)", ascending=False), use_container_width=True, hide_index=True)
