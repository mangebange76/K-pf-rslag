# -*- coding: utf-8 -*-
"""
stockapp/views/edit.py
Lägg till / uppdatera bolag:
- Formulär med dubblettskydd och GAV i SEK
- Uppdatera endast kurs
- Full auto (per-bolag)
- Robust bläddring
- Manuell prognoslista (äldst TS för Omsättning idag/nästa år)
"""

from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# yfinance används endast som no-regrets fallback
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # kör utan fallback om inte installerad

# --- Lokal Stockholm-tid om pytz finns (annars systemtid) ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def _now_stamp() -> str:
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def _now_stamp() -> str:
        return datetime.now().strftime("%Y-%m-%d")

# --- Konstanter (håll i sync med övriga moduler) ---
TS_FIELDS: Dict[str, str] = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

MANUELL_FALT_FOR_DATUM = [
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år"
]

# --- Hjälpare ---------------------------------------------------------------

def _stamp_ts_for_field(df: pd.DataFrame, row_idx: int, field: str, when: Optional[str] = None):
    ts_col = TS_FIELDS.get(field)
    if not ts_col:
        return
    try:
        df.at[row_idx, ts_col] = when or _now_stamp()
    except Exception:
        pass

def _note_manual_update(df: pd.DataFrame, row_idx: int):
    try:
        df.at[row_idx, "Senast manuellt uppdaterad"] = _now_stamp()
    except Exception:
        pass

def _note_auto_update(df: pd.DataFrame, row_idx: int, source: str):
    try:
        df.at[row_idx, "Senast auto-uppdaterad"] = _now_stamp()
        df.at[row_idx, "Senast uppdaterad källa"] = source
    except Exception:
        pass

def _apply_updates_to_row(df: pd.DataFrame, ridx: int, new_vals: Dict[str, Any], source: Optional[str] = None) -> List[str]:
    """
    Skriv endast meningsfulla värden. Returnerar lista på ändrade fält.
    """
    changed: List[str] = []
    for f, v in (new_vals or {}).items():
        if f not in df.columns:
            continue
        old = df.at[ridx, f]
        write_ok = False
        if isinstance(v, (int, float, np.floating)):
            # tillåt 0 för vissa fält, annars >0
            if f in ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Utestående aktier"]:
                write_ok = float(v) > 0
            else:
                write_ok = float(v) >= 0
        else:
            write_ok = str(v).strip() != ""
        if not write_ok:
            continue
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[ridx, f] = v
            changed.append(f)
            if f in TS_FIELDS:
                _stamp_ts_for_field(df, ridx, f)

    if changed and source:
        _note_auto_update(df, ridx, source)
    return changed

def _yahoo_price_fallback(ticker: str) -> Dict[str, Any]:
    """Minimal pris/valuta/namn via yfinance som fallback."""
    out = {}
    if yf is None:
        return out
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        px = info.get("regularMarketPrice", None)
        if px is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                px = float(h["Close"].iloc[-1])
        if px is not None:
            out["Aktuell kurs"] = float(px)
        if info.get("currency"):
            out["Valuta"] = str(info["currency"]).upper()
        name = info.get("shortName") or info.get("longName")
        if name:
            out["Bolagsnamn"] = str(name)
    except Exception:
        pass
    return out

def _full_auto_fallback(ticker: str) -> Dict[str, Any]:
    """
    Försiktig fallback: pris + implied shares + enkel P/S om marketCap & revenueTTM finns.
    (Bättre runners bör sättas i st.session_state["runner_full"]).
    """
    res = _yahoo_price_fallback(ticker)
    if yf is None:
        return res
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        mcap = info.get("marketCap") or 0
        try:
            mcap = float(mcap)
        except Exception:
            mcap = 0.0
        px = float(res.get("Aktuell kurs", 0) or 0)
        if mcap > 0 and px > 0:
            res["Utestående aktier"] = mcap / px / 1e6  # i miljoner
        # revenue TTM → enkel P/S
        # (yfinance har inte alltid lättillgänglig TTM; hoppa om saknas)
        # här sätter vi endast pris, namn, valuta, ev implied shares
    except Exception:
        pass
    return res

def _forecast_watchlist(df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """
    Manuell prognoslista: sortera på äldsta av TS_Omsättning idag / TS_Omsättning nästa år.
    """
    work = df.copy()
    for c in ["TS_Omsättning idag", "TS_Omsättning nästa år"]:
        if c in work.columns:
            work[c] = pd.to_datetime(work[c], errors="coerce")
        else:
            work[c] = pd.NaT
    work["_äldst_prognos_ts"] = work[["TS_Omsättning idag", "TS_Omsättning nästa år"]].min(axis=1)
    work["_äldst_prognos_ts_fill"] = work["_äldst_prognos_ts"].fillna(pd.Timestamp("1900-01-01"))
    cols = ["Ticker", "Bolagsnamn", "TS_Omsättning idag", "TS_Omsättning nästa år", "_äldst_prognos_ts"]
    cols = [c for c in cols if c in work.columns]
    out = work.sort_values(by="_äldst_prognos_ts_fill", ascending=True)[cols].head(limit)
    return out

# --- Själva vyn --------------------------------------------------------------

def lagg_till_eller_uppdatera(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame, Dict[str, float]], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Form + actions. Returnerar ev. uppdaterad df.
    """
    st.header("➕ Lägg till / uppdatera bolag")

    # välj sortering för bläddring
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)", "Äldst uppdaterade först (alla fält)"])
    if sort_val.startswith("Äldst"):
        # återanvänd hjälpare från Kontroll (om laddad), annars lokal fallback
        if "views.control.add_oldest_ts_col" in st.session_state.get("_symbols", []):
            from stockapp.views.control import add_oldest_ts_col  # type: ignore
            work = add_oldest_ts_col(df.copy())
            vis_df = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn"])
        else:
            # minimalistisk fallback: sortera på TS_P/S om finns
            key_cols = [c for c in df.columns if str(c).startswith("TS_")]
            work = df.copy()
            if key_cols:
                first = key_cols[0]
                order = pd.to_datetime(work[first], errors="coerce")
                work["_k"] = order.fillna(pd.Timestamp("1900-01-01"))
                vis_df = work.sort_values(by=["_k", "Bolagsnamn"])
            else:
                vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"])

    namn_map = {f"{r.get('Bolagsnamn','')} ({r.get('Ticker','')})": r.get('Ticker','') for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(val_lista)-1))

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista, index=st.session_state.edit_index)

    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    # Hämta befintlig rad
    if valt_label and valt_label in namn_map:
        ticker_vald = namn_map[valt_label]
        mask = (df["Ticker"].astype(str).str.upper() == str(ticker_vald).upper())
        bef = df[mask].iloc[0] if mask.any() else pd.Series({}, dtype=object)
    else:
        ticker_vald = ""
        bef = pd.Series({}, dtype=object)

    # --- Formulär ------------------------------------------------------------
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0) or 0.0))
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0) or 0.0))
            gav_sek = st.number_input("GAV i SEK", value=float(bef.get("GAV i SEK",0.0) or 0.0), step=0.01, format="%.2f")

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0) or 0.0))
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0) or 0.0))
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0) or 0.0))
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0) or 0.0))
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0) or 0.0))
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag",0.0) or 0.0))
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0) or 0.0))

            st.caption("Vid spara uppdateras även: Bolagsnamn, Valuta, Aktuell kurs via runner/yfinance. Beräkningar räknas om om du skickar in `recompute_cb`.")

        spar = st.form_submit_button("💾 Spara")

    # --- Spara ---------------------------------------------------------------
    if spar and ticker:
        # dubblettskydd (case-insensitive)
        dupe = df["Ticker"].astype(str).str.upper() == ticker.upper()
        if bef.empty and dupe.any():
            st.error(f"Ticker '{ticker}' finns redan i tabellen.")
            return df
        if (not bef.empty) and (ticker.upper() != str(bef.get("Ticker","")).upper()) and dupe.any():
            st.error(f"Kan inte byta till '{ticker}' – tickern finns redan.")
            return df

        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV i SEK": gav_sek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Bestäm om manuell TS ska sättas + vilka fält som ändrats
        datum_sätt = False
        changed_manual_fields: List[str] = []
        if not bef.empty:
            before = {f: float(bef.get(f,0.0) or 0.0) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0) or 0.0)  for f in MANUELL_FALT_FOR_DATUM}
            for k in MANUELL_FALT_FOR_DATUM:
                if before[k] != after[k]:
                    datum_sätt = True
                    changed_manual_fields.append(k)
        else:
            if any(float(ny.get(f,0.0) or 0.0) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
                changed_manual_fields = [f for f in MANUELL_FALT_FOR_DATUM if float(ny.get(f,0.0) or 0.0) != 0.0]

        # Skriv nya fält
        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"].astype(str).str.upper()==str(bef.get("Ticker","")).upper(), k] = v
            ridx = df.index[df["Ticker"].astype(str).str.upper()==ticker.upper()][0]
        else:
            # initiera tom rad med rimliga default
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa"] and not str(c).startswith("TS_") else "") for c in df.columns}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            ridx = df.index[-1]

        # Tidsstämplar (manuellt)
        if datum_sätt:
            _note_manual_update(df, ridx)
            for f in changed_manual_fields:
                _stamp_ts_for_field(df, ridx, f)

        # Lätt basuppdatering via runner_price (eller yfinance fallback)
        price_runner = st.session_state.get("runner_price")
        try:
            base = price_runner(ticker) if callable(price_runner) else _yahoo_price_fallback(ticker)
            changed = _apply_updates_to_row(df, ridx, base, source="Form (price)")
        except Exception as e:
            st.warning(f"Kunde inte hämta basfält för {ticker}: {e}")

        # Recompute + spara
        if recompute_cb:
            df = recompute_cb(df, user_rates)
        saver = save_cb or st.session_state.get("_save_df")
        if callable(saver):
            try:
                saver(df)
                st.success("Sparat.")
            except Exception as e:
                st.error(f"Kunde inte spara: {e}")
        else:
            st.info("Ingen save-callback angiven – data sparas inte till Sheets i denna konfiguration.")

    # --- Action-knappar för valt bolag --------------------------------------
    st.markdown("### ⚙️ Snabbåtgärder för valt bolag")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📈 Uppdatera **KURS** för valt bolag", disabled=(not ticker_vald and bef.empty)):
            if not ticker_vald and not bef.empty:
                ticker_vald = str(bef.get("Ticker","")).upper()
            if ticker_vald:
                price_runner = st.session_state.get("runner_price")
                try:
                    updates = price_runner(ticker_vald) if callable(price_runner) else _yahoo_price_fallback(ticker_vald)
                    ridx = df.index[df["Ticker"].astype(str).str.upper()==ticker_vald.upper()][0]
                    changed = _apply_updates_to_row(df, ridx, updates, source="Price-only")
                    if recompute_cb: df = recompute_cb(df, user_rates)
                    saver = save_cb or st.session_state.get("_save_df")
                    if callable(saver): saver(df)
                    st.success(f"Kurs uppdaterad för {ticker_vald}. Ändrade fält: {', '.join(changed) if changed else '(inga)'}")
                except Exception as e:
                    st.error(f"Kunde inte uppdatera kurs för {ticker_vald}: {e}")
            else:
                st.info("Välj ett bolag först.")

    with c2:
        if st.button("🧠 Full auto för **valt bolag**", disabled=(not ticker_vald and bef.empty)):
            if not ticker_vald and not bef.empty:
                ticker_vald = str(bef.get("Ticker","")).upper()
            if ticker_vald:
                full_runner = st.session_state.get("runner_full")
                try:
                    updates = full_runner(ticker_vald) if callable(full_runner) else _full_auto_fallback(ticker_vald)
                    ridx = df.index[df["Ticker"].astype(str).str.upper()==ticker_vald.upper()][0]
                    changed = _apply_updates_to_row(df, ridx, updates, source="Full auto (runner/fallback)")
                    if recompute_cb: df = recompute_cb(df, user_rates)
                    saver = save_cb or st.session_state.get("_save_df")
                    if callable(saver): saver(df)
                    if changed:
                        st.success(f"Full auto klar för {ticker_vald}. Ändrade fält: {', '.join(changed)}")
                    else:
                        st.warning("Inga ändringar hittades vid full auto.")
                except Exception as e:
                    st.error(f"Kunde inte köra full auto för {ticker_vald}: {e}")
            else:
                st.info("Välj ett bolag först.")

    st.divider()

    # --- Manuell prognoslista ------------------------------------------------
    st.markdown("### 📝 Manuell prognoslista (äldsta prognoser först)")
    lista = _forecast_watchlist(df, limit=20)
    if lista.empty:
        st.info("Inga prognosrader hittades (saknar TS-kolumner).")
    else:
        st.dataframe(lista, use_container_width=True, hide_index=True)

    return df
