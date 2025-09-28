# stockapp/views/edit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Any, List
import streamlit as st
import pandas as pd
import numpy as np

from ..config import (
    säkerställ_kolumner,
    konvertera_typer,
    TS_FIELDS,
    add_oldest_ts_col,
    now_stamp,
)
from ..calc import update_calculations, safe_float

# -----------------------------
# Konstanter
# -----------------------------

MANUELL_PROGNOS_FALT = ("Omsättning idag", "Omsättning nästa år")
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# -----------------------------
# Små hjälp-funktioner
# -----------------------------

def _chip(text: str, color: str = "#F1F5F9", fg: str = "#0F172A") -> str:
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;"
        f"background:{color};color:{fg};font-size:12px;margin-right:6px;margin-bottom:6px;'>{text}</span>"
    )

def _pick_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """Bygg ordning av tickers utifrån valt sortläge."""
    if df is None or df.empty or "Ticker" not in df.columns:
        return []
    if sort_mode == "Äldst TS först":
        work = add_oldest_ts_col(df.copy())
        work = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"], ascending=[True,True,True])
        return [str(t).upper().strip() for t in work["Ticker"].tolist()]
    work = df.sort_values(by=["Bolagsnamn","Ticker"], ascending=[True,True]).copy()
    return [str(t).upper().strip() for t in work["Ticker"].tolist()]

def _stamp_manual_ts(df: pd.DataFrame, ticker: str, changed_fields: List[str]) -> None:
    """Stämpla 'Senast manuellt uppdaterad' och TS_ för ändrade manuella fält."""
    mask = (df["Ticker"].astype(str).str.upper().str.strip() == str(ticker).upper().strip())
    if not mask.any():
        return
    ridx = df.index[mask][0]
    df.at[ridx, "Senast manuellt uppdaterad"] = now_stamp()
    for f in changed_fields:
        ts_col = TS_FIELDS.get(f)
        if ts_col:
            df.at[ridx, ts_col] = now_stamp()

def _apply_auto_fields_to_row(df: pd.DataFrame, row_idx: int, new_vals: Dict[str, Any], source_label: str, changes_map: Dict[str, List[str]]) -> bool:
    """
    Precis som i batch: skriv automatiskt hämtade fält i en rad.
    - Skriv EJ manuella prognosfält.
    - Stämpla TS_ för spårade fält (även om lika värde inkom).
    - Sätt 'Senast auto-uppdaterad' + 'Senast uppdaterad källa'.
    """
    changed = False
    tkr = str(df.at[row_idx, "Ticker"]).strip().upper()
    changed_fields: List[str] = []

    for f, v in (new_vals or {}).items():
        if f in MANUELL_PROGNOS_FALT:
            continue  # manuella fält lämnas orörda

        if f not in df.columns:
            # Skapa kolumn (försiktigt gissad typ)
            if any(x in f.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","aktier","snitt","score","market cap","kassa","skuld"]):
                df[f] = 0.0
            else:
                df[f] = ""

        old = df.at[row_idx, f]
        write_ok = True
        if isinstance(v, (int, float, np.floating)):
            # Skriv säkert: för P/S, shares, market cap kräver vi >0
            if f.lower().startswith("p/s") or "market cap" in f.lower() or f in ("Utestående aktier",):
                write_ok = (float(v) > 0)
            else:
                write_ok = (f not in ("P/S", "Utestående aktier") and float(v) >= 0)
        elif isinstance(v, str):
            write_ok = (v.strip() != "")

        if not write_ok:
            # stämpla ändå TS om spårat fält levererades (utan att skriva)
            ts_col = TS_FIELDS.get(f)
            if ts_col:
                df.at[row_idx, ts_col] = now_stamp()
            continue

        # skriv och markera ändring
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            df.at[row_idx, f] = v
            changed = True
            changed_fields.append(f)

        # stämpla TS_ (även om lika)
        ts_col = TS_FIELDS.get(f)
        if ts_col:
            df.at[row_idx, ts_col] = now_stamp()

    # meta
    df.at[row_idx, "Senast auto-uppdaterad"] = now_stamp()
    df.at[row_idx, "Senast uppdaterad källa"] = source_label

    if changed_fields:
        changes_map.setdefault(tkr, []).extend(changed_fields)

    return changed

def _manual_forecast_queue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bygger lista över bolag där 'Omsättning idag'/'Omsättning nästa år' saknar
    värde/TS eller har äldst TS – sorterat på äldsta av de två TS.
    """
    work = df.copy()

    # Hämta TS-kolumner för våra två fält
    ts_cols = [TS_FIELDS.get("Omsättning idag"), TS_FIELDS.get("Omsättning nästa år")]
    for c in ts_cols:
        if c and c not in work.columns:
            work[c] = ""

    rows = []
    for _, r in work.iterrows():
        ts_idag = str(r.get(TS_FIELDS.get("Omsättning idag",""), "") or "").strip()
        ts_next = str(r.get(TS_FIELDS.get("Omsättning nästa år",""), "") or "").strip()
        # plocka min-datum av två (tomt räknas som väldigt gammalt)
        try:
            d_idag = pd.to_datetime(ts_idag, errors="coerce")
        except Exception:
            d_idag = pd.NaT
        try:
            d_next = pd.to_datetime(ts_next, errors="coerce")
        except Exception:
            d_next = pd.NaT

        # Äldsta av de två
        cand = [d for d in [d_idag, d_next] if pd.notna(d)]
        oldest = min(cand) if cand else pd.Timestamp("1900-01-01")

        # saknar värde?
        miss_val = (safe_float(r.get("Omsättning idag", 0.0)) <= 0.0) or (safe_float(r.get("Omsättning nästa år", 0.0)) <= 0.0)
        miss_ts  = (not ts_idag) or (not ts_next)

        rows.append({
            "Ticker": r.get("Ticker",""),
            "Bolagsnamn": r.get("Bolagsnamn",""),
            "TS_Oms_idag": ts_idag,
            "TS_Oms_next": ts_next,
            "Äldsta TS (manuell)": "" if oldest == pd.Timestamp("1900-01-01") else oldest.strftime("%Y-%m-%d"),
            "Saknar värde?": "Ja" if miss_val else "Nej",
            "Saknar TS?": "Ja" if miss_ts else "Nej",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # sortera på äldsta datum (tomma sist)
    def _key(x: str):
        try:
            return pd.to_datetime(x, errors="coerce")
        except Exception:
            return pd.NaT
    out["_sort"] = out["Äldsta TS (manuell)"].apply(_key)
    out = out.sort_values(by=["_sort","Bolagsnamn"]).drop(columns=["_sort"])
    return out

def _no_runner(_ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return {}, {"info": "no-runner"}

# -----------------------------
# Public view
# -----------------------------

def lagg_till_eller_uppdatera(
    df: pd.DataFrame,
    user_rates: Dict[str, float],
    save_cb: Optional[Callable[[pd.DataFrame], None]] = None,
    recompute_cb: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    runner_full: Optional[Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    runner_price: Optional[Callable[[str], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
) -> pd.DataFrame:
    """
    Lägg till / uppdatera bolag
    - Bläddra A–Ö eller Äldst TS
    - Formulär för manuell editering (inkl. GAV i SEK)
    - Knapp: Uppdatera KURS
    - Knapp: Full auto för denna
    - Manuell prognoslista (Omsättning idag / nästa år)
    """
    st.header("➕ Lägg till / uppdatera bolag")

    # Defensivt schema/typer
    df = säkerställ_kolumner(df.copy())
    df = konvertera_typer(df)

    # Sortering
    st.session_state.setdefault("edit_sort_mode", "A–Ö (bolagsnamn)")
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst TS först (alla fält)"], index=0, key="edit_sort_mode")

    if sort_val.startswith("Äldst"):
        work = add_oldest_ts_col(df.copy())
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    namn_map = {lab: vis_df.iloc[i]["Ticker"] for i, lab in enumerate(etiketter)}

    # Navigering/val
    st.session_state.setdefault("edit_idx", 0)
    st.session_state.edit_idx = int(np.clip(st.session_state.edit_idx, 0, max(0, len(etiketter)-1)))

    val_lista = ["(nytt bolag)"] + etiketter
    init_index = 0 if not etiketter else min(st.session_state.edit_idx+1, len(val_lista)-1)
    valt_label = st.selectbox("Välj bolag", val_lista, index=init_index, key="edit_selectbox")

    nav1, nav2, nav3 = st.columns([1,2,1])
    with nav1:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_idx = max(0, st.session_state.edit_idx - 1)
    with nav2:
        st.write(f"Post {st.session_state.edit_idx+1}/{max(1, len(etiketter))}")
    with nav3:
        if st.button("➡️ Nästa"):
            st.session_state.edit_idx = min(max(0, len(etiketter)-1), st.session_state.edit_idx + 1)

    if valt_label and valt_label != "(nytt bolag)":
        tkr = namn_map.get(valt_label, "")
        bef_mask = (df["Ticker"].astype(str).str.upper().str.strip() == str(tkr).upper().strip())
        bef = df[bef_mask].iloc[0] if bef_mask.any() else pd.Series({}, dtype=object)
    else:
        tkr = ""
        bef = pd.Series({}, dtype=object)

    # Badges (om befintligt)
    if not bef.empty:
        man_dt = str(bef.get("Senast manuellt uppdaterad","") or "").strip()
        auto_dt = str(bef.get("Senast auto-uppdaterad","") or "").strip()
        source  = str(bef.get("Senast uppdaterad källa","") or "").strip()
        badge_html = ""
        if auto_dt: badge_html += _chip(f"Auto: {auto_dt}", color="#DCFCE7", fg="#065F46")
        if man_dt:  badge_html += _chip(f"Manuellt: {man_dt}", color="#E0E7FF", fg="#1E3A8A")
        if source:  badge_html += _chip(f"Källa: {source}", color="#FFE4E6", fg="#9F1239")
        if badge_html:
            st.markdown(badge_html, unsafe_allow_html=True)

    # Formulär
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
            gav_sek = st.number_input("GAV i SEK (per aktie)", value=float(bef.get("GAV i SEK",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner) – MANUELL",  value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner) – MANUELL", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**OBS:** Dessa två är *alltid* manuella och hämtas aldrig automatiskt.")

        spar = st.form_submit_button("💾 Spara")

    # Spara-formlogik (utan auto-hämtning)
    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal, "GAV i SEK": gav_sek,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # vilka manuella fält ändrades?
        changed_manual_fields: List[str] = []
        if not bef.empty:
            before = {f: safe_float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: safe_float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            for k in MANUELL_FALT_FOR_DATUM:
                if before.get(k, 0.0) != after.get(k, 0.0):
                    changed_manual_fields.append(k)
        else:
            for k in MANUELL_FALT_FOR_DATUM:
                if safe_float(ny.get(k,0.0)) != 0.0:
                    changed_manual_fields.append(k)

        # skriv in
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            # skapa rad med default för alla FINAL_COLS
            # (låter säkerställ_kolumner sköta detaljerna)
            row = {c: "" for c in df.columns}
            for k,v in ny.items():
                row[k] = v
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df = säkerställ_kolumner(df)
            df = konvertera_typer(df)

        # stämpla manuella fält
        if changed_manual_fields:
            _stamp_manual_ts(df, ticker, changed_manual_fields)

        # räkna om & spara
        try:
            if recompute_cb:
                df = recompute_cb(df)
            else:
                df = update_calculations(df, user_rates)
        except Exception as e:
            st.warning(f"Kunde inte räkna om formler: {e}")

        if save_cb:
            try:
                save_cb(df)
                st.success("Sparat.")
            except Exception as e:
                st.error(f"Misslyckades spara: {e}")
        else:
            st.info("Ingen save_cb angiven – ändringar ligger i minnet.")

    st.markdown("---")

    # —————————————
    # Auto-knappar för valt bolag (kurs / full)
    # —————————————
    st.subheader("⚙️ Snabbuppdatering för valt bolag")

    colu1, colu2, colu3 = st.columns([1,1,2])
    with colu1:
        to_next = st.checkbox("Gå till nästa efter uppdatering", value=False, key="edit_go_next")
    with colu2:
        price_btn = st.button("🔂 Uppdatera KURS")
    with colu3:
        full_btn  = st.button("🔁 Full auto för denna")

    if (price_btn or full_btn):
        if not tkr:
            st.warning("Välj ett befintligt bolag först.")
        else:
            mask = (df["Ticker"].astype(str).str.upper().str.strip() == str(tkr).upper().strip())
            if not mask.any():
                st.warning(f"{tkr} hittades inte i tabellen.")
            else:
                ridx = df.index[mask][0]
                # välj runner
                run_fn = None
                src_label = ""
                if price_btn:
                    run_fn = runner_price or _no_runner
                    src_label = "Auto (KURS)"
                else:
                    run_fn = runner_full or _no_runner
                    src_label = "Auto (Full)"

                # progress 1/1
                p = st.progress(0.0)
                step = st.empty()
                step.write(f"Uppdaterar 1/1 – **{tkr}**")
                try:
                    vals, dbg = run_fn(tkr)
                    chmap: Dict[str, List[str]] = {}
                    _apply_auto_fields_to_row(df, ridx, vals, source_label=src_label, changes_map=chmap)
                except Exception as e:
                    st.error(f"Uppdatering misslyckades: {e}")
                p.progress(1.0)

                # räkna om & spara
                try:
                    if recompute_cb:
                        df = recompute_cb(df)
                    else:
                        df = update_calculations(df, user_rates)
                except Exception as e:
                    st.warning(f"Kunde inte räkna om formler: {e}")

                if save_cb:
                    try:
                        save_cb(df)
                        st.success("Ändringar sparade.")
                    except Exception as e:
                        st.error(f"Misslyckades spara: {e}")

                # ev. gå vidare
                if to_next and etiketter:
                    st.session_state.edit_idx = min(len(etiketter)-1, st.session_state.edit_idx + 1)

    # —————————————
    # Manuell prognoslista (i denna vy)
    # —————————————
    st.markdown("---")
    st.subheader("📝 Manuell prognoslista (Omsättning idag & nästa år)")
    qdf = _manual_forecast_queue(df)
    if qdf.empty:
        st.success("Inga uppenbara manuella uppdateringar behövs just nu.")
    else:
        st.dataframe(qdf[["Ticker","Bolagsnamn","Äldsta TS (manuell)","TS_Oms_idag","TS_Oms_next","Saknar värde?","Saknar TS?"]],
                     use_container_width=True, hide_index=True)

    return df
