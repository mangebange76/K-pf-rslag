# stockapp/views/edit.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import pytz
except Exception:
    pytz = None

try:
    import streamlit as st
except Exception:
    st = None

# Runners (hämtningar)
from ..sources import fetch_all_fields_for_ticker, fetch_price_only

# ------------------------------------------------------------
# Standard TS-fält (samma som i batch)
# ------------------------------------------------------------
DEFAULT_TS_FIELDS = {
    "Utestående aktier": "TS_Utestående aktier",
    "P/S": "TS_P/S",
    "P/S Q1": "TS_P/S Q1",
    "P/S Q2": "TS_P/S Q2",
    "P/S Q3": "TS_P/S Q3",
    "P/S Q4": "TS_P/S Q4",
    "Omsättning idag": "TS_Omsättning idag",
    "Omsättning nästa år": "TS_Omsättning nästa år",
}

MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _now_stamp() -> str:
    if pytz:
        try:
            tz = pytz.timezone("Europe/Stockholm")
            return datetime.now(tz).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.now().strftime("%Y-%m-%d")

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            if any(k in c.lower() for k in ["p/s","omsättning","kurs","marginal","utdelning","cagr","antal","riktkurs","värde","debt","cash","kassa","fcf","runway","market cap","gav"]):
                df[c] = 0.0
            else:
                df[c] = ""
    return df

def _stamp_ts_for_field(df: pd.DataFrame, ridx: int, field: str, ts_fields: Dict[str, str]):
    ts_col = ts_fields.get(field)
    if ts_col:
        df.at[ridx, ts_col] = _now_stamp()

def _note_manual_update(df: pd.DataFrame, ridx: int):
    _ensure_cols(df, ["Senast manuellt uppdaterad"])
    df.at[ridx, "Senast manuellt uppdaterad"] = _now_stamp()

def _note_auto_update(df: pd.DataFrame, ridx: int, source: str):
    _ensure_cols(df, ["Senast auto-uppdaterad","Senast uppdaterad källa"])
    df.at[ridx, "Senast auto-uppdaterad"] = _now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = source

def _row_index_for_ticker(df: pd.DataFrame, ticker: str) -> Optional[int]:
    t = str(ticker).strip().upper()
    idx = df.index[df["Ticker"].astype(str).str.upper() == t].tolist()
    return idx[0] if idx else None

def _oldest_any_ts(row: pd.Series, ts_fields: Dict[str, str]) -> Optional[pd.Timestamp]:
    dates = []
    for c in ts_fields.values():
        if c in row and str(row[c]).strip():
            d = pd.to_datetime(str(row[c]).strip(), errors="coerce")
            if pd.notna(d):
                dates.append(d)
    return min(dates) if dates else None

def _add_oldest_ts_col(df: pd.DataFrame, ts_fields: Dict[str, str]) -> pd.DataFrame:
    df["_oldest_any_ts"] = df.apply(lambda r: _oldest_any_ts(r, ts_fields), axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return df

def _ps_avg_from_row(row: pd.Series) -> float:
    vals = []
    for k in ["P/S Q1","P/S Q2","P/S Q3","P/S Q4"]:
        try:
            v = float(row.get(k, 0.0) or 0.0)
            if v > 0:
                vals.append(v)
        except Exception:
            pass
    return round(float(np.mean(vals)), 2) if vals else 0.0

def _recompute_locally(df: pd.DataFrame, ridx: int):
    """Minimal lokal omräkning (om appen inte skickar in recompute_cb)."""
    # P/S-snitt
    df.at[ridx, "P/S-snitt"] = _ps_avg_from_row(df.loc[ridx])
    # Riktkurser om möjligt
    r = df.loc[ridx]
    ps_snitt = float(r.get("P/S-snitt", 0.0) or 0.0)
    shares_m = float(r.get("Utestående aktier", 0.0) or 0.0)
    shares = shares_m * 1e6 if shares_m > 0 else 0.0
    if ps_snitt > 0 and shares > 0:
        for field_src, field_dst in [
            ("Omsättning idag",    "Riktkurs idag"),
            ("Omsättning nästa år","Riktkurs om 1 år"),
            ("Omsättning om 2 år", "Riktkurs om 2 år"),
            ("Omsättning om 3 år", "Riktkurs om 3 år"),
        ]:
            val = float(r.get(field_src, 0.0) or 0.0)
            if val > 0:
                df.at[ridx, field_dst] = round((val * ps_snitt) / shares_m, 2) if shares_m > 0 else 0.0

# ------------------------------------------------------------
# Applicera vals → df (stämpla TS alltid för spårade fält)
# ------------------------------------------------------------

def _apply_vals_to_row(df: pd.DataFrame, ridx: int, vals: Dict, ts_fields: Dict[str, str], source_label: str) -> Tuple[bool, List[str]]:
    changed = False
    changed_fields = []
    _ensure_cols(df, list(vals.keys()) + ["Senast auto-uppdaterad","Senast uppdaterad källa"])
    for k, v in vals.items():
        if k not in df.columns:
            df[k] = np.nan
        old = df.at[ridx, k]
        df.at[ridx, k] = v
        if (pd.isna(old) and not pd.isna(v)) or (str(old) != str(v)):
            changed = True
            changed_fields.append(k)
        if k in ts_fields:
            _stamp_ts_for_field(df, ridx, k, ts_fields)
    _note_auto_update(df, ridx, source_label)
    return changed, changed_fields

# ------------------------------------------------------------
# Manuell prognoslista
# ------------------------------------------------------------

def _manual_forecast_list(df: pd.DataFrame, ts_fields: Dict[str, str]) -> pd.DataFrame:
    need_cols = ["Omsättning idag","Omsättning nästa år"]
    ts_cols = [ts_fields[c] for c in need_cols if c in ts_fields]
    for c in need_cols + ts_cols + ["Bolagsnamn","Ticker"]:
        if c not in df.columns:
            df[c] = 0.0 if c in need_cols else ""
    out = []
    for _, r in df.iterrows():
        missing_val = any((float(r.get(c,0.0)) <= 0.0) for c in need_cols)
        missing_ts = any((not str(r.get(ts, "")).strip()) for ts in ts_cols)
        oldest = None
        # specifikt bara för de två fälten:
        dts = []
        for c in ts_cols:
            if str(r.get(c,"")).strip():
                d = pd.to_datetime(str(r[c]).strip(), errors="coerce")
                if pd.notna(d): dts.append(d)
        if dts:
            oldest = min(dts)
        out.append({
            "Ticker": r.get("Ticker",""),
            "Bolagsnamn": r.get("Bolagsnamn",""),
            "TS_Omsättning idag": r.get(ts_fields["Omsättning idag"], ""),
            "TS_Omsättning nästa år": r.get(ts_fields["Omsättning nästa år"], ""),
            "Äldsta TS": oldest.strftime("%Y-%m-%d") if oldest is not None else "",
            "Saknar värde?": "Ja" if missing_val else "Nej",
            "Saknar TS?": "Ja" if missing_ts else "Nej",
        })
    out_df = pd.DataFrame(out)
    # sortera på äldst TS (tomma sist)
    out_df["_old"] = pd.to_datetime(out_df["Äldsta TS"], errors="coerce")
    out_df["_old_fill"] = out_df["_old"].fillna(pd.Timestamp("2099-12-31"))
    out_df = out_df.sort_values(by=["_old_fill","Bolagsnamn"]).drop(columns=["_old","_old_fill"]).reset_index(drop=True)
    return out_df

# ------------------------------------------------------------
# Huvudvy
# ------------------------------------------------------------

def lagg_till_eller_uppdatera(
    df: pd.DataFrame,
    user_rates: dict,
    save_cb=None,           # callable(df) -> None
    recompute_cb=None,      # callable(df) -> df  (hela df)  – valfri, annars lokal omräkning för raden
    ts_fields: Dict[str, str] = None
) -> pd.DataFrame:

    if ts_fields is None:
        ts_fields = DEFAULT_TS_FIELDS

    st.header("➕ Lägg till / uppdatera bolag")

    # Init session_state keys FÖRE widgets
    if "edit_idx" not in st.session_state:
        st.session_state.edit_idx = 0
    if "last_single_log" not in st.session_state:
        st.session_state.last_single_log = {}

    # Säkerställ viktiga kolumner
    must_cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
        "Utestående aktier","Antal aktier","GAV (SEK)",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Senast manuellt uppdaterad","Senast auto-uppdaterad","Senast uppdaterad källa",
    ] + list(ts_fields.values())
    df = _ensure_cols(df, must_cols)

    # Sorteringsval
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst uppdaterade först (alla fält)"], index=0)
    if sort_val.startswith("Äldst"):
        work = _add_oldest_ts_col(df.copy(), ts_fields)
        vis_df = work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # rullista
    labels = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if not labels:
        st.info("Inga bolag i databasen ännu.")
        return df

    # map label → index in vis_df
    label_to_vis_idx = {labels[i]: i for i in range(len(labels))}
    # map vis_idx → df-index
    vis_idx_to_df_idx = {i: vis_df.index[i] for i in range(len(vis_df))}

    # Håll edit_idx inom bounds
    st.session_state.edit_idx = int(max(0, min(st.session_state.edit_idx, len(labels)-1)))

    # Välj label (default enligt edit_idx)
    cur_label = st.selectbox("Välj bolag", labels, index=st.session_state.edit_idx)
    # Synka edit_idx om användaren bytt val i listan
    st.session_state.edit_idx = label_to_vis_idx.get(cur_label, st.session_state.edit_idx)

    # Prev/next knappar
    cprev, cmid, cnext = st.columns([1,2,1])
    with cprev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_idx = max(0, st.session_state.edit_idx - 1)
    with cnext:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_idx = min(len(labels)-1, st.session_state.edit_idx + 1)
    with cmid:
        st.write(f"Post {st.session_state.edit_idx+1}/{len(labels)}")

    # Aktiv rad (df-index)
    ridx = vis_idx_to_df_idx[st.session_state.edit_idx]
    r = df.loc[ridx]

    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    # Visa uppdaterings-metadata
    meta_left, meta_right = st.columns([1,1])
    with meta_left:
        st.caption(f"Senast manuellt uppdaterad: **{r.get('Senast manuellt uppdaterad','')}**")
        st.caption(f"Senast auto-uppdaterad: **{r.get('Senast auto-uppdaterad','')}**")
        st.caption(f"Källa: **{r.get('Senast uppdaterad källa','')}**")
    with meta_right:
        # visa små TS-chippar
        chips = []
        for k, ts in ts_fields.items():
            if ts in df.columns:
                val = str(r.get(ts,"")).strip() or "—"
                chips.append(f"{ts.replace('TS_','')}: {val}")
        st.caption(" • ".join(chips))

    # Form
    with st.form("form_edit_company"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=str(r.get("Ticker","")).upper())
            utest  = st.number_input("Utestående aktier (miljoner)", value=float(r.get("Utestående aktier",0.0)), step=1.0)
            antal  = st.number_input("Antal aktier du äger", value=float(r.get("Antal aktier",0.0)), step=1.0)
            gavsek = st.number_input("GAV (SEK)", value=float(r.get("GAV (SEK)",0.0)), step=0.01, format="%.4f")

            ps   = st.number_input("P/S",   value=float(r.get("P/S",0.0)), step=0.01, format="%.4f")
            ps1  = st.number_input("P/S Q1", value=float(r.get("P/S Q1",0.0)), step=0.01, format="%.4f")
            ps2  = st.number_input("P/S Q2", value=float(r.get("P/S Q2",0.0)), step=0.01, format="%.4f")
            ps3  = st.number_input("P/S Q3", value=float(r.get("P/S Q3",0.0)), step=0.01, format="%.4f")
            ps4  = st.number_input("P/S Q4", value=float(r.get("P/S Q4",0.0)), step=0.01, format="%.4f")
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(r.get("Omsättning idag",0.0)), step=1.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(r.get("Omsättning nästa år",0.0)), step=1.0)

            st.markdown("**Vid spara uppdateras också automatiskt (utan att skriva över manuella 0-värden):**")
            st.write("- P/S-snitt och Riktkurser räknas om")
            st.write("- TS för ändrade manuella fält stämplas")
        save_btn = st.form_submit_button("💾 Spara")

    # Spara
    if save_btn:
        if not ticker:
            st.error("Ticker kan inte vara tom.")
        else:
            # skriv värden
            df.at[ridx, "Ticker"] = str(ticker).upper()
            df.at[ridx, "Utestående aktier"] = float(utest)
            df.at[ridx, "Antal aktier"] = float(antal)
            df.at[ridx, "GAV (SEK)"] = float(gavsek)

            # manuella fält (spårade)
            before = {f: float(r.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {"P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
                      "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next}
            for k, v in after.items():
                df.at[ridx, k] = float(v)

            # sätt manuell TS om ändrat
            changed_manual = [k for k in MANUELL_FALT_FOR_DATUM if float(before.get(k,0.0)) != float(after.get(k,0.0))]
            if changed_manual:
                _note_manual_update(df, ridx)
                for f in changed_manual:
                    _stamp_ts_for_field(df, ridx, f, ts_fields)

            # omräkning
            if recompute_cb:
                df2 = recompute_cb(df.copy())
                df[:] = df2
            else:
                _recompute_locally(df, ridx)

            # spara
            if save_cb:
                try:
                    save_cb(df)
                    st.success("Sparat.")
                except Exception as e:
                    st.error(f"Kunde inte spara: {e}")

    st.divider()

    # Enskilda uppdateringar (runner)
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("🔄 Endast kurs"):
            tkr = str(df.at[ridx,"Ticker"]).strip().upper()
            if not tkr:
                st.warning("Ticker saknas.")
            else:
                try:
                    vals, dbg = fetch_price_only(tkr)
                    changed, fields = _apply_vals_to_row(df, ridx, vals, ts_fields, source_label="Auto (price)")
                    # omräkning
                    if recompute_cb:
                        df2 = recompute_cb(df.copy())
                        df[:] = df2
                    else:
                        _recompute_locally(df, ridx)
                    # spara
                    if save_cb:
                        try:
                            save_cb(df)
                            st.success(f"Kurs uppdaterad för {tkr}.")
                        except Exception as e:
                            st.error(f"Kunde inte spara: {e}")
                except Exception as e:
                    st.error(f"Fel vid prisuppdatering: {e}")
    with cB:
        if st.button("🤖 Full auto för bolaget"):
            tkr = str(df.at[ridx,"Ticker"]).strip().upper()
            if not tkr:
                st.warning("Ticker saknas.")
            else:
                try:
                    with st.spinner(f"Hämtar data för {tkr}…"):
                        vals, dbg = fetch_all_fields_for_ticker(tkr)
                    changed, fields = _apply_vals_to_row(df, ridx, vals, ts_fields, source_label="Auto (single)")
                    # omräkning
                    if recompute_cb:
                        df2 = recompute_cb(df.copy())
                        df[:] = df2
                    else:
                        _recompute_locally(df, ridx)
                    # spara
                    if save_cb:
                        try:
                            save_cb(df)
                            if changed:
                                st.success(f"{tkr}: {', '.join(fields)} uppdaterade.")
                            else:
                                st.info("Inga fält ändrades, men tidsstämplar och meta uppdaterades.")
                        except Exception as e:
                            st.error(f"Kunde inte spara: {e}")
                except Exception as e:
                    st.error(f"Fel vid auto-uppdatering: {e}")
    with cC:
        st.write("")  # spacer

    st.markdown("### 📝 Manuell prognoslista (Omsättning idag / nästa år)")
    need = _manual_forecast_list(df, ts_fields)
    if need.empty:
        st.success("Inga uppenbara kandidater för manuell prognos just nu.")
    else:
        st.warning(f"{len(need)} bolag kan behöva manuell uppdatering.")
        st.dataframe(need[["Ticker","Bolagsnamn","TS_Omsättning idag","TS_Omsättning nästa år","Äldsta TS","Saknar värde?","Saknar TS?"]],
                     use_container_width=True, hide_index=True)
        # snabb-jump
        jump_list = [""] + [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in need.iterrows()]
        pick = st.selectbox("Hoppa till bolag", jump_list, index=0)
        if pick:
            # hitta i huvudlistan och sätt edit_idx
            try:
                tkr = pick.split("(")[-1].split(")")[0].strip().upper()
                # hitta vis-idx
                for i, lab in enumerate(labels):
                    if lab.endswith(f"({tkr})"):
                        st.session_state.edit_idx = i
                        st.info(f"Hoppar till {pick}.")
                        break
            except Exception:
                pass

    return df
