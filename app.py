# app.py
from __future__ import annotations
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import gspread
import yfinance as yf
from google.oauth2.service_account import Credentials

# =========================
# Grundinställningar
# =========================
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Lokal Stockholm-tid om pytz finns (annars systemtid)
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")

# =========================
# Google Sheets-koppling
# =========================
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    """Liten backoff-hjälpare för 429/kvotfel."""
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return get_spreadsheet().worksheet(SHEET_NAME)

def skapa_rates_sheet_if_missing():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        ws.update([["Valuta","Kurs"]])
        return ws

# =========================
# Svensk decimal-hjälp
# =========================
_S_THOUSANDS = re.compile(r"[ \u00A0\u202F]")  # mellanslag, no-break space, narrow no-break

def _sv_to_float_scalar(x) -> float | np.nan:
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace("kr", "").replace("SEK", "").replace("USD", "").strip()
    if "," in s and "." in s:
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        s = _S_THOUSANDS.sub("", s)
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def _sv_to_float_series(s: pd.Series) -> pd.Series:
    return s.apply(_sv_to_float_scalar).astype(float)

def _float_to_sv(x: float) -> str:
    if pd.isna(x):
        return ""
    val = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return val.replace(".", ",")

# =========================
# Kolumnschema
# =========================
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "GAV (SEK)", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad",
    "Fair value",  # nytt fält
]

NUMERIC_COLS = [
    "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "GAV (SEK)", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Fair value",
]

TEXT_COLS = ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if kol in NUMERIC_COLS:
                df[kol] = 0.0
            else:
                df[kol] = ""
    df = df[[c for c in FINAL_COLS]]
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "Riktkurs 2026": "Riktkurs om 1 år",
        "Riktkurs 2027": "Riktkurs om 2 år",
        "Riktkurs 2028": "Riktkurs om 3 år",
        "Riktkurs om idag": "Riktkurs idag",
    }
    for old, new in mapping.items():
        if old in df.columns:
            if new not in df.columns:
                df[new] = 0.0
            new_vals = pd.to_numeric(df[new], errors="coerce").fillna(0.0)
            old_vals = pd.to_numeric(df[old], errors="coerce").fillna(0.0)
            mask = (new_vals == 0.0) & (old_vals > 0.0)
            df.loc[mask, new] = old_vals[mask]
            df = df.drop(columns=[old])
    return df

def konvertera_typer_sv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = _sv_to_float_series(df[c]).fillna(0.0)
        else:
            df[c] = 0.0
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str)
        else:
            df[c] = ""
    return df

# =========================
# Data I/O (formaterade strängar in/ut)
# =========================
def hamta_data() -> pd.DataFrame:
    ws = skapa_koppling()
    vals = _with_backoff(ws.get_all_values, value_render_option='FORMATTED_VALUE')
    if not vals:
        return pd.DataFrame(columns=FINAL_COLS)
    header = vals[0]
    rows = vals[1:]
    df = pd.DataFrame(rows, columns=header)
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer_sv(df)
    return df

def spara_data(df: pd.DataFrame):
    ws = skapa_koppling()
    out = säkerställ_kolumner(df).copy()
    for c in NUMERIC_COLS:
        out[c] = out[c].apply(_float_to_sv)
    for c in TEXT_COLS:
        out[c] = out[c].astype(str)
    body = [out.columns.tolist()] + out.astype(object).where(pd.notnull(out), "").values.tolist()
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

# =========================
# Valutakurser
# =========================
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int) -> dict:
    ws = skapa_rates_sheet_if_missing()
    vals = _with_backoff(ws.get_all_values, value_render_option='FORMATTED_VALUE')
    out = {}
    if not vals:
        return out
    header = [h.strip() for h in vals[0]]
    try:
        i_val = header.index("Valuta")
        i_kur = header.index("Kurs")
    except ValueError:
        return out
    for r in vals[1:]:
        if not r or len(r) <= max(i_val, i_kur):
            continue
        cur = (r[i_val] or "").strip().upper()
        raw = (r[i_kur] or "").strip()
        v = _sv_to_float_scalar(raw)
        if pd.notna(v):
            out[cur] = float(v)
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = skapa_rates_sheet_if_missing()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))
        body.append([k, _float_to_sv(v)])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))

@st.cache_data(show_spinner=False, ttl=3600)
def hamta_valutakurser_automatiskt() -> dict:
    par = {"USD": "USDSEK=X","NOK": "NOKSEK=X","CAD": "CADSEK=X","EUR": "EURSEK=X"}
    res = {"SEK": 1.0}
    for code, ysym in par.items():
        try:
            h = yf.Ticker(ysym).history(period="1d")
            if not h.empty and "Close" in h:
                val = float(h["Close"].iloc[-1])
                if val > 0:
                    res[code] = float(val)
        except Exception:
            pass
    return res

def auto_update_valutakurser_if_stale() -> bool:
    try:
        saved = las_sparade_valutakurser()
        live = hamta_valutakurser_automatiskt()
        if not live:
            return False
        changed = False
        for k in ("USD", "NOK", "CAD", "EUR"):
            lv = live.get(k, None)
            if lv is None:
                continue
            sv = float(saved.get(k, 0.0))
            if abs(lv - sv) > 1e-4:
                changed = True
                break
        if changed or not saved:
            merged = saved.copy()
            merged.update(live)
            spara_valutakurser(merged)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            return True
        return False
    except Exception:
        return False

# =========================
# Yahoo – enklare fält
# =========================
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    try:
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0
        if series.empty or len(series) < 2:
            return 0.0
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def hamta_yahoo_fält(ticker: str) -> dict:
    out = {"Bolagsnamn": "","Aktuell kurs": 0.0,"Valuta": "USD","Årlig utdelning": 0.0,"CAGR 5 år (%)": 0.0}
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)
        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# =========================
# Beräkningar
# =========================
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    df = df.copy()
    for i, rad in df.iterrows():
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# =========================
# Massuppdatering Yahoo
# =========================
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
            data = hamta_yahoo_fält(tkr)
            failed_fields = []

            if data.get("Bolagsnamn"): df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
            else: failed_fields.append("Bolagsnamn")

            if data.get("Aktuell kurs", 0) > 0: df.at[i, "Aktuell kurs"] = data["Aktuell kurs"]
            else: failed_fields.append("Aktuell kurs")

            if data.get("Valuta"): df.at[i, "Valuta"] = data["Valuta"]
            else: failed_fields.append("Valuta")

            if "Årlig utdelning" in data:
                df.at[i, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            else:
                failed_fields.append("Årlig utdelning")

            if "CAGR 5 år (%)" in data:
                df.at[i, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)
            else:
                failed_fields.append("CAGR 5 år (%)")

            if failed_fields:
                misslyckade.append(f"{tkr}: {', '.join(failed_fields)}")

            time.sleep(0.8)
            bar.progress((i+1)/max(1,total))

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# =========================
# Formfält som triggar datum
# =========================
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# =========================
# Vyer
# =========================
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
        row_idx = df.index[df["Ticker"] == namn_map[valt_label]][0]
    else:
        bef = pd.Series({}, dtype=object)
        row_idx = None

    # ⇩ NYTT: reset av formulär-state vid byte av valt bolag
    sel_token = namn_map.get(valt_label, "__NEW__")
    if st.session_state.get("form_row_token") != sel_token:
        st.session_state["form_row_token"] = sel_token
        st.session_state["ti_ticker"]  = bef.get("Ticker","") if not bef.empty else ""
        st.session_state["ni_utest"]   = float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0
        st.session_state["ni_antal"]   = float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0
        st.session_state["ni_gav"]     = float(bef.get("GAV (SEK)",0.0)) if not bef.empty else 0.0
        st.session_state["ni_fair"]    = float(bef.get("Fair value",0.0)) if not bef.empty else 0.0
        st.session_state["ni_ps"]      = float(bef.get("P/S",0.0)) if not bef.empty else 0.0
        st.session_state["ni_ps1"]     = float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0
        st.session_state["ni_ps2"]     = float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0
        st.session_state["ni_ps3"]     = float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0
        st.session_state["ni_ps4"]     = float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0
        st.session_state["ni_oms_idag"]= float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0
        st.session_state["ni_oms_next"]= float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker_raw = st.text_input("Ticker (Yahoo-format)",
                                       value=st.session_state.get("ti_ticker",""),
                                       key="ti_ticker")
            utest = st.number_input("Utestående aktier (miljoner)",
                                    value=float(st.session_state.get("ni_utest",0.0)),
                                    key="ni_utest")
            antal = st.number_input("Antal aktier du äger",
                                    value=float(st.session_state.get("ni_antal",0.0)),
                                    key="ni_antal")
            gav_sek = st.number_input("GAV (SEK)",
                                      value=float(st.session_state.get("ni_gav",0.0)),
                                      key="ni_gav")
            fair_value = st.number_input("Fair value",
                                         value=float(st.session_state.get("ni_fair",0.0)),
                                         key="ni_fair")
            ps  = st.number_input("P/S",
                                  value=float(st.session_state.get("ni_ps",0.0)),
                                  key="ni_ps")
            ps1 = st.number_input("P/S Q1",
                                  value=float(st.session_state.get("ni_ps1",0.0)),
                                  key="ni_ps1")
            ps2 = st.number_input("P/S Q2",
                                  value=float(st.session_state.get("ni_ps2",0.0)),
                                  key="ni_ps2")
            ps3 = st.number_input("P/S Q3",
                                  value=float(st.session_state.get("ni_ps3",0.0)),
                                  key="ni_ps3")
            ps4 = st.number_input("P/S Q4",
                                  value=float(st.session_state.get("ni_ps4",0.0)),
                                  key="ni_ps4")
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",
                                        value=float(st.session_state.get("ni_oms_idag",0.0)),
                                        key="ni_oms_idag")
            oms_next  = st.number_input("Omsättning nästa år (miljoner)",
                                        value=float(st.session_state.get("ni_oms_next",0.0)),
                                        key="ni_oms_next")

            st.markdown("**Uppdateras automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas om")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker_raw:
        new_tkr = (ticker_raw or "").strip().upper()

        # DUBBLETTKONTROLL
        tkr_norm = df["Ticker"].astype(str).str.strip().str.upper()
        cur_tkr = (bef.get("Ticker","") if not bef.empty else "").strip().upper()
        if bef.empty:
            if (tkr_norm == new_tkr).any():
                st.error(f"Tickern **{new_tkr}** finns redan i databasen. Välj den i listan för att redigera.")
                st.stop()
        else:
            if new_tkr != cur_tkr and (tkr_norm == new_tkr).any():
                st.error(f"Kan inte byta till tickern **{new_tkr}** – den finns redan i en annan rad.")
                st.stop()

        ny = {
            "Ticker": new_tkr, "Utestående aktier": utest, "Antal aktier": antal,
            "GAV (SEK)": gav_sek,
            "Fair value": fair_value,
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
                df.at[row_idx, k] = v
            if new_tkr != cur_tkr:
                df.at[row_idx, "Ticker"] = new_tkr
        else:
            tom = {c: (0.0 if c in NUMERIC_COLS else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"]==new_tkr, "Senast manuellt uppdaterad"] = now_stamp()

        data = hamta_yahoo_fält(new_tkr)
        if data.get("Bolagsnamn"): df.loc[df["Ticker"]==new_tkr, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[df["Ticker"]==new_tkr, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[df["Ticker"]==new_tkr, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data:    df.loc[df["Ticker"]==new_tkr, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:      df.loc[df["Ticker"]==new_tkr, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")
        st.rerun()

    # Tabellen: äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )

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
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = ["Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
                "P/S-snitt","Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
                "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
                "CAGR 5 år (%)","Antal aktier","GAV (SEK)","Årlig utdelning","Senast manuellt uppdaterad"]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs och marknadsvärde
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]

    # Anskaffningsvärde (köpt för): Antal × GAV
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]

    # Vinst/förlust
    port["Vinst/Förlust (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst/Förlust (%)"] = np.where(
        port["Anskaffningsvärde (SEK)"] > 0,
        (port["Vinst/Förlust (SEK)"] / port["Anskaffningsvärde (SEK)"]) * 100.0,
        0.0
    )

    # Andelar och utdelning
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde > 0, round(port["Värde (SEK)"] / total_värde * 100.0, 2), 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    tot_ansk = float(port["Anskaffningsvärde (SEK)"].sum())
    tot_pl = float(port["Vinst/Förlust (SEK)"].sum())
    tot_pl_pct = (tot_pl / tot_ansk * 100.0) if tot_ansk > 0 else 0.0

    # Sammanfattning
    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Totalt anskaffningsvärde:** {round(tot_ansk,2)} SEK")
    st.markdown(f"**Orealiserad vinst/förlust:** {round(tot_pl,2)} SEK ({round(tot_pl_pct,2)} %)")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    # Sorteringsreglage (nytt)
    sort_val = st.radio("Sortera efter anskaffningsvärde", ["Störst först","Minst först"], horizontal=True)
    ascending = (sort_val == "Minst först")
    port = port.sort_values(by="Anskaffningsvärde (SEK)", ascending=ascending).reset_index(drop=True)

    # Bläddring (nytt)
    if "port_idx" not in st.session_state:
        st.session_state.port_idx = 0
    st.session_state.port_idx = min(st.session_state.port_idx, len(port)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående innehav"):
            st.session_state.port_idx = max(0, st.session_state.port_idx - 1)
    with col_mid:
        st.write(f"Post {st.session_state.port_idx+1}/{len(port)}")
    with col_next:
        if st.button("➡️ Nästa innehav"):
            st.session_state.port_idx = min(len(port)-1, st.session_state.port_idx + 1)

    r = port.iloc[st.session_state.port_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
    st.markdown(
        f"""
- **Antal aktier:** {int(r['Antal aktier'])}
- **GAV (SEK):** {round(r['GAV (SEK)'],2)}
- **Anskaffningsvärde (SEK):** {round(r['Anskaffningsvärde (SEK)'],2)}
- **Aktuell kurs:** {round(r['Aktuell kurs'],2)} {r['Valuta']}
- **Växelkurs:** {round(r['Växelkurs'],6)}
- **Värde (SEK):** {round(r['Värde (SEK)'],2)}
- **Vinst/Förlust (SEK):** {round(r['Vinst/Förlust (SEK)'],2)} ({round(r['Vinst/Förlust (%)'],2)} %)
- **Andel av portfölj:** {r['Andel (%)']} %
- **Årlig utdelning per aktie:** {round(r['Årlig utdelning'],2)} {r['Valuta']}
- **Total årlig utdelning (SEK):** {round(r['Total årlig utdelning (SEK)'],2)}
"""
    )

    # Hela tabellen (oförändrad)
    st.dataframe(
        port[[
            "Ticker","Bolagsnamn","Antal aktier","GAV (SEK)","Anskaffningsvärde (SEK)",
            "Aktuell kurs","Valuta","Växelkurs","Värde (SEK)",
            "Vinst/Förlust (SEK)","Vinst/Förlust (%)",
            "Årlig utdelning","Total årlig utdelning (SEK)","Andel (%)"
        ]],
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

    ps_filter = st.selectbox(
        "Filtrera på P/S i förhållande till P/S-snitt",
        ["Alla", "P/S under snitt", "P/S över snitt"],
        index=0
    )
    fv_filter = st.selectbox(
        "Filtrera på aktiekurs i förhållande till Fair value",
        ["Alla", "Under fair value", "Över fair value"],
        index=0
    )

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()

    if ps_filter == "P/S under snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] < base["P/S-snitt"])].copy()
    elif ps_filter == "P/S över snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] > base["P/S-snitt"])].copy()

    if fv_filter == "Under fair value":
        base = base[(base["Fair value"] > 0) & (base["Aktuell kurs"] < base["Fair value"])].copy()
    elif fv_filter == "Över fair value":
        base = base[(base["Fair value"] > 0) & (base["Aktuell kurs"] > base["Fair value"])].copy()

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
- **Fair value:** {round(rad.get('Fair value', 0.0), 2)} {rad['Valuta']}
- **Nuvarande P/S (TTM):** {round(rad.get('P/S', 0.0), 2)}
- **P/S-snitt (Q1–Q4):** {round(rad.get('P/S-snitt', 0.0), 2)}
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

# =========================
# Sidomeny – valutor + data
# =========================
def valutakurser_sidebar() -> dict:
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = las_sparade_valutakurser()
    usd = st.sidebar.number_input("USD → SEK", value=float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, format="%.6f", key="fx_usd")
    nok = st.sidebar.number_input("NOK → SEK", value=float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, format="%.6f", key="fx_nok")
    cad = st.sidebar.number_input("CAD → SEK", value=float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, format="%.6f", key="fx_cad")
    eur = st.sidebar.number_input("EUR → SEK", value=float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, format="%.6f", key="fx_eur")
    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("💾 Spara", key="fx_save"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with c2:
        if st.button("↻ Läs", key="fx_read"):
            st.cache_data.clear()
            st.rerun()
    with c3:
        if st.button("🌐 Live", key="fx_live"):
            live = hamta_valutakurser_automatiskt()
            if live:
                merged = las_sparade_valutakurser()
                merged.update(live)
                spara_valutakurser(merged)
                st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
                st.sidebar.success("Live-kurser hämtade & sparade.")
                st.rerun()
            else:
                st.sidebar.warning("Kunde inte hämta live-kurser just nu.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets", key="reload_sheet"):
        st.cache_data.clear()
        st.rerun()

    return user_rates

# =========================
# MAIN
# =========================
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    auto_update_valutakurser_if_stale()

    user_rates = valutakurser_sidebar()

    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    meny = st.sidebar.radio("📌 Välj vy", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])

    if meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
