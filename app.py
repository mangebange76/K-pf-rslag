import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
import random
from datetime import datetime
from google.oauth2.service_account import Credentials

# ---- Lokal tid (Stockholm) om pytz finns, annars systemtid ----
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---- Google Sheets koppling ----
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# --------- GSpread klient & Spreadsheet med cache ----------
@st.cache_resource
def get_gspread_client():
    creds = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    return gspread.authorize(creds)

@st.cache_resource
def get_spreadsheet(url: str):
    return get_gspread_client().open_by_url(url)

def _with_backoff(func, *args, **kwargs):
    """Kör gspread-funktion med exponentiell backoff vid kvotfel."""
    delay = 1.0
    for _ in range(6):
        try:
            return func(*args, **kwargs)
        except gspread.exceptions.APIError as e:
            msg = str(e)
            if "429" in msg or "Quota exceeded" in msg:
                time.sleep(delay + random.uniform(0, 0.25))
                delay = min(delay * 2, 16)
                continue
            raise
    raise

def skapa_koppling():
    ss = get_spreadsheet(SHEET_URL)
    return _with_backoff(ss.worksheet, SHEET_NAME)

# --------- Läs/Spara huvuddata med cache ----------
@st.cache_data(ttl=30)
def hamta_data_cached(reload_key: int) -> pd.DataFrame:
    sheet = skapa_koppling()
    rows = _with_backoff(sheet.get_all_records)
    return pd.DataFrame(rows)

def hamta_data() -> pd.DataFrame:
    rk = st.session_state.get("reload_key", 0)
    return hamta_data_cached(rk)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    payload = [df.columns.values.tolist()] + df.astype(str).values.tolist()
    _with_backoff(sheet.update, payload)
    st.session_state["reload_key"] = st.session_state.get("reload_key", 0) + 1

# --------- Valutakurser i separat blad (persistens) ----------
def _hamta_rates_sheet():
    ss = get_spreadsheet(SHEET_URL)
    try:
        return _with_backoff(ss.worksheet, RATES_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws = _with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=10, cols=4)
        _with_backoff(ws.update, [["Valuta", "Kurs", "Sparad", "Av"]])
        return ws

@st.cache_data(ttl=3600)
def las_sparade_valutakurser_cached(rk: int) -> dict:
    ws = _hamta_rates_sheet()
    rows = _with_backoff(ws.get_all_records)
    out = {}
    for r in rows:
        val = str(r.get("Valuta", "")).strip().upper()
        try:
            kurs = float(str(r.get("Kurs", "")).replace(",", "."))
        except Exception:
            continue
        if val in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            out[val] = kurs
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(user_rates: dict):
    ws = _hamta_rates_sheet()
    data = [["Valuta", "Kurs", "Sparad", "Av"]]
    datum = now_stamp()
    for val in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        data.append([val, float(user_rates.get(val, 1.0)), datum, "App"])
    _with_backoff(ws.clear)
    _with_backoff(ws.update, data)
    st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
    st.sidebar.success("Valutakurser sparade.")

# ---- Standard valutakurser till SEK (fallback) ----
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# ---- Kolumnschema ----
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad",
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs", "omsättning", "p/s", "utdelning", "cagr", "antal", "riktkurs"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    # ta bort gamla riktkurs-kolumner om de finns
    for old in ["Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Riktkurs om idag"]:
        if old in df.columns:
            df = df.drop(columns=[old])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# ---- CAGR från Yahoo (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    try:
        # prova nya API:t (income_stmt) först
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc("Total Revenue") if callable(df_fin.loc) else df_fin.loc["Total Revenue"]
                series = series.dropna()
            else:
                return 0.0

        if series.empty or len(series) < 2:
            return 0.0

        # Kronologisk ordning: äldst → nyast
        series = series.sort_index()
        start = float(series.iloc[0])
        end = float(series.iloc[-1])
        years = max(1, len(series) - 1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

# ---- Hämtning från Yahoo (namn, kurs, valuta, utdelning, CAGR) ----
def hamta_yahoo_fält(ticker: str) -> dict:
    out = {"Bolagsnamn": "", "Aktuell kurs": 0.0, "Valuta": "USD", "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0}
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

        if "dividendRate" in info and info["dividendRate"] is not None:
            out["Årlig utdelning"] = float(info["dividendRate"])

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---- Beräkningar (P/S-snitt, omsättning år2/3 med clamp, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    for i, rad in df.iterrows():
        # P/S-snitt = snitt av positiva Q1–Q4
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: >100% → 50%, <0% → 2%
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # lämna ev. befintliga
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0)
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))     * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0)) * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])      * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# ---- Massuppdatera från Yahoo (1s delay, kopierbar felrapport) ----
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

            if "Årlig utdelning" in data: df.at[i, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            else: failed_fields.append("Årlig utdelning")

            if "CAGR 5 år (%)" in data: df.at[i, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)
            else: failed_fields.append("CAGR 5 år (%)")

            if failed_fields:
                misslyckade.append(f"{tkr}: {', '.join(failed_fields)}")

            time.sleep(1.0)
            bar.progress((i+1)/total)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")
    return df

# ---- Lägg till / uppdatera bolag ----
MANUELL_FALT_FOR_DATUM = ["P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år"]

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)", "Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum", "Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"])
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    tickers = [r["Ticker"] for _, r in vis_df.iterrows()]

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    if etiketter:
        st.session_state.edit_index = min(st.session_state.edit_index, len(etiketter)-1)

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", [""] + etiketter, index=0 if not etiketter else st.session_state.edit_index + 1)
    # bläddringsknappar
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            if etiketter:
                st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index+1 if etiketter else 0}/{len(etiketter)}")
    with col_next:
        if st.button("➡️ Nästa"):
            if etiketter:
                st.session_state.edit_index = min(len(etiketter)-1, st.session_state.edit_index + 1)

    if valt_label and etiketter:
        # justera index efter eventuella hopp via selectbox
        try:
            st.session_state.edit_index = etiketter.index(valt_label)
        except ValueError:
            pass

    bef = pd.Series({}, dtype=object)
    if etiketter and valt_label:
        tkr = tickers[st.session_state.edit_index]
        match = df[df["Ticker"] == tkr]
        if not match.empty:
            bef = match.iloc[0]

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker", "") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier", 0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier", 0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",    value=float(bef.get("P/S", 0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)",  value=float(bef.get("Omsättning idag", 0.0)) if not bef.empty else 0.0)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år", 0.0)) if not bef.empty else 0.0)

            st.markdown("**Uppdateras automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas om")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f, 0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f, 0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f, 0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"] == ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"] == ticker, "Senast manuellt uppdaterad"] = now_stamp()

        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[df["Ticker"] == ticker, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[df["Ticker"] == ticker, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs", 0) > 0: df.loc[df["Ticker"] == ticker, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data: df.loc[df["Ticker"] == ticker, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:   df.loc[df["Ticker"] == ticker, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")

    # Tipslista: äldst manuell uppdatering
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    tmp = df.copy()
    tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = tmp.sort_values(by=["_sort_datum", "Bolagsnamn"]).head(10)
    st.dataframe(tips[[
        "Ticker","Bolagsnamn","Senast manuellt uppdaterad",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"
    ]], use_container_width=True)

    return df

# ---- Analysvy ----
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    if etiketter:
        st.session_state.analys_idx = min(st.session_state.analys_idx, len(etiketter)-1)

    # välj via index eller lista
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx, step=1
    )
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx if etiketter else 0, key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx + 1)

    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")

    if len(vis_df) > 0 and etiketter:
        r = vis_df.iloc[st.session_state.analys_idx]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = [
            "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Senast manuellt uppdaterad"
        ]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

# ---- Portfölj ----
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
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
              "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

# ---- Investeringsförslag ----
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox("Vilken riktkurs ska användas?",
                                ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
                                index=1)

    subset = st.radio("Vilka bolag?", ["Alla bolag", "Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential", "Närmast riktkurs"], horizontal=True)

    base = df.copy() if subset == "Alla bolag" else df[df["Antal aktier"] > 0].copy()
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

    # Portföljdata i SEK (för andelar)
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
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2)   if port_värde > 0 else 0.0

    # Presentationskort (visa alla riktkurser, fetmarkera vald)
    def mark(line, use_bold): return f"**{line}**" if use_bold else line
    lines = []
    lines.append(f"Aktuell kurs: {round(rad['Aktuell kurs'],2)} {rad['Valuta']}")
    for lbl in ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]:
        val = round(rad[lbl], 2)
        lines.append(mark(f"{lbl}: {val} {rad['Valuta']}", lbl == riktkurs_val))
    lines.append(f"Uppsida (valda riktkursen): {round(rad['Potential (%)'],2)} %")
    lines.append(f"Antal att köpa för {int(kapital_sek)} SEK: {antal_köp} st")
    lines.append(f"Nuvarande andel: {nuv_andel} %")
    lines.append(f"Andel efter köp: {ny_andel} %")

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown("- " + "\n- ".join(lines))

# ---- MAIN ----
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Sidopanel: valutakurser (persistenta) ---
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = las_sparade_valutakurser()
    # Starta med sparade, fallback till standard
    start_rates = {k: saved_rates.get(k, STANDARD_VALUTAKURSER[k]) for k in ["USD","NOK","CAD","EUR","SEK"]}

    usd = st.sidebar.number_input("USD → SEK", value=float(start_rates["USD"]), step=0.01, key="usd")
    nok = st.sidebar.number_input("NOK → SEK", value=float(start_rates["NOK"]), step=0.01, key="nok")
    cad = st.sidebar.number_input("CAD → SEK", value=float(start_rates["CAD"]), step=0.01, key="cad")
    eur = st.sidebar.number_input("EUR → SEK", value=float(start_rates["EUR"]), step=0.01, key="eur")
    sek = st.sidebar.number_input("SEK → SEK", value=float(start_rates["SEK"]), step=0.01, key="sek")

    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": sek}

    if st.sidebar.button("💾 Spara valutakurser"):
        spara_valutakurser(user_rates)

    # Global massuppdatering i sidopanelen (anropar i slutet av laddning)
    # Läs data
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Massuppdateringsknapp i sidopanelen
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        # säkerställ färska beräkningar
        df = uppdatera_berakningar(df, user_rates)
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
