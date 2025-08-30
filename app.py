import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
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
RATES_SHEET_NAME = "Valutor"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def get_spreadsheet(url: str):
    return client.open_by_url(url)

def skapa_koppling():
    ss = get_spreadsheet(SHEET_URL)
    return ss.worksheet(SHEET_NAME)

# ---- Backoff-wrapper för Google API-anrop ----
def _with_backoff(func, *args, **kwargs):
    delay = 0.8
    for _ in range(6):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            time.sleep(delay)
            delay *= 1.7
    # sista försök – låt exception bubbla upp
    return func(*args, **kwargs)

# ---- Läs & spara huvudarket ----
def hamta_data():
    sheet = skapa_koppling()
    try:
        data = _with_backoff(sheet.get_all_records)
        df = pd.DataFrame(data)
        return df
    except Exception:
        # Om arket är tomt – returnera tom df (skapas i main)
        return pd.DataFrame({})

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    _with_backoff(sheet.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Standard valutakurser till SEK (kan justeras & sparas) ----
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

# ---- Valutablad: säkerställ header & robust läsning ----
def _hamta_rates_sheet():
    ss = get_spreadsheet(SHEET_URL)
    try:
        ws = _with_backoff(ss.worksheet, RATES_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws = _with_backoff(ss.add_worksheet, title=RATES_SHEET_NAME, rows=10, cols=4)
        _with_backoff(ws.update, [["Valuta", "Kurs", "Sparad", "Av"]])
        return ws

    # Säkerställ header om bladet är tomt eller trasigt
    try:
        vals = _with_backoff(ws.get_all_values)
    except Exception:
        vals = []

    if not vals or len(vals) == 0 or not vals[0] or vals[0][:2] != ["Valuta", "Kurs"]:
        _with_backoff(ws.update, [["Valuta", "Kurs", "Sparad", "Av"]])
    return ws

@st.cache_data(ttl=3600)
def las_sparade_valutakurser_cached(cache_buster: int) -> dict:
    ws = _hamta_rates_sheet()

    # Försök med get_all_records
    try:
        rows = _with_backoff(ws.get_all_records)
    except gspread.exceptions.GSpreadException:
        # Fallback: get_all_values och bygg dicts
        try:
            vals = _with_backoff(ws.get_all_values)
        except Exception:
            vals = []

        if not vals or len(vals) < 2:
            return {}

        header = vals[0]
        data_rows = vals[1:]
        rows = []
        for r in data_rows:
            d = {header[i]: (r[i] if i < len(r) else "") for i in range(len(header))}
            rows.append(d)
    except Exception:
        return {}

    out = {}
    for r in rows:
        val = str(r.get("Valuta", "")).strip().upper()
        raw_kurs = str(r.get("Kurs", "")).replace(",", ".").strip()
        try:
            kurs = float(raw_kurs)
        except Exception:
            continue
        if val in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            out[val] = kurs
    return out

def las_sparade_valutakurser() -> dict:
    # använd session-buster för att forcera omläsning efter spar
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    ws = _hamta_rates_sheet()
    rows = [["Valuta", "Kurs", "Sparad", "Av"]]
    idag = now_stamp()
    for val in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        kurs = float(rates.get(val, STANDARD_VALUTAKURSER.get(val, 1.0)))
        rows.append([val, kurs, idag, ""])
    # skriv om hela lilla bladet
    _with_backoff(ws.clear)
    _with_backoff(ws.update, rows)
    # bump cache
    st.session_state["rates_reload"] = int(time.time())

# ---- Kolumnschema (huvudarket) ----
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad"
]

# Manuell-fält som triggar datumstämpling vid förändring
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år"
]

# ---- Kolumnsäkring, migrering, typer ----
def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({c: [] for c in FINAL_COLS})
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    # Mappa ev. gamla kolumner → nya
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
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# ---- Valutakurs-hjälpare (till SEK när vi behöver det, t.ex. i portfölj) ----
def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    try:
        # Nya yfinance
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            # Fallback: gamla attributet
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0

        if series.empty or len(series) < 2:
            return 0.0

        # sortera kronologiskt
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

# ---- Hämtning från Yahoo (namn, kurs, valuta, utdelning, CAGR) ----
def hamta_yahoo_fält(ticker: str) -> dict:
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice", None)
        if pris is None:
            # fallback via historik
            h = t.history(period="5d")
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

        # Årlig utdelning per aktie (kan vara None/0)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate or 0.0)

        # CAGR 5 år (%), beräknad från Total Revenue
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---- Beräkningar (P/S-snitt, omsättning år2/3 med clamp, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    for i, rad in df.iterrows():
        # P/S-snitt: snitt av positiva Q1–Q4
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
            # lämna befintliga om redan ifyllda, annars 0
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (i aktiens egen valuta)
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
        misslyckade = []   # ["TICKER: fält1, fält2 ..."]
        total = len(df)
        for i, row in df.iterrows():
            tkr = str(row.get("Ticker","")).strip()
            if not tkr:
                continue
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)
            failed_fields = []

            if data.get("Bolagsnamn"):
                df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
            else:
                failed_fields.append("Bolagsnamn")

            if data.get("Aktuell kurs", 0) > 0:
                df.at[i, "Aktuell kurs"] = data["Aktuell kurs"]
            else:
                failed_fields.append("Aktuell kurs")

            if data.get("Valuta"):
                df.at[i, "Valuta"] = data["Valuta"]
            else:
                failed_fields.append("Valuta")

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

            time.sleep(1.0)
            bar.progress(int((i+1)/max(1,total)*100))

        # Beräkna om efter hämtning
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# ---- Konstanter för datumspårning av manuell uppdatering ----
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år"
]

# ---- Lägg till / uppdatera bolag ----
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringshjälp för rullistan
    sort_val = st.selectbox(
        "Sortera för redigering",
        ["A–Ö (bolagsnamn)", "Äldst manuell uppdatering först"]
    )
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum","Bolagsnamn","Ticker"]).drop(columns=["_sort_datum"], errors="ignore")
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    ticker_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r["Ticker"] for _, r in vis_df.iterrows()}

    # Init index i session
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(etiketter)-1))

    # Rullista + bläddringsknappar
    valt_label = st.selectbox(
        "Välj bolag (lämna tomt för nytt)",
        [""] + etiketter,
        index=(st.session_state.edit_index+1 if etiketter else 0)
    )
    c_prev, c_mid, c_next = st.columns([1,2,1])
    with c_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with c_mid:
        st.write(f"Post {st.session_state.edit_index+1}/{max(1,len(etiketter))}")
    with c_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(etiketter)-1, st.session_state.edit_index + 1)

    # Hämta befintlig rad
    if valt_label and valt_label in ticker_map:
        bef = df[df["Ticker"] == ticker_map[valt_label]].iloc[0]
    elif etiketter and st.session_state.edit_index < len(etiketter):
        # Om användaren bara bläddrar utan att välja i rullistan
        label = etiketter[st.session_state.edit_index]
        bef = df[df["Ticker"] == ticker_map[label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0, step=1.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0, step=1.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, step=0.01, format="%.2f")
            ps1 = st.number_input("P/S Q1",value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, step=0.01, format="%.2f")
            ps2 = st.number_input("P/S Q2",value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, step=0.01, format="%.2f")
            ps3 = st.number_input("P/S Q3",value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, step=0.01, format="%.2f")
            ps4 = st.number_input("P/S Q4",value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, step=0.01, format="%.2f")
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0, step=1.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0, step=1.0)

            st.markdown("**Hämtas automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas om")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    # Spara + hämta Yahoo
    if spar and ticker:
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Sätt datum om manuellfält ändrats
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # Skriv till df
        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"] == ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"] == ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta från Yahoo
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"):       df.loc[df["Ticker"]==ticker, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):           df.loc[df["Ticker"]==ticker, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[df["Ticker"]==ticker, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data:    df.loc[df["Ticker"]==ticker, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:      df.loc[df["Ticker"]==ticker, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

        # Beräkna & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")

    # Tips: äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    tmp = df.copy()
    tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = tmp.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10).drop(columns=["_sort_datum"])
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )

    return df

# ---- Analysvy ----
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(st.session_state.analys_idx, max(0, len(etiketter)-1))

    # Välj / bläddra
    valt = st.selectbox("Välj bolag", etiketter if etiketter else ["—"])
    if etiketter:
        st.session_state.analys_idx = etiketter.index(valt)

    c_prev, c_mid, c_next = st.columns([1,2,1])
    with c_prev:
        if st.button("⬅️ Föregående", key="analys_prev_btn"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
    with c_mid:
        st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")
    with c_next:
        if st.button("➡️ Nästa", key="analys_next_btn"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)

    if len(vis_df) > 0:
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

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde, 2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd, 2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd / 12.0, 2)} SEK")

    st.dataframe(
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
              "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

# ---- Investeringsförslag ----
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge   = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    # Filtergrund
    if subset == "Endast portfölj":
        base = df[df["Antal aktier"] > 0].copy()
    else:
        base = df.copy()

    # Kräver giltiga värden
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Potential (%)
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    # Avvikelse mot mål (%): + = över mål, - = under mål
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Bläddring
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    c_prev, c_mid, c_next = st.columns([1,2,1])
    with c_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with c_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with c_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljvärde i SEK för andelsberäkning
    port = df[df["Antal aktier"] > 0].copy()
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
    else:
        port_värde = 0.0

    # Köpförslag i SEK (kurs i egen valuta * växelkurs)
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

    # Presentationskort (valda riktkursen markeras)
    def mark(label):  # fetmarkera om vald
        return "**⬅ vald**" if label == riktkurs_val else ""

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {mark('Riktkurs idag')}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {mark('Riktkurs om 1 år')}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {mark('Riktkurs om 2 år')}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {mark('Riktkurs om 3 år')}
- **Uppsida (vald riktkurs):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# ---- main ----
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Läs data från Google Sheets
    df = hamta_data()
    if df.empty:
        # Om arket skulle vara tomt – skapa mall så att appen inte kraschar
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # 2) Säkerställ schema och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 3) Sidopanel: valutakurser (med försök att läsa sparade kurser)
    st.sidebar.header("💱 Valutakurser → SEK")

    # Försök hämta sparade kurser från Rates-bladet; falla tillbaka till standard om något går fel
    try:
        saved_rates = las_sparade_valutakurser()
        if not isinstance(saved_rates, dict):
            saved_rates = {}
    except Exception:
        saved_rates = {}

    # slå ihop: sparat > standard
    _base_rates = STANDARD_VALUTAKURSER.copy()
    _base_rates.update(saved_rates)

    user_rates = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(_base_rates.get("USD", 9.75)), step=0.01, format="%.4f", key="usd"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(_base_rates.get("NOK", 0.95)), step=0.01, format="%.4f", key="nok"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(_base_rates.get("CAD", 7.05)), step=0.01, format="%.4f", key="cad"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(_base_rates.get("EUR", 11.18)), step=0.01, format="%.4f", key="eur"),
    }

    # Knappar i sidopanel: spara valutakurser + global Yahoo-uppdatering
    c_rates_save, c_spacer, c_dummy = st.sidebar.columns([1,0.1,1])
    with c_rates_save:
        if st.button("💾 Spara valutakurser"):
            try:
                spara_valutakurser(user_rates)
                st.sidebar.success("Valutakurser sparade i arket!")
            except Exception as e:
                st.sidebar.warning(f"Kunde inte spara valutakurser: {e}")

    # Global massuppdatering från Yahoo (1s delay, felrapport)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # 4) Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    # 5) Växla vy
    if meny == "Analys":
        # Beräkna alltid innan visning (ifall något just ändrats via massuppdatera)
        df = uppdatera_berakningar(df, user_rates)
        analysvy(df, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
        # (lagg_till_eller_uppdatera sköter spara_data när man trycker Spara)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

    # 6) Extra: visa en liten statusrad längst ned
    st.caption("✅ Appen är igång. Uppdatera gärna från Yahoo i sidopanelen efter att du gjort manuella ändringar.")
    

if __name__ == "__main__":
    main()
