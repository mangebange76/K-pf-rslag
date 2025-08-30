# app.py — Del 1/4
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
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return get_spreadsheet().worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Standard valutakurser till SEK (kan justeras & sparas) ----
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# ---- Rates-blad: robust hantering ----
RATES_SHEET_NAME_PRIMARY = "Rates"
RATES_SHEET_NAME_ALT = "Valutor"

def get_or_create_rates_sheet():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME_PRIMARY)
    except gspread.WorksheetNotFound:
        pass
    try:
        return ss.worksheet(RATES_SHEET_NAME_ALT)
    except gspread.WorksheetNotFound:
        pass
    ws = ss.add_worksheet(title=RATES_SHEET_NAME_PRIMARY, rows=10, cols=2)
    ws.update("A1:B1", [["Valuta", "Kurs"]])
    return ws

def _coerce_float_sv(x):
    if x is None:
        return None
    s = str(x).strip().replace(" ", "")
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def las_sparade_valutakurser():
    """Läs sparade kurser från Rates-bladet. Returnerar dict. {} om inget hittas."""
    try:
        ws = get_or_create_rates_sheet()
        rates = {}
        rows = ws.get_all_records()  # kräver headers Valuta/Kurs
        if rows:
            for r in rows:
                code = str(r.get("Valuta", "")).strip().upper()
                val = _coerce_float_sv(r.get("Kurs", ""))
                if code and (val is not None):
                    rates[code] = val
        if not rates:
            raw = ws.get_values("A2:B")
            for row in raw:
                if len(row) < 2:
                    continue
                code = str(row[0]).strip().upper()
                val = _coerce_float_sv(row[1])
                if code and (val is not None):
                    rates[code] = val
        return rates
    except Exception:
        return {}

def spara_valutakurser(rates: dict):
    """Spara kurser i Rates-bladet, säkerställ headers."""
    ws = get_or_create_rates_sheet()
    headers = ws.row_values(1)
    if len(headers) < 2 or headers[0] != "Valuta" or headers[1] != "Kurs":
        ws.update("A1:B1", [["Valuta", "Kurs"]])

    order = ["USD", "NOK", "CAD", "EUR", "SEK"]
    data = []
    for code in order:
        if code in rates:
            val = rates[code]
            data.append([code, f"{float(val):.6f}"])
    if not data:
        data = [[k, f"{float(v):.6f}"] for k, v in rates.items()]

    ws.batch_clear(["A2:B100"])
    if data:
        ws.update(f"A2:B{1+len(data)}", data)

# ---- Kolumnschema ----
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
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

# app.py — Del 2/4
# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """
    Försök beräkna CAGR (5 år ~ bästa möjliga med tillgängliga år)
    från 'Total Revenue' i årsredovisningen via yfinance.
    Returnerar procenttal (t.ex. 12.34) eller 0.0 om ej möjligt.
    """
    try:
        # Nya yfinance: .income_stmt, fallback .financials
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

        # sortera kronologiskt (äldst→nyast)
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series) - 1)
        if start <= 0:
            return 0.0

        cagr = (end / start) ** (1.0 / years) - 1.0
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
        # yfinance .info kan ibland kasta; skydda
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        # Valuta
        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()

        # Namn
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        # Årlig utdelning (kan saknas)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        # CAGR 5 år (%)
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---- Beräkningar (P/S-snitt, omsättning år2/3 med clamp, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    - P/S-snitt = snitt av positiva (Q1..Q4)
    - CAGR clamp: >100% → 50%, <0% → 2%
    - Omsättning om 2 & 3 år baserat på 'Omsättning nästa år'
    - Riktkurser = (Omsättning * P/S-snitt) / Utestående aktier
    """
    for i, rad in df.iterrows():
        # 1) P/S-snitt
        ps_vals = [
            rad.get("P/S Q1", 0), rad.get("P/S Q2", 0),
            rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)
        ]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # 2) CAGR clamp
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # 3) Omsättning framåt från "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # lämna om befintligt, annars 0
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # 4) Riktkurser
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# ---- Massuppdatera från Yahoo (1s delay, kopierbar felrapport) ----
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    """
    Sidopanel-knapp som hämtar (Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år)
    för alla tickers. Beräknar om och sparar. Visar kopierbar felista.
    """
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []   # ["TICKER: fält1, fält2 ..."]
        total = len(df)

        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
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

            time.sleep(1.0)  # 1s paus mellan anrop
            bar.progress((i+1)/total)

        # Beräkna om efter hämtning
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# ---- Lista över manuell-fält som sätter datum ----
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

# ---------- Del 3: Yahoo-hämtning, CAGR, beräkningar, massuppdatering, formulär ----------

# Manuellfält som triggar "Senast manuellt uppdaterad"
MANUELL_FALT_FOR_DATUM = [
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år"
]

def berakna_cagr_fran_finansiella(tkr: yf.Ticker) -> float:
    """
    Försök räkna CAGR på total omsättning (Total Revenue) från yfinance årsdata.
    Returnerar procent (t.ex. 12.34), 0.0 vid fel/brist.
    """
    try:
        # Nya yfinance
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            # Äldre yfinance
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0

        if series.empty or len(series) < 2:
            return 0.0

        # Sortera kronologiskt (äldst -> nyast)
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


def hamta_yahoo_falt(ticker: str) -> dict:
    """
    Hämtar Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%) från yfinance.
    Fält som saknas fylls med rimliga default.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)

        # info kan kasta — wrappa
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        pris = info.get("regularMarketPrice")
        if pris is None:
            # fallback: senast stängning
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        # Valuta
        val = info.get("currency")
        if val:
            out["Valuta"] = str(val).upper()

        # Namn
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        # Utdelning per aktie (årstakt)
        div_rate = info.get("dividendRate")
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        # CAGR
        out["CAGR 5 år (%)"] = berakna_cagr_fran_finansiella(t)

    except Exception:
        # lämna defaults
        pass

    return out


def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    - P/S-snitt = snitt av P/S Q1- Q4 (endast positiva värden)
    - CAGR clamp: >100% → 50%, <0% → 2%
    - Omsättning om 2 & 3 år = 'Omsättning nästa år' * (1+g)^n
    - Riktkurser = Omsättning(n) * P/S-snitt / Utestående aktier
      (observera: ingen valutakonvertering här — allt i aktiens valuta)
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if pd.notna(x) and float(x) > 0]
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR-justering
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        if cagr > 100.0:
            just_cagr = 50.0
        elif cagr < 0.0:
            just_cagr = 2.0  # inflation/golv
        else:
            just_cagr = cagr
        g = just_cagr / 100.0

        # Omsättning n+2, n+3
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev. manuellt ifyllt
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser
        utest = float(rad.get("Utestående aktier", 0.0))
        if utest > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]   = round((float(rad.get("Omsättning idag", 0.0))     * ps_snitt) / utest, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0)) * ps_snitt) / utest, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])      * ps_snitt) / utest, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])      * ps_snitt) / utest, 2)
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df


def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    """
    Sidopanelknapp för att uppdatera ALLA bolag från Yahoo.
    - 1s delay per ticker
    - sammanställer kopierbar lista med fält som inte kunde hämtas
    - kör beräkningar & sparar
    """
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []  # ["TICKER: fält1, fält2, ..."]
        total = len(df) if len(df) > 0 else 1

        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip().upper()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            failed_fields = []

            data = hamta_yahoo_falt(tkr)

            # Skriv fält om de finns; annars notera miss
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

            # 1 sekund paus
            time.sleep(1.0)
            bar.progress((i + 1) / total)

        # Beräkna om och spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)

        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area(
                "Misslyckade fält (kopierbar)",
                "\n".join(misslyckade),
                height=180,
                key=f"{key_prefix}_miss"
            )

    return df


def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    Formulär för att lägga till eller uppdatera ett enskilt bolag.
    - Bläddringsknappar + rullista
    - Vid spara hämtas Yahoo-fält, beräkningar körs, datum stämplas om manuellfält ändrats, och allt sparas.
    """
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringshjälp/rullista + bläddring
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)", "Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum", "Bolagsnamn", "Ticker"]).reset_index(drop=True)
    else:
        vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    if etiketter:
        st.session_state.edit_index = min(st.session_state.edit_index, len(etiketter) - 1)

    # Rullista + bläddringsknappar
    valt_idx = st.selectbox(
        "Välj befintligt bolag (eller lämna tomt för nytt)",
        options=["<ny post>"] + etiketter,
        index=st.session_state.edit_index + 1 if etiketter else 0
    )

    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            if etiketter:
                st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_mid:
        if etiketter:
            st.write(f"Post {st.session_state.edit_index+1}/{len(etiketter)}")
        else:
            st.write("Ingen befintlig post")
    with col_next:
        if st.button("➡️ Nästa"):
            if etiketter:
                st.session_state.edit_index = min(len(etiketter) - 1, st.session_state.edit_index + 1)

    # Hämta befintlig rad
    if valt_idx != "<ny post>" and etiketter:
        bef = vis_df.iloc[st.session_state.edit_index]
    else:
        bef = pd.Series({}, dtype=object)

    # Form
    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker", "") if not bef.empty else "").upper()
            utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier", 0.0)) if not bef.empty else 0.0)
            antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier", 0.0)) if not bef.empty else 0.0)

            ps = st.number_input("P/S", value=float(bef.get("P/S", 0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag", 0.0)) if not bef.empty else 0.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år", 0.0)) if not bef.empty else 0.0)

            st.markdown("**Hämtas automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # Ny data från formuläret
        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
        }

        # Har manuellfält ändrats?
        datum_satt = False
        if not bef.empty:
            before = {f: float(bef.get(f, 0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after = {f: float(ny.get(f, 0.0)) for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_satt = True
        else:
            if any(float(ny.get(f, 0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_satt = True

        # Skriv in i df
        if not bef.empty:
            # Uppdatera den rad som matchar ticker (i original-DF, inte sorterad vy)
            df.loc[df["Ticker"] == bef["Ticker"], list(ny.keys())] = list(ny.values())
            # Om användaren bytt ticker på bef post:
            if bef["Ticker"] != ticker:
                # flytta även "Ticker"-fältet korrekt
                df.loc[df["Ticker"] == bef["Ticker"], "Ticker"] = ticker
        else:
            # skapa tom rad med alla kolumner, fyll sedan
            tom = {c: (0.0 if c not in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Datumstämpel (om manuellfälten ändrats)
        if datum_satt:
            df.loc[df["Ticker"] == ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta Yahoo-fält för detta ticker
        data = hamta_yahoo_falt(ticker)
        if data.get("Bolagsnamn"): df.loc[df["Ticker"] == ticker, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[df["Ticker"] == ticker, "Valuta"]     = data["Valuta"]
        if data.get("Aktuell kurs", 0) > 0: df.loc[df["Ticker"] == ticker, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data:       df.loc[df["Ticker"] == ticker, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:         df.loc[df["Ticker"] == ticker, "CAGR 5 år (%)"]   = float(data.get("CAGR 5 år (%)") or 0.0)

        # Kör beräkningar & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat & uppdaterat från Yahoo.")

    # Tips: äldst uppdaterade först
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    tmp = df.copy()
    tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = tmp.sort_values(by=["_sort_datum", "Bolagsnamn", "Ticker"]).head(10)
    st.dataframe(
        tips[[
            "Ticker", "Bolagsnamn", "Senast manuellt uppdaterad",
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
            "Omsättning idag", "Omsättning nästa år"
        ]],
        use_container_width=True
    )

    return df

# ---------- Del 4: Analys, Portfölj, Investeringsförslag och main ----------

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    # Sortera för visning
    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    # Bläddring + rullista
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    if etiketter:
        st.session_state.analys_idx = min(st.session_state.analys_idx, len(etiketter)-1)

    valt = st.selectbox(
        "Välj bolag",
        options=etiketter if etiketter else ["(tom databas)"],
        index=st.session_state.analys_idx if etiketter else 0,
        key="analys_selectbox"
    )

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="analys_prev_btn"):
            if etiketter:
                st.session_state.analys_idx = max(0, st.session_state.analys_idx-1)
                st.experimental_set_query_params(analys_idx=st.session_state.analys_idx)
    with col_mid:
        if etiketter:
            st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")
        else:
            st.write("Ingen post att visa")
    with col_next:
        if st.button("➡️ Nästa", key="analys_next_btn"):
            if etiketter:
                st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx+1)
                st.experimental_set_query_params(analys_idx=st.session_state.analys_idx)

    if etiketter:
        r = vis_df.iloc[st.session_state.analys_idx]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = [
            "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
            "Utestående aktier",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Senast manuellt uppdaterad"
        ]
        vis_cols = [c for c in cols if c in r.index]
        st.dataframe(pd.DataFrame([r[vis_cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)


def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Värdering i SEK (endast portföljvyer)
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round((port["Värde (SEK)"] / total_värde) * 100.0, 2) if total_värde > 0 else 0.0

    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[[
            "Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
            "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"
        ]],
        use_container_width=True
    )


def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0, key="kapital_sek")

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1,
        key="rk_val"
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True, key="subset_val")
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True, key="sort_mode")

    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()

    # Kräver riktkurs & aktuell kurs
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

    # Bläddring
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("⬅️ Föregående förslag", key="prev_prop"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with c2:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with c3:
        if st.button("➡️ Nästa förslag", key="next_prop"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljvärde i SEK (för andelsberäkning)
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    # Köpberäkning (konvertera endast för köpbelopp)
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

    # Presentation (allt i aktiens valuta för kurs/riktkurs; andelar i %)
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    def mark(vald): return "**⬅ vald**" if riktkurs_val == vald else ""
    st.markdown(
f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {mark("Riktkurs idag")}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {mark("Riktkurs om 1 år")}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {mark("Riktkurs om 2 år")}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {mark("Riktkurs om 3 år")}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Läs huvudark ---
    try:
        df = hamta_data()
    except Exception as e:
        st.error(f"Kunde inte läsa Google Sheet: {e}")
        return

    # Skapa tom mall om arket är tomt
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        try:
            spara_data(df)
        except Exception:
            pass

    # Säkerställ schema & typer (+ migrering av ev. gamla kolumner)
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Sidopanel: valutakurser -> SEK ---
    st.sidebar.header("💱 Valutakurser → SEK")

    # Försök läsa sparade kurser (om funktioner finns); annars default
    try:
        saved_rates = las_sparade_valutakurser()
    except Exception:
        saved_rates = {}

    def init_rate(code: str, fallback: float) -> float:
        try:
            return float(saved_rates.get(code, fallback))
        except Exception:
            return fallback

    user_rates = {
        "USD": st.sidebar.number_input("USD → SEK", value=init_rate("USD", STANDARD_VALUTAKURSER["USD"]), step=0.01, key="usd_in"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=init_rate("NOK", STANDARD_VALUTAKURSER["NOK"]), step=0.01, key="nok_in"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=init_rate("CAD", STANDARD_VALUTAKURSER["CAD"]), step=0.01, key="cad_in"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=init_rate("EUR", STANDARD_VALUTAKURSER["EUR"]), step=0.01, key="eur_in"),
    }

    col_sv, col_ld = st.sidebar.columns(2)
    with col_sv:
        if st.button("💾 Spara kurser", key="save_rates_btn"):
            try:
                spara_valutakurser(user_rates)
                st.sidebar.success("Valutakurser sparade.")
                # Invalidera ev. cache
                st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            except Exception as e:
                st.sidebar.error(f"Kunde inte spara kurser: {e}")
    with col_ld:
        if st.button("↻ Ladda sparade", key="load_rates_btn"):
            st.experimental_rerun()

    # Global massuppdateringsknapp (i sidopanelen)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # --- Meny ---
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"],
        index=0
    )

    if meny == "Analys":
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
