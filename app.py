# =========================
# app.py — Del 1/4
# (Import, konstanter, Google Sheets I/O, valutablad, kolumnschema)
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from gspread.exceptions import WorksheetNotFound
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
SHEET_URL = st.secrets["SHEET_URL"]           # ditt huvuddokument
SHEET_NAME = "Blad1"                          # databladet
VALUTA_SHEET_NAME = "Valutor"                 # nytt blad där valutakurser sparas

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data() -> pd.DataFrame:
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Standard valutakurser till SEK (fallback) ----
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
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))

# ---- Valutablad: läs/spara persistenta kurser i separat sheet ----
def _hamta_valutablad():
    ss = client.open_by_url(SHEET_URL)
    try:
        ws = ss.worksheet(VALUTA_SHEET_NAME)
    except WorksheetNotFound:
        ws = ss.add_worksheet(title=VALUTA_SHEET_NAME, rows=10, cols=3)
        rows = [["Valuta", "SEK"]]
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            rows.append([k, STANDARD_VALUTAKURSER.get(k, 1.0)])
        ws.update(rows)
    vals = ws.get_all_values()
    if not vals or vals[0][:2] != ["Valuta", "SEK"]:
        ws.clear()
        rows = [["Valuta", "SEK"]]
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            rows.append([k, STANDARD_VALUTAKURSER.get(k, 1.0)])
        ws.update(rows)
    return ws

def lasa_valutakurser_fran_sheet() -> dict:
    try:
        ws = _hamta_valutablad()
        records = ws.get_all_records()
        out = {}
        for r in records:
            kod = str(r.get("Valuta", "")).strip().upper()
            try:
                sek = float(r.get("SEK", ""))
            except Exception:
                continue
            if kod:
                out[kod] = sek
        # fyll ut med standard där det saknas
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            out[k] = float(out.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))
        return out
    except Exception:
        return STANDARD_VALUTAKURSER.copy()

def spara_valutakurser_till_sheet(rates: dict):
    ws = _hamta_valutablad()
    rows = [["Valuta", "SEK"]]
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        rows.append([k, float(rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))])
    ws.clear()
    ws.update(rows)

# ---- Kolumnschema (ska alltid finnas i databasen) ----
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

# =========================
# app.py — Del 2/4
# (Yahoo-hämtning, CAGR, beräkningar)
# =========================

# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """Beräkna CAGR ~5 år från 'Total Revenue' (årsdata) om möjligt.
       Returnerar procent (t.ex. 12.34) eller 0.0 vid fel/brist."""
    try:
        # Nya yfinance: income_stmt, fallback: financials
        df_is = getattr(tkr, "income_stmt", None)
        series = None
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()

        if series is None or series.empty or len(series) < 2:
            return 0.0

        # sortera kronologiskt (äldst → nyast)
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
    """Hämta Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%)."""
    result = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)

        # info kan ibland kasta, så fånga separat
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            # fallback: senaste stängning
            hist = t.history(period="1d")
            if not hist.empty and "Close" in hist:
                pris = float(hist["Close"].iloc[-1])
        if pris is not None:
            result["Aktuell kurs"] = float(pris)

        # Valuta
        val = info.get("currency", None)
        if val:
            result["Valuta"] = str(val).upper()

        # Namn
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            result["Bolagsnamn"] = str(namn)

        # Utdelning per aktie (årstakt om tillgängligt)
        if "dividendRate" in info and info["dividendRate"] is not None:
            try:
                result["Årlig utdelning"] = float(info["dividendRate"])
            except Exception:
                pass

        # CAGR 5 år (%)
        result["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

    except Exception:
        # lämna tomma/defaults
        pass

    return result

# ---- Beräkningar (P/S-snitt, omsättning år2/3 (CAGR clamp), riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Uppdatera P/S-snitt, Omsättning om 2/3 år (från 'Omsättning nästa år' + CAGR-clamp),
       samt riktkurser. Inga valutakonverteringar i riktkurserna (allt i aktiens valuta)."""
    for i, rad in df.iterrows():
        # 1) P/S-snitt = snitt av positiva Q1–Q4
        ps_vals = [
            rad.get("P/S Q1", 0), rad.get("P/S Q2", 0),
            rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)
        ]
        ps_clean = [float(x) for x in ps_vals if pd.notna(x) and float(x) > 0]
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # 2) CAGR clamp för framtidsberäkningar
        #    >100% → 50%; <0% → 2%; annars oförändrad
        cagr_pct = float(rad.get("CAGR 5 år (%)", 0.0))
        if cagr_pct > 100.0:
            just_cagr = 50.0
        elif cagr_pct < 0.0:
            just_cagr = 2.0
        else:
            just_cagr = cagr_pct
        g = just_cagr / 100.0

        # 3) Omsättning om 2 & 3 år utifrån "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # lämna befintliga om redan ifyllda
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # 4) Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            oms_idag   = float(rad.get("Omsättning idag", 0.0))
            oms_1y     = float(rad.get("Omsättning nästa år", 0.0))
            oms_2y     = float(df.at[i, "Omsättning om 2 år"])
            oms_3y     = float(df.at[i, "Omsättning om 3 år"])

            df.at[i, "Riktkurs idag"]    = round((oms_idag * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((oms_1y   * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((oms_2y   * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((oms_3y   * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# =========================
# app.py — Del 3/4
# (Massuppdatering, formulär + datumlogik, analysvy)
# =========================

# Vilka fält räknas som "manuellt uppdaterade" för datumstämpel
MANUELL_FALT_FOR_DATUM = [
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år"
]

def _parse_float(text: str) -> float:
    """Tillåt tomma fält och komma → punkt. Tomt = 0.0."""
    if text is None:
        return 0.0
    s = str(text).strip().replace(" ", "").replace(",", ".")
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    """Sidopanel-knapp som hämtar Yahoo-fält för alla bolag (1s delay).
       Skriver INTE 'Senast manuellt uppdaterad'."""
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

            time.sleep(1.0)  # 1s paus
            bar.progress((i+1)/max(1,total))

        # Beräkna om efter hämtning
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade),
                                 height=160, key=f"{key_prefix}_miss")

    return df

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Formulär för att lägga till/uppdatera enskilt bolag.
       - Datumstämpel sätts ENDAST om manuell-fält ändrats.
       - Hämtar Yahoo-fält och räknar om riktkurser efter spar."""
    st.header("➕ Lägg till / uppdatera bolag")

    # Sortering för rullista
    sort_val = st.selectbox("Sortera för redigering",
                            ["A–Ö (bolagsnamn)", "Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        tmp = df.copy()
        tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = tmp.sort_values(by=["_sort_datum", "Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"])

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    till_ticker = {f"{r['Bolagsnamn']} ({r['Ticker']})": r["Ticker"] for _, r in vis_df.iterrows()}

    # Bläddringsindex i sessionen
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(etiketter)-1))

    # Rullista + bläddringsknappar
    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", [""] + etiketter,
                              index=0 if len(etiketter)==0 else min(st.session_state.edit_index+1, len(etiketter)))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {0 if len(etiketter)==0 else st.session_state.edit_index+1}/{max(1, len(etiketter))}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(max(0, len(etiketter)-1), st.session_state.edit_index + 1)

    # Hämta befintlig rad
    if valt_label and valt_label in till_ticker:
        ticker_vald = till_ticker[valt_label]
        bef = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)",
                                   value=bef.get("Ticker", "") if not bef.empty else "").upper()

            utest = _parse_float(st.text_input(
                "Utestående aktier (miljoner)",
                value=str(bef.get("Utestående aktier", 0.0)) if not bef.empty else ""
            ))
            antal = _parse_float(st.text_input(
                "Antal aktier du äger",
                value=str(bef.get("Antal aktier", 0.0)) if not bef.empty else ""
            ))

            ps  = _parse_float(st.text_input("P/S",    value=str(bef.get("P/S", 0.0)) if not bef.empty else ""))
            ps1 = _parse_float(st.text_input("P/S Q1", value=str(bef.get("P/S Q1", 0.0)) if not bef.empty else ""))
            ps2 = _parse_float(st.text_input("P/S Q2", value=str(bef.get("P/S Q2", 0.0)) if not bef.empty else ""))
            ps3 = _parse_float(st.text_input("P/S Q3", value=str(bef.get("P/S Q3", 0.0)) if not bef.empty else ""))
            ps4 = _parse_float(st.text_input("P/S Q4", value=str(bef.get("P/S Q4", 0.0)) if not bef.empty else ""))
        with c2:
            oms_idag = _parse_float(st.text_input(
                "Omsättning idag (miljoner)",
                value=str(bef.get("Omsättning idag", 0.0)) if not bef.empty else ""
            ))
            oms_next = _parse_float(st.text_input(
                "Omsättning nästa år (miljoner)",
                value=str(bef.get("Omsättning nästa år", 0.0)) if not bef.empty else ""
            ))

            st.markdown("**Uppdateras automatiskt vid spar:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas om")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # Ny data (manuellt styrda fält)
        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Har manuellfält ändrats? → datum
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f, 0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f, 0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f, 0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # Skriv ny data i df
        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"] == ticker, k] = v
        else:
            # Skapa tom rad med alla kolumner
            tom = {c: (0.0 if c not in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad"] else "")
                   for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"] == ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta Yahoo för just detta bolag
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"):
            df.loc[df["Ticker"] == ticker, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):
            df.loc[df["Ticker"] == ticker, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs", 0) > 0:
            df.loc[df["Ticker"] == ticker, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data:
            df.loc[df["Ticker"] == ticker, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:
            df.loc[df["Ticker"] == ticker, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

        # Räkna om och spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")

    # Tipslista: äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    tmp2 = df.copy()
    tmp2["_sort_datum"] = tmp2["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = tmp2.sort_values(by=["_sort_datum", "Bolagsnamn"]).head(10)
    st.dataframe(
        tips[[
            "Ticker", "Bolagsnamn", "Senast manuellt uppdaterad",
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
            "Omsättning idag", "Omsättning nästa år"
        ]],
        use_container_width=True
    )

    return df

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    """Analys: visa ett valt bolag (med bläddring) och hela databasen nedan."""
    st.header("📈 Analys")

    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(st.session_state.analys_idx, max(0, len(etiketter)-1))

    # Välj via rullista eller bläddra
    if etiketter:
        st.selectbox("Välj bolag", etiketter, index=st.session_state.analys_idx, key="analys_select")
    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(max(0, len(etiketter)-1), st.session_state.analys_idx + 1)

    st.write(f"Post { (st.session_state.analys_idx+1) if etiketter else 0 }/{ max(1, len(etiketter)) }")

    # Visa vald rad
    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        cols = [
            "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs", "Utestående aktier",
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
            "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
            "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
            "CAGR 5 år (%)", "Antal aktier", "Årlig utdelning", "Senast manuellt uppdaterad"
        ]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

# =========================
# app.py — Del 4/4
# (Valutablad, Portfölj, Investeringsförslag, main)
# =========================

# --- Valutablad i samma Spreadsheet ---
def _get_spreadsheet():
    # Använd samma klient/URL som för huvudbladet
    return client.open_by_url(SHEET_URL)

def get_valutablad():
    """Säkerställ att ett blad 'Valutakurser' finns, annars skapa med rubriker."""
    ss = _get_spreadsheet()
    try:
        sh = ss.worksheet("Valutakurser")
    except gspread.exceptions.WorksheetNotFound:
        sh = ss.add_worksheet(title="Valutakurser", rows=200, cols=6)
        sh.update([["Datum", "USD", "NOK", "CAD", "EUR", "SEK"]])
        # Skriv standardraden
        sh.append_row([now_stamp(),
                       STANDARD_VALUTAKURSER["USD"],
                       STANDARD_VALUTAKURSER["NOK"],
                       STANDARD_VALUTAKURSER["CAD"],
                       STANDARD_VALUTAKURSER["EUR"],
                       STANDARD_VALUTAKURSER["SEK"]])
    return sh

def load_user_rates() -> dict:
    """Läs senaste sparade valutakurser från bladet 'Valutakurser'."""
    try:
        sh = get_valutablad()
        rows = sh.get_all_records()
        if not rows:
            return STANDARD_VALUTAKURSER.copy()
        last = rows[-1]
        rates = {
            "USD": _parse_float(last.get("USD", STANDARD_VALUTAKURSER["USD"])),
            "NOK": _parse_float(last.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
            "CAD": _parse_float(last.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
            "EUR": _parse_float(last.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
            "SEK": 1.0,  # SEK alltid 1
        }
        return rates
    except Exception:
        return STANDARD_VALUTAKURSER.copy()

def save_user_rates(rates: dict):
    """Append:a en ny rad med valutakurser i bladet 'Valutakurser'."""
    try:
        sh = get_valutablad()
        sh.append_row([
            now_stamp(),
            float(rates.get("USD", STANDARD_VALUTAKURSER["USD"])),
            float(rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])),
            float(rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])),
            float(rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])),
            1.0
        ])
    except Exception as e:
        st.warning(f"Kunde inte spara valutakurser: {e}")

# --- Portfölj ---
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2) if total_värde > 0 else 0.0
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

# --- Investeringsförslag ---
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = _parse_float(st.text_input("Tillgängligt kapital (SEK)", value="500"))

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    # Endast-portfölj-filter (på begäran)
    endast_port = st.checkbox("Visa endast bolag du äger", value=False)

    läge = st.radio("Sortering", ["Störst potential", "Närmast riktkurs"], horizontal=True)

    base = df.copy()
    if endast_port:
        base = base[base["Antal aktier"] > 0].copy()

    # Kräver riktkurs & aktuell kurs > 0
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Beräkna potential (visas i kortet) och diff till mål för "Närmast riktkurs"
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

    # Portföljdata i SEK (för andelar)
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx if vx > 0 else 0.0
    antal_köp = int(kapital_sek // kurs_sek) if kurs_sek > 0 else 0
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Presentationskort – visa ALLA riktkurser, fetmarkera vald
    def mark(line, chosen):
        return f"**{line}**" if chosen else line

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        "\n".join([
            f"- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}",
            mark(f"- Riktkurs idag: {round(rad['Riktkurs idag'],2)} {rad['Valuta']}",
                 riktkurs_val == "Riktkurs idag"),
            mark(f"- Riktkurs om 1 år: {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']}",
                 riktkurs_val == "Riktkurs om 1 år"),
            mark(f"- Riktkurs om 2 år: {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']}",
                 riktkurs_val == "Riktkurs om 2 år"),
            mark(f"- Riktkurs om 3 år: {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']}",
                 riktkurs_val == "Riktkurs om 3 år"),
            f"- **Uppsida (vald riktkurs): {round(rad['Potential (%)'],2)} %**",
            f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
            f"- **Nuvarande andel:** {nuv_andel} %",
            f"- **Andel efter köp:** {ny_andel} %",
        ])
    )

# --- main ---
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Läs data från Google Sheets
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # 2) Säkerställ schema/migrera/typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 3) Valutakurser i sidopanelen (persistens i bladet 'Valutakurser')
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = load_user_rates()
    usd_in = st.sidebar.text_input("USD → SEK", value=str(saved_rates["USD"]), key="usd_rate")
    nok_in = st.sidebar.text_input("NOK → SEK", value=str(saved_rates["NOK"]), key="nok_rate")
    cad_in = st.sidebar.text_input("CAD → SEK", value=str(saved_rates["CAD"]), key="cad_rate")
    eur_in = st.sidebar.text_input("EUR → SEK", value=str(saved_rates["EUR"]), key="eur_rate")

    user_rates = {
        "USD": _parse_float(usd_in),
        "NOK": _parse_float(nok_in),
        "CAD": _parse_float(cad_in),
        "EUR": _parse_float(eur_in),
        "SEK": 1.0,
    }

    if st.sidebar.button("💾 Spara valutakurser"):
        save_user_rates(user_rates)
        st.sidebar.success("Valutakurser sparade i bladet 'Valutakurser'.")

    # 4) Global massuppdateringsknapp i sidopanelen
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # 5) Meny
    meny = st.sidebar.radio("📌 Välj vy",
                            ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        # säkerställ färska beräkningar i vyer som visar siffror
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
