# =========================
# Del 1/4 – Importer, GSheet-koppling, valutakurser, kolumnschema
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound

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
VALUTA_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()  # FIX: inga walrus-operatorer
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def _get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def hamta_valutakurser_sheet() -> dict:
    """Läs användarens sparade valutakurser från arket 'Valutakurser' (skapa om saknas)."""
    ss = _get_spreadsheet()
    try:
        ws = ss.worksheet(VALUTA_SHEET_NAME)
    except WorksheetNotFound:
        ws = ss.add_worksheet(title=VALUTA_SHEET_NAME, rows=10, cols=2)
        ws.update([["Valuta", "SEK_kurs"], ["USD", "9.75"], ["NOK", "0.95"], ["CAD", "7.05"], ["EUR", "11.18"], ["SEK", "1.0"]])
    rows = ws.get_all_records()
    out = {}
    for r in rows:
        cur = str(r.get("Valuta","")).upper().strip()
        try:
            rate = float(str(r.get("SEK_kurs","")).replace(",", "."))
        except Exception:
            rate = None
        if cur and rate is not None:
            out[cur] = rate
    return out

def spara_valutakurser_sheet(rates: dict):
    """Skriv användarens valutakurser till arket 'Valutakurser'."""
    ss = _get_spreadsheet()
    try:
        ws = ss.worksheet(VALUTA_SHEET_NAME)
    except WorksheetNotFound:
        ws = ss.add_worksheet(title=VALUTA_SHEET_NAME, rows=10, cols=2)
    rows = [["Valuta", "SEK_kurs"]]
    for k, v in rates.items():
        rows.append([k.upper(), str(v)])
    ws.clear()
    ws.update(rows)

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
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

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

# =========================
# Del 2/4 – Yahoo-funktioner, CAGR, beräkningar, massuppdatering
# =========================

# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """Försök beräkna CAGR (5 år om möjligt) från 'Total Revenue' i income statement/financials."""
    try:
        # Nyare yfinance
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            # Fallback till äldre attribut
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0

        if series.empty or len(series) < 2:
            return 0.0

        # Säkerställ kronologisk ordning (äldst -> nyast)
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
    """Returnerar dict med Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%). Tomma om ej hittas."""
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

        # Kurs
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            # Fallback till senaste close
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

        # Årlig utdelning (kan vara None)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        # CAGR 5 år (%) – från finansiella rapporter
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass

    return out

# ---- Beräkningar (P/S-snitt, omsättning år2/3 med clamp, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Uppdatera P/S-snitt, CAGR-justering, omsättning om 2/3 år, samt riktkurser."""
    for i, rad in df.iterrows():
        # P/S-snitt: snitt av positiva P/S Q1–Q4
        ps_vals = [
            rad.get("P/S Q1", 0),
            rad.get("P/S Q2", 0),
            rad.get("P/S Q3", 0),
            rad.get("P/S Q4", 0),
        ]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: > 100% → 50%, < 0% → 2%
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år baserat på "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # lämna befintliga om redan ifyllda, annars 0
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och ps_snitt > 0)
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))   * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])        * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])        * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# ---- Massuppdatera från Yahoo (1s delay, kopierbar felrapport) ----
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    """Sidopanelknapp för att uppdatera alla rader från Yahoo. Returnerar ev. uppdaterad df."""
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []   # Lista med "TICKER: fält1, fält2 ..."
        total = len(df) if len(df) > 0 else 1

        for i, row in df.iterrows():
            tkr = str(row["Ticker"]).strip()
            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)
            failed_fields = []

            # Sätt bara om något faktiskt kom
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

            # Årlig utdelning kan vara 0 – notera som miss bara om fältet helt saknas
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

            time.sleep(1.0)  # paus mellan anrop
            bar.progress((i + 1) / total)

        # Beräkna om och spara efter massuppdatering
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# =========================
# Del 3/4 – Formulär, Analys, Portfölj, Investeringsförslag
# =========================

# Vilka fält räknas som "manuellt uppdaterade" (för datumstämpel)
MANUELL_FALT_FOR_DATUM = [
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år",
]

def _fmt_blank(v) -> str:
    """Visa tomt istället för 0.0 i textfält."""
    try:
        f = float(v)
        return "" if f == 0.0 else str(f)
    except Exception:
        return str(v) if v not in (None, "nan") else ""

def _parse_float(txt: str) -> float:
    """Tillåt tomt, komma, punkt; returnera float (0.0 om tomt/ogiltigt)."""
    if txt is None:
        return 0.0
    s = str(txt).strip().replace(" ", "").replace(",", ".")
    if s == "":
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

# ---- Lägg till / uppdatera bolag ----
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringshjälp för rullistan
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

    # Välj rad via lista
    valt_label = st.selectbox(
        "Välj bolag (lämna tomt för nytt)",
        [""] + etiketter,
        index=(st.session_state.edit_index + 1) if etiketter else 0,
        key="edit_select"
    )

    # Bläddringsknappar
    col_prev, col_pos, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            if etiketter:
                st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
                st.session_state.edit_select = etiketter[st.session_state.edit_index]
    with col_pos:
        st.write(f"Post {st.session_state.edit_index + 1}/{len(etiketter) if etiketter else 1}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            if etiketter:
                st.session_state.edit_index = min(len(etiketter) - 1, st.session_state.edit_index + 1)
                st.session_state.edit_select = etiketter[st.session_state.edit_index]

    # Plocka aktuell rad (om någon vald)
    if valt_label and etiketter:
        try:
            idx = etiketter.index(valt_label)
            bef = vis_df.iloc[idx]
        except ValueError:
            bef = pd.Series({}, dtype=object)
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)

        # ---- Kolumn 1: text_inputs för att enkelt kunna tömma värden ----
        with c1:
            ticker = st.text_input(
                "Ticker (Yahoo-format)",
                value=(bef.get("Ticker", "") if not bef.empty else "")
            ).upper()

            utest_txt = st.text_input("Utestående aktier (miljoner)", value=_fmt_blank(bef.get("Utestående aktier", "")) if not bef.empty else "")
            antal_txt = st.text_input("Antal aktier du äger", value=_fmt_blank(bef.get("Antal aktier", "")) if not bef.empty else "")

            ps_txt  = st.text_input("P/S",    value=_fmt_blank(bef.get("P/S", "")) if not bef.empty else "")
            ps1_txt = st.text_input("P/S Q1", value=_fmt_blank(bef.get("P/S Q1", "")) if not bef.empty else "")
            ps2_txt = st.text_input("P/S Q2", value=_fmt_blank(bef.get("P/S Q2", "")) if not bef.empty else "")
            ps3_txt = st.text_input("P/S Q3", value=_fmt_blank(bef.get("P/S Q3", "")) if not bef.empty else "")
            ps4_txt = st.text_input("P/S Q4", value=_fmt_blank(bef.get("P/S Q4", "")) if not bef.empty else "")

        # ---- Kolumn 2: omsättningar (text_inputs) + info ----
        with c2:
            oms_idag_txt  = st.text_input("Omsättning idag (miljoner)",  value=_fmt_blank(bef.get("Omsättning idag", "")) if not bef.empty else "")
            oms_next_txt  = st.text_input("Omsättning nästa år (miljoner)", value=_fmt_blank(bef.get("Omsättning nästa år", "")) if not bef.empty else "")

            st.markdown("**Fält som hämtas/beräknas automatiskt vid spar:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- P/S-snitt, Omsättning om 2 & 3 år, samt alla riktkurser")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    # ---- Vid spar ----
    if spar and ticker:
        # Parsea inmatning
        utest = _parse_float(utest_txt)
        antal = _parse_float(antal_txt)

        ps   = _parse_float(ps_txt)
        ps1  = _parse_float(ps1_txt)
        ps2  = _parse_float(ps2_txt)
        ps3  = _parse_float(ps3_txt)
        ps4  = _parse_float(ps4_txt)

        oms_idag = _parse_float(oms_idag_txt)
        oms_next = _parse_float(oms_next_txt)

        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
        }

        # Datumstämpel: sätt bara om manuella fält ändrats
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f, 0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f, 0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f, 0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # Skriv in i df
        if not bef.empty:
            for k, v in ny.items():
                df.loc[df["Ticker"] == bef["Ticker"], k] = v
        else:
            # skapa tom rad med alla kolumner
            tom = {c: (0.0 if c not in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuellt uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.DataFrame(pd.concat([df, pd.DataFrame([tom])], ignore_index=True))

        if datum_sätt:
            df.loc[df["Ticker"] == ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta Yahoo-fält & skriv
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

        # Beräkna & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")

    # Tipslista: äldst uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum", "Bolagsnamn"]).head(10)
    st.dataframe(
        tips[[
            "Ticker", "Bolagsnamn", "Senast manuellt uppdaterad",
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
            "Omsättning idag", "Omsättning nästa år"
        ]],
        use_container_width=True
    )

    return df


# ---- Analysvy ----
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    if etiketter:
        st.session_state.analys_idx = min(st.session_state.analys_idx, len(etiketter) - 1)

    valt_label = st.selectbox(
        "Välj bolag",
        etiketter if etiketter else ["(tom databas)"],
        index=st.session_state.analys_idx if etiketter else 0,
        key="analys_select"
    )
    if etiketter and valt_label in etiketter:
        st.session_state.analys_idx = etiketter.index(valt_label)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            if etiketter:
                st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            if etiketter:
                st.session_state.analys_idx = min(len(etiketter) - 1, st.session_state.analys_idx + 1)

    st.write(f"Post {st.session_state.analys_idx + 1}/{len(etiketter) if etiketter else 1}")

    if etiketter:
        r = vis_df.iloc[st.session_state.analys_idx]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = [
            "Ticker", "Bolagsnamn", "Valuta", "Aktuell kurs",
            "Utestående aktier",
            "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
            "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
            "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
            "CAGR 5 år (%)", "Antal aktier", "Årlig utdelning",
            "Senast manuellt uppdaterad",
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
        port[[
            "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
            "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning (SEK)"
        ]],
        use_container_width=True
    )


# ---- Investeringsförslag ----
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag", "Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential", "Närmast riktkurs"], horizontal=True)

    # Filtera
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0]

    # Kräver riktkurs > 0 & aktuell kurs > 0
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Beräkna potential & diff
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Bläddring i förslag
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base) - 1)

    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index + 1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base) - 1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljdata i SEK (för andelsberäkning)
    port = df[df["Antal aktier"] > 0].copy()
    if not port.empty:
        port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
        port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
        port_värde = float(port["Värde (SEK)"].sum())
    else:
        port_värde = 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if port_värde > 0 and not port.empty:
        r_ = port[port["Ticker"] == rad["Ticker"]]
        if not r_.empty:
            nuv_innehav = float(r_["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'], 2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# =========================
# Del 4/4 – Main & Router
# =========================

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Läs in databasen ---
    df = hamta_data()
    if df.empty:
        # skapa tom mall om arket är tomt (alla kolumner)
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # Säkerställ schema och typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Sidopanel: valutakurser till SEK (persistenta via separat ark) ---
    st.sidebar.header("💱 Valutakurser → SEK")

    # Försök läsa sparade kurser; fallback till standard
    try:
        saved_rates = hamta_valutakurser_sheet()
        if not saved_rates:
            saved_rates = STANDARD_VALUTAKURSER.copy()
    except Exception:
        saved_rates = STANDARD_VALUTAKURSER.copy()

    # Redigerbara inputs
    user_rates = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, key="usd"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, key="nok"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, key="cad"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, key="eur"),
        "SEK": 1.0,  # alltid 1.0
    }

    # Spara valutakurser till ark
    if st.sidebar.button("💾 Spara valutakurser"):
        try:
            spara_valutakurser_sheet(user_rates)
            st.sidebar.success("Valutakurser sparade.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara valutakurser: {e}")

    # Global Yahoo-uppdatering (med 1s fördröjning + kopierbar felrapport)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # --- Meny ---
    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        # beräkna för säkerhets skull inför visning
        df = uppdatera_berakningar(df, user_rates)
        analysvy(df, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
        # lagg_till_eller_uppdatera sparar redan vid submit

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)

if __name__ == "__main__":
    main()
