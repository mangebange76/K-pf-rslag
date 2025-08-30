# ---------- app.py — DEL 1/4 ----------
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

# Ev. lokal tid (Stockholm) om pytz finns
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d %H:%M")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------------- Google Sheets ----------------
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"                  # huvudbladet för bolagsdata
VALUTA_SHEET_NAME = "Valutakurser"    # separat blad för sparade valutakurser

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    # Huvudbladet (bolagsdata)
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hämta_valutablad(skapas_vid_behov: bool = True):
    ss = client.open_by_url(SHEET_URL)
    try:
        ws = ss.worksheet(VALUTA_SHEET_NAME)
    except gspread.WorksheetNotFound:
        if not skapas_vid_behov:
            return None
        # Skapa blad och lägg in rubriker
        ws = ss.add_worksheet(title=VALUTA_SHEET_NAME, rows=10, cols=3)
        ws.update([["Valuta", "SEK_per_1", "Senast sparad"]])
    return ws

def hamta_data() -> pd.DataFrame:
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---------------- Valutakurser ----------------
# Standard (fallback) om inget är sparat i bladet
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.00,
}

def ladda_sparade_valutakurser() -> dict:
    """Läser in sparade kurser från bladet 'Valutakurser'.
       Returnerar alltid en komplett dict med standardvärden som fallback."""
    ws = hämta_valutablad(skapas_vid_behov=True)
    rows = ws.get_all_records() if ws else []
    saved = {}
    for r in rows:
        kod = str(r.get("Valuta", "")).strip().upper()
        try:
            kurs = float(r.get("SEK_per_1", ""))
        except Exception:
            kurs = None
        if kod and kurs and kurs > 0:
            saved[kod] = kurs

    # Slå ihop sparade med standard (standard fyller luckor)
    out = STANDARD_VALUTAKURSER.copy()
    out.update(saved)
    # Säkerställ bara dessa valutor (kan enkelt utökas vid behov)
    clean = {k: float(out.get(k, STANDARD_VALUTAKURSER[k])) for k in ["USD", "NOK", "CAD", "EUR", "SEK"]}
    return clean

def spara_valutakurser(rates: dict):
    """Skriver över bladet 'Valutakurser' med aktuella värden."""
    ws = hämta_valutablad(skapas_vid_behov=True)
    rows = [["Valuta", "SEK_per_1", "Senast sparad"]]
    ts = now_stamp()
    for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        rows.append([k, float(rates.get(k, STANDARD_VALUTAKURSER[k])), ts])
    ws.clear()
    ws.update(rows)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))

# ---------------- Kolumnschema & konvertering ----------------
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
            if any(x in kol.lower() for x in ["kurs", "omsättning", "p/s", "utdelning", "cagr", "antal", "riktkurs"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    # Ta bort ev. gamla riktkurskolumner om de råkar finnas kvar
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

# ---------- app.py — DEL 2/4 ----------
# --- CAGR från Yahoo (Total Revenue, årligen) ---
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """Försök läsa årlig 'Total Revenue' och räkna CAGR mellan första och sista året.
       Returnerar %-tal (t.ex. 12.34) eller 0.0 om det inte går."""
    try:
        # yfinance kan exponera "income_stmt" (ny) eller "financials" (gammal)
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

        # sortera kronologiskt (äldst -> nyast)
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series) - 1)
        if start <= 0:
            return 0.0

        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)  # i procent
    except Exception:
        return 0.0


# --- Hämtning från Yahoo (namn, kurs, valuta, utdelning, CAGR) ---
def hamta_yahoo_fält(ticker: str) -> dict:
    """Returnerar dict med Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%)."""
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

        # Pris (försök via info, annars fallback till senaste Close)
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

        # Årlig utdelning per aktie (kan vara None om ingen utdelning)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        # CAGR 5 år (%) mha finansiella
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out


# --- Beräkningar: P/S-snitt, omsättning år 2/3 (CAGR clamp), riktkurser ---
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Beräknar P/S-snitt, omsättning om 2/3 år (från 'Omsättning nästa år' + CAGR clamp)
       samt riktkurser (utan valutakonvertering – alla riktkurser i aktiens egen valuta)."""
    for i, rad in df.iterrows():
        # 1) P/S-snitt (positiva Q1–Q4)
        ps_vals = [
            rad.get("P/S Q1", 0.0),
            rad.get("P/S Q2", 0.0),
            rad.get("P/S Q3", 0.0),
            rad.get("P/S Q4", 0.0),
        ]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # 2) CAGR clamp: >100% => 50%, <0% => 2%
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # 3) Omsättning om 2 & 3 år från "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # Lämna ev. befintliga ifyllda värden orörda
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # 4) Riktkurser (i aktiens valuta). Kräver Utestående aktier > 0 och P/S-snitt > 0
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))     * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0)) * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])      * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0
    return df


# --- Massuppdatering från Yahoo (sidopanel, 1s delay, kopierbar felrapport) ---
def massuppdatera(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """Sidopanelknapp: hämta Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%)
       för alla rader. Spara, och visa kopierbar lista över misslyckade fält."""
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []   # ["TICKER: fält1, fält2 ..."]
        total = len(df) if len(df) > 0 else 1

        with st.spinner("Hämtar från Yahoo…"):
            for i, row in df.iterrows():
                tkr = str(row.get("Ticker", "")).strip()
                status.write(f"Uppdaterar {i+1}/{total} – {tkr or '(saknar ticker)'}")

                failed_fields = []
                if not tkr:
                    misslyckade.append("(tom ticker): alla fält")
                    bar.progress((i + 1) / total)
                    time.sleep(1.0)
                    continue

                data = hamta_yahoo_fält(tkr)

                # Skriv endast om något faktiskt kom
                if data.get("Bolagsnamn"):
                    df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
                else:
                    failed_fields.append("Bolagsnamn")

                if data.get("Aktuell kurs", 0) > 0:
                    df.at[i, "Aktuell kurs"] = float(data["Aktuell kurs"])
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

                # 1s paus mellan anrop
                time.sleep(1.0)
                bar.progress((i + 1) / total)

        # Beräkna om & spara
        df = uppdatera_berakningar(df, user_rates={})  # user_rates behövs ej här för riktkurser
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")

        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# ---------- app.py — DEL 3/4 ----------
# Vilka manuella fält som triggar datumstämpel "Senast manuellt uppdaterad"
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år"
]

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsval för enklare manuell uppföljning
    sort_val = st.selectbox("Sortera lista för redigering", ["A–Ö (bolagsnamn)", "Äldst manuell uppdatering först"])
    vis_df = df.copy()
    if sort_val.startswith("Äldst"):
        vis_df["_sort_datum"] = vis_df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = vis_df.sort_values(by=["_sort_datum", "Bolagsnamn", "Ticker"])
    else:
        vis_df = vis_df.sort_values(by=["Bolagsnamn", "Ticker"])

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    etiketter = ["(ny post)"] + etiketter

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0  # 0 = (ny post)

    # Rullista + bläddring
    valt_label = st.selectbox("Välj bolag (eller '(ny post)')", etiketter, index=st.session_state.edit_index, key="edit_select")

    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(0, len(etiketter)-1)}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(etiketter)-1, st.session_state.edit_index + 1)

    # Plocka ut befintlig rad om ej ny post
    if valt_label != "(ny post)":
        # mappa etikett -> index i vis_df
        try:
            idx_vis = etiketter.index(valt_label) - 1  # -1 pga "(ny post)" i början
            bef = vis_df.iloc[idx_vis]
            # hitta radens "globala" index i df
            glob_idx = df.index[df["Ticker"] == bef["Ticker"]].tolist()
            glob_idx = glob_idx[0] if glob_idx else None
        except Exception:
            bef, glob_idx = pd.Series({}, dtype=object), None
    else:
        bef, glob_idx = pd.Series({}, dtype=object), None

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=(bef.get("Ticker","") if not bef.empty else "")).upper()
            utest  = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0, step=0.01)
            antal  = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0, step=1.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0, step=0.01)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0, step=0.01)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0, step=0.01)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0, step=0.01)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0, step=0.01)
        with c2:
            oms_idag  = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0, step=0.01)
            oms_next  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0, step=0.01)

            st.markdown("**Fält som hämtas automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas om")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # Ny data att skriva
        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Bestäm om "Senast manuellt uppdaterad" ska sättas
        datum_sätt = False
        if not bef.empty and glob_idx is not None:
            before = {f: float(df.at[glob_idx, f]) if f in df.columns else 0.0 for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            # Ny post: sätt datum om något manuellfält ≠ 0
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # Skriv in i df
        if not bef.empty and glob_idx is not None:
            for k, v in ny.items():
                df.at[glob_idx, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            # flytta markören till posten vi just skapade
            st.session_state.edit_index = 1 + (len(etiketter)-1)  # grovt; uppdateras efter spar

        # Datumstämpel
        if datum_sätt:
            mask = (df["Ticker"] == ticker)
            df.loc[mask, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta Yahoo-fält för denna ticker
        y = hamta_yahoo_fält(ticker)
        mask = (df["Ticker"] == ticker)
        if y.get("Bolagsnamn"): df.loc[mask, "Bolagsnamn"] = y["Bolagsnamn"]
        if y.get("Valuta"):     df.loc[mask, "Valuta"] = y["Valuta"]
        if y.get("Aktuell kurs", 0) > 0: df.loc[mask, "Aktuell kurs"] = float(y["Aktuell kurs"])
        if "Årlig utdelning" in y: df.loc[mask, "Årlig utdelning"] = float(y.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in y:   df.loc[mask, "CAGR 5 år (%)"]   = float(y.get("CAGR 5 år (%)") or 0.0)

        # Räkna om & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("✅ Sparat och uppdaterat från Yahoo.")

    # Tipp-lista (äldst uppdaterade manuellt)
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    tmp = df.copy()
    tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = tmp.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )
    return df


def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    # Välj med index eller lista
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #", min_value=0, max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx, step=1
    )
    st.selectbox("…eller välj i lista", etiketter if etiketter else ["(tom databas)"],
                 index=st.session_state.analys_idx if etiketter else 0,
                 key="analys_select")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(max(0, len(etiketter)-1), st.session_state.analys_idx + 1)

    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")

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
        port[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"]],
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

    # Filter
    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Nyckeltal
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

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag", key="forslag_prev"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag", key="forslag_next"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljdata i SEK (för andelsberäkning)
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

    # Presentationskort (alla riktkurser visas; valda fetmarkeras)
    def mark(radnamn: str) -> str:
        return "**⬅ vald**" if riktkurs_val == radnamn else ""

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {mark("Riktkurs idag")}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {mark("Riktkurs om 1 år")}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {mark("Riktkurs om 2 år")}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {mark("Riktkurs om 3 år")}
- **Uppsida (vald riktkurs):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# ---------- app.py — DEL 4/4 ----------
def _float_from_text(txt: str, fallback: float) -> float:
    """Tillåt fri textinmatning (t.ex. '9,55') och parsa robust till float."""
    if txt is None:
        return fallback
    t = str(txt).strip().replace(" ", "").replace(",", ".")
    try:
        return float(t)
    except Exception:
        return fallback


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Läs huvuddata
    df = hamta_data()
    if df.empty:
        # skapa tom mall om arket är tomt
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # Säkerställ schema + migrera gamla kolumner + konvertera typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 2) Sidopanel: Hämta & redigera valutakurser (sparas i eget blad "Valutor")
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = hamta_valutakurser()  # från bladet "Valutor" om finns, annars standard

    # använde text_input så man kan sudda fritt, och vi parsar manuellt
    usd_txt = st.sidebar.text_input("USD → SEK",  value=str(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])))
    nok_txt = st.sidebar.text_input("NOK → SEK",  value=str(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])))
    cad_txt = st.sidebar.text_input("CAD → SEK",  value=str(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])))
    eur_txt = st.sidebar.text_input("EUR → SEK",  value=str(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])))

    user_rates = {
        "USD": _float_from_text(usd_txt, STANDARD_VALUTAKURSER["USD"]),
        "NOK": _float_from_text(nok_txt, STANDARD_VALUTAKURSER["NOK"]),
        "CAD": _float_from_text(cad_txt, STANDARD_VALUTAKURSER["CAD"]),
        "EUR": _float_from_text(eur_txt, STANDARD_VALUTAKURSER["EUR"]),
        "SEK": 1.0,
    }

    if st.sidebar.button("💾 Spara valutakurser"):
        spara_valutakurser(user_rates)
        st.sidebar.success("Valutakurser sparade i bladet ‘Valutor’. De laddas automatiskt vid nästa start.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Massuppdatering hämtar: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%) "
                       "för alla rader och räknar om riktkurser (1 sek paus/ticker).")
    # 3) Global massuppdateringsknapp i sidopanelen
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # 4) Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        # säkerställ färska beräkningar innan visning
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
