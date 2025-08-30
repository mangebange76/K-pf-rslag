# ============================
# app.py — DEL 1/4 (bas + valuta)
# ============================

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

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Kolumnschema (slutlig lista vi jobbar med) ----
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
    # Mappa ev. gamla till nya och ta bort de gamla
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

# ============================
#  Persistent VALUTAKURSER i separat blad
# ============================
VALUTA_SHEET_NAME = "Valutakurser"

STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

def _get_or_create_ws(title: str):
    ss = client.open_by_url(SHEET_URL)
    try:
        return ss.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = ss.add_worksheet(title=title, rows=10, cols=3)
        ws.update([["Valuta", "SEK_per_1_unit", "Updated"]])
        return ws

def load_valutakurser() -> dict:
    try:
        ws = _get_or_create_ws(VALUTA_SHEET_NAME)
        rows = ws.get_all_records()
        rates = {}
        for r in rows:
            cur = str(r.get("Valuta", "")).upper().strip()
            raw = str(r.get("SEK_per_1_unit", "")).strip()
            if not cur:
                continue
            try:
                val = float(raw.replace(",", "."))
            except Exception:
                val = None
            if val is not None and val > 0:
                rates[cur] = val
        # fyll på saknade
        for c, v in STANDARD_VALUTAKURSER.items():
            rates.setdefault(c, v)
        return rates
    except Exception:
        return dict(STANDARD_VALUTAKURSER)

def save_valutakurser(rates: dict) -> None:
    ws = _get_or_create_ws(VALUTA_SHEET_NAME)
    now = now_stamp()
    out = [["Valuta", "SEK_per_1_unit", "Updated"]]
    for cur in ["USD", "NOK", "CAD", "EUR", "SEK"]:
        out.append([cur, float(rates.get(cur, STANDARD_VALUTAKURSER.get(cur, 1.0))), now])
    ws.clear()
    ws.update(out)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    v = str(valuta).upper()
    return float(user_rates.get(v, STANDARD_VALUTAKURSER.get(v, 1.0)))

def sidebar_valutakurser() -> dict:
    st.sidebar.subheader("💱 Valutakurser → SEK")

    current = load_valutakurser()

    usd = st.sidebar.number_input("USD → SEK", value=float(current["USD"]), step=0.01, key="rate_usd")
    nok = st.sidebar.number_input("NOK → SEK", value=float(current["NOK"]), step=0.01, key="rate_nok")
    cad = st.sidebar.number_input("CAD → SEK", value=float(current["CAD"]), step=0.01, key="rate_cad")
    eur = st.sidebar.number_input("EUR → SEK", value=float(current["EUR"]), step=0.01, key="rate_eur")
    sek = 1.0  # alltid 1

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Spara kurser", key="rates_save"):
        to_save = {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": float(sek)}
        try:
            save_valutakurser(to_save)
            st.sidebar.success("Valutakurser sparade i Google Sheet.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte spara kurser: {e}")

    if c2.button("Återställ standard", key="rates_reset"):
        try:
            save_valutakurser(STANDARD_VALUTAKURSER.copy())
            st.sidebar.info("Återställde till standardvärden.")
        except Exception as e:
            st.sidebar.error(f"Kunde inte återställa: {e}")

    return {"USD": float(usd), "NOK": float(nok), "CAD": float(cad), "EUR": float(eur), "SEK": float(sek)}

# ============================
# app.py — DEL 2/4 (Yahoo, CAGR, beräkningar, massuppdatering)
# ============================

# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """Beräknar CAGR på total omsättning utifrån Yahoo 'Total Revenue' (årsdata)."""
    try:
        # Prova nya egenskapen
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            # Fallback: äldre 'financials'
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0

        if series.empty or len(series) < 2:
            return 0.0

        # indexen är datumperioder – sortera Kronologiskt (äldst → nyast)
        series = series.sort_index()
        start = float(series.iloc[0])
        slut  = float(series.iloc[-1])
        år = max(1, len(series) - 1)
        if start <= 0:
            return 0.0

        cagr = (slut / start) ** (1.0 / år) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


# ---- Hämtning från Yahoo (namn, kurs, valuta, utdelning, CAGR) ----
def hamta_yahoo_fält(ticker: str) -> dict:
    """Returnerar dict med Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%)."""
    ut = {"Bolagsnamn": "", "Aktuell kurs": 0.0, "Valuta": "USD", "Årlig utdelning": 0.0, "CAGR 5 år (%)": 0.0}
    try:
        t = yf.Ticker(ticker)

        # info kan kasta/vara tomt i vissa versioner → säkra hantering
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice")
        if pris is None:
            # Fallback: senaste stängning
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            ut["Aktuell kurs"] = float(pris)

        valuta = info.get("currency")
        if valuta:
            ut["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            ut["Bolagsnamn"] = str(namn)

        # Utdelningsrate/aktie (kan vara None → lämna 0.0)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            ut["Årlig utdelning"] = float(div_rate)

        # CAGR (5 år) beräknad från finansiella rapporter
        ut["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return ut


# ---- Beräkningar (P/S-snitt, omsättning år 2 & 3, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    - P/S-snitt = snitt av positiva P/S Q1–Q4
    - CAGR clamp: >100% → 50%, <0% → 2% (inflationsgolv)
    - Omsättning om 2 & 3 år = 'Omsättning nästa år' växt med justerad CAGR
    - Riktkurser = (Omsättning * P/S-snitt) / Utestående aktier
    OBS: Antas att 'Omsättning *' och 'Utestående aktier' båda är i "miljoner"
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if pd.notnull(x) and float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR-justering
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning år 2 & 3 utifrån "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll befintliga värden om de finns
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurs-beräkningar
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


# ---- Massuppdatera från Yahoo (1s delay + kopierbar felrapport) ----
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    """
    Sidopanelsknapp "Uppdatera alla från Yahoo".
    Uppdaterar: Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%).
    Gör sedan om-beräkningar och sparar tillbaka arket.
    """
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []   # strängar "TICKER: fält1, fält2, ..."
        total = len(df)

        for i, row in df.iterrows():
            tkr = str(row.get("Ticker", "")).strip()
            if not tkr:
                continue

            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)
            failed_fields = []

            # skriv endast om vi fick något
            if data.get("Bolagsnamn"):
                df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
            else:
                failed_fields.append("Bolagsnamn")

            if float(data.get("Aktuell kurs", 0) or 0) > 0:
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

            time.sleep(1.0)  # API-hygien
            bar.progress((i+1)/max(1, total))

        # Beräkna om & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# ============================
# app.py — DEL 3/4 (VYER)
# ============================

# Vilka fält räknas som "manuellt uppdaterade" (för datumstämpling)
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
        ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"]
    )
    if sort_val.startswith("Äldst"):
        tmp = df.copy()
        tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = tmp.sort_values(by=["_sort_datum","Bolagsnamn","Ticker"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"])

    # Etiketter för rullistan
    val_list = [""] + [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0

    valt_label = st.selectbox(
        "Välj bolag (lämna tomt för nytt)",
        val_list,
        index=min(st.session_state.edit_index, len(val_list)-1)
    )

    # Bläddringsknappar
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(0, len(val_list)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_list)-1, st.session_state.edit_index + 1)

    # Hämta vald rad (om någon)
    if valt_label and "(" in valt_label and ")" in valt_label:
        tkr = valt_label.split("(")[-1].split(")")[0].strip()
        bef_mask = df["Ticker"].astype(str).str.upper().eq(tkr.upper())
        bef = df[bef_mask].iloc[0] if bef_mask.any() else pd.Series({}, dtype=object)
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input(
                "Ticker (Yahoo-format)",
                value=bef.get("Ticker","") if not bef.empty else ""
            ).upper()
            utest = st.number_input(
                "Utestående aktier (miljoner)",
                value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0
            )
            antal = st.number_input(
                "Antal aktier du äger",
                value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0
            )

            ps  = st.number_input("P/S",    value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)

        with c2:
            oms_idag = st.number_input(
                "Omsättning idag (miljoner)",
                value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0
            )
            oms_next = st.number_input(
                "Omsättning nästa år (miljoner)",
                value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0
            )

            st.markdown("**Uppdateras automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # Förbered ny data (manuell)
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Har manuellfält ändrats? → sätt datum
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # Skriv in i df (uppdatera eller skapa)
        if not bef.empty:
            idx = df.index[df["Ticker"].astype(str).str.upper() == ticker.upper()]
            if len(idx) > 0:
                for k,v in ny.items():
                    df.at[idx[0], k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"] else "")
                   for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Datumstämpla (endast manuella ändringar)
        if datum_sätt:
            df.loc[df["Ticker"].astype(str).str.upper() == ticker.upper(), "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta från Yahoo för just detta bolag
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"):
            df.loc[df["Ticker"].astype(str).str.upper() == ticker.upper(), "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):
            df.loc[df["Ticker"].astype(str).str.upper() == ticker.upper(), "Valuta"] = data["Valuta"]
        if float(data.get("Aktuell kurs",0) or 0) > 0:
            df.loc[df["Ticker"].astype(str).str.upper() == ticker.upper(), "Aktuell kurs"] = float(data["Aktuell kurs"])
        if "Årlig utdelning" in data:
            df.loc[df["Ticker"].astype(str).str.upper() == ticker.upper(), "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data:
            df.loc[df["Ticker"].astype(str).str.upper() == ticker.upper(), "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

        # Beräkna om + spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")

    # Tips: topp 10 äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    tmp = df.copy()
    tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = tmp.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
              "Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )

    return df


# ---- Analys ----
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Databasen är tom.")
        return

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(st.session_state.analys_idx, max(0, len(etiketter)-1))

    # Välj rad via index eller lista
    st.session_state.analys_idx = st.number_input(
        "Visa bolag #",
        min_value=0,
        max_value=max(0, len(etiketter)-1),
        value=st.session_state.analys_idx,
        step=1
    )
    st.selectbox("Eller välj i lista", etiketter, index=st.session_state.analys_idx, key="analys_select")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx + 1)

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

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    # Filtrera grundmängd
    if subset == "Endast portfölj":
        base = df[df["Antal aktier"] > 0].copy()
    else:
        base = df.copy()

    # Kräver riktkurs och aktuell kurs
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Potential och differens mot mål
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
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portfölj-helhet i SEK för andelsberäkning
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    # Köpberäkning: kapital (SEK) → antal aktier (konvertera aktiekursen till SEK)
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"].astype(str) == str(rad["Ticker"])]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Presentationskort (valutorna i aktiens egen valuta — ej omräknat utom SEK-raderna)
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 3 år" else ""}

- **Uppsida (vald riktkurs):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# ============================
# app.py — DEL 4/4 (MAIN + Valutablad)
# ============================

VALUTA_SHEET = "Valutor"

def _ensure_valutablad():
    """Skapa bladet 'Valutor' om det saknas, med standardvärden."""
    ss = client.open_by_url(SHEET_URL)
    try:
        ws = ss.worksheet(VALUTA_SHEET)
        return ws
    except Exception:
        ws = ss.add_worksheet(title=VALUTA_SHEET, rows="10", cols="3")
        rows = [["Valuta", "SEK"]]
        for k, v in STANDARD_VALUTAKURSER.items():
            rows.append([k, str(v)])
        ws.update(rows)
        return ws

def _load_rates_from_sheet() -> dict:
    """Läs valutakurser från bladet 'Valutor'."""
    ws = _ensure_valutablad()
    records = ws.get_all_records()
    rates = {}
    for r in records:
        code = str(r.get("Valuta","")).strip().upper()
        val  = str(r.get("SEK","")).strip().replace(",", ".")
        try:
            fval = float(val)
        except Exception:
            fval = None
        if code and fval is not None:
            rates[code] = fval
    # komplettera ev. saknade med standard
    for k, v in STANDARD_VALUTAKURSER.items():
        rates.setdefault(k, v)
    return rates

def _save_rates_to_sheet(rates: dict):
    """Skriv valutakurser till bladet 'Valutor' (ersätter allt)."""
    ws = _ensure_valutablad()
    rows = [["Valuta", "SEK"]]
    # skriv endast kända koder i en bestämd ordning
    for code in ["USD","NOK","CAD","EUR","SEK"]:
        if code in rates:
            rows.append([code, str(rates[code])])
    ws.clear()
    ws.update(rows)

def _sidebar_valutor() -> dict:
    """Sidopanel för valutakurser (fri text, kan suddas) + spara-knapp."""
    st.sidebar.header("💱 Valutakurser → SEK (sparas i bladet 'Valutor')")
    persisted = _load_rates_from_sheet()

    # Visa som textfält för enklare redigering
    def _txt(kod, key):
        default = str(persisted.get(kod, STANDARD_VALUTAKURSER.get(kod, 1.0)))
        return st.sidebar.text_input(f"{kod} → SEK", value=default, key=key)

    usd_txt = _txt("USD", "v_usd")
    nok_txt = _txt("NOK", "v_nok")
    cad_txt = _txt("CAD", "v_cad")
    eur_txt = _txt("EUR", "v_eur")
    sek_txt = _txt("SEK", "v_sek")

    # Parse till floats (komma → punkt)
    def _p(s):
        try:
            return float((s or "").replace(",", "."))
        except Exception:
            return None

    edited = {
        "USD": _p(usd_txt),
        "NOK": _p(nok_txt),
        "CAD": _p(cad_txt),
        "EUR": _p(eur_txt),
        "SEK": _p(sek_txt) if _p(sek_txt) not in (None,0) else 1.0,  # SEK ska vara 1.0
    }
    # Fyll tomma/ogiltiga med tidigare/pålitliga värden
    user_rates = {}
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        if edited[k] is not None:
            user_rates[k] = edited[k]
        else:
            user_rates[k] = persisted.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))

    if st.sidebar.button("💾 Spara valutakurser"):
        _save_rates_to_sheet(user_rates)
        st.sidebar.success("Valutakurser sparade i bladet 'Valutor'.")

    return user_rates

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Läs data
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # 2) Säkerställ schema, migrera ev. gamla kolumner, konvertera typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 3) Sidopanel: valutakurser från bladet "Valutor" (persistens)
    user_rates = _sidebar_valutor()

    # 4) Global massuppdatering i sidopanelen (1 s delay / ticker)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # 5) Meny
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"]
    )

    # 6) Visa vald vy
    if meny == "Analys":
        # Visa (utan extra spar här – vyerna sparar själva vid behov)
        analysvy(df, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # Om användaren sparade något i formuläret sköts spara i funktionen.
        # Visa uppdaterad vy (hämtas om från arket för att vara i synk).
        df = hamta_data()
        df = säkerställ_kolumner(df)
        df = migrera_gamla_riktkurskolumner(df)
        df = konvertera_typer(df)
        st.success("Klar.")

    elif meny == "Investeringsförslag":
        # Se till att beräkningar är färska
        df_calc = uppdatera_berakningar(df.copy(), user_rates)
        visa_investeringsforslag(df_calc, user_rates)

    elif meny == "Portfölj":
        df_calc = uppdatera_berakningar(df.copy(), user_rates)
        visa_portfolj(df_calc, user_rates)


if __name__ == "__main__":
    main()
