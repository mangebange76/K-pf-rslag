import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

# ── Lokal tid (Stockholm) om pytz finns, annars systemtid ──────────────────────
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ── Google Sheets koppling ─────────────────────────────────────────────────────
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"                 # Huvudark (data)
RATES_SHEET_NAME = "Valutakurser"    # Arket där användarens valutakurser sparas

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return get_spreadsheet().worksheet(SHEET_NAME)

def skapa_koppling_rates(create_if_missing: bool = True):
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except gspread.WorksheetNotFound:
        if not create_if_missing:
            raise
        # Skapa nytt ark för valutakurser
        ws = ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=3)
        ws.update([["Valuta", "Kurs", "Uppdaterad"]])
        return ws

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ── Standard valutakurser till SEK (fallback) ──────────────────────────────────
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

def load_saved_rates() -> dict:
    """Läser sparade valutakurser från arket 'Valutakurser'. Fallback till standard."""
    try:
        ws = skapa_koppling_rates(create_if_missing=True)
        rows = ws.get_all_records()
        out = {}
        for r in rows:
            val = str(r.get("Valuta", "")).upper().strip()
            kurs = r.get("Kurs", "")
            try:
                kurs = float(str(kurs).replace(",", "."))
            except Exception:
                kurs = None
            if val and kurs and kurs > 0:
                out[val] = kurs
        # Fyll upp med ev. saknade standard
        for k, v in STANDARD_VALUTAKURSER.items():
            out.setdefault(k, v)
        return out
    except Exception:
        return STANDARD_VALUTAKURSER.copy()

def save_rates_to_sheet(rates: dict):
    """Sparar användarens valutakurser till arket 'Valutakurser'."""
    try:
        ws = skapa_koppling_rates(create_if_missing=True)
        rows = [["Valuta", "Kurs", "Uppdaterad"]]
        today = now_stamp()
        for k in ["USD", "NOK", "CAD", "EUR", "SEK"]:
            if k in rates:
                rows.append([k, float(rates[k]), today])
        ws.clear()
        ws.update(rows)
    except Exception:
        # Om något går fel här ignorerar vi tyst (appens övriga delar ska fungera)
        pass

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# ── Kolumnschema ───────────────────────────────────────────────────────────────
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
    # Ta bort ev. kolumner som inte längre används (mjuk – endast om de finns)
    drop_cols = [c for c in df.columns if c not in FINAL_COLS]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Migrerar ev. gamla riktkurs-kolumner till nya namn och tar bort de gamla:
      - 'Riktkurs 2026' -> 'Riktkurs om 1 år'
      - 'Riktkurs 2027' -> 'Riktkurs om 2 år'
      - 'Riktkurs 2028' -> 'Riktkurs om 3 år'
      - 'Riktkurs om idag' -> 'Riktkurs idag'
    Behåller befintliga värden om nya kolumnen redan är ifylld (>0).
    """
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

# ── CAGR från yfinance (Total Revenue, årligen) ────────────────────────────────
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

        # sortera kronologiskt
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

# ── Hämtning från Yahoo (namn, kurs, valuta, utdelning, CAGR) ─────────────────
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

# ── Beräkningar (P/S-snitt, omsättning år 2/3 med clamp, riktkurser) ───────────
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
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

        # Riktkurser (kräver Utestående aktier > 0)
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


# ── Massuppdatera från Yahoo (1s delay, kopierbar felrapport) ─────────────────
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
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

            # Skriv endast om något faktiskt kom
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

            # Årlig utdelning kan vara 0 – räkna som OK (vi noterar miss om nyckeln saknas helt)
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

            # 1s paus mellan anropen
            time.sleep(1.0)
            bar.progress((i+1)/max(1, total))

        # Beräkna om efter hämtning
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df


# ── Lägg till / uppdatera bolag ───────────────────────────────────────────────
MANUELL_FALT_FOR_DATUM = ["P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringshjälp för rullistan
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)","Äldst manuell uppdatering först"], key="edit_sort")
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum","Bolagsnamn","Ticker"]).reset_index(drop=True)
    else:
        vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())

    # Hålla koll på “position” vid bläddring
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    chosen = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista,
                          index=min(st.session_state.edit_index, len(val_lista)-1), key="edit_select")

    # Bläddringsknappar
    c_prev, c_pos, c_next = st.columns([1,2,1])
    with c_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with c_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with c_next:
        if st.button("➡️ Nästa", key="edit_next"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if chosen and chosen in namn_map:
        bef = df[df["Ticker"] == namn_map[chosen]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest  = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal  = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps  = st.number_input("P/S",   value=float(bef.get("P/S",0.0))   if not bef.empty else 0.0)
            ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Uppdateras automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, P/S-snitt och riktkurser beräknas om")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo", use_container_width=True)

    if spar and ticker:
        # ny rad-data
        ny = {
            "Ticker": ticker, "Utestående aktier": utest, "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # kolla om manuellfält ändrats (för datum)
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            # ny post → datum sätts om man matat in något i manuellfälten
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # skriv in ny data i df
        if not bef.empty:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
        else:
            # skapa tom rad med alla kolumner
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # datum
        if datum_sätt:
            df.loc[df["Ticker"]==ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # hämta Yahoo-fält för detta ticker
        data = hamta_yahoo_fält(ticker)
        if data.get("Bolagsnamn"): df.loc[df["Ticker"]==ticker, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"): df.loc[df["Ticker"]==ticker, "Valuta"] = data["Valuta"]
        if data.get("Aktuell kurs",0)>0: df.loc[df["Ticker"]==ticker, "Aktuell kurs"] = data["Aktuell kurs"]
        if "Årlig utdelning" in data: df.loc[df["Ticker"]==ticker, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in data: df.loc[df["Ticker"]==ticker, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)

        # beräkna om & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")

    # Visa tipslista (äldst uppdaterad)
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )
    # städa hjälpkolumn om den råkar finnas kvar
    if "_sort_datum" in df.columns:
        df.drop(columns=["_sort_datum"], inplace=True)

    return df

# ── Analysvy ───────────────────────────────────────────────────────────────────
def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    # Sorterad lista för visning + bläddring
    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    # Välj via rullista eller index
    if etiketter:
        st.session_state.analys_idx = st.number_input(
            "Visa bolag #", min_value=0, max_value=len(etiketter)-1,
            value=st.session_state.analys_idx, step=1, key="analys_num"
        )
        st.selectbox("Eller välj i lista", etiketter,
                     index=st.session_state.analys_idx, key="analys_select")
    else:
        st.info("Inga bolag i databasen ännu.")
        return

    # Bläddringsknappar
    c_prev, c_pos, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with c_pos:
        st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter)}")
    with c_next:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx + 1)

    # Visa vald rad (kompakt tabell)
    r = vis_df.iloc[st.session_state.analys_idx]
    st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")

    cols = [
        "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Senast manuellt uppdaterad"
    ]
    cols = [c for c in cols if c in r.index]
    st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)


# ── Portfölj ───────────────────────────────────────────────────────────────────
def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Enda platsen vi räknar om till SEK
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]

    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2)

    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde, 2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd, 2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0, 2)} SEK")

    st.dataframe(
        port[[
            "Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
            "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"
        ]],
        use_container_width=True
    )


# ── Investeringsförslag ───────────────────────────────────────────────────────
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    sort_läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    # Filtrera ev. på portfölj
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()

    # Kräver riktkurs + aktuell kurs
    base = base[(base.get(riktkurs_val, 0) > 0) & (base.get("Aktuell kurs", 0) > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Potential (i aktiens egen valuta – INTE konverterad)
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    # Diff till målkurs (negativ = under målet, positiv = över)
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if sort_läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Bläddring bland förslag
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    c_prev, c_mid, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("⬅️ Föregående förslag", key="sugg_prev"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with c_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with c_next:
        if st.button("➡️ Nästa förslag", key="sugg_next"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljdata i SEK för andelsberäkning
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
        rmatch = port[port["Ticker"] == rad["Ticker"]]
        if not rmatch.empty:
            nuv_innehav = float(rmatch["Värde (SEK)"].sum())

    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Presentationskort – alla fyra riktkurser, vald fetmarkeras
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val=="Riktkurs om 3 år" else ""}

- **Uppsida (vald riktkurs):** {round(rad['Potential (%)'],2)} %
- **Avvikelse från vald riktkurs:** {round(rad['Diff till mål (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# ── Valutakurser: läs/spara i separat blad ─────────────────────────────────────
RATES_SHEET_NAME = "Valutakurser"

def _hamta_rates_sheet():
    """Returnera worksheet-objekt för bladet 'Valutakurser'. Skapa om saknas."""
    ss = client.open_by_url(SHEET_URL)
    try:
        ws = ss.worksheet(RATES_SHEET_NAME)
    except gspread.exceptions.WorksheetNotFound:
        ws = ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=4)
        ws.update([["Valuta", "Kurs", "Sparad", "Av"]])
    return ws

def las_sparade_valutakurser() -> dict:
    """Läs tidigare sparade valutakurser (om finns). Returnerar t.ex. {"USD": 10.12, ...}"""
    try:
        ws = _hamta_rates_sheet()
        rows = ws.get_all_records()
        rates = {}
        for r in rows:
            val = str(r.get("Valuta","")).strip().upper()
            kurs = r.get("Kurs", "")
            try:
                kurs = float(kurs)
            except Exception:
                continue
            if val in ["USD","NOK","CAD","EUR","SEK"]:
                rates[val] = kurs
        return rates
    except Exception:
        return {}

def spara_valutakurser(user_rates: dict):
    """Skriv nuvarande användar-kurser till 'Valutakurser' (skriver över hela tabellen)."""
    try:
        ws = _hamta_rates_sheet()
        header = ["Valuta","Kurs","Sparad","Av"]
        data = [header]
        datum = now_stamp()
        for val in ["USD","NOK","CAD","EUR","SEK"]:
            data.append([val, float(user_rates.get(val, 1.0)), datum, "App"])
        ws.clear()
        ws.update(data)
        st.sidebar.success("Valutakurser sparade.")
    except Exception as e:
        st.sidebar.warning(f"Kunde inte spara valutakurser: {e}")

# ── Huvudprogram ───────────────────────────────────────────────────────────────
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel: läs sparade valutakurser (fallback till standard)
    sparade = las_sparade_valutakurser()
    start_rates = {
        "USD": sparade.get("USD", STANDARD_VALUTAKURSER["USD"]),
        "NOK": sparade.get("NOK", STANDARD_VALUTAKURSER["NOK"]),
        "CAD": sparade.get("CAD", STANDARD_VALUTAKURSER["CAD"]),
        "EUR": sparade.get("EUR", STANDARD_VALUTAKURSER["EUR"]),
        "SEK": sparade.get("SEK", STANDARD_VALUTAKURSER["SEK"]),
    }

    st.sidebar.header("💱 Valutakurser → SEK")
    user_rates = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(start_rates["USD"]), step=0.01, format="%.6f", key="rate_usd"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(start_rates["NOK"]), step=0.01, format="%.6f", key="rate_nok"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(start_rates["CAD"]), step=0.01, format="%.6f", key="rate_cad"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(start_rates["EUR"]), step=0.01, format="%.6f", key="rate_eur"),
        "SEK": 1.0,
    }

    c_spara, c_las = st.sidebar.columns(2)
    with c_spara:
        if st.button("💾 Spara valutakurser", key="save_rates_btn"):
            spara_valutakurser(user_rates)
    with c_las:
        if st.button("⟳ Läs sparade", key="reload_rates_btn"):
            # Ladda om från blad och uppdatera widgets via session_state
            nya = las_sparade_valutakurser()
            for k in ["USD","NOK","CAD","EUR"]:
                if k in nya:
                    st.session_state[f"rate_{k.lower()}"] = float(nya[k])
            st.sidebar.success("Inlästa sparade valutakurser.")

    # Läs huvuddata
    df = hamta_data()
    if df.empty:
        # Skapa tom mall om arket är tomt
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # Säkerställ schema + migrera ev. gamla kolumner + typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Global massuppdatering (knapp ligger i sidopanelen inuti funktionen)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # Meny
    meny = st.sidebar.radio("📌 Välj vy",
                            ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        # Uppdatera beräkningar för visningen (sparar inte automatiskt här)
        df_view = uppdatera_berakningar(df.copy(), user_rates)
        analysvy(df_view, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df.copy(), user_rates)
        if not df2.equals(df):
            # Spara endast om ändrat
            spara_data(df2)
        # Visa uppdaterad analysdel under formuläret (valfritt)
        st.markdown("---")
        st.subheader("Snabböversikt (efter spar)")
        df_view = uppdatera_berakningar(df2.copy(), user_rates)
        st.dataframe(df_view, use_container_width=True)

    elif meny == "Investeringsförslag":
        df_calc = uppdatera_berakningar(df.copy(), user_rates)
        visa_investeringsforslag(df_calc, user_rates)

    elif meny == "Portfölj":
        df_calc = uppdatera_berakningar(df.copy(), user_rates)
        visa_portfolj(df_calc, user_rates)


if __name__ == "__main__":
    main()
