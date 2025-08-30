# =========================
# app.py — Del 1/4
# Importer, tidsstämplar, Sheets-koppling, valuta-blad
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

# Lokal tid (Stockholm) om pytz finns, annars systemtid
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---- Google Sheets ----
SHEET_URL  = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"     # databasfliken
VALUTA_SHEET_NAME = "Valutor"  # flik för sparade valutakurser

scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

def _open_spreadsheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return _open_spreadsheet().worksheet(SHEET_NAME)

def hamta_data() -> pd.DataFrame:
    sheet = skapA_koppling := skapa_koppling()  # alias för enklare felsökning i loggar
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    """Säker skrivning: rensa + skriv OM och endast om df har rader.
       Om df är tomt: skriv bara rubriker (ingen Clear)."""
    ws = skapa_koppling()
    # skydda mot tom skrivning
    if df is None:
        st.warning("Ingen data att spara (df är None). Hoppar över skrivning.")
        return
    if len(df) == 0:
        # skriv bara headerrad – rensa inte hela arket
        ws.update([df.columns.values.tolist()])
        st.info("Arket uppdaterades med kolumnrubriker (ingen data).")
        return
    # normal helskrivning
    ws.clear()
    ws.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Standard-valutor (fallback om inget sparat) ----
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

def ensure_valutor_sheet():
    """Skapar fliken 'Valutor' om den saknas och fyller med standardvärden."""
    ss = _open_spreadsheet()
    try:
        ss.worksheet(VALUTA_SHEET_NAME)
        return
    except gspread.WorksheetNotFound:
        ws = ss.add_worksheet(title=VALUTA_SHEET_NAME, rows=10, cols=2)
        header = [["Valuta", "SEK_per_1"]]
        rows = [[k, str(v)] for k, v in STANDARD_VALUTAKURSER.items()]
        ws.update(header + rows)

def load_user_rates() -> dict:
    """Läs valutakurser från fliken 'Valutor'. Fallback till STANDARD_VALUTAKURSER."""
    ensure_valutor_sheet()
    ws = _open_spreadsheet().worksheet(VALUTA_SHEET_NAME)
    values = ws.get_all_records()
    rates = {r.get("Valuta", "").upper(): float(str(r.get("SEK_per_1", "0")).replace(",", ".") or 0)
             for r in values if r.get("Valuta")}
    # komplettera ev. saknade med standard
    for k, v in STANDARD_VALUTAKURSER.items():
        rates.setdefault(k, v)
    return rates

def save_user_rates(rates: dict):
    """Skriv helt nya kurser till 'Valutor' (ersätter allt innehåll)."""
    ws = _open_spreadsheet().worksheet(VALUTA_SHEET_NAME)
    rows = [["Valuta", "SEK_per_1"]] + [[k, str(v)] for k, v in rates.items()]
    ws.clear()
    ws.update(rows)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return float(user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0)))

# =========================
# app.py — Del 2/4
# Kolumnschema, typer, Yahoo-hämtning, beräkningar, massuppdatera
# =========================

# ---- Kolumnschema (inkl. datum för manuell uppdatering) ----
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
    """Se till att alla nödvändiga kolumner finns."""
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Flytta ev. gamla riktkurs-kolumner till nya namn och ta bort gamla."""
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
    """Tvinga numeriska kolumner till float och textkolumner till str."""
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

# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """Beräkna CAGR baserat på 'Total Revenue' (årliga siffror) från yfinance."""
    try:
        # Nya yfinance: income_stmt, fallback: financials
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

# ---- Hämtning från Yahoo: namn, kurs, valuta, utdelning, CAGR ----
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

        # Kurs
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

        # Utdelning (per aktie, årlig takt om tillgänglig)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        # CAGR 5 år (%)
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out

# ---- Beräkningar (P/S-snitt, omsättning år 2 & 3, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """Beräkna P/S-snitt, justera CAGR enligt regler och uppdatera riktkurser."""
    for i, rad in df.iterrows():
        # P/S-snitt = snitt av positiva Q1-Q4
        ps_vals = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR-justering: >100% → 50%, <0% → 2%, annars som den är
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år baserat på "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * (1.0 + g) ** 2, 2)
        else:
            # lämna befintliga (om de finns), annars 0.0
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurs = (Omsättning * P/S-snitt) / Utestående aktier
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])       * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])       * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

# ---- Massuppdatera från Yahoo (1 sekunds paus, kopierbar fel-lista) ----
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []   # lista med "TICKER: fält1, fält2"
        total = len(df)

        for i, row in df.iterrows():
            tkr = str(row.get("Ticker","")).strip()
            if not tkr:
                continue

            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)
            failed_fields = []

            # skriv endast om värden faktiskt fanns
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

            time.sleep(1.0)  # 1s paus per ticker
            bar.progress((i+1)/max(1,total))

        # Beräkna om och spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)

        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")
    return df

# =========================
# DEL 3 — VYER (ANALYS / PORTFÖLJ / FÖRSLAG)
# =========================

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    # Sortera och bygg etiketter
    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    # Initiera bläddringsindex
    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(st.session_state.analys_idx, max(0, len(etiketter)-1))

    # Välj via rullista
    if len(etiketter) > 0:
        valt_label = st.selectbox("Välj bolag", etiketter, index=st.session_state.analys_idx, key="analys_selectbox")
        if valt_label in etiketter:
            st.session_state.analys_idx = etiketter.index(valt_label)

    # Bläddringsknappar + positionsinfo
    c_prev, c_pos, c_next = st.columns([1,2,1])
    with c_prev:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with c_pos:
        st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")
    with c_next:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(max(0, len(etiketter)-1), st.session_state.analys_idx + 1)

    # Visa vald rad kompakt
    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = [
            "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
            "Utestående aktier",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "CAGR 5 år (%)","Årlig utdelning","Antal aktier",
            "Senast manuellt uppdaterad"
        ]
        existerar = [c for c in cols if c in r.index]
        st.dataframe(pd.DataFrame([r[existerar].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)


def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")

    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # SEK-värden
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())

    port["Andel (%)"] = round(np.where(total_värde > 0, port["Värde (SEK)"] / total_värde * 100.0, 0.0), 2)
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

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0, key="inv_capital")

    riktkurs_val = st.selectbox(
        "Val för uppsida/”närmast”-beräkning",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1,
        key="inv_rk_val"
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True, key="inv_subset")
    sort_mode = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True, key="inv_sort")

    # Filtrera databas
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0]

    # Kräver meningsfull data
    base = base[(base.get(riktkurs_val, 0) > 0) & (base.get("Aktuell kurs", 0) > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Beräkningar för sortering
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if sort_mode == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Bläddringsindex
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    c_prev, c_pos, c_next = st.columns([1,2,1])
    with c_prev:
        if st.button("⬅️ Föregående förslag", key="inv_prev"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with c_pos:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with c_next:
        if st.button("➡️ Nästa förslag", key="inv_next"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljdata i SEK för andelar
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    # Antal att köpa för givet SEK-kapital (konvertera den valda aktiens valuta till SEK)
    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    # Nuvarande innehav i SEK för just denna ticker
    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())

    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Presentationskort
    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    # Markera vald riktkurs rad i fet
    def mark(line, chosen):
        return f"**{line}**" if chosen else line

    rader = [
        ("Riktkurs idag",        rad["Riktkurs idag"],        riktkurs_val == "Riktkurs idag"),
        ("Riktkurs om 1 år",     rad["Riktkurs om 1 år"],     riktkurs_val == "Riktkurs om 1 år"),
        ("Riktkurs om 2 år",     rad["Riktkurs om 2 år"],     riktkurs_val == "Riktkurs om 2 år"),
        ("Riktkurs om 3 år",     rad["Riktkurs om 3 år"],     riktkurs_val == "Riktkurs om 3 år"),
    ]
    lines = "\n".join([
        f"- {mark(f'**Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}', False)}",
        *[f"- {mark(f'{rub}: {round(v,2)} {rad['Valuta']}', ch)}" for rub, v, ch in rader],
        f"- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %",
        f"- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st",
        f"- **Nuvarande andel:** {nuv_andel} %",
        f"- **Andel efter köp:** {ny_andel} %",
    ])
    st.markdown(lines)

# =========================
# DEL 4 — KONFIG & MAIN
# =========================

# -- Valutakurser i separat flik ("Konfig") --
def hamta_valutakurser_fran_konfig() -> dict:
    """Läs sparade valutakurser från fliken 'Konfig'.
       Förväntat format:
         Rad 1:  Valuta | SEK
         Rader:  USD    | 10.12  (float)
       Returnerar dict och faller tillbaka till STANDARD_VALUTAKURSER om fliken saknas/tom.
    """
    try:
        wb = client.open_by_url(SHEET_URL)
        try:
            ws = wb.worksheet("Konfig")
        except gspread.WorksheetNotFound:
            return STANDARD_VALUTAKURSER.copy()

        rows = ws.get_all_records()
        if not rows:
            return STANDARD_VALUTAKURSER.copy()

        out = {}
        for r in rows:
            k = str(r.get("Valuta","")).strip().upper()
            v = r.get("SEK", "")
            try:
                v = float(v)
            except Exception:
                continue
            if k:
                out[k] = v
        # se till att vi åtminstone täcker dessa:
        for k, v in STANDARD_VALUTAKURSER.items():
            out.setdefault(k, v)
        return out
    except Exception:
        return STANDARD_VALUTAKURSER.copy()


def spara_valutakurser_till_konfig(rates: dict) -> None:
    """Skriv valutakurserna till fliken 'Konfig' utan att röra övriga ark."""
    try:
        wb = client.open_by_url(SHEET_URL)
        try:
            ws = wb.worksheet("Konfig")
        except gspread.WorksheetNotFound:
            ws = wb.add_worksheet(title="Konfig", rows=10, cols=2)

        # Gör en enkel tabell: header + rader (sortera för stabilitet)
        header = [["Valuta", "SEK"]]
        body = [[k, str(rates[k])] for k in sorted(rates.keys())]
        ws.clear()
        ws.update(header + body)
    except Exception:
        # Vid fel: inte kritiskt för appens drift, ignorera
        pass


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Sidopanel: valutakurser (persistenta) ---
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = hamta_valutakurser_fran_konfig()
    # Visa inputs (med sparade värden som default)
    user_rates = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(saved_rates.get("USD", STANDARD_VALUTAKURSER["USD"])), step=0.01, key="rate_usd"),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(saved_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"])), step=0.01, key="rate_nok"),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(saved_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"])), step=0.01, key="rate_cad"),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(saved_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"])), step=0.01, key="rate_eur"),
    }
    # Lägg även med SEK själv (används internt vid mapping)
    user_rates["SEK"] = 1.0

    # Spara valutakurser om de ändrats
    changed = any(abs(float(saved_rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0))) - float(user_rates.get(k, 1.0))) > 1e-9
                  for k in ["USD","NOK","CAD","EUR"])
    if changed:
        if st.sidebar.button("💾 Spara valutakurser"):
            spara_valutakurser_till_konfig(user_rates)
            st.sidebar.success("Valutakurser sparade i fliken 'Konfig'.")

    # --- Läs huvuddata ---
    df = hamta_data()
    if df.empty:
        # skapa tom mall om arket är helt tomt
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        try:
            spara_data(df)
        except Exception:
            pass  # om vi saknar rättigheter just nu, fortsätt ändå i minnet

    # Schema → migrering → typer
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # Global Yahoo-uppdatering i sidopanelen (finns alltid)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # --- Meny ---
    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"]
    )

    # --- Visa vald vy ---
    if meny == "Analys":
        # beräkna innan visning (utan att tvinga spara)
        df_view = uppdatera_berakningar(df.copy(), user_rates)
        analysvy(df_view, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df.copy(), user_rates)
        # Om något ändrats där inne har vi redan sparat. Men för säkerhets skull:
        if not df2.equals(df):
            try:
                spara_data(df2)
            except Exception:
                st.warning("Kunde inte spara till Google Sheets just nu.")
            df = df2

    elif meny == "Investeringsförslag":
        df_view = uppdatera_berakningar(df.copy(), user_rates)
        visa_investeringsforslag(df_view, user_rates)

    elif meny == "Portfölj":
        df_view = uppdatera_berakningar(df.copy(), user_rates)
        visa_portfolj(df_view, user_rates)


if __name__ == "__main__":
    main()
