# ---------------------------------------
# Aktieanalys & investeringsförslag – FULLSTÄNDIG VERSION
# Del 1: Importer, autentisering, databaskoppling, hjälpmetoder
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="📊 Aktieanalys och investeringsförslag", layout="wide")

# ---------------------------------------
# KONFIGURATION – GOOGLE SHEETS
# ---------------------------------------

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ---------------------------------------
# DATAHANTERING – LADDA OCH SPARA
# ---------------------------------------

def skapa_koppling(sheetname):
    return client.open_by_url(SHEET_URL).worksheet(sheetname)

def hamta_data():
    df = pd.DataFrame(skapa_koppling(SHEET_NAME).get_all_records())
    return konvertera_typer(säkerställ_kolumner(df))

def spara_data(df):
    sheet = skapa_koppling(SHEET_NAME)
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    numeriska = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier", "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år"
    ]
    for kol in numeriska:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år", "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() else ""
    return df

# ---------------------------------------
# INSTÄLLNINGAR – Läs & spara från bladet "Inställningar"
# ---------------------------------------

def skapa_inställningsblad():
    try:
        sheet = skapa_koppling(SETTINGS_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        ss = client.open_by_url(SHEET_URL)
        sheet = ss.add_worksheet(title=SETTINGS_SHEET, rows=10, cols=4)
        sheet.append_row(["Namn", "Värde", "Senast ändrad"])
        sheet.append_row(["Valutakurs", "10.0", datetime.today().strftime("%Y-%m-%d")])
        sheet.append_row(["Max portföljandel (%)", "20", datetime.today().strftime("%Y-%m-%d")])
        sheet.append_row(["Max högriskandel (%)", "2", datetime.today().strftime("%Y-%m-%d")])
    return skapa_koppling(SETTINGS_SHEET)

def load_settings():
    sheet = skapa_inställningsblad()
    data = sheet.get_all_records()
    return {rad["Namn"]: float(rad["Värde"]) for rad in data}

def save_settings(settings_dict):
    sheet = skapa_inställningsblad()
    data = [["Namn", "Värde", "Senast ändrad"]]
    for key, value in settings_dict.items():
        data.append([key, str(value), datetime.today().strftime("%Y-%m-%d")])
    sheet.clear()
    sheet.update(data)

def visa_sidopanel():
    st.sidebar.title("⚙️ Inställningar")
    inst = load_settings()
    valutakurs = st.sidebar.number_input("Valutakurs (USD → SEK)", value=inst.get("Valutakurs", 10.0), step=0.1)
    max_andel = st.sidebar.number_input("Max portföljandel (%)", value=inst.get("Max portföljandel (%)", 20.0), step=1.0)
    max_risk = st.sidebar.number_input("Max högriskandel (%)", value=inst.get("Max högriskandel (%)", 2.0), step=0.5)

    if st.sidebar.button("💾 Spara inställningar"):
        save_settings({
            "Valutakurs": valutakurs,
            "Max portföljandel (%)": max_andel,
            "Max högriskandel (%)": max_risk
        })
        st.sidebar.success("Inställningar sparade!")

    return valutakurs, max_andel, max_risk

# ---------------------------------------
# BERÄKNINGAR & DATAINMATNING
# ---------------------------------------

def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "Antal aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S-snitt",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år"
    ]
    for kol in kolumner:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() else ""
    return df

def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs nu"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = df["Ticker"].tolist()
    valt = st.selectbox("Välj existerande bolag att uppdatera (eller lämna tom för nytt)", [""] + tickers)

    if valt:
        befintlig = df[df["Ticker"] == valt].iloc[0]
    else:
        befintlig = {}

    with st.form("form"):
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "")).upper()
        namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", ""))
        kurs = st.number_input("Aktuell kurs (USD)", value=float(befintlig.get("Aktuell kurs", 0.0)))
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)))
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)))

        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)))
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)))
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)))
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)))
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)))

        oms_idag = st.number_input("Omsättning idag (miljoner USD)", value=float(befintlig.get("Omsättning idag", 0.0)))
        oms_1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)))
        oms_2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)))

        sparaknapp = st.form_submit_button("💾 Spara")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs,
            "Utestående aktier": aktier, "Antal aktier": antal_aktier,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1, "Omsättning om 2 år": oms_2
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

# ---------------------------------------
# INVESTERINGSFÖRSLAG & PORTFÖLJ
# ---------------------------------------

def visa_investeringsforslag(df, valutakurs, max_andel, max_hogrisk):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=500.0)
    st.markdown("---")

    df = df[df["Riktkurs om 1 år"] > df["Aktuell kurs"]].copy()
    df["Potential"] = df["Riktkurs om 1 år"] - df["Aktuell kurs"]
    df = df.sort_values(by="Potential", ascending=False).reset_index(drop=True)

    if valutakurs == 0:
        st.warning("Valutakursen får inte vara 0.")
        return

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    totalt_portfoljvarde = df["Värde (SEK)"].sum()
    df["Portföljandel (%)"] = round(df["Värde (SEK)"] / totalt_portfoljvarde * 100, 2)
    df["Högrisk"] = (df["Omsättning idag"] < 1000) & (df["Portföljandel (%)"] >= max_hogrisk)

    if "visat_index" not in st.session_state:
        st.session_state.visat_index = 0

    # Visning av investeringsförslag ett i taget
    for i in range(st.session_state.visat_index, len(df)):
        rad = df.iloc[i]
        antal = int((kapital_sek / valutakurs) // rad["Aktuell kurs"])
        if rad["Aktuell kurs"] <= 0:
            continue

        st.markdown(f"### Förslag {i+1}: {rad['Ticker']} – {rad['Bolagsnamn']}")
        st.write(f"**Aktuell kurs**: {rad['Aktuell kurs']} USD")
        st.write(f"**Riktkurs om 1 år**: {rad['Riktkurs om 1 år']} USD")
        st.write(f"**Uppsidepotential**: {round(rad['Potential'], 2)} USD")
        st.write(f"**Rekommenderat antal aktier**: {antal}")
        st.write(f"**Kostnad i SEK**: {round(antal * rad['Aktuell kurs'] * valutakurs, 2)}")

        if rad["Portföljandel (%)"] > max_andel:
            st.warning(f"⚠️ Portföljandelen är redan {rad['Portföljandel (%)']}% vilket överstiger maxgränsen på {max_andel}%.")

        if rad["Högrisk"]:
            st.error("⚠️ Högrisk: Bolaget har omsättning < 1 miljard USD och överstiger max tillåten andel för högriskbolag.")

        st.session_state.visat_index += 1
        if st.button("Nästa förslag"):
            st.experimental_rerun()
        break
    else:
        st.info("Inga fler förslag att visa.")

    st.markdown("---")
    visa_ombalanseringstabeller(df, max_andel, max_hogrisk)

def visa_ombalanseringstabeller(df, max_andel, max_hogrisk):
    st.subheader("⚖️ Ombalanseringsförslag")

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * st.session_state.valutakurs
    totalt = df["Värde (SEK)"].sum()
    df["Portföljandel (%)"] = round(df["Värde (SEK)"] / totalt * 100, 2)

    minskas = df[df["Portföljandel (%)"] > max_andel]
    ökas = df[(df["Potential"] > 0) & (df["Portföljandel (%)"] < max_andel)]
    högrisk = df[(df["Omsättning idag"] < 1000) & (df["Portföljandel (%)"] > max_hogrisk)]

    if not minskas.empty:
        st.markdown("#### 📉 Bolag att minska i")
        st.dataframe(minskas[["Ticker", "Bolagsnamn", "Portföljandel (%)"]], use_container_width=True)

    if not ökas.empty:
        st.markdown("#### 📈 Bolag att öka i")
        st.dataframe(ökas[["Ticker", "Bolagsnamn", "Potential", "Portföljandel (%)"]], use_container_width=True)

    if not högrisk.empty:
        st.markdown("#### ⚠️ Högriskvarningar")
        st.dataframe(högrisk[["Ticker", "Bolagsnamn", "Omsättning idag", "Portföljandel (%)"]], use_container_width=True)

def visa_portfolj(df, valutakurs):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return

    df["Värde (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    st.dataframe(df[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]], use_container_width=True)

# ---------------------------------------
# INSTÄLLNINGAR (inställningsblad + sidopanel)
# ---------------------------------------

INSTÄLLNINGAR_SHEET = "Inställningar"

def skapa_inställningsblad():
    try:
        blad = client.open_by_url(SHEET_URL)
        if INSTÄLLNINGAR_SHEET not in [ws.title for ws in blad.worksheets()]:
            blad.add_worksheet(title=INSTÄLLNINGAR_SHEET, rows=10, cols=2)
            inst_sheet = blad.worksheet(INSTÄLLNINGAR_SHEET)
            inst_sheet.update("A1:B4", [
                ["Inställning", "Värde"],
                ["Valutakurs", "10.0"],
                ["Max portföljandel (%)", "20.0"],
                ["Max högriskandel (%)", "2.0"],
            ])
            inst_sheet.update("C1", "Senast ändrad")
    except Exception as e:
        st.error(f"Fel vid skapande av inställningsblad: {e}")

def hamta_inställningar():
    try:
        blad = client.open_by_url(SHEET_URL).worksheet(INSTÄLLNINGAR_SHEET)
        data = blad.get_all_records()
        inst = {rad["Inställning"]: float(rad["Värde"]) for rad in data}
        return inst
    except Exception as e:
        st.warning("Inställningar saknas eller kunde inte hämtas.")
        return {"Valutakurs": 10.0, "Max portföljandel (%)": 20.0, "Max högriskandel (%)": 2.0}

def spara_inställningar(valutakurs, max_andel, max_hogrisk):
    try:
        blad = client.open_by_url(SHEET_URL).worksheet(INSTÄLLNINGAR_SHEET)
        blad.update("A2:B4", [
            ["Valutakurs", valutakurs],
            ["Max portföljandel (%)", max_andel],
            ["Max högriskandel (%)", max_hogrisk]
        ])
        blad.update("C2", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        st.error(f"Fel vid uppdatering av inställningar: {e}")

def visa_inställningar_sido():
    st.sidebar.markdown("### ⚙️ Inställningar")
    inst = hamta_inställningar()

    valutakurs = st.sidebar.number_input("Valutakurs (USD → SEK)", value=inst["Valutakurs"], step=0.1)
    max_andel = st.sidebar.number_input("Max portföljandel (%)", value=inst["Max portföljandel (%)"], step=1.0)
    max_hogrisk = st.sidebar.number_input("Max högriskandel (%)", value=inst["Max högriskandel (%)"], step=0.5)

    if st.sidebar.button("💾 Spara inställningar"):
        spara_inställningar(valutakurs, max_andel, max_hogrisk)
        st.sidebar.success("Inställningar sparade!")

    return valutakurs, max_andel, max_hogrisk

# ---------------------------------------
# MAIN-FUNKTION
# ---------------------------------------

def main():
    st.set_page_config(page_title="📊 Aktieanalys och investeringsförslag", layout="wide")
    skapa_inställningsblad()

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = uppdatera_berakningar(df)

    valutakurs, max_andel, max_hogrisk = visa_inställningar_sido()

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        st.subheader("📈 Aktieanalys")
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        df = uppdatera_berakningar(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, valutakurs, max_andel, max_hogrisk)

    elif meny == "Portfölj":
        visa_portfolj(df, valutakurs)

if __name__ == "__main__":
    main()
