# ---------------------------------------
# Aktieanalys & investeringsförslag – Manuell valutakurs och aktiekurs
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------------------------------------
# KONFIGURATION OCH KOPPLING TILL GOOGLE SHEETS
# ---------------------------------------

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
INST_NAME = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling(sheet_name):
    return client.open_by_url(SHEET_URL).worksheet(sheet_name)

def hamta_data():
    sheet = skapa_koppling(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling(SHEET_NAME)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def skapa_inställningsblad():
    try:
        sheet = skapa_koppling(INST_NAME)
    except:
        sh = client.open_by_url(SHEET_URL)
        sheet = sh.add_worksheet(title=INST_NAME, rows=10, cols=2)
        sheet.update("A1:B4", [
            ["Inställning", "Värde"],
            ["Valutakurs", "10.00"],
            ["Max portföljandel (%)", "20.00"],
            ["Max högriskandel (%)", "2.00"]
        ])
        sheet.update("C1", "Senast ändrad")
        sheet.update("C2", datetime.today().strftime("%Y-%m-%d"))

def load_settings():
    skapa_inställningsblad()
    ws = skapa_koppling(INST_NAME)
    values = ws.get_all_values()
    df = pd.DataFrame(values[1:], columns=values[0])
    df["Värde"] = df["Värde"].str.replace(",", ".").astype(float)
    inst = dict(zip(df["Inställning"], df["Värde"]))
    return inst

def spara_inställningar(inst):
    ws = skapa_koppling(INST_NAME)
    for key, val in inst.items():
        val_str = str(round(val, 2)).replace(".", ",")
        cell = ws.find(key)
        if cell:
            ws.update_cell(cell.row, cell.col + 1, val_str)
    ws.update("C2", datetime.today().strftime("%Y-%m-%d"))

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

def visa_investeringsforslag(df, valutakurs, inställningar):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=500.0)
    kapital_usd = kapital_sek / valutakurs if valutakurs > 0 else 0

    df = df.copy()
    df["Potential"] = df["Riktkurs om 1 år"] - df["Aktuell kurs"]
    df = df[df["Potential"] > 0]
    df = df.sort_values(by="Potential", ascending=False).reset_index(drop=True)

    st.markdown(f"💰 **Restkapital:** {round(kapital_sek, 2)} SEK")

    if "förslag_index" not in st.session_state:
        st.session_state["förslag_index"] = 0

    if st.button("Nästa förslag"):
        st.session_state["förslag_index"] += 1

    if st.session_state["förslag_index"] >= len(df):
        st.info("Inga fler förslag.")
        st.session_state["förslag_index"] = 0
        return

    rad = df.iloc[st.session_state["förslag_index"]]
    antal = int(kapital_usd // rad["Aktuell kurs"])
    total_sek = antal * rad["Aktuell kurs"] * valutakurs

    st.markdown(f"""
        **Förslag {st.session_state["förslag_index"] + 1}:**
        Köp **{antal} st {rad['Ticker']}** för ca **{round(total_sek,2)} SEK**  
        Riktkurs om 1 år: **{rad['Riktkurs om 1 år']} USD**, Aktuell kurs: **{rad['Aktuell kurs']} USD**
    """)

    # Ombalanseringsförslag
    st.subheader("📉 Ombalansering")
    max_andel = inställningar.get("max_portfoljandel", 20.0)
    max_hogrisk = inställningar.get("max_hogriskandel", 2.0)

    df_port = df[df["Antal aktier"] > 0].copy()
    df_port["Värde (SEK)"] = df_port["Antal aktier"] * df_port["Aktuell kurs"] * valutakurs
    total = df_port["Värde (SEK)"].sum()
    df_port["Andel (%)"] = (df_port["Värde (SEK)"] / total) * 100
    df_port["Högrisk"] = (df_port["Omsättning idag"] < 1000)

    minska = df_port[df_port["Andel (%)"] > max_andel]
    öka = df[df["Antal aktier"] == 0]
    högrisk = df_port[(df_port["Högrisk"]) & (df_port["Andel (%)"] > max_hogrisk)]

    with st.expander("📉 Bolag att minska i"):
        if minska.empty:
            st.write("Inga bolag att minska i.")
        else:
            st.dataframe(minska[["Ticker", "Andel (%)", "Värde (SEK)"]])

    with st.expander("📈 Bolag att öka i"):
        if öka.empty:
            st.write("Inga bolag med positiv potential.")
        else:
            st.dataframe(öka[["Ticker", "Potential", "Aktuell kurs"]])

    with st.expander("⚠️ Högriskvarningar"):
        if högrisk.empty:
            st.write("Inga högriskvarningar.")
        else:
            st.dataframe(högrisk[["Ticker", "Andel (%)", "Omsättning idag"]])

def visa_portfolj(df, valutakurs):
    st.subheader("📦 Min portfölj")

    df_port = df[df["Antal aktier"] > 0].copy()
    if df_port.empty:
        st.info("Du äger inga aktier.")
        return

    df_port["Värde (SEK)"] = df_port["Antal aktier"] * df_port["Aktuell kurs"] * valutakurs
    total_värde = df_port["Värde (SEK)"].sum()
    df_port["Andel (%)"] = round((df_port["Värde (SEK)"] / total_värde) * 100, 2)

    st.dataframe(
        df_port[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde (SEK)", "Andel (%)"]],
        use_container_width=True
    )

def visa_analysvy(df):
    st.subheader("📈 Analys")
    st.dataframe(df, use_container_width=True)

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 🟩 Läs data och inställningar
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    inställningar = load_settings()

    valutakurs = inställningar["Valutakurs"]
    max_portfoljandel = inställningar["Max portföljandel"]
    max_hogriskandel = inställningar["Max högriskandel"]

    # 🟩 Sidopanel – redigera inställningar
    st.sidebar.subheader("⚙️ Inställningar")
    ny_valutakurs = st.sidebar.number_input("Valutakurs (USD → SEK)", value=valutakurs, format="%.2f")
    ny_max_port = st.sidebar.number_input("Max portföljandel (%)", value=max_portfoljandel, format="%.2f")
    ny_max_risk = st.sidebar.number_input("Max högriskandel (%)", value=max_hogriskandel, format="%.2f")

    if st.sidebar.button("💾 Spara inställningar"):
        uppdatera_inställningar(ny_valutakurs, ny_max_port, ny_max_risk)
        st.sidebar.success("Inställningar uppdaterade. Ladda om appen för att se ändringar.")

    # 🟩 Meny
    meny = st.sidebar.radio("Meny", [
        "Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"
    ])

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        visa_analysvy(df)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurs, max_portfoljandel, max_hogriskandel)
    elif meny == "Portfölj":
        visa_portfolj(df, valutakurs)

if __name__ == "__main__":
    main()
