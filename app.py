# ---------------------------------------
# Aktieanalys & investeringsförslag – Manuell valutakurs och aktiekurs (med inställningsfixar)
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---------------------------------------
# KONFIGURATION OCH GOOGLE SHEETS
# ---------------------------------------

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def skapa_inställningsblad():
    fil = client.open_by_url(SHEET_URL)
    if SETTINGS_SHEET not in [w.title for w in fil.worksheets()]:
        sheet = fil.add_worksheet(title=SETTINGS_SHEET, rows="10", cols="4")
        sheet.update("A1:D1", [["Inställning", "Värde", "Typ", "Senast ändrad"]])
        sheet.append_row(["Valutakurs", "10.0", "float", str(datetime.today().date())])
        sheet.append_row(["Max portföljandel (%)", "20.0", "float", str(datetime.today().date())])
        sheet.append_row(["Max högriskandel (%)", "2.0", "float", str(datetime.today().date())])

def load_settings():
    skapa_inställningsblad()
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    data = sheet.get_all_records()
    settings = {}
    for rad in data:
        try:
            value = float(str(rad["Värde"]).replace(",", "."))
        except:
            value = 0.0
        settings[rad["Inställning"]] = value
    return settings

def save_settings(new_settings):
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    data = sheet.get_all_records()
    for idx, row in enumerate(data, start=2):
        inst_namn = row["Inställning"]
        if inst_namn in new_settings:
            nytt_värde = str(new_settings[inst_namn]).replace(",", ".")
            sheet.update(f"B{idx}", nytt_värde)
            sheet.update(f"D{idx}", str(datetime.today().date()))

# ---------------------------------------
# DATAHANTERING OCH BERÄKNINGAR
# ---------------------------------------

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    df = df.fillna("")
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

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
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år",
        "P/S-snitt", "Riktkurs nu", "Riktkurs om 1 år", "Riktkurs om 2 år", "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            df[kol] = 0.0 if "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() else ""
    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs nu"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df

# ---------------------------------------
# INMATNING AV BOLAG & INSTÄLLNINGAR
# ---------------------------------------

def skapa_inställningsblad():
    try:
        sheet = client.open_by_url(SHEET_URL)
        if "Inställningar" not in [w.title for w in sheet.worksheets()]:
            inst_sheet = sheet.add_worksheet(title="Inställningar", rows=10, cols=2)
            inst_sheet.update("A1:B4", [
                ["Inställning", "Värde"],
                ["Valutakurs", "10.0"],
                ["Max portföljandel (%)", "20.0"],
                ["Max högriskandel (%)", "2.0"],
            ])
    except Exception as e:
        st.error(f"Kunde inte skapa inställningsblad: {e}")

def load_settings():
    skapa_inställningsblad()
    sheet = client.open_by_url(SHEET_URL).worksheet("Inställningar")
    data = sheet.get_all_records()
    inst = {rad["Inställning"]: float(str(rad["Värde"]).replace(",", ".")) for rad in data if rad["Inställning"] and rad["Värde"]}
    return inst

def save_settings(inst):
    sheet = client.open_by_url(SHEET_URL).worksheet("Inställningar")
    sheet.update("B2", str(inst["Valutakurs"]))
    sheet.update("B3", str(inst["Max portföljandel (%)"]))
    sheet.update("B4", str(inst["Max högriskandel (%)"]))

def inställningar_sidopanel():
    inst = load_settings()
    st.sidebar.subheader("⚙️ Inställningar")

    valutakurs = st.sidebar.number_input("Valutakurs USD → SEK", value=inst.get("Valutakurs", 10.0), step=0.1, format="%.2f")
    max_andel = st.sidebar.number_input("Max portföljandel (%)", value=inst.get("Max portföljandel (%)", 20.0), step=0.1, format="%.2f")
    max_högrisk = st.sidebar.number_input("Max högriskandel (%)", value=inst.get("Max högriskandel (%)", 2.0), step=0.1, format="%.2f")

    if st.sidebar.button("💾 Spara inställningar"):
        save_settings({
            "Valutakurs": valutakurs,
            "Max portföljandel (%)": max_andel,
            "Max högriskandel (%)": max_högrisk,
        })
        st.sidebar.success("Inställningar sparade!")

    return valutakurs, max_andel, max_högrisk

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = df["Ticker"].dropna().unique().tolist()
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
# INVESTERINGSFÖRSLAG & PORTFÖLJVY
# ---------------------------------------

def visa_investeringsforslag(df, valutakurs, max_andel, max_högrisk):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=500.0)
    kapital_usd = kapital_sek / valutakurs if valutakurs > 0 else 0

    df = df.copy()
    df["Potential"] = df["Riktkurs om 1 år"] - df["Aktuell kurs"]
    df = df[df["Potential"] > 0]
    df = df.sort_values(by="Potential", ascending=False).reset_index(drop=True)

    if "visat_index" not in st.session_state:
        st.session_state.visat_index = 0

    if not df.empty:
        i = st.session_state.visat_index % len(df)
        rad = df.iloc[i]

        totalpris_usd = rad["Aktuell kurs"]
        antal = int(kapital_usd // totalpris_usd)

        st.markdown(f"**{rad['Bolagsnamn']} ({rad['Ticker']})**")
        st.markdown(f"Aktuell kurs: {rad['Aktuell kurs']} USD")
        st.markdown(f"Riktkurs om 1 år: {rad['Riktkurs om 1 år']} USD → **Potential: {round(rad['Potential'],2)} USD**")
        st.markdown(f"Kapital räcker till: {antal} aktier")

        if st.button("➡️ Nästa förslag"):
            st.session_state.visat_index += 1
    else:
        st.info("Inga bolag med positiv potential just nu.")

    visa_ombalansering(df, valutakurs, max_andel, max_högrisk)

def visa_ombalansering(df, valutakurs, max_andel, max_högrisk):
    df_portfölj = df[df["Antal aktier"] > 0].copy()
    df_portfölj["Värde (SEK)"] = df_portfölj["Antal aktier"] * df_portfölj["Aktuell kurs"] * valutakurs
    tot_värde = df_portfölj["Värde (SEK)"].sum()
    df_portfölj["Andel (%)"] = df_portfölj["Värde (SEK)"] / tot_värde * 100
    df_portfölj["Omsättning idag"] = df_portfölj["Omsättning idag"].astype(float)

    minska = df_portfölj[df_portfölj["Andel (%)"] > max_andel]
    öka = df_portfölj[(df_portfölj["Potential"] > 0) & (df_portfölj["Andel (%)"] < max_andel)]
    högrisk = df_portfölj[(df_portfölj["Omsättning idag"] < 1000) & (df_portfölj["Andel (%)"] >= max_högrisk)]

    with st.expander("🔻 Bolag att minska i"):
        if minska.empty:
            st.write("Inga bolag över max andel.")
        else:
            st.dataframe(minska[["Ticker", "Andel (%)", "Värde (SEK)"]], use_container_width=True)

    with st.expander("📈 Bolag att öka i"):
        if öka.empty:
            st.write("Inga bolag att öka i just nu.")
        else:
            st.dataframe(öka[["Ticker", "Potential", "Andel (%)"]], use_container_width=True)

    with st.expander("⚠️ Högriskvarningar"):
        if högrisk.empty:
            st.write("Inga högriskbolag över gräns.")
        else:
            st.dataframe(högrisk[["Ticker", "Omsättning idag", "Andel (%)"]], use_container_width=True)

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
# HUVUDFUNKTION
# ---------------------------------------

def skapa_inställningsblad():
    try:
        sheet = client.open_by_url(SHEET_URL)
        bladnamn = "Inställningar"
        existerar = any(ws.title == bladnamn for ws in sheet.worksheets())
        if not existerar:
            sheet.add_worksheet(title=bladnamn, rows="10", cols="3")
            ws = sheet.worksheet(bladnamn)
            ws.append_row(["Inställning", "Värde", "Senast ändrad"])
            ws.append_row(["Valutakurs", "10.0", str(pd.Timestamp.today().date())])
            ws.append_row(["Max portföljandel (%)", "20.0", str(pd.Timestamp.today().date())])
            ws.append_row(["Max högriskandel (%)", "2.0", str(pd.Timestamp.today().date())])
    except Exception as e:
        st.error(f"Fel vid skapande av inställningsblad: {e}")

def load_settings():
    skapa_inställningsblad()
    ws = client.open_by_url(SHEET_URL).worksheet("Inställningar")
    df = pd.DataFrame(ws.get_all_records())
    df["Värde"] = df["Värde"].astype(str).str.replace(",", ".").astype(float)
    settings = {row["Inställning"]: row["Värde"] for _, row in df.iterrows()}
    return settings

def save_setting(namn, nytt_värde):
    ws = client.open_by_url(SHEET_URL).worksheet("Inställningar")
    cell = ws.find(namn)
    ws.update_cell(cell.row, 2, str(nytt_värde).replace(",", "."))
    ws.update_cell(cell.row, 3, str(pd.Timestamp.today().date()))

def main():
    st.set_page_config(layout="wide")
    st.title("📈 Aktieanalys")

    # Ladda inställningar
    inställningar = load_settings()
    valutakurs = inställningar.get("Valutakurs", 10.0)
    max_andel = inställningar.get("Max portföljandel (%)", 20.0)
    max_högrisk = inställningar.get("Max högriskandel (%)", 2.0)

    # Sidopanel för inställningar
    st.sidebar.header("⚙️ Inställningar")
    ny_valuta = st.sidebar.number_input("Valutakurs (USD → SEK)", value=valutakurs, step=0.1)
    ny_max = st.sidebar.number_input("Max portföljandel (%)", value=max_andel, step=0.5)
    ny_hrisk = st.sidebar.number_input("Max högriskandel (%)", value=max_högrisk, step=0.5)

    if ny_valuta != valutakurs:
        save_setting("Valutakurs", ny_valuta)
    if ny_max != max_andel:
        save_setting("Max portföljandel (%)", ny_max)
    if ny_hrisk != max_högrisk:
        save_setting("Max högriskandel (%)", ny_hrisk)

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, ny_valuta, ny_max, ny_hrisk)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, ny_valuta)

if __name__ == "__main__":
    main()
