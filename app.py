import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# ---- Google Sheets-konfiguration ----
SHEET_URL = "https://docs.google.com/spreadsheets/d/1-5JSJpqBB0j7sm3cgEGZmnFoBL_oJDPMpLdleggL0HQ/edit?usp=drivesdk"
SHEET_NAME = "Blad1"
SETTINGS_SHEET = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("secrets/norremt.json", scope)
client = gspread.authorize(credentials)

# ---- Hjälpfunktioner ----
def load_data():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return konvertera_till_ratt_typ(df)

def save_data(df):
    df_clean = df.fillna("").astype(str)
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    sheet.update([df_clean.columns.values.tolist()] + df_clean.values.tolist())

def load_settings():
    try:
        sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        # Skapa nytt blad med standardinställningar
        sheet = client.open_by_url(SHEET_URL).add_worksheet(title=SETTINGS_SHEET, rows="10", cols="3")
        sheet.update("A1:C1", [["Inställning", "Värde", "Senast ändrad"]])
        sheet.update("A2:C4", [
            ["Valutakurs", "9.53", str(datetime.now().date())],
            ["Max portföljandel (%)", "20", str(datetime.now().date())],
            ["Max högriskandel (%)", "2", str(datetime.now().date())]
        ])
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_settings(settings):
    df = pd.DataFrame([
        {"Inställning": k, "Värde": v, "Senast ändrad": str(datetime.now().date())}
        for k, v in settings.items()
    ])
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    sheet.update([df.columns.tolist()] + df.values.tolist())

def konvertera_till_ratt_typ(df):
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        try:
            df[col] = df[col].astype(float)
        except:
            pass
    return df

# ---- Appens sektioner ----
def sidopanel():
    st.sidebar.title("Inställningar")
    settings_df = load_settings()
    inst_dict = {}
    for _, row in settings_df.iterrows():
        ny_val = st.sidebar.text_input(row["Inställning"], value=row["Värde"])
        inst_dict[row["Inställning"]] = ny_val
    if st.sidebar.button("💾 Spara inställningar"):
        save_settings(inst_dict)
        st.sidebar.success("Inställningar sparade!")

def visa_tabell(df):
    st.subheader("📊 Portföljöversikt")
    st.dataframe(df)

def lagg_till_bolag(df):
    st.subheader("➕ Lägg till eller uppdatera bolag")
    val = st.selectbox("Välj bolag att uppdatera eller skapa nytt", ["Nytt bolag"] + sorted(df["Bolag"].unique().tolist()))
    ny_rad = {}
    if val != "Nytt bolag":
        rad = df[df["Bolag"] == val].iloc[0]
        ny_rad = {k: st.text_input(k, str(v)) for k, v in rad.items()}
    else:
        for col in df.columns:
            ny_rad[col] = st.text_input(col)

    if st.button("💾 Spara bolag"):
        for key in ny_rad:
            ny_rad[key] = ny_rad[key].replace(",", ".")
        df = df[df["Bolag"] != ny_rad["Bolag"]]
        df = df.append(ny_rad, ignore_index=True)
        save_data(df)
        st.success("Bolag sparat!")
    return df

def investeringsforslag(df, kapital_sek, valutakurs):
    kapital_usd = float(kapital_sek) / float(valutakurs)
    max_andel = float(load_settings().set_index("Inställning").loc["Max portföljandel (%)"]["Värde"]) / 100
    max_risk_andel = float(load_settings().set_index("Inställning").loc["Max högriskandel (%)"]["Värde"]) / 100

    df["Portföljandel (%)"] = df["Position (USD)"] / df["Portföljvärde (USD)"] * 100
    minska = df[df["Portföljandel (%)"] > max_andel * 100]
    öka = df[(df["Potential (%)"] > 0) & (df["Portföljandel (%)"] < max_andel * 100)]

    högrisk = df[(df["Omsättning (milj USD)"] < 1000) & (df["Portföljandel (%)"] >= max_risk_andel * 100)]

    return {"Minska": minska, "Öka": öka, "Högrisk": högrisk}

def visa_investeringsforslag(df):
    st.subheader("📈 Investeringsförslag")
    inst = load_settings().set_index("Inställning")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=1000)
    valutakurs = float(inst.loc["Valutakurs"]["Värde"])
    forslag = investeringsforslag(df, kapital_sek, valutakurs)

    st.markdown("### 🔻 Minska innehav")
    st.dataframe(forslag["Minska"])

    st.markdown("### 🔺 Öka innehav")
    st.dataframe(forslag["Öka"])

    st.markdown("### ⚠️ Högriskvarningar")
    st.dataframe(forslag["Högrisk"])

# ---- Huvudfunktion ----
def main():
    st.title("📘 Investeringsanalysapp")
    sidopanel()
    df = load_data()

    meny = st.radio("Navigera", ["📊 Portfölj", "➕ Lägg till/uppdatera", "📈 Investeringsförslag"])
    if meny == "📊 Portfölj":
        visa_tabell(df)
    elif meny == "➕ Lägg till/uppdatera":
        df = lagg_till_bolag(df)
    elif meny == "📈 Investeringsförslag":
        visa_investeringsforslag(df)

if __name__ == "__main__":
    main()
