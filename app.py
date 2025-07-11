# app.py – Aktieanalys och investeringsförslag
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# ---------------------------------------
# KONFIGURATION
# ---------------------------------------
st.set_page_config(page_title="📈 Aktieanalys", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
DATA_SHEET = "Blad1"
SETTINGS_SHEET = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ---------------------------------------
# GOOGLE SHEETS
# ---------------------------------------

def get_data():
    sheet = client.open_by_url(SHEET_URL).worksheet(DATA_SHEET)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def save_data(df):
    sheet = client.open_by_url(SHEET_URL).worksheet(DATA_SHEET)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def load_settings():
    try:
        sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
        rows = sheet.get_all_records()
        return {row["Inställning"]: float(row["Värde"]) for row in rows}
    except:
        return {"Valutakurs": 10.0, "Max portföljandel (%)": 20.0, "Max högriskandel (%)": 2.0}

def save_settings(settings_dict):
    df = pd.DataFrame([{"Inställning": k, "Värde": v} for k, v in settings_dict.items()])
    df["Senast ändrad"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    sheet = client.open_by_url(SHEET_URL).worksheet(SETTINGS_SHEET)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# ---------------------------------------
# BERÄKNINGAR
# ---------------------------------------

def konvertera_typer(df):
    num_cols = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Aktuell kurs", "Antal aktier"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "Omsättning idag", "Omsättning nästa år",
        "Omsättning om 2 år", "Omsättning om 3 år", "P/S-snitt", "Riktkurs idag", 
        "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Antal aktier"
    ]
    for k in kolumner:
        if k not in df.columns:
            df[k] = 0.0 if "kurs" in k.lower() or "P/S" in k or "Omsättning" in k else ""
    return df

def uppdatera_berakningar(df):
    for i, row in df.iterrows():
        ps_values = [row[f"P/S Q{q}"] for q in range(1, 5) if row[f"P/S Q{q}"] > 0]
        ps_snitt = round(np.mean(ps_values), 2) if ps_values else 0
        df.at[i, "P/S-snitt"] = ps_snitt
        aktier = row["Utestående aktier"]
        df.at[i, "Riktkurs idag"] = round((row["Omsättning idag"] * ps_snitt / aktier), 2) if aktier > 0 else 0
        df.at[i, "Riktkurs 2026"] = round((row["Omsättning nästa år"] * ps_snitt / aktier), 2) if aktier > 0 else 0
        df.at[i, "Riktkurs 2027"] = round((row["Omsättning om 2 år"] * ps_snitt / aktier), 2) if aktier > 0 else 0
        df.at[i, "Riktkurs 2028"] = round((row["Omsättning om 3 år"] * ps_snitt / aktier), 2) if aktier > 0 else 0
    return df

# ---------------------------------------
# FORMULÄR
# ---------------------------------------

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    val = st.selectbox("Välj bolag att uppdatera eller skriv nytt", [""] + sorted(df["Ticker"].dropna().unique().tolist()))
    if val:
        existerande = df[df["Ticker"] == val].iloc[0].to_dict()
    else:
        existerande = {k: 0.0 for k in df.columns}
        existerande["Ticker"] = ""

    with st.form("form_bolag"):
        ticker = st.text_input("Ticker", value=existerande["Ticker"]).upper()
        namn = st.text_input("Bolagsnamn", value=existerande["Bolagsnamn"])
        kurs = st.number_input("Aktuell kurs", value=existerande["Aktuell kurs"])
        u_aktier = st.number_input("Utestående aktier", value=existerande["Utestående aktier"])
        ps_q1 = st.number_input("P/S Q1", value=existerande["P/S Q1"])
        ps_q2 = st.number_input("P/S Q2", value=existerande["P/S Q2"])
        ps_q3 = st.number_input("P/S Q3", value=existerande["P/S Q3"])
        ps_q4 = st.number_input("P/S Q4", value=existerande["P/S Q4"])
        oms_idag = st.number_input("Omsättning idag", value=existerande["Omsättning idag"])
        oms1 = st.number_input("Omsättning nästa år", value=existerande["Omsättning nästa år"])
        oms2 = st.number_input("Omsättning om 2 år", value=existerande["Omsättning om 2 år"])
        oms3 = st.number_input("Omsättning om 3 år", value=existerande["Omsättning om 3 år"])
        antal = st.number_input("Antal aktier du äger", value=existerande["Antal aktier"])
        spara = st.form_submit_button("💾 Spara")

    if spara and ticker:
        ny = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Utestående aktier": u_aktier,
            "P/S Q1": ps_q1, "P/S Q2": ps_q2, "P/S Q3": ps_q3, "P/S Q4": ps_q4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms1,
            "Omsättning om 2 år": oms2, "Omsättning om 3 år": oms3,
            "Antal aktier": antal
        }
        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny.keys()] = ny.values()
            st.success(f"Uppdaterade {ticker}")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success(f"La till {ticker}")
    return df

# ---------------------------------------
# INVESTERINGSFÖRSLAG
# ---------------------------------------

def investeringsforslag(df, kapital_sek, valutakurs, max_andel, max_hr_andel):
    df = df.copy()
    df["Potential"] = df["Riktkurs 2026"] - df["Aktuell kurs"]
    df = df.sort_values(by="Potential", ascending=False)

    portfolj = df[df["Antal aktier"] > 0]
    portfolj["Värde i SEK"] = portfolj["Antal aktier"] * portfolj["Aktuell kurs"] * valutakurs
    totalvarde = portfolj["Värde i SEK"].sum()

    forslag = []
    kapital_usd = kapital_sek / valutakurs

    for _, rad in df.iterrows():
        if rad["Aktuell kurs"] <= 0 or rad["Ticker"] in portfolj["Ticker"].values:
            continue

        kurs = rad["Aktuell kurs"]
        bolagsvarde = portfolj.loc[portfolj["Ticker"] == rad["Ticker"], "Värde i SEK"].sum()
        andel = bolagsvarde / totalvarde * 100 if totalvarde > 0 else 0

        högrisk = rad["Omsättning om 2 år"] < 1000
        maxandel = max_hr_andel if högrisk else max_andel

        if andel >= maxandel:
            continue

        antal = int(kapital_usd // kurs)
        if antal == 0:
            forslag.append((rad["Ticker"], kurs, True))
            continue

        forslag.append((rad["Ticker"], kurs, False))
        break

    return forslag

def visa_investeringsrad(df, valutakurs, inst):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("💰 Tillgängligt kapital (SEK)", value=10000.0)
    df = uppdatera_berakningar(df)

    forslag = investeringsforslag(df, kapital_sek, valutakurs, inst["Max portföljandel (%)"], inst["Max högriskandel (%)"])
    if not forslag:
        st.info("Inga bolag uppfyller kriterierna")
        return

    ticker, pris, räcker_inte = forslag[0]
    if räcker_inte:
        st.warning(f"💸 `{ticker}` ser lovande ut men tillgängligt kapital räcker inte till ett köp (pris {pris} USD).")
    else:
        st.success(f"Köp `{ticker}` á `{pris} USD` föreslås!")

# ---------------------------------------
# SIDOMENY & MAIN
# ---------------------------------------

def main():
    inst = load_settings()
    with st.sidebar:
        st.markdown("⚙️ **Inställningar**")
        valutakurs = st.number_input("USD/SEK", value=inst["Valutakurs"])
        max_andel = st.number_input("Max portföljandel (%)", value=inst["Max portföljandel (%)"])
        max_hr = st.number_input("Max andel högrisk (%)", value=inst["Max högriskandel (%)"])
        if st.button("💾 Spara inställningar"):
            save_settings({
                "Valutakurs": valutakurs,
                "Max portföljandel (%)": max_andel,
                "Max högriskandel (%)": max_hr
            })
            st.success("Inställningar sparade!")

    df = get_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("Meny", ["📊 Analys", "➕ Uppdatera bolag", "💡 Investeringsförslag", "📦 Portfölj"])
    if meny == "📊 Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df, use_container_width=True)
    elif meny == "➕ Uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        save_data(df)
    elif meny == "💡 Investeringsförslag":
        visa_investeringsrad(df, valutakurs, inst)
    elif meny == "📦 Portfölj":
        port = df[df["Antal aktier"] > 0].copy()
        port["Värde i SEK"] = port["Antal aktier"] * port["Aktuell kurs"] * valutakurs
        st.dataframe(port[["Ticker", "Bolagsnamn", "Antal aktier", "Värde i SEK"]], use_container_width=True)

if __name__ == "__main__":
    main()
