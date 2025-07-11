# ---------------------------------------
# app.py – Aktieanalys och investeringsförslag
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ---------------------------------------
# KONFIGURATION
# ---------------------------------------

st.set_page_config(page_title="📈 Aktieanalys", layout="wide")
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
INSTÄLLNINGAR_SHEET = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling(sheet_name):
    return client.open_by_url(SHEET_URL).worksheet(sheet_name)

# ---------------------------------------
# HÄMTA OCH SPARA DATA
# ---------------------------------------

def hamta_data():
    sheet = skapa_koppling(SHEET_NAME)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling(SHEET_NAME)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def hamta_inställningar():
    sheet = skapa_koppling(INSTÄLLNINGAR_SHEET)
    data = sheet.get_all_records()
    return {rad["Inställning"]: float(rad["Värde"]) for rad in data}

def spara_inställning(namn, värde):
    sheet = skapa_koppling(INSTÄLLNINGAR_SHEET)
    df = pd.DataFrame(sheet.get_all_records())
    if namn in df["Inställning"].values:
        index = df[df["Inställning"] == namn].index[0]
        df.at[index, "Värde"] = värde
    else:
        df = pd.concat([df, pd.DataFrame([{"Inställning": namn, "Värde": värde}])], ignore_index=True)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# ---------------------------------------
# DATAPREPARERING
# ---------------------------------------

def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Senast uppdaterad"
    ]
    for kol in kolumner:
        if kol not in df.columns:
            df[kol] = 0.0 if "P/S" in kol or "Omsättning" in kol or "kurs" in kol.lower() else ""
    return df

def konvertera_typer(df):
    numeriska_kolumner = [k for k in df.columns if any(x in k for x in ["P/S", "Omsättning", "kurs", "aktier"])]
    for kol in numeriska_kolumner:
        df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def uppdatera_beräkningar(df):
    for index, row in df.iterrows():
        ps_kv = [row["P/S Q1"], row["P/S Q2"], row["P/S Q3"], row["P/S Q4"]]
        ps_giltiga = [x for x in ps_kv if x > 0]
        ps_snitt = round(np.mean(ps_giltiga), 2) if ps_giltiga else 0
        aktier = row["Utestående aktier"]
        df.at[index, "P/S-snitt"] = ps_snitt
        for i, label in enumerate(["idag", "2026", "2027", "2028"]):
            oms = row["Omsättning idag"] if i == 0 else row[f"Omsättning om {i} år"] if i > 1 else row["Omsättning nästa år"]
            riktkurs = (oms * ps_snitt) / aktier if aktier > 0 else 0
            df.at[index, f"Riktkurs {label}"] = round(riktkurs, 2)
    return df

# ---------------------------------------
# FORMULÄR – LÄGG TILL / UPPDATERA BOLAG
# ---------------------------------------

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    tickers_sorterade = sorted(df["Ticker"].dropna().unique())
    val = st.selectbox("Välj bolag att uppdatera eller lämna tomt för nytt", [""] + tickers_sorterade)

    befintlig = df[df["Ticker"] == val].iloc[0] if val else {}

    with st.form("bolagsform"):
        ticker = st.text_input("Ticker", value=val).upper()
        namn = st.text_input("Bolagsnamn", value=befintlig.get("Bolagsnamn", ""))
        kurs = st.number_input("Aktuell kurs (USD)", value=float(befintlig.get("Aktuell kurs", 0.0)))
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)))
        ps = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)))
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)))
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)))
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)))
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)))
        oms_idag = st.number_input("Omsättning idag (milj USD)", value=float(befintlig.get("Omsättning idag", 0.0)))
        oms1 = st.number_input("Omsättning nästa år", value=float(befintlig.get("Omsättning nästa år", 0.0)))
        oms2 = st.number_input("Omsättning om 2 år", value=float(befintlig.get("Omsättning om 2 år", 0.0)))
        oms3 = st.number_input("Omsättning om 3 år", value=float(befintlig.get("Omsättning om 3 år", 0.0)))
        antal = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)))
        spara = st.form_submit_button("💾 Spara")

    if spara and ticker:
        ny = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Utestående aktier": aktier,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms1, "Omsättning om 2 år": oms2,
            "Omsättning om 3 år": oms3, "Antal aktier": antal,
            "Senast uppdaterad": datetime.now().strftime("%Y-%m-%d")
        }
        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny.keys()] = ny.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

# (fortsättning kommer i nästa cell p.g.a. längdbegränsning)


# ---------------------------------------
# DEL 8: INSTÄLLNINGAR – SPARAS I GOOGLE SHEETS
# ---------------------------------------

INSTÄLLNINGAR_SHEET = "Inställningar"

def hamta_inställningar():
    try:
        sheet = client.open_by_url(SHEET_URL).worksheet(INSTÄLLNINGAR_SHEET)
        inst = sheet.get_all_records()
        if inst:
            return inst[0]
    except:
        pass
    return {"max_portfoljandel": 20.0, "max_hogriskandel": 2.0}

def spara_inställningar(max_portfoljandel, max_hogriskandel):
    sheet = client.open_by_url(SHEET_URL).worksheet(INSTÄLLNINGAR_SHEET)
    sheet.clear()
    sheet.update([["max_portfoljandel", "max_hogriskandel"], [max_portfoljandel, max_hogriskandel]])

def visa_sidomeny_inställningar():
    st.sidebar.subheader("⚙️ Inställningar")
    inst = hamta_inställningar()
    max_portf = st.sidebar.number_input("Max portföljandel (%)", min_value=1.0, max_value=100.0, value=float(inst.get("max_portfoljandel", 20.0)))
    max_hogrisk = st.sidebar.number_input("Max andel i högriskbolag (%)", min_value=0.5, max_value=10.0, value=float(inst.get("max_hogriskandel", 2.0)))

    if st.sidebar.button("💾 Spara inställningar"):
        spara_inställningar(max_portf, max_hogrisk)
        st.sidebar.success("✅ Inställningar sparade.")

# ---------------------------------------
# DEL 9: HÖGRISK- OCH STORA POSITIONSVARNINGAR
# ---------------------------------------

def kontrollera_risker(df, valutakurs):
    portfolj = df[df["Antal aktier"] > 0].copy()
    portfolj["Värde i SEK"] = portfolj["Antal aktier"] * portfolj["Aktuell kurs"] * valutakurs
    total_varde = portfolj["Värde i SEK"].sum()

    inst = hamta_inställningar()
    max_andel = float(inst.get("max_portfoljandel", 20.0))
    max_hogrisk = float(inst.get("max_hogriskandel", 2.0))

    varningar = []
    for _, row in portfolj.iterrows():
        andel = row["Värde i SEK"] / total_varde * 100
        framtida_oms = max(row["Omsättning om 2 år"], row["Omsättning om 3 år"])
        if framtida_oms < 1000 and andel >= max_hogrisk:
            varningar.append(f"⚠️ {row['Ticker']}: Högriskinnehav på {round(andel,2)}% med omsättning {framtida_oms} MUSD")
        if andel > max_andel:
            varningar.append(f"⚠️ {row['Ticker']}: Portföljandel är {round(andel,2)}% (över {max_andel}%)")
    return varningar

# ---------------------------------------
# DEL 10: START MED HUVUDFUNKTION IGEN
# ---------------------------------------

def main():
    st.set_page_config(page_title="📈 Aktieanalys", layout="wide")
    st.title("📊 Aktieanalys och investeringsförslag")

    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_till_ratt_typ(df)

    visa_sidomeny_inställningar()
    valutakurs = visa_valutakurs_sparad()

    menyval = st.sidebar.radio("📁 Meny", [
        "📊 Analys",
        "➕ Lägg till/uppdatera bolag",
        "🔁 Uppdatera värderingar",
        "💼 Investeringsråd",
        "📦 Portfölj"
    ])

    if menyval == "📊 Analys":
        df = uppdatera_berakningar(df)
        visa_tabell(df)

    elif menyval == "➕ Lägg till/uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif menyval == "🔁 Uppdatera värderingar":
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.success("✅ Alla värderingar har uppdaterats!")

    elif menyval == "💼 Investeringsråd":
        visa_investeringsrad(df, valutakurs)

    elif menyval == "📦 Portfölj":
        visa_portfolj(df, valutakurs)
        for v in kontrollera_risker(df, valutakurs):
            st.warning(v)

# ---------------------------------------
# KÖR APPEN
# ---------------------------------------

if __name__ == "__main__":
    main()
