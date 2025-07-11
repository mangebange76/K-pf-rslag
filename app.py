import streamlit as st
import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    return sheet

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def konvertera_till_ratt_typ(df):
    numeriska_kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier"
    ]
    for kolumn in numeriska_kolumner:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce").fillna(0.0)
    return df

@st.cache_data(ttl=3600)
def hamta_valutakurs():
    try:
        url = "https://api.exchangerate.host/latest?base=USD&symbols=SEK"
        response = pd.read_json(url)
        return float(response["rates"]["SEK"])
    except:
        return None

def uppdatera_berakningar(df):
    ps_kvartal = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
    for index, row in df.iterrows():
        giltiga_ps = [row[k] for k in ps_kvartal if row[k] > 0]
        ps_snitt = round(np.mean(giltiga_ps), 2) if giltiga_ps else 0

        try:
            riktkurs_idag = (row["Omsättning idag"] * ps_snitt) / row["Utestående aktier"] if row["Utestående aktier"] > 0 else 0
            riktkurs_1 = (row["Omsättning nästa år"] * ps_snitt) / row["Utestående aktier"] if row["Utestående aktier"] > 0 else 0
            riktkurs_2 = (row["Omsättning om 2 år"] * ps_snitt) / row["Utestående aktier"] if row["Utestående aktier"] > 0 else 0
            riktkurs_3 = (row["Omsättning om 3 år"] * ps_snitt) / row["Utestående aktier"] if row["Utestående aktier"] > 0 else 0
        except:
            riktkurs_idag = riktkurs_1 = riktkurs_2 = riktkurs_3 = 0

        df.at[index, "P/S-snitt"] = ps_snitt
        df.at[index, "Riktkurs idag"] = round(riktkurs_idag, 2)
        df.at[index, "Riktkurs 2026"] = round(riktkurs_1, 2)
        df.at[index, "Riktkurs 2027"] = round(riktkurs_2, 2)
        df.at[index, "Riktkurs 2028"] = round(riktkurs_3, 2)
    return df

def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier"
    ]
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = 0.0 if "P/S" in kolumn or "Omsättning" in kolumn or "kurs" in kolumn.lower() else ""
    return df

def uppdatera_aktuell_kurs(df):
    for index, row in df.iterrows():
        ticker = row.get("Ticker", "")
        if not ticker:
            continue
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
            r = requests.get(url)
            data = r.json()
            ny_kurs = data["quoteResponse"]["result"][0]["regularMarketPrice"]
            if ny_kurs:
                df.at[index, "Aktuell kurs"] = round(ny_kurs, 2)
        except:
            st.warning(f"⚠️ Kursen kunde inte hämtas för {ticker}. Ange den manuellt om du vill uppdatera.")
    return df

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    with st.form("bolagsformulär"):
        ticker = st.text_input("Ticker").upper()
        namn = st.text_input("Bolagsnamn")
        kurs = st.number_input("Aktuell kurs (om automatisk hämtning ej fungerar)", value=0.0)
        aktier = st.number_input("Utestående aktier (miljoner)", value=0.0)
        ps_idag = st.number_input("P/S idag", value=0.0)
        ps1 = st.number_input("P/S Q1", value=0.0)
        ps2 = st.number_input("P/S Q2", value=0.0)
        ps3 = st.number_input("P/S Q3", value=0.0)
        ps4 = st.number_input("P/S Q4", value=0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner USD)", value=0.0)
        oms_1 = st.number_input("Omsättning nästa år", value=0.0)
        oms_2 = st.number_input("Omsättning om 2 år", value=0.0)
        oms_3 = st.number_input("Omsättning om 3 år", value=0.0)

        antal_aktier = st.number_input("Antal aktier du äger", value=0.0)

        sparaknapp = st.form_submit_button("💾 Spara bolag")

    if sparaknapp and ticker:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "Utestående aktier": aktier,
            "P/S": ps_idag,
            "P/S Q1": ps1,
            "P/S Q2": ps2,
            "P/S Q3": ps3,
            "P/S Q4": ps4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_1,
            "Omsättning om 2 år": oms_2,
            "Omsättning om 3 år": oms_3,
            "Antal aktier": antal_aktier
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} har uppdaterats.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} har lagts till.")
    return df

if "hoppade_over" not in st.session_state:
    st.session_state.hoppade_over = []

def investeringsforslag(df, kapital_sek, valutakurs):
    if valutakurs == 0:
        st.error("❌ Valutakursen är 0. Kan inte räkna om SEK till USD.")
        return [], kapital_sek

    df = df.copy()
    df = df[df["Riktkurs 2026"] > df["Aktuell kurs"]]
    df = df[~df["Ticker"].isin(st.session_state.hoppade_over)]
    df["Potential"] = df["Riktkurs 2026"] - df["Aktuell kurs"]
    df = df.sort_values(by="Potential", ascending=False)

    forslag = []
    kapital_usd = kapital_sek / valutakurs
    kapital_kvar = kapital_usd

    for _, rad in df.iterrows():
        pris = rad["Aktuell kurs"]
        if pris <= 0 or kapital_kvar < pris:
            continue

        antal = int(kapital_kvar // pris)
        if antal > 0:
            total_usd = antal * pris
            total_sek = round(total_usd * valutakurs, 2)
            forslag.append({
                "Ticker": rad["Ticker"],
                "Köp antal": antal,
                "Pris per aktie (USD)": pris,
                "Totalt (SEK)": total_sek
            })
            kapital_kvar -= total_usd
            break  # Visa bara ett förslag i taget

    return forslag, kapital_kvar * valutakurs

def visa_investeringsrad(df, valutakurs):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("💰 Tillgängligt kapital (SEK)", min_value=0.0, value=10000.0, step=500.0)

    df = uppdatera_berakningar(df)
    forslag, rest_sek = investeringsforslag(df, kapital_sek, valutakurs)

    if forslag:
        f = forslag[0]
        st.markdown(
            f"**Förslag:** Köp `{f['Köp antal']}` st `{f['Ticker']}` à `{f['Pris per aktie (USD)']}` USD – Totalt `{f['Totalt (SEK)']} SEK`"
        )
        st.markdown(f"💵 Kvarvarande kapital: `{round(rest_sek, 2)} SEK`")

        if st.button("⏭️ Nästa förslag"):
            st.session_state.hoppade_over.append(f["Ticker"])
            st.experimental_rerun()
    else:
        st.info("🚫 Inga fler bolag uppfyller kriterierna just nu.")

def visa_portfolj(df, valutakurs):
    st.subheader("📦 Min portfölj")
    df["Antal aktier"] = df["Antal aktier"].fillna(0.0)
    portfolj = df[df["Antal aktier"] > 0].copy()

    if portfolj.empty:
        st.info("📭 Du äger inga aktier just nu.")
        return

    portfolj["Värde i SEK"] = portfolj["Antal aktier"] * portfolj["Aktuell kurs"] * valutakurs
    totalvärde = portfolj["Värde i SEK"].sum()
    portfolj["Andel (%)"] = (portfolj["Värde i SEK"] / totalvärde * 100).round(2)

    visa_df = portfolj[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde i SEK", "Andel (%)"]]
    st.dataframe(visa_df, use_container_width=True)
    st.markdown(f"💼 **Totalt portföljvärde:** `{round(totalvärde, 2)} SEK`")

def visa_tabell(df):
    st.subheader("📈 Datatabell")
    st.dataframe(df, use_container_width=True)

def visa_valutakurs():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=SEK")
        data = r.json()
        kurs = data["rates"]["SEK"]
        st.sidebar.markdown(f"💱 **USD/SEK:** {round(kurs, 2)}")
        return kurs
    except:
        st.sidebar.warning("⚠️ Kunde inte hämta valutakurs.")
        return 0.0

# ---------------------------------------
# DEL 7: HUVUDFUNKTION – MAIN OCH START
# ---------------------------------------

def main():
    st.set_page_config(page_title="📈 Aktieanalys", layout="wide")
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Hämta och förbered data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_till_ratt_typ(df)

    # 2) Hämta valutakurs
    valutakurs = visa_valutakurs()

    # 3) Menyval
    menyval = st.sidebar.radio("📁 Meny", [
        "📊 Analys",
        "➕ Lägg till/uppdatera bolag",
        "🔁 Uppdatera värderingar",
        "💼 Investeringsråd",
        "📦 Portfölj"
    ])

    # 4) Rutin för varje vy
    if menyval == "📊 Analys":
        df = uppdatera_berakningar(df)
        visa_tabell(df)

    elif menyval == "➕ Lägg till/uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
        st.success("✅ Bolagsdatabasen är uppdaterad!")

    elif menyval == "🔁 Uppdatera värderingar":
        df = uppdatera_aktuell_kurs(df)
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.success("✅ Kurser och riktkurser uppdaterade!")

    elif menyval == "💼 Investeringsråd":
        visa_investeringsrad(df, valutakurs)

    elif menyval == "📦 Portfölj":
        visa_portfolj(df, valutakurs)

# Kör appen
if __name__ == "__main__":
    main()
