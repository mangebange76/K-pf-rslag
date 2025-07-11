import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import yfinance as yf
import requests

# -------------------------------
# KONFIGURATION – GOOGLE SHEETS
# -------------------------------

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

def skapa_koppling():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    return sheet

# -------------------------------------
# HÄMTA OCH SPARA DATA TILL GOOGLE SHEET
# -------------------------------------

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

def spara_data(sheet, df):
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# --------------------------
# KOLUMNER & DATATYPER
# --------------------------

REQUIRED_COLUMNS = [
    "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta",
    "Utestående aktier", "Omsättning idag", "Omsättning nästa år", "Omsättning om två år",
    "P/S idag", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt",
    "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027",
    "Undervärdering idag", "Undervärdering 2026", "Undervärdering 2027",
    "Antal aktier", "Kommentar"
]

def säkerställ_kolumner(df):
    for kolumn in REQUIRED_COLUMNS:
        if kolumn not in df.columns:
            df[kolumn] = 0.0 if any(x in kolumn.lower() for x in ["kurs", "p/s", "omsättning", "undervärdering", "aktier"]) else ""
    return df

def konvertera_till_ratt_typ(df):
    numeriska = [
        "Aktuell kurs", "Utestående aktier",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om två år",
        "P/S idag", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt",
        "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027",
        "Undervärdering idag", "Undervärdering 2026", "Undervärdering 2027",
        "Antal aktier"
    ]
    for kolumn in numeriska:
        if kolumn in df.columns:
            df[kolumn] = pd.to_numeric(df[kolumn], errors="coerce").fillna(0.0)
    return df

def beräkna_snitt_ps(rad):
    värden = [rad.get(f"P/S Q{i}", 0) for i in range(1, 5)]
    giltiga = [v for v in värden if v > 0]
    return round(sum(giltiga) / len(giltiga), 2) if giltiga else 0.0

def beräkna_riktkurs(omsättning, snitt_ps, aktier):
    if omsättning > 0 and snitt_ps > 0 and aktier > 0:
        return round((omsättning * snitt_ps) / aktier, 2)
    return 0.0

def beräkna_undervärdering(riktkurs, aktuell_kurs):
    if riktkurs > 0 and aktuell_kurs > 0:
        return round(((riktkurs - aktuell_kurs) / aktuell_kurs) * 100, 2)
    return 0.0

# --------------------------
# KURSUPPDATERING & BERÄKNINGAR
# --------------------------

def uppdatera_aktuell_kurs(df):
    for i, rad in df.iterrows():
        ticker = rad["Ticker"]
        try:
            aktie = yf.Ticker(ticker)
            pris = aktie.info.get("currentPrice", None)
            if pris:
                df.at[i, "Aktuell kurs"] = round(pris, 2)
        except Exception:
            st.warning(f"Kunde inte hämta kurs för {ticker}")
    return df

def uppdatera_beräkningar(df):
    for i, rad in df.iterrows():
        snitt_ps = beräkna_snitt_ps(rad)
        df.at[i, "P/S snitt"] = snitt_ps

        df.at[i, "Riktkurs idag"] = beräkna_riktkurs(rad["Omsättning idag"], snitt_ps, rad["Utestående aktier"])
        df.at[i, "Riktkurs 2026"] = beräkna_riktkurs(rad["Omsättning nästa år"], snitt_ps, rad["Utestående aktier"])
        df.at[i, "Riktkurs 2027"] = beräkna_riktkurs(rad["Omsättning om två år"], snitt_ps, rad["Utestående aktier"])

        df.at[i, "Undervärdering idag"] = beräkna_undervärdering(df.at[i, "Riktkurs idag"], rad["Aktuell kurs"])
        df.at[i, "Undervärdering 2026"] = beräkna_undervärdering(df.at[i, "Riktkurs 2026"], rad["Aktuell kurs"])
        df.at[i, "Undervärdering 2027"] = beräkna_undervärdering(df.at[i, "Riktkurs 2027"], rad["Aktuell kurs"])
    return df

# --------------------------
# LÄGG TILL/UPPDATERA BOLAG
# --------------------------

def lagg_till_bolag(df):
    with st.form("lägg_till_bolag_form"):
        ticker = st.text_input("Ticker").upper()
        namn = st.text_input("Bolagsnamn")
        kurs = st.number_input("Aktuell kurs (USD)", min_value=0.0, value=0.0)
        ps1 = st.number_input("P/S Q1", min_value=0.0, value=0.0)
        ps2 = st.number_input("P/S Q2", min_value=0.0, value=0.0)
        ps3 = st.number_input("P/S Q3", min_value=0.0, value=0.0)
        ps4 = st.number_input("P/S Q4", min_value=0.0, value=0.0)
        oms_idag = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0, value=0.0)
        oms_2026 = st.number_input("Omsättning nästa år (miljoner USD)", min_value=0.0, value=0.0)
        oms_2027 = st.number_input("Omsättning om två år (miljoner USD)", min_value=0.0, value=0.0)
        aktier = st.number_input("Utestående aktier (miljoner)", min_value=0.0, value=0.0)
        antal_aktier = st.number_input("Antal aktier (ägda)", min_value=0.0, value=0.0)

        submit = st.form_submit_button("Spara")

    if submit and ticker:
        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Aktuell kurs": kurs,
            "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_2026,
            "Omsättning om två år": oms_2027,
            "Utestående aktier": aktier,
            "Antal aktier": antal_aktier,
        }

        df = säkerställ_kolumner(df)

        index = df[df["Ticker"] == ticker].index
        if not index.empty:
            for k, v in ny_rad.items():
                df.at[index[0], k] = v
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)

        st.success(f"{ticker} har sparats.")
    return df

# --------------------------
# INVESTERINGSFÖRSLAG
# --------------------------

if "hoppade_over" not in st.session_state:
    st.session_state.hoppade_over = []

def investeringsforslag(df, kapital):
    df = df[df["Riktkurs 2026"] > df["Aktuell kurs"]]
    df = df[~df["Ticker"].isin(st.session_state.hoppade_over)]
    df = df.copy()
    df["Potential"] = df["Riktkurs 2026"] - df["Aktuell kurs"]
    df = df.sort_values(by="Potential", ascending=False)

    forslag = []
    kapital_kvar = kapital

    for i, rad in df.iterrows():
        pris = rad["Aktuell kurs"]
        if pris <= 0 or kapital_kvar < pris:
            continue

        antal = int(kapital_kvar // pris)
        if antal > 0:
            forslag.append({
                "Ticker": rad["Ticker"],
                "Köp antal": antal,
                "Pris per aktie": pris,
                "Totalt": round(antal * pris, 2)
            })
            break

    return forslag, kapital_kvar

def visa_investeringsrad(df):
    st.subheader("📌 Investeringsförslag")
    kapital = st.number_input("💰 Tillgängligt kapital (USD)", min_value=0.0, value=1000.0, step=100.0)
    df = uppdatera_beräkningar(df)
    forslag, rest = investeringsforslag(df, kapital)

    if forslag:
        f = forslag[0]
        st.markdown(f"- **{f['Ticker']}**: Köp {f['Köp antal']} st à {f['Pris per aktie']} USD (Totalt {f['Totalt']} USD)")
        st.markdown(f"💵 **Kvarvarande kapital:** {round(rest, 2)} USD")
        if st.button("⏭️ Nästa förslag"):
            st.session_state.hoppade_over.append(f["Ticker"])
            st.experimental_rerun()
    else:
        st.info("🚫 Inga fler förslag just nu. Starta om sidan för att återställa listan.")

# --------------------------
# PORTFÖLJ & VALUTAKURS
# --------------------------

def visa_portfolj(df, valutakurs):
    st.subheader("📦 Min portfölj")
    df["Antal aktier"] = df["Antal aktier"].fillna(0.0)
    portfolj = df[df["Antal aktier"] > 0].copy()

    if portfolj.empty:
        st.info("Du äger inga aktier just nu.")
        return

    portfolj["Värde i SEK"] = portfolj["Antal aktier"] * portfolj["Aktuell kurs"] * valutakurs
    totalvärde = portfolj["Värde i SEK"].sum()
    portfolj["Andel (%)"] = (portfolj["Värde i SEK"] / totalvärde * 100).round(2)

    visa_df = portfolj[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Värde i SEK", "Andel (%)"]]
    st.dataframe(visa_df, use_container_width=True)
    st.markdown(f"💼 **Totalt portföljvärde:** {round(totalvärde, 2)} SEK")

def visa_valutakurs():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=SEK")
        data = r.json()
        kurs = data["rates"]["SEK"]
        st.sidebar.markdown(f"💱 **USD/SEK:** {round(kurs, 2)}")
        return kurs
    except:
        st.sidebar.warning("Kunde inte hämta valutakurs.")
        return 0.0

# --------------------------
# ANALYSVY – visa hela datatabellen
# --------------------------

def visa_tabell(df):
    st.subheader("📈 Datatabell")
    st.dataframe(df, use_container_width=True)

# --------------------------
# HUVUDFUNKTION
# --------------------------

def main():
    st.set_page_config(page_title="📈 Aktieanalys", layout="wide")
    st.title("📊 Aktieanalys och investeringsförslag")

    sheet = skapa_koppling()
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_till_ratt_typ(df)

    valutakurs = visa_valutakurs()

    menyval = st.sidebar.radio("📁 Meny", [
        "📊 Analys",
        "➕ Lägg till/uppdatera bolag",
        "🔁 Uppdatera värderingar",
        "💼 Investeringsråd",
        "📦 Portfölj"
    ])

    if menyval == "📊 Analys":
        df = uppdatera_beräkningar(df)
        visa_tabell(df)

    elif menyval == "➕ Lägg till/uppdatera bolag":
        df = lagg_till_bolag(df)
        spara_data(sheet, df)

    elif menyval == "🔁 Uppdatera värderingar":
        df = uppdatera_aktuell_kurs(df)
        df = uppdatera_beräkningar(df)
        spara_data(sheet, df)
        st.success("✅ Alla kurser och värderingar har uppdaterats!")

    elif menyval == "💼 Investeringsråd":
        visa_investeringsrad(df)

    elif menyval == "📦 Portfölj":
        visa_portfolj(df, valutakurs)

if __name__ == "__main__":
    main()
