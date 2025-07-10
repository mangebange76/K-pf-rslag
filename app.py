import streamlit as st
import pandas as pd
import gspread
from google.oauth2 import service_account
import yfinance as yf
import datetime

# ==== Google Sheets Setup ====
secrets = st.secrets
SHEET_URL = secrets["SHEET_URL"]
credentials_dict = secrets["GOOGLE_CREDENTIALS"]

scoped_credentials = service_account.Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
client = gspread.authorize(scoped_credentials)

# Flexibel Sheet-anslutning: använder SHEET_NAME om det finns, annars första arket
try:
    SHEET_NAME = secrets.get("SHEET_NAME", None)
    if SHEET_NAME:
        sheet = client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)
    else:
        sheet = client.open_by_url(SHEET_URL).get_worksheet(0)
except Exception as e:
    st.error(f"❌ Fel vid åtkomst till Google Sheet: {e}")
    st.stop()

# Läs in data
@st.cache_data(ttl=60)
def load_data():
    try:
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"❌ Kunde inte läsa data från arket: {e}")
        return pd.DataFrame()

df = load_data()

# ====== Kolumnstruktur ======
ALL_COLUMNS = [
    "Ticker", "Bolagsnamn", "Antal", "Aktuell kurs (USD)", "Valutakurs (USD/SEK)",
    "Innehavsvärde (SEK)", "Omsättning idag (MUSD)", "Omsättning nästa år (MUSD)",
    "Omsättning om 2 år (MUSD)", "Omsättning om 3 år (MUSD)",
    "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S Snitt",
    "Utestående aktier (miljoner)", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027",
    "Kommentar", "Datum"
]

# Säkerställ att alla kolumner finns i arket
def ensure_columns_exist(df):
    missing_cols = [col for col in ALL_COLUMNS if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            df[col] = ""
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
    return df

df = ensure_columns_exist(df)

# ====== Hjälpfunktioner för beräkningar ======

def beräkna_ps_snitt(row):
    ps_värden = [row.get(f"P/S Q{i}", 0) for i in range(1, 5)]
    giltiga = [v for v in ps_värden if isinstance(v, (int, float)) and v > 0]
    return round(sum(giltiga) / len(giltiga), 2) if giltiga else ""

def beräkna_riktkurs(omsättning, ps_snitt, aktier):
    try:
        if omsättning and ps_snitt and aktier:
            return round((omsättning * ps_snitt) / aktier, 2)
    except:
        return ""
    return ""

def beräkna_innehavsvärde(row):
    try:
        return round(float(row["Antal"]) * float(row["Aktuell kurs (USD)"]) * float(row["Valutakurs (USD/SEK)"]), 2)
    except:
        return ""

def uppdatera_beräkningar(df):
    for i, row in df.iterrows():
        try:
            df.at[i, "P/S Snitt"] = beräkna_ps_snitt(row)
            df.at[i, "Riktkurs idag"] = beräkna_riktkurs(
                float(row.get("Omsättning idag (MUSD)", 0)),
                float(df.at[i, "P/S Snitt"]),
                float(row.get("Utestående aktier (miljoner)", 0))
            )
            df.at[i, "Riktkurs 2026"] = beräkna_riktkurs(
                float(row.get("Omsättning om 2 år (MUSD)", 0)),
                float(df.at[i, "P/S Snitt"]),
                float(row.get("Utestående aktier (miljoner)", 0))
            )
            df.at[i, "Riktkurs 2027"] = beräkna_riktkurs(
                float(row.get("Omsättning om 3 år (MUSD)", 0)),
                float(df.at[i, "P/S Snitt"]),
                float(row.get("Utestående aktier (miljoner)", 0))
            )
            df.at[i, "Innehavsvärde (SEK)"] = beräkna_innehavsvärde(row)
        except Exception as e:
            print(f"Fel vid beräkning för rad {i}: {e}")
    return df

# ====== Funktion för att hämta aktuell kurs via yfinance ======

def hamta_aktuell_kurs(ticker):
    try:
        aktie = yf.Ticker(ticker)
        data = aktie.history(period="1d")
        pris = data["Close"].iloc[-1]
        return round(pris, 2)
    except:
        return None

# ====== Funktion för att uppdatera kurser ======

def uppdatera_kurser(df):
    for i, row in df.iterrows():
        ticker = row.get("Ticker", "")
        if not ticker or not isinstance(ticker, str):
            continue

        aktuell_kurs = hamta_aktuell_kurs(ticker)
        if aktuell_kurs:
            df.at[i, "Aktuell kurs (USD)"] = aktuell_kurs
        else:
            st.warning(f"Kunde inte hämta kurs för {row.get('Namn', '')} ({ticker}). Ange den manuellt.")

    return df

# ====== Funktion för att spara DataFrame till Google Sheet ======

def spara_data(df):
    sheet.clear()
    sheet.append_row(df.columns.tolist())
    for _, row in df.iterrows():
        sheet.append_row(row.fillna("").tolist())

# ====== Funktion för att lägga till nytt bolag ======

def lagg_till_bolag(df):
    st.subheader("➕ Lägg till nytt bolag")

    with st.form("nytt_bolag_formulär"):
        namn = st.text_input("Bolagsnamn")
        ticker = st.text_input("Ticker (ex: AAPL, OS)")
        valuta = st.selectbox("Valuta", ["USD", "SEK"])
        aktier = st.number_input("Antal aktier", min_value=0.0, value=0.0)
        ps_nu = st.number_input("P/S just nu", min_value=0.0, value=0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0, value=0.0)
        oms_2025 = st.number_input("Omsättning 2026 (miljoner USD)", min_value=0.0, value=0.0)
        oms_2026 = st.number_input("Omsättning 2027 (miljoner USD)", min_value=0.0, value=0.0)
        oms_2027 = st.number_input("Omsättning 2028 (miljoner USD)", min_value=0.0, value=0.0)

        ps_q1 = st.number_input("P/S Q1", min_value=0.0, value=0.0)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, value=0.0)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, value=0.0)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, value=0.0)

        utest_aktier = st.number_input("Utestående aktier (miljoner)", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Spara bolag")

    if submitted:
        ny_rad = {
            "Namn": namn,
            "Ticker": ticker,
            "Valuta": valuta,
            "Antal aktier": aktier,
            "P/S just nu": ps_nu,
            "Omsättning idag": oms_idag,
            "Omsättning 2026": oms_2025,
            "Omsättning 2027": oms_2026,
            "Omsättning 2028": oms_2027,
            "P/S Q1": ps_q1,
            "P/S Q2": ps_q2,
            "P/S Q3": ps_q3,
            "P/S Q4": ps_q4,
            "Utestående aktier": utest_aktier
        }

        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        df = berakna_varderingar(df)
        spara_data(df)
        st.success("Bolaget har sparats.")
        st.experimental_rerun()

    return df

# ====== Funktion för att visa investeringsrekommendationer ======

def visa_investeringsförslag(df):
    st.subheader("📈 Investeringsförslag")

    tillgängligt_kapital = st.number_input("Tillgängligt kapital (SEK)", min_value=0, value=1000)

    if df.empty or "Riktkurs nu (SEK)" not in df.columns:
        st.info("Ingen data tillgänglig eller felaktigt format.")
        return

    df_sorterad = df.copy()
    df_sorterad["Skillnad (%)"] = round((df_sorterad["Riktkurs nu (SEK)"] - df_sorterad["Aktuell kurs (SEK)"]) / df_sorterad["Aktuell kurs (SEK)"] * 100, 2)
    df_sorterad = df_sorterad.sort_values(by="Skillnad (%)", ascending=False)

    köp_förslag = []

    for _, rad in df_sorterad.iterrows():
        if rad["Aktuell kurs (SEK)"] <= tillgängligt_kapital:
            antal = int(tillgängligt_kapital // rad["Aktuell kurs (SEK)"])
            if antal > 0:
                köp_förslag.append({
                    "Namn": rad["Namn"],
                    "Ticker": rad["Ticker"],
                    "Köp antal": antal,
                    "Pris per aktie (SEK)": round(rad["Aktuell kurs (SEK)"], 2),
                    "Totalt (SEK)": round(antal * rad["Aktuell kurs (SEK)"], 2),
                    "Riktkurs nu (SEK)": round(rad["Riktkurs nu (SEK)"], 2),
                    "Skillnad (%)": rad["Skillnad (%)"]
                })
                tillgängligt_kapital -= antal * rad["Aktuell kurs (SEK)"]

    if köp_förslag:
        st.success("Följande investeringar föreslås:")
        st.dataframe(pd.DataFrame(köp_förslag))
    else:
        st.warning("Inget bolag kunde köpas för det tillgängliga kapitalet.")

# 🟩 DEL 7 – Fortsättning med hamta_data och main()

def hamta_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["SHEET_URL"]).worksheet(st.secrets["SHEET_NAME"])
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df

# ----------------------------------------------
# 🔁 Main-funktion
# ----------------------------------------------

def main():
    st.title("📊 Aktieanalys & Portföljförslag")
    
    # 🟡 Hämta data
    df = hamta_data()

    # 🟡 Säkerställ kolumner och rätt format
    df = ensure_all_columns_exist(df)
    df = konvertera_till_ratt_typ(df)

    # 🟡 Uppdatera aktuell kurs
    df, manuella_kurser = uppdatera_aktuell_kurs(df)

    # 🟡 Spara tillbaka kurser som angavs manuellt
    if manuella_kurser:
        st.warning("Kurser kunde inte hämtas för följande bolag. Ange dem manuellt:")
        for ticker, namn in manuella_kurser:
            ny_kurs = st.number_input(f"{namn} ({ticker}) – Ange kurs manuellt", min_value=0.01, value=0.0, step=0.01)
            df.loc[df["Ticker"] == ticker, "Aktuell kurs"] = ny_kurs

    # 🟡 Gör beräkningar
    df = berakna_varderingar(df)

    # 🟡 Visa formulär
    visa_lagg_till_bolag_form(df)

    # 🟡 Visa datatabell
    visa_bolagsdata(df)

    # 🟡 Visa portföljförslag
    visa_portfoljforslag(df)

if __name__ == "__main__":
    main()
