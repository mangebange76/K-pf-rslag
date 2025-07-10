import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# === Autentisering från Streamlit Cloud secrets ===
SHEET_URL = st.secrets["SHEET_URL"]
credentials_dict = st.secrets["GOOGLE_CREDENTIALS"]

credentials = Credentials.from_service_account_info(
    credentials_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)

client = gspread.authorize(credentials)
spreadsheet = client.open_by_url(SHEET_URL)
worksheet = spreadsheet.sheet1

# === Kolumner som används i appen ===
ALL_COLUMNS = [
    "Ticker", "Bolagsnamn", "Antal aktier", "Innehav i kr", "Tillgängligt belopp (kr)",
    "Aktuell kurs", "Valutakurs", "P/S nu", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4", "P/S snitt",
    "Utestående aktier",
    "Omsättning idag",  # 👈 Viktigt fält för riktkurs idag
    "Omsättning Y1", "Omsättning Y2", "Omsättning Y3",
    "Riktkurs idag", "Riktkurs Y1", "Riktkurs Y2", "Riktkurs Y3"
]

# === Ladda data från Google Sheet ===
def load_data():
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    df = ensure_columns_exist(df)
    return df

# === Spara data till Google Sheet ===
def save_data(df):
    worksheet.clear()
    worksheet.append_row(ALL_COLUMNS)
    for _, row in df.iterrows():
        worksheet.append_row([row.get(col, "") for col in ALL_COLUMNS])

# === Säkerställ att alla kolumner finns i df ===
def ensure_columns_exist(df):
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[ALL_COLUMNS]

import yfinance as yf
import requests

def hamta_valutakurs():
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=SEK")
        return round(r.json()["rates"]["SEK"], 2)
    except:
        return 10.0  # fallback

def hamta_aktuell_kurs(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        kurs = info.get("regularMarketPrice", None)

        if kurs is None:
            kurs_hist = yf_ticker.history(period="1d")["Close"]
            kurs = kurs_hist.iloc[-1] if not kurs_hist.empty else None

        return round(kurs, 2) if kurs else 0
    except:
        return 0

# === Lägg till eller redigera bolag ===
def lagg_till_bolag(df):
    st.subheader("➕ Lägg till eller redigera bolag")

    with st.form("ny_bolagsform"):
        ticker = st.text_input("Ticker (t.ex. NVDA)").upper()

        # Hämta automatiskt kurs & valutakurs
        auto_kurs = hamta_aktuell_kurs(ticker)
        valutakurs = hamta_valutakurs()

        st.markdown(f"💵 Automatisk USD-kurs: **{auto_kurs} USD**")
        st.markdown(f"💱 Valutakurs (USD/SEK): **{valutakurs}**")

        aktuell_kurs = st.number_input("Bekräfta eller justera aktuell kurs (USD)", min_value=0.0, value=auto_kurs, format="%.2f")

        antal = st.number_input("Antal aktier", min_value=0, value=0)
        tillgangligt_belopp = st.number_input("Tillgängligt belopp (kr)", min_value=0, value=0)

        ps_nu = st.number_input("Nuvarande P/S", min_value=0.0, value=0.0)
        ps_q1 = st.number_input("P/S Q1", min_value=0.0, value=0.0)
        ps_q2 = st.number_input("P/S Q2", min_value=0.0, value=0.0)
        ps_q3 = st.number_input("P/S Q3", min_value=0.0, value=0.0)
        ps_q4 = st.number_input("P/S Q4", min_value=0.0, value=0.0)

        aktier = st.number_input("Utestående aktier (miljoner)", min_value=0.0, value=0.0, format="%.2f")

        # 📌 Omsättning idag först
        oms0 = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0, value=0.0)
        oms1 = st.number_input("Förväntad omsättning nästa år (miljoner USD)", min_value=0.0, value=0.0)
        oms2 = st.number_input("Förväntad omsättning om 2 år (miljoner USD)", min_value=0.0, value=0.0)
        oms3 = st.number_input("Förväntad omsättning om 3 år (miljoner USD)", min_value=0.0, value=0.0)

        submitted = st.form_submit_button("Spara bolag")

    if submitted and ticker:
        try:
            namn = yf.Ticker(ticker).info.get("shortName", ticker)
        except:
            namn = ticker

        # Lägg till eller uppdatera i df
        if ticker in df["Ticker"].values:
            idx = df[df["Ticker"] == ticker].index[0]
        else:
            idx = len(df)
            df.loc[idx] = [None] * len(ALL_COLUMNS)

        df.at[idx, "Ticker"] = ticker
        df.at[idx, "Bolagsnamn"] = namn
        df.at[idx, "Antal aktier"] = antal
        df.at[idx, "Tillgängligt belopp (kr)"] = tillgangligt_belopp
        df.at[idx, "Valutakurs"] = valutakurs
        df.at[idx, "Aktuell kurs"] = aktuell_kurs
        df.at[idx, "P/S nu"] = ps_nu
        df.at[idx, "P/S Q1"] = ps_q1
        df.at[idx, "P/S Q2"] = ps_q2
        df.at[idx, "P/S Q3"] = ps_q3
        df.at[idx, "P/S Q4"] = ps_q4
        df.at[idx, "Utestående aktier"] = aktier
        df.at[idx, "Omsättning idag"] = oms0
        df.at[idx, "Omsättning Y1"] = oms1
        df.at[idx, "Omsättning Y2"] = oms2
        df.at[idx, "Omsättning Y3"] = oms3

        st.success(f"{ticker} sparat.")
        return df

    return df

# === Beräkna P/S-snitt, riktkurser och innehavsvärden ===
def berakna_allt(df):
    for i, row in df.iterrows():
        # P/S-snitt baserat på ifyllda kvartal
        try:
            ps_q = [row["P/S Q1"], row["P/S Q2"], row["P/S Q3"], row["P/S Q4"]]
            ps_q = [x for x in ps_q if x > 0]
            ps_snitt = round(sum(ps_q) / len(ps_q), 2) if ps_q else 0
        except:
            ps_snitt = 0

        df.at[i, "P/S snitt"] = ps_snitt

        # Riktkurser (omsättning i miljoner USD, aktier i miljoner → båda × 1 000 000)
        try:
            aktier_milj = row["Utestående aktier"] * 1_000_000
            if aktier_milj > 0:
                df.at[i, "Riktkurs idag"] = round((row["Omsättning idag"] * 1_000_000 * ps_snitt) / aktier_milj, 2)
                df.at[i, "Riktkurs Y1"] = round((row["Omsättning Y1"] * 1_000_000 * ps_snitt) / aktier_milj, 2)
                df.at[i, "Riktkurs Y2"] = round((row["Omsättning Y2"] * 1_000_000 * ps_snitt) / aktier_milj, 2)
                df.at[i, "Riktkurs Y3"] = round((row["Omsättning Y3"] * 1_000_000 * ps_snitt) / aktier_milj, 2)
            else:
                df.at[i, "Riktkurs idag"] = 0
                df.at[i, "Riktkurs Y1"] = 0
                df.at[i, "Riktkurs Y2"] = 0
                df.at[i, "Riktkurs Y3"] = 0
        except:
            df.at[i, "Riktkurs idag"] = 0
            df.at[i, "Riktkurs Y1"] = 0
            df.at[i, "Riktkurs Y2"] = 0
            df.at[i, "Riktkurs Y3"] = 0

        # Innehav i kronor
        try:
            df.at[i, "Innehav i kr"] = round(row["Antal aktier"] * row["Aktuell kurs"] * row["Valutakurs"], 2)
        except:
            df.at[i, "Innehav i kr"] = 0

    return df

# === Visa portföljen i tabellform ===
def visa_portfolj(df):
    st.subheader("📊 Portföljöversikt")

    if df.empty:
        st.info("Inga bolag har lagts till ännu.")
        return df

    visa_df = df[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Innehav i kr", "Tillgängligt belopp (kr)",
        "Aktuell kurs", "Valutakurs", "P/S snitt",
        "Omsättning idag", "Omsättning Y1", "Omsättning Y2", "Omsättning Y3",
        "Riktkurs idag", "Riktkurs Y1", "Riktkurs Y2", "Riktkurs Y3"
    ]].copy()

    visa_df = visa_df.sort_values("Riktkurs Y1", ascending=False)
    st.dataframe(visa_df, use_container_width=True)

    return df

# === Investeringsråd baserat på riktkurs och tillgängligt belopp ===
def investeringsrad(df):
    st.subheader("🧠 Investeringsråd")

    for _, row in df.iterrows():
        ticker = row["Ticker"]
        namn = row["Bolagsnamn"]
        aktuell_kurs = row["Aktuell kurs"]
        riktkurs = row["Riktkurs idag"]
        tillgängligt = row["Tillgängligt belopp (kr)"]
        innehav_kr = row["Innehav i kr"]
        valuta = row["Valutakurs"]

        # Undvik division med 0
        if aktuell_kurs == 0 or valuta == 0:
            continue

        pris_i_sek = aktuell_kurs * valuta

        # Köpråd
        if riktkurs > aktuell_kurs and riktkurs > 0:
            köptext = f"💡 **{namn} ({ticker})** är undervärderad – riktkurs {riktkurs} USD > aktuell kurs {aktuell_kurs} USD"

            # Kan vi köpa minst 1 aktie?
            if pris_i_sek <= tillgängligt:
                antal_att_kopa = int(tillgängligt // pris_i_sek)
                köptext += f"\n✅ Rekommenderat: **Köp {antal_att_kopa} st** ({pris_i_sek:.2f} kr/st)"
            else:
                köptext += f"\n⚠️ **För dyrt** just nu (kostar {pris_i_sek:.2f} kr) – fundera på att skjuta till mer kapital, spara eller omfördela."

            st.markdown(köptext)

        # Överviktad position
        if innehav_kr > 3 * tillgängligt and innehav_kr > 0:
            st.warning(
                f"📉 **{namn} ({ticker})** utgör en stor del av portföljen ({innehav_kr:.0f} kr). Överväg att ta hem vinst eller minska position."
            )

        # Undervärde/neutral
        if riktkurs <= aktuell_kurs and riktkurs > 0:
            st.info(
                f"📎 **{namn} ({ticker})** har redan nått eller passerat riktkurs ({riktkurs} USD ≤ {aktuell_kurs} USD)."
            )

# === Uppdatera alla beräkningar ===
def uppdatera(df):
    st.subheader("🔄 Uppdatera beräkningar")

    if st.button("Uppdatera allt"):
        df = berakna_allt(df)
        st.success("Beräkningar uppdaterade.")
    return df

# === Huvudfunktion för appen ===
def main():
    st.set_page_config(page_title="Aktieanalys App", layout="wide")
    st.title("📈 Aktieanalys & Köprekommendationer")

    df = load_data()

    menyval = st.sidebar.radio("Meny", ["Lägg till bolag", "Portfölj", "Investeringsråd", "Uppdatera", "🔧 Exportera/Importera"])

    if menyval == "Lägg till bolag":
        df = lagg_till_bolag(df)
        save_data(df)

    elif menyval == "Portfölj":
        df = berakna_allt(df)
        visa_portfolj(df)

    elif menyval == "Investeringsråd":
        df = berakna_allt(df)
        investeringsrad(df)

    elif menyval == "Uppdatera":
        df = uppdatera(df)
        save_data(df)

    elif menyval == "🔧 Exportera/Importera":
        st.subheader("💾 Export / Import")
        st.markdown("Data sparas automatiskt till Google Sheets.")
        st.markdown("Om du vill ladda upp en backupfil, implementera import senare.")

    save_data(df)

# === Lägg till bolag manuellt ===
def lagg_till_bolag(df):
    st.subheader("➕ Lägg till nytt bolag")

    with st.form("lägg_till_form"):
        kol1, kol2 = st.columns(2)

        with kol1:
            ticker = st.text_input("Ticker").upper()
            bolagsnamn = st.text_input("Bolagsnamn")
            antal = st.number_input("Antal aktier", min_value=0, step=1)
            belopp = st.number_input("Tillgängligt belopp (kr)", min_value=0.0, step=100.0)
            kurs = st.number_input("Aktuell kurs (USD)", min_value=0.0)
            valuta = st.number_input("Valutakurs USD/SEK", min_value=0.0, value=11.0)
            ps_nu = st.number_input("P/S nu", min_value=0.0)
            ps_q1 = st.number_input("P/S Q1", min_value=0.0)
            ps_q2 = st.number_input("P/S Q2", min_value=0.0)
            ps_q3 = st.number_input("P/S Q3", min_value=0.0)
            ps_q4 = st.number_input("P/S Q4", min_value=0.0)

        with kol2:
            utest_aktier = st.number_input("Utestående aktier (miljoner)", min_value=0.0)
            oms_idag = st.number_input("Omsättning idag (miljoner USD)", min_value=0.0)
            oms_y1 = st.number_input("Omsättning Y1 (miljoner USD)", min_value=0.0)
            oms_y2 = st.number_input("Omsättning Y2 (miljoner USD)", min_value=0.0)
            oms_y3 = st.number_input("Omsättning Y3 (miljoner USD)", min_value=0.0)

        submitted = st.form_submit_button("Lägg till bolag")

        if submitted:
            ny_rad = {
                "Ticker": ticker,
                "Bolagsnamn": bolagsnamn,
                "Antal aktier": antal,
                "Innehav i kr": 0,
                "Tillgängligt belopp (kr)": belopp,
                "Aktuell kurs": kurs,
                "Valutakurs": valuta,
                "P/S nu": ps_nu,
                "P/S Q1": ps_q1,
                "P/S Q2": ps_q2,
                "P/S Q3": ps_q3,
                "P/S Q4": ps_q4,
                "P/S snitt": 0,
                "Utestående aktier": utest_aktier,
                "Omsättning idag": oms_idag,
                "Omsättning Y1": oms_y1,
                "Omsättning Y2": oms_y2,
                "Omsättning Y3": oms_y3,
                "Riktkurs idag": 0,
                "Riktkurs Y1": 0,
                "Riktkurs Y2": 0,
                "Riktkurs Y3": 0
            }

            df = df.append(ny_rad, ignore_index=True)
            st.success(f"{ticker} har lagts till.")

    return df
