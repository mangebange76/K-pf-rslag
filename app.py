import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# -----------------------------
# GOOGLE SHEETS KONFIGURATION
# -----------------------------
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# -----------------------------
# HJÄLPFUNKTIONER
# -----------------------------
def konvertera_typer(df):
    kolumner = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier", "Årlig utdelning"
    ]
    for kol in kolumner:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    nödvändiga = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Valuta", "Årlig utdelning", "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if kol in ["Ticker", "Bolagsnamn", "Valuta"]:
                df[kol] = ""
            elif "Riktkurs" in kol or "kurs" in kol.lower() or "omsättning" in kol.lower() or "p/s" in kol.lower() or kol == "Årlig utdelning":
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def uppdatera_berakningar(df):
    for i, rad in df.iterrows():
        ps = [rad["P/S Q1"], rad["P/S Q2"], rad["P/S Q3"], rad["P/S Q4"]]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0
        df.at[i, "P/S-snitt"] = ps_snitt

        if rad["Utestående aktier"] > 0:
            df.at[i, "Riktkurs idag"] = round((rad["Omsättning idag"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 1 år"] = round((rad["Omsättning nästa år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 2 år"] = round((rad["Omsättning om 2 år"] * ps_snitt) / rad["Utestående aktier"], 2)
            df.at[i, "Riktkurs om 3 år"] = round((rad["Omsättning om 3 år"] * ps_snitt) / rad["Utestående aktier"], 2)
    return df

def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    df_portfolj = df[df["Antal aktier"] > 0].copy()
    # Konvertera kurs till SEK med valutakurserna
    df_portfolj["Värde (SEK)"] = df_portfolj.apply(
        lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1.0),
        axis=1
    )
    portfoljvarde = df_portfolj["Värde (SEK)"].sum()

    if filterval == "Endast portföljen":
        df_forslag = df_portfolj[df_portfolj[riktkurs_val] > df_portfolj["Aktuell kurs"]].copy()
    else:
        df_forslag = df[df[riktkurs_val] > df["Aktuell kurs"]].copy()

    df_forslag["Potential (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if kapital_sek <= 0:
        st.warning("Kapitalet måste vara större än 0.")
        return

    kapital_usd = kapital_sek / valutakurser.get("USD", 1.0)

    if 'forslags_index' not in st.session_state:
        st.session_state.forslags_index = 0

    if df_forslag.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    index = st.session_state.forslags_index
    if index >= len(df_forslag):
        st.info("Inga fler förslag att visa.")
        return

    rad = df_forslag.iloc[index]
    if rad["Aktuell kurs"] <= 0:
        st.warning("Felaktig aktiekurs – kan inte visa förslag.")
        return

    växelkurs = valutakurser.get(rad["Valuta"], 1.0)
    antal = int(kapital_sek // (rad["Aktuell kurs"] * växelkurs))
    investering_sek = antal * rad["Aktuell kurs"] * växelkurs

    nuvarande_innehav = df_portfolj[df_portfolj["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
    nuvarande_andel = round((nuvarande_innehav / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0
    ny_andel = round(((nuvarande_innehav + investering_sek) / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

    st.markdown(f"""
        ### 💰 Förslag {index+1} av {len(df_forslag)}
        - **Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})
        - **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {rad['Valuta']}
        - **{riktkurs_val}:** {round(rad[riktkurs_val], 2)} {rad['Valuta']}
        - **Potential:** {round(rad['Potential (%)'], 2)}%
        - **Antal att köpa:** {antal} st
        - **Beräknad investering:** {round(investering_sek, 2)} SEK
        - **Nuvarande andel i portföljen:** {nuvarande_andel}%
        - **Andel efter köp:** {ny_andel}%
    """)

    if st.button("➡️ Nästa förslag"):
        st.session_state.forslags_index += 1


def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    df = df[df["Antal aktier"] > 0].copy()
    if df.empty:
        st.info("Du äger inga aktier.")
        return

    df["Värde (SEK)"] = df.apply(
        lambda r: r["Antal aktier"] * r["Aktuell kurs"] * valutakurser.get(r["Valuta"], 1.0),
        axis=1
    )
    df["Andel (%)"] = round(df["Värde (SEK)"] / df["Värde (SEK)"].sum() * 100, 2)
    total_varde = df["Värde (SEK)"].sum()

    total_utdelning_arlig = (df["Årlig utdelning"] * df["Antal aktier"] * df.apply(lambda r: valutakurser.get(r["Valuta"], 1.0), axis=1)).sum()
    total_utdelning_manad = total_utdelning_arlig / 12

    st.markdown(f"**Totalt portföljvärde:** {round(total_varde, 2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utdelning_arlig, 2)} SEK")
    st.markdown(f"**Förväntad månadsutdelning (snitt):** {round(total_utdelning_manad, 2)} SEK")

    st.dataframe(df[[
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
        "Värde (SEK)", "Andel (%)", "Årlig utdelning"
    ]], use_container_width=True)

def analysvy(df):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)

    # Valutakurser i sidopanelen
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=9.50, step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=0.93, step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=11.10, step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=7.00, step=0.01)
    }

    if st.button("🔄 Uppdatera alla från Yahoo"):
        misslyckade = []
        uppdaterade = 0
        total = len(df)
        status = st.empty()

        with st.spinner("Uppdaterar alla bolag..."):
            for i, row in df.iterrows():
                ticker = str(row["Ticker"]).strip().upper()
                fail_fields = []
                status.text(f"🔄 Uppdaterar {i+1}/{total} ({ticker})...")

                try:
                    info = yf.Ticker(ticker).info

                    # Kurs
                    kurs = info.get("regularMarketPrice", None)
                    if kurs is not None:
                        df.at[i, "Aktuell kurs"] = kurs
                    else:
                        fail_fields.append("kurs")

                    # Bolagsnamn
                    namn = info.get("shortName", None)
                    if namn:
                        df.at[i, "Bolagsnamn"] = namn
                    else:
                        fail_fields.append("bolagsnamn")

                    # Valuta
                    valuta = info.get("currency", None)
                    if valuta:
                        df.at[i, "Valuta"] = valuta
                    else:
                        fail_fields.append("valuta")

                    # Utdelning
                    utd = info.get("dividendRate", None)
                    if utd is not None:
                        df.at[i, "Årlig utdelning"] = utd
                    else:
                        fail_fields.append("utdelning")

                    # CAGR för omsättningar
                    try:
                        hist = yf.Ticker(ticker).financials
                        oms_hist = hist.loc["Total Revenue"].dropna().values
                        if len(oms_hist) >= 5:
                            cagr = (oms_hist[0] / oms_hist[-1]) ** (1 / (len(oms_hist) - 1)) - 1
                        else:
                            cagr = 0
                    except Exception:
                        cagr = 0

                    if cagr > 0.5:
                        cagr = 0.5
                    elif cagr < -0.5:
                        cagr = -0.5

                    # Beräkna år 2 & 3
                    if row["Omsättning nästa år"] != 0:
                        df.at[i, "Omsättning om 2 år"] = round(row["Omsättning nästa år"] * (1 + cagr), 2)
                        df.at[i, "Omsättning om 3 år"] = round(df.at[i, "Omsättning om 2 år"] * (1 + cagr), 2)
                    else:
                        fail_fields.append("omsättning nästa år")

                except Exception:
                    fail_fields.append("ALLA FÄLT")

                if fail_fields:
                    misslyckade.append(f"{ticker}: {', '.join(fail_fields)}")
                else:
                    uppdaterade += 1

                time.sleep(1)  # Vänta mellan anrop

        spara_data(df)
        status.text("✅ Uppdatering slutförd.")
        st.success(f"{uppdaterade} av {total} bolag uppdaterades.")

        if misslyckade:
            miss_str = "\n".join(misslyckade)
            st.warning(f"Misslyckade fält:\n{miss_str}")
            st.code(miss_str)

    # Visa hela databasen
    st.dataframe(df, use_container_width=True)
    return valutakurser

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0

    namn_map = {f"{rad['Bolagsnamn']} ({rad['Ticker']})": idx for idx, rad in df.iterrows()}
    bolagslista = list(namn_map.keys())

    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + bolagslista,
                        index=st.session_state.edit_index + 1 if st.session_state.edit_index < len(bolagslista) else 0)

    if valt:
        idx = namn_map[valt]
        befintlig = df.iloc[idx]
    else:
        befintlig = pd.Series(dtype=object)

    with st.form("form"):
        # Fält du anger manuellt
        ticker = st.text_input("Ticker", value=befintlig.get("Ticker", "") if not befintlig.empty else "").upper()
        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0, step=1.0)
        ps_idag = st.number_input("P/S idag", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)
        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara och hämta automatiskt från Yahoo")

    if sparaknapp and ticker:
        # Hämta från Yahoo
        try:
            info = yf.Ticker(ticker).info
            namn = info.get("shortName", befintlig.get("Bolagsnamn", ""))
            kurs = info.get("regularMarketPrice", befintlig.get("Aktuell kurs", 0.0))
            valuta = info.get("currency", befintlig.get("Valuta", "USD"))
        except Exception:
            namn = befintlig.get("Bolagsnamn", "")
            kurs = befintlig.get("Aktuell kurs", 0.0)
            valuta = befintlig.get("Valuta", "USD")

        # Hämta CAGR för omsättning (5 år)
        try:
            hist = yf.Ticker(ticker).financials
            oms_hist = hist.loc["Total Revenue"].dropna().values
            if len(oms_hist) >= 5:
                cagr = (oms_hist[0] / oms_hist[-1]) ** (1 / (len(oms_hist) - 1)) - 1
            else:
                cagr = 0
        except Exception:
            cagr = 0

        # Tak/golv för CAGR
        if cagr > 0.5: 
            cagr = 0.5
        elif cagr < -0.5:
            cagr = -0.5

        # Räkna omsättning år 2 och 3
        oms_ar2 = round(oms_next * (1 + cagr), 2) if oms_next != 0 else 0
        oms_ar3 = round(oms_ar2 * (1 + cagr), 2) if oms_ar2 != 0 else 0

        ny_rad = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Valuta": valuta,
            "Antal aktier": antal_aktier,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
            "Omsättning om 2 år": oms_ar2, "Omsättning om 3 år": oms_ar3
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat med automatisk data från Yahoo.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt med automatisk data från Yahoo.")

        spara_data(df)

        # Bläddra till nästa bolag om möjligt
        if valt:
            nu_idx = bolagslista.index(valt)
            if nu_idx + 1 < len(bolagslista):
                st.session_state.edit_index = nu_idx + 1
                st.experimental_rerun()

    return df

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Hämta data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        valutakurser = analysvy(df)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)

    elif meny == "Investeringsförslag":
        valutakurs_usd = st.sidebar.number_input("Valutakurs USD → SEK", value=9.50, step=0.01)
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurs_usd)

    elif meny == "Portfölj":
        valutakurs_usd = st.sidebar.number_input("Valutakurs USD → SEK", value=9.50, step=0.01)
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurs_usd)


if __name__ == "__main__":
    main()
