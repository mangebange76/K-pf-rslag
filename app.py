import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# --- Google Sheets ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# --- Hårdkodade standard-valutakurser till SEK (kan justeras i sidopanelen) ---
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0
}

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    # numeriska fält
    kolumner_num = [
        "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier",
        "Årlig utdelning",
        "Aktuell kurs",
        "CAGR 5 år (%)",
        "P/S-snitt"
    ]
    for kol in kolumner_num:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)
    # textfält
    for kol in ["Ticker", "Bolagsnamn", "Valuta"]:
        if kol in df.columns:
            df[kol] = df[kol].astype(str)
    return df

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # Exakt kolumnuppsättning enligt din lista
    önskade = [
        "Ticker",
        "Bolagsnamn",
        "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag",
        "Omsättning nästa år",
        "Omsättning om 2 år",
        "Omsättning om 3 år",
        "Riktkurs idag",
        "Riktkurs om 1 år",
        "Riktkurs om 2 år",
        "Riktkurs om 3 år",
        "Antal aktier",
        "Valuta",
        "Årlig utdelning",
        "Aktuell kurs",
        "CAGR 5 år (%)",
        "P/S-snitt"
    ]
    for kol in önskade:
        if kol not in df.columns:
            if kol in ["Ticker", "Bolagsnamn", "Valuta"]:
                df[kol] = ""
            else:
                df[kol] = 0.0
    # Ta bort eventuella överskottskolumner (håller arket rent)
    df = df[önskade]
    return df

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    # P/S-snitt baserat på Q1–Q4 > 0
    for i, rad in df.iterrows():
        ps = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps = [x for x in ps if x and x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # Riktkurser: pris = (omsättning * P/S-snitt) / utestående aktier
        uts = float(rad.get("Utestående aktier", 0.0))
        if uts > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))      * ps_snitt) / uts, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0))  * ps_snitt) / uts, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(rad.get("Omsättning om 2 år", 0.0))   * ps_snitt) / uts, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(rad.get("Omsättning om 3 år", 0.0))   * ps_snitt) / uts, 2)
        else:
            # Låt ev tidigare värden stå kvar? Vi nollar försiktigt om indata saknas
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0
    return df

def hamta_kurs_och_valuta(ticker: str):
    """Hämtar aktuell kurs och valuta från Yahoo Finance. Returnerar (pris, valuta) eller (None, 'USD')."""
    try:
        info = yf.Ticker(ticker).info
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        # Bolagsnamn om vi vill sätta det i formuläret (frivilligt)
        namn = info.get("longName") or info.get("shortName") or ""
        return pris, valuta, namn
    except Exception:
        return None, "USD", ""

def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Rullista "Bolagsnamn (Ticker)"
    namn_map = {
        f"{str(rad.get('Bolagsnamn','')).strip()} ({str(rad.get('Ticker','')).strip()})": str(rad.get('Ticker','')).strip()
        for _, rad in df.iterrows() if str(rad.get('Ticker','')).strip() != ""
    }
    valt_visningsnamn = st.selectbox(
        "Välj bolag att uppdatera (eller lämna tomt för nytt)",
        [""] + sorted(namn_map.keys())
    )

    if valt_visningsnamn:
        ticker_vald = namn_map[valt_visningsnamn]
        bef = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        bef = pd.Series(dtype=object)

    with st.form("form_lagg_till_uppdatera", clear_on_submit=False):
        st.markdown("### Grunddata")
        ticker = st.text_input("Ticker (börskod)", value=bef.get("Ticker", "") if not bef.empty else "").upper()
        namn = st.text_input("Bolagsnamn (hämtas om möjligt)", value=bef.get("Bolagsnamn", "") if not bef.empty else "")
        kurs = st.number_input("Aktuell kurs (i aktiens valuta)", value=float(bef.get("Aktuell kurs", 0.0)) if not bef.empty else 0.0)
        uts = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier", 0.0)) if not bef.empty else 0.0)
        antal = st.number_input("Antal aktier du äger (st)", value=float(bef.get("Antal aktier", 0.0)) if not bef.empty else 0.0)
        valuta = st.selectbox(
            "Valuta (aktiekursens valuta, hämtas om möjligt)",
            ["USD", "NOK", "CAD", "SEK", "EUR"],
            index=(["USD","NOK","CAD","SEK","EUR"].index(bef.get("Valuta","USD")) if not bef.empty and bef.get("Valuta","USD") in ["USD","NOK","CAD","SEK","EUR"] else 0)
        )
        utd = st.number_input("Årlig utdelning per aktie (i aktiens valuta)", value=float(bef.get("Årlig utdelning", 0.0)) if not bef.empty else 0.0)

        st.markdown("### P/S (pris/omsättning)")
        ps_idag = st.number_input("P/S (idag)", value=float(bef.get("P/S", 0.0)) if not bef.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0.0)) if not bef.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0.0)) if not bef.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0.0)) if not bef.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0.0)) if not bef.empty else 0.0)

        st.markdown("### Omsättning (i aktiens valuta, miljoner)")
        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag", 0.0)) if not bef.empty else 0.0)
        oms_1 = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år", 0.0)) if not bef.empty else 0.0)
        oms_2 = st.number_input("Omsättning om 2 år (miljoner)", value=float(bef.get("Omsättning om 2 år", 0.0)) if not bef.empty else 0.0)
        oms_3 = st.number_input("Omsättning om 3 år (miljoner)", value=float(bef.get("Omsättning om 3 år", 0.0)) if not bef.empty else 0.0)

        hamta_yahoo_vid_spar = st.checkbox("Hämta fält från Yahoo när jag sparar (valfritt)", value=False)
        sparaknapp = st.form_submit_button("💾 Spara")

    if not sparaknapp:
        return df

    if not ticker:
        st.error("Ange minst en Ticker.")
        return df

    ny_rad = {
        "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs, "Utestående aktier": uts,
        "Antal aktier": antal, "Valuta": valuta, "Årlig utdelning": utd,
        "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
        "Omsättning idag": oms_idag, "Omsättning nästa år": oms_1,
        "Omsättning om 2 år": oms_2, "Omsättning om 3 år": oms_3
    }

    if hamta_yahoo_vid_spar:
        try:
            with st.spinner("Hämtar från Yahoo…"):
                pris, valuta_y, namn_y = hamta_kurs_och_valuta(ticker)
                if pris is not None:
                    ny_rad["Aktuell kurs"] = round(float(pris), 2)
                if valuta_y:
                    ny_rad["Valuta"] = valuta_y
                if not namn and namn_y:
                    ny_rad["Bolagsnamn"] = namn_y
            st.success("Yahoo-data hämtad.")
        except Exception:
            st.warning("Kunde inte hämta från Yahoo just nu. Sparar dina manuella ändringar ändå.")

    if ticker in df["Ticker"].values:
        for k, v in ny_rad.items():
            df.loc[df["Ticker"] == ticker, k] = v
        st.success(f"{ticker} uppdaterat.")
    else:
        df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
        st.success(f"{ticker} tillagt.")

    # Spara direkt till Google Sheet
    try:
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.info("✅ Ändringar sparade till Google Sheet.")
    except Exception as e:
        st.error("Kunde inte spara till Google Sheet just nu.")
        st.exception(e)

    return df

def visa_investeringsforslag(df: pd.DataFrame, valutakurser: dict):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    # SEK används endast för att räkna antal & portföljandel
    df["Växelkurs"] = df["Valuta"].map(valutakurser).fillna(1.0)
    df_port = df[df["Antal aktier"] > 0].copy()
    df_port["Värde (SEK)"] = df_port["Antal aktier"] * df_port["Aktuell kurs"] * df_port["Växelkurs"]
    portfoljvarde = df_port["Värde (SEK)"].sum()

    bas = df_port if filterval == "Endast portföljen" else df
    # Filtrera där riktkurs > aktuell
    df_forslag = bas[bas[riktkurs_val] > bas["Aktuell kurs"]].copy()

    df_forslag["Uppsida (%)"] = ((df_forslag[riktkurs_val] - df_forslag["Aktuell kurs"]) / df_forslag["Aktuell kurs"]) * 100
    df_forslag = df_forslag.sort_values(by="Uppsida (%)", ascending=False).reset_index(drop=True)

    if df_forslag.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    index = st.session_state.forslags_index
    if index >= len(df_forslag):
        st.info("Inga fler förslag att visa.")
        return

    rad = df_forslag.iloc[index]
    kurs_sek = rad["Aktuell kurs"] * rad["Växelkurs"]
    antal = int(kapital_sek // kurs_sek) if kurs_sek > 0 else 0
    investering_sek = antal * kurs_sek

    nuvarande_innehav = df_port[df_port["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
    ny_total = nuvarande_innehav + investering_sek
    nuvarande_andel = round((nuvarande_innehav / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0
    ny_andel = round((ny_total / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

    # Lista alla riktkurser, fetmarkera den valda
    rikt_labels = ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"]
    rader = []
    for lab in rikt_labels:
        valtxt = f"**{lab}: {round(rad.get(lab, 0.0), 2)} {rad['Valuta']}**" if lab == riktkurs_val else f"{lab}: {round(rad.get(lab, 0.0), 2)} {rad['Valuta']}"
        rader.append(f"- {valtxt}")

    st.markdown(f"""
### 💰 Förslag {index+1} av {len(df_forslag)}
**{rad.get('Bolagsnamn','')} ({rad.get('Ticker','')})**

- Aktuell kurs: {round(rad.get('Aktuell kurs',0.0), 2)} {rad.get('Valuta','')}
{chr(10).join(rader)}
- **Uppsida mot vald riktkurs:** {round(rad.get('Uppsida (%)',0.0), 2)}%
- Antal att köpa (för {int(kapital_sek)} SEK): {antal} st
- Beräknad investering: {round(investering_sek, 2)} SEK
- Nuvarande andel i portföljen: {nuvarande_andel}%
- Andel efter köp: {ny_andel}%
    """)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Föregående förslag", use_container_width=True):
            st.session_state.forslags_index = max(0, index - 1)
    with c2:
        if st.button("➡️ Nästa förslag", use_container_width=True):
            st.session_state.forslags_index = min(len(df_forslag) - 1, index + 1)


def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📦 Min portfölj")

    dfp = df[df["Antal aktier"] > 0].copy()
    if dfp.empty:
        st.info("Du äger inga aktier.")
        return

    dfp["Växelkurs"] = dfp["Valuta"].map(valutakurser).fillna(1.0)
    dfp["Värde (SEK)"] = dfp["Antal aktier"] * dfp["Aktuell kurs"] * dfp["Växelkurs"]
    dfp["Andel (%)"] = round(dfp["Värde (SEK)"] / dfp["Värde (SEK)"].sum() * 100, 2)
    dfp["Total årlig utdelning (SEK)"] = dfp["Antal aktier"] * dfp["Årlig utdelning"] * dfp["Växelkurs"]

    total_utdelning = dfp["Total årlig utdelning (SEK)"].sum()
    total_varde = dfp["Värde (SEK)"].sum()

    st.markdown(f"**Portföljvärde (SEK):** {round(total_varde, 2)}")
    st.markdown(f"**Förväntad årlig utdelning (SEK):** {round(total_utdelning, 2)}")
    st.markdown(f"**Utdelning per månad (SEK):** {round(total_utdelning / 12, 2)}")

    visnings_df = dfp.rename(columns={
        "Antal aktier": "Antal aktier (st)",
        "Aktuell kurs": "Aktuell kurs (i aktiens valuta)",
        "Total årlig utdelning (SEK)": "Total årlig utdelning (SEK)"
    })
    st.dataframe(
        visnings_df[[
            "Ticker", "Bolagsnamn", "Antal aktier (st)",
            "Aktuell kurs (i aktiens valuta)", "Valuta",
            "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning (SEK)"
        ]],
        use_container_width=True
    )

def analysvy(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📈 Analys")

    # Valfri filtrering (endast visning)
    namn_map = {
        f"{str(rad.get('Bolagsnamn','')).strip()} ({str(rad.get('Ticker','')).strip()})": str(rad.get('Ticker','')).strip()
        for _, rad in df.iterrows() if str(rad.get('Ticker','')).strip() != ""
    }
    valt = st.selectbox("Filtrera bolag (valfritt)", ["— visa alla —"] + sorted(namn_map.keys()))
    visa = df.copy()
    if valt != "— visa alla —":
        visa = df[df["Ticker"] == namn_map[valt]].copy()

    # Etiketter (display only)
    etiketter = {
        "Ticker": "Ticker (kortnamn)",
        "Bolagsnamn": "Bolagsnamn",
        "Aktuell kurs": "Aktuell kurs (i aktiens valuta)",
        "Valuta": "Valuta",
        "Årlig utdelning": "Årlig utdelning per aktie",
        "CAGR 5 år (%)": "Omsättningstillväxt 5 år (%)",
        "P/S": "P/S (idag)",
        "P/S Q1": "P/S Q1",
        "P/S Q2": "P/S Q2",
        "P/S Q3": "P/S Q3",
        "P/S Q4": "P/S Q4",
        "P/S-snitt": "P/S-snitt",
        "Omsättning idag": "Omsättning idag (M)",
        "Omsättning nästa år": "Omsättning nästa år (M)",
        "Omsättning om 2 år": "Omsättning om 2 år (M)",
        "Omsättning om 3 år": "Omsättning om 3 år (M)",
        "Riktkurs idag": "Riktkurs idag",
        "Riktkurs om 1 år": "Riktkurs om 1 år",
        "Riktkurs om 2 år": "Riktkurs om 2 år",
        "Riktkurs om 3 år": "Riktkurs om 3 år",
        "Utestående aktier": "Utestående aktier (M)",
        "Antal aktier": "Antal aktier (st)",
    }
    visnings_df = visa.rename(columns={k: v for k, v in etiketter.items() if k in visa.columns})
    st.dataframe(visnings_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🔄 Uppdatera aktuella kurser (Yahoo Finance)")
    delay = st.number_input("Fördröjning mellan anrop (sek)", value=1.0, min_value=0.0, step=0.5)

    if st.button("Uppdatera alla nu"):
        misslyckade = []
        uppdaterade = 0
        total = len(df.index)
        status = st.empty()
        bar = st.progress(0)

        with st.spinner("Hämtar kurser…"):
            for i, row in df.iterrows():
                ticker = str(row.get("Ticker", "")).strip().upper()
                status.text(f"({i+1}/{total}) {ticker}…")

                if not ticker:
                    misslyckade.append("(tom ticker)")
                    bar.progress((i + 1) / max(total, 1))
                    continue

                try:
                    pris, valuta, namn = hamta_kurs_och_valuta(ticker)
                    if pris is None:
                        misslyckade.append(ticker)
                    else:
                        df.at[i, "Aktuell kurs"] = round(float(pris), 2)
                        if valuta:
                            df.at[i, "Valuta"] = valuta
                        if namn and not str(df.at[i, "Bolagsnamn"]).strip():
                            df.at[i, "Bolagsnamn"] = namn
                        uppdaterade += 1
                except Exception:
                    misslyckade.append(ticker)

                bar.progress((i + 1) / max(total, 1))
                time.sleep(max(0.0, float(delay)))  # ingen busy-loop

        # Kör beräkningar och spara
        try:
            df = uppdatera_berakningar(df)
            spara_data(df)
            status.text("✅ Uppdatering klar.")
            st.success(f"{uppdaterade} av {total} tickers uppdaterade.")
        except Exception as e:
            st.warning("Uppdaterat i minnet, men kunde inte skriva till Google Sheet.")
            st.exception(e)

        if misslyckade:
            st.info("Kunde inte hämta för:\n" + ", ".join(sorted(set(misslyckade))))


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Sidopanel: manuella valutakurser (till SEK), påverkar endast SEK-beräkningar
    st.sidebar.header("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(STANDARD_VALUTAKURSER["USD"]), step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(STANDARD_VALUTAKURSER["NOK"]), step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(STANDARD_VALUTAKURSER["CAD"]), step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(STANDARD_VALUTAKURSER["EUR"]), step=0.01),
    }

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        # Visning + manuell massuppdatering
        analysvy(df, valutakurser)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        # OBS: spara_data sker redan inne i funktionen

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurser)


if __name__ == "__main__":
    main()
