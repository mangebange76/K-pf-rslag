import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Google Sheets-autentisering
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    return pd.DataFrame(skapa_koppling().get_all_records())

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.astype(str).values.tolist())

def konvertera_typer(df):
    kol = ["Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
           "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
           "Aktuell kurs","Antal aktier","Årlig utdelning"]
    for k in kol:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    behövs = ["Ticker","Bolagsnamn","Aktuell kurs","Utestående aktier","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
              "P/S-snitt","Riktkurs idag","Riktkurs 2026","Riktkurs 2027","Riktkurs 2028",
              "Antal aktier","Valuta","Årlig utdelning"]
    for k in behövs:
        if k not in df.columns:
            df[k] = "" if k in ["Ticker","Bolagsnamn","Valuta"] else 0.0
    return df

def uppdatera_berakningar(df):
    for i, r in df.iterrows():
        ps = [r["P/S Q1"],r["P/S Q2"],r["P/S Q3"],r["P/S Q4"]]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps),2) if ps else 0.0
        df.at[i,"P/S-snitt"] = ps_snitt
        if r["Utestående aktier"] > 0:
            df.at[i,"Riktkurs idag"] = round((r["Omsättning idag"] * ps_snitt) / r["Utestående aktier"], 2)
            df.at[i,"Riktkurs 2026"] = round((r["Omsättning nästa år"] * ps_snitt) / r["Utestående aktier"], 2)
            df.at[i,"Riktkurs 2027"] = round((r["Omsättning om 2 år"] * ps_snitt) / r["Utestående aktier"], 2)
            df.at[i,"Riktkurs 2028"] = round((r["Omsättning om 3 år"] * ps_snitt) / r["Utestående aktier"], 2)
    return df

def hamta_kurs_och_valuta(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("regularMarketPrice", None), info.get("currency", "USD")
    except Exception:
        return None, None

#######################
# Del 2: Formulär + Investeringsförslag
#######################

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r["Ticker"] for _, r in df.iterrows()}
    valt = st.selectbox("Välj bolag att uppdatera (eller lämna tom)", [""] + sorted(namn_map.keys()))
    if valt:
        bef = df[df["Ticker"] == namn_map[valt]].iloc[0]
    else:
        bef = pd.Series(dtype=object)

    with st.form("form"):
        ticker = st.text_input("Ticker", value=bef.get("Ticker","") if not bef.empty else "").upper()
        namn = st.text_input("Bolagsnamn", value=bef.get("Bolagsnamn","") if not bef.empty else "")
        kurs = st.number_input("Aktuell kurs (USD)", value=float(bef.get("Aktuell kurs",0.0)) if not bef.empty else 0.0)
        aktier = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
        antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)
        valuta = st.selectbox("Valuta", ["USD","NOK","CAD","SEK","EUR"],
                              index=0 if bef.empty else ["USD","NOK","CAD","SEK","EUR"].index(bef.get("Valuta","USD")))
        utd = st.number_input("Årlig utdelning (per aktie, i angiven valuta)",
                              value=float(bef.get("Årlig utdelning",0.0)) if not bef.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        oms0 = st.number_input("Omsättning idag (miljoner USD)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
        oms1 = st.number_input("Omsättning nästa år", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)
        oms2 = st.number_input("Omsättning om 2 år", value=float(bef.get("Omsättning om 2 år",0.0)) if not bef.empty else 0.0)
        oms3 = st.number_input("Omsättning om 3 år", value=float(bef.get("Omsättning om 3 år",0.0)) if not bef.empty else 0.0)
        submit = st.form_submit_button("💾 Spara")

    if submit and ticker:
        ny = {"Ticker":ticker,"Bolagsnamn":namn,"Aktuell kurs":kurs,"Utestående aktier":aktier,
              "Antal aktier":antal,"Valuta":valuta,"Årlig utdelning":utd,"P/S Q1":ps1,"P/S Q2":ps2,
              "P/S Q3":ps3,"P/S Q4":ps4,"Omsättning idag":oms0,"Omsättning nästa år":oms1,
              "Omsättning om 2 år":oms2,"Omsättning om 3 år":oms3}
        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny.keys()] = ny.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

def visa_investeringsforslag(df, valutakurs):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)
    riktkurs_val = st.selectbox("Vilken riktkurs?", ["Riktkurs idag","Riktkurs 2026","Riktkurs 2027","Riktkurs 2028"], index=1)
    filterval = st.radio("Visa förslag för:", ["Alla bolag","Endast portföljen"])

    df2 = df.copy()
    df2["Potential (%)"] = ((df2[riktkurs_val] - df2["Aktuell kurs"]) / df2["Aktuell kurs"]) * 100
    df2 = df2[df2["Potential (%)"] > 0]
    if filterval == "Endast portföljen":
        df2 = df2[df2["Antal aktier"] > 0]
    df2 = df2.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    if df2.empty:
        st.info("Inga bolag matchar kriterierna.")
        return

    if 'forslags_index' not in st.session_state:
        st.session_state.forslags_index = 0
    idx = st.session_state.forslags_index
    if idx >= len(df2):
        st.info("Inga fler förslag.")
        return

    rad = df2.iloc[idx]
    kapital_usd = kapital_sek / valutakurs
    antal = int(kapital_usd // rad["Aktuell kurs"])
    investering = antal * rad["Aktuell kurs"] * valutakurs

    dfp = df[df["Antal aktier"] > 0].copy()
    dfp["Värde (SEK)"] = dfp["Antal aktier"] * dfp["Aktuell kurs"] * valutakurs
    portvarde = dfp["Värde (SEK)"].sum()
    nuv = dfp.loc[dfp["Ticker"] == rad["Ticker"], "Värde (SEK)"].sum()
    nuv_andel = round((nuv / portvarde) * 100, 2) if portvarde else 0
    ny_andel = round(((nuv + investering) / portvarde) * 100, 2) if portvarde else 0

    st.markdown(f"""
### 💰 Förslag {idx+1} av {len(df2)}
- **{rad['Bolagsnamn']} ({rad['Ticker']})**
- Kurs: {round(rad['Aktuell kurs'],2)} USD  
- {riktkurs_val}: {round(rad[riktkurs_val],2)} USD  
- Potential: {round(rad['Potential (%)'],2)}%  
- Antal att köpa: {antal} → investering {round(investering,2)} SEK  
- Nuvarande andel: {nuv_andel}%  
- Andel efter köp: {ny_andel}%  
""")

    if st.button("➡️ Nästa förslag"):
        st.session_state.forslags_index += 1

#######################
# Del 3: Portföljvy inkl. utdelningar och valutakursinmatningar
#######################

def visa_portfolj(df, valutakurs, nok_usd, cad_usd, sek_usd, eur_usd):
    st.subheader("📦 Min portfölj")
    dfp = df[df["Antal aktier"] > 0].copy()
    if dfp.empty:
        st.info("Du äger inga aktier.")
        return

    valutapar = {"NOK": nok_usd, "CAD": cad_usd, "SEK": sek_usd, "EUR": eur_usd, "USD":1.0}

    dfp["Värde (SEK)"] = dfp["Antal aktier"] * dfp["Aktuell kurs"] * valutakurs
    dfp["Utdelning_USD"] = dfp.apply(lambda r: r["Årlig utdelning"] * valutapar.get(r["Valuta"],1.0), axis=1)
    dfp["Årlig utdelning SEK"] = dfp["Utdelning_USD"] * dfp["Antal aktier"] * valutakurs

    total_utdel = dfp["Årlig utdelning SEK"].sum()
    m_utdel = round(total_utdel / 12, 2)
    total = dfp["Värde (SEK)"].sum()
    dfp["Andel (%)"] = round(dfp["Värde (SEK)"] / total * 100, 2)

    st.markdown(f"**Totalt portföljvärde:** {round(total,2)} SEK  \n"
                f"**Förväntad årlig utdelning:** {round(total_utdel,2)} SEK  \n"
                f"**Genomsnittlig månadsutdelning:** {m_utdel} SEK")
    st.dataframe(dfp[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Värde (SEK)","Andel (%)","Årlig utdelning SEK"]],
                 use_container_width=True)

#######################
# Del 4: Analysvy + main()
#######################

def analysvy(df, valutakurs, nok_usd, cad_usd, sek_usd, eur_usd):
    st.subheader("📈 Analysläge")
    df = uppdatera_berakningar(df)
    if st.button("🔄 Uppdatera alla aktuella kurser från Yahoo"):
        misslyckade, uppdat = [], 0
        total = len(df)
        status = st.empty()

        with st.spinner("Uppdaterar kurser..."):
            for i, r in df.iterrows():
                ticker = str(r["Ticker"]).strip().upper()
                status.text(f"🔄 Uppdaterar {i+1} av {total}… {ticker}")
                pris, curr = hamta_kurs_och_valuta(ticker)
                if pris is None or curr is None:
                    misslyckade.append(ticker)
                else:
                    kurs_usd = pris * {"NOK":nok_usd,"CAD":cad_usd,"SEK":sek_usd,"EUR":eur_usd,"USD":1.0}.get(curr,1.0)
                    df.at[i,"Aktuell kurs"] = round(kurs_usd,2)
                    uppdat += 1
                time.sleep(2)

        spara_data(df)
        status.text("✅ Uppdatering slutförd.")
        st.success(f"{uppdat} tickers uppdaterade.")
        if misslyckade:
            st.warning("Kunde inte uppdatera: " + ", ".join(misslyckade))
    st.dataframe(df, use_container_width=True)

def main():
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    st.sidebar.header("Valutakurser")
    nok_usd = st.sidebar.number_input("NOK → USD", value=0.093)
    cad_usd = st.sidebar.number_input("CAD → USD", value=0.74)
    sek_usd = st.sidebar.number_input("SEK → USD", value=0.10)
    eur_usd = st.sidebar.number_input("EUR → USD", value=1.10)
    valutakurs = st.sidebar.number_input("Valutakurs USD → SEK", value=10.0, step=0.1)

    meny = st.sidebar.radio("Meny", ["Analys","Lägg till / uppdatera bolag","Investeringsförslag","Portfölj"])
    if meny == "Analys":
        analysvy(df, valutakurs, nok_usd, cad_usd, sek_usd, eur_usd)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        spara_data(df)
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, valutakurs)
    elif meny == "Portfölj":
        visa_portfolj(df, valutakurs, nok_usd, cad_usd, sek_usd, eur_usd)

if __name__ == "__main__":
    main()
