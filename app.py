# ---------------------------------------
# app.py – Komplett version med förbättringar
# ---------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

st.set_page_config(page_title="📈 Aktieanalys", layout="wide")

SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
INST_NAME = "Inställningar"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def skapa_koppling(sheetname):
    return client.open_by_url(SHEET_URL).worksheet(sheetname)

def hamta_data():
    data = skapa_koppling(SHEET_NAME).get_all_records()
    return pd.DataFrame(data)

def hamta_inställningar():
    try:
        df = pd.DataFrame(skapa_koppling(INST_NAME).get_all_records())
        return df.set_index("Inställning")["Värde"].to_dict()
    except:
        return {"Valutakurs": "10", "MaxPortföljAndel": "20", "MaxHögrisk": "2"}

def spara_data(df):
    sheet = skapa_koppling(SHEET_NAME)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def spara_inställningar(inst_dict):
    df = pd.DataFrame([{"Inställning": k, "Värde": str(v)} for k, v in inst_dict.items()])
    sheet = skapa_koppling(INST_NAME)
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def konvertera_till_ratt_typ(df):
    num_cols = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Aktuell kurs", "Utestående aktier", "P/S",
        "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "P/S-snitt", "Riktkurs idag", "Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028",
        "Antal aktier", "Senast uppdaterad"
    ]
    for k in kolumner:
        if k not in df.columns:
            df[k] = 0.0 if "P/S" in k or "Omsättning" in k or "kurs" in k.lower() else ""
    return df

def uppdatera_berakningar(df):
    for index, row in df.iterrows():
        ps_values = [row[k] for k in ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"] if row[k] > 0]
        ps_snitt = round(np.mean(ps_values), 2) if ps_values else 0
        aktier = row["Utestående aktier"]
        df.at[index, "P/S-snitt"] = ps_snitt
        for i, label in enumerate(["Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år"]):
            riktkurs = (row[label] * ps_snitt / aktier) if aktier > 0 else 0
            df.at[index, f"Riktkurs {['idag', '2026', '2027', '2028'][i]}"] = round(riktkurs, 2)
    return df

def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")
    tickers = sorted(df["Ticker"].dropna().unique())
    val = st.selectbox("Välj bolag att uppdatera eller lämna tomt för nytt", [""] + tickers)

    existerande = df[df["Ticker"] == val].iloc[0] if val else {}

    with st.form("form"):
        ticker = st.text_input("Ticker", value=existerande.get("Ticker", "")).upper()
        namn = st.text_input("Bolagsnamn", value=existerande.get("Bolagsnamn", ""))
        kurs = st.number_input("Aktuell kurs", value=existerande.get("Aktuell kurs", 0.0))
        aktier = st.number_input("Utestående aktier (milj)", value=existerande.get("Utestående aktier", 0.0))
        ps = st.number_input("P/S idag", value=existerande.get("P/S", 0.0))
        psq = [st.number_input(f"P/S Q{i+1}", value=existerande.get(f"P/S Q{i+1}", 0.0)) for i in range(4)]
        oms = [st.number_input(f"Omsättning {t}", value=existerande.get(f"Omsättning {t}", 0.0)) for t in ["idag", "nästa år", "om 2 år", "om 3 år"]]
        antal_aktier = st.number_input("Antal aktier du äger", value=existerande.get("Antal aktier", 0.0))
        spara = st.form_submit_button("💾 Spara bolag")

    if spara and ticker:
        ny = {
            "Ticker": ticker, "Bolagsnamn": namn, "Aktuell kurs": kurs,
            "Utestående aktier": aktier, "P/S": ps, "Antal aktier": antal_aktier,
            "Senast uppdaterad": datetime.today().strftime("%Y-%m-%d")
        }
        ny.update({f"P/S Q{i+1}": psq[i] for i in range(4)})
        ny.update({f"Omsättning {['idag', 'nästa år', 'om 2 år', 'om 3 år'][i]}": oms[i] for i in range(4)})

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny.keys()] = ny.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
    return df

def investeringsforslag(df, kapital_sek, valutakurs, maxandel, maxhogrisk):
    df = df.copy()
    df = df[df["Riktkurs 2026"] > df["Aktuell kurs"]]
    df["Potential"] = df["Riktkurs 2026"] - df["Aktuell kurs"]
    df = df.sort_values("Potential", ascending=False)

    for _, row in df.iterrows():
        pris = row["Aktuell kurs"]
        om2 = row["Omsättning om 2 år"]
        ticker = row["Ticker"]
        if pris <= 0:
            continue
        # Riskgräns
        andel = row["Antal aktier"] * pris * valutakurs
        total = (df["Antal aktier"] * df["Aktuell kurs"] * valutakurs).sum()
        portand = (andel / total) * 100 if total > 0 else 0
        risk = om2 < 1000

        if portand >= maxandel:
            continue
        varning = ""
        if risk and portand >= maxhogrisk:
            varning = "⚠️ Högriskbolag, andel redan ≥ gräns!"
        sek = kapital_sek if kapital_sek >= pris * valutakurs else 0
        return {
            "Ticker": ticker,
            "Pris": pris,
            "Maxköp (SEK)": round(sek, 2),
            "Varning": varning
        }
    return {}

def visa_investeringsrad(df, valutakurs, maxandel, maxhogrisk):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=10000.0, step=500.0)

    forslag = investeringsforslag(df, kapital_sek, valutakurs, maxandel, maxhogrisk)

    if forslag:
        st.markdown(f"**Köp:** `{forslag['Ticker']}` à `{forslag['Pris']}` USD")
        st.markdown(f"🔹 Maxköp för SEK: `{forslag['Maxköp (SEK)']}`")
        if forslag["Varning"]:
            st.warning(forslag["Varning"])
    else:
        st.info("🚫 Inget investeringsförslag just nu.")

def visa_portfolj(df, valutakurs):
    port = df[df["Antal aktier"] > 0].copy()
    port["Värde SEK"] = port["Antal aktier"] * port["Aktuell kurs"] * valutakurs
    tot = port["Värde SEK"].sum()
    port["Andel (%)"] = (port["Värde SEK"] / tot * 100).round(2)
    st.dataframe(port[["Ticker", "Antal aktier", "Aktuell kurs", "Värde SEK", "Andel (%)"]])
    st.markdown(f"💼 **Totalt värde:** {round(tot, 2)} SEK")

def main():
    df = hamta_data()
    inst = hamta_inställningar()
    df = säkerställ_kolumner(df)
    df = konvertera_till_ratt_typ(df)

    valutakurs = float(inst.get("Valutakurs", "10"))
    maxandel = float(inst.get("MaxPortföljAndel", "20"))
    maxhogrisk = float(inst.get("MaxHögrisk", "2"))

    st.sidebar.subheader("⚙️ Inställningar")
    valutakurs = st.sidebar.number_input("USD → SEK", value=valutakurs, step=0.01)
    maxandel = st.sidebar.number_input("Max portföljandel (%)", value=maxandel)
    maxhogrisk = st.sidebar.number_input("Max högrisk-andel (%)", value=maxhogrisk)
    if st.sidebar.button("💾 Spara inställningar"):
        spara_inställningar({
            "Valutakurs": valutakurs,
            "MaxPortföljAndel": maxandel,
            "MaxHögrisk": maxhogrisk
        })
        st.sidebar.success("Inställningar sparade.")

    meny = st.sidebar.radio("📁 Meny", ["📊 Analys", "➕ Lägg till/uppdatera bolag", "💡 Investeringsförslag", "📦 Portfölj"])

    if meny == "📊 Analys":
        df = uppdatera_berakningar(df)
        st.dataframe(df)
    elif meny == "➕ Lägg till/uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        df = uppdatera_berakningar(df)
        spara_data(df)
    elif meny == "💡 Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsrad(df, valutakurs, maxandel, maxhogrisk)
    elif meny == "📦 Portfölj":
        visa_portfolj(df, valutakurs)

if __name__ == "__main__":
    main()

        ticker = row["Ticker"]
        innehav = row["Antal aktier"] * pris
        total_portfolj = (df["Antal aktier"] * df["Aktuell kurs"]).sum()
        andel = innehav / total_portfolj * 100 if total_portfolj > 0 else 0

        if om2 < 1000:
            if andel >= float(maxhogrisk):
                st.warning(f"⚠️ {ticker}: Högriskbolag (omsättning < 1 mdr USD om 2 år) redan {andel:.1f}% av portföljen.")
        elif andel >= float(maxandel):
            st.warning(f"⚠️ {ticker}: Över maxgräns på {maxandel}% i portföljen.")
        else:
            antal = int(kapital_sek / (pris * valutakurs))
            if antal > 0:
                return f"Köp {antal} aktier i {row['Bolagsnamn']} ({ticker})", kapital_sek - antal * pris * valutakurs
            else:
                return f"💡 {row['Bolagsnamn']} ({ticker}) är ett bra förslag, men mer kapital krävs.", kapital_sek
    return "Inget lämpligt förslag just nu", kapital_sek

def visa_portfolj(df):
    st.subheader("📊 Portföljöversikt")
    df = df[df["Antal aktier"] > 0].copy()
    df["Position (SEK)"] = df["Antal aktier"] * df["Aktuell kurs"] * valutakurs
    df = df.sort_values("Position (SEK)", ascending=False)
    st.dataframe(df[["Bolagsnamn", "Antal aktier", "Aktuell kurs", "Position (SEK)", "Senast uppdaterad"]], use_container_width=True)

def inställningar_panel():
    st.sidebar.header("⚙️ Inställningar")
    inst = hamta_inställningar()

    valutakurs = st.sidebar.number_input("Valutakurs USD/SEK", value=float(inst.get("Valutakurs", 10.0)), step=0.01)
    maxandel = st.sidebar.number_input("Max portföljandel (%)", value=float(inst.get("MaxPortföljAndel", 20.0)))
    maxhogrisk = st.sidebar.number_input("Max andel högriskbolag (%)", value=float(inst.get("MaxHögrisk", 2.0)))
    kapital = st.sidebar.number_input("Tillgängligt kapital (SEK)", value=1000)

    if st.sidebar.button("💾 Spara inställningar"):
        spara_inställningar({
            "Valutakurs": valutakurs,
            "MaxPortföljAndel": maxandel,
            "MaxHögrisk": maxhogrisk
        })
        st.sidebar.success("Inställningar sparade")

    return valutakurs, maxandel, maxhogrisk, kapital

def main():
    st.title("📈 Investeringsförslag & Portföljanalys")
    global valutakurs
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_till_ratt_typ(df)
    valutakurs, maxandel, maxhogrisk, kapital = inställningar_panel()
    df = uppdatera_berakningar(df)
    df = lagg_till_eller_uppdatera(df)
    visa_portfolj(df)

    st.subheader("💡 Investeringsförslag")
    for i in range(3):
        forslag, kapital = investeringsforslag(df, kapital, valutakurs, maxandel, maxhogrisk)
        st.info(f"{forslag}")
        if st.button(f"Nästa förslag ({i+1})"):
            continue

    if st.button("📤 Spara förändringar till kalkylarket"):
        spara_data(df)
        st.success("Data sparad!")

if __name__ == "__main__":
    main()
