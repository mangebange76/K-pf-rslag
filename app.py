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

# --- Hårdkodade standard-växelkurser (till SEK) för snabb start ---
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.00,
}

# ----------------------------
# Google Sheets helpers
# ----------------------------
def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data() -> pd.DataFrame:
    data = skapa_koppling().get_all_records()
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ----------------------------
# Kolumn-setup och typer
# ----------------------------
def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # Exakt kolumnlista enligt dina krav
    kolumner = [
        "Ticker",
        "Bolagsnamn",
        "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier",
        "Valuta",
        "Årlig utdelning",
        "Aktuell kurs",
        "CAGR 5 år (%)",
        "P/S-snitt",
    ]
    for k in kolumner:
        if k not in df.columns:
            # Siffror vs text
            if any(x in k.lower() for x in ["p/s", "omsättning", "kurs", "antal", "utdelning", "cagr"]):
                df[k] = 0.0
            else:
                df[k] = ""
    # Ta bort ev. gamla kolumner som du inte använder längre (valfritt – kommentera bort om du vill spara allt historiskt)
    drop_candidates = ["Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Riktkurs om idag"]
    for c in drop_candidates:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    numeriska = [
        "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier",
        "Årlig utdelning",
        "Aktuell kurs",
        "CAGR 5 år (%)",
        "P/S-snitt",
    ]
    for k in numeriska:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)
    # Ticker, Bolagsnamn, Valuta som text
    for k in ["Ticker", "Bolagsnamn", "Valuta"]:
        if k in df.columns:
            df[k] = df[k].astype(str)
    return df

# ----------------------------
# Yahoo helpers (pris + valuta + bolagsnamn)
# ----------------------------
def hamta_kurs_och_valuta_och_namn(ticker: str):
    """
    Returnerar (pris, valuta, bolagsnamn). Om något saknas → None/"USD"/"".
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        namn = info.get("shortName") or info.get("longName") or ""
        return pris, valuta, namn
    except Exception:
        return None, "USD", ""

def hamta_valutakurs(valuta: str, manuella: dict) -> float:
    """
    Valuta→SEK enligt manuella inmatningar i sidopanelen (fallback till STANDARD_VALUTAKURSER).
    """
    if not valuta:
        return 1.0
    v = valuta.upper().strip()
    if v in manuella and manuella[v] > 0:
        return float(manuella[v])
    return STANDARD_VALUTAKURSER.get(v, 1.0)

# ----------------------------
# CAGR-justering (tak/golv) + beräkningar
# ----------------------------
def _justerad_tillvaxt(cagr_pct: float) -> float:
    """
    Tar CAGR 5 år (%) och justerar enligt reglerna:
    - > 100%  -> 50%
    - < 0%    -> 2%
    - annars  -> oförändrat
    Returnerar decimal (0.12 = 12%).
    """
    if cagr_pct is None or pd.isna(cagr_pct):
        return 0.0
    try:
        c = float(cagr_pct)
    except Exception:
        return 0.0
    if c > 100:
        return 0.50
    if c < 0:
        return 0.02
    return c / 100.0

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Omsättning om 2 & 3 år från "Omsättning nästa år" och "CAGR 5 år (%)"
    for i, rad in df.iterrows():
        oms_next = float(rad.get("Omsättning nästa år", 0.0) or 0.0)
        g = _justerad_tillvaxt(float(rad.get("CAGR 5 år (%)", 0.0) or 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * (1.0 + g) * (1.0 + g), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = 0.0
            df.at[i, "Omsättning om 3 år"] = 0.0

    # 2) P/S-snitt (snitt av Q1..Q4 > 0)
    for i, rad in df.iterrows():
        ps = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps = [float(x) for x in ps if pd.notna(x) and float(x) > 0]
        df.at[i, "P/S-snitt"] = round(np.mean(ps), 2) if ps else 0.0

    # 3) Riktkurser
    for i, rad in df.iterrows():
        uts = float(rad.get("Utestående aktier", 0.0) or 0.0)
        psn = float(rad.get("P/S-snitt", 0.0) or 0.0)
        if uts > 0 and psn > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))     * psn) / uts, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0)) * psn) / uts, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(rad.get("Omsättning om 2 år", 0.0))  * psn) / uts, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(rad.get("Omsättning om 3 år", 0.0))  * psn) / uts, 2)
        else:
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

# ----------------------------
# Lägg till / uppdatera bolag (form)
# ----------------------------
def lagg_till_eller_uppdatera(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Välj befintligt via bolagsnamn (Ticker)
    namn_map = {f"{rad.get('Bolagsnamn','')} ({rad['Ticker']})": rad['Ticker'] for _, rad in df.iterrows() if str(rad.get("Ticker","")).strip() != ""}
    valt = st.selectbox("Välj bolag (eller lämna tom för nytt)", [""] + sorted(namn_map.keys()))

    if valt:
        tick = namn_map[valt]
        bef = df[df["Ticker"] == tick].iloc[0]
    else:
        bef = pd.Series(dtype=object)

    with st.form("form_bolag"):
        # --- Manuella inputs (enligt dina regler) ---
        ticker = st.text_input("Ticker", value=bef.get("Ticker","") if not bef.empty else "").upper().strip()
        utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0, step=1.0)
        antal  = st.number_input("Antal aktier (du äger)", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0, step=1.0)

        ps_idag = st.number_input("P/S (nuvarande)", value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
        ps1     = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
        ps2     = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
        ps3     = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
        ps4     = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
        oms_next = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

        # Submit
        spara = st.form_submit_button("💾 Spara")

    if spara and ticker:
        # Skriv manuella fält
        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
        }

        if ticker in df["Ticker"].values:
            for k, v in ny.items():
                df.loc[df["Ticker"] == ticker, k] = v
        else:
            # skapa basrad med alla kolumner
            tom = {c: ("" if c in ["Ticker","Bolagsnamn","Valuta"] else 0.0) for c in df.columns}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # Hämta från Yahoo (namn/kurs/valuta/utdelning – om möjligt)
        pris, valuta, namn = hamta_kurs_och_valuta_och_namn(ticker)
        idx = df["Ticker"] == ticker
        if namn:
            df.loc[idx, "Bolagsnamn"] = namn
        if pris is not None:
            df.loc[idx, "Aktuell kurs"] = round(float(pris), 2)
        if valuta:
            df.loc[idx, "Valuta"] = valuta

        # För utdelning: yfinance har ibland "dividendRate" i info:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            div = info.get("dividendRate", None)
            if div is not None:
                df.loc[idx, "Årlig utdelning"] = float(div)
        except Exception:
            pass

        # Kör beräkningar och spara
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.success(f"{ticker} sparat & beräkningar uppdaterade.")

    return df

def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📦 Min portfölj")

    dfp = df.copy()
    dfp = dfp[dfp["Antal aktier"] > 0]
    if dfp.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs per rad (för att visa SEK-värde)
    dfp["Växelkurs"] = dfp["Valuta"].map(lambda v: hamta_valutakurs(v, valutakurser)).fillna(1.0)
    dfp["Värde (SEK)"] = dfp["Antal aktier"] * dfp["Aktuell kurs"] * dfp["Växelkurs"]
    total_varde = dfp["Värde (SEK)"].sum()

    # Utdelning i SEK
    dfp["Total årlig utdelning (SEK)"] = dfp["Antal aktier"] * dfp["Årlig utdelning"] * dfp["Växelkurs"]
    total_utd = dfp["Total årlig utdelning (SEK)"].sum()

    dfp["Andel (%)"] = (dfp["Värde (SEK)"] / total_varde * 100).round(2)

    st.markdown(f"**Totalt portföljvärde:** {round(total_varde,2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utd,2)} SEK")
    st.markdown(f"**Genomsnittlig månadsutdelning:** {round(total_utd/12,2)} SEK")

    st.dataframe(
        dfp[[
            "Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
            "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"
        ]],
        use_container_width=True
    )

def visa_investeringsforslag(df: pd.DataFrame, valutakurser: dict):
    st.subheader("💡 Investeringsförslag")

    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas i uppsidan?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )
    filterval = st.radio("Visa förslag för:", ["Alla bolag", "Endast portföljen"])

    df_calc = df.copy()
    # Växelkurs behövs bara för att räkna *antal* (kapital i SEK / pris i SEK), inte för att visa priserna (de visas i aktiens valuta).
    df_calc["Växelkurs"] = df_calc["Valuta"].map(lambda v: hamta_valutakurs(v, valutakurser)).fillna(1.0)

    if filterval == "Endast portföljen":
        dfc = df_calc[df_calc["Antal aktier"] > 0]
    else:
        dfc = df_calc

    # Filtrera bara de där riktkurs_val > aktuell kurs (potentiell uppsida)
    dfc = dfc[dfc[riktkurs_val] > dfc["Aktuell kurs"]].copy()
    if dfc.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    dfc["Potential (%)"] = ((dfc[riktkurs_val] - dfc["Aktuell kurs"]) / dfc["Aktuell kurs"]) * 100.0
    dfc = dfc.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    # Bläddring
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    idx = st.session_state.forslags_index
    if idx >= len(dfc):
        idx = 0
        st.session_state.forslags_index = 0

    rad = dfc.iloc[idx]
    # Antal att köpa beräknas i SEK, men priser visas i originalvalutan
    pris_sek = float(rad["Aktuell kurs"]) * float(rad["Växelkurs"])
    antal_kop = int(kapital_sek // pris_sek) if pris_sek > 0 else 0

    # Portföljandel före/efter (i SEK)
    dff = df_calc[df_calc["Antal aktier"] > 0].copy()
    dff["Värde (SEK)"] = dff["Antal aktier"] * dff["Aktuell kurs"] * dff["Växelkurs"]
    port_sum = dff["Värde (SEK)"].sum()
    nu_innehav = dff[dff["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
    ny_total = nu_innehav + antal_kop * pris_sek
    nu_andel = round((nu_innehav / port_sum) * 100, 2) if port_sum > 0 else 0.0
    ny_andel = round((ny_total / port_sum) * 100, 2) if port_sum > 0 else 0.0

    # Visa all fyra riktkurser (fetmarkera vald rad)
    def fmt_rk(label):
        val = float(rad.get(label, 0.0) or 0.0)
        if label == riktkurs_val:
            return f"**{label}: {val:.2f} {rad['Valuta']}**"
        return f"{label}: {val:.2f} {rad['Valuta']}"

    st.markdown(f"**Förslag {idx+1} / {len(dfc)}**")
    st.markdown(f"""
**Bolag:** {rad['Bolagsnamn']} ({rad['Ticker']})  
**Aktuell kurs:** {rad['Aktuell kurs']:.2f} {rad['Valuta']}  
{fmt_rk("Riktkurs idag")}  
{fmt_rk("Riktkurs om 1 år")}  
{fmt_rk("Riktkurs om 2 år")}  
{fmt_rk("Riktkurs om 3 år")}  

**Potential (baserat på {riktkurs_val}):** {rad['Potential (%)']:.2f}%  
**Förslag antal att köpa:** {antal_kop} st  
**Beräknad investering:** {antal_kop * pris_sek:.2f} SEK  
**Nuvarande andel i portföljen:** {nu_andel}%  
**Andel efter köp:** {ny_andel}%  
""")

    cols = st.columns(2)
    if cols[0].button("⬅️ Föregående"):
        st.session_state.forslags_index = (idx - 1) % len(dfc)
    if cols[1].button("➡️ Nästa"):
        st.session_state.forslags_index = (idx + 1) % len(dfc)

def analysvy(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📈 Analysläge")

    # Valfritt filter överst (visa enskilt bolag + hela tabellen under)
    tickers = ["(Visa alla)"] + sorted([t for t in df["Ticker"].astype(str).tolist() if t.strip() != ""])
    val = st.selectbox("Filtrera visning (ticker):", tickers)
    if val != "(Visa alla)":
        visa = df[df["Ticker"] == val].copy()
        st.markdown("#### Detaljer (filtrerat)")
        st.dataframe(visa, use_container_width=True)

    # Hela tabellen alltid under
    st.markdown("#### Databasen")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("### Uppdatera från Yahoo")
    delay = st.number_input("Fördröjning mellan anrop (sek)", value=1.0, min_value=0.0, step=0.5)
    if st.button("🔄 Uppdatera alla nu"):
        miss = []
        uppd = 0
        total = len(df)
        status = st.empty()
        bar = st.progress(0)

        for i, row in df.iterrows():
            ticker = str(row.get("Ticker","")).strip().upper()
            status.text(f"Uppdaterar {i+1}/{total}: {ticker}")
            if not ticker:
                miss.append("(tom ticker)")
                bar.progress((i+1)/total)
                continue
            try:
                pris, valuta, namn = hamta_kurs_och_valuta_och_namn(ticker)
                if pris is not None:
                    df.at[i, "Aktuell kurs"] = round(float(pris),2)
                if valuta:
                    df.at[i, "Valuta"] = valuta
                if namn:
                    df.at[i, "Bolagsnamn"] = namn
                uppd += 1
            except Exception:
                miss.append(ticker)
            bar.progress((i+1)/total)
            time.sleep(max(0.0, float(delay)))

        # Kör beräkningar + spara
        df = uppdatera_berakningar(df)
        try:
            spara_data(df)
            status.text("✅ Uppdatering & beräkningar sparade.")
            st.success(f"Uppdaterade {uppd} av {total} tickers.")
        except Exception as e:
            st.warning("Kunde inte skriva till Google Sheet just nu.")
            st.exception(e)

        if miss:
            st.warning("Kunde inte uppdatera:\n" + ", ".join(miss))

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs in data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Manuella växelkurser i sidopanelen (till SEK)
    st.sidebar.header("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=STANDARD_VALUTAKURSER["USD"], step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=STANDARD_VALUTAKURSER["NOK"], step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=STANDARD_VALUTAKURSER["CAD"], step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=STANDARD_VALUTAKURSER["EUR"], step=0.01),
    }

    # Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if st.sidebar.button("💾 Spara nu (utan ändringar)"):
        # Kör alltid beräkningar innan man spar “manuellt”
        df = uppdatera_berakningar(df)
        spara_data(df)
        st.success("Sparat till Google Sheet.")

    if meny == "Analys":
        # Räkna om innan visning så allt är fräscht
        df = uppdatera_berakningar(df)
        analysvy(df, valutakurser)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        # (lagg_till_eller_uppdatera kör redan uppdatera_berakningar + spara)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, valutakurser)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df)
        visa_portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
