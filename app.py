import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

# ===== App-inställning =====
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ===== Google Sheets =====
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ===== Standard växelkurser (X -> SEK) för sidopanel =====
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

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # Kolumner enligt din specifikation (+ Omsättningsvaluta)
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
        "Omsättningsvaluta",
    ]
    for k in kolumner:
        if k not in df.columns:
            if any(x in k.lower() for x in ["p/s", "omsättning", "kurs", "aktier", "utdelning", "cagr"]):
                df[k] = 0.0
            else:
                df[k] = ""
            if k == "Omsättningsvaluta":
                df[k] = "USD"

    # Ta bort kända gamla/överflödiga kolumner om de råkar finnas
    for c in ["Riktkurs 2026", "Riktkurs 2027", "Riktkurs 2028", "Riktkurs om idag"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier",
        "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
        "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
        "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
        "Antal aktier","Årlig utdelning","Aktuell kurs","CAGR 5 år (%)","P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Omsättningsvaluta"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def hamta_kurs_och_valuta_och_namn(ticker: str):
    """Hämtar aktuell kurs, valuta, och namn (short/long) från Yahoo."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        namn = info.get("shortName") or info.get("longName") or ""
        # Utdelning per aktie (årstakt) om tillgängligt
        utd = info.get("dividendRate", None)
        if utd is None:
            utd = 0.0
        return pris, valuta, namn, float(utd)
    except Exception:
        return None, "USD", "", 0.0

def konvertera_belopp(belopp: float, fran: str, till: str, kurser: dict) -> float:
    """Kurser är X->SEK. Konvertera belopp från 'fran' till 'till' via SEK."""
    if belopp is None:
        return 0.0
    fran = (fran or "USD").upper().strip()
    till = (till or "USD").upper().strip()
    kurs_fran = float(kurser.get(fran, 1.0))
    kurs_till = float(kurser.get(till, 1.0))
    if kurs_till <= 0:
        kurs_till = 1.0
    return float(belopp) * kurs_fran / kurs_till

def beräkna_cagr_serie(revenues: list[float]) -> float:
    """
    CAGR i % givet en lista med årsomsättningar i kronologisk ordning (minst 2 punkter).
    Om 5+ punkter finns, använder vi första och sista (annualiserar över N-1 år).
    """
    series = [float(x) for x in revenues if x is not None and x > 0]
    if len(series) < 2:
        return 0.0
    first = series[0]
    last = series[-1]
    years = len(series) - 1
    if first <= 0 or years <= 0:
        return 0.0
    cagr = (last / first) ** (1.0 / years) - 1.0
    return round(cagr * 100.0, 2)

def hamta_cagr_5_ar_fran_yahoo(ticker: str) -> float:
    """
    Försök läsa årlig Total Revenue via yfinance och räkna CAGR 5 år (eller över tillgängliga år >= 2).
    Om ej möjligt: returnera 0.0.
    """
    try:
        t = yf.Ticker(ticker)
        # yfinance: annual financials -> DataFrame med rader och kolumner per år
        fin = t.financials  # kan vara tom i vissa fall
        if fin is None or fin.empty:
            return 0.0
        # Försök få 'Total Revenue' rad (kan heta olika ibland)
        possible_rows = ["Total Revenue", "TotalRevenue", "Revenue"]
        row = None
        for r in possible_rows:
            if r in fin.index:
                row = r
                break
        if row is None:
            return 0.0
        # Hämta årsserien, kolumnordning är oftast senaste först – vi vänder till kronologisk
        vals = fin.loc[row].dropna().values.tolist()
        if len(vals) < 2:
            return 0.0
        vals = vals[::-1]  # äldst -> nyast
        return beräkna_cagr_serie(vals)
    except Exception:
        return 0.0

def _justerad_tillvaxt(cagr_pct: float) -> float:
    """
    Justera CAGR enligt överenskommelse:
    > 100% -> 50% tillväxt
    <   0% -> +2% (inflation)
    övrigt -> cagr/100
    """
    if cagr_pct is None or pd.isna(cagr_pct):
        return 0.0
    c = float(cagr_pct)
    if c > 100:
        return 0.50
    if c < 0:
        return 0.02
    return c / 100.0

def uppdatera_fran_yahoo(df: pd.DataFrame, tickers: list[str] | None, delay_sec: float = 1.0):
    """
    Uppdaterar: Aktuell kurs, Valuta, Bolagsnamn, Årlig utdelning, CAGR 5 år (%)
    för valda tickers (eller alla om tickers=None).
    Sparar INTE – returnerar uppdaterad df.
    """
    if tickers is None:
        todo = df["Ticker"].dropna().astype(str).str.strip().tolist()
    else:
        todo = [t for t in tickers if isinstance(t, str) and t.strip()]
    if not todo:
        return df, []

    fail = []
    bar = st.progress(0)
    for i, t in enumerate(todo):
        try:
            pris, valuta, namn, utd = hamta_kurs_och_valuta_och_namn(t)
            if pris is not None:
                df.loc[df["Ticker"] == t, "Aktuell kurs"] = round(float(pris), 2)
            if valuta:
                df.loc[df["Ticker"] == t, "Valuta"] = valuta
            if namn:
                df.loc[df["Ticker"] == t, "Bolagsnamn"] = namn
            df.loc[df["Ticker"] == t, "Årlig utdelning"] = round(float(utd), 4)

            cagr = hamta_cagr_5_ar_fran_yahoo(t)
            df.loc[df["Ticker"] == t, "CAGR 5 år (%)"] = round(float(cagr), 2)
        except Exception:
            fail.append(t)
        bar.progress((i + 1) / len(todo))
        time.sleep(max(0.0, delay_sec))
    return df, fail

def uppdatera_berakningar(df: pd.DataFrame, valutakurser: dict) -> pd.DataFrame:
    # 1) Omsättning om 2/3 år från "Omsättning nästa år" via justerad CAGR
    for i, rad in df.iterrows():
        oms_next = float(rad.get("Omsättning nästa år", 0.0) or 0.0)
        g = _justerad_tillvaxt(float(rad.get("CAGR 5 år (%)", 0.0) or 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * (1.0 + g) * (1.0 + g), 2)
        else:
            df.at[i, "Omsättning om 2 år"] = 0.0
            df.at[i, "Omsättning om 3 år"] = 0.0

    # 2) P/S-snitt (positiva värden)
    for i, rad in df.iterrows():
        ps = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps = [float(x) for x in ps if pd.notna(x) and float(x) > 0]
        df.at[i, "P/S-snitt"] = round(np.mean(ps), 2) if ps else 0.0

    # 3) Riktkurser – konvertera omsättning (Omsättningsvaluta -> Valuta) innan formeln
    for i, rad in df.iterrows():
        uts = float(rad.get("Utestående aktier", 0.0) or 0.0)
        psn = float(rad.get("P/S-snitt", 0.0) or 0.0)
        aktie_val = (rad.get("Valuta") or "USD").strip().upper()
        oms_val  = (rad.get("Omsättningsvaluta") or "USD").strip().upper()

        if uts > 0 and psn > 0:
            oms0 = konvertera_belopp(float(rad.get("Omsättning idag", 0.0) or 0.0),  oms_val, aktie_val, valutakurser)
            oms1 = konvertera_belopp(float(rad.get("Omsättning nästa år", 0.0) or 0.0), oms_val, aktie_val, valutakurser)
            oms2 = konvertera_belopp(float(rad.get("Omsättning om 2 år", 0.0) or 0.0),  oms_val, aktie_val, valutakurser)
            oms3 = konvertera_belopp(float(rad.get("Omsättning om 3 år", 0.0) or 0.0),  oms_val, aktie_val, valutakurser)

            df.at[i, "Riktkurs idag"]    = round((oms0 * psn) / uts, 2)
            df.at[i, "Riktkurs om 1 år"] = round((oms1 * psn) / uts, 2)
            df.at[i, "Riktkurs om 2 år"] = round((oms2 * psn) / uts, 2)
            df.at[i, "Riktkurs om 3 år"] = round((oms3 * psn) / uts, 2)
        else:
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df

def lagg_till_eller_uppdatera(df: pd.DataFrame, valutakurser: dict) -> pd.DataFrame:
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Rullista: Bolagsnamn (Ticker)
    namn_map = {f"{rad.get('Bolagsnamn','').strip()} ({rad.get('Ticker','').strip()})": rad.get('Ticker','').strip()
                for _, rad in df.iterrows() if str(rad.get('Ticker','')).strip()}
    valt = st.selectbox("Välj bolag (eller lämna tom för nytt)", [""] + sorted(namn_map.keys()))

    if valt:
        ticker_vald = namn_map[valt]
        bef = df[df["Ticker"] == ticker_vald].iloc[0]
    else:
        bef = pd.Series(dtype=object)

    with st.form("form_bolag"):
        # Manuella fält
        ticker = st.text_input("Ticker", value=bef.get("Ticker","") if not bef.empty else "").upper()
        utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
        ps_idag = st.number_input("P/S", value=float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)

        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
        oms_nxt  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

        # Omsättningens valuta (direkt under omsättningsfälten)
        val_list = ["USD","SEK","NOK","EUR","CAD"]
        default_oms_v = bef.get("Omsättningsvaluta","USD") if not bef.empty else "USD"
        if default_oms_v not in val_list:
            default_oms_v = "USD"
        oms_val = st.selectbox("Omsättningsvaluta", val_list, index=val_list.index(default_oms_v))

        antal_ag = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

        st.caption("Följande hämtas från Yahoo via knapparna nedan eller i andra vyer:")
        st.write(f"- Bolagsnamn: {bef.get('Bolagsnamn','') if not bef.empty else ''}")
        st.write(f"- Aktuell kurs: {bef.get('Aktuell kurs',0.0) if not bef.empty else 0.0}")
        st.write(f"- Valuta: {bef.get('Valuta','') if not bef.empty else ''}")
        st.write(f"- Årlig utdelning: {bef.get('Årlig utdelning',0.0) if not bef.empty else 0.0}")
        st.write(f"- CAGR 5 år (%): {bef.get('CAGR 5 år (%)',0.0) if not bef.empty else 0.0}")

        col1, col2 = st.columns(2)
        with col1:
            spara = st.form_submit_button("💾 Spara")
        with col2:
            uppd_vald = st.form_submit_button("🔄 Uppdatera vald från Yahoo")

    if spara and ticker:
        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_nxt,
            "Omsättningsvaluta": oms_val,
            "Antal aktier": antal_ag,
        }
        if ticker in df["Ticker"].values:
            for k,v in ny.items():
                df.loc[df["Ticker"]==ticker, k] = v
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)
            st.success(f"{ticker} tillagt.")
        # Räkna och spara direkt
        df = uppdatera_berakningar(df, valutakurser)
        spara_data(df)

    if uppd_vald:
        if not valt and not (ticker and ticker in df["Ticker"].values):
            st.warning("Välj ett befintligt bolag i rullistan eller spara det nya först.")
        else:
            target = [namn_map[valt]] if valt else [ticker]
            df, fail = uppdatera_fran_yahoo(df, target, delay_sec=1.0)
            df = uppdatera_berakningar(df, valutakurser)
            spara_data(df)
            if fail:
                st.warning("Kunde inte uppdatera: " + ", ".join(fail))
            else:
                st.success("Vald ticker uppdaterad från Yahoo.")
    return df

def analysvy(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📈 Analys")

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("🔄 Uppdatera alla från Yahoo (Analys)"):
            df, fail = uppdatera_fran_yahoo(df, None, delay_sec=1.0)
            df = uppdatera_berakningar(df, valutakurser)
            spara_data(df)
            if fail:
                st.warning("Kunde inte uppdatera: " + ", ".join(fail))
            else:
                st.success("Alla tickers uppdaterade.")

    # Rullista + visning
    tickers = df["Ticker"].fillna("").astype(str).tolist() if "Ticker" in df.columns else []
    valt = st.selectbox("Välj bolag att visa", ["(alla)"] + tickers, index=0)
    if valt != "(alla)":
        filtrerad = df[df["Ticker"]==valt]
        st.dataframe(filtrerad, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📦 Min portfölj")

    if st.button("🔄 Uppdatera alla från Yahoo (Portfölj)"):
        df, fail = uppdatera_fran_yahoo(df, None, delay_sec=1.0)
        df = uppdatera_berakningar(df, valutakurser)
        spara_data(df)
        if fail:
            st.warning("Kunde inte uppdatera: " + ", ".join(fail))
        else:
            st.success("Alla tickers uppdaterade.")

    if df.empty or "Antal aktier" not in df.columns:
        st.info("Ingen data.")
        return

    dfp = df.copy()
    dfp["Växelkurs"] = dfp["Valuta"].str.upper().map(valutakurser).fillna(1.0)
    dfp["Värde (SEK)"] = dfp["Antal aktier"] * dfp["Aktuell kurs"] * dfp["Växelkurs"]
    dfp["Andel (%)"] = (dfp["Värde (SEK)"] / dfp["Värde (SEK)"].sum()*100.0).round(2)
    dfp["Total årlig utdelning"] = dfp["Antal aktier"] * dfp["Årlig utdelning"] * dfp["Växelkurs"]

    total_varde = dfp["Värde (SEK)"].sum()
    total_utd = dfp["Total årlig utdelning"].sum()

    st.markdown(f"**Totalt portföljvärde:** {round(total_varde,2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utd,2)} SEK")
    st.markdown(f"**Genomsnittlig månadsutdelning:** {round(total_utd/12,2)} SEK")

    st.dataframe(
        dfp[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning"]],
        use_container_width=True
    )

def visa_investeringsforslag(df: pd.DataFrame, portfoljfilter: bool):
    st.subheader("💡 Investeringsförslag")

    if st.button("🔄 Uppdatera alla från Yahoo (Förslag)"):
        # Detta uppdaterar ej lokala df här – main kör om vyn efter uppdatering/spar
        st.session_state.trigger_update_all = True

    rikt_alternativ = ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"]
    riktval = st.selectbox("Jämför mot:", rikt_alternativ, index=1)
    kapital = st.number_input("Tillgängligt kapital (i aktiens valuta)", value=500.0, step=100.0)

    data = df.copy()
    if portfoljfilter:
        data = data[data["Antal aktier"] > 0]

    data = data[(data[riktval] > 0) & (data["Aktuell kurs"] > 0)].copy()
    if data.empty:
        st.info("Inga kandidater.")
        return

    data["Potential (%)"] = ((data[riktval] - data["Aktuell kurs"]) / data["Aktuell kurs"] * 100.0).round(2)
    data = data.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)

    # Bläddring
    if "forslag_index" not in st.session_state:
        st.session_state.forslag_index = 0
    n = len(data)
    if st.session_state.forslag_index >= n:
        st.session_state.forslag_index = 0

    col_prev, col_info, col_next = st.columns([1,3,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.forslag_index = (st.session_state.forslag_index - 1) % n
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.forslag_index = (st.session_state.forslag_index + 1) % n

    rad = data.iloc[st.session_state.forslag_index]
    st.markdown(f"**Förslag {st.session_state.forslag_index+1}/{n}**")

    # Antal för givet kapital (i aktiens valuta)
    antal = int(kapital // max(rad["Aktuell kurs"], 1e-9))

    def mk(label):
        return f"**{label}**" if label == riktval else label

    st.markdown(f"""
**{rad['Bolagsnamn']} ({rad['Ticker']})**

- Aktuell kurs: {rad['Aktuell kurs']:.2f} {rad['Valuta']}
- {mk('Riktkurs idag')}: {rad['Riktkurs idag']:.2f} {rad['Valuta']}
- {mk('Riktkurs om 1 år')}: {rad['Riktkurs om 1 år']:.2f} {rad['Valuta']}
- {mk('Riktkurs om 2 år')}: {rad['Riktkurs om 2 år']:.2f} {rad['Valuta']}
- {mk('Riktkurs om 3 år')}: {rad['Riktkurs om 3 år']:.2f} {rad['Valuta']}

**Uppsida (enligt val)**: {rad['Potential (%)']:.2f}%

**Förslag:** Köp {antal} st
""")

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    # Valutakurser i sidopanel (X -> SEK)
    st.sidebar.header("💱 Valutakurser till SEK")
    valutakurser = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(STANDARD_VALUTAKURSER["USD"]), step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(STANDARD_VALUTAKURSER["NOK"]), step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(STANDARD_VALUTAKURSER["CAD"]), step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(STANDARD_VALUTAKURSER["EUR"]), step=0.01),
        "SEK": 1.0,
    }

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    # Global hantering av "trigger_update_all" från Förslag-vyn
    if st.session_state.get("trigger_update_all"):
        df, fail = uppdatera_fran_yahoo(df, None, delay_sec=1.0)
        df = uppdatera_berakningar(df, valutakurser)
        spara_data(df)
        if fail:
            st.warning("Kunde inte uppdatera: " + ", ".join(fail))
        else:
            st.success("Alla tickers uppdaterade.")
        st.session_state.trigger_update_all = False

    if meny == "Analys":
        df = uppdatera_berakningar(df, valutakurser)
        analysvy(df, valutakurser)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, valutakurser)

    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, valutakurser)
        bara_port = st.checkbox("Visa endast portföljinnehav", value=False)
        visa_investeringsforslag(df, portfoljfilter=bara_port)

    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, valutakurser)
        visa_portfolj(df, valutakurser)

if __name__ == "__main__":
    main()
