import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials
from datetime import datetime

# ---- Streamlit config ----
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---- Google Sheets koppling ----
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(
    st.secrets["GOOGLE_CREDENTIALS"], scopes=scope
)
client = gspread.authorize(credentials)

def skapa_koppling():
    return client.open_by_url(SHEET_URL).worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def spara_data(df):
    sheet = skapa_koppling()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Säkerställ kolumner ----
def säkerställ_kolumner(df):
    kolumner = [
        "Ticker", "Bolagsnamn", "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt",
        "Senast manuell uppdatering"
    ]
    for kol in kolumner:
        if kol not in df.columns:
            df[kol] = ""
    return df[kolumner]

# ---- Valutakonvertering ----
def valutakurs_sek(valuta, user_rates):
    if valuta.upper() == "SEK":
        return 1.0
    return float(user_rates.get(f"{valuta.upper()}_SEK", 1.0))

# ---- Beräkningar ----
def konvertera_typer(df):
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def beräkna_kolumner(df):
    # Beräkna P/S-snitt
    df["P/S-snitt"] = df[["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]].mean(axis=1)

    # Beräkna omsättning 2 och 3 år framåt från CAGR
    for i, row in df.iterrows():
        cagr = row["CAGR 5 år (%)"] / 100
        omsn = row["Omsättning nästa år"]

        if cagr > 1.0:  # >100%
            cagr = 0.5
        elif cagr < 0:  # negativ
            cagr = 0.02

        df.at[i, "Omsättning om 2 år"] = omsn * (1 + cagr)
        df.at[i, "Omsättning om 3 år"] = df.at[i, "Omsättning om 2 år"] * (1 + cagr)

    # Beräkna riktkurser
    for i, row in df.iterrows():
        if row["P/S-snitt"] > 0 and row["Utestående aktier"] > 0:
            riktkurs_idag = (row["Omsättning idag"] * row["P/S-snitt"]) / row["Utestående aktier"]
            riktkurs_1 = (row["Omsättning nästa år"] * row["P/S-snitt"]) / row["Utestående aktier"]
            riktkurs_2 = (row["Omsättning om 2 år"] * row["P/S-snitt"]) / row["Utestående aktier"]
            riktkurs_3 = (row["Omsättning om 3 år"] * row["P/S-snitt"]) / row["Utestående aktier"]

            df.at[i, "Riktkurs idag"] = riktkurs_idag
            df.at[i, "Riktkurs om 1 år"] = riktkurs_1
            df.at[i, "Riktkurs om 2 år"] = riktkurs_2
            df.at[i, "Riktkurs om 3 år"] = riktkurs_3
    return df

# ---- Yahoo helpers ----
def hamta_kurs_valuta_namn_utdelning(ticker: str):
    """
    Hämtar (pris, valuta, bolagsnamn, årlig utdelning per aktie) för en ticker via yfinance.
    Returnerar tuple: (pris: float|None, valuta: str|None, namn: str|None, utd: float|None)
    """
    try:
        t = yf.Ticker(str(ticker).strip())
        info = getattr(t, "info", {}) or {}

        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", None)

        namn = info.get("longName") or info.get("shortName") or None

        # yfinance kan ge utdelning på olika nycklar; vi försöker i rimlig ordning:
        utd = None
        for k in ["trailingAnnualDividendRate", "dividendRate", "lastDividendValue"]:
            v = info.get(k)
            if v is not None:
                try:
                    utd = float(v)
                    break
                except Exception:
                    pass

        # Säkerställ typer
        if pris is not None:
            pris = float(pris)
        if utd is not None:
            utd = float(utd)

        return pris, valuta, namn, utd
    except Exception:
        return None, None, None, None


def massuppdatera_yahoo(df: pd.DataFrame, user_rates: dict, sleep_sec: float = 1.0):
    """
    Uppdaterar 'Aktuell kurs', 'Valuta', 'Bolagsnamn', 'Årlig utdelning' för alla rader.
    Stryker INTE manuella fält. Sätter bara dessa 4 om värde hittas.
    """
    misslyckade = []
    uppdaterade = 0
    total = len(df)
    status = st.empty()
    bar = st.progress(0)

    with st.spinner("Uppdaterar bolag från Yahoo…"):
        for i, row in df.iterrows():
            ticker = str(row.get("Ticker", "")).strip()
            status.text(f"🔄 {i+1}/{total} • {ticker}")

            if not ticker:
                misslyckade.append("(tom ticker)")
                bar.progress((i + 1) / total)
                continue

            pris, valuta, namn, utd = hamta_kurs_valuta_namn_utdelning(ticker)

            try:
                if pris is not None and pris >= 0:
                    df.at[i, "Aktuell kurs"] = float(pris)
                if isinstance(valuta, str) and valuta:
                    df.at[i, "Valuta"] = valuta
                if isinstance(namn, str) and namn:
                    df.at[i, "Bolagsnamn"] = namn
                if utd is not None and utd >= 0:
                    df.at[i, "Årlig utdelning"] = float(utd)   # <-- rätt nyckel (fix från 'udt' -> 'utd')
                uppdaterade += 1
            except Exception:
                misslyckade.append(ticker)

            bar.progress((i + 1) / total)
            time.sleep(max(sleep_sec, 0.0))

    spara_data(df)
    status.text("✅ Klart.")
    st.success(f"Uppdaterade {uppdaterade} av {total} rader.")
    if misslyckade:
        st.warning("Kunde inte uppdatera följande:\n" + ", ".join(misslyckade))


# ---- Portföljvy ----
def visa_portfolj(df: pd.DataFrame, user_rates: dict):
    st.subheader("📦 Min portfölj")

    df = df.copy()
    # Endast innehav
    df["Antal aktier"] = pd.to_numeric(df["Antal aktier"], errors="coerce").fillna(0.0)
    innehav = df[df["Antal aktier"] > 0].copy()

    if innehav.empty:
        st.info("Du äger inga aktier ännu.")
        return

    # Växelkurs per rad (för SEK-konvertering av värde & utdelning)
    innehav["Valuta"] = innehav["Valuta"].fillna("USD").astype(str)
    innehav["Växelkurs"] = innehav["Valuta"].apply(lambda v: valutakurs_sek(v, user_rates))

    # Numeriska fält
    innehav["Aktuell kurs"] = pd.to_numeric(innehav["Aktuell kurs"], errors="coerce").fillna(0.0)
    innehav["Årlig utdelning"] = pd.to_numeric(innehav["Årlig utdelning"], errors="coerce").fillna(0.0)

    # Beräkningar i SEK
    innehav["Värde (SEK)"] = innehav["Antal aktier"] * innehav["Aktuell kurs"] * innehav["Växelkurs"]
    total_varde = float(innehav["Värde (SEK)"].sum())

    innehav["Total årlig utdelning (SEK)"] = innehav["Antal aktier"] * innehav["Årlig utdelning"] * innehav["Växelkurs"]
    total_utdelning = float(innehav["Total årlig utdelning (SEK)"].sum())
    manads_utdelning = total_utdelning / 12.0 if total_utdelning else 0.0

    # Andelar
    if total_varde > 0:
        innehav["Andel (%)"] = (innehav["Värde (SEK)"] / total_varde) * 100.0
    else:
        innehav["Andel (%)"] = 0.0

    # Topprad med totaler
    c1, c2, c3 = st.columns(3)
    c1.metric("Totalt portföljvärde (SEK)", f"{total_varde:,.2f}")
    c2.metric("Förväntad årlig utdelning (SEK)", f"{total_utdelning:,.2f}")
    c3.metric("Snitt månadsutdelning (SEK)", f"{manads_utdelning:,.2f}")

    # Tabell
    visa_cols = [
        "Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta",
        "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning (SEK)"
    ]
    st.dataframe(innehav[visa_cols], use_container_width=True)

    # Liten uppdateringsrad längst ner (valfritt)
    with st.expander("Uppdatera kurser/valuta/namn/utdelning (Yahoo)"):
        if st.button("🔄 Uppdatera portföljens rader från Yahoo", key="pf_update_yahoo"):
            massuppdatera_yahoo(df, user_rates, sleep_sec=1.0)
            st.rerun()

# ---- Lägg till / uppdatera bolag ----
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict):
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Mappa namn för rullistan
    namn_map = {
        f"{row.get('Bolagsnamn','') or ''} ({row.get('Ticker','')})": idx
        for idx, row in df.reset_index().iterrows()
        if str(row.get("Ticker","")).strip()
    }
    namn_lista = ["(nytt bolag)"] + sorted(namn_map.keys())

    if "form_index" not in st.session_state:
        st.session_state.form_index = 0

    # rullista
    valt_visningsnamn = st.selectbox("Välj bolag (eller välj '(nytt bolag)')", namn_lista, index=st.session_state.form_index)
    # bläddringsknappar
    if namn_lista:
        pos = namn_lista.index(valt_visningsnamn)
        c_prev, c_pos, c_next = st.columns([1,2,1])
        if c_prev.button("⬅️ Föregående", key="form_prev") and pos > 0:
            st.session_state.form_index = pos - 1
            st.rerun()
        c_pos.write(f"**{pos}/{len(namn_lista)-1}**" if pos>0 else "**Nytt bolag**")
        if c_next.button("Nästa ➡️", key="form_next") and pos < len(namn_lista)-1:
            st.session_state.form_index = pos + 1
            st.rerun()

    # Hämta befintlig rad (eller tom)
    if valt_visningsnamn != "(nytt bolag)":
        i = namn_map[valt_visningsnamn]
        bef = df.iloc[i].copy()
    else:
        bef = pd.Series(dtype=object)

    with st.form("bolag_form"):
        # Fält du anger manuellt
        ticker = st.text_input("Ticker", value=str(bef.get("Ticker","")) if not bef.empty else "").upper().strip()
        utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier", 0.0)) if not bef.empty else 0.0)
        antal = st.number_input("Antal aktier (du äger)", value=float(bef.get("Antal aktier", 0.0)) if not bef.empty else 0.0)

        # P/S (manuella)
        ps_idag = st.number_input("P/S (nuvarande)", value=float(bef.get("P/S", 0.0)) if not bef.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0.0)) if not bef.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0.0)) if not bef.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0.0)) if not bef.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0.0)) if not bef.empty else 0.0)

        # Omsättningar (manuella input för idag + nästa år)
        oms_idag = st.number_input("Omsättning idag", value=float(bef.get("Omsättning idag", 0.0)) if not bef.empty else 0.0)
        oms_next = st.number_input("Omsättning nästa år", value=float(bef.get("Omsättning nästa år", 0.0)) if not bef.empty else 0.0)

        # Info (hämtas automatiskt vid uppdatering/spara)
        st.markdown("**Hämtas från Yahoo när du sparar/uppdaterar:** Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning.")
        spara = st.form_submit_button("💾 Spara")

    if spara:
        if not ticker:
            st.error("Ange en ticker.")
            return df

        # Bygg ny rad med dina manuella fält
        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps_idag, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next
        }

        # Lägg in/uppdatera i df
        if (df["Ticker"].astype(str).str.upper() == ticker).any():
            idx = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]
            for k,v in ny.items():
                df.at[idx, k] = v
            # markera manuell uppdateringstid (endast vid manuell spar)
            df.at[idx, "Senast manuell uppdatering"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success(f"{ticker} uppdaterat.")
        else:
            ny_full = {c: "" for c in df.columns}
            ny_full.update(ny)
            ny_full["Senast manuell uppdatering"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.concat([df, pd.DataFrame([ny_full])], ignore_index=True)
            st.success(f"{ticker} tillagt.")

        # Hämta Yahoo-data för just denna ticker (namn/kurs/valuta/utdelning)
        pris, valuta, namn, utd = hamta_kurs_valuta_namn_utdelning(ticker)
        idx2 = df.index[df["Ticker"].astype(str).str.upper() == ticker][0]
        if isinstance(namn, str) and namn:
            df.at[idx2, "Bolagsnamn"] = namn
        if pris is not None and pris >= 0:
            df.at[idx2, "Aktuell kurs"] = float(pris)
        if isinstance(valuta, str) and valuta:
            df.at[idx2, "Valuta"] = valuta
        if utd is not None and utd >= 0:
            df.at[idx2, "Årlig utdelning"] = float(utd)

        # Konvertera/beräkna och spara
        df = konvertera_typer(df)
        df = beräkna_kolumner(df)
        spara_data(df)

        st.success("Sparat och uppdaterat från Yahoo.")
        st.rerun()

    # Liten hjälpruta för massuppdatering
    with st.expander("Uppdatera alla bolag från Yahoo"):
        if st.button("🔄 Massuppdatera (Yahoo)", key="mass_yahoo_in_form"):
            massuppdatera_yahoo(df, user_rates, sleep_sec=1.0)
            st.rerun()


# ---- Analys-vy ----
def analysvy(df: pd.DataFrame, user_rates: dict):
    st.subheader("📈 Analys")

    # Rullista + bläddringsknappar för att se ett bolag i taget
    namn_map = {
        f"{row.get('Bolagsnamn','') or ''} ({row.get('Ticker','')})": idx
        for idx, row in df.reset_index().iterrows()
        if str(row.get("Ticker","")).strip()
    }
    namn_lista = sorted(namn_map.keys())
    if "analys_index" not in st.session_state:
        st.session_state.analys_index = 0

    valt = st.selectbox("Välj bolag", namn_lista, index=min(st.session_state.analys_index, max(len(namn_lista)-1,0)) if namn_lista else 0)

    if namn_lista:
        pos = namn_lista.index(valt)
        cprev, cpos, cnext = st.columns([1,2,1])
        if cprev.button("⬅️ Föregående", key="analys_prev") and pos > 0:
            st.session_state.analys_index = pos - 1
            st.rerun()
        cpos.write(f"**{pos+1}/{len(namn_lista)}**")
        if cnext.button("Nästa ➡️", key="analys_next") and pos < len(namn_lista)-1:
            st.session_state.analys_index = pos + 1
            st.rerun()

        # Visa vald rad
        rad = df.iloc[namn_map[valt]]
        visa = rad[[
            "Ticker","Bolagsnamn","Aktuell kurs","Valuta",
            "Utestående aktier","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "Antal aktier","Årlig utdelning","CAGR 5 år (%)","Senast manuell uppdatering"
        ]].to_frame(name="Värde")
        st.table(visa)

    # Hela databasen under
    st.markdown("### Databas")
    st.dataframe(df, use_container_width=True)

    # Massuppdatera från Yahoo
    with st.expander("Uppdatera alla från Yahoo"):
        if st.button("🔄 Uppdatera alla (Yahoo)", key="analys_mass_yahoo_btn"):
            massuppdatera_yahoo(df, user_rates, sleep_sec=1.0)
            st.rerun()

# ---- Investeringsförslag ----
def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict):
    st.subheader("💡 Investeringsförslag")

    # Val av riktkursfält
    riktkurs_val = st.selectbox(
        "Baserar uppsidan på:",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    # Sorterings/filtreringsläge
    läge = st.radio(
        "Visa",
        ["Störst uppsida först", "Närmast/toppat riktkurs (absolut avvikelse)"],
        index=0,
        horizontal=True
    )

    # Portföljfilter
    portf_filter = st.radio("Vilka bolag?", ["Alla bolag", "Endast portföljen"], index=0, horizontal=True)

    # Tillgängligt kapital (SEK) för köpförslag
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=100.0)

    data = df.copy()
    data["Antal aktier"] = pd.to_numeric(data["Antal aktier"], errors="coerce").fillna(0.0)
    if portf_filter == "Endast portföljen":
        data = data[data["Antal aktier"] > 0]

    # Filtrera bort rader utan nödvändiga fält
    data["Aktuell kurs"] = pd.to_numeric(data["Aktuell kurs"], errors="coerce").fillna(0.0)
    data[riktkurs_val] = pd.to_numeric(data[riktkurs_val], errors="coerce").fillna(0.0)
    data = data[(data["Aktuell kurs"] > 0) & (data[riktkurs_val] > 0)]

    if data.empty:
        st.info("Inget att visa. Kontrollera att du har fyllt i data och beräkningar.")
        return

    # Växelkurs för portföljandelar (SEK) – endast för beräkning av andelar
    data["Valuta"] = data["Valuta"].fillna("USD").astype(str)
    data["Växelkurs"] = data["Valuta"].apply(lambda v: valutakurs_sek(v, user_rates))
    data["Värde (SEK)"] = data["Antal aktier"] * data["Aktuell kurs"] * data["Växelkurs"]
    total_pf_sek = float(data["Värde (SEK)"].sum())

    # Uppside/avvikelse
    data["Uppsida (%)"] = ((data[riktkurs_val] - data["Aktuell kurs"]) / data["Aktuell kurs"]) * 100.0
    data["Avvikelse (%)"] = ((data[riktkurs_val] - data["Aktuell kurs"]) / data["Aktuell kurs"]) * 100.0
    data["Abs avvikelse"] = data["Avvikelse (%)"].abs()

    # Sortering
    if läge == "Störst uppsida först":
        data = data.sort_values(by="Uppsida (%)", ascending=False)
    else:
        data = data.sort_values(by="Abs avvikelse", ascending=True)

    data = data.reset_index(drop=True)

    # Paginering/bläddring
    key_idx = f"inv_idx_{riktkurs_val}_{läge}_{portf_filter}"
    if key_idx not in st.session_state:
        st.session_state[key_idx] = 0

    i = st.session_state[key_idx]
    if i >= len(data):
        i = 0
        st.session_state[key_idx] = 0

    rad = data.iloc[i]

    # Köpförslag i aktiens egen valuta för antal-beräkning?
    # Antalet beräknas i SEK via växelkurs (kapital dividerat med kurs i SEK)
    kurs_sek = rad["Aktuell kurs"] * rad["Växelkurs"]
    antal_att_kopa = int(kapital_sek // kurs_sek) if kurs_sek > 0 else 0
    investering_sek = antal_att_kopa * kurs_sek

    # Andelar i portfölj
    innehav_sek = float(data.loc[data["Ticker"] == rad["Ticker"], "Värde (SEK)"].sum())
    nu_andel = (innehav_sek / total_pf_sek) * 100.0 if total_pf_sek > 0 else 0.0
    ny_andel = ((innehav_sek + investering_sek) / total_pf_sek) * 100.0 if total_pf_sek > 0 else 0.0

    # Presenterad ruta
    st.markdown(f"### {rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"- **Aktuell kurs:** {rad['Aktuell kurs']:.2f} {rad['Valuta']}"
    )
    # lista alla riktkurser; fetmarkera den valda
    def rk(label):
        val = float(rad[label])
        if label == riktkurs_val:
            return f"**{label}: {val:.2f} {rad['Valuta']}**"
        return f"{label}: {val:.2f} {rad['Valuta']}"

    st.markdown(
        "- " + "\n- ".join([
            rk("Riktkurs idag"),
            rk("Riktkurs om 1 år"),
            rk("Riktkurs om 2 år"),
            rk("Riktkurs om 3 år"),
        ])
    )

    # Uppsida/avvikelse enligt valt läge
    if läge == "Störst uppsida först":
        st.markdown(f"**Uppsida (baserat på '{riktkurs_val}'): {rad['Uppsida (%)']:.2f}%**")
    else:
        rikt = float(rad[riktkurs_val])
        diff_pct = ((rikt - float(rad["Aktuell kurs"])) / float(rad["Aktuell kurs"])) * 100.0
        trendtxt = "över riktkurs" if diff_pct < 0 else "under riktkurs"
        st.markdown(f"**Avvikelse mot '{riktkurs_val}': {abs(diff_pct):.2f}% {trendtxt}**")

    # Köpförslag (SEK)
    st.markdown(
        f"- **Antal att köpa:** {antal_att_kopa} st\n"
        f"- **Beräknad investering:** {investering_sek:,.2f} SEK\n"
        f"- **Nuvarande andel av portfölj:** {nu_andel:.2f}%\n"
        f"- **Andel efter köp:** {ny_andel:.2f}%"
    )

    # Bläddringsknappar + index
    c_prev, c_idx, c_next = st.columns([1,2,1])
    if c_prev.button("⬅️ Föregående", key=f"inv_prev_{key_idx}") and i > 0:
        st.session_state[key_idx] = i - 1
        st.rerun()
    c_idx.write(f"**Förslag {i+1}/{len(data)}**")
    if c_next.button("Nästa ➡️", key=f"inv_next_{key_idx}") and i < len(data)-1:
        st.session_state[key_idx] = i + 1
        st.rerun()

    # Yahoo-uppdatering expander (valfritt)
    with st.expander("Uppdatera alla från Yahoo"):
        if st.button("🔄 Uppdatera alla (Yahoo)", key="inv_mass_yahoo_btn"):
            massuppdatera_yahoo(df, user_rates, sleep_sec=1.0)
            st.rerun()


# ---- Huvudprogram ----
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)
    df = beräkna_kolumner(df)

    # Valutakurser (SEK) – manuellt justerbara med förval
    st.sidebar.header("💱 Valutakurser → SEK")
    user_rates = {
        "USD_SEK": st.sidebar.number_input("USD → SEK", value=9.75, step=0.01),
        "NOK_SEK": st.sidebar.number_input("NOK → SEK", value=0.95, step=0.01),
        "CAD_SEK": st.sidebar.number_input("CAD → SEK", value=7.05, step=0.01),
        "EUR_SEK": st.sidebar.number_input("EUR → SEK", value=11.18, step=0.01),
    }

    meny = st.sidebar.radio(
        "📌 Välj vy",
        ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"],
        index=0
    )

    if meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # (lagg_till_eller_uppdatera sköter spar och rerun när man sparar)
        if df2 is not None and not df2.equals(df):
            df = df2
    elif meny == "Investeringsförslag":
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        visa_portfolj(df, user_rates)


if __name__ == "__main__":
    main()
