import streamlit as st
import pandas as pd
import gspread
import yfinance as yf
import time
from google.oauth2.service_account import Credentials

# ==============================
# Streamlit-inställningar
# ==============================
st.set_page_config(page_title="Aktieanalys & Investeringsförslag", layout="wide")

# ==============================
# Google Sheets-anslutning
# ==============================
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

# ==============================
# Säkerställ kolumner
# ==============================
def säkerställ_kolumner(df):
    kolumner = [
        "Ticker",
        "Bolagsnamn",
        "Utestående aktier",
        "P/S",
        "P/S Q1",
        "P/S Q2",
        "P/S Q3",
        "P/S Q4",
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
    for kolumn in kolumner:
        if kolumn not in df.columns:
            df[kolumn] = ""
    # Ta bort extra kolumner som inte ska vara där
    df = df[kolumner]
    return df

# ==============================
# Hjälpfunktioner
# ==============================
def beräkna_cagr(start, slut):
    try:
        if start > 0 and slut > 0:
            år = 5
            return ((slut / start) ** (1 / år) - 1) * 100
        else:
            return None
    except:
        return None

def justera_tillväxt(cagr):
    """Om CAGR > 100% → sätt till 50%, om < 0% → sätt till 2%."""
    if cagr is None:
        return None
    if cagr > 100:
        return 50
    elif cagr < 0:
        return 2
    else:
        return cagr

# ==============================
# Hämta data från Yahoo Finance
# ==============================
def hamta_yf_data(ticker):
    try:
        aktie = yf.Ticker(ticker)
        
        # Hämta bolagsnamn
        bolagsnamn = aktie.info.get("longName") or aktie.info.get("shortName") or ""

        # Hämta aktuell kurs & valuta
        kurs = aktie.info.get("currentPrice")
        valuta = aktie.info.get("currency")

        # Hämta årlig utdelning
        utdelning = aktie.info.get("dividendRate")

        # Hämta historisk omsättning (för CAGR-beräkning)
        # annual_total_revenue kommer i en DataFrame
        oms_hist = aktie.financials
        oms_år = None
        oms_5år = None
        if not oms_hist.empty:
            # Sortera på senaste år först
            oms_values = oms_hist.loc["Total Revenue"].sort_index(ascending=False)
            if len(oms_values) >= 5:
                oms_år = oms_values.iloc[0]
                oms_5år = oms_values.iloc[4]

        # Beräkna CAGR
        cagr = None
        if oms_år and oms_5år:
            cagr = beräkna_cagr(oms_5år, oms_år)
            cagr = justera_tillväxt(cagr)

        return {
            "Bolagsnamn": bolagsnamn,
            "Aktuell kurs": kurs,
            "Valuta": valuta,
            "Årlig utdelning": utdelning,
            "CAGR 5 år (%)": cagr
        }
    except Exception as e:
        st.error(f"Fel vid hämtning av {ticker}: {e}")
        return None

# ==============================
# Beräkningar (P/S-snitt, omsättning 2–3 år, riktkurser)
# ==============================
import numpy as np

def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def beräkna_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Säkerställ numeriska fält
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Antal aktier", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"
    ]
    df = _to_num(df, num_cols)

    # 2) P/S-snitt = snitt av P/S Q1..Q4 som är > 0
    ps_quarters = ["P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4"]
    ps_df = df[ps_quarters].where(df[ps_quarters] > 0)
    df["P/S-snitt"] = ps_df.mean(axis=1, skipna=True).fillna(0.0).round(2)

    # 3) Omsättning om 2 & 3 år
    #    Bas = "Omsättning nästa år"
    #    Tillväxt = justerad CAGR som redan ligger i "CAGR 5 år (%)" (procent)
    def proj_oms(row, år):
        bas = row.get("Omsättning nästa år", np.nan)
        cagr_pct = row.get("CAGR 5 år (%)", np.nan)
        if pd.isna(bas) or pd.isna(cagr_pct):
            return np.nan
        try:
            cagr_dec = float(cagr_pct) / 100.0
            return round(float(bas) * ((1.0 + cagr_dec) ** år), 2)
        except:
            return np.nan

    df["Omsättning om 2 år"] = df.apply(lambda r: proj_oms(r, 1), axis=1)
    df["Omsättning om 3 år"] = df.apply(lambda r: proj_oms(r, 2), axis=1)

    # 4) Riktkurser
    def riktkurs(oms, ps, aktier):
        try:
            if pd.isna(oms) or pd.isna(ps) or pd.isna(aktier):
                return np.nan
            oms = float(oms); ps = float(ps); aktier = float(aktier)
            if aktier <= 0 or ps <= 0:
                return np.nan
            return round((oms * ps) / aktier, 2)
        except:
            return np.nan

    df["Riktkurs idag"]   = df.apply(lambda r: riktkurs(r.get("Omsättning idag"),     r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)
    df["Riktkurs om 1 år"] = df.apply(lambda r: riktkurs(r.get("Omsättning nästa år"), r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)
    df["Riktkurs om 2 år"] = df.apply(lambda r: riktkurs(r.get("Omsättning om 2 år"),  r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)
    df["Riktkurs om 3 år"] = df.apply(lambda r: riktkurs(r.get("Omsättning om 3 år"),  r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)

    return df

# ==============================
# CAGR-rå från Yahoo (för att spara i DB), separat från Del 2
# ==============================
def hamta_hist_oms_rå_cagr(ticker: str):
    """
    Hämtar årlig historisk omsättning via yfinance och returnerar RÅ CAGR i procent.
    (Ingen 50%/2% justering här – den görs i framåträkningen.)
    """
    try:
        t = yf.Ticker(ticker)
        df_is = None
        for attr in ("income_stmt", "financials"):
            try:
                cand = getattr(t, attr)
                if isinstance(cand, pd.DataFrame) and "Total Revenue" in cand.index and cand.shape[1] >= 2:
                    df_is = cand
                    break
            except:
                pass
        if df_is is None:
            return None

        cols_sorted = sorted(df_is.columns)
        start_col = cols_sorted[0]
        end_col   = cols_sorted[-1]
        oms_start = df_is.loc["Total Revenue", start_col]
        oms_slut  = df_is.loc["Total Revenue", end_col]

        # Antal år
        try:
            years = max(1, (end_col.year - start_col.year))
        except:
            years = max(1, len(cols_sorted) - 1)

        if pd.isna(oms_start) or pd.isna(oms_slut) or float(oms_start) <= 0:
            return None

        cagr_decimal = (float(oms_slut) / float(oms_start)) ** (1.0 / years) - 1.0
        return round(cagr_decimal * 100.0, 2)
    except:
        return None


# ==============================
# Hjälpare: uppdatera EN ticker i df (hämtar auto-fält + RÅ CAGR)
# ==============================
def uppdatera_en_ticker_i_df(df: pd.DataFrame, ticker: str, fel_lista: list):
    t = ticker.strip()
    if not t:
        return df

    try:
        aktie = yf.Ticker(t)
        info = aktie.info if hasattr(aktie, "info") else {}

        bolagsnamn = info.get("longName") or info.get("shortName") or ""
        kurs = info.get("currentPrice")
        valuta = info.get("currency")
        utdelning = info.get("dividendRate")

        # Spara RÅ CAGR (utan justering) i DB
        cagr_rå = hamta_hist_oms_rå_cagr(t)

        mask = df["Ticker"].astype(str).str.upper() == t.upper()
        if not mask.any():
            # finns inte – skapa tom rad med Ticker
            tom = {k: "" for k in df.columns}
            tom["Ticker"] = t
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)
            mask = df["Ticker"].astype(str).str.upper() == t.upper()

        i = df[mask].index[0]
        # Auto-fält överskrivs
        if bolagsnamn != "":
            df.at[i, "Bolagsnamn"] = bolagsnamn
        if kurs is not None:
            df.at[i, "Aktuell kurs"] = kurs
        if valuta:
            df.at[i, "Valuta"] = valuta
        if utdelning is not None:
            df.at[i, "Årlig utdelning"] = utdelning
        if cagr_rå is not None:
            df.at[i, "CAGR 5 år (%)"] = cagr_rå

        # Samla info om fält som inte gick att hämta
        miss = []
        if bolagsnamn == "":
            miss.append("bolagsnamn")
        if kurs is None:
            miss.append("kurs")
        if not valuta:
            miss.append("valuta")
        if utdelning is None:
            miss.append("utdelning")
        if cagr_rå is None:
            miss.append("CAGR")

        if miss:
            fel_lista.append(f"{t}: saknar {', '.join(miss)}")

    except Exception as e:
        fel_lista.append(f"{t}: fel {e}")

    return df


# ==============================
# Kör beräkningar inkl. justerad CAGR för framåträkning
# ==============================
def kör_beräkningar_med_justering(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Se till kolumner finns och numeriska typer
    df = säkerställ_kolumner(df)
    df = beräkna_kolumner(df)  # P/S-snitt + rå CAGR finns redan i DF här

    # 2) Justera framåträkning (50%/2%) för "Omsättning om 2/3 år"
    def proj_just(r):
        cagr_rå = r.get("CAGR 5 år (%)")
        if pd.isna(cagr_rå):
            return None, None
        cagr_j = justera_tillväxt(float(cagr_rå))  # 50%/2%/eller rå
        if cagr_j is None:
            return None, None
        bas = r.get("Omsättning nästa år")
        if pd.isna(bas):
            return None, None
        try:
            oms2 = round(float(bas) * (1.0 + (cagr_j/100.0)) ** 1, 2)
            oms3 = round(float(bas) * (1.0 + (cagr_j/100.0)) ** 2, 2)
            return oms2, oms3
        except:
            return None, None

    res = df.apply(lambda r: proj_just(r), axis=1, result_type="expand")
    if isinstance(res, pd.DataFrame) and res.shape[1] == 2:
        df["Omsättning om 2 år"] = res.iloc[:, 0]
        df["Omsättning om 3 år"] = res.iloc[:, 1]

    # 3) Räkna om riktkurser baserat på uppdaterade projicerade omsättningar
    def riktkurs(oms, ps, aktier):
        try:
            if pd.isna(oms) or pd.isna(ps) or pd.isna(aktier):
                return None
            oms = float(oms); ps = float(ps); aktier = float(aktier)
            if aktier <= 0 or ps <= 0:
                return None
            return round((oms * ps) / aktier, 2)
        except:
            return None

    df["Riktkurs idag"]    = df.apply(lambda r: riktkurs(r.get("Omsättning idag"),     r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)
    df["Riktkurs om 1 år"] = df.apply(lambda r: riktkurs(r.get("Omsättning nästa år"), r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)
    df["Riktkurs om 2 år"] = df.apply(lambda r: riktkurs(r.get("Omsättning om 2 år"),  r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)
    df["Riktkurs om 3 år"] = df.apply(lambda r: riktkurs(r.get("Omsättning om 3 år"),  r.get("P/S-snitt"), r.get("Utestående aktier")), axis=1)

    return df


# ==============================
# Formulär: Lägg till / uppdatera bolag
# ==============================
def formulär(df: pd.DataFrame):
    st.header("➕ Lägg till / uppdatera bolag")

    # Välj befintlig eller nytt
    tickers = [""] + sorted(df["Ticker"].astype(str).unique().tolist())
    vald = st.selectbox("Välj bolag (eller lämna tomt för nytt)", tickers)

    bef = df[df["Ticker"].astype(str) == vald].iloc[0] if vald else pd.Series(dtype="object")

    with st.form("bolagsformulär"):
        ticker = st.text_input("Ticker", value=bef.get("Ticker", "") if not bef.empty else "").strip().upper()

        # Manuella fält (du matar in)
        utest = st.number_input("Utestående aktier", min_value=0.0, value=float(bef.get("Utestående aktier", 0) or 0.0))
        antal = st.number_input("Antal aktier (ägda)", min_value=0.0, value=float(bef.get("Antal aktier", 0) or 0.0))

        ps = st.number_input("P/S", value=float(bef.get("P/S", 0) or 0.0))
        ps1 = st.number_input("P/S Q1", value=float(bef.get("P/S Q1", 0) or 0.0))
        ps2 = st.number_input("P/S Q2", value=float(bef.get("P/S Q2", 0) or 0.0))
        ps3 = st.number_input("P/S Q3", value=float(bef.get("P/S Q3", 0) or 0.0))
        ps4 = st.number_input("P/S Q4", value=float(bef.get("P/S Q4", 0) or 0.0))

        oms_idag  = st.number_input("Omsättning idag", value=float(bef.get("Omsättning idag", 0) or 0.0))
        oms_next  = st.number_input("Omsättning nästa år", value=float(bef.get("Omsättning nästa år", 0) or 0.0))

        col_b = st.columns(3)
        with col_b[0]:
            hämta = st.form_submit_button("🔎 Hämta från Yahoo för denna ticker")
        with col_b[1]:
            spara = st.form_submit_button("💾 Spara")
        with col_b[2]:
            avbryt = st.form_submit_button("Avbryt")

    if avbryt:
        st.stop()

    # Hämta från Yahoo (enskild)
    if hämta:
        if not ticker:
            st.warning("Ange en ticker först.")
            return df
        fel = []
        df = uppdatera_en_ticker_i_df(df, ticker, fel)
        df = kör_beräkningar_med_justering(df)
        spara_data(df)
        st.success("Data hämtad och beräkningar uppdaterade.")
        if fel:
            st.info("Misslyckade/partiella fält (kopiera nedan):")
            st.code("\n".join(fel))
        return df

    # Spara (manuellt + auto-fält om de redan finns)
    if spara:
        if not ticker:
            st.warning("Ange en ticker.")
            return df

        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag, "Omsättning nästa år": oms_next,
        }

        # Behåll befintliga auto-fält om de finns
        if not bef.empty:
            for k in ["Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"]:
                ny[k] = bef.get(k, "")
        else:
            for k in ["Bolagsnamn", "Valuta", "Aktuell kurs", "Årlig utdelning", "CAGR 5 år (%)"]:
                ny[k] = ""

        # Skriv in/uppdatera
        df = df[df["Ticker"].astype(str).str.upper() != ticker.upper()]
        df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)

        # Kör beräkningar
        df = kör_beräkningar_med_justering(df)
        spara_data(df)
        st.success("Bolag sparat och beräkningar uppdaterade.")
        return df

    return df


# ==============================
# Analysvy (tabell + massuppdatering)
# ==============================
def analysvy(df: pd.DataFrame):
    st.header("📈 Analys")

    # Filtrera visning på ett bolag (överst), men visa alltid hela tabellen under
    lista = ["(visa alla)"] + sorted(df["Ticker"].astype(str).unique().tolist())
    val = st.selectbox("Visa ett bolag (förhandsgranskning)", lista, index=0)
    if val != "(visa alla)":
        st.subheader(f"Detaljer för {val}")
        st.dataframe(df[df["Ticker"].astype(str) == val], use_container_width=True)

    st.subheader("Hela databasen")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    if st.button("🔄 Uppdatera ALLA från Yahoo (1 s mellan anrop)"):
        fel = []
        tot = len(df)
        bar = st.progress(0)
        status = st.empty()

        for i, t in enumerate(df["Ticker"].astype(str).tolist(), start=1):
            status.text(f"Hämtar {i}/{tot} — {t}")
            df = uppdatera_en_ticker_i_df(df, t, fel)
            time.sleep(1)
            bar.progress(i / tot)

        df = kör_beräkningar_med_justering(df)
        spara_data(df)
        status.text("✅ Klar")
        st.success("Alla bolag uppdaterade.")

        if fel:
            st.info("Misslyckade/partiella fält (kopiera listan):")
            st.code("\n".join(fel))


# ==============================
# Investeringsförslag
# ==============================
def investeringsförslag(df: pd.DataFrame):
    st.header("💡 Investeringsförslag")

    val = st.selectbox(
        "Baserat på riktkurs …",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    # Potentialsiffror
    df = df.copy()
    df["Aktuell kurs"] = pd.to_numeric(df["Aktuell kurs"], errors="coerce")
    df[val] = pd.to_numeric(df[val], errors="coerce")
    df["Uppside (%)"] = ((df[val] - df["Aktuell kurs"]) / df["Aktuell kurs"]) * 100.0
    df = df.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=["Aktuell kurs", val, "Uppside (%)"])
    df = df.sort_values("Uppside (%)", ascending=False).reset_index(drop=True)

    if df.empty:
        st.info("Inga förslag – saknar värden för vald riktkurs.")
        return

    if "forslag_idx" not in st.session_state:
        st.session_state.forslag_idx = 0

    col_nav = st.columns(3)
    with col_nav[0]:
        if st.button("⬅️ Föregående") and st.session_state.forslag_idx > 0:
            st.session_state.forslag_idx -= 1
    with col_nav[2]:
        if st.button("Nästa ➡️") and st.session_state.forslag_idx < len(df) - 1:
            st.session_state.forslag_idx += 1

    rad = df.iloc[st.session_state.forslag_idx]

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.write(f"Aktuell kurs: **{rad['Aktuell kurs']} {rad.get('Valuta','')}**")
    st.write(f"{val}: **{rad[val]} {rad.get('Valuta','')}**")
    st.write(f"Uppsida (valet ovan): **{rad['Uppside (%)']:.2f}%**")

    kapital = st.number_input("Tillgängligt belopp (samma valuta som aktuell kurs)", min_value=0.0, value=0.0, step=100.0)
    if kapital > 0 and rad["Aktuell kurs"] and rad["Aktuell kurs"] > 0:
        antal = int(kapital // float(rad["Aktuell kurs"]))
        st.write(f"Förslag: köp **{antal}** st")


# ==============================
# Portfölj
# ==============================
def portfölj(df: pd.DataFrame):
    st.header("📦 Portfölj")
    d = df.copy()
    d["Antal aktier"] = pd.to_numeric(d["Antal aktier"], errors="coerce").fillna(0.0)
    d["Aktuell kurs"] = pd.to_numeric(d["Aktuell kurs"], errors="coerce").fillna(0.0)

    ägda = d[d["Antal aktier"] > 0].copy()
    if ägda.empty:
        st.info("Inga innehav registrerade ännu.")
        return

    ägda["Värde"] = (ägda["Antal aktier"] * ägda["Aktuell kurs"]).round(2)
    ägda["Årlig utdelning"] = pd.to_numeric(ägda["Årlig utdelning"], errors="coerce").fillna(0.0)
    ägda["Utdelning/år"] = (ägda["Antal aktier"] * ägda["Årlig utdelning"]).round(2)

    tot_värde = float(ägda["Värde"].sum())
    tot_utd = float(ägda["Utdelning/år"].sum())
    mån = tot_utd / 12.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Totalt portföljvärde", f"{tot_värde:,.0f}")
    c2.metric("Total årlig utdelning", f"{tot_utd:,.0f}")
    c3.metric("Utdelning per månad (snitt)", f"{mån:,.0f}")

    st.dataframe(ägda[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Värde","Årlig utdelning","Utdelning/år"]], use_container_width=True)


# ==============================
# MAIN
# ==============================
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Läs data
    df = hamta_data()
    df = säkerställ_kolumner(df)

    # Kör bas-beräkningar (P/S-snitt, riktkurser mm) och justerad framåträkning
    df = kör_beräkningar_med_justering(df)

    meny = st.sidebar.radio("Meny", ["Analys", "Lägg till / uppdatera", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df)
    elif meny == "Lägg till / uppdatera":
        df = formulär(df)
    elif meny == "Investeringsförslag":
        investeringsförslag(df)
    elif meny == "Portfölj":
        portfölj(df)

if __name__ == "__main__":
    main()
