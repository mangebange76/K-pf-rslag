import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError, WorksheetNotFound

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---- Google Sheets ----
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ---- Standard-valutakurser (SEK) ----
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0
}

# =========================================================
#  Google Sheets: stabil anslutning (retry), cache & I/O
# =========================================================
def skapa_koppling(max_retries: int = 5, backoff_start: float = 0.6):
    """Öppnar kalkylarket och 'Blad1' med retry/backoff och cachar worksheet i session_state."""
    if "worksheet_obj" in st.session_state and st.session_state["worksheet_obj"] is not None:
        return st.session_state["worksheet_obj"]

    last_err = None
    delay = backoff_start
    for _ in range(max_retries):
        try:
            sh = client.open_by_url(SHEET_URL)
            ws = sh.worksheet(SHEET_NAME)
            st.session_state["worksheet_obj"] = ws
            return ws
        except WorksheetNotFound as e:
            last_err = e
            break
        except APIError as e:
            last_err = e
            time.sleep(delay)
            delay *= 1.7
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 1.7

    st.error(
        "Kunde inte ansluta till Google Sheet. "
        f"Kontrollera åtkomst och att bladet heter **{SHEET_NAME}**."
    )
    if last_err:
        st.caption(f"Teknisk info: {type(last_err).__name__}")
    return None


def _hamta_data_fran_google(max_retries: int = 5, backoff_start: float = 0.6):
    ws = skapa_koppling()
    if ws is None:
        return None
    last_err = None
    delay = backoff_start
    for _ in range(max_retries):
        try:
            data = ws.get_all_records()
            return pd.DataFrame(data)
        except APIError as e:
            last_err = e
            time.sleep(delay)
            delay *= 1.7
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 1.7
    return None


def hamta_data(force_reload: bool = False):
    """
    Hämtar data EN gång per session (eller när du trycker “Ladda om från Google”).
    - Använder session_state['df_cache'] om den finns och vi inte tvingar omladdning.
    - Vid Google-fel används senast fungerande data ('last_df_ok') om sådan finns.
    """
    if not force_reload and "df_cache" in st.session_state and st.session_state["df_cache"] is not None:
        return st.session_state["df_cache"].copy()

    df = _hamta_data_fran_google()
    if df is None or df.empty:
        if "last_df_ok" in st.session_state and st.session_state["last_df_ok"] is not None:
            st.warning("Kunde inte läsa från Google just nu – visar senaste inlästa data.")
            st.session_state["df_cache"] = st.session_state["last_df_ok"].copy()
            return st.session_state["df_cache"].copy()
        else:
            st.error("Kunde inte läsa från Google och ingen tidigare data finns i minnet.")
            return pd.DataFrame()

    df = säkerställ_kolumner(df)
    st.session_state["last_df_ok"] = df.copy()
    st.session_state["df_cache"] = df.copy()
    return df.copy()


def spara_data(df):
    ws = skapa_koppling()
    if ws is None:
        st.error("Kan inte spara – ingen anslutning till Google Sheet.")
        return
    ws.clear()
    ws.update([df.columns.values.tolist()] + df.astype(str).values.tolist())
    st.session_state["last_df_ok"] = df.copy()
    st.session_state["df_cache"] = df.copy()

# =========================
#   Hjälpfunktioner & logik
# =========================
ALLA_VALUTOR = ["USD", "NOK", "CAD", "SEK", "EUR"]

def konvertera_typer(df):
    kol_float = [
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Aktuell kurs", "Antal aktier", "Årlig utdelning", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for kol in kol_float:
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors="coerce").fillna(0.0)

    # Säkerställ text-kolumner
    for kol in ["Ticker", "Bolagsnamn", "Valuta", "Omsättningsvaluta", "Senast manuell uppdatering"]:
        if kol in df.columns:
            df[kol] = df[kol].astype(str).fillna("")

    return df


def säkerställ_kolumner(df):
    nödvändiga = [
        # manuellt + auto
        "Ticker", "Bolagsnamn",
        "Utestående aktier",
        "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
        "CAGR 5 år (%)", "P/S-snitt",
        # nya
        "Omsättningsvaluta", "Senast manuell uppdatering"
    ]
    for kol in nödvändiga:
        if kol not in df.columns:
            if kol in ["Ticker", "Bolagsnamn", "Valuta", "Omsättningsvaluta", "Senast manuell uppdatering"]:
                df[kol] = ""
            else:
                df[kol] = 0.0
    return df


def hamta_valutakurs_sek_map():
    # Läser från sidopanelns inmatningar (fallback till standard)
    return {
        "USD": float(st.session_state.get("rate_USD", STANDARD_VALUTAKURSER["USD"])),
        "NOK": float(st.session_state.get("rate_NOK", STANDARD_VALUTAKURSER["NOK"])),
        "CAD": float(st.session_state.get("rate_CAD", STANDARD_VALUTAKURSER["CAD"])),
        "EUR": float(st.session_state.get("rate_EUR", STANDARD_VALUTAKURSER["EUR"])),
        "SEK": 1.0
    }


def konvertera_belopp(belopp: float, fran: str, till: str, rates_sek: dict) -> float:
    """
    Konvertera belopp från 'från-valuta' till 'till-valuta' via SEK.
    rates_sek: t.ex. {"USD": 9.75, ...} (= hur många SEK per 1 enhet valuta)
    """
    if belopp == 0 or fran == till:
        return belopp
    fran_rate = rates_sek.get(fran.upper(), 1.0)
    till_rate = rates_sek.get(till.upper(), 1.0)
    if fran_rate <= 0 or till_rate <= 0:
        return belopp
    # belopp_fran -> SEK -> till
    sek = belopp * fran_rate
    return sek / till_rate


def beräkna_cagr_från_yahoo(ticker: str) -> float:
    """
    Försöker beräkna CAGR 5 år från Yahoo (års-data: Total Revenue).
    Om inte möjligt returneras 0.0
    """
    try:
        t = yf.Ticker(ticker)
        # income_stmt (annual) – kan variera i yfinance versioner:
        # Prova .income_stmt (nyare) och .financials (äldre) och läs 'Total Revenue'
        df_income = None
        try:
            df_income = t.income_stmt
        except Exception:
            pass
        if df_income is None or df_income.empty:
            try:
                df_income = t.financials
            except Exception:
                df_income = None
        if df_income is None or df_income.empty:
            return 0.0

        # Normalisera index/kolumner
        df_income.index = [str(x) for x in df_income.index]
        rev_key_candidates = [k for k in df_income.index if "Total Revenue" in k or "TotalRevenue" in k or "totalRevenue" in k]
        if not rev_key_candidates:
            return 0.0
        rev_key = rev_key_candidates[0]

        series = df_income.loc[rev_key]
        series = series.dropna().sort_index()  # äldst->nyast
        if len(series) < 5:
            return 0.0

        # Ta de senaste 6 åren om möjligt, räkna CAGR mellan år-5 och år-1 (5 intervall)
        vals = series.values.astype(float)
        first = float(vals[-6]) if len(vals) >= 6 else float(vals[0])
        last = float(vals[-1])
        if first <= 0 or last <= 0:
            return 0.0
        years = 5
        cagr = (last / first) ** (1/years) - 1
        return round(cagr * 100.0, 2)  # procent
    except Exception:
        return 0.0


def uppdatera_oms_prognos_med_cagr(df):
    """
    Använder 'CAGR 5 år (%)' (med tak/golv) för att räkna fram:
    - Omsättning om 2 år
    - Omsättning om 3 år
    utifrån 'Omsättning nästa år'.
    Tar hänsyn till 'Omsättningsvaluta' och räknar riktkurser i aktiens 'Valuta'.
    (Själva prognos-OMS sparas i miljoner i omsättningsvalutan – vi konverterar först
     när vi räknar riktkurser.)
    """
    rates = hamta_valutakurs_sek_map()
    for i, rad in df.iterrows():
        # Hämtad / tidigare lagrad CAGR
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        # tak/golv
        if cagr > 100.0:
            eff = 50.0
        elif cagr < 0.0:
            eff = 2.0
        else:
            eff = cagr

        oms_na = float(rad.get("Omsättning nästa år", 0.0))
        # räkna fram (i OMS-värdets angivna valuta, miljoner)
        oms_2 = oms_na * (1 + eff/100.0)
        oms_3 = oms_2 * (1 + eff/100.0)

        df.at[i, "Omsättning om 2 år"] = round(oms_2, 2)
        df.at[i, "Omsättning om 3 år"] = round(oms_3, 2)
    return df


def uppdatera_berakningar(df):
    """
    Räknar P/S-snitt och riktkurser. Riktkurser i aktiens VALUTA.
    Om omsättningsvalutan skiljer sig från aktiens valuta konverteras omsättningsbeloppen.
    """
    rates = hamta_valutakurs_sek_map()

    for i, rad in df.iterrows():
        ps = [rad.get("P/S Q1", 0), rad.get("P/S Q2", 0), rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)]
        ps = [x for x in ps if x > 0]
        ps_snitt = round(np.mean(ps), 2) if ps else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        utst = float(rad.get("Utestående aktier", 0.0))
        aktie_val = (rad.get("Valuta") or "USD").strip().upper()
        oms_val = (rad.get("Omsättningsvaluta") or aktie_val).strip().upper()

        # Konvertera omsättning -> aktiens valuta (miljoner)
        def conv(miljon):
            return konvertera_belopp(float(miljon), oms_val, aktie_val, rates)

        oms_idag = conv(rad.get("Omsättning idag", 0.0))
        oms_1   = conv(rad.get("Omsättning nästa år", 0.0))
        oms_2   = conv(rad.get("Omsättning om 2 år", 0.0))
        oms_3   = conv(rad.get("Omsättning om 3 år", 0.0))

        if utst > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((oms_idag * ps_snitt) / utst, 2)
            df.at[i, "Riktkurs om 1 år"] = round((oms_1   * ps_snitt) / utst, 2)
            df.at[i, "Riktkurs om 2 år"] = round((oms_2   * ps_snitt) / utst, 2)
            df.at[i, "Riktkurs om 3 år"] = round((oms_3   * ps_snitt) / utst, 2)
        else:
            df.at[i, "Riktkurs idag"] = df.at[i, "Riktkurs om 1 år"] = df.at[i, "Riktkurs om 2 år"] = df.at[i, "Riktkurs om 3 år"] = 0.0

    return df


def hamta_kurs_valuta_namn(ticker):
    try:
        info = yf.Ticker(ticker).info
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        namn = info.get("shortName") or info.get("longName") or ""
        return pris, valuta, namn
    except Exception:
        return None, "USD", ""

# =========================
#   Vyer
# =========================
def lagg_till_eller_uppdatera(df):
    st.subheader("➕ Lägg till / uppdatera bolag")

    # Snabb-kontroller
    c0, c1 = st.columns([1,1])
    with c0:
        if st.button("🔄 Ladda om från Google", key="reload_from_google_form"):
            st.session_state.pop("df_cache", None)
            _ = hamta_data(force_reload=True)
            st.success("Data laddad från Google.")
            st.rerun()
    with c1:
        if st.button("🌐 Uppdatera alla kurser (Yahoo)", key="yahoo_all_form"):
            massuppdatera_kurser(df)  # definieras nedan
            st.rerun()

    # Rullista + Föregående/Nästa
    df = df.reset_index(drop=True)
    etiketter = [f"{row['Bolagsnamn']} ({row['Ticker']})" if str(row['Bolagsnamn']).strip() else row['Ticker'] for _, row in df.iterrows()]
    etiketter_vis = ["— nytt bolag —"] + etiketter

    # Håll koll på vilket index som är valt/bläddras
    if "form_bolag_index" not in st.session_state:
        st.session_state.form_bolag_index = 0  # 0 betyder "nytt bolag", annars 1..len

    valt_label = st.selectbox("Välj bolag", etiketter_vis, index=st.session_state.form_bolag_index)
    if valt_label == "— nytt bolag —":
        befintlig = pd.Series(dtype=object)
        current_i = None
        total = len(df)
        pos_text = "–/{}".format(total)
    else:
        current_i = etiketter.index(valt_label)
        befintlig = df.iloc[current_i]
        total = len(df)
        pos_text = f"{current_i+1}/{total}"

    cprev, cpos, cnext = st.columns([1,2,1])
    with cprev:
        if st.button("⬅️ Föregående", use_container_width=True):
            if current_i is None:
                st.session_state.form_bolag_index = 1 if len(etiketter) > 0 else 0
            else:
                ny = 1 + ((current_i - 1) % len(etiketter))
                st.session_state.form_bolag_index = ny
            st.rerun()
    with cpos:
        st.info(f"Post: {pos_text}")
    with cnext:
        if st.button("➡️ Nästa", use_container_width=True):
            if current_i is None:
                st.session_state.form_bolag_index = 1 if len(etiketter) > 0 else 0
            else:
                ny = 1 + ((current_i + 1) % len(etiketter))
                st.session_state.form_bolag_index = ny
            st.rerun()

    with st.form("form_lagg_till"):
        # Manuella fält
        ticker = st.text_input("Ticker", value=str(befintlig.get("Ticker", "")) if not befintlig.empty else "").upper()
        utest = st.number_input("Utestående aktier (miljoner)", value=float(befintlig.get("Utestående aktier", 0.0)) if not befintlig.empty else 0.0)

        ps_idag = st.number_input("P/S", value=float(befintlig.get("P/S", 0.0)) if not befintlig.empty else 0.0)
        ps1 = st.number_input("P/S Q1", value=float(befintlig.get("P/S Q1", 0.0)) if not befintlig.empty else 0.0)
        ps2 = st.number_input("P/S Q2", value=float(befintlig.get("P/S Q2", 0.0)) if not befintlig.empty else 0.0)
        ps3 = st.number_input("P/S Q3", value=float(befintlig.get("P/S Q3", 0.0)) if not befintlig.empty else 0.0)
        ps4 = st.number_input("P/S Q4", value=float(befintlig.get("P/S Q4", 0.0)) if not befintlig.empty else 0.0)

        oms_val = st.selectbox("Omsättningsvaluta (för omsättningsfält)", ALLA_VALUTOR,
                               index=(ALLA_VALUTOR.index(befintlig.get("Omsättningsvaluta", "USD")) if not befintlig.empty else ALLA_VALUTOR.index("USD")))
        oms_idag = st.number_input("Omsättning idag (miljoner)", value=float(befintlig.get("Omsättning idag", 0.0)) if not befintlig.empty else 0.0)
        oms_1 = st.number_input("Omsättning nästa år (miljoner)", value=float(befintlig.get("Omsättning nästa år", 0.0)) if not befintlig.empty else 0.0)

        antal_aktier = st.number_input("Antal aktier du äger", value=float(befintlig.get("Antal aktier", 0.0)) if not befintlig.empty else 0.0)

        sparaknapp = st.form_submit_button("💾 Spara & hämta namn/kurs/valuta (Yahoo)")

    if sparaknapp:
        if not ticker:
            st.error("Ange minst en Ticker.")
            return df

        pris, valuta, namn = hamta_kurs_valuta_namn(ticker)
        if pris is None:
            st.warning(f"Kunde inte hämta kurs för {ticker}. Kurs/valuta/namn lämnas oförändrat om de finns.")
            pris = float(befintlig.get("Aktuell kurs", 0.0)) if not befintlig.empty else 0.0
            valuta = (befintlig.get("Valuta", "USD") if not befintlig.empty else "USD") or "USD"
            namn = (befintlig.get("Bolagsnamn", "") if not befintlig.empty else "")

        # Hämta/uppdatera CAGR 5 år (%) (kan vara 0 om det inte går att räkna)
        cagr5 = beräkna_cagr_från_yahoo(ticker)

        ny_rad = {
            "Ticker": ticker,
            "Bolagsnamn": namn,
            "Utestående aktier": utest,
            "P/S": ps_idag,
            "P/S Q1": ps1,
            "P/S Q2": ps2,
            "P/S Q3": ps3,
            "P/S Q4": ps4,
            "Omsättningsvaluta": oms_val,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_1,
            # prognos fylls efter CAGR-uppdatering
            "Omsättning om 2 år": float(befintlig.get("Omsättning om 2 år", 0.0)) if not befintlig.empty else 0.0,
            "Omsättning om 3 år": float(befintlig.get("Omsättning om 3 år", 0.0)) if not befintlig.empty else 0.0,
            # riktkurser fylls efter beräkning
            "Riktkurs idag": float(befintlig.get("Riktkurs idag", 0.0)) if not befintlig.empty else 0.0,
            "Riktkurs om 1 år": float(befintlig.get("Riktkurs om 1 år", 0.0)) if not befintlig.empty else 0.0,
            "Riktkurs om 2 år": float(befintlig.get("Riktkurs om 2 år", 0.0)) if not befintlig.empty else 0.0,
            "Riktkurs om 3 år": float(befintlig.get("Riktkurs om 3 år", 0.0)) if not befintlig.empty else 0.0,
            "Antal aktier": antal_aktier,
            "Valuta": valuta,
            "Årlig utdelning": float(befintlig.get("Årlig utdelning", 0.0)) if not befintlig.empty else 0.0,
            "Aktuell kurs": round(float(pris or 0.0), 2),
            "CAGR 5 år (%)": cagr5,
            "P/S-snitt": float(befintlig.get("P/S-snitt", 0.0)) if not befintlig.empty else 0.0,
            "Senast manuell uppdatering": datetime.now().strftime("%Y-%m-%d %H:%M")
        }

        if ticker in df["Ticker"].values:
            df.loc[df["Ticker"] == ticker, ny_rad.keys()] = ny_rad.values()
            st.success(f"{ticker} uppdaterat.")
        else:
            df = pd.concat([df, pd.DataFrame([ny_rad])], ignore_index=True)
            st.success(f"{ticker} tillagt.")

        # Uppdatera prognos med CAGR (tak/golv), sedan riktkurser
        df = uppdatera_oms_prognos_med_cagr(df)
        df = uppdatera_berakningar(df)
        spara_data(df)

        # Synka rullistans index till ny/uppdaterad post
        if ticker in df["Ticker"].values:
            idx = df.index[df["Ticker"] == ticker][0]
            st.session_state.form_bolag_index = 1 + idx  # +1 pga "— nytt bolag —"
        st.rerun()

    return df


def massuppdatera_kurser(df):
    misslyckade = []
    uppdaterade = 0
    total = len(df)
    status = st.empty()
    bar = st.progress(0)

    with st.spinner("Uppdaterar kurser..."):
        for i, row in df.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            status.text(f"Uppdaterar {i + 1} av {total} ({ticker})...")
            try:
                pris, valuta, namn = hamta_kurs_valuta_namn(ticker)
                if pris is None:
                    misslyckade.append(ticker)
                else:
                    df.at[i, "Aktuell kurs"] = round(float(pris), 2)
                    df.at[i, "Valuta"] = valuta
                    if namn:
                        df.at[i, "Bolagsnamn"] = namn
                    uppdaterade += 1
            except Exception:
                misslyckade.append(ticker)

            bar.progress((i + 1) / total)
            time.sleep(1.0)

    # Efter uppdatering av kurser kan vi uppdatera beräkningar
    df = uppdatera_oms_prognos_med_cagr(df)
    df = uppdatera_berakningar(df)
    spara_data(df)

    status.text("✅ Uppdatering klar.")
    st.success(f"{uppdaterade} tickers uppdaterade.")
    if misslyckade:
        st.warning("Kunde inte uppdatera följande tickers:\n" + ", ".join(misslyckade))


def visa_portfolj(df, valutakurser):
    st.subheader("📦 Min portfölj")
    dfv = df[df["Antal aktier"] > 0].copy()
    if dfv.empty:
        st.info("Du äger inga aktier.")
        return

    dfv["Växelkurs"] = dfv["Valuta"].map(valutakurser).fillna(1.0)
    dfv["Värde (SEK)"] = dfv["Antal aktier"] * dfv["Aktuell kurs"] * dfv["Växelkurs"]
    dfv["Andel (%)"] = round(dfv["Värde (SEK)"] / dfv["Värde (SEK)"].sum() * 100, 2)

    dfv["Total årlig utdelning"] = dfv["Antal aktier"] * dfv["Årlig utdelning"] * dfv["Växelkurs"]
    total_utdelning = dfv["Total årlig utdelning"].sum()
    total_varde = dfv["Värde (SEK)"].sum()

    st.markdown(f"**Totalt portföljvärde:** {round(total_varde, 2)} SEK")
    st.markdown(f"**Förväntad årlig utdelning:** {round(total_utdelning, 2)} SEK")
    st.markdown(f"**Genomsnittlig månadsutdelning:** {round(total_utdelning / 12, 2)} SEK")
    st.divider()

    st.dataframe(
        dfv[["Ticker", "Bolagsnamn", "Antal aktier", "Aktuell kurs", "Valuta", "Värde (SEK)", "Andel (%)", "Årlig utdelning", "Total årlig utdelning"]],
        use_container_width=True
    )

def visa_investeringsforslag(df, valutakurser):
    st.subheader("💡 Investeringsförslag")
    kapital_sek = st.number_input("Tillgängligt kapital (SEK)", value=500.0, step=500.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )
    läge = st.radio("Sortering", ["Störst uppsida", "Närmast riktkurs"], horizontal=True)
    visa_endast_portfolj = st.checkbox("Visa endast portföljens innehav", value=False)

    df["Växelkurs"] = df["Valuta"].map(valutakurser).fillna(1.0)
    df_portf = df[df["Antal aktier"] > 0].copy()
    df_portf["Värde (SEK)"] = df_portf["Antal aktier"] * df_portf["Aktuell kurs"] * df_portf["Växelkurs"]
    portfoljvarde = df_portf["Värde (SEK)"].sum()

    data = df_portf if visa_endast_portfolj else df
    data = data.copy()
    data = data[(data[riktkurs_val] > 0) & (data["Aktuell kurs"] > 0)]

    if data.empty:
        st.info("Inga bolag matchar kriterierna just nu.")
        return

    if läge == "Störst uppsida":
        data["Metric (%)"] = ((data[riktkurs_val] - data["Aktuell kurs"]) / data["Aktuell kurs"]) * 100
        data = data.sort_values(by="Metric (%)", ascending=False).reset_index(drop=True)
    else:
        # Närmast riktkurs (absolut %-avvikelse – både under & över)
        data["Metric (%)"] = (abs(data[riktkurs_val] - data["Aktuell kurs"]) / data["Aktuell kurs"]) * 100
        data = data.sort_values(by="Metric (%)", ascending=True).reset_index(drop=True)

    key_index = f"forslags_index_{'portf' if visa_endast_portfolj else 'alla'}_{riktkurs_val}_{läge}"
    if key_index not in st.session_state:
        st.session_state[key_index] = 0

    index = st.session_state[key_index] % len(data)
    rad = data.iloc[index]

    kurs_sek = rad["Aktuell kurs"] * rad["Växelkurs"]
    antal_kop = int(kapital_sek // kurs_sek) if kurs_sek > 0 else 0
    investering_sek = antal_kop * kurs_sek

    nuv_innehav = df_portf[df_portf["Ticker"] == rad["Ticker"]]["Värde (SEK)"].sum()
    ny_total = nuv_innehav + investering_sek
    nuv_andel = round((nuv_innehav / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0
    ny_andel = round((ny_total / portfoljvarde) * 100, 2) if portfoljvarde > 0 else 0

    # Visa alla riktkurser – fetmarkera den valda
    def fetta(label):  # helper för fetmarkering
        return f"**{label}**" if label == riktkurs_val else label

    uppsida_vald = ((rad[riktkurs_val] - rad["Aktuell kurs"]) / rad["Aktuell kurs"]) * 100

    st.markdown(f"**Förslag {index+1} av {len(data)}**")
    st.markdown(f"### {rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"- **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {rad['Valuta']}\n"
        f"- {fetta('Riktkurs idag')}: {round(rad['Riktkurs idag'], 2)} {rad['Valuta']}\n"
        f"- {fetta('Riktkurs om 1 år')}: {round(rad['Riktkurs om 1 år'], 2)} {rad['Valuta']}\n"
        f"- {fetta('Riktkurs om 2 år')}: {round(rad['Riktkurs om 2 år'], 2)} {rad['Valuta']}\n"
        f"- {fetta('Riktkurs om 3 år')}: {round(rad['Riktkurs om 3 år'], 2)} {rad['Valuta']}\n"
        f"- **{('Potential' if läge=='Störst uppsida' else 'Avvikelse')} ({riktkurs_val}):** {round(uppsida_vald, 2)}%"
    )

    st.markdown(
        f"- **Antal att köpa:** {antal_kop} st\n"
        f"- **Beräknad investering:** {round(investering_sek, 2)} SEK\n"
        f"- **Nuvarande andel i portföljen:** {nuv_andel}%\n"
        f"- **Andel efter köp:** {ny_andel}%"
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Föregående", key=f"prev_{key_index}", use_container_width=True):
            st.session_state[key_index] = (index - 1) % len(data)
            st.rerun()
    with c2:
        if st.button("➡️ Nästa", key=f"next_{key_index}", use_container_width=True):
            st.session_state[key_index] = (index + 1) % len(data)
            st.rerun()


def analysvy(df, valutakurser):
    st.subheader("📈 Analysläge")

    with st.expander("Uppdatera data"):
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔄 Ladda om från Google", key="reload_from_google_analysis"):
                st.session_state.pop("df_cache", None)
                _ = hamta_data(force_reload=True)
                st.success("Data laddad från Google.")
                st.rerun()
        with c2:
            if st.button("🌐 Uppdatera alla kurser (Yahoo)", key="yahoo_all_analysis"):
                massuppdatera_kurser(df)
                st.rerun()
        with c3:
            if st.button("♻️ Re-beräkna riktkurser (CAGR/valuta)", key="recalc_analysis"):
                df = uppdatera_oms_prognos_med_cagr(df)
                df = uppdatera_berakningar(df)
                spara_data(df)
                st.success("Beräkningar uppdaterade.")
                st.rerun()

    st.dataframe(df, use_container_width=True)


# =========================
#   Huvudprogram
# =========================
def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Valutakurser i sidopanel (används bl.a. i portfölj och investeringsförslag)
    st.sidebar.header("💱 Valutakurser till SEK")
    st.session_state["rate_USD"] = st.sidebar.number_input("USD → SEK", value=float(STANDARD_VALUTAKURSER["USD"]), step=0.01)
    st.session_state["rate_NOK"] = st.sidebar.number_input("NOK → SEK", value=float(STANDARD_VALUTAKURSER["NOK"]), step=0.01)
    st.session_state["rate_CAD"] = st.sidebar.number_input("CAD → SEK", value=float(STANDARD_VALUTAKURSER["CAD"]), step=0.01)
    st.session_state["rate_EUR"] = st.sidebar.number_input("EUR → SEK", value=float(STANDARD_VALUTAKURSER["EUR"]), step=0.01)

    with st.sidebar.expander("🔄 Google Sheet"):
        if st.button("Ladda om från Google", key="reload_sidebar"):
            st.session_state.pop("df_cache", None)
            _ = hamta_data(force_reload=True)
            st.success("Data laddad från Google.")
            st.rerun()

    with st.sidebar.expander("🌐 Yahoo"):
        if st.button("Uppdatera ALLA kurser", key="yahoo_all_sidebar"):
            df_tmp = hamta_data()
            massuppdatera_kurser(df_tmp)
            st.rerun()

    df = hamta_data(force_reload=False)
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df, hamta_valutakurs_sek_map())
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df)
        # sparas i funktionen
    elif meny == "Investeringsförslag":
        # säkerställ beräkningar
        df = uppdatera_oms_prognos_med_cagr(df)
        df = uppdatera_berakningar(df)
        visa_investeringsforslag(df, hamta_valutakurs_sek_map())
    elif meny == "Portfölj":
        df = uppdatera_oms_prognos_med_cagr(df)
        df = uppdatera_berakningar(df)
        visa_portfolj(df, hamta_valutakurs_sek_map())


if __name__ == "__main__":
    main()
