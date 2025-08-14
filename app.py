import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from gspread.exceptions import APIError, WorksheetNotFound
from google.oauth2.service_account import Credentials

# ====== App-setup ======
st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# Läs från secrets (ändra inte din secrets-konfig)
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# ====== Valutakurser (default till SEK) ======
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0
}

# ====== Kolumnordning/definition (enligt dina specifikationer) ======
KOLUMNER = [
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
    "P/S-snitt",
    "Senast manuell uppdatering"  # nytt fält för att följa manuell uppdatering
]

# =====================================================================================
# Google Sheets: stabil anslutning (retry), cache och säkra I/O
# =====================================================================================

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

@st.cache_data(ttl=30, show_spinner=False)
def _hamta_data_cached():
    ws = skapa_koppling()
    if ws is None:
        return None
    try:
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except APIError:
        return None

def hamta_data():
    """Hämta DF med cache + fallback till senast fungerande i sessionen."""
    df = _hamta_data_cached()
    if df is None or df.empty:
        if "last_df_ok" in st.session_state and st.session_state["last_df_ok"] is not None:
            st.warning("Kunde inte läsa från Google just nu – visar senaste inlästa data.")
            return st.session_state["last_df_ok"].copy()
        else:
            st.error("Kunde inte läsa från Google och ingen tidigare data finns i minnet.")
            return pd.DataFrame()
    # Säkerställ kolumner
    df = säkerställ_kolumner(df)
    st.session_state["last_df_ok"] = df.copy()
    return df

def spara_data(df: pd.DataFrame, max_retries: int = 5, backoff_start: float = 0.6):
    ws = skapa_koppling()
    if ws is None:
        st.error("Ingen anslutning till arket – kan inte spara just nu.")
        return

    # Säkerställ kolumnordning innan skrivning
    for col in KOLUMNER:
        if col not in df.columns:
            df[col] = "" if col in ["Ticker","Bolagsnamn","Valuta","Senast manuell uppdatering"] else 0.0
    df = df[KOLUMNER]

    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    last_err = None
    delay = backoff_start
    for _ in range(max_retries):
        try:
            ws.clear()
            ws.update(values)
            st.session_state["last_df_ok"] = df.copy()
            _hamta_data_cached.clear()  # bust cache
            return
        except APIError as e:
            last_err = e
            time.sleep(delay)
            delay *= 1.7
        except Exception as e:
            last_err = e
            time.sleep(delay)
            delay *= 1.7

    st.error("Kunde inte spara till Google Sheet. Försök igen om en stund.")
    if last_err:
        st.caption(f"Teknisk info: {type(last_err).__name__}")

# =====================================================================================
# Säkerställ kolumner, typkonvertering & beräkningshjälp
# =====================================================================================

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    # Lägg till saknade kolumner
    for col in KOLUMNER:
        if col not in df.columns:
            if col in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuell uppdatering"]:
                df[col] = ""
            else:
                df[col] = 0.0
    # Ta bort kolumner som inte längre används
    cols_to_keep = set(KOLUMNER)
    df = df[[c for c in df.columns if c in cols_to_keep]]
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # Strängkolumner
    for col in ["Ticker", "Bolagsnamn", "Valuta", "Senast manuell uppdatering"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def beräkna_ps_snitt(row: pd.Series) -> float:
    vals = [row.get("P/S Q1", 0), row.get("P/S Q2", 0), row.get("P/S Q3", 0), row.get("P/S Q4", 0)]
    vals = [v for v in vals if float(v) > 0]
    if not vals:
        return 0.0
    return float(np.mean(vals))

def clamp_cagr_to_forward_rules(cagr_pct: float) -> float:
    """
    Tillämpa regler för framåträkning:
    - Om CAGR > 100% → använd 50%
    - Om CAGR < 0% → använd 2% (inflationsgolv)
    - Annars använd CAGR som den är
    """
    if cagr_pct > 100.0:
        return 50.0
    if cagr_pct < 0.0:
        return 2.0
    return cagr_pct

def framåträkna_omsättning(oms_bas: float, cagr_pct: float, år_från_bas: int) -> float:
    """
    Räkna fram omsättning år 2/3 baserat på 'Omsättning nästa år' (bas) och CAGR-reglerna.
    """
    eff_cagr = clamp_cagr_to_forward_rules(cagr_pct)
    factor = (1.0 + eff_cagr / 100.0) ** (år_från_bas - 1)  # år 2 -> ^1, år 3 -> ^2
    return float(oms_bas) * factor

def uppdatera_berakningar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # P/S-snitt
    df["P/S-snitt"] = df.apply(beräkna_ps_snitt, axis=1)

    # Omsättning om 2 & 3 år utifrån CAGR och "Omsättning nästa år"
    def _calc_oms2(row):
        cagr = float(row.get("CAGR 5 år (%)", 0.0))
        bas = float(row.get("Omsättning nästa år", 0.0))
        if bas > 0 and cagr != 0:
            return round(framåträkna_omsättning(bas, cagr, 2), 2)
        return float(row.get("Omsättning om 2 år", 0.0))

    def _calc_oms3(row):
        cagr = float(row.get("CAGR 5 år (%)", 0.0))
        bas = float(row.get("Omsättning nästa år", 0.0))
        if bas > 0 and cagr != 0:
            return round(framåträkna_omsättning(bas, cagr, 3), 2)
        return float(row.get("Omsättning om 3 år", 0.0))

    df["Omsättning om 2 år"] = df.apply(_calc_oms2, axis=1)
    df["Omsättning om 3 år"] = df.apply(_calc_oms3, axis=1)

    # Riktkurser (i aktiens egen valuta) baserat på P/S-snitt
    def _rk(oms, ps, shares):
        if shares and shares > 0 and ps and ps > 0 and oms and oms > 0:
            return round((float(oms) * float(ps)) / float(shares), 2)
        return 0.0

    df["Riktkurs idag"]    = df.apply(lambda r: _rk(r["Omsättning idag"],     r["P/S-snitt"], r["Utestående aktier"]), axis=1)
    df["Riktkurs om 1 år"] = df.apply(lambda r: _rk(r["Omsättning nästa år"], r["P/S-snitt"], r["Utestående aktier"]), axis=1)
    df["Riktkurs om 2 år"] = df.apply(lambda r: _rk(r["Omsättning om 2 år"],  r["P/S-snitt"], r["Utestående aktier"]), axis=1)
    df["Riktkurs om 3 år"] = df.apply(lambda r: _rk(r["Omsättning om 3 år"],  r["P/S-snitt"], r["Utestående aktier"]), axis=1)

    return df

# =====================================================================================
# Yahoo-hämtning (pris/valuta/namn/utdelning) & CAGR från historisk omsättning
# =====================================================================================

def yahoo_hämta_bas(ticker: str):
    """Hämta Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning (om möjligt) via yfinance."""
    try:
        tk = yf.Ticker(ticker)
        info = getattr(tk, "info", {}) or {}
        pris = info.get("regularMarketPrice", None)
        valuta = info.get("currency", "USD")
        namn = info.get("longName") or info.get("shortName") or ""
        utd_per_aktie = info.get("dividendRate", 0.0) or 0.0  # kan vara None
        return namn, pris, valuta, float(utd_per_aktie)
    except Exception:
        return "", None, "USD", 0.0

def yahoo_hämta_cagr_totalrevenue_5år(ticker: str) -> float:
    """
    Beräkna CAGR ~5 år bakåt baserat på annuala 'Total Revenue' från yfinance financials.
    Tar första och sista punkten i serien (minst 2 års datapunkter).
    """
    try:
        tk = yf.Ticker(ticker)
        fin = tk.financials
        if fin is None or fin.empty:
            return 0.0
        possible_rows = ["Total Revenue", "TotalRevenue", "totalRevenue"]
        row_name = next((r for r in possible_rows if r in fin.index), None)
        if row_name is None:
            return 0.0

        series = fin.loc[row_name].dropna()
        if series.empty:
            return 0.0

        # Sortera kronologiskt (äldst->nyast)
        series = series[sorted(series.index)]
        values = series.values.astype(float)

        if len(values) < 2:
            return 0.0

        v0 = values[0]
        vn = values[-1]
        n_years = len(values) - 1
        if v0 <= 0 or vn <= 0 or n_years <= 0:
            return 0.0

        cagr = (vn / v0) ** (1.0 / n_years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def uppdatera_från_yahoo_enkelrad(row: pd.Series) -> pd.Series:
    """Returnera en kopia av raden med Yahoo-fält uppdaterade + CAGR beräknad."""
    r = row.copy()
    ticker = str(r.get("Ticker", "")).strip()
    if not ticker:
        return r

    namn, pris, valuta, utd = yahoo_hämta_bas(ticker)
    if namn:
        r["Bolagsnamn"] = namn
    if pris is not None:
        r["Aktuell kurs"] = float(pris)
    if valuta:
        r["Valuta"] = str(valuta).upper()
    if utd is not None:
        r["Årlig utdelning"] = float(utd)

    # CAGR 5 år (%) baserat på historisk Total Revenue
    cagr = yahoo_hämta_cagr_totalrevenue_5år(ticker)
    r["CAGR 5 år (%)"] = float(cagr)

    return r

# =====================================================================================
# Vyer: Analys, Lägg till/uppdatera, Investeringsförslag, Portfölj
# =====================================================================================

def _init_nav_state(key:str):
    if key not in st.session_state:
        st.session_state[key] = 0

def _nav_prev(key:str, length:int):
    if length <= 0: return
    st.session_state[key] = (st.session_state[key] - 1) % length

def _nav_next(key:str, length:int):
    if length <= 0: return
    st.session_state[key] = (st.session_state[key] + 1) % length

# ====== Analys-vy ======
def analysvy(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📈 Analys")

    # Manuell "Ladda om från Google"
    with st.expander("🔄 Google Sheet"):
        if st.button("Ladda om från Google"):
            _hamta_data_cached.clear()
            if "worksheet_obj" in st.session_state:
                st.session_state["worksheet_obj"] = None
            st.success("Cache rensad – laddar om…")
            st.rerun()

    # Filtrera för ett bolag (rullista) + bläddring
    namn_map = {f"{row['Bolagsnamn']} ({row['Ticker']})": idx for idx, row in df.reset_index().iterrows()}
    val = st.selectbox("Välj bolag", [""] + sorted(namn_map.keys()))
    _init_nav_state("analys_index")

    if val:
        st.session_state["analys_index"] = namn_map[val]

    total = len(df)
    if total > 0:
        idx = st.session_state["analys_index"] % total
        st.write(f"Visar bolag **{idx+1}/{total}**")
        colA, colB = st.columns(2)
        with colA:
            if st.button("⬅️ Föregående"):
                _nav_prev("analys_index", total)
        with colB:
            if st.button("Nästa ➡️"):
                _nav_next("analys_index", total)

        rad = df.iloc[idx]
        st.markdown(f"### {rad['Bolagsnamn']} ({rad['Ticker']})")
        st.write(rad.to_frame().T)

    # Visa alltid hela databasen under
    st.markdown("### Databas")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Uppdatera alla från Yahoo")
    if st.button("🔄 Hämta aktuell kurs/valuta/namn/utdelning + CAGR för alla"):
        miss = []
        uppd = 0
        total = len(df)
        bar = st.progress(0)
        status = st.empty()
        for i in range(total):
            try:
                df.iloc[i] = uppdatera_från_yahoo_enkelrad(df.iloc[i])
                uppd += 1
            except Exception:
                miss.append(df.iloc[i]["Ticker"])
            bar.progress((i+1)/total)
            time.sleep(1.0)  # 1 sekund mellan anrop
        # Efter Yahoo: kör beräkningar
        df = uppdatera_berakningar(df)
        spara_data(df)
        status.text("✅ Klar.")
        st.success(f"Uppdaterade {uppd} rader.")
        if miss:
            st.warning("Kunde inte uppdatera: " + ", ".join([m for m in miss if m]))

# ====== Lägg till / uppdatera bolag ======
def lagg_till_eller_uppdatera(df: pd.DataFrame, valutkurser_dummy: dict) -> pd.DataFrame:
    st.subheader("➕ Lägg till / uppdatera bolag")

    namn_map = {f"{row['Bolagsnamn']} ({row['Ticker']})": idx for idx, row in df.reset_index().iterrows()}
    val = st.selectbox("Välj bolag att uppdatera (eller lämna tom för nytt)", [""] + sorted(namn_map.keys()))

    _init_nav_state("edit_index")
    if val:
        st.session_state["edit_index"] = namn_map[val]

    total = len(df)
    idx = (st.session_state["edit_index"] % total) if total > 0 else 0
    st.write(f"Post **{(idx+1 if total>0 else 0)}/{total}**")
    colL, colR = st.columns(2)
    with colL:
        if st.button("⬅️ Föregående bolag"):
            _nav_prev("edit_index", total)
    with colR:
        if st.button("Nästa bolag ➡️"):
            _nav_next("edit_index", total)

    # Hämta rad för editering eller ny blank
    if total > 0:
        bef = df.iloc[idx].copy()
    else:
        bef = pd.Series({c: ("" if c in ["Ticker","Bolagsnamn","Valuta","Senast manuell uppdatering"] else 0.0) for c in KOLUMNER})

    with st.form("edit_form"):
        # Manuell inmatning (dessa driver datumstämplingen)
        ticker = st.text_input("Ticker", value=str(bef.get("Ticker","")).upper())
        utest = st.number_input("Utestående aktier (miljoner)", value=float(bef.get("Utestående aktier",0.0)))
        ps0   = st.number_input("P/S", value=float(bef.get("P/S",0.0)))
        ps1   = st.number_input("P/S Q1", value=float(bef.get("P/S Q1",0.0)))
        ps2   = st.number_input("P/S Q2", value=float(bef.get("P/S Q2",0.0)))
        ps3   = st.number_input("P/S Q3", value=float(bef.get("P/S Q3",0.0)))
        ps4   = st.number_input("P/S Q4", value=float(bef.get("P/S Q4",0.0)))
        oms0  = st.number_input("Omsättning idag (miljoner)", value=float(bef.get("Omsättning idag",0.0)))
        oms1  = st.number_input("Omsättning nästa år (miljoner)", value=float(bef.get("Omsättning nästa år",0.0)))
        antal = st.number_input("Antal aktier du äger", value=float(bef.get("Antal aktier",0.0)))

        st.markdown("—")
        st.caption("Dessa fält uppdateras automatiskt från Yahoo när du sparar:")
        st.write(f"Bolagsnamn (nu: {bef.get('Bolagsnamn','')})")
        st.write(f"Aktuell kurs (nu: {bef.get('Aktuell kurs',0.0)})")
        st.write(f"Valuta (nu: {bef.get('Valuta','')})")
        st.write(f"Årlig utdelning (nu: {bef.get('Årlig utdelning',0.0)})")
        st.write(f"CAGR 5 år (%) (nu: {bef.get('CAGR 5 år (%)',0.0)})")

        spar = st.form_submit_button("💾 Spara")

    if spar:
        # Uppdatera/infoga i df
        ny = bef.copy()
        ny["Ticker"] = ticker
        ny["Utestående aktier"] = utest
        ny["P/S"] = ps0
        ny["P/S Q1"] = ps1
        ny["P/S Q2"] = ps2
        ny["P/S Q3"] = ps3
        ny["P/S Q4"] = ps4
        ny["Omsättning idag"] = oms0
        ny["Omsättning nästa år"] = oms1
        ny["Antal aktier"] = antal

        # Stämpla datum ENDAST om manuellfält ändrats
        def man_fields(r):
            return (
                float(r.get("Utestående aktier",0)),
                float(r.get("P/S",0)), float(r.get("P/S Q1",0)), float(r.get("P/S Q2",0)),
                float(r.get("P/S Q3",0)), float(r.get("P/S Q4",0)),
                float(r.get("Omsättning idag",0)), float(r.get("Omsättning nästa år",0)),
                float(r.get("Antal aktier",0))
            )

        changed = man_fields(ny) != man_fields(bef)
        if changed:
            ny["Senast manuell uppdatering"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Sätt in i DF (om ny ticker och det inte finns rader)
        if total == 0:
            df = pd.DataFrame(columns=KOLUMNER)

        if ticker and ticker in df["Ticker"].astype(str).values:
            # Uppdatera befintlig ticker
            df.loc[df["Ticker"] == ticker, ny.index] = ny.values
        else:
            # Ny rad
            df = pd.concat([df, pd.DataFrame([ny])], ignore_index=True)

        # Hämta Yahoo-fält för just denna rad och gör beräkningar
        ix = df.index[df["Ticker"] == ticker]
        if len(ix) > 0:
            i0 = ix[0]
            df.iloc[i0] = uppdatera_från_yahoo_enkelrad(df.iloc[i0])

        df = uppdatera_berakningar(df)
        spara_data(df)
        st.success(f"{ticker} sparat och uppdaterat från Yahoo.")

    return df

# ====== Investeringsförslag ======
def visa_investeringsforslag(df: pd.DataFrame, endast_portfölj: bool):
    st.subheader("💡 Investeringsförslag")

    # Val: vilken riktkurs styr potentialen
    riktkurs_val = st.selectbox(
        "Jämför mot vilken riktkurs?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1
    )

    sortläge = st.radio(
        "Sortering:",
        ["Störst potential", "Närmast riktkurs (kan vara under/över)"],
        horizontal=True
    )

    kapital = st.number_input("Tillgängligt kapital (i aktiens egen valuta räknas per bolag)", value=500.0, step=100.0)

    # Filtrera ev bara portfölj (Antal aktier > 0)
    df_view = df.copy()
    if endast_portfölj:
        df_view = df_view[df_view["Antal aktier"] > 0]

    # Behöver riktkurs > 0 och aktuell kurs > 0 för meningsfull sortering
    df_view = df_view[(df_view[riktkurs_val] > 0) & (df_view["Aktuell kurs"] > 0)].copy()
    if df_view.empty:
        st.info("Inga bolag matchar för investeringsförslag just nu.")
        return

    if sortläge == "Störst potential":
        df_view["Potential (%)"] = (df_view[riktkurs_val] - df_view["Aktuell kurs"]) / df_view["Aktuell kurs"] * 100.0
        df_view = df_view.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        # Närmast riktkurs: minsta absoluta skillnad i %
        df_view["Avvikelse (%)"] = (df_view["Aktuell kurs"] - df_view[riktkurs_val]) / df_view[riktkurs_val] * 100.0
        df_view["|Avvikelse|"] = df_view["Avvikelse (%)"].abs()
        df_view = df_view.sort_values(by="|Avvikelse|", ascending=True).reset_index(drop=True)

    # Navigering
    _init_nav_state("forslags_index")
    total = len(df_view)
    st.write(f"Förslag **{(st.session_state['forslags_index']%total)+1}/{total}**")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Föregående förslag"):
            _nav_prev("forslags_index", total)
    with c2:
        if st.button("Nästa förslag ➡️"):
            _nav_next("forslags_index", total)

    idx = st.session_state["forslags_index"] % total
    rad = df_view.iloc[idx]

    # Antal man kan köpa för angivet kapital i aktiens valuta
    if rad["Aktuell kurs"] > 0:
        antal_köp = int(kapital // rad["Aktuell kurs"])
    else:
        antal_köp = 0

    # Visa alla fyra riktkurser, highlighta vald
    def _line(lbl, val, active=False):
        if active:
            return f"**{lbl}: {val:.2f} {rad['Valuta']}**"
        return f"{lbl}: {val:.2f} {rad['Valuta']}"

    pot = ((rad[riktkurs_val] - rad["Aktuell kurs"]) / rad["Aktuell kurs"]) * 100.0

    st.markdown(f"""
### {rad['Bolagsnamn']} ({rad['Ticker']})
- **Aktuell kurs:** {rad['Aktuell kurs']:.2f} {rad['Valuta']}
- {_line("Riktkurs idag",   rad['Riktkurs idag'],   riktkurs_val=="Riktkurs idag")}
- {_line("Riktkurs om 1 år", rad['Riktkurs om 1 år'], riktkurs_val=="Riktkurs om 1 år")}
- {_line("Riktkurs om 2 år", rad['Riktkurs om 2 år'], riktkurs_val=="Riktkurs om 2 år")}
- {_line("Riktkurs om 3 år", rad['Riktkurs om 3 år'], riktkurs_val=="Riktkurs om 3 år")}
- **Uppsida enligt valt mått:** {pot:.2f} %
- **Antal att köpa för {kapital:.0f} {rad['Valuta']}:** {antal_köp} st
- **Äger redan:** {int(rad['Antal aktier'])} st
""")

# ====== Portfölj-vy ======
def visa_portfolj(df: pd.DataFrame, valutakurser: dict):
    st.subheader("📦 Min portfölj")

    dfp = df[df["Antal aktier"] > 0].copy()
    if dfp.empty:
        st.info("Du äger inga aktier.")
        return

    # SEK-växling endast här
    dfp["Växelkurs"] = dfp["Valuta"].map(valutakurser).fillna(1.0)
    dfp["Värde (SEK)"] = dfp["Antal aktier"] * dfp["Aktuell kurs"] * dfp["Växelkurs"]
    total_värde = dfp["Värde (SEK)"].sum()

    dfp["Total årlig utdelning (SEK)"] = dfp["Antal aktier"] * dfp["Årlig utdelning"] * dfp["Växelkurs"]
    total_utdelning = dfp["Total årlig utdelning (SEK)"].sum()
    månads_snitt = total_utdelning / 12.0 if total_utdelning else 0.0

    st.markdown(f"**Totalt portföljvärde:** {total_värde:,.2f} SEK")
    st.markdown(f"**Förväntad årlig utdelning (SEK):** {total_utdelning:,.2f}")
    st.markdown(f"**Ungefärlig månadsutdelning (SEK):** {månads_snitt:,.2f}")

    st.dataframe(
        dfp[["Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta","Värde (SEK)","Årlig utdelning","Total årlig utdelning (SEK)"]],
        use_container_width=True
    )

# =====================================================================================
# main()
# =====================================================================================

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # Sidopanel — valutakurser (endast SEK-presentation använder dessa)
    st.sidebar.header("💱 Valutakurser → SEK")
    user_rates = {
        "USD": st.sidebar.number_input("USD → SEK", value=float(STANDARD_VALUTAKURSER["USD"]), step=0.01),
        "NOK": st.sidebar.number_input("NOK → SEK", value=float(STANDARD_VALUTAKURSER["NOK"]), step=0.01),
        "CAD": st.sidebar.number_input("CAD → SEK", value=float(STANDARD_VALUTAKURSER["CAD"]), step=0.01),
        "EUR": st.sidebar.number_input("EUR → SEK", value=float(STANDARD_VALUTAKURSER["EUR"]), step=0.01),
        "SEK": 1.0,
    }

    # Manuell “Ladda om från Google” i sidopanelen
    with st.sidebar.expander("🔄 Google Sheet"):
        if st.button("Ladda om från Google", key="reload_sidebar"):
            _hamta_data_cached.clear()
            if "worksheet_obj" in st.session_state:
                st.session_state["worksheet_obj"] = None
            st.success("Cache rensad – laddar om…")
            st.rerun()

    # Läs data
    df = hamta_data()
    df = säkerställ_kolumner(df)
    df = konvertera_typer(df)

    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df2 = lagg_till_eller_uppdatera(df, user_rates)
        # df2 kan vara uppdaterad — spara endast om något faktiskt ändrades:
        if not df2.equals(df):
            spara_data(df2)

    elif meny == "Investeringsförslag":
        df_calc = uppdatera_berakningar(df)
        endast_pf = st.checkbox("Endast portföljens innehav", value=False)
        visa_investeringsforslag(df_calc, endast_pf)

    elif meny == "Portfölj":
        df_calc = uppdatera_berakningar(df)
        visa_portfolj(df_calc, user_rates)


if __name__ == "__main__":
    main()
