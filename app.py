# =========================
# Del 1/4 – Importer, kopplingar, hjälpfunktioner & valutakurser I/O
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

# ---- Lokal tid (Stockholm) om pytz finns, annars systemtid ----
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# ---- Google Sheets koppling ----
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

# Liten backoff-wrapper för att undvika kvotspikar
def _with_backoff(func, *args, **kwargs):
    delay = 0.7
    for i in range(5):
        try:
            return func(*args, **kwargs)
        except Exception:
            if i == 4:
                raise
            time.sleep(delay)
            delay *= 1.6

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return get_spreadsheet().worksheet(SHEET_NAME)

def hamta_data():
    sheet = skapa_koppling()
    data = _with_backoff(sheet.get_all_records)
    return pd.DataFrame(data)

def spara_data(df: pd.DataFrame):
    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    _with_backoff(sheet.update, [df.columns.values.tolist()] + df.astype(str).values.tolist())

# ---- Hjälpare: tolka svenska/engelska decimaler, ta bort tusentalsavskiljare ----
def parse_decimal(val):
    """Accepterar t.ex. '9,46', '9.46', '1 234,56' och returnerar float eller None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # ta bort olika mellanslag som tusentalsavskiljare
    s = s.replace('\xa0', '').replace('\u202f', '').replace(' ', '')
    # ersätt komma med punkt
    s = s.replace(',', '.')
    # om flera punkter (t.ex. "1.234.56") → behåll sista som decimal
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    try:
        return float(s)
    except Exception:
        return None

# ---- Standard valutakurser till SEK (kan överstyras av sparade värden / sidopanel) ----
STANDARD_VALUTAKURSER = {
    "USD": 9.75,
    "NOK": 0.95,
    "CAD": 7.05,
    "EUR": 11.18,
    "SEK": 1.0,
}

def _ensure_valuta_sheet():
    """Skapa bladet 'Valutakurser' om det saknas."""
    ss = get_spreadsheet()
    try:
        ss.worksheet("Valutakurser")
    except Exception:
        ss.add_worksheet(title="Valutakurser", rows=10, cols=2)
        ws = ss.worksheet("Valutakurser")
        _with_backoff(ws.update, [["Valuta", "Kurs"]])

@st.cache_data(ttl=60)
def las_sparade_valutakurser_cached(_bump: int = 0) -> dict:
    """Läser valutakurser från bladet 'Valutakurser' och returnerar dict."""
    ss = get_spreadsheet()
    try:
        ws = ss.worksheet("Valutakurser")
    except Exception:
        return {}
    rows = _with_backoff(ws.get_all_records)
    rates = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).strip().upper()
        val = parse_decimal(r.get("Kurs"))
        if cur and val is not None:
            rates[cur] = val
    return rates

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict) -> None:
    """Sparar till bladet 'Valutakurser' med punkt-decimal (robust)."""
    _ensure_valuta_sheet()
    ss = get_spreadsheet()
    ws = ss.worksheet("Valutakurser")
    out = [["Valuta", "Kurs"]]
    for k, v in rates.items():
        try:
            out.append([k.upper(), f"{float(v):.6f}"])
        except Exception:
            continue
    _with_backoff(ws.clear)
    _with_backoff(ws.update, out)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    if not valuta:
        return 1.0
    return user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))

# ---- Kolumnschema & datarengöring ----
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs"]):
                df[kol] = 0.0
            else:
                df[kol] = ""
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Riktkurs 2026": "Riktkurs om 1 år",
        "Riktkurs 2027": "Riktkurs om 2 år",
        "Riktkurs 2028": "Riktkurs om 3 år",
        "Riktkurs om idag": "Riktkurs idag",
    }
    for old, new in mapping.items():
        if old in df.columns:
            if new not in df.columns:
                df[new] = 0.0
            new_vals = pd.to_numeric(df[new], errors="coerce").fillna(0.0)
            old_vals = pd.to_numeric(df[old], errors="coerce").fillna(0.0)
            mask = (new_vals == 0.0) & (old_vals > 0.0)
            df.loc[mask, new] = old_vals[mask]
            df = df.drop(columns=[old])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
        "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
        "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
        "Antal aktier", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# =========================
# Del 2/4 – Yahoo-hämtning (namn/kurs/valuta/utdelning/CAGR) + beräkningar + massuppdatering
# =========================

# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    """
    Försöker läsa årliga intäkter (Total Revenue) och beräknar CAGR över tillgängliga år.
    Returnerar procent (t.ex. 12.34) eller 0.0 om ej möjligt.
    """
    try:
        # Nya yfinance (income_stmt) först, annars fallback till financials
        df_is = getattr(tkr, "income_stmt", None)
        if isinstance(df_is, pd.DataFrame) and not df_is.empty and "Total Revenue" in df_is.index:
            series = df_is.loc["Total Revenue"].dropna()
        else:
            df_fin = getattr(tkr, "financials", None)
            if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Total Revenue" in df_fin.index:
                series = df_fin.loc["Total Revenue"].dropna()
            else:
                return 0.0

        if series.empty or len(series) < 2:
            return 0.0

        # sortera kronologiskt (äldst → nyast)
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series) - 1)

        if start <= 0:
            return 0.0

        cagr = (end / start) ** (1.0 / years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


# ---- Hämtning från Yahoo (namn, kurs, valuta, utdelning, CAGR) ----
def hamta_yahoo_fält(ticker: str) -> dict:
    """
    Returnerar:
      Bolagsnamn, Aktuell kurs, Valuta, Årlig utdelning, CAGR 5 år (%)
    Saknas ett fält lämnas tomt/0.
    """
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)

        # info kan kasta i vissa miljöer → skydda
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        # Pris
        pris = info.get("regularMarketPrice", None)
        if pris is None:
            # Fallback via historik (senaste Close)
            h = t.history(period="5d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        # Valuta
        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()

        # Namn
        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        # Årlig utdelning per aktie (kan vara None)
        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        # CAGR 5 år (%) baserat på Total Revenue (årligen)
        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)

    except Exception:
        # lämna default
        pass

    return out


# ---- Beräkningar (P/S-snitt, omsättning år2/3 m. clamp, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    """
    - P/S-snitt = snitt av positiva P/S Q1–Q4
    - CAGR clamp: >100% → 50%, <0% → 2%
    - Omsättning om 2 & 3 år = "Omsättning nästa år" växt med clampad CAGR
    - Riktkurser = (Omsättning * P/S-snitt) / Utestående aktier
    """
    for i, rad in df.iterrows():
        # P/S-snitt
        ps_vals = [
            rad.get("P/S Q1", 0), rad.get("P/S Q2", 0),
            rad.get("P/S Q3", 0), rad.get("P/S Q4", 0)
        ]
        ps_clean = [float(x) for x in ps_vals if float(x) > 0]
        ps_snitt = round(np.mean(ps_clean), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp
        cagr = float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 år & 3 år från "Omsättning nästa år"
        oms_next = float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # behåll ev. befintliga ifyllda, annars 0
            df.at[i, "Omsättning om 2 år"] = float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser
        aktier_ut = float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((float(rad.get("Omsättning idag", 0.0))     * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((float(rad.get("Omsättning nästa år", 0.0)) * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round((float(df.at[i, "Omsättning om 2 år"])      * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round((float(df.at[i, "Omsättning om 3 år"])      * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"]    = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df


# ---- Massuppdatera från Yahoo (1s delay, kopierbar felrapport). Ändrar inte "Senast manuellt uppdaterad". ----
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []   # lista med strängar: "TICKER: fält1, fält2 ..."
        total = len(df)

        for i, row in df.iterrows():
            tkr = str(row.get("Ticker", "")).strip()
            if not tkr:
                continue

            status.write(f"Uppdaterar {i+1}/{total} – {tkr}")
            data = hamta_yahoo_fält(tkr)
            failed_fields = []

            # Skriv endast om något faktiskt kom
            if data.get("Bolagsnamn"):
                df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
            else:
                failed_fields.append("Bolagsnamn")

            if data.get("Aktuell kurs", 0) > 0:
                df.at[i, "Aktuell kurs"] = data["Aktuell kurs"]
            else:
                failed_fields.append("Aktuell kurs")

            if data.get("Valuta"):
                df.at[i, "Valuta"] = data["Valuta"]
            else:
                failed_fields.append("Valuta")

            # Årlig utdelning kan vara 0 – skriv ändå (om None: 0.0)
            if "Årlig utdelning" in data:
                df.at[i, "Årlig utdelning"] = float(data.get("Årlig utdelning") or 0.0)
            else:
                failed_fields.append("Årlig utdelning")

            if "CAGR 5 år (%)" in data:
                df.at[i, "CAGR 5 år (%)"] = float(data.get("CAGR 5 år (%)") or 0.0)
            else:
                failed_fields.append("CAGR 5 år (%)")

            if failed_fields:
                misslyckade.append(f"{tkr}: {', '.join(failed_fields)}")

            # 1 sekund paus mellan anropen
            time.sleep(1.0)
            bar.progress((i + 1) / max(1, total))

        # Beräkna om efter hämtning
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera listan nedan:")
            st.sidebar.text_area("Misslyckade fält (kopierbar)", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df

# =========================
# Del 3/4 – Lägg till/uppdatera bolag (med datumlogik) + Analysvy
# =========================

# Vilka fält som räknas som "manuella" för datumstämpling
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år"
]

def _num_text_input(label: str, default_val: float) -> float:
    """
    Textfält som accepterar blankt och både komma & punkt.
    Returnerar float (blank → 0.0).
    """
    s = "" if default_val == 0 or pd.isna(default_val) else str(default_val)
    raw = st.text_input(label, value=s)
    if raw is None or raw.strip() == "":
        return 0.0
    raw = raw.replace(" ", "").replace(",", ".")
    try:
        return float(raw)
    except Exception:
        return 0.0

def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sorteringsval för rullistan
    sort_val = st.selectbox(
        "Sortera för redigering",
        ["A–Ö (bolagsnamn)", "Äldst manuell uppdatering först"]
    )
    vis_df = df.copy()
    if sort_val.startswith("Äldst"):
        vis_df["_sort_datum"] = vis_df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = vis_df.sort_values(by=["_sort_datum", "Bolagsnamn"])
    else:
        vis_df = vis_df.sort_values(by=["Bolagsnamn", "Ticker"])

    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]
    idx_map = list(vis_df["Ticker"])

    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0
    # Säkerställ rimligt index
    st.session_state.edit_index = min(st.session_state.edit_index, max(0, len(etiketter)-1))

    # Rullista + bläddring
    sel = st.selectbox(
        "Välj bolag (lämna tomt för nytt)",
        [""] + etiketter,
        index=(st.session_state.edit_index + 1) if len(etiketter) > 0 else 0
    )

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående", key="edit_prev"):
            if len(etiketter) > 0:
                st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_mid:
        st.write(f"Post {0 if len(etiketter)==0 else (st.session_state.edit_index+1)}/{max(1, len(etiketter))}")
    with col_next:
        if st.button("➡️ Nästa", key="edit_next"):
            if len(etiketter) > 0:
                st.session_state.edit_index = min(len(etiketter)-1, st.session_state.edit_index + 1)

    # Hämta befintlig rad (om något valt)
    if sel and len(etiketter) > 0:
        # mappa label -> ticker
        label_to_ticker = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
        tick = label_to_ticker.get(sel, "")
        bef = df[df["Ticker"] == tick].iloc[0] if tick in df["Ticker"].values else pd.Series(dtype=object)
        # synka index till vald post
        if tick in list(vis_df["Ticker"]):
            st.session_state.edit_index = list(vis_df["Ticker"]).index(tick)
    else:
        bef = pd.Series(dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)", value=bef.get("Ticker","") if not bef.empty else "").upper()
            utest = _num_text_input("Utestående aktier (miljoner)", float(bef.get("Utestående aktier",0.0)) if not bef.empty else 0.0)
            antal = _num_text_input("Antal aktier du äger", float(bef.get("Antal aktier",0.0)) if not bef.empty else 0.0)

            ps  = _num_text_input("P/S", float(bef.get("P/S",0.0)) if not bef.empty else 0.0)
            ps1 = _num_text_input("P/S Q1", float(bef.get("P/S Q1",0.0)) if not bef.empty else 0.0)
            ps2 = _num_text_input("P/S Q2", float(bef.get("P/S Q2",0.0)) if not bef.empty else 0.0)
            ps3 = _num_text_input("P/S Q3", float(bef.get("P/S Q3",0.0)) if not bef.empty else 0.0)
            ps4 = _num_text_input("P/S Q4", float(bef.get("P/S Q4",0.0)) if not bef.empty else 0.0)
        with c2:
            oms_idag  = _num_text_input("Omsättning idag (miljoner)", float(bef.get("Omsättning idag",0.0)) if not bef.empty else 0.0)
            oms_next  = _num_text_input("Omsättning nästa år (miljoner)", float(bef.get("Omsättning nästa år",0.0)) if not bef.empty else 0.0)

            st.markdown("**Uppdateras automatiskt vid spara:**")
            st.write("- Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.write("- Omsättning om 2 & 3 år, Riktkurser och P/S-snitt beräknas om")

            # Visa (read-only) senast manuellt uppdaterad
            st.text_input("Senast manuellt uppdaterad", value=bef.get("Senast manuellt uppdaterad","") if not bef.empty else "", disabled=True)

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # bygg ny värde-dict för manuell data
        ny = {
            "Ticker": ticker,
            "Utestående aktier": utest,
            "Antal aktier": antal,
            "P/S": ps, "P/S Q1": ps1, "P/S Q2": ps2, "P/S Q3": ps3, "P/S Q4": ps4,
            "Omsättning idag": oms_idag,
            "Omsättning nästa år": oms_next
        }

        # avgör om datum ska uppdateras
        datum_sätt = False
        if not bef.empty:
            before = {f: float(bef.get(f,0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: float(ny.get(f,0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(float(ny.get(f,0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # skriv in/foga rad
        if ticker in df["Ticker"].values:
            for k, v in ny.items():
                df.loc[df["Ticker"] == ticker, k] = v
        else:
            tom = {c: (0.0 if c not in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"] else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        # datum vid manuell ändring
        if datum_sätt:
            df.loc[df["Ticker"] == ticker, "Senast manuellt uppdaterad"] = now_stamp()

        # hämta Yahoo-fält → skriv om
        y = hamta_yahoo_fält(ticker)
        if y.get("Bolagsnamn"):           df.loc[df["Ticker"]==ticker, "Bolagsnamn"] = y["Bolagsnamn"]
        if y.get("Valuta"):               df.loc[df["Ticker"]==ticker, "Valuta"]     = y["Valuta"]
        if y.get("Aktuell kurs",0)>0:     df.loc[df["Ticker"]==ticker, "Aktuell kurs"] = y["Aktuell kurs"]
        if "Årlig utdelning" in y:        df.loc[df["Ticker"]==ticker, "Årlig utdelning"] = float(y.get("Årlig utdelning") or 0.0)
        if "CAGR 5 år (%)" in y:          df.loc[df["Ticker"]==ticker, "CAGR 5 år (%)"]   = float(y.get("CAGR 5 år (%)") or 0.0)

        # räkna om & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)
        st.success("Sparat och uppdaterat från Yahoo.")

    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    tmp = df.copy()
    tmp["_sort_datum"] = tmp["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = tmp.sort_values(by=["_sort_datum","Bolagsnamn"]).head(10)
    st.dataframe(
        tips[["Ticker","Bolagsnamn","Senast manuellt uppdaterad","P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","Omsättning idag","Omsättning nästa år"]],
        use_container_width=True
    )

    return df


def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")

    vis_df = df.sort_values(by=["Bolagsnamn","Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0
    st.session_state.analys_idx = min(st.session_state.analys_idx, max(0, len(etiketter)-1))

    sel = st.selectbox(
        "Välj bolag att visa",
        etiketter if etiketter else ["(tomt)"],
        index=st.session_state.analys_idx if etiketter else 0
    )

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("⬅️ Föregående", key="analys_prev_btn"):
            if len(etiketter) > 0:
                st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with colB:
        if st.button("➡️ Nästa", key="analys_next_btn"):
            if len(etiketter) > 0:
                st.session_state.analys_idx = min(len(etiketter)-1, st.session_state.analys_idx + 1)

    st.write(f"Post {0 if len(etiketter)==0 else (st.session_state.analys_idx+1)}/{max(1,len(etiketter))}")

    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = [
            "Ticker","Bolagsnamn","Valuta","Aktuell kurs",
            "Utestående aktier",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "CAGR 5 år (%)","Antal aktier","Årlig utdelning","Senast manuellt uppdaterad"
        ]
        # skydda om någon kolumn mot förmodan saknas
        cols = [c for c in cols if c in r.index]
        st.dataframe(pd.DataFrame([r[cols].to_dict()]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)

# =========================
# Del 4/4 – Portfölj, Investeringsförslag & main()
# =========================

def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = round(port["Värde (SEK)"] / total_värde * 100.0, 2) if total_värde > 0 else 0.0
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[[
            "Ticker","Bolagsnamn","Antal aktier","Aktuell kurs","Valuta",
            "Värde (SEK)","Andel (%)","Årlig utdelning","Total årlig utdelning (SEK)"
        ]],
        use_container_width=True
    )


def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")

    # Kapital i SEK (för beräkning av hur många man kan köpa)
    kapital_sek = _num_text_input("Tillgängligt kapital (SEK)", 500.0)

    # Val av riktkurs för potential/diff
    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år"],
        index=1
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag","Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential","Närmast riktkurs"], horizontal=True)

    # Filtrera databas
    base = df.copy()
    if subset == "Endast portfölj":
        base = base[base["Antal aktier"] > 0].copy()

    # Kräver riktkurs och aktuell kurs > 0
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()
    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    # Potential och diff i instrumentets valuta (ingen SEK-konvertering här)
    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Bläddringsläge
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base)-1)

    col_prev, col_mid, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base)-1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljvärden i SEK (för andelsberäkningar)
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(kapital_sek // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    # Markera vald riktkurs med fetstil
    def mark(label: str) -> str:
        return "**⬅ vald**" if riktkurs_val == label else ""

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'],2)} {rad['Valuta']}
- **Riktkurs idag:** {round(rad['Riktkurs idag'],2)} {rad['Valuta']} {mark("Riktkurs idag")}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'],2)} {rad['Valuta']} {mark("Riktkurs om 1 år")}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'],2)} {rad['Valuta']} {mark("Riktkurs om 2 år")}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'],2)} {rad['Valuta']} {mark("Riktkurs om 3 år")}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'],2)} %
- **Antal att köpa för {int(kapital_sek)} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )


def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # --- Sidopanel: valutakurser (läs sparade; editera; spara) ---
    st.sidebar.header("💱 Valutakurser → SEK")

    # Läs sparade kurser (Del 1/2 definierar las_sparade_valutakurser / spara_valutakurser)
    try:
        saved_rates = las_sparade_valutakurser()
    except Exception:
        saved_rates = {}

    # Slå ihop: sparade överstyr standard
    current_rates = STANDARD_VALUTAKURSER.copy()
    for k, v in saved_rates.items():
        try:
            current_rates[k] = float(str(v).replace(",", "."))
        except Exception:
            pass

    # Editera via textfält (tillåter att man raderar helt)
    usd_edit = _num_text_input("USD → SEK", current_rates.get("USD", STANDARD_VALUTAKURSER["USD"]))
    nok_edit = _num_text_input("NOK → SEK", current_rates.get("NOK", STANDARD_VALUTAKURSER["NOK"]))
    cad_edit = _num_text_input("CAD → SEK", current_rates.get("CAD", STANDARD_VALUTAKURSER["CAD"]))
    eur_edit = _num_text_input("EUR → SEK", current_rates.get("EUR", STANDARD_VALUTAKURSER["EUR"]))

    user_rates = {"USD": usd_edit, "NOK": nok_edit, "CAD": cad_edit, "EUR": eur_edit, "SEK": 1.0}

    # Knappar för sparning / återläsning
    col_sr1, col_sr2 = st.sidebar.columns(2)
    with col_sr1:
        if st.button("💾 Spara kurser"):
            try:
                spara_valutakurser(user_rates)
                st.sidebar.success("Valutakurser sparade.")
            except Exception as e:
                st.sidebar.error(f"Kunde inte spara: {e}")
    with col_sr2:
        if st.button("↺ Läs sparade"):
            # Tvinga omcache-läsning (Del 1/2 hanterar cacheklarering)
            if "rates_reload" not in st.session_state:
                st.session_state.rates_reload = 0
            st.session_state.rates_reload += 1
            st.sidebar.success("Läser om sparade kurser – uppdatera sidan om inget syns direkt.")

    st.sidebar.markdown("---")
    st.sidebar.caption("Tips: valutakurser används endast för SEK-värden i portfölj och vid köpbelopp i SEK.")

    # --- Läs data & säkerställ schema ---
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # --- Global Yahoo-uppdatering i sidopanelen ---
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # --- Meny ---
    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        # Säkra färska beräkningar för visningen
        df_view = uppdatera_berakningar(df.copy(), user_rates)
        analysvy(df_view, user_rates)

    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
        # Spara direkt efter ev. ändringar
        spara_data(df)

    elif meny == "Investeringsförslag":
        df_calc = uppdatera_berakningar(df.copy(), user_rates)
        visa_investeringsforslag(df_calc, user_rates)

    elif meny == "Portfölj":
        df_calc = uppdatera_berakningar(df.copy(), user_rates)
        visa_portfolj(df_calc, user_rates)


if __name__ == "__main__":
    main()
