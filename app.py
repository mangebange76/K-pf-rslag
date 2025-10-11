# ===== DEL 1/6 — BAS & DECIMALFIX =====
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import yfinance as yf
import time
from datetime import datetime
from google.oauth2.service_account import Credentials

# --- Lokal Stockholm-tid om pytz finns (annars systemtid) ---
try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def now_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def now_stamp():
        return datetime.now().strftime("%Y-%m-%d")

st.set_page_config(page_title="Aktieanalys och investeringsförslag", layout="wide")

# --- Google Sheets-koppling ---
SHEET_URL = st.secrets["SHEET_URL"]
SHEET_NAME = "Blad1"
RATES_SHEET_NAME = "Valutakurser"

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(st.secrets["GOOGLE_CREDENTIALS"], scopes=scope)
client = gspread.authorize(credentials)

def _with_backoff(func, *args, **kwargs):
    """Liten backoff-hjälpare för att mildra 429/kvotfel."""
    delays = [0, 0.5, 1.0, 2.0]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

def get_spreadsheet():
    return client.open_by_url(SHEET_URL)

def skapa_koppling():
    return get_spreadsheet().worksheet(SHEET_NAME)

def skapa_rates_sheet_if_missing():
    ss = get_spreadsheet()
    try:
        return ss.worksheet(RATES_SHEET_NAME)
    except Exception:
        ss.add_worksheet(title=RATES_SHEET_NAME, rows=10, cols=5)
        ws = ss.worksheet(RATES_SHEET_NAME)
        ws.update([["Valuta","Kurs"]])
        return ws

# ---------- SVE-decimalparser & formatter ----------
def _parse_sv_float(val) -> float:
    """
    Accepterar svenska/engelska format:
    '10,61' '1 234,56' '1,234.56' 1061 '' -> float.
    """
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 0.0
    s = s.replace(" ", "").replace("\u202f", "")
    if "," in s and "." in s:
        # sista av ,/. antas vara decimaltecken
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0

def _fmt_sv(val: float, decimals: int = 4) -> str:
    try:
        return f"{float(val):.{decimals}f}".replace(".", ",")
    except Exception:
        return "0," + "0"*decimals
# --------------------------------------------------

# --- Standard valutakurser till SEK (startvärden) ---
STANDARD_VALUTAKURSER = {"USD": 9.75, "NOK": 0.95, "CAD": 7.05, "EUR": 11.18, "SEK": 1.0}

# --- Kolumnschema (inkl. 'GAV (SEK)') ---
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "GAV (SEK)", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if any(x in kol.lower() for x in ["kurs","omsättning","p/s","utdelning","cagr","antal","riktkurs","gav"]):
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
        "Antal aktier", "GAV (SEK)", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].apply(_parse_sv_float)
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df
# ===== Slut DEL 1/6 =====

# ===== DEL 2/6 — SHEETS I/O & VALUTAKURSER (SVE-DECIMAL) =====

# --------- Läs & spara huvudbladet (ALLT genom svensk-decimalfilter) ---------
def hamta_data() -> pd.DataFrame:
    """Hämtar databasen från Google Sheets och ser till att kolumner/typer är rätt."""
    try:
        sheet = skapa_koppling()
        data = _with_backoff(sheet.get_all_records)  # list[dict]
        df = pd.DataFrame(data)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})

    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)   # <- använder _parse_sv_float för alla numeriska
    return df


# Hur många decimaler vi vill skriva till arket per numerisk kolumn
_DECIMALS_MAP = {
    # kurs/kvoter
    "P/S": 2, "P/S Q1": 2, "P/S Q2": 2, "P/S Q3": 2, "P/S Q4": 2, "P/S-snitt": 2,
    # omsättning/riktkurs (miljoner och pris/aktie)
    "Omsättning idag": 2, "Omsättning nästa år": 2, "Omsättning om 2 år": 2, "Omsättning om 3 år": 2,
    "Riktkurs idag": 2, "Riktkurs om 1 år": 2, "Riktkurs om 2 år": 2, "Riktkurs om 3 år": 2,
    # övrigt
    "Utestående aktier": 2, "Antal aktier": 2, "GAV (SEK)": 4, "Årlig utdelning": 4,
    "Aktuell kurs": 4, "CAGR 5 år (%)": 2,
}

def _to_sheet_cell(col: str, val):
    """Formatera en cell för ark – alla numeriska går via svensk formattering."""
    try:
        f = _parse_sv_float(val)
        # Om talet är heltal → skriv utan decimaler
        if float(f).is_integer() and _DECIMALS_MAP.get(col, 2) <= 2:
            return str(int(f))
        return _fmt_sv(f, _DECIMALS_MAP.get(col, 2))
    except Exception:
        return str(val) if val is not None else ""

def spara_data(df: pd.DataFrame):
    """Sparar HELA df till arket, med svenska decimaler. Bevarar kolumnordning."""
    df = säkerställ_kolumner(df.copy())
    # Format­tera rader
    body = [list(FINAL_COLS)]
    for _, row in df.iterrows():
        out_row = []
        for col in FINAL_COLS:
            v = row.get(col, "")
            if col in _DECIMALS_MAP:
                out_row.append(_to_sheet_cell(col, v))
            else:
                out_row.append("" if v is None else str(v))
        body.append(out_row)

    sheet = skapa_koppling()
    _with_backoff(sheet.clear)
    _with_backoff(sheet.update, body)


# ---------------------- Valutakurser (separat blad) --------------------------
@st.cache_data(show_spinner=False)
def las_sparade_valutakurser_cached(nonce: int):
    """
    Läser 'Valutakurser'-bladet. Tolkar alltid som svenska tal.
    Returnerar t.ex. {'USD': 10.61, 'SEK': 1.0}
    """
    ws = skapa_rates_sheet_if_missing()
    rows = _with_backoff(ws.get_all_records)  # [{'Valuta': 'USD', 'Kurs': '10,61'}, ...]
    out = {}
    for r in rows:
        cur = str(r.get("Valuta", "")).upper().strip()
        if not cur:
            continue
        out[cur] = _parse_sv_float(r.get("Kurs", ""))
    # Garanti
    out["SEK"] = 1.0
    return out

def las_sparade_valutakurser() -> dict:
    return las_sparade_valutakurser_cached(st.session_state.get("rates_reload", 0))

def spara_valutakurser(rates: dict):
    """Skriver valutakurser till bladet med KOMMA som decimal (svenskt)."""
    ws = skapa_rates_sheet_if_missing()
    body = [["Valuta","Kurs"]]
    for k in ["USD","NOK","CAD","EUR","SEK"]:
        v = _parse_sv_float(rates.get(k, STANDARD_VALUTAKURSER.get(k, 1.0)))
        body.append([k, _fmt_sv(v, 6)])  # 6 decimaler för valutor
    _with_backoff(ws.clear)
    _with_backoff(ws.update, body)

def hamta_valutakurs(valuta: str, user_rates: dict) -> float:
    """Hämta kurs ur user_rates/standard (siffran används i beräkningar)."""
    if not valuta:
        return 1.0
    v = user_rates.get(valuta.upper(), STANDARD_VALUTAKURSER.get(valuta.upper(), 1.0))
    return _parse_sv_float(v)


# --------- Automatisk valutahämtning via Yahoo (cache 1h) ----------
@st.cache_data(show_spinner=False, ttl=3600)
def hamta_valutakurser_automatiskt() -> dict:
    """
    Hämtar USD/NOK/CAD/EUR → SEK från Yahoo Finance (senaste Close).
    Returnerar dict med flyttal (inte strängar).
    """
    par = {
        "USD": "USDSEK=X",
        "NOK": "NOKSEK=X",
        "CAD": "CADSEK=X",
        "EUR": "EURSEK=X",
    }
    res = {"SEK": 1.0}
    for code, ysym in par.items():
        try:
            h = yf.Ticker(ysym).history(period="1d")
            if not h.empty and "Close" in h:
                val = float(h["Close"].iloc[-1])
                if val > 0:
                    res[code] = round(val, 6)
        except Exception:
            pass
    return res

def auto_update_valutakurser_if_stale() -> bool:
    """
    Jämför sparade kurser mot live-kurser. Om skillnad (eller tom sparfil),
    skriv till Google Sheets (svenska decimaler) och bumpa cache-nyckeln.
    """
    try:
        saved = las_sparade_valutakurser()
        live = hamta_valutakurser_automatiskt()
        if not live:
            return False
        changed = False
        for k in ("USD", "NOK", "CAD", "EUR"):
            lv = _parse_sv_float(live.get(k, None))
            sv = _parse_sv_float(saved.get(k, 0.0))
            if lv and abs(lv - sv) > 1e-6:
                changed = True
                break
        if changed or not saved:
            merged = saved.copy()
            merged.update(live)
            spara_valutakurser(merged)  # <- skriver KOMMA-decimaler
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            return True
        return False
    except Exception:
        return False

# ===== Slut DEL 2/6 =====

# ===== DEL 3/6 — SCHEMA, TYPER, YAHOO & BERÄKNINGAR =====

# ---- Kolumnschema (håll samma ordning överallt) ----
FINAL_COLS = [
    "Ticker", "Bolagsnamn", "Utestående aktier",
    "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "GAV (SEK)", "Valuta", "Årlig utdelning", "Aktuell kurs",
    "CAGR 5 år (%)", "P/S-snitt",
    "Senast manuellt uppdaterad",
]

# Kolumner som ska behandlas som numeriska (svensk-decimal)
_NUMERIC_COLS = [
    "Utestående aktier", "P/S", "P/S Q1", "P/S Q2", "P/S Q3", "P/S Q4",
    "Omsättning idag", "Omsättning nästa år", "Omsättning om 2 år", "Omsättning om 3 år",
    "Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år",
    "Antal aktier", "GAV (SEK)", "Årlig utdelning", "Aktuell kurs", "CAGR 5 år (%)", "P/S-snitt"
]

def säkerställ_kolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Lägg till saknade kolumner med rätt default-typ."""
    df = df.copy()
    for kol in FINAL_COLS:
        if kol not in df.columns:
            if kol in _NUMERIC_COLS:
                df[kol] = 0.0
            else:
                df[kol] = ""
    # Ta bort okända kolumner? Nej – bevara, men vi sparar bara FINAL_COLS.
    return df

def migrera_gamla_riktkurskolumner(df: pd.DataFrame) -> pd.DataFrame:
    """Mappar ev. gamla riktkursnamn till nya; tolkar svenska decimaler."""
    df = df.copy()
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
            new_vals = df[new].map(_parse_sv_float)
            old_vals = df[old].map(_parse_sv_float)
            mask = (new_vals == 0.0) & (old_vals > 0.0)
            df.loc[mask, new] = old_vals[mask]
            df = df.drop(columns=[old])
    return df

def konvertera_typer(df: pd.DataFrame) -> pd.DataFrame:
    """Tvinga numeriska kolumner via svensk-decimaltolkning, textkolumner till str."""
    df = df.copy()
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].map(_parse_sv_float)
    for c in ["Ticker","Bolagsnamn","Valuta","Senast manuellt uppdaterad"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


# ---- CAGR från yfinance (Total Revenue, årligen) ----
def beräkna_cagr_från_finansiella(tkr: yf.Ticker) -> float:
    try:
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
        series = series.sort_index()  # kronologisk
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0


# ---- Hämtning från Yahoo (namn, kurs, valuta, utdelning, CAGR) ----
def hamta_yahoo_fält(ticker: str) -> dict:
    out = {
        "Bolagsnamn": "",
        "Aktuell kurs": 0.0,
        "Valuta": "USD",
        "Årlig utdelning": 0.0,
        "CAGR 5 år (%)": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice", None)
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["Aktuell kurs"] = float(pris)

        valuta = info.get("currency", None)
        if valuta:
            out["Valuta"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["Bolagsnamn"] = str(namn)

        div_rate = info.get("dividendRate", None)
        if div_rate is not None:
            out["Årlig utdelning"] = float(div_rate)

        out["CAGR 5 år (%)"] = beräkna_cagr_från_finansiella(t)
    except Exception:
        pass
    return out


# ---- Beräkningar (P/S-snitt, omsättning år2/3 med clamp, riktkurser) ----
def uppdatera_berakningar(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    df = df.copy()
    for i, rad in df.iterrows():
        # P/S-snitt: snitt av positiva Q1–Q4
        ps_vals = [
            _parse_sv_float(rad.get("P/S Q1", 0)),
            _parse_sv_float(rad.get("P/S Q2", 0)),
            _parse_sv_float(rad.get("P/S Q3", 0)),
            _parse_sv_float(rad.get("P/S Q4", 0)),
        ]
        ps_clean = [x for x in ps_vals if x > 0]
        ps_snitt = round(float(np.mean(ps_clean)), 2) if ps_clean else 0.0
        df.at[i, "P/S-snitt"] = ps_snitt

        # CAGR clamp: >100% → 50%, <0% → 2%
        cagr = _parse_sv_float(rad.get("CAGR 5 år (%)", 0.0))
        just_cagr = 50.0 if cagr > 100.0 else (2.0 if cagr < 0.0 else cagr)
        g = just_cagr / 100.0

        # Omsättning om 2 & 3 år från "Omsättning nästa år"
        oms_next = _parse_sv_float(rad.get("Omsättning nästa år", 0.0))
        if oms_next > 0:
            df.at[i, "Omsättning om 2 år"] = round(oms_next * (1.0 + g), 2)
            df.at[i, "Omsättning om 3 år"] = round(oms_next * ((1.0 + g) ** 2), 2)
        else:
            # bevara ev. manuella värden
            df.at[i, "Omsättning om 2 år"] = _parse_sv_float(rad.get("Omsättning om 2 år", 0.0))
            df.at[i, "Omsättning om 3 år"] = _parse_sv_float(rad.get("Omsättning om 3 år", 0.0))

        # Riktkurser (kräver Utestående aktier > 0 och P/S-snitt > 0)
        aktier_ut = _parse_sv_float(rad.get("Utestående aktier", 0.0))
        if aktier_ut > 0 and ps_snitt > 0:
            df.at[i, "Riktkurs idag"]    = round((_parse_sv_float(rad.get("Omsättning idag", 0.0))     * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 1 år"] = round((_parse_sv_float(rad.get("Omsättning nästa år", 0.0)) * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 2 år"] = round(( _parse_sv_float(df.at[i, "Omsättning om 2 år"])     * ps_snitt) / aktier_ut, 2)
            df.at[i, "Riktkurs om 3 år"] = round(( _parse_sv_float(df.at[i, "Omsättning om 3 år"])     * ps_snitt) / aktier_ut, 2)
        else:
            df.at[i, "Riktkurs idag"] = 0.0
            df.at[i, "Riktkurs om 1 år"] = 0.0
            df.at[i, "Riktkurs om 2 år"] = 0.0
            df.at[i, "Riktkurs om 3 år"] = 0.0

    return df


# Fält som triggar tidsstämpel "Senast manuellt uppdaterad"
MANUELL_FALT_FOR_DATUM = [
    "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
    "Omsättning idag","Omsättning nästa år"
]

# ===== Slut DEL 3/6 =====

# ===== DEL 4/6 — MASSUPPDATERA & LÄGG TILL/UPPDATERA =====

# --- Hjälpetikett (visar när posten ändrades manuellt) ---
def _lbl_with_ts(base: str, df_row: pd.Series) -> str:
    if df_row is None or getattr(df_row, "empty", True):
        return base
    ts = str(df_row.get("Senast manuellt uppdaterad", "") or "—")
    return f"{base}  [{ts}]"


def _normalize_ticker(t: str) -> str:
    return (t or "").strip().upper()


# --- Massuppdatera från Yahoo (enkel, robust) ---
def massuppdatera(df: pd.DataFrame, key_prefix: str, user_rates: dict) -> pd.DataFrame:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Uppdatera alla från Yahoo", key=f"{key_prefix}_massupd_btn"):
        status = st.sidebar.empty()
        bar = st.sidebar.progress(0)
        misslyckade = []
        total = max(1, len(df))
        for i, row in df.iterrows():
            tkr = _normalize_ticker(row.get("Ticker", ""))
            if not tkr:
                continue
            status.write(f"Uppdaterar {i+1}/{len(df)} – {tkr}")

            data = hamta_yahoo_fält(tkr)
            failed = []

            if data.get("Bolagsnamn"): df.at[i, "Bolagsnamn"] = data["Bolagsnamn"]
            else: failed.append("Bolagsnamn")

            if _parse_sv_float(data.get("Aktuell kurs", 0.0)) > 0:
                df.at[i, "Aktuell kurs"] = float(data["Aktuell kurs"])
            else: failed.append("Aktuell kurs")

            if data.get("Valuta"):
                df.at[i, "Valuta"] = str(data["Valuta"])
            else: failed.append("Valuta")

            if "Årlig utdelning" in data:
                df.at[i, "Årlig utdelning"] = _parse_sv_float(data.get("Årlig utdelning"))
            else:
                failed.append("Årlig utdelning")

            if "CAGR 5 år (%)" in data:
                df.at[i, "CAGR 5 år (%)"] = _parse_sv_float(data.get("CAGR 5 år (%)"))
            else:
                failed.append("CAGR 5 år (%)")

            if failed:
                misslyckade.append(f"{tkr}: {', '.join(failed)}")

            time.sleep(0.6)  # undvik kvot
            bar.progress((i+1)/total)

        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)  # skriver med svensk-decimalformat
        st.sidebar.success("Klart! Alla bolag uppdaterade.")
        if misslyckade:
            st.sidebar.warning("Vissa fält kunde inte hämtas. Kopiera vid behov:")
            st.sidebar.text_area("Misslyckade fält", "\n".join(misslyckade), height=160, key=f"{key_prefix}_miss")

    return df


# --- Lägg till / uppdatera bolag (med dubblettkontroll & sv-decimalinputs) ---
def lagg_till_eller_uppdatera(df: pd.DataFrame, user_rates: dict) -> pd.DataFrame:
    st.header("➕ Lägg till / uppdatera bolag")

    # Sortering för redigering
    sort_val = st.selectbox("Sortera för redigering", ["A–Ö (bolagsnamn)", "Äldst manuell uppdatering först"])
    if sort_val.startswith("Äldst"):
        df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
        vis_df = df.sort_values(by=["_sort_datum", "Bolagsnamn"])
    else:
        vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"])

    namn_map = {f"{r['Bolagsnamn']} ({r['Ticker']})": r['Ticker'] for _, r in vis_df.iterrows()}
    val_lista = [""] + list(namn_map.keys())
    if "edit_index" not in st.session_state:
        st.session_state.edit_index = 0

    valt_label = st.selectbox("Välj bolag (lämna tomt för nytt)", val_lista,
                              index=min(st.session_state.edit_index, len(val_lista)-1))
    col_prev, col_pos, col_next = st.columns([1,2,1])
    with col_prev:
        if st.button("⬅️ Föregående"):
            st.session_state.edit_index = max(0, st.session_state.edit_index - 1)
    with col_pos:
        st.write(f"Post {st.session_state.edit_index}/{max(1, len(val_lista)-1)}")
    with col_next:
        if st.button("➡️ Nästa"):
            st.session_state.edit_index = min(len(val_lista)-1, st.session_state.edit_index + 1)

    if valt_label and valt_label in namn_map:
        bef = df[df["Ticker"] == namn_map[valt_label]].iloc[0]
    else:
        bef = pd.Series({}, dtype=object)

    with st.form("form_bolag"):
        c1, c2 = st.columns(2)
        with c1:
            ticker = st.text_input("Ticker (Yahoo-format)",
                                   value=bef.get("Ticker", "") if not bef.empty else "").upper().strip()

            utest = number_input_sv(_lbl_with_ts("Utestående aktier (miljoner)", bef),
                                    default=_parse_sv_float(bef.get("Utestående aktier", 0.0)) if not bef.empty else 0.0)

            antal = number_input_sv("Antal aktier du äger",
                                    default=_parse_sv_float(bef.get("Antal aktier", 0.0)) if not bef.empty else 0.0)

            gav_sek = number_input_sv("GAV (SEK)",
                                      default=_parse_sv_float(bef.get("GAV (SEK)", 0.0)) if not bef.empty else 0.0)

            ps  = number_input_sv(_lbl_with_ts("P/S (TTM)", bef),
                                  default=_parse_sv_float(bef.get("P/S", 0.0)) if not bef.empty else 0.0)
            ps1 = number_input_sv(_lbl_with_ts("P/S Q1", bef),
                                  default=_parse_sv_float(bef.get("P/S Q1", 0.0)) if not bef.empty else 0.0)
            ps2 = number_input_sv(_lbl_with_ts("P/S Q2", bef),
                                  default=_parse_sv_float(bef.get("P/S Q2", 0.0)) if not bef.empty else 0.0)
            ps3 = number_input_sv(_lbl_with_ts("P/S Q3", bef),
                                  default=_parse_sv_float(bef.get("P/S Q3", 0.0)) if not bef.empty else 0.0)
            ps4 = number_input_sv(_lbl_with_ts("P/S Q4", bef),
                                  default=_parse_sv_float(bef.get("P/S Q4", 0.0)) if not bef.empty else 0.0)

        with c2:
            oms_idag = number_input_sv(_lbl_with_ts("Omsättning idag (miljoner)", bef),
                                       default=_parse_sv_float(bef.get("Omsättning idag", 0.0)) if not bef.empty else 0.0)
            oms_next = number_input_sv(_lbl_with_ts("Omsättning nästa år (miljoner)", bef),
                                       default=_parse_sv_float(bef.get("Omsättning nästa år", 0.0)) if not bef.empty else 0.0)

            st.caption("Följande hämtas automatiskt vid spara: Bolagsnamn, Valuta, Aktuell kurs, Årlig utdelning, CAGR 5 år (%)")
            st.caption("P/S-snitt, Omsättning år 2 & 3 samt Riktkurser räknas om automatiskt.")

        spar = st.form_submit_button("💾 Spara & hämta från Yahoo")

    if spar and ticker:
        # === DUBBLETTKONTROLL (case-insensitiv, trim) ===
        new_tkr = _normalize_ticker(ticker)
        cur_tkr = _normalize_ticker(bef.get("Ticker", "") if not bef.empty else "")
        tkr_norm = df["Ticker"].astype(str).str.strip().str.upper()

        if bef.empty:
            if (tkr_norm == new_tkr).any():
                st.error(f"Tickern **{new_tkr}** finns redan i databasen. Välj den i listan för att redigera.")
                st.stop()
        else:
            if new_tkr != cur_tkr and (tkr_norm == new_tkr).any():
                st.error(f"Kan inte byta till tickern **{new_tkr}** – den finns redan i en annan rad.")
                st.stop()
        # ================================================

        ny = {
            "Ticker": new_tkr,
            "Utestående aktier": _parse_sv_float(utest),
            "Antal aktier": _parse_sv_float(antal),
            "GAV (SEK)": _parse_sv_float(gav_sek),
            "P/S": _parse_sv_float(ps),
            "P/S Q1": _parse_sv_float(ps1),
            "P/S Q2": _parse_sv_float(ps2),
            "P/S Q3": _parse_sv_float(ps3),
            "P/S Q4": _parse_sv_float(ps4),
            "Omsättning idag": _parse_sv_float(oms_idag),
            "Omsättning nästa år": _parse_sv_float(oms_next),
        }

        # Sätt “Senast manuellt uppdaterad” om något av manuella fälten har ändrats
        datum_sätt = False
        if not bef.empty:
            before = {f: _parse_sv_float(bef.get(f, 0.0)) for f in MANUELL_FALT_FOR_DATUM}
            after  = {f: _parse_sv_float(ny.get(f, 0.0))  for f in MANUELL_FALT_FOR_DATUM}
            if any(before[k] != after[k] for k in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True
        else:
            if any(_parse_sv_float(ny.get(f, 0.0)) != 0.0 for f in MANUELL_FALT_FOR_DATUM):
                datum_sätt = True

        # Skriv in i df (ny rad eller uppdatera existerande)
        if not bef.empty:
            # Om användaren ändrade tickern, uppdatera radens nyckel först
            if new_tkr != cur_tkr:
                df.loc[df["Ticker"] == cur_tkr, "Ticker"] = new_tkr
            for k, v in ny.items():
                df.loc[df["Ticker"] == new_tkr, k] = v
        else:
            tom = {c: (0.0 if c in _NUMERIC_COLS else "") for c in FINAL_COLS}
            tom.update(ny)
            df = pd.concat([df, pd.DataFrame([tom])], ignore_index=True)

        if datum_sätt:
            df.loc[df["Ticker"] == new_tkr, "Senast manuellt uppdaterad"] = now_stamp()

        # Hämta automatiska fält från Yahoo
        data = hamta_yahoo_fält(new_tkr)
        if data.get("Bolagsnamn"): df.loc[df["Ticker"] == new_tkr, "Bolagsnamn"] = data["Bolagsnamn"]
        if data.get("Valuta"):     df.loc[df["Ticker"] == new_tkr, "Valuta"] = data["Valuta"]
        if _parse_sv_float(data.get("Aktuell kurs", 0.0)) > 0: df.loc[df["Ticker"] == new_tkr, "Aktuell kurs"] = float(data["Aktuell kurs"])
        if "Årlig utdelning" in data: df.loc[df["Ticker"] == new_tkr, "Årlig utdelning"] = _parse_sv_float(data.get("Årlig utdelning"))
        if "CAGR 5 år (%)" in data:   df.loc[df["Ticker"] == new_tkr, "CAGR 5 år (%)"] = _parse_sv_float(data.get("CAGR 5 år (%)"))

        # Beräkna & spara
        df = uppdatera_berakningar(df, user_rates)
        spara_data(df)  # skriver med svensk-decimalformat
        st.success("Sparat och uppdaterat från Yahoo.")
        st.rerun()

    # Lista: äldst manuellt uppdaterade
    st.markdown("### ⏱️ Äldst manuellt uppdaterade (topp 10)")
    df["_sort_datum"] = df["Senast manuellt uppdaterad"].replace("", "0000-00-00")
    tips = df.sort_values(by=["_sort_datum", "Bolagsnamn"]).head(10)
    st.dataframe(
        tips[[
            "Ticker","Bolagsnamn","Senast manuellt uppdaterad",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4",
            "Omsättning idag","Omsättning nästa år"
        ]],
        use_container_width=True
    )

    return df

# ===== Slut DEL 4/6 =====

# ===== DEL 5/6 — ANALYS, PORTFÖLJ & INVESTERINGSFÖRSLAG =====

def analysvy(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📈 Analys")
    if df.empty:
        st.info("Databasen är tom.")
        return

    vis_df = df.sort_values(by=["Bolagsnamn", "Ticker"]).reset_index(drop=True)
    etiketter = [f"{r['Bolagsnamn']} ({r['Ticker']})" for _, r in vis_df.iterrows()]

    if "analys_idx" not in st.session_state:
        st.session_state.analys_idx = 0

    st.session_state.analys_idx = st.number_input(
        "Visa bolag #",
        min_value=0,
        max_value=max(0, len(etiketter) - 1),
        value=st.session_state.analys_idx,
        step=1,
    )
    st.selectbox(
        "Eller välj i lista",
        etiketter,
        index=st.session_state.analys_idx if etiketter else 0,
        key="analys_select",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("⬅️ Föregående", key="analys_prev"):
            st.session_state.analys_idx = max(0, st.session_state.analys_idx - 1)
    with col_b:
        if st.button("➡️ Nästa", key="analys_next"):
            st.session_state.analys_idx = min(len(etiketter) - 1, st.session_state.analys_idx + 1)

    st.write(f"Post {st.session_state.analys_idx+1}/{len(etiketter) if etiketter else 1}")

    if len(vis_df) > 0:
        r = vis_df.iloc[st.session_state.analys_idx]
        st.subheader(f"{r['Bolagsnamn']} ({r['Ticker']})")
        cols = [
            "Ticker","Bolagsnamn","Valuta","Aktuell kurs","Utestående aktier",
            "P/S","P/S Q1","P/S Q2","P/S Q3","P/S Q4","P/S-snitt",
            "Omsättning idag","Omsättning nästa år","Omsättning om 2 år","Omsättning om 3 år",
            "Riktkurs idag","Riktkurs om 1 år","Riktkurs om 2 år","Riktkurs om 3 år",
            "CAGR 5 år (%)","Antal aktier","GAV (SEK)","Årlig utdelning","Senast manuellt uppdaterad"
        ]
        show = {c: r.get(c, "") for c in cols if c in df.columns}
        st.dataframe(pd.DataFrame([show]), use_container_width=True)

    st.markdown("### Hela databasen")
    st.dataframe(df, use_container_width=True)


def visa_portfolj(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("📦 Min portfölj")
    port = df[df["Antal aktier"] > 0].copy()
    if port.empty:
        st.info("Du äger inga aktier.")
        return

    # Växelkurs och marknadsvärde
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]

    # Anskaffningsvärde, P/L
    port["Anskaffningsvärde (SEK)"] = port["Antal aktier"] * port["GAV (SEK)"]
    port["Vinst/Förlust (SEK)"] = port["Värde (SEK)"] - port["Anskaffningsvärde (SEK)"]
    port["Vinst/Förlust (%)"] = np.where(
        port["Anskaffningsvärde (SEK)"] > 0,
        (port["Vinst/Förlust (SEK)"] / port["Anskaffningsvärde (SEK)"]) * 100.0,
        0.0,
    )

    # Andelar och utdelning
    total_värde = float(port["Värde (SEK)"].sum())
    port["Andel (%)"] = np.where(total_värde > 0, round(port["Värde (SEK)"] / total_värde * 100.0, 2), 0.0)
    port["Total årlig utdelning (SEK)"] = port["Antal aktier"] * port["Årlig utdelning"] * port["Växelkurs"]
    tot_utd = float(port["Total årlig utdelning (SEK)"].sum())
    tot_ansk = float(port["Anskaffningsvärde (SEK)"].sum())
    tot_pl = float(port["Vinst/Förlust (SEK)"].sum())
    tot_pl_pct = (tot_pl / tot_ansk * 100.0) if tot_ansk > 0 else 0.0

    st.markdown(f"**Totalt portföljvärde:** {round(total_värde,2)} SEK")
    st.markdown(f"**Totalt anskaffningsvärde:** {round(tot_ansk,2)} SEK")
    st.markdown(f"**Orealiserad vinst/förlust:** {round(tot_pl,2)} SEK ({round(tot_pl_pct,2)} %)")
    st.markdown(f"**Total kommande utdelning:** {round(tot_utd,2)} SEK")
    st.markdown(f"**Ungefärlig månadsutdelning:** {round(tot_utd/12.0,2)} SEK")

    st.dataframe(
        port[
            [
                "Ticker","Bolagsnamn","Antal aktier","GAV (SEK)","Anskaffningsvärde (SEK)",
                "Aktuell kurs","Valuta","Växelkurs","Värde (SEK)",
                "Vinst/Förlust (SEK)","Vinst/Förlust (%)",
                "Årlig utdelning","Total årlig utdelning (SEK)","Andel (%)",
            ]
        ],
        use_container_width=True,
    )


def visa_investeringsforslag(df: pd.DataFrame, user_rates: dict) -> None:
    st.header("💡 Investeringsförslag")
    if df.empty:
        st.info("Databasen är tom.")
        return

    kapital_sek = number_input_sv("Tillgängligt kapital (SEK)", default=500.0, step=100.0)

    riktkurs_val = st.selectbox(
        "Vilken riktkurs ska användas?",
        ["Riktkurs idag", "Riktkurs om 1 år", "Riktkurs om 2 år", "Riktkurs om 3 år"],
        index=1,
    )

    subset = st.radio("Vilka bolag?", ["Alla bolag", "Endast portfölj"], horizontal=True)
    läge = st.radio("Sortering", ["Störst potential", "Närmast riktkurs"], horizontal=True)

    # 🔽 P/S-filter (nuvarande P/S vs P/S-snitt)
    ps_filter = st.selectbox(
        "Filtrera på P/S i förhållande till P/S-snitt",
        ["Alla", "P/S under snitt", "P/S över snitt"],
        index=0,
    )

    base = df[df["Antal aktier"] > 0].copy() if subset == "Endast portfölj" else df.copy()
    base = base[(base[riktkurs_val] > 0) & (base["Aktuell kurs"] > 0)].copy()

    # Tillämpa P/S-filter (kräv att båda finns och > 0)
    if ps_filter == "P/S under snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] < base["P/S-snitt"])].copy()
    elif ps_filter == "P/S över snitt":
        base = base[(base["P/S"] > 0) & (base["P/S-snitt"] > 0) & (base["P/S"] > base["P/S-snitt"])].copy()

    if base.empty:
        st.info("Inga bolag matchar just nu.")
        return

    base["Potential (%)"] = (base[riktkurs_val] - base["Aktuell kurs"]) / base["Aktuell kurs"] * 100.0
    base["Diff till mål (%)"] = (base["Aktuell kurs"] - base[riktkurs_val]) / base[riktkurs_val] * 100.0

    if läge == "Störst potential":
        base = base.sort_values(by="Potential (%)", ascending=False).reset_index(drop=True)
    else:
        base["absdiff"] = base["Diff till mål (%)"].abs()
        base = base.sort_values(by="absdiff", ascending=True).reset_index(drop=True)

    # Bläddring
    if "forslags_index" not in st.session_state:
        st.session_state.forslags_index = 0
    st.session_state.forslags_index = min(st.session_state.forslags_index, len(base) - 1)

    col_prev, col_mid, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("⬅️ Föregående förslag"):
            st.session_state.forslags_index = max(0, st.session_state.forslags_index - 1)
    with col_mid:
        st.write(f"Förslag {st.session_state.forslags_index+1}/{len(base)}")
    with col_next:
        if st.button("➡️ Nästa förslag"):
            st.session_state.forslags_index = min(len(base) - 1, st.session_state.forslags_index + 1)

    rad = base.iloc[st.session_state.forslags_index]

    # Portföljandelar
    port = df[df["Antal aktier"] > 0].copy()
    port["Växelkurs"] = port["Valuta"].apply(lambda v: hamta_valutakurs(v, user_rates))
    port["Värde (SEK)"] = port["Antal aktier"] * port["Aktuell kurs"] * port["Växelkurs"]
    port_värde = float(port["Värde (SEK)"].sum()) if not port.empty else 0.0

    vx = hamta_valutakurs(rad["Valuta"], user_rates)
    kurs_sek = rad["Aktuell kurs"] * vx
    antal_köp = int(_parse_sv_float(kapital_sek) // max(kurs_sek, 1e-9))
    investering = antal_köp * kurs_sek

    nuv_innehav = 0.0
    if not port.empty:
        r = port[port["Ticker"] == rad["Ticker"]]
        if not r.empty:
            nuv_innehav = float(r["Värde (SEK)"].sum())
    ny_total = nuv_innehav + investering
    nuv_andel = round((nuv_innehav / port_värde) * 100.0, 2) if port_värde > 0 else 0.0
    ny_andel  = round((ny_total   / port_värde) * 100.0, 2) if port_värde > 0 else 0.0

    st.subheader(f"{rad['Bolagsnamn']} ({rad['Ticker']})")
    st.markdown(
        f"""
- **Aktuell kurs:** {round(rad['Aktuell kurs'], 2)} {rad['Valuta']}
- **Nuvarande P/S (TTM):** {round(rad.get('P/S', 0.0), 2)}
- **P/S-snitt (Q1–Q4):** {round(rad.get('P/S-snitt', 0.0), 2)}
- **Riktkurs idag:** {round(rad['Riktkurs idag'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs idag" else ""}
- **Riktkurs om 1 år:** {round(rad['Riktkurs om 1 år'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs om 1 år" else ""}
- **Riktkurs om 2 år:** {round(rad['Riktkurs om 2 år'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs om 2 år" else ""}
- **Riktkurs om 3 år:** {round(rad['Riktkurs om 3 år'], 2)} {rad['Valuta']} {"**⬅ vald**" if riktkurs_val == "Riktkurs om 3 år" else ""}
- **Uppsida (valda riktkursen):** {round(rad['Potential (%)'], 2)} %
- **Antal att köpa för {int(_parse_sv_float(kapital_sek))} SEK:** {antal_köp} st
- **Nuvarande andel:** {nuv_andel} %
- **Andel efter köp:** {ny_andel} %
"""
    )

# ===== Slut DEL 5/6 =====

# ===== DEL 6/6 — MAIN & APP-ENTRY =====

def main():
    st.title("📊 Aktieanalys och investeringsförslag")

    # 1) Hämta live-valutor tyst om sparade skiljer sig
    auto_update_valutakurser_if_stale()

    # 2) Sidopanel: valutakurser (svenska decimaler) + spara/läs
    st.sidebar.header("💱 Valutakurser → SEK")
    saved_rates = las_sparade_valutakurser()

    usd_in = number_input_sv("USD → SEK", default=saved_rates.get("USD", 9.75), step=0.01, key="fx_usd")
    nok_in = number_input_sv("NOK → SEK", default=saved_rates.get("NOK", 0.95), step=0.01, key="fx_nok")
    cad_in = number_input_sv("CAD → SEK", default=saved_rates.get("CAD", 7.05), step=0.01, key="fx_cad")
    eur_in = number_input_sv("EUR → SEK", default=saved_rates.get("EUR", 11.18), step=0.01, key="fx_eur")

    usd = _parse_sv_float(usd_in)
    nok = _parse_sv_float(nok_in)
    cad = _parse_sv_float(cad_in)
    eur = _parse_sv_float(eur_in)

    user_rates = {"USD": usd, "NOK": nok, "CAD": cad, "EUR": eur, "SEK": 1.0}

    col_fx1, col_fx2, col_fx3 = st.sidebar.columns(3)
    with col_fx1:
        if st.button("💾 Spara", key="fx_save"):
            spara_valutakurser(user_rates)
            st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
            st.sidebar.success("Valutakurser sparade.")
    with col_fx2:
        if st.button("↻ Läs", key="fx_reload"):
            st.cache_data.clear()
            st.rerun()
    with col_fx3:
        if st.button("🌐 Yahoo", key="fx_yahoo"):
            live = hamta_valutakurser_automatiskt()
            if live and any(k in live for k in ("USD", "NOK", "CAD", "EUR")):
                merged = las_sparade_valutakurser()
                merged.update(live)
                spara_valutakurser(merged)
                st.session_state["rates_reload"] = st.session_state.get("rates_reload", 0) + 1
                st.sidebar.success("Valutakurser uppdaterade från Yahoo.")
                st.rerun()
            else:
                st.sidebar.error("Kunde inte hämta kurser just nu (Yahoo). Försök igen senare.")

    st.sidebar.markdown("---")
    if st.sidebar.button("↻ Läs om data från Google Sheets", key="reload_sheet"):
        st.cache_data.clear()
        st.rerun()

    # 3) Läs databasen
    df = hamta_data()
    if df.empty:
        df = pd.DataFrame({c: [] for c in FINAL_COLS})
        spara_data(df)

    # 4) Säkerställ schema/typer (svenska decimaler hanteras i inmatning & vid spar)
    df = säkerställ_kolumner(df)
    df = migrera_gamla_riktkurskolumner(df)
    df = konvertera_typer(df)

    # 5) Global massuppdatering (Yahoo, ej SEC i denna version)
    df = massuppdatera(df, key_prefix="global", user_rates=user_rates)

    # 6) Meny
    meny = st.sidebar.radio("📌 Välj vy", ["Analys", "Lägg till / uppdatera bolag", "Investeringsförslag", "Portfölj"])

    if meny == "Analys":
        analysvy(df, user_rates)
    elif meny == "Lägg till / uppdatera bolag":
        df = lagg_till_eller_uppdatera(df, user_rates)
    elif meny == "Investeringsförslag":
        df = uppdatera_berakningar(df, user_rates)
        visa_investeringsforslag(df, user_rates)
    elif meny == "Portfölj":
        df = uppdatera_berakningar(df, user_rates)
        visa_portfolj(df, user_rates)


if __name__ == "__main__":
    main()
# ===== Slut DEL 6/6 =====
