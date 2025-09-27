# stockapp/sources.py
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime

try:
    import pytz
    TZ_STHLM = pytz.timezone("Europe/Stockholm")
    def _today_stamp():
        return datetime.now(TZ_STHLM).strftime("%Y-%m-%d")
except Exception:
    def _today_stamp():
        return datetime.now().strftime("%Y-%m-%d")

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _yahoo_last_price_currency_name(ticker: str):
    """
    Returnerar (price, currency, name) från yfinance.
    Prioriterar fast_info -> info -> history fallback.
    """
    import yfinance as yf

    t = yf.Ticker(ticker)

    # 1) fast_info
    try:
        fi = getattr(t, "fast_info", None) or {}
        px = _safe_float(fi.get("last_price"))
        cur = fi.get("currency")
        nm = None
        if px:
            return px, cur, nm
    except Exception:
        pass

    # 2) info
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    px = _safe_float(info.get("regularMarketPrice"))
    cur = info.get("currency")
    nm = info.get("shortName") or info.get("longName")
    if px:
        return px, cur, nm

    # 3) history fallback
    try:
        hist = t.history(period="1d")
        if hist is not None and not hist.empty and "Close" in hist:
            px = _safe_float(hist["Close"].iloc[-1])
            if px:
                return px, cur, nm
    except Exception:
        pass

    return None, cur, nm

def update_price_only_runner(df: pd.DataFrame, ticker: str, user_rates: dict):
    """
    Runner-signatur som batchpanelen förväntar sig:
        (df_new, changed_fields|list, err|None)

    - Uppdaterar 'Aktuell kurs' (och 'Valuta'/'Bolagsnamn' om vi hittar dem).
    - Sätter alltid 'Senast auto-uppdaterad' + 'Senast uppdaterad källa'.
    - Returnerar 'Aktuell kurs' i changed_fields även om värdet är detsamma
      (så batch-loggen inte blir "miss" i onödan).
    """
    if df is None or df.empty or "Ticker" not in df.columns:
        return df, [], f"Ingen data eller saknar kolumnen 'Ticker'."

    mask = df["Ticker"].astype(str).str.upper() == str(ticker).strip().upper()
    idx_list = df.index[mask].tolist()
    if not idx_list:
        return df, [], f"{ticker} hittades inte i tabellen."
    ridx = idx_list[0]

    # hämta pris/valuta/namn
    try:
        price, currency, name = _yahoo_last_price_currency_name(ticker)
    except Exception as e:
        return df, [], f"Fel vid pris-hämtning: {e}"

    changed = []

    # pris (räkna alltid som ändrat om vi fick ett pris)
    if price is not None and price > 0:
        df.at[ridx, "Aktuell kurs"] = float(price)
        changed.append("Aktuell kurs")

    # valuta
    if currency and "Valuta" in df.columns:
        cur_up = str(currency).upper()
        if str(df.at[ridx, "Valuta"]).upper() != cur_up:
            df.at[ridx, "Valuta"] = cur_up
            changed.append("Valuta")

    # bolagsnamn
    if name and "Bolagsnamn" in df.columns:
        if str(df.at[ridx, "Bolagsnamn"]).strip() != str(name).strip():
            df.at[ridx, "Bolagsnamn"] = str(name).strip()
            changed.append("Bolagsnamn")

    # meta: senast auto-uppdaterad + källa
    if "Senast auto-uppdaterad" in df.columns:
        df.at[ridx, "Senast auto-uppdaterad"] = _today_stamp()
    if "Senast uppdaterad källa" in df.columns:
        df.at[ridx, "Senast uppdaterad källa"] = "Auto (Yahoo/yfinance: price-only)"

    if not changed:
        return df, [], "Ingen ny kurs hittades."
    return df, changed, None
