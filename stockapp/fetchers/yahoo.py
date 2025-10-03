# stockapp/fetchers/yahoo.py
from __future__ import annotations
import pandas as pd
import yfinance as yf

def _cagr_from_financials(tkr: yf.Ticker) -> float:
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
        series = series.sort_index()
        start = float(series.iloc[0])
        end   = float(series.iloc[-1])
        years = max(1, len(series)-1)
        if start <= 0:
            return 0.0
        cagr = (end / start) ** (1.0/years) - 1.0
        return round(cagr * 100.0, 2)
    except Exception:
        return 0.0

def get_all(ticker: str) -> dict:
    """
    Returnerar ett litet paket med vanliga fält för enkel sammanslagning.
    Nycklarna är “neutrala” (engelska) för att minska krockar.
    """
    out = {
        "name": "",
        "price": 0.0,
        "currency": "USD",
        "dividend": 0.0,
        "cagr5": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        pris = info.get("regularMarketPrice")
        if pris is None:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                pris = float(h["Close"].iloc[-1])
        if pris is not None:
            out["price"] = float(pris)

        valuta = info.get("currency")
        if valuta:
            out["currency"] = str(valuta).upper()

        namn = info.get("shortName") or info.get("longName") or ""
        if namn:
            out["name"] = str(namn)

        div_rate = info.get("dividendRate")
        if div_rate is not None:
            out["dividend"] = float(div_rate)

        out["cagr5"] = _cagr_from_financials(t)
    except Exception:
        pass
    return out
