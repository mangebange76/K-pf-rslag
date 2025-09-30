# stockapp/batch.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional, Tuple, Dict

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# För sortering "Äldst först"
from .config import TS_FIELDS  # dict över spårade fält -> TS-kolumn
from .utils import now_stamp

# För uppdatering (om modul finns)
_HAVE_UPDATE = False
try:
    from .update import run_update_price_only, run_update_full  # type: ignore
    _HAVE_UPDATE = True
except Exception:
    _HAVE_UPDATE = False

# Fallback: yfinance för snabb kursuppdatering om update-modul saknas
try:
    import yfinance as yf
    _HAVE_YF = True
except Exception:
    _HAVE_YF = False


# ---------------------------------------------------------------------
# Hjälpare: äldsta tidsstämpeln per rad (bland alla TS_-kolumner)
# ---------------------------------------------------------------------
def _oldest_any_ts(row: pd.Series) -> Optional[pd.Timestamp]:
    dates = []
    for f, ts_col in TS_FIELDS.items():
        if ts_col in row and str(row[ts_col]).strip():
            try:
                d = pd.to_datetime(str(row[ts_col]).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else None


def _add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    work = df.copy()
    work["_oldest_any_ts"] = work.apply(_oldest_any_ts, axis=1)
    # fyll för sortering
    work["_oldest_any_ts_fill"] = work["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return work


# ---------------------------------------------------------------------
# Sortering
# ---------------------------------------------------------------------
def _pick_order(df: pd.DataFrame, mode: str) -> List[str]:
    """
    Returnerar en lista med tickers i den valda ordningen.
    mode: "A–Ö" eller "Äldst först"
    """
    if df is None or df.empty or "Ticker" not in df.columns:
        return []

    if mode == "Äldst först":
        work = _add_oldest_ts_col(df)
        work = work.sort_values(by=["_oldest_any_ts_fill", "Bolagsnamn", "Ticker"], ascending=[True, True, True])
        return [str(t).upper() for t in work["Ticker"].tolist()]

    # default: A–Ö
    work = df.copy()
    if "Bolagsnamn" in work.columns:
        work = work.sort_values(by=["Bolagsnamn", "Ticker"], ascending=[True, True])
    else:
        work = work.sort_values(by=["Ticker"], ascending=[True])
    return [str(t).upper() for t in work["Ticker"].tolist()]


# ---------------------------------------------------------------------
# Fallback: uppdatera kurs via yfinance om update-modul saknas
# ---------------------------------------------------------------------
def _fallback_update_price_only(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, str]:
    if not _HAVE_YF:
        return df, "yfinance saknas – kan inte uppdatera kurs."
    tkr = str(ticker).upper().strip()
    if not tkr:
        return df, "Tom ticker."
    if "Ticker" not in df.columns or tkr not in [str(x).upper() for x in df["Ticker"].tolist()]:
        return df, f"{tkr}: hittades inte i tabellen."

    t = yf.Ticker(tkr)
    price = None
    currency = None
    name = None

    try:
        info = t.info or {}
        price = info.get("regularMarketPrice")
        currency = (info.get("currency") or "").upper() if info.get("currency") else None
        name = info.get("shortName") or info.get("longName")
    except Exception:
        # fallback på hist
        try:
            h = t.history(period="1d")
            if not h.empty and "Close" in h:
                price = float(h["Close"].iloc[-1])
        except Exception:
            pass

    idx = df.index[df["Ticker"].astype(str).str.upper() == tkr]
    if len(idx) == 0:
        return df, f"{tkr}: hittades inte i tabellen."
    ridx = idx[0]

    changed = False
    if price is not None and price > 0:
        if float(df.at[ridx, "Aktuell kurs"]) != float(price):
            df.at[ridx, "Aktuell kurs"] = float(price)
            changed = True

    if name:
        if str(df.at[ridx, "Bolagsnamn"]).strip() != str(name).strip():
            df.at[ridx, "Bolagsnamn"] = str(name)
            changed = True

    if currency:
        if str(df.at[ridx, "Valuta"]).upper() != currency:
            df.at[ridx, "Valuta"] = currency
            changed = True

    # Stämpla "Senast auto-uppdaterad" & källa, även om kursen råkar vara samma
    df.at[ridx, "Senast auto-uppdaterad"] = now_stamp()
    df.at[ridx, "Senast uppdaterad källa"] = "Batch (Yahoo pris fallback)"

    if changed:
        return df, f"{tkr}: pris uppdaterat."
    else:
        return df, f"{tkr}: inga värdeförändringar, men tidsstämpel satt."


# ---------------------------------------------------------------------
# Kör uppdatering för en ticker (enligt valt läge)
# ---------------------------------------------------------------------
def _run_one(df: pd.DataFrame, ticker: str, mode: str, user_rates: Dict[str, float]) -> Tuple[pd.DataFrame, str]:
    tkr = str(ticker).upper().strip()
    if not tkr:
        return df, "Tom ticker."

    # Endast kurs
    if mode == "Endast kurs":
        if _HAVE_UPDATE:
            # tolerant signatur
            try:
                df2, msg = run_update_price_only(df.copy(), user_rates, tkr)  # type: ignore
                return df2, msg or f"{tkr}: kurs uppdaterad."
            except TypeError:
                try:
                    df2, msg = run_update_price_only(df.copy(), tkr)  # type: ignore
                    return df2, msg or f"{tkr}: kurs uppdaterad."
                except TypeError:
                    df2, msg = run_update_price_only(tkr, df.copy())  # type: ignore
                    return df2, msg or f"{tkr}: kurs uppdaterad."
            except Exception as e:
                return df, f"{tkr}: fel i run_update_price_only – {e}"
        else:
            # fallback
            return _fallback_update_price_only(df.copy(), tkr)

    # Full auto
    if _HAVE_UPDATE:
        try:
            df2, log, note = run_update_full(df.copy(), user_rates, tkr)  # type: ignore
            msg = note or f"{tkr}: full auto OK"
            return df2, msg
        except TypeError:
            try:
                df2, log, note = run_update_full(df.copy(), tkr, user_rates)  # type: ignore
                msg = note or f"{tkr}: full auto OK"
                return df2, msg
            except TypeError:
                df2, log = run_update_full(df.copy(), tkr)  # type: ignore
                msg = f"{tkr}: full auto OK (utan note)"
                return df2, msg
        except Exception as e:
            return df, f"{tkr}: fel i run_update_full – {e}"
    else:
        # fallback till pris-uppdatering
        df2, m = _fallback_update_price_only(df.copy(), tkr)
        return df2, f"{m} (full auto fallback: pris)"

# ---------------------------------------------------------------------
# Kö-hantering
# ---------------------------------------------------------------------
def _init_state_defaults():
    # Säkra nycklar innan widgets
    st.session_state.setdefault("_batch_queue", [])      # List[str]
    st.session_state.setdefault("_batch_done", set())    # Set[str]
    st.session_state.setdefault("_batch_cursor", 0)      # int (offset i sorted order)
    st.session_state.setdefault("_batch_logs", [])       # List[dict]
    st.session_state.setdefault("_batch_mode", "Full auto")
    st.session_state.setdefault("_batch_sort_mode", "Äldst först")
    st.session_state.setdefault("_batch_n", 10)


def _enqueue_next(df: pd.DataFrame, sort_mode: str, n: int):
    order = _pick_order(df, sort_mode)
    q: List[str] = list(st.session_state["_batch_queue"])
    done: set = set(st.session_state["_batch_done"])
    cur = int(st.session_state["_batch_cursor"])

    added = 0
    i = cur
    while i < len(order) and added < n:
        t = order[i]
        i += 1
        if t in q or t in done:
            continue
        q.append(t)
        added += 1

    st.session_state["_batch_queue"] = q
    st.session_state["_batch_cursor"] = i  # flytta fram “pekaren”
    return added, len(q)


# ---------------------------------------------------------------------
# Publik: Sidopanels-UI
# ---------------------------------------------------------------------
def sidebar_batch_controls(df: pd.DataFrame, user_rates: Dict[str, float]) -> Optional[pd.DataFrame]:
    """
    Bygger batch-panelen i sidopanelen, kör uppdateringar och returnerar ett nytt df
    om något faktiskt har ändrats (annars None).
    """
    if df is None or df.empty or "Ticker" not in df.columns:
        st.sidebar.info("Ingen data att batcha just nu.")
        return None

    _init_state_defaults()

    # --- UI ---
    st.sidebar.markdown("### 🧰 Batch-uppdatering")

    # Välj sortering & läge (keys stabila!)
    sort_mode = st.sidebar.selectbox("Sortering", ["Äldst först", "A–Ö"], key="_batch_sort_mode")
    mode = st.sidebar.radio("Uppdateringstyp", ["Full auto", "Endast kurs"], horizontal=True, key="_batch_mode")

    col_top1, col_top2 = st.sidebar.columns([1, 1])
    with col_top1:
        n = st.number_input("Antal att lägga i kö", min_value=1, max_value=200, value=int(st.session_state["_batch_n"]), step=1, key="_batch_n")
    with col_top2:
        add_click = st.button("➕ Lägg till N")

    if add_click:
        added, qlen = _enqueue_next(df, sort_mode, int(n))
        if added > 0:
            st.sidebar.success(f"Lade till {added} tickers i kön (totalt {qlen}).")
        else:
            st.sidebar.info("Inga fler tickers att lägga till just nu (kön kan vara full eller slut på listan).")

    # Visa aktuell kö
    q: List[str] = list(st.session_state["_batch_queue"])
    if q:
        st.sidebar.caption(f"Kö ({len(q)}): " + ", ".join(q[:20]) + (" …" if len(q) > 20 else ""))
    else:
        st.sidebar.caption("Kö: (tom)")

    # Kör-knappar
    col_run1, col_run2, col_run3 = st.sidebar.columns([1, 1, 1])
    do_next = col_run1.button("▶ Kör nästa")
    do_all  = col_run2.button("⏩ Kör hela kön")
    do_clear = col_run3.button("🧹 Töm kö")

    # Extra åtgärder
    col_extra1, col_extra2 = st.sidebar.columns([1, 1])
    do_reset_cursor = col_extra1.button("🔄 Nollställ cursor")
    do_skip = col_extra2.button("⤼ Hoppa över första")

    if do_clear:
        st.session_state["_batch_queue"] = []
        st.sidebar.info("Kön rensad.")

    if do_reset_cursor:
        st.session_state["_batch_cursor"] = 0
        st.sidebar.info("Cursor nollställd – nästa 'Lägg till N' börjar från början av listan (med exkludering av 'Done').")

    if do_skip and q:
        skipped = q.pop(0)
        st.session_state["_batch_queue"] = q
        st.sidebar.info(f"Hoppade över {skipped}.")

    # --- Kör nästa ---
    df_out: Optional[pd.DataFrame] = None
    changed_any = False

    if do_next and q:
        tkr = q.pop(0)
        st.session_state["_batch_queue"] = q
        try:
            df2, msg = _run_one(df.copy(), tkr, mode, user_rates)
            if isinstance(df2, pd.DataFrame) and not df2.equals(df):
                df = df2
                df_out = df2
                changed_any = True
            st.session_state["_batch_logs"].append({"ticker": tkr, "status": "ok", "msg": msg, "ts": now_stamp(), "mode": mode})
            # markera done
            done: set = set(st.session_state["_batch_done"])
            done.add(tkr)
            st.session_state["_batch_done"] = done
            st.sidebar.success(msg)
        except Exception as e:
            st.session_state["_batch_logs"].append({"ticker": tkr, "status": "error", "msg": str(e), "ts": now_stamp(), "mode": mode})
            st.sidebar.error(f"{tkr}: {e}")

    # --- Kör hela kön ---
    if do_all and q:
        total = len(q)
        prog = st.sidebar.progress(0.0, text=f"0/{total}")
        local_q = q[:]  # kopia
        processed = 0

        for t in local_q:
            try:
                df2, msg = _run_one(df.copy(), t, mode, user_rates)
                if isinstance(df2, pd.DataFrame) and not df2.equals(df):
                    df = df2
                    df_out = df2
                    changed_any = True
                st.session_state["_batch_logs"].append({"ticker": t, "status": "ok", "msg": msg, "ts": now_stamp(), "mode": mode})
                # markera done
                done: set = set(st.session_state["_batch_done"])
                done.add(t)
                st.session_state["_batch_done"] = done
                # poppa från kön
                q.pop(0)
            except Exception as e:
                st.session_state["_batch_logs"].append({"ticker": t, "status": "error", "msg": str(e), "ts": now_stamp(), "mode": mode})

            processed += 1
            frac = processed / max(1, total)
            prog.progress(frac, text=f"{processed}/{total}")

        st.session_state["_batch_queue"] = q
        st.sidebar.success(f"Klar: {processed}/{total} tickers körda.")

    # Loggvisning
    with st.sidebar.expander("📒 Senaste batchlogg (senast 30)", expanded=False):
        logs = list(st.session_state["_batch_logs"])[-30:]
        if logs:
            st.json(logs, expanded=False)
        else:
            st.caption("–")

    return df_out if changed_any else None
