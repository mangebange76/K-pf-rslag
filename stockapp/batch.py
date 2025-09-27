# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------------
# Hjälpare: hitta "äldst TS" per rad
# -----------------------------------
def _parse_date_safe(s: str):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # Pandas sista chans
    try:
        d = pd.to_datetime(s, errors="coerce")
        if pd.notna(d):
            # returnera naive datetime
            return d.to_pydatetime()
    except Exception:
        pass
    return None

def _oldest_ts_value(row: pd.Series):
    dates = []
    for c in row.index:
        if str(c).startswith("TS_"):
            d = _parse_date_safe(row.get(c, ""))
            if isinstance(d, datetime):
                dates.append(d)
    return min(dates) if dates else None

def _add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["_oldest_any_ts"] = pd.NaT
        df["_oldest_any_ts_fill"] = pd.Timestamp.max
        return df
    df = df.copy()
    df["_oldest_any_ts"] = df.apply(_oldest_ts_value, axis=1)
    df["_oldest_any_ts"] = pd.to_datetime(df["_oldest_any_ts"], errors="coerce")
    df["_oldest_any_ts_fill"] = df["_oldest_any_ts"].fillna(pd.Timestamp.max)
    return df

# -----------------------------------
# Plocka ordning för batch-listan
# -----------------------------------
def _pick_order(df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if df.empty:
        return df
    if sort_mode.startswith("Äldst"):
        work = _add_oldest_ts_col(df)
        return work.sort_values(by=["_oldest_any_ts_fill","Bolagsnamn","Ticker"])
    else:
        # A–Ö på bolagsnamn, därefter ticker
        return df.sort_values(by=["Bolagsnamn","Ticker"])

# -----------------------------------
# Kör en batch uppsättning tickers
# -----------------------------------
def run_batch_update(df: pd.DataFrame,
                     user_rates: dict,
                     tickers: list,
                     make_snapshot: bool = False,
                     runner=None,
                     save_cb=None,
                     recompute_cb=None):
    """
    Kör en uppdatering på en lista tickers.

    - runner(df, ticker, user_rates) -> (df_updated, changed_fields|None, error|None)
      Om runner saknas försöker vi anropa st.session_state["run_update_for_ticker"].
      Finns inget → ingen faktisk uppdatering görs (endast recompute_cb).

    - save_cb(df): spara till Sheets
    - recompute_cb(df) -> df: räkna om beräkningar (P/S-snitt etc)

    Returnerar: (df_new, log) där log={"changed":{ticker:[fält...]}, "misses":{ticker:[.../fel]}}
    """
    log = {"changed": {}, "misses": {}}
    df_out = df.copy()

    if runner is None:
        runner = st.session_state.get("run_update_for_ticker")

    total = len(tickers)
    if total == 0:
        st.info("Ingen ticker i kön att köra.")
        return df_out, log

    # Progress + 1/X text
    pb = st.sidebar.progress(0.0, text=f"Startar batch: 0/{total}")
    status = st.sidebar.empty()

    for i, tkr in enumerate(tickers, start=1):
        tkr_s = str(tkr).strip().upper()
        status.write(f"Kör {i}/{total}: {tkr_s}")

        if runner is not None:
            try:
                df_out, changed_fields, err = runner(df_out, tkr_s, user_rates)
                if err:
                    log["misses"][tkr_s] = [str(err)]
                elif changed_fields:
                    log["changed"][tkr_s] = list(changed_fields)
                else:
                    log["misses"][tkr_s] = ["(inga ändringar)"]
            except Exception as e:
                log["misses"][tkr_s] = [f"error: {e}"]
        else:
            # ingen runner – hoppa över logiskt men tillämpa recompute
            log["misses"][tkr_s] = ["runner saknas (ingen fetch)"]

        pb.progress(i/total, text=f"Kör: {i}/{total}")

    # recompute + spara
    if callable(recompute_cb):
        try:
            df_out = recompute_cb(df_out)
        except Exception as e:
            st.warning(f"Kunde inte räkna om beräkningar: {e}")

    if callable(save_cb):
        try:
            save_cb(df_out)
            st.sidebar.success("Batch-skrivning klar.")
        except Exception as e:
            st.sidebar.error(f"Skrivning misslyckades: {e}")

    return df_out, log

# -----------------------------------
# Sidopanel: batch-styrning
# -----------------------------------
def sidebar_batch_controls(df: pd.DataFrame,
                           user_rates: dict,
                           save_cb=None,
                           recompute_cb=None,
                           runner=None) -> pd.DataFrame:
    """
    Skapa/kör/återställ en batch-kö. Kö och avklarade lagras i session_state.

    Parametrar:
      - save_cb(df): funktion att spara
      - recompute_cb(df)->df: räkna om
      - runner(df, ticker, user_rates) -> (df_new, changed_fields|None, error|None)
    """
    st.sidebar.subheader("🧵 Batch-uppdatering")

    # init state
    if "batch_queue" not in st.session_state: st.session_state.batch_queue = []
    if "batch_done" not in st.session_state: st.session_state.batch_done = []
    if "batch_sort_mode" not in st.session_state: st.session_state.batch_sort_mode = "Äldst uppdaterade först (TS_)"

    sort_mode = st.sidebar.selectbox(
        "Sortera kö",
        ["Äldst uppdaterade först (TS_)","A–Ö (Bolagsnamn)"],
        index=0 if st.session_state.batch_sort_mode.startswith("Äldst") else 1
    )
    st.session_state.batch_sort_mode = sort_mode

    batch_size = st.sidebar.number_input("Antal att lägga i kö", min_value=1, max_value=200, value=20, step=1)
    step_size = st.sidebar.slider("Kör nästa (stegstorlek)", min_value=1, max_value=50, value=10, step=1)

    # Bygg lista (filtrera bort redan körda)
    with st.sidebar.expander("📋 Skapa kö", expanded=False):
        candidate_df = _pick_order(df, sort_mode)
        tickers_all = [str(x).strip().upper() for x in candidate_df["Ticker"].astype(str).tolist() if str(x).strip()]
        already = set(st.session_state.batch_done)
        remaining = [t for t in tickers_all if t not in already]

        st.write(f"Totalt i datan: {len(tickers_all)} | Kvar (ej körda): {len(remaining)}")

        if st.button("Lägg till topp N i kö"):
            st.session_state.batch_queue.extend(remaining[:int(batch_size)])
            # ta bort dubbletter, behåll ordning
            seen = set()
            newq = []
            for t in st.session_state.batch_queue:
                if t not in seen:
                    newq.append(t); seen.add(t)
            st.session_state.batch_queue = newq
            st.success(f"Lade till {min(len(remaining), int(batch_size))} st i kö.")

        st.write("Nuvarande kö:", ", ".join(st.session_state.batch_queue) if st.session_state.batch_queue else "–")

    # Kör / Återställ
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button(f"▶️ Kör nästa {step_size}"):
            to_run = st.session_state.batch_queue[:int(step_size)]
            st.session_state.batch_queue = st.session_state.batch_queue[int(step_size):]

            if not to_run:
                st.info("Kön är tom.")
                return df

            df_new, log = run_batch_update(
                df, user_rates, to_run,
                make_snapshot=False,
                runner=runner,  # ev. override
                save_cb=save_cb,
                recompute_cb=recompute_cb
            )
            # markera som körda
            st.session_state.batch_done.extend(to_run)

            # visa log
            with st.sidebar.expander("📒 Senaste körlogg (Batch)", expanded=True):
                if log.get("changed"):
                    st.write("**Ändringar**")
                    st.json(log["changed"])
                if log.get("misses"):
                    st.write("**Missar**")
                    st.json(log["misses"])

            return df_new

    with col2:
        if st.button("🗑️ Återställ kö"):
            st.session_state.batch_queue = []
            st.success("Kön återställd.")

    # Status
    total_done = len(st.session_state.batch_done)
    st.sidebar.caption(f"Körda totalt: {total_done} | I kö: {len(st.session_state.batch_queue)}")

    return df
