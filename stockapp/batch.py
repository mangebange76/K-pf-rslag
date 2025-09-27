# stockapp/batch.py
import streamlit as st
import pandas as pd
from typing import List, Tuple, Dict

# -----------------------------
# Hjälpare: äldsta TS per rad
# -----------------------------
def _oldest_any_ts(row: pd.Series):
    dates = []
    for c, v in row.items():
        if str(c).startswith("TS_") and str(v).strip():
            try:
                d = pd.to_datetime(str(v).strip(), errors="coerce")
                if pd.notna(d):
                    dates.append(d)
            except Exception:
                pass
    return min(dates) if dates else pd.NaT

def add_oldest_ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """Publik (utan underscore) så andra moduler kan använda den."""
    out = df.copy()
    out["_oldest_any_ts"] = out.apply(_oldest_any_ts, axis=1)
    out["_oldest_any_ts_fill"] = out["_oldest_any_ts"].fillna(pd.Timestamp("2099-12-31"))
    return out

# -----------------------------
# Sorteringsordning för batch
# -----------------------------
def _pick_order(df: pd.DataFrame, sort_mode: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    if sort_mode == "Äldst uppdaterade först":
        work = add_oldest_ts_col(df)
        by = [c for c in ["_oldest_any_ts_fill", "Bolagsnamn", "Ticker"] if c in work.columns]
        return work.sort_values(by=by, ascending=True)
    else:
        by = [c for c in ["Bolagsnamn", "Ticker"] if c in df.columns]
        return df.sort_values(by=by, ascending=True)

# -----------------------------
# Skapa batch ur ordning
# -----------------------------
def _build_batch_list(df: pd.DataFrame, size: int, sort_mode: str) -> List[str]:
    ordered = _pick_order(df, sort_mode)
    if ordered.empty or "Ticker" not in ordered.columns:
        return []
    tickers = [str(t).upper().strip() for t in ordered["Ticker"].tolist() if str(t).strip()]
    return tickers[: max(0, int(size))]

# ----------------------------------------------------
# Kör batchen och returnera (df, log) – ingen sparning
# ----------------------------------------------------
def run_batch_update(
    df: pd.DataFrame,
    user_rates: dict,
    tickers_to_run: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Kör auto_fetch → apply_auto_updates_to_row för valda tickers.
    Sparar INTE till Sheets – det gör app.py efteråt.
    """
    # Importer här för att undvika cirkulära beroenden vid app-start
    from .fetchers import auto_fetch_for_ticker
    from .calc import apply_auto_updates_to_row, uppdatera_berakningar

    log = {"changed": {}, "misses": {}}
    total = len(tickers_to_run)
    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()

    any_changed = False

    # Snabb index-karta för rader
    ticker_to_idx = {}
    if "Ticker" in df.columns:
        for i, t in enumerate(df["Ticker"].tolist()):
            ticker_to_idx[str(t).upper().strip()] = i

    for i, tkr in enumerate(tickers_to_run, start=1):
        status.write(f"Uppdaterar {i}/{total}: {tkr}")
        try:
            new_vals, debug = auto_fetch_for_ticker(tkr)
            ridx = ticker_to_idx.get(str(tkr).upper().strip())
            if ridx is None:
                # okänt ticker i df → logga miss
                log["misses"][tkr] = ["Ticker saknas i tabellen"]
            else:
                changed = apply_auto_updates_to_row(
                    df, ridx, new_vals, source="Batch (SEC/Yahoo→Finnhub→FMP)", changes_map=log["changed"]
                )
                any_changed = any_changed or bool(changed)
        except Exception as e:
            log["misses"].setdefault(tkr, []).append(str(e))

        progress.progress(i / max(1, total))

    # Recompute lokalt
    df = uppdatera_berakningar(df, user_rates)
    return df, log

# ----------------------------------------------------
# Sidopanel: bygga & köra batch
# ----------------------------------------------------
def sidebar_batch_controls(
    df: pd.DataFrame,
    user_rates: dict,
    save_cb=None,         # callable(df) -> None (app.py ansvarar för Sheets-skriv)
    recompute_cb=None,    # callable(df) -> pd.DataFrame (om app.py vill göra extra)
):
    st.sidebar.subheader("🧰 Batch-uppdatering")

    # Init state
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 10
    if "batch_sort_mode" not in st.session_state:
        st.session_state.batch_sort_mode = "Äldst uppdaterade först"
    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []   # tickers i aktuell batch
    if "batch_ran" not in st.session_state:
        st.session_state.batch_ran = []     # tickers som körts denna session

    st.session_state.batch_size = int(st.sidebar.number_input("Batchstorlek", min_value=1, max_value=200, value=st.session_state.batch_size, step=1))
    st.session_state.batch_sort_mode = st.sidebar.selectbox(
        "Ordning", ["Äldst uppdaterade först", "A–Ö (bolagsnamn)"], index=0 if st.session_state.batch_sort_mode.startswith("Äldst") else 1
    )

    col1, col2, col3 = st.sidebar.columns([1,1,1])

    with col1:
        if st.button("Skapa batch"):
            st.session_state.batch_queue = _build_batch_list(df, st.session_state.batch_size, st.session_state.batch_sort_mode)
            st.session_state.batch_ran = []
            st.sidebar.success(f"Batch skapad ({len(st.session_state.batch_queue)} tickers).")

    with col2:
        if st.button("Kör batch"):
            if not st.session_state.batch_queue:
                st.sidebar.warning("Ingen batch skapad ännu.")
            else:
                to_run = [t for t in st.session_state.batch_queue if t not in st.session_state.batch_ran]
                if not to_run:
                    st.sidebar.info("Inget kvar i batchen. Skapa ny eller återställ.")
                else:
                    df2, log = run_batch_update(df.copy(), user_rates, to_run)
                    st.session_state.batch_ran.extend(to_run)
                    # valfri recompute/spara
                    if recompute_cb:
                        df2 = recompute_cb(df2)
                    if save_cb:
                        save_cb(df2)
                    st.session_state["_df_ref"] = df2
                    st.sidebar.success(f"Klar: {len(to_run)} tickers uppdaterade.")
                    st.session_state["last_auto_log"] = log
                    return df2  # tillbaka till app.py med *nytt* df

    with col3:
        if st.button("Återställ batch"):
            st.session_state.batch_queue = []
            st.session_state.batch_ran = []
            st.sidebar.info("Batch återställd.")

    # Visa batch-status
    q = st.session_state.batch_queue
    ran = set(st.session_state.batch_ran)
    if q:
        left = [t for t in q if t not in ran]
        st.sidebar.caption(f"Batch: {len(q)} tickers • Kvar: {len(left)} • Klara: {len(ran)}")
        if left:
            st.sidebar.write(", ".join(left[:10]) + (" ..." if len(left) > 10 else ""))
    else:
        st.sidebar.caption("Ingen aktiv batch.")

    # Returnera oförändrat df om inget körts
    return df
