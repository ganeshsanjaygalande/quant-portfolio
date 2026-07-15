#!/usr/bin/env python3
"""
MT5 MACRO BASKET EXTRACTOR  (run on the Windows VPS with the MT5 terminal open)
==============================================================================
Pulls N years of H1 + D1 bars for a diversified macro basket and writes CSVs in
the EXACT tab-delimited MetaTrader export format that tsmom_engine.py ingests:
    <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    2024.12.31  05:20:00  21191.95  21200.25  21191.15  21192.85  305  0  210

Usage:  python mt5_extract.py
Requires:  pip install MetaTrader5 pandas pytz
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz, os, sys, time

# ── CONFIG ──────────────────────────────────────────────────────────────────
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_data")
YEARS     = 5
# Broker symbol names DIFFER. Edit the right-hand side to match YOUR Market Watch.
# (Right-click Market Watch -> Symbols to see exact names. Crypto/WTI often have suffixes.)
SYMBOLS   = {
    "US100":   "US100.cash",
    "US500":   "US500.cash",
    "XAUUSD":  "XAUUSD",
    "USOIL":   "USOIL.cash",        # some brokers: "WTI", "XTIUSD", "CrudeOIL"
    "BTCUSD":  "BTCUSD",       # some brokers: "BTCUSD.bc", "Bitcoin"
    "EURUSD":  "EURUSD",
}
TIMEFRAMES = {                 # MT5 enum -> label used in filename
    "H1": mt5.TIMEFRAME_H1,
    "D1": mt5.TIMEFRAME_D1,
}
# MT5 bar timestamps are in the BROKER SERVER timezone (commonly UTC+2/+3).
# We fetch in UTC and DO NOT silently shift; we record the server offset so the
# engine can align. Set SERVER_TZ if you know it; else leave None to auto-probe.
SERVER_TZ = None               # e.g. "Etc/GMT-2"  (note: Etc/GMT-2 == UTC+2)
# ────────────────────────────────────────────────────────────────────────────

def init_mt5():
    if not mt5.initialize():
        print(f"[FATAL] mt5.initialize() failed: {mt5.last_error()}")
        sys.exit(1)
    info = mt5.terminal_info()
    acc  = mt5.account_info()
    print(f"[OK] Terminal: {info.name}  build {info.build}")
    if acc: print(f"[OK] Account: {acc.login} @ {acc.server}  ({acc.company})")

def probe_server_offset():
    """Estimate broker-server UTC offset from the latest M1 tick vs system UTC."""
    t = mt5.symbol_info_tick("EURUSD") or mt5.symbol_info_tick(list(SYMBOLS.values())[0])
    if not t: return 0
    server_dt = datetime.utcfromtimestamp(t.time)         # MT5 epoch is server-local-as-UTC
    sys_utc   = datetime.utcnow()
    off_hours = round((server_dt - sys_utc).total_seconds()/3600)
    print(f"[INFO] Estimated broker server offset vs UTC: {off_hours:+d}h "
          f"(server bar-time = UTC{off_hours:+d})")
    return off_hours

def ensure_symbol(sym):
    if not mt5.symbol_select(sym, True):
        print(f"[WARN] Could not select symbol '{sym}': {mt5.last_error()}")
        return False
    # give the terminal a moment to populate history
    time.sleep(0.3)
    return True

def fetch(sym, tf_enum, years):
    utc = pytz.utc
    end   = datetime.now(tz=utc)
    start = end - timedelta(days=365*years + 5)
    rates = mt5.copy_rates_range(sym, tf_enum, start, end)
    if rates is None or len(rates) == 0:
        # fallback: pull by count (some brokers cap range queries)
        rates = mt5.copy_rates_from(sym, tf_enum, end, years*365*24)
    if rates is None or len(rates) == 0:
        print(f"[WARN] No data for {sym}: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["dt"] = pd.to_datetime(df["time"], unit="s")        # server-time epoch
    return df

def write_csv(df, label, tf_label):
    os.makedirs(OUT_DIR, exist_ok=True)
    # exact column structure tsmom_engine expects (tab-delimited, MT5 header style)
    out = pd.DataFrame({
        "<DATE>":   df["dt"].dt.strftime("%Y.%m.%d"),
        "<TIME>":   df["dt"].dt.strftime("%H:%M:%S"),
        "<OPEN>":   df["open"],
        "<HIGH>":   df["high"],
        "<LOW>":    df["low"],
        "<CLOSE>":  df["close"],
        "<TICKVOL>":df["tick_volume"],
        "<VOL>":    df["real_volume"] if "real_volume" in df else 0,
        "<SPREAD>": df["spread"] if "spread" in df else 0,
    })
    first = df["dt"].iloc[0].strftime("%Y%m%d%H%M")
    last  = df["dt"].iloc[-1].strftime("%Y%m%d%H%M")
    path = os.path.join(OUT_DIR, f"{label}_{tf_label}_{first}_{last}.csv")
    out.to_csv(path, sep="\t", index=False)
    print(f"[SAVE] {label} {tf_label}: {len(out):>7,} bars -> {os.path.basename(path)}")
    return path

def main():
    init_mt5()
    off = probe_server_offset()
    manifest = []
    for label, sym in SYMBOLS.items():
        if not ensure_symbol(sym):
            continue
        for tf_label, tf_enum in TIMEFRAMES.items():
            df = fetch(sym, tf_enum, YEARS)
            if df is None or len(df) == 0:
                continue
            p = write_csv(df, label, tf_label)
            manifest.append((label, tf_label, sym, len(df), p))
    print("\n=== MANIFEST ===")
    for label, tf, sym, n, p in manifest:
        print(f"  {label:8s} {tf:3s} <- {sym:12s} {n:>8,} bars")
    print(f"\nServer->UTC offset recorded as {off:+d}h. "
          f"Pass this to the engine if you align H1 bars across instruments.")
    mt5.shutdown()

if __name__ == "__main__":
    main()
