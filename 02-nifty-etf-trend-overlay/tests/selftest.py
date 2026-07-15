#!/usr/bin/env python3
"""No-dependency self-test of data_pipeline (determinism + QC-fail + invalidation)."""
import os, sys, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data_pipeline as dp

HDR = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"
def good(sym, base):
    rows = [HDR]
    px = base
    for d in range(1, 60):
        px *= 1.001
        rows.append(f"2024.01.{d:02d}\t00:00:00\t{px:.2f}\t{px*1.01:.2f}\t{px*0.99:.2f}\t{px:.2f}\t100\t0\t1")
    return "\n".join(rows) + "\n"

def run(raw):
    with tempfile.TemporaryDirectory() as out:
        return dp.build_panel(raw, out, freq="D", require_ok=False)

fails = 0
with tempfile.TemporaryDirectory() as raw:
    open(os.path.join(raw, "AAA_D1.csv"), "w").write(good("AAA", 100))
    open(os.path.join(raw, "BBB_D1.csv"), "w").write(good("BBB", 50))
    h1 = run(raw)["dataset_hash"]; h2 = run(raw)["dataset_hash"]
    print("determinism:", "PASS" if h1 == h2 else "FAIL"); fails += h1 != h2
    # invalidation
    with open(os.path.join(raw, "AAA_D1.csv"), "a") as f:
        f.write("2024.03.01\t00:00:00\t999\t999\t999\t999\t100\t0\t1\n")
    h3 = run(raw)["dataset_hash"]
    print("invalidation:", "PASS" if h3 != h1 else "FAIL"); fails += h3 == h1

with tempfile.TemporaryDirectory() as raw:
    open(os.path.join(raw, "AAA_D1.csv"), "w").write(good("AAA", 100))
    bad = HDR + "\n2024.01.01\t00:00:00\t-5\t-5\t-5\t-5\t100\t0\t1\n"  # negative price
    open(os.path.join(raw, "BAD_D1.csv"), "w").write(bad)
    m = run(raw)
    bad_fail = any(q["symbol"] == "BAD" and q["status"] == "FAIL" for q in m["qc"])
    print("qc_negative_price_FAIL:", "PASS" if bad_fail else "FAIL"); fails += not bad_fail

print("SELFTEST", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(1 if fails else 0)
