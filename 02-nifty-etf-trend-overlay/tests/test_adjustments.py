#!/usr/bin/env python3
"""Permanent regression test: NIFTYBEES 10:1 split adjustment (offline, no network)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import nse_adapter as na

def test_split_removes_cliff():
    raw = pd.DataFrame({
        "date":  ["2019-12-17","2019-12-18","2019-12-19","2019-12-20"],
        "open":  [1200.0,1210.0,121.0,122.0],
        "high":  [1215.0,1225.0,122.5,123.0],
        "low":   [1190.0,1200.0,120.0,121.5],
        "close": [1210.0,1220.0,122.0,122.5],   # raw shows a fake ~-90% cliff on the 19th
        "volume":[100000.0,110000.0,1000000.0,1050000.0],
    })
    adj = na.apply_adjustments(raw, "NIFTYBEES")
    # pre-split prices divided by 10
    assert abs(adj.loc[1,"close"] - 122.0) < 1e-9
    # post-split untouched
    assert abs(adj.loc[2,"close"] - 122.0) < 1e-9
    # the cross-boundary overnight return is now small, not -90%
    assert abs(adj["close"].pct_change().iloc[2]) < 0.05
    # volume scaled x10 pre-split
    assert abs(adj.loc[0,"volume"] - 1000000.0) < 1e-6
    return True

if __name__ == "__main__":
    ok = test_split_removes_cliff()
    print("test_split_removes_cliff:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
