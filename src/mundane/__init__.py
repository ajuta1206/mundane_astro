"""Mundane astrology x markets: ephemeris, factors, and rigorous backtests."""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(prog="mundane", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="run the full factor x market backtest grid")
    b.add_argument("--refresh", action="store_true", help="re-download market data")
    sub.add_parser("signals", help="rebuild the cached astrological factor table")
    m = sub.add_parser("model", help="train/evaluate the multi-horizon forecast + options model")
    m.add_argument("--nulls", type=int, default=20, help="shuffled-astro null refits per astro model")
    m.add_argument("--assets", nargs="*", help="subset of markets, e.g. SPX GOLD")
    m.add_argument("--horizons", nargs="*", default=["W", "M", "Q", "H", "Y"])
    sub.add_parser("technicals", help="stand-alone technical-rule backtests")
    f = sub.add_parser("forecast", help="upcoming astro periods with their backtest record")
    f.add_argument("--asset", default="SPX")
    f.add_argument("--days", type=int, default=365)
    sub.add_parser("app", help="launch the Streamlit dashboard")
    a = ap.parse_args()

    if a.cmd == "backtest":
        from .pipeline import run_all
        run_all(refresh_data=a.refresh)
    elif a.cmd == "model":
        from .pipeline import run_model_all
        run_model_all(n_null=a.nulls, horizons=a.horizons, assets=a.assets)
    elif a.cmd == "technicals":
        from .pipeline import run_technicals
        run_technicals()
    elif a.cmd == "signals":
        from .pipeline import get_signals
        s = get_signals(refresh=True)
        print(f"built {s.shape[1]} columns x {len(s)} days")
    elif a.cmd == "forecast":
        import pandas as pd
        from .forecast import annotate, upcoming_periods
        from .pipeline import RESULTS, get_signals
        per = upcoming_periods(get_signals(), days=a.days)
        p = RESULTS / "factor_tests.csv"
        if p.exists():
            per = annotate(per, pd.read_csv(p), a.asset)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(per.to_string(index=False))
    elif a.cmd == "app":
        import subprocess
        import sys
        from pathlib import Path
        app = Path(__file__).resolve().parents[2] / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)])
