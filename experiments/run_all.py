"""Run experiments 02-05 (the long part), resumable and shardable.

    python experiments/run_all.py                       # everything, one process
    python experiments/run_all.py --shard 0/2           # half of the buildings
    python experiments/run_all.py --quick               # smoke test (~2 min)
    MG_RESULTS_DIR=/content/drive/MyDrive/mg_results python experiments/run_all.py ...

Finished runs are JSON files and are skipped on restart; unfinished long runs resume
from weekly checkpoints. Experiments 06-07 are short: run them separately.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mgdispatch.runner import all_buildings, load_config, results_root, run_one  # noqa: E402
from plan import PASSES                                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", default="0/1", help="i/n: run buildings i, i+n, i+2n ...")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--passes", default="1,2,3,4,5")
    args = ap.parse_args()
    cfg = load_config(args.config)
    i, n = map(int, args.shard.split("/"))
    buildings = all_buildings(cfg)
    if args.quick:
        buildings = [(b.split("_")[0], b) for b in cfg["quick"]["buildings"]]
    mine = buildings[i::n]
    passes = [int(x) for x in args.passes.split(",")]
    print(f"Results -> {results_root()}\nShard {i}/{n}: {len(mine)} buildings, passes {passes}",
          flush=True)
    t0 = time.time()
    failures = 0
    for pnum in passes:
        print(f"\n=== PASS {pnum} ({(time.time() - t0) / 60:.1f} min elapsed) ===", flush=True)
        cache = {}
        for site, b in mine:
            for exp, spec in PASSES[pnum - 1]:
                if args.quick and spec["policy"] in ("mpc_stochastic", "mpc_robust") \
                        and spec.get("seed", 0) > 0:
                    continue
                try:
                    run_one(cfg, exp, site, b, spec, quick=args.quick, cache=cache)
                except Exception:                                   # noqa: BLE001
                    failures += 1
                    print(f"!! FAILED {exp} {b} {spec}\n{traceback.format_exc()}", flush=True)
    print(f"\nDONE shard {i}/{n} in {(time.time() - t0) / 60:.1f} min, failures: {failures}",
          flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
