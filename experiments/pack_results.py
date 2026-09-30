"""Pack per-run JSON files into one JSON-lines file per experiment (and back).

GitHub's web upload takes at most 100 files per commit; 468 per-run JSONs would need
6 uploads. Each line of results/<exp>/runs.jsonl is exactly one original run JSON.

    python experiments/pack_results.py            # raw/*/*.json -> runs.jsonl, delete raw/
    python experiments/pack_results.py --unpack   # runs.jsonl -> raw/<building>/<label>.json
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mgdispatch.runner import results_root  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unpack", action="store_true")
    args = ap.parse_args()
    root = results_root()
    for exp in sorted(p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        jl, raw = exp / "runs.jsonl", exp / "raw"
        if args.unpack and jl.exists():
            for line in jl.read_text().splitlines():
                r = json.loads(line)
                out = raw / r["building"] / f"{r['label']}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(r, indent=1))
            print(f"unpacked {exp.name}")
        elif not args.unpack and raw.exists():
            files = sorted(raw.glob("*/*.json"))
            lines = [json.dumps(json.loads(f.read_text()), separators=(",", ":")) for f in files]
            old = jl.read_text().splitlines() if jl.exists() else []
            keys = {(json.loads(l)["building"], json.loads(l)["label"]) for l in lines}
            old = [l for l in old if (json.loads(l)["building"], json.loads(l)["label"]) not in keys]
            jl.write_text("\n".join(old + lines) + "\n")
            shutil.rmtree(raw)
            print(f"packed {exp.name}: {len(lines)} runs -> {jl.name}")


if __name__ == "__main__":
    main()
