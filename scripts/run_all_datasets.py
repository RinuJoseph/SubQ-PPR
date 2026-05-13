#!/usr/bin/env python
"""scripts/run_all_datasets.py — run retrieval + (optional) QA on every dataset.

For each dataset, runs:
  1. build_cache.py    — ensures decomp, passage embs, NER, triples are cached
  2. run_retrieval.py  — full pipeline, saves to results/<dataset>/retrieval/
  3. (optional) run_qa.py — saves to results/<dataset>/qa/

Cache is per-dataset (cache/<dataset>/...) so reruns short-circuit.

Usage:
  python scripts/run_all_datasets.py
  python scripts/run_all_datasets.py --datasets musique hover
  python scripts/run_all_datasets.py --skip-qa  --limit 50
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_DATASETS = [
    # Datasets with chunk-level gold (used in the Recall@K table)
    "musique", "hover", "hotpotqa", "2wikimultihopqa", "popqa", "nq_rear",
    # QA-F1 only (no chunk-level gold; pipeline still runs end-to-end)
    "lveval", "narrativeqa_dev_10_doc",
]

HERE = Path(__file__).resolve().parent.parent


def run(cmd, cwd=HERE):
    print(f"\n  $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=cwd)
    if rc != 0:
        print(f"  ! exit {rc}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                    help=f"datasets to run (default: {DEFAULT_DATASETS})")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit queries per dataset (for smoke runs)")
    ap.add_argument("--skip-build", action="store_true",
                    help="skip the build_cache step")
    ap.add_argument("--skip-qa", action="store_true",
                    help="skip the QA stage")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    log = HERE / "results" / "all_runs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[plan] datasets: {args.datasets}")
    print(f"[plan] config:   {args.config}")
    print(f"[plan] limit:    {args.limit}")
    print(f"[plan] log tail: {log}")

    t0_all = time.time()
    summary = []
    for ds in args.datasets:
        print(f"\n{'='*100}")
        print(f"  DATASET: {ds}")
        print(f"  {time.ctime()}")
        print(f"{'='*100}")
        t0 = time.time()

        if not args.skip_build:
            rc = run([sys.executable, "scripts/build_cache.py",
                        "--config", args.config, "--dataset", ds,
                        "--workers", str(args.workers)])
            if rc != 0:
                summary.append((ds, "build_cache FAILED", time.time()-t0))
                continue

        retrieval_cmd = [sys.executable, "scripts/run_retrieval.py",
                            "--config", args.config, "--dataset", ds,
                            "--workers", str(args.workers)]
        if args.limit: retrieval_cmd += ["--limit", str(args.limit)]
        rc = run(retrieval_cmd)
        if rc != 0:
            summary.append((ds, "retrieval FAILED", time.time()-t0))
            continue

        if not args.skip_qa:
            qa_cmd = [sys.executable, "scripts/run_qa.py",
                        "--config", args.config, "--dataset", ds,
                        "--workers", str(args.workers)]
            if args.limit: qa_cmd += ["--limit", str(args.limit)]
            rc = run(qa_cmd)
            if rc != 0:
                summary.append((ds, "qa FAILED", time.time()-t0))
                continue

        # Aggregate report
        run([sys.executable, "scripts/eval.py",
                "--config", args.config, "--dataset", ds])

        summary.append((ds, "OK", time.time() - t0))

    print(f"\n{'='*100}")
    print(f"  ALL DONE in {(time.time()-t0_all)/60:.1f} min")
    print(f"{'='*100}")
    for ds, status, dt in summary:
        print(f"  {ds:<18s}  {status:<24s}  {dt/60:>6.1f} min")


if __name__ == "__main__":
    main()
