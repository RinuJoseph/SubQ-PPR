#!/usr/bin/env python
"""scripts/eval.py — aggregate evaluation report.

Reads results/retrieval/summary.json and (optionally) results/qa/qa_results.json
and prints a final report comparing against HippoRAG-2's published numbers.
Also writes results/eval_report.{json,md}.

Usage:
  python scripts/eval.py --config configs/default.yaml
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sqppr.utils import load_config, save_json
from sqppr.metrics import recall_at_k

KS = [5, 10, 15, 20, 30, 40, 50]

# HippoRAG-2 published baselines per dataset (R@5 / F1 / EM).
# Sources: HippoRAG-2 paper Tables 2–3.  null = not published / not applicable.
HIPPO_BASELINES = {
    "musique":                {"R@5": 0.7418, "F1": 0.4809, "EM": 0.343},
    "hotpotqa":               {"R@5": 0.928,  "F1": 0.713,  "EM": 0.564},
    "2wikimultihopqa":        {"R@5": 0.901,  "F1": 0.706,  "EM": 0.617},
    "hover":                  {"R@5": 0.748,  "F1": None,   "EM": None},
    "popqa":                  {"R@5": 0.628,  "F1": 0.514,  "EM": 0.450},
    "nq_rear":                {"R@5": None,   "F1": None,   "EM": None},
    "lveval":                 {"R@5": None,   "F1": 0.205,  "EM": None},
    "narrativeqa_dev_10_doc": {"R@5": None,   "F1": 0.232,  "EM": None},
}


def hop_class(qid):
    m = re.match(r"^(\d+)hop", qid)
    return f"{m.group(1)}hop" if m else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (overrides config.dataset.name). "
                         "Supported: musique, hotpotqa, 2wikimultihopqa, hover, popqa, nq_rear")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_override=args.dataset)
    out_dir = Path(cfg["output_dir"])
    retr_path = out_dir / "retrieval" / "summary.json"
    qa_path   = out_dir / "qa" / "qa_results.json"

    if not retr_path.exists():
        sys.exit(f"missing {retr_path} — run scripts/run_retrieval.py first.")
    retr = json.load(open(retr_path))
    rows = retr["per_query"]
    n = len(rows)

    # Recall@k from saved per-query recall (rerun safely)
    recall = {f"recall@{k}": 0.0 for k in KS}
    counts = defaultdict(int)
    by_hop_r5 = defaultdict(list)
    for r in rows:
        for k in KS:
            v = r["recall"].get(f"recall@{k}")
            if v is not None:
                recall[f"recall@{k}"] += v
                counts[k] += 1
        v5 = r["recall"].get("recall@5")
        if v5 is not None:
            by_hop_r5[hop_class(r["qid"])].append(v5)
    for k in KS:
        recall[f"recall@{k}"] = recall[f"recall@{k}"] / counts[k] if counts[k] else None

    cov = [r["final_pool_coverage"] for r in rows if r["final_pool_coverage"] is not None]
    avg_cov  = sum(cov)/len(cov) if cov else None
    avg_pool = sum(r["n_pool"] for r in rows) / max(1, n)
    avg_kept = sum(r["n_facts_kept_by_llm"] for r in rows) / max(1, n)

    # QA
    qa = None
    qa_by_hop = defaultdict(list)
    if qa_path.exists():
        qa_blob = json.load(open(qa_path))
        qa = {
            "n_queries": qa_blob["n_queries"],
            "mean_f1":   qa_blob["mean_f1"],
            "mean_em":   qa_blob["mean_em"],
            "qa_model":  qa_blob["qa_model"],
            "top_k":     qa_blob["top_k"],
        }
        for r in qa_blob["per_query"]:
            qa_by_hop[hop_class(r["qid"])].append((r["f1"], r["em"]))

    # Print report
    print()
    print("=" * 90)
    print(f"  SQ-PPR-Final  Evaluation Report")
    print("=" * 90)
    print(f"  Queries:                {n}")
    print(f"  Avg pool size:          {avg_pool:.2f}")
    print(f"  Avg final coverage:     {avg_cov:.4f}" if avg_cov is not None else "")
    print(f"  Avg LLM-kept facts:     {avg_kept:.2f}")
    print()
    print(f"  Retrieval (PPR ranking)")
    print(f"    " + "  ".join(f"@{k:>3d}" for k in KS))
    print(f"    " + "  ".join(f"{recall[f'recall@{k}']:.4f}" if recall[f'recall@{k}'] is not None else "  n/a"
                              for k in KS))
    print()
    print(f"  Per-hop R@5:")
    for hc in ("2hop", "3hop", "4hop"):
        vs = by_hop_r5.get(hc, [])
        if vs:
            print(f"    {hc:<8s}  n={len(vs):>4d}  R@5={sum(vs)/len(vs):.4f}")
    # Per-dataset HippoRAG-2 baselines (None = not published)
    dataset_name = cfg["dataset"]["name"]
    base = HIPPO_BASELINES.get(dataset_name, {})
    HIPPO_R5 = base.get("R@5"); HIPPO_F1 = base.get("F1"); HIPPO_EM = base.get("EM")
    print()
    if HIPPO_R5 is not None:
        print(f"  HippoRAG-2 reported ({dataset_name}):    R@5 = {HIPPO_R5:.4f}")
        delta = (recall["recall@5"] - HIPPO_R5) if recall["recall@5"] is not None else None
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            print(f"  Δ vs HippoRAG-2:        {sign}{delta:.4f}  ({sign}{delta*100:.2f} pts)")
    else:
        print(f"  HippoRAG-2 reported ({dataset_name}):    R@5 not published")
        delta = None

    if qa:
        print()
        print(f"  QA  ({qa['qa_model']}, top-{qa['top_k']})")
        print(f"    Mean F1:  {qa['mean_f1']:.4f}")
        print(f"    Mean EM:  {qa['mean_em']:.4f}")
        if HIPPO_F1 is not None:
            em_s = f"   EM = {HIPPO_EM:.4f}" if HIPPO_EM is not None else ""
            print(f"  HippoRAG-2 reported:   F1 = {HIPPO_F1:.4f}{em_s}")
        else:
            print(f"  HippoRAG-2 reported:   F1/EM not published for {dataset_name}")
        print(f"  Per-hop:")
        for hc in ("2hop", "3hop", "4hop"):
            vs = qa_by_hop.get(hc, [])
            if vs:
                f1 = sum(x[0] for x in vs)/len(vs)
                em = sum(x[1] for x in vs)/len(vs)
                print(f"    {hc:<8s}  n={len(vs):>4d}  F1={f1:.4f}  EM={em:.4f}")
    else:
        print()
        print(f"  QA: not run yet (no {qa_path.name})")

    # Save JSON + Markdown report
    report = {
        "n_queries":         n,
        "avg_pool_size":     round(avg_pool, 4),
        "avg_final_coverage": round(avg_cov, 6) if avg_cov is not None else None,
        "avg_llm_kept_facts": round(avg_kept, 4),
        "retrieval":         {k: (round(v, 6) if v is not None else None) for k, v in recall.items()},
        "retrieval_by_hop":  {hc: {"n": len(vs), "recall@5": round(sum(vs)/len(vs), 6)}
                              for hc, vs in by_hop_r5.items() if vs},
        "dataset":             dataset_name,
        "hipporag2_reported":  {"recall@5": HIPPO_R5, "f1": HIPPO_F1, "em": HIPPO_EM},
        "delta_recall@5_vs_hipporag2": round(delta, 6) if delta is not None else None,
        "qa":                qa,
        "qa_by_hop":         {hc: {"n": len(vs),
                                    "f1": round(sum(x[0] for x in vs)/len(vs), 6),
                                    "em": round(sum(x[1] for x in vs)/len(vs), 6)}
                              for hc, vs in qa_by_hop.items() if vs} if qa else None,
    }
    save_json(report, out_dir / "eval_report.json")
    print(f"\nsaved -> {out_dir/'eval_report.json'}")

    md = []
    md.append(f"# SQ-PPR-Final Evaluation Report\n")
    md.append(f"- Queries: **{n}**")
    md.append(f"- Avg pool size: **{avg_pool:.2f}**")
    if avg_cov is not None:
        md.append(f"- Avg final pool coverage: **{avg_cov:.4f}**")
    md.append(f"- Avg LLM-kept facts: **{avg_kept:.2f}**\n")
    md.append(f"## Retrieval (PPR ranking)\n")
    md.append("| " + " | ".join(f"R@{k}" for k in KS) + " |")
    md.append("|" + "---|" * len(KS))
    md.append("| " + " | ".join(f"{recall[f'recall@{k}']:.4f}" if recall[f'recall@{k}'] is not None else "n/a"
                                  for k in KS) + " |\n")
    if HIPPO_R5 is not None:
        md.append(f"HippoRAG-2 reported R@5 ({dataset_name}): **{HIPPO_R5:.4f}**")
        if delta is not None:
            sign = "+" if delta >= 0 else ""
            md.append(f"Δ vs HippoRAG-2: **{sign}{delta:.4f}** ({sign}{delta*100:.2f} pts)\n")
    else:
        md.append(f"HippoRAG-2: R@5 not published for {dataset_name}\n")
    md.append(f"## Per-hop R@5\n")
    md.append("| hop | n | R@5 |")
    md.append("|---|---|---|")
    for hc in ("2hop", "3hop", "4hop"):
        vs = by_hop_r5.get(hc, [])
        if vs:
            md.append(f"| {hc} | {len(vs)} | {sum(vs)/len(vs):.4f} |")
    if qa:
        md.append(f"\n## QA — {qa['qa_model']}, top-{qa['top_k']}\n")
        f1_ref = f" (HippoRAG-2: {HIPPO_F1:.4f})" if HIPPO_F1 is not None else ""
        em_ref = f" (HippoRAG-2: {HIPPO_EM:.4f})" if HIPPO_EM is not None else ""
        md.append(f"- Mean F1: **{qa['mean_f1']:.4f}**{f1_ref}")
        md.append(f"- Mean EM: **{qa['mean_em']:.4f}**{em_ref}\n")
        md.append("| hop | n | F1 | EM |")
        md.append("|---|---|---|---|")
        for hc in ("2hop", "3hop", "4hop"):
            vs = qa_by_hop.get(hc, [])
            if vs:
                f1 = sum(x[0] for x in vs)/len(vs)
                em = sum(x[1] for x in vs)/len(vs)
                md.append(f"| {hc} | {len(vs)} | {f1:.4f} | {em:.4f} |")
    (out_dir / "eval_report.md").write_text("\n".join(md))
    print(f"saved -> {out_dir/'eval_report.md'}")


if __name__ == "__main__":
    main()
