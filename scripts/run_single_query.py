#!/usr/bin/env python
"""scripts/run_single_query.py — verbose end-to-end pipeline for ONE qid.

Useful for paper inspection / debugging. Prints pool, graph stats, LLM-kept
facts, PPR top-5, gold landings, QA answer + F1/EM.

Usage:
  python scripts/run_single_query.py --config configs/default.yaml --qid 2hop__13548_13529
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sqppr.utils import load_config, save_json
from sqppr.data import load_dataset_with_decomp
from sqppr.embedding import passage_embeddings_and_keys
from sqppr.retrieval import retrieve_one
from sqppr.qa import qa_one
from sqppr.metrics import compute_f1, compute_em, recall_at_k

KS = [5, 10, 15, 20, 30, 40, 50]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    ap.add_argument("--qid", required=True)
    ap.add_argument("--no-qa", action="store_true", help="Skip the QA stage")
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (overrides config.dataset.name).")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_override=args.dataset)
    corpus, queries = load_dataset_with_decomp(
        cfg["dataset"]["name"], Path(cfg["dataset"]["root"]),
        Path(cfg["cache"]["decompositions"]),
    )
    qr = next((q for q in queries if q["qid"] == args.qid), None)
    if qr is None:
        sys.exit(f"qid {args.qid!r} not found")

    print(f"\n{'='*80}\n  qid: {qr['qid']}\n  query: {qr['query']}\n{'='*80}")
    print(f"\nSub-questions ({len(qr['llm_decomposition'])}):")
    for i, sub in enumerate(qr["llm_decomposition"], 1):
        print(f"  Q{i}: {sub}")
    print(f"\nGold supports ({len(qr['gold'])}):")
    for g in qr["gold"]:
        cid = g.get("chunk_id")
        if cid:
            print(f"  hop {g['hop']}: {corpus[cid]['doc_id']}  "
                    f"{corpus[cid]['title'][:55]!r}  answer={g['answer']!r}")
        else:
            print(f"  hop {g['hop']}: <no chunk_id>  answer={g['answer']!r}")

    print(f"\n[load] passage embeddings …")
    passage_embs, passage_keys = passage_embeddings_and_keys(cache_dir=Path(cfg["cache"]["passage_embeddings"]), corpus=corpus)

    # ---- Retrieval ----
    print(f"\n[retrieval] running …")
    rec = retrieve_one(qr, corpus, passage_embs, passage_keys, cfg)

    gold = [g["chunk_id"] for g in qr["gold"] if g.get("chunk_id")]
    print(f"\nPool size:       {rec['n_pool']}")
    print(f"Entity nodes:    {rec['n_entity_nodes']}")
    print(f"Total edges:     {rec['n_edges']}")
    print(f"Triples in pool: {rec['n_triples']}")
    print(f"LLM-kept facts:  {rec['n_facts_kept_by_llm']}")
    print(f"Final pool coverage: {rec['final_pool_coverage']}")

    print(f"\nRecall (PPR ranking):")
    for k in KS:
        v = recall_at_k(rec["ranking"], gold, k)
        print(f"  recall@{k:<3d}  {'n/a' if v is None else f'{v:.4f}'}")

    print(f"\nPPR top-5:")
    print(f"  {'rk':>3s}  {'score':>10s}  {'gold?':<5s}  {'doc':<10s}  title")
    for p in rec["top_5"]:
        gflag = "GOLD" if p["is_gold"] else ""
        print(f"  [{p['rank']:>2d}]  {p['score']:>10.6f}  {gflag:<5s}  "
                f"{p['doc_id']:<10s}  {p['title'][:60]!r}")

    print(f"\nGold landings:")
    for g in rec["gold_landings"]:
        cid = g.get("chunk_id")
        if not cid:
            print(f"  hop {g['hop']}: <no chunk_id>"); continue
        if g.get("in_pool"):
            print(f"  hop {g['hop']}: PPR rank #{g['rank']}  "
                    f"doc={corpus[cid]['doc_id']}  title={corpus[cid]['title'][:50]!r}  "
                    f"answer={g['answer']!r}")
        else:
            print(f"  hop {g['hop']}: NOT IN POOL  doc={corpus[cid]['doc_id']}  "
                    f"title={corpus[cid]['title'][:50]!r}  answer={g['answer']!r}")

    # ---- QA ----
    qa_result = None
    if not args.no_qa:
        print(f"\n[qa] running …")
        top = rec["top_5"][:cfg["qa"]["top_k_passages"]]
        gold_answers = [g["answer"] for g in qr["gold"] if g.get("answer")]
        final_gold = gold_answers[-1] if gold_answers else ""
        r = qa_one(
            qr["query"], top, corpus,
            model=cfg["llm"]["qa_model"],
            temperature=cfg["qa"]["temperature"],
            max_completion_tokens=cfg["qa"]["max_completion_tokens"],
        )
        f1 = compute_f1(r["predicted_answer"], final_gold)
        em = compute_em(r["predicted_answer"], final_gold)
        qa_result = {
            "predicted_answer": r["predicted_answer"],
            "raw_response":     r["raw_response"],
            "gold_answer":      final_gold,
            "all_hop_answers":  gold_answers,
            "f1":               f1,
            "em":               em,
        }
        print(f"\nQA:")
        print(f"  Gold:      {final_gold!r}")
        print(f"  Predicted: {r['predicted_answer']!r}")
        print(f"  F1: {f1:.4f}   EM: {em:.4f}")

    # Save full per-qid blob
    out_dir = Path(cfg["output_dir"]) / "single_query"
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = {
        "qid": qr["qid"], "query": qr["query"],
        "sub_questions": qr["llm_decomposition"],
        "gold": qr["gold"],
        "retrieval": {
            "n_pool": rec["n_pool"],
            "n_entity_nodes": rec["n_entity_nodes"],
            "n_edges": rec["n_edges"],
            "n_triples": rec["n_triples"],
            "n_facts_kept_by_llm": rec["n_facts_kept_by_llm"],
            "final_pool_coverage": rec["final_pool_coverage"],
            "top_5": rec["top_5"],
            "gold_landings": rec["gold_landings"],
            "recall": {f"recall@{k}": recall_at_k(rec["ranking"], gold, k) for k in KS},
        },
        "qa": qa_result,
    }
    save_json(blob, out_dir / f"{qr['qid']}.json")
    print(f"\nsaved -> {out_dir / f'{qr['qid']}.json'}")


if __name__ == "__main__":
    main()
