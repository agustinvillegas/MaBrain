"""
Evaluate a trained brain against a gold set of validated schemas.

Gold set format (JSONL, one item per line):
    {"schema_id": "...", "relation_sequence": ["IS_A"],
     "roles": ["entity", "class"], "instances": [{"entity":"dog","class":"mammal"}, ...],
     "valid": true}

Metrics:
  - analogy top-1 / top-5 accuracy
  - relation precision (find_by_relation, query_relation)
  - definition recall (IS_A chain)

Usage:
    python eval_gold.py --brain data/brain_states/brain_massive_v1.json \\
        --gold data/gold/schemas_to_validate.jsonl \
        --output data/gold/eval_results.json
"""
import json
import os
import sys
from collections import defaultdict, Counter

from braincell import Brain


# ── Loaders ──────────────────────────────────────────────────────
def load_brain(path):
    brain = Brain()
    brain.load(path)
    return brain


def load_gold(path):
    """Load validated gold schemas (valid: true only)."""
    items = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("valid") is True:
                items.append(item)
    return items


# ── Evaluators ───────────────────────────────────────────────────
def evaluate_analogy(brain, gold_items, top_k=3):
    """For each schema with a relation sequence, test analogy performance:
    Given (a, relation, b) and (c, relation, ?), can brain find d?
    """
    results = []
    hits_top1 = 0
    hits_topk = 0
    total = 0

    for item in gold_items:
        rels = item.get("relation_sequence", [])
        instances = item.get("instances", [])
        roles = item.get("roles", [])
        if len(rels) < 1 or len(instances) < 2:
            continue

        # For single-relation schemas (IS_A, FUNCTION, CAUSE, etc.)
        rel = rels[0]
        role_a = roles[0] if roles else "concept_a"
        role_b = roles[1] if len(roles) > 1 else "concept_b"

        # Test each pair of instances as analogy queries
        for i in range(len(instances)):
            for j in range(i + 1, len(instances)):
                a = instances[i].get(role_a, "")
                b = instances[i].get(role_b, "")
                c = instances[j].get(role_a, "")
                d_expected = instances[j].get(role_b, "")

                if not a or not b or not c or not d_expected:
                    continue

                # Try direct analogy
                analogies = brain.analogy(a, b, c, top_k=top_k)
                if analogies:
                    found_d = [r["d"] for r in analogies]
                    if found_d and found_d[0] == d_expected:
                        hits_top1 += 1
                    if d_expected in found_d[:top_k]:
                        hits_topk += 1
                total += 1

    return {
        "total_analogy_queries": total,
        "top1_accuracy": round(hits_top1 / max(total, 1), 4),
        "topk_accuracy": round(hits_topk / max(total, 1), 4),
        "top1_hits": hits_top1,
        "topk_hits": hits_topk,
    }


def evaluate_relation(brain, gold_items):
    """Test that the brain has correct edges for each gold instance."""
    results = []
    found = 0
    total = 0

    for item in gold_items:
        rels = item.get("relation_sequence", [])
        instances = item.get("instances", [])
        roles = item.get("roles", [])
        if len(rels) < 1:
            continue

        rel = rels[0]
        role_a = roles[0] if roles else "concept_a"
        role_b = roles[1] if len(roles) > 1 else "concept_b"

        for inst in instances:
            a = inst.get(role_a, "")
            b = inst.get(role_b, "")
            if not a or not b:
                continue
            total += 1

            # Check via query_relation
            found_rel = brain.query_relation(a, b)
            if found_rel == rel:
                found += 1
                continue

            # Fallback: check via find_by_relation
            candidates = brain.find_by_relation(a, rel)
            if any(c[0] == b for c in candidates):
                found += 1
                continue

    return {
        "total_relation_checks": total,
        "relation_accuracy": round(found / max(total, 1), 4),
        "edges_found": found,
    }


# ── Main ─────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate brain against gold schemas")
    parser.add_argument("--brain", required=True, help="Path to brain state JSON")
    parser.add_argument("--gold", required=True, help="Path to validated gold JSONL")
    parser.add_argument("--output", "-o", default="data/gold/eval_results.json",
                        help="Output results path")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-k for analogy evaluation (default: 3)")
    args = parser.parse_args()

    print("=== Gold Set Evaluation ===\n")

    # Load
    print(f"Loading brain: {args.brain}")
    brain = load_brain(args.brain)
    print(f"  Cells: {len(brain.cells):,}, Synapses: {len(brain.synapses):,}")

    print(f"Loading gold: {args.gold}")
    gold_items = load_gold(args.gold)
    print(f"  Validated schemas: {len(gold_items)}")

    if not gold_items:
        print("  No valid gold items found. Did you set 'valid': true?")
        sys.exit(1)

    # Evaluate
    print("\n--- Analogy Evaluation ---")
    analogy_metrics = evaluate_analogy(brain, gold_items, top_k=args.top_k)
    for k, v in analogy_metrics.items():
        print(f"  {k}: {v}")

    print("\n--- Relation Evaluation ---")
    rel_metrics = evaluate_relation(brain, gold_items)
    for k, v in rel_metrics.items():
        print(f"  {k}: {v}")

    # Combined
    results = {
        "brain": args.brain,
        "gold": args.gold,
        "num_gold_items": len(gold_items),
        "analogy": analogy_metrics,
        "relation": rel_metrics,
        "summary": {
            "analogy_top1": analogy_metrics["top1_accuracy"],
            "analogy_topk": analogy_metrics["topk_accuracy"],
            "relation_accuracy": rel_metrics["relation_accuracy"],
        }
    }

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results: {args.output}")


if __name__ == "__main__":
    main()
