import json
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, ".")

from braincell import Brain


def build_benchmark_from_graph(brain, max_analogies=None, max_per_relation=500):

    by_relation = defaultdict(list)

    for syn in brain.synapses.values():
        origin_word = syn.origin.word
        target_word = syn.target.word
        if origin_word and target_word and syn.relation:
            by_relation[syn.relation].append({
                "from": origin_word,
                "to": target_word,
                "strength": syn.strength,
                "cost": syn.cost,
                "syn_id": syn.id
            })

    analogies = []

    for rel, pairs in by_relation.items():

        if len(pairs) < 2:
            continue

        shuffled = pairs[:]
        random.shuffle(shuffled)
        count = 0

        for i, p1 in enumerate(shuffled):
            for j, p2 in enumerate(shuffled):
                if i == j:
                    continue

                analogies.append({
                    "a": p1["from"],
                    "b": p1["to"],
                    "c": p2["from"],
                    "d": p2["to"],
                    "relation": rel,
                    "a_b": (p1["from"], p1["to"]),
                    "c_d": (p2["from"], p2["to"])
                })
                count += 1
                if max_per_relation and count >= max_per_relation:
                    break
            if max_per_relation and count >= max_per_relation:
                break

        if max_analogies is not None and len(analogies) >= max_analogies:
            break

    return analogies


def run_ma_brain(brain, analogies):

    total = len(analogies)
    correct_top1 = 0
    correct_top5 = 0
    not_found = 0
    results = []

    start = time.perf_counter()
    total_ops = 0

    for item in analogies:
        a, b, c, d = item["a"], item["b"], item["c"], item["d"]

        candidates = brain.analogy(a, b, c)

        total_ops += len(candidates) + 1

        top1 = candidates[0]["d"] if candidates else None
        top5 = [r["d"] for r in candidates[:5]]

        if top1 == d:
            correct_top1 += 1
        if d in top5:
            correct_top5 += 1
        if not candidates:
            not_found += 1

        results.append({
            "analogy": f"{a}:{b} :: {c}:?",
            "expected": d,
            "top1": top1,
            "top5": top5,
            "correct": top1 == d,
            "in_top5": d in top5,
            "trace": candidates[0]["trace"] if candidates else None,
            "score": candidates[0]["score"] if candidates else 0
        })

    elapsed = time.perf_counter() - start

    return {
        "total": total,
        "correct_top1": correct_top1,
        "correct_top5": correct_top5,
        "not_found": not_found,
        "accuracy_top1": correct_top1 / total * 100 if total else 0,
        "accuracy_top5": correct_top5 / total * 100 if total else 0,
        "elapsed_seconds": elapsed,
        "ops_estimated": total_ops,
        "results": results
    }


STRUCTURAL_RELATIONS = {
    "CAUSE", "FUNCTION", "PART_OF", "INSTRUMENT", "LOCATED_IN",
    "EFFECT_OF", "FUNCTION_OF", "CONTAINS", "CONTAINED_IN",
    "USED_FOR", "OPPOSITE", "SIMILAR_TO", "HAS_PROPERTY", "PROPERTY_OF",
    "SYMBOL_OF", "SYMBOLIZED_BY",
    "MADE_OF", "DESIRES", "AVOIDS",
}

# Inverse map: if A→B relation=R, what's the reverse?
INVERSE_RELATIONS = {
    "CAUSE": "EFFECT_OF",
    "EFFECT_OF": "CAUSE",
    "FUNCTION": "FUNCTION_OF",
    "FUNCTION_OF": "FUNCTION",
    "PART_OF": "CONTAINS",
    "CONTAINS": "PART_OF",
    "LOCATED_IN": "CONTAINED_IN",
    "CONTAINED_IN": "LOCATED_IN",
    "HAS_PROPERTY": "PROPERTY_OF",
    "PROPERTY_OF": "HAS_PROPERTY",
    "INSTRUMENT": "USED_FOR",
    "USED_FOR": "INSTRUMENT",
    "SYMBOL_OF": "SYMBOLIZED_BY",
    "SYMBOLIZED_BY": "SYMBOL_OF",
    "MADE_OF": "MATERIAL_OF",
    "OPPOSITE": "OPPOSITE",
    "SIMILAR_TO": "SIMILAR_TO",
}


def build_benchmark_structural(brain, max_analogies=None, max_per_relation=200):
    """Genera analogías solo para relaciones estructurales, con muestreo por relación."""

    by_relation = {}
    for syn in brain.synapses.values():
        ow = syn.origin.word
        tw = syn.target.word
        rel = syn.relation
        if not ow or not tw or rel not in STRUCTURAL_RELATIONS:
            continue
        by_relation.setdefault(rel, []).append({
            "from": ow, "to": tw, "strength": syn.strength, "cost": syn.cost
        })

    analogies = []
    for rel, pairs in by_relation.items():
        if len(pairs) < 2:
            continue
        shuffled = pairs[:]
        random.shuffle(shuffled)
        count = 0
        for i, p1 in enumerate(shuffled):
            for j, p2 in enumerate(shuffled):
                if i == j:
                    continue
                analogies.append({
                    "a": p1["from"], "b": p1["to"],
                    "c": p2["from"], "d": p2["to"],
                    "relation": rel,
                })
                count += 1
                if max_per_relation and count >= max_per_relation:
                    break
            if max_per_relation and count >= max_per_relation:
                break
        if max_analogies is not None and len(analogies) >= max_analogies:
            break

    return analogies


def run_ma_brain_structural(brain, analogies, hops=2, min_sim=0.1):

    total = len(analogies)
    correct_top1 = 0
    correct_top5 = 0
    not_found = 0
    results = []

    start = time.perf_counter()

    for item in analogies:
        a, b, c, d = item["a"], item["b"], item["c"], item["d"]
        candidates = brain.analogy_structural(a, b, c, top_k=5, hops=hops, min_sim=min_sim)

        top1 = candidates[0]["d"] if candidates else None
        top5 = [r["d"] for r in candidates[:5]]

        if top1 == d:
            correct_top1 += 1
        if d in top5:
            correct_top5 += 1
        if not candidates:
            not_found += 1

        results.append({
            "analogy": f"{a}:{b} :: {c}:?",
            "expected": d,
            "top1": top1,
            "top5": top5,
            "correct": top1 == d,
            "in_top5": d in top5,
            "relation": item.get("relation"),
            "structural_score": candidates[0]["structural_score"] if candidates else 0,
        })

    elapsed = time.perf_counter() - start

    return {
        "total": total,
        "correct_top1": correct_top1,
        "correct_top5": correct_top5,
        "not_found": not_found,
        "accuracy_top1": correct_top1 / total * 100 if total else 0,
        "accuracy_top5": correct_top5 / total * 100 if total else 0,
        "elapsed_seconds": elapsed,
        "results": results,
    }


def print_report(stats):

    print("=" * 60)
    print("  MaBrain - Evaluacion de Analogias Semanticas")
    print("=" * 60)
    print(f"  Total analogias:     {stats['total']}")
    print(f"  Correctos top-1:     {stats['correct_top1']}  ({stats['accuracy_top1']:.1f}%)")
    print(f"  Correctos top-5:     {stats['correct_top5']}  ({stats['accuracy_top5']:.1f}%)")
    print(f"  No encontrados:      {stats['not_found']}")
    print(f"  Tiempo total:        {stats['elapsed_seconds']:.4f}s")
    print(f"  Ops estimadas:       {stats['ops_estimated']}")
    print()

    by_rel = defaultdict(list)
    for r in stats["results"]:
        rel = r["trace"]["a_b"]["relation"] if r["trace"] else "?"
        by_rel[rel].append(r)

    print("  --- Por relacion ---")
    for rel in sorted(by_rel.keys()):
        items = by_rel[rel]
        correct = sum(1 for i in items if i["correct"])
        print(f"    {rel}: {correct}/{len(items)} ({correct/len(items)*100:.1f}%)")
    print()

    fallos = [r for r in stats["results"] if not r["correct"]]
    aciertos = [r for r in stats["results"] if r["correct"]]

    if fallos:
        print(f"  --- Fallos top-1 ({len(fallos)}) ---")
        for r in fallos[:20]:
            print(f"    X {r['analogy']}  (esperado: {r['expected']}, obtuvo: {r['top1']})")
        if len(fallos) > 20:
            print(f"    ... y {len(fallos)-20} mas")
        print()

    if aciertos:
        print(f"  --- Aciertos top-1 ({len(aciertos)}) ---")
        for r in aciertos:
            t = r["trace"]
            if t:
                print(f"    OK {r['analogy']} -> {r['top1']}")
                print(f"       {t['a_b']['from']} -({t['a_b']['relation']})-> {t['a_b']['to']}"
                      f"  |  {t['c_d']['from']} -({t['c_d']['relation']})-> {t['c_d']['to']}")
        print()


def main():
    brain = Brain()
    brain.load()

    n_relations = sum(1 for s in brain.synapses.values() if s.relation is not None)
    print(f"  Cargado: {len(brain.cells)} neuronas, {len(brain.synapses)} sinapsis")
    print(f"  Sinapsis con relacion: {n_relations}")
    if n_relations == 0:
        print("  Asignando relaciones via auto_infer_relations()...")
        brain.auto_infer_relations()
        n_relations = sum(1 for s in brain.synapses.values() if s.relation is not None)
        print(f"  Ahora: {n_relations} sinapsis con relacion")
    print()

    analogies = build_benchmark_from_graph(brain, max_per_relation=500)
    print(f"  Benchmark generado: {len(analogies)} analogias")
    rels = set(a["relation"] for a in analogies)
    print(f"  Relaciones: {sorted(rels)}")
    print()

    stats = run_ma_brain(brain, analogies)
    print_report(stats)


if __name__ == "__main__":
    main()
