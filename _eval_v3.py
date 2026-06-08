import sys
sys.path.insert(0, ".")
from braincell import Brain
from eval.evaluate import build_benchmark_from_graph, run_ma_brain, build_benchmark_structural, run_ma_brain_structural
from collections import Counter, defaultdict
import json

b = Brain()
b.load(r'D:\ma_brain_data\brain_state_v8_v3_embed.json')
print(f'Brain: {len(b.cells)} cells, {len(b.synapses)} synapses')

rel_counts = Counter(s.relation for s in b.synapses.values() if s.relation)
print(f'Relations: {dict(rel_counts.most_common(10))}')

# Full benchmark
analogies = build_benchmark_from_graph(b, max_per_relation=500)
stats = run_ma_brain(b, analogies)

# Per-relation
by_rel = defaultdict(list)
for r in stats["results"]:
    rel = r["trace"]["a_b"]["relation"] if r["trace"] else "?"
    by_rel[rel].append(r)

print('\n--- Full benchmark (max_per_relation=500) ---')
print(f'Overall: {stats["accuracy_top1"]:.1f}% top-1, {stats["accuracy_top5"]:.1f}% top-5')
for rel in sorted(by_rel.keys()):
    items = by_rel[rel]
    corr = sum(1 for i in items if i["correct"])
    top5 = sum(1 for i in items if i["in_top5"])
    print(f'  {rel}: {corr}/{len(items)} ({corr/len(items)*100:.1f}%) top-1, top-5 {top5}/{len(items)} ({top5/len(items)*100:.1f}%)')

# Structural benchmark
struct_analogies = build_benchmark_structural(b, max_per_relation=200)
print(f'\n--- Structural benchmark ---')
print(f'Total analogies: {len(struct_analogies)}')
if struct_analogies:
    struct_breakdown = Counter(a["relation"] for a in struct_analogies)
    print(f'Breakdown: {dict(struct_breakdown)}')
    sstats = run_ma_brain_structural(b, struct_analogies, hops=2, min_sim=0.1)
    print(f'Structural: top-1 {sstats["accuracy_top1"]:.1f}%, top-5 {sstats["accuracy_top5"]:.1f}%, time {sstats["elapsed_seconds"]:.4f}s')
    
    # Per-relation structural
    s_by_rel = defaultdict(list)
    for r in sstats["results"]:
        rel = r["analogy"].split(" :: ")[0].split(":")[1] if ":" in r["analogy"].split(" :: ")[0] else "?"
        s_by_rel[r.get("structural_score", 0)].append(r)
    # simpler: just use the relation from the analogy
    for r in sstats["results"]:
        rel = r.get("structural_score", "?")
    print(f'Results per relation not directly available - scores present in each result')

b.save(r'D:\ma_brain_data\brain_state_v8_v3_evaled.json')
print('Saved')
