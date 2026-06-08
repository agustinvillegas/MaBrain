import sys
sys.path.insert(0, ".")
from braincell import Brain
from eval.evaluate import build_benchmark_from_graph, run_ma_brain, build_benchmark_structural, run_ma_brain_structural
from collections import Counter, defaultdict

b = Brain()
b.load(r'D:\ma_brain_data\brain_state_v8_v3_pruned.json')
print(f'Brain: {len(b.cells)} cells, {len(b.synapses)} synapses')

rel_counts = Counter(s.relation for s in b.synapses.values() if s.relation)
print(f'Relations: {dict(rel_counts.most_common(10))}')

# 1. Full benchmark
analogies = build_benchmark_from_graph(b, max_per_relation=500)
stats = run_ma_brain(b, analogies)

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

# 2. Structural benchmark
struct_analogies = build_benchmark_structural(b, max_per_relation=200)
print(f'\n--- Structural benchmark ---')
print(f'Total: {len(struct_analogies)}')
struct_breakdown = Counter(a["relation"] for a in struct_analogies)
print(f'Breakdown: {dict(struct_breakdown)}')

sstats = run_ma_brain_structural(b, struct_analogies, hops=2, min_sim=0.1)
print(f'Overall: top-1 {sstats["accuracy_top1"]:.1f}%, top-5 {sstats["accuracy_top5"]:.1f}%, time {sstats["elapsed_seconds"]:.4f}s')

s_by_rel = defaultdict(list)
for r in sstats["results"]:
    rel = r.get("relation", "?")
    s_by_rel[rel].append(r)

print('\n--- Per-relation structural ---')
for rel in sorted(s_by_rel.keys()):
    items = s_by_rel[rel]
    corr = sum(1 for i in items if i["correct"])
    top5 = sum(1 for i in items if i["in_top5"])
    print(f'  {rel}: {corr}/{len(items)} ({corr/len(items)*100:.1f}%) top-1, top-5 {top5}/{len(items)} ({top5/len(items)*100:.1f}%)')

b.save(r'D:\ma_brain_data\brain_state_v8_v3_pruned_evaled.json')
print('\nSaved')
