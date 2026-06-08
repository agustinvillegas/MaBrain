import sys
sys.path.insert(0, ".")
from braincell import Brain
from eval.evaluate import build_benchmark_from_graph, run_ma_brain, build_benchmark_structural, run_ma_brain_structural
from collections import Counter, defaultdict

b = Brain()
b.load(r'D:\ma_brain_data\brain_state_v8_v3_evaled.json')

# Structural benchmark
struct_analogies = build_benchmark_structural(b, max_per_relation=200)
print(f'Structural benchmark: {len(struct_analogies)} analogies')
struct_breakdown = Counter(a["relation"] for a in struct_analogies)
print(f'Breakdown: {dict(struct_breakdown)}')

sstats = run_ma_brain_structural(b, struct_analogies, hops=2, min_sim=0.1)
print(f'\nOverall structural: top-1 {sstats["accuracy_top1"]:.1f}%, top-5 {sstats["accuracy_top5"]:.1f}%, time {sstats["elapsed_seconds"]:.4f}s')

# Per-relation
by_rel = defaultdict(list)
for r in sstats["results"]:
    rel = r.get("relation", "?")
    by_rel[rel].append(r)

print('\n--- Per-relation structural ---')
for rel in sorted(by_rel.keys()):
    items = by_rel[rel]
    corr = sum(1 for i in items if i["correct"])
    top5 = sum(1 for i in items if i["in_top5"])
    print(f'  {rel}: {corr}/{len(items)} ({corr/len(items)*100:.1f}%) top-1, top-5 {top5}/{len(items)} ({top5/len(items)*100:.1f}%)')
