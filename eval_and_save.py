"""Load brain, run eval to populate inference_usage, then save."""
import sys
sys.path.insert(0, ".")
from braincell import Brain
from eval.evaluate import build_benchmark_from_graph, run_ma_brain, print_report

brain = Brain()
brain.load("states/brain_state_v5_embed.json")
print(f"Cargado: {len(brain.cells)} celulas, {len(brain.synapses)} sinapsis")

analogies = build_benchmark_from_graph(brain)
print(f"Benchmark: {len(analogies)} analogias")

stats = run_ma_brain(brain, analogies)
print_report(stats)

brain.save("states/brain_state_v5_evaled.json")
print("Guardado con inference_usage en states/brain_state_v5_evaled.json")
