import argparse
from braincell import Brain
from embedding_bridge import EmbeddingBridge

def main():
    parser = argparse.ArgumentParser(description="Podar sinapsis guiada por embeddings")
    parser.add_argument("--load", default="states/brain_state_v6_evaled.json", help="Estado del cerebro a cargar")
    parser.add_argument("--save", default="states/brain_state_v6_pruned.json", help="Archivo de salida")
    parser.add_argument("--embedding-model", default="models/cc.es.300.bin", help="Modelo FastText")
    parser.add_argument("--min-usage", type=int, default=0, help="Uso mínimo (0 = ignorar uso)")
    parser.add_argument("--min-sim", type=float, default=0.3, help="Similitud semántica mínima (default: 0.3)")
    args = parser.parse_args()

    print(f"Cargando cerebro desde {args.load}...")
    brain = Brain()
    brain.load(args.load)

    print(f"Cargando modelo de embeddings desde {args.embedding_model}...")
    bridge = EmbeddingBridge(model_path=args.embedding_model)

    total = len(brain.synapses)
    to_remove = []

    for syn_id, syn in list(brain.synapses.items()):
        w1 = syn.origin.word
        w2 = syn.target.word
        if w1 is None or w2 is None:
            continue
        sim = bridge.similarity(w1, w2)
        if sim < args.min_sim:
            if args.min_usage == 0 or syn.inference_usage < args.min_usage:
                to_remove.append(syn_id)

    for syn_id in to_remove:
        syn = brain.synapses[syn_id]
        # Remove from cell lists
        if syn in syn.origin.synapses_out:
            syn.origin.synapses_out.remove(syn)
        if syn in syn.target.synapses_in:
            syn.target.synapses_in.remove(syn)
        del brain.synapses[syn_id]

    removed = len(to_remove)
    kept = total - removed

    print(f"  Sinapsis total: {total}")
    print(f"  Eliminadas: {removed}")
    print(f"  Conservadas: {kept}")

    brain.save(args.save)
    print(f"  Guardado en: {args.save}")

if __name__ == "__main__":
    main()
