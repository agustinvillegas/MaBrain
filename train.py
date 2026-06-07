import argparse
import json
import os
import sys
from braincell import Brain


def train_sequence(brain, sequence, relations=None, reward=0.1):

    previous_neuron = None
    rel_idx = 0

    for concept in sequence:

        current_neuron = brain.get_or_create_cell(concept)

        if previous_neuron:

            rel = relations[rel_idx] if relations and rel_idx < len(relations) else None

            synapse = brain.connect(
                previous_neuron,
                current_neuron,
                concept,
                relation=rel
            )

            synapse.activation_trace = 1.0
            rel_idx += 1

        previous_neuron = current_neuron

    brain.reward_thought(reward)


def sample_thoughts(brain, n=5, steps=5, epsilon=0.1):

    seen = set()
    samples = []

    for syn in brain.synapses.values():
        word = syn.origin.word
        if word and word not in seen:
            seen.add(word)
            start = brain.get_or_create_cell(word)
            thought = brain.think(start, steps=steps, temperature=None, epsilon=epsilon)
            samples.append((word, thought))
            if len(samples) >= n:
                break

    return samples


def main():

    parser = argparse.ArgumentParser(
        description="MaBrain - Entrenamiento del grafo de conceptos"
    )

    parser.add_argument(
        "--dataset", "-d",
        default="datasets/dataset.json",
        help="Archivo del dataset (default: dataset.json)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Cantidad de epocas de entrenamiento (default: 3)"
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=0.1,
        help="Recompensa por secuencia por epoca (default: 0.1)"
    )
    parser.add_argument(
        "--load", "-l",
        default=None,
        help="Cargar estado previo del cerebro desde archivo"
    )
    parser.add_argument(
        "--save", "-s",
        default="states/brain_state.json",
        help="Guardar estado final en archivo (default: brain_state.json)"
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Eliminar sinapsis sin uso al finalizar"
    )
    parser.add_argument(
        "--prune-min-usage",
        type=int,
        default=1,
        help="Uso minimo para no podar (default: 1)"
    )
    parser.add_argument(
        "--auto-relations", "-r",
        action="store_true",
        help="Inferir relaciones (IS_A, PROPERTY, NEXT) segun topologia"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Ejecutar evaluacion de analogias al finalizar"
    )
    parser.add_argument(
        "--think-samples",
        type=int,
        default=5,
        help="Cantidad de pensamientos de ejemplo (default: 5)"
    )
    parser.add_argument(
        "--think-steps",
        type=int,
        default=5,
        help="Pasos por pensamiento de ejemplo (default: 5)"
    )

    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Ruta al modelo FastText (ej: cc.es.300.bin)"
    )
    parser.add_argument(
        "--embedding-weight",
        type=float,
        default=0.0,
        help="Peso de similitud semantica en scoring (default: 0.0)"
    )

    args = parser.parse_args()

    bridge = None
    if args.embedding_model:
        from embedding_bridge import EmbeddingBridge
        bridge = EmbeddingBridge(args.embedding_model)

    brain = Brain(
        embedding_bridge=bridge,
        embedding_weight=args.embedding_weight
    )

    if args.load and os.path.exists(args.load):
        brain.load(args.load)
        print(f"Cargado: {len(brain.cells)} neuronas, {len(brain.synapses)} sinapsis")
    else:
        print("Cerebro nuevo")

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    sequences = dataset["sequences"]
    print(f"Dataset: {len(sequences)} secuencias, {args.epochs} epoch(s), reward={args.reward}")
    print()

    for epoch in range(args.epochs):
        for item in sequences:
            train_sequence(
                brain,
                item["sequence"],
                relations=item.get("relations"),
                reward=args.reward
            )

        total_strength = sum(s.strength for s in brain.synapses.values())
        avg_cost = sum(s.cost for s in brain.synapses.values()) / max(len(brain.synapses), 1)

        print(
            f"  Epoca {epoch + 1}/{args.epochs}: "
            f"{len(brain.cells)} neuronas, "
            f"{len(brain.synapses)} sinapsis, "
            f"fuerza total={total_strength:.1f}, "
            f"costo prom={avg_cost:.4f}"
        )

    if args.auto_relations:
        brain.auto_infer_relations()
        n_rels = sum(1 for s in brain.synapses.values() if s.relation is not None)
        rel_types = {}
        for s in brain.synapses.values():
            if s.relation:
                rel_types[s.relation] = rel_types.get(s.relation, 0) + 1
        print(f"\nRelaciones inferidas: {n_rels}/{len(brain.synapses)} sinapsis")
        for rel, count in sorted(rel_types.items()):
            print(f"  {rel}: {count}")

    if args.prune:
        removed = brain.prune(min_usage=args.prune_min_usage)
        print(f"\nPoda: {removed} sinapsis eliminadas (min_usage={args.prune_min_usage})")

    brain.save(args.save)

    print(f"\n--- Resumen final ---")
    print(f"  Neuronas: {len(brain.cells)}")
    print(f"  Sinapsis: {len(brain.synapses)}")
    print(f"  Pensamientos registrados: {len(brain.thoughts)}")
    print(f"  Guardado en: {args.save}")

    if args.think_samples > 0:
        print(f"\n--- Pensamientos de ejemplo ---")
        samples = sample_thoughts(
            brain,
            n=args.think_samples,
            steps=args.think_steps,
            epsilon=0.1
        )
        for word, thought in samples:
            print(f"  [{word}] -> {thought}")

    if args.eval:
        try:
            sys.path.insert(0, "eval")
            from evaluate import run_ma_brain, build_benchmark_from_graph

            analogies = build_benchmark_from_graph(brain)
            if len(analogies) == 0:
                print("\nEvaluacion: no se generaron analogias (muy pocas relaciones)")
            else:
                stats = run_ma_brain(brain, analogies)
                print(
                    f"\nEvaluacion: {stats['total']} analogias, "
                    f"top-1: {stats['accuracy_top1']:.1f}%, "
                    f"top-5: {stats['accuracy_top5']:.1f}%, "
                    f"{stats['elapsed_seconds']:.4f}s"
                )
        except ImportError:
            print("\nEvaluacion no disponible (falta eval/evaluate.py)")

    print("\nListo.")


if __name__ == "__main__":
    main()
