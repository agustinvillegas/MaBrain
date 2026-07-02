import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from braincell import Brain


def print_banner():
    print()
    print("  MaBrain - CLI interactiva")
    print("  " + "-" * 40)
    print("  comandos:")
    print("    chat <mensaje>           - responder con WM contextual")
    print("    wm                       - mostrar memoria de trabajo")
    print("    think <concepto> [pasos] - caminar el grafo")
    print("    analogia <a> <b> <c>    - A:B :: C:?")
    print("    predice <texto> [top_k] - predecir siguiente")
    print("    explica <concepto>      - conexiones salientes")
    print("    entrena [n]             - re-entrenar n epocas")
    print("    evalua                  - benchmark de analogias")
    print("    estadisticas            - resumen del grafo")
    print("    ayuda                   - esta pantalla")
    print("    salir                   - salir")
    print()


def cmd_think(brain, args):

    if not args:
        print("  uso: think <concepto> [pasos]")
        return

    concept = args[0]
    steps = int(args[1]) if len(args) > 1 else 5

    cell_id = brain.concept_registry.get(concept)
    if not cell_id:
        print(f"  X '{concept}' no encontrado en el grafo")
        return

    cell = brain.cells[cell_id]

    print(f"  Pensando desde '{concept}' ({steps} pasos, temp=dinamica):")
    print()

    current = cell
    total_cost = 0.0

    for i in range(steps):
        if not current.synapses_out:
            print(f"    [{i + 1}] '{current.word}' -> callejon sin salida")
            break

        options = current.synapses_out
        choice = brain._select_synapse(options, temperature=None, epsilon=0.1)

        cost_step = choice.efficiency()
        total_cost += cost_step
        rel = f" ({choice.relation})" if choice.relation else ""

        print(
            f"    [{i + 1}] {current.word} -{rel}-> {choice.concept}"
            f"  (costo={cost_step:.3f}, fuerza={choice.strength:.2f})"
        )

        choice.inference_usage += 1
        current = choice.target

    print(f"\n  Costo total: {total_cost:.3f}")

    if len(options) > 1:
        print(f"  Opciones en paso 1: {len(options)} caminos posibles")


def cmd_analogy(brain, args):

    if len(args) < 3:
        print("  uso: analogia <a> <b> <c>")
        return

    a, b, c = args[0], args[1], args[2]
    results = brain.analogy(a, b, c, top_k=5)

    if not results:
        print(f"  X '{a}:{b} :: {c}:?' -> sin candidatos")
        return

    print(f"  '{a}:{b} :: {c}:?' -> candidatos:")
    print()

    for i, r in enumerate(results):
        tick = ">" if i == 0 else " "
        print(
            f"    {tick} {r['d']}"
            f"  (score={r['score']:.3f}, relacion={r['relation']})"
        )

        t = r["trace"]
        print(
            f"       traza: {t['a_b']['from']} -({t['a_b']['relation']})-> {t['a_b']['to']}"
            f"  |  {t['c_d']['from']} -({t['c_d']['relation']})-> {t['c_d']['to']}"
        )


def cmd_predict(brain, args):

    if not args:
        print("  uso: predice <texto> [top_k]")
        return

    top_k = 5
    if len(args) > 1 and args[-1].isdigit():
        top_k = int(args[-1])
        text = " ".join(args[:-1])
    else:
        text = " ".join(args)

    preds = brain.predict_next(text, top_k=top_k)

    if not preds:
        last_word = text.split()[-1] if text.split() else ""
        cell_id = brain.concept_registry.get(last_word)
        if not cell_id:
            print(f"  X '{last_word}' no encontrado en el grafo")
        else:
            print(f"  X '{last_word}' no tiene salidas")
        return

    print(f"  Contexto: '{text}'")
    print(f"  Predicciones:")
    for word, prob in preds:
        bar = "=" * int(prob * 30)
        print(f"    {bar} {word}  ({prob:.1%})")


def cmd_explain(brain, args):

    if not args:
        print("  uso: explica <concepto>")
        return

    concept = args[0]

    cell_id = brain.concept_registry.get(concept)
    if not cell_id:
        print(f"  X '{concept}' no encontrado")
        return

    cell = brain.cells[cell_id]
    print(f"  '{concept}' ({cell.id[:8]}):")
    print(f"    energia: {cell.energy:.3f}, activacion: {cell.activation:.3f}")
    print(f"    sinapsis salientes: {len(cell.synapses_out)}")
    print(f"    sinapsis entrantes: {len(cell.synapses_in)}")
    print()

    if cell.synapses_out:
        print(f"  Salidas:")
        for syn in sorted(cell.synapses_out, key=lambda x: -x.strength):
            target_word = syn.target.word or syn.target.id[:8]
            rel = syn.relation or "?"
            print(
                f"    -> {target_word}"
                f"  [{rel}]"
                f"  fuerza={syn.strength:.3f}"
                f"  costo={syn.cost:.3f}"
                f"  eff={syn.efficiency():.3f}"
                f"  usos={syn.inference_usage}"
            )

    if cell.synapses_in:
        print(f"  Entradas:")
        for syn in sorted(cell.synapses_in, key=lambda x: -x.strength)[:5]:
            origin_word = syn.origin.word or syn.origin.id[:8]
            rel = syn.relation or "?"
            print(
                f"    <- {origin_word}"
                f"  [{rel}]"
                f"  fuerza={syn.strength:.3f}"
                f"  costo={syn.cost:.3f}"
            )


def train_sequence(brain, sequence, relations=None, reward=0.1):

    previous_neuron = None
    rel_idx = 0

    for concept in sequence:
        current_neuron = brain.get_or_create_cell(concept)
        if previous_neuron:
            rel = relations[rel_idx] if relations and rel_idx < len(relations) else None
            synapse = brain.connect(previous_neuron, current_neuron, concept, relation=rel)
            synapse.activation_trace = 1.0
            rel_idx += 1
        previous_neuron = current_neuron

    brain.reward_thought(reward)


def cmd_train(brain, args):

    dataset_path = "datasets/dataset.json"
    epochs = int(args[0]) if args else 3

    if not os.path.exists(dataset_path):
        print(f"  X dataset '{dataset_path}' no encontrado")
        return

    import json

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    sequences = dataset["sequences"]
    print(f"  Entrenando {len(sequences)} secuencias x {epochs} epocas...")

    for epoch in range(epochs):
        for item in sequences:
            train_sequence(
                brain,
                item["sequence"],
                relations=item.get("relations"),
                reward=0.1
            )
            print(
                f"    Epoca {epoch + 1}/{epochs}:"
                f" {len(brain.cells)} neuronas,"
                f" {len(brain.synapses)} sinapsis",
                end="\r"
            )
        print()

    brain.auto_infer_relations()
    print(f"  Relaciones inferidas.")

    brain.save()
    print(f"  Guardado.")


def cmd_evaluate(brain, args):

    try:
        sys.path.insert(0, "eval")
        from evaluate import run_ma_brain, build_benchmark_from_graph
    except ImportError:
        print("  X evaluacion no disponible (falta eval/evaluate.py)")
        return

    n_rels = sum(1 for s in brain.synapses.values() if s.relation)
    if n_rels == 0:
        print("  Sin relaciones. Ejecutando auto_infer_relations()...")
        brain.auto_infer_relations()

    analogies = build_benchmark_from_graph(brain)
    if len(analogies) == 0:
        print("  X no se generaron analogias (muy pocas relaciones)")
        return

    stats = run_ma_brain(brain, analogies)
    print(
        f"  {stats['total']} analogias generadas,"
        f" top-1: {stats['accuracy_top1']:.1f}%,"
        f" top-5: {stats['accuracy_top5']:.1f}%"
    )
    print(f"  Tiempo: {stats['elapsed_seconds']:.4f}s, ops: {stats['ops_estimated']}")


def cmd_stats(brain, args):

    n_rels = sum(1 for s in brain.synapses.values() if s.relation)
    rel_types = {}
    for s in brain.synapses.values():
        if s.relation:
            rel_types[s.relation] = rel_types.get(s.relation, 0) + 1

    total_energy = sum(c.energy for c in brain.cells.values())
    total_activation = sum(c.activation for c in brain.cells.values())
    avg_strength = sum(s.strength for s in brain.synapses.values()) / max(len(brain.synapses), 1)
    avg_cost = sum(s.cost for s in brain.synapses.values()) / max(len(brain.synapses), 1)

    print(f"  Grafo:")
    print(f"    Neuronas:     {len(brain.cells)}")
    print(f"    Sinapsis:     {len(brain.synapses)}")
    print(f"    Conceptos:    {len(brain.concept_registry)}")
    print(f"    Pensamientos: {len(brain.thoughts)}")
    print()
    print(f"  Sinapsis con relacion: {n_rels}/{len(brain.synapses)}")
    for rel, count in sorted(rel_types.items()):
        print(f"    {rel}: {count}")
    print()
    print(f"  Fuerza promedio: {avg_strength:.3f}")
    print(f"  Costo promedio:  {avg_cost:.4f}")
    print(f"  Energia total:   {total_energy:.3f}")
    print(f"  Activacion total: {total_activation:.3f}")

    hubs = sorted(
        [(cid, c) for cid, c in brain.cells.items()],
        key=lambda x: -len(x[1].synapses_out)
    )[:3]
    if hubs[0][1].synapses_out:
        print()
        print(f"  Top hubs (mas salidas):")
        for cid, c in hubs:
            label = c.word or cid[:8]
            print(f"    {label}: {len(c.synapses_out)} salidas, {len(c.synapses_in)} entradas")


def cmd_chat(brain, args):
    if not args:
        print("  uso: chat <mensaje>")
        return
    msg = " ".join(args)
    resp = brain.get_response(msg, steps=8)
    print(f"  Tu: {msg}")
    print(f"  Brain: {resp}")
    # Show WM context summary
    active = brain.wm.get_active_entities(min_salience=0.1)
    if active:
        print(f"  [entidades activas: {', '.join(active[:5])}]")
    print(f"  [turnos: {len(brain.wm.turns)}]")


def cmd_wm(brain, args):
    print(f"  Memoria de trabajo:")
    print(f"    Turnos: {len(brain.wm.turns)}/{brain.wm.max_turns}")
    for t in brain.wm.last_turns(5):
        roles = {"user": "Tu", "bot": "Brain"}
        label = roles.get(t.role, t.role)
        preview = t.message[:60] + ("..." if len(t.message) > 60 else "")
        ents = f" [{', '.join(t.entities)}]" if t.entities else ""
        print(f"    {label}: {preview}{ents}")
    active = brain.wm.get_active_entities(min_salience=0.1)
    if active:
        print(f"    Entidades activas: {', '.join(active[:8])}")
    else:
        print(f"    Entidades activas: (ninguna)")
    topic = brain.wm.current_topic()
    if topic:
        print(f"    Tema actual: {topic}")


def main():

    brain = Brain(context_bias_weight=0.3)
    loaded_path = "data/brain_states/brain_massive_v1.json"
    if os.path.exists(loaded_path):
        brain.load(loaded_path)
        print(f"Cargado: {len(brain.cells)} neuronas, {len(brain.synapses)} sinapsis")
        n_rels = sum(1 for s in brain.synapses.values() if s.relation)
        if n_rels == 0:
            print("Infiltrando relaciones...")
            brain.auto_infer_relations()
            n_rels = sum(1 for s in brain.synapses.values() if s.relation)
            print(f"  {n_rels} relaciones inferidas")
    else:
        print("Cerebro nuevo. Usa 'entrena' para entrenar.")

    print_banner()

    while True:
        try:
            line = input("  brain> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("salir", "exit", "quit"):
            break
        elif cmd in ("ayuda", "help", "?"):
            print_banner()
        elif cmd == "think":
            cmd_think(brain, args)
        elif cmd in ("analogia", "analogy"):
            cmd_analogy(brain, args)
        elif cmd in ("predice", "predict"):
            cmd_predict(brain, args)
        elif cmd in ("explica", "explain"):
            cmd_explain(brain, args)
        elif cmd in ("entrena", "train"):
            cmd_train(brain, args)
        elif cmd == "evalua":
            cmd_evaluate(brain, args)
        elif cmd in ("estadisticas", "stats"):
            cmd_stats(brain, args)
        elif cmd == "chat":
            cmd_chat(brain, args)
        elif cmd == "wm":
            cmd_wm(brain, args)
        else:
            # Auto-route unknown input to chat
            cmd_chat(brain, parts)

        print()

    print("  Chao.")


if __name__ == "__main__":
    main()
