import json
from pathlib import Path

from braincell import Brain


DATASET_PATH = Path("dataset.json")


def quality_score(predicted_text: str, expected_text: str) -> dict:
    predicted_tokens = [token for token in Brain.tokenize(predicted_text) if token and token not in Brain.STOPWORDS]
    expected_tokens = [token for token in Brain.tokenize(expected_text) if token and token not in Brain.STOPWORDS]
    predicted_set = set(predicted_tokens)
    expected_set = set(expected_tokens)

    overlap = len(predicted_set & expected_set)
    extra = len(predicted_set - expected_set)
    missing = len(expected_set - predicted_set)
    expected_count = max(1, len(expected_set))

    coverage = overlap / expected_count
    noise = extra / max(1, len(predicted_set))
    completeness = 1.0 - (missing / expected_count)
    quality = max(0.0, min(1.0, (coverage * 0.65) + (completeness * 0.35) - (noise * 0.25)))

    return {
        "coverage": coverage,
        "noise": noise,
        "completeness": completeness,
        "quality": quality,
    }


def format_quality(metrics: dict) -> str:
    return (
        f"cov={metrics['coverage']:.2f} "
        f"noise={metrics['noise']:.2f} "
        f"comp={metrics['completeness']:.2f} "
        f"q={metrics['quality']:.2f}"
    )


def estimate_response_quality(thought) -> float:
    if not thought.output_values:
        return 0.0
    average_value = sum(thought.output_values) / len(thought.output_values)
    diversity_bonus = min(0.25, len(thought.output_tokens) * 0.05)
    return max(0.0, min(1.0, (average_value * 0.8) + diversity_bonus))


def classify_quality(value: float) -> str:
    if value >= 0.85:
        return "buena"
    if value >= 0.65:
        return "aceptable"
    return "mejorable"


def block_tokens(text: str) -> list[str]:
    blocks: list[str] = []
    for segment in text.split(","):
        tokens = [token for token in Brain.tokenize(segment) if token and token not in Brain.STOPWORDS]
        if tokens:
            blocks.append(" ".join(tokens))
    return blocks


def block_labels_from_thought(thought) -> list[str]:
    labels = [token.strip() for token in thought.output_tokens if token.strip()]
    if len(labels) >= 2 and not thought.output_values:
        return labels
    return labels


def block_quality(predicted_text: str, expected_text: str) -> dict:
    predicted_blocks = block_tokens(predicted_text)
    expected_blocks = block_tokens(expected_text)

    predicted_set = set(predicted_blocks)
    expected_set = set(expected_blocks)
    overlap = predicted_set & expected_set
    extra = predicted_set - expected_set
    missing = expected_set - predicted_set

    expected_count = max(1, len(expected_set))
    coverage = len(overlap) / expected_count
    noise = len(extra) / max(1, len(predicted_set))
    completeness = 1.0 - (len(missing) / expected_count)
    quality = max(0.0, min(1.0, (coverage * 0.7) + (completeness * 0.3) - (noise * 0.2)))

    return {
        "coverage": coverage,
        "noise": noise,
        "completeness": completeness,
        "quality": quality,
        "matched": sorted(overlap),
        "extra": sorted(extra),
        "missing": sorted(missing),
    }


def build_block_analysis(thought, correction: str) -> list[dict]:
    predicted_blocks = block_labels_from_thought(thought)
    expected_blocks = block_tokens(correction)
    predicted_set = set(predicted_blocks)
    expected_set = set(expected_blocks)

    analysis: list[dict] = []
    for block in sorted(predicted_set | expected_set):
        analysis.append(
            {
                "block": block,
                "predicted": block in predicted_set,
                "expected": block in expected_set,
                "status": (
                    "matched"
                    if block in predicted_set and block in expected_set
                    else "extra"
                    if block in predicted_set
                    else "missing"
                ),
            }
        )
    return analysis


def normalize_text(text: str) -> str:
    tokens = [token for token in Brain.tokenize(text) if token and token not in Brain.STOPWORDS]
    return " ".join(tokens)


def normalize_feedback(text: str) -> str:
    return text.strip().lower().lstrip("> ").strip()


def tokenize_text(text: str) -> str:
    tokens = [token for token in Brain.tokenize(text) if token and token not in Brain.STOPWORDS]
    return " ".join(tokens)


def main() -> None:
    brain = Brain()
    brain.load()

    print("Red de rutas dinamicas")
    print("Comandos:")
    print("  train <entrada> => <salida>")
    print("  batch")
    print("  autotrain")
    print("  reset_memory")
    print("  ask <entrada>")
    print("  save")
    print("  exit")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        if raw.lower() == "exit":
            break

        if raw.lower() == "save":
            brain.save()
            print("guardado")
            continue

        if raw.lower() == "reset_memory":
            confirmation = normalize_feedback(input("esto borra la memoria. escribir 'reset' para confirmar: "))
            if confirmation == "reset":
                brain.reset_memory()
                print("memoria reiniciada")
            else:
                print("cancelado")
            continue

        if raw.lower() == "batch":
            if not DATASET_PATH.exists():
                print("no existe dataset.json")
                continue

            data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            pairs = data.get("pairs", [])
            if not pairs:
                print("dataset vacio")
                continue

            for pair in pairs:
                input_text = pair["input"]
                output_text = pair["output"]
                thought = brain.train_pair(input_text, output_text)
                brain.reward(thought)

            brain.save()
            print(f"entrenamiento completado: {len(pairs)} pares")
            continue

        if raw.lower() == "autotrain":
            if not DATASET_PATH.exists():
                print("no existe dataset.json")
                continue

            data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            pairs = data.get("pairs", [])
            if not pairs:
                print("dataset vacio")
                continue

            correct = 0
            for pair in pairs:
                input_text = pair["input"]
                expected_text = pair["output"].strip().lower()
                thought = brain.infer(input_text)
                predicted_text = " ".join(thought.output_tokens).strip().lower()
                metrics = quality_score(predicted_text, expected_text)
                score = metrics["quality"]

                if score >= 0.75:
                    brain.reward(thought, amount=score)
                    correct += 1
                else:
                    brain.punish(thought, amount=max(0.1, 1.0 - score))

                brain.train_pair(input_text, expected_text)

            brain.save()
            print(f"autotrain completado: {correct}/{len(pairs)} correctos")
            continue

        if raw.startswith("train "):
            payload = raw[6:].strip()
            if "=>" not in payload:
                print("usa: train entrada => salida")
                continue
            left, right = [part.strip() for part in payload.split("=>", 1)]
            thought = brain.train_pair(left, right)
            print("ruta creada:", ", ".join(map(str, thought.active_synapses)) or "ninguna")
            continue

        if raw.startswith("ask "):
            query = raw[4:].strip()
            thought = brain.infer(query)
            ranked_outputs = brain.explain_thought(thought)
            if ranked_outputs:
                response = ", ".join(
                    f"{label} [{score:.2f}]" for label, score in ranked_outputs[:3]
                )
            else:
                response = "(sin ruta)"
            print(response)
            if thought.output_tokens:
                token_details = ", ".join(
                    f"{token}:{value:.2f}"
                    for token, value in zip(thought.output_tokens, thought.output_values)
                )
                print(f"tokens -> {token_details}")
            estimated_quality = estimate_response_quality(thought)
            print(f"calidad estimada -> {estimated_quality:.2f} ({classify_quality(estimated_quality)})")
            feedback = normalize_feedback(input("correcto? (y/n): "))
            if feedback in {"y", "yes", "si", "s"}:
                predicted_text = tokenize_text(" ".join(thought.output_tokens))
                metrics = quality_score(predicted_text, predicted_text)
                print(format_quality(metrics))
                brain.reward(thought, amount=max(0.5, metrics["quality"]))
            elif feedback in {"n", "no"}:
                correction = input("corrige la respuesta: ").strip()
                correction_text = tokenize_text(correction)
                predicted_text = tokenize_text(" ".join(thought.output_tokens))
                metrics = quality_score(predicted_text, correction_text)
                predicted_blocks = ", ".join(block_labels_from_thought(thought))
                block_metrics = block_quality(predicted_blocks, correction)
                print(format_quality(metrics))
                if block_metrics["matched"] or block_metrics["extra"] or block_metrics["missing"]:
                    print(
                        "bloques -> "
                        f"ok={', '.join(block_metrics['matched']) or '(ninguno)'} | "
                        f"extra={', '.join(block_metrics['extra']) or '(ninguno)'} | "
                        f"faltan={', '.join(block_metrics['missing']) or '(ninguno)'}"
                    )

                if block_metrics["coverage"] > 0:
                    brain.reward(thought, amount=0.75 + (block_metrics["coverage"] * 0.75))
                if block_metrics["extra"]:
                    brain.punish(thought, amount=max(0.3, len(block_metrics["extra"]) * 0.4))
                if block_metrics["missing"]:
                    brain.punish(thought, amount=max(0.3, len(block_metrics["missing"]) * 0.35))
                if metrics["noise"] > 0:
                    brain.punish(thought, amount=max(0.2, metrics["noise"] * 1.0))
                if metrics["quality"] < 0.75:
                    brain.punish(thought, amount=max(0.2, (1.0 - metrics["quality"]) * 1.25))

                if correction_text:
                    brain.train_pair(query, correction)
                    corrected_thought = brain.thoughts[-1]
                    corrected_thought.block_analysis = build_block_analysis(corrected_thought, correction)
                    print(
                        "analisis -> "
                        + ", ".join(
                            f"{item['block']}:{item['status']}" for item in corrected_thought.block_analysis
                        )
                    )
                    brain.reward(corrected_thought, amount=max(0.5, metrics["quality"]))
            else:
                brain.punish(thought, amount=1.0)
            continue

        if raw.startswith("debug "):
            query = raw[6:].strip()
            thought = brain.infer(query)
            details = brain.debug_thought(thought)
            if not details:
                print("(sin ruta)")
            else:
                for item in details[:10]:
                    print(
                        f"{item['origin']} -> {item['target']} | "
                        f"score={item['score']:.2f} | "
                        f"strength={item['strength']:.2f} | "
                        f"cost={item['cost']:.2f} | "
                        f"merit={item['merit']:.2f} | "
                        f"usage={item['usage']} | "
                        f"reward={item['reward']:.2f}"
                    )
            continue

        print("comando no reconocido")

    brain.save()


if __name__ == "__main__":
    main()
