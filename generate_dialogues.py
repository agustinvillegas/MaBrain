"""
Generate conversational dialogues (Capa 4) from ConceptNet knowledge graph.
Output: JSONL with multi-turn dialogues grounded in real entities + relations.

Each dialogue is a list of turns:
  {"user": "question string", "topic": "entity",
   "expected": "target concept", "relation": "RELATION_TYPE"}

Usage:
    python generate_dialogues.py --output data/processed/layer4_dialogues.jsonl
"""
import json
import os
import random
from collections import defaultdict

random.seed(42)

# ── Template patterns for each relation ─────────────────────────
QUESTION_TEMPLATES = {
    "IS_A":        ["What is {e}?", "What kind of thing is {e}?", "Tell me about {e}."],
    "LOCATED_IN":  ["Where is {e} found?", "Where can I find {e}?", "Where is {e} located?"],
    "USED_FOR":    ["What is {e} used for?", "How is {e} used?", "What do people use {e} for?"],
    "HAS_PROPERTY":["What is {e} like?", "Tell me something about {e}.", "What properties does {e} have?"],
    "PART_OF":     ["What is {e} part of?", "What does {e} belong to?"],
    "CAUSE":       ["What does {e} cause?", "What happens because of {e}?", "What results from {e}?"],
    "FUNCTION":    ["What does {e} do?", "How does {e} work?", "What is the function of {e}?"],
    "OPPOSITE":    ["What is the opposite of {e}?", "What is {e} opposite of?"],
    "MADE_OF":     ["What is {e} made of?", "What materials make up {e}?"],
}

RESPONSE_TEMPLATES = {
    "IS_A":        "{E} is a kind of {t}.",
    "LOCATED_IN":  "{E} can be found in {t}.",
    "USED_FOR":    "{E} is used for {t}.",
    "HAS_PROPERTY":"{E} is {t}.",
    "PART_OF":     "{E} is part of {t}.",
    "CAUSE":       "{E} causes {t}.",
    "FUNCTION":    "{E} functions as {t}.",
    "OPPOSITE":    "The opposite of {e} is {t}.",
    "MADE_OF":     "{E} is made of {t}.",
}


def generate_dialogues(schemas_path, output_path, min_relation_types=2,
                       turns_per_dialogue=4, max_dialogues=800,
                       chain_proportion=0.4):
    """Generate multi-turn dialogues grounded in schema entities.

    Generates two types:
      - chain dialogues: walk consecutive steps in multi-step schemas (e.g. IS_A chain)
      - mixed dialogues: ask about different relations for the same entity
    """
    with open(schemas_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── Build entity connection map ─────────────────────────────
    entity_conns = defaultdict(list)
    chain_instances = []  # multi-step chains for chain dialogues
    for schema in data.get("schemas", []):
        rel_seq = schema.get("relation_sequence", [])
        primary_rel = rel_seq[0] if rel_seq else "UNKNOWN"
        for inst in schema.get("instances", []):
            vals = inst[:]  # list [entity, class_1, class_2, ...]
            if len(vals) >= 2:
                entity_conns[vals[0]].append((vals[1], primary_rel))
            # Collect multi-step chains (2+ steps) for dialogue chaining
            if len(vals) >= 3 and primary_rel in ("IS_A", "LOCATED_IN", "PART_OF", "CAUSE"):
                chain_instances.append((primary_rel, vals))

    # ── Single-entity mixed-relation candidates ─────────────────
    candidates = []
    for entity, conns in entity_conns.items():
        rel_types = set(r for _, r in conns)
        if len(rel_types) >= min_relation_types:
            by_rel = defaultdict(list)
            for t, r in conns:
                by_rel[r].append(t)
            candidates.append((entity, dict(by_rel)))

    random.shuffle(candidates)
    random.shuffle(chain_instances)
    print(f"  Entities with >= {min_relation_types} rel-types: {len(candidates)}")
    print(f"  Multi-step chain instances: {len(chain_instances)}")

    dialogues = []
    used_pairs = set()

    # ── Type A: Chain dialogues (follow consecutive steps) ──────
    n_chain = int(max_dialogues * chain_proportion)
    for rel, vals in chain_instances:
        if len(dialogues) >= n_chain:
            break
        # Build turns walking the chain: step 1, step 2, ...
        turns = []
        for i in range(len(vals) - 1):
            entity = vals[i]
            target = vals[i + 1]
            pair_key = (entity, target, rel)
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            if i == 0:
                q = random.choice(QUESTION_TEMPLATES.get(rel, ["What is {e}?"])).format(e=entity)
            else:
                q = random.choice([
                    f"And what is {entity}?",
                    f"What about {entity}?",
                    f"Tell me about {entity}.",
                ])
            turns.append({
                "user": q,
                "topic": entity,
                "expected": target,
                "relation": rel,
            })
            if len(turns) >= 3:  # max 3 chain steps
                break
        if len(turns) >= 2:
            dialogues.append({"turns": turns, "type": "chain"})

    # ── Type B: Mixed-relation dialogues (different rels, same entity) ──
    for entity, by_rel in candidates:
        if len(dialogues) >= max_dialogues:
            break

        rels = sorted(by_rel.keys())
        random.shuffle(rels)
        selected_rels = rels[:turns_per_dialogue]
        if len(selected_rels) < 2:
            continue

        turns = []
        for rel in selected_rels:
            targets = by_rel[rel]
            target = random.choice(targets)
            pair_key = (entity, target, rel)
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            q_templates = QUESTION_TEMPLATES.get(rel, ["Tell me about {e}."])
            question = random.choice(q_templates).format(e=entity)
            turns.append({
                "user": question,
                "topic": entity,
                "expected": target,
                "relation": rel,
            })
        if len(turns) >= 2:
            dialogues.append({"turns": turns, "type": "mixed"})

    # ── Save ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n  Generated {len(dialogues)} dialogues ({sum(1 for d in dialogues if d.get('type')=='chain')} chain, {sum(1 for d in dialogues if d.get('type')=='mixed')} mixed)")
    total_turns = sum(len(d["turns"]) for d in dialogues)
    print(f"  Total turns: {total_turns}")
    print(f"  Saved: {output_path}")

    # ── Sample ──────────────────────────────────────────────────
    if dialogues:
        sample = dialogues[0]
        print(f"\n  Sample dialogue ({sample.get('type')}):")
        for i, turn in enumerate(sample["turns"]):
            resp = RESPONSE_TEMPLATES.get(turn["relation"], "{E} -> {t}")
            response_text = resp.format(E=turn["topic"].capitalize(),
                                        e=turn["topic"], t=turn["expected"])
            print(f"    [{i+1}] User: {turn['user']}")
            print(f"           Bot: {response_text}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate conversational dialogues (Capa 4)")
    parser.add_argument("--layer3", default="data/processed/layer3_schemas.json",
                        help="Capa 3 schemas JSON")
    parser.add_argument("--output", "-o", default="data/processed/layer4_dialogues.jsonl",
                        help="Output dialogues JSONL")
    parser.add_argument("--min-rels", type=int, default=2,
                        help="Minimum relation types per entity (default: 2)")
    parser.add_argument("--turns", type=int, default=4,
                        help="Turns per dialogue")
    parser.add_argument("--max", type=int, default=800,
                        help="Max dialogues (default: 800)")
    parser.add_argument("--chain-prop", type=float, default=0.4,
                        help="Proportion of chain dialogues (default: 0.4)")
    args = parser.parse_args()

    print("=== Generate Dialogues (Capa 4) ===\n")
    generate_dialogues(
        args.layer3, args.output,
        min_relation_types=args.min_rels,
        turns_per_dialogue=args.turns,
        max_dialogues=args.max,
        chain_proportion=args.chain_prop,
    )
    print("\nDone.")
