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


def generate_dialogues(schemas_path, output_path, min_relation_types=3,
                       turns_per_dialogue=4, max_dialogues=500):
    """Generate multi-turn dialogues grounded in schema entities."""
    with open(schemas_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build entity connection map: entity -> [(target, relation)]
    entity_conns = defaultdict(list)
    for schema in data.get("schemas", []):
        rel_seq = schema.get("relation_sequence", [])
        primary_rel = rel_seq[0] if rel_seq else "UNKNOWN"
        for inst in schema.get("instances", []):
            vals = list(inst.values())
            if len(vals) >= 2:
                entity_conns[vals[0]].append((vals[1], primary_rel))

    # Find entities with enough relation variety
    candidates = []
    for entity, conns in entity_conns.items():
        rel_types = set(r for _, r in conns)
        if len(rel_types) >= min_relation_types:
            # Group by relation, pick best target for each
            by_rel = defaultdict(list)
            for t, r in conns:
                by_rel[r].append(t)
            candidates.append((entity, dict(by_rel)))

    random.shuffle(candidates)
    print(f"  Candidates with >= {min_relation_types} relation types: {len(candidates)}")

    dialogues = []
    used_pairs = set()

    for entity, by_rel in candidates[:max_dialogues * 2]:
        if len(dialogues) >= max_dialogues:
            break

        # Pick up to turns_per_dialogue different relations
        rels = sorted(by_rel.keys())
        random.shuffle(rels)
        selected_rels = rels[:turns_per_dialogue]

        if len(selected_rels) < 2:
            continue

        turns = []
        topic = entity
        for rel in selected_rels:
            targets = by_rel[rel]
            target = random.choice(targets)

            # Avoid duplicate (entity, target, rel) across dialogues
            pair_key = (entity, target, rel)
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)

            question_templates = QUESTION_TEMPLATES.get(rel, ["Tell me about {e}."])
            question = random.choice(question_templates).format(e=entity)

            turns.append({
                "user": question,
                "topic": topic,
                "expected": target,
                "relation": rel,
            })

        if len(turns) >= 2:
            dialogues.append({"turns": turns})

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"  Generated {len(dialogues)} dialogues")
    total_turns = sum(len(d["turns"]) for d in dialogues)
    print(f"  Total turns: {total_turns}")
    print(f"  Saved: {output_path}")

    # Show a sample
    if dialogues:
        print(f"\n  Sample dialogue:")
        for i, turn in enumerate(dialogues[0]["turns"]):
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
    parser.add_argument("--min-rels", type=int, default=3,
                        help="Minimum relation types per entity")
    parser.add_argument("--turns", type=int, default=4,
                        help="Turns per dialogue")
    parser.add_argument("--max", type=int, default=500,
                        help="Max dialogues")
    args = parser.parse_args()

    print("=== Generate Dialogues (Capa 4) ===\n")
    generate_dialogues(
        args.layer3, args.output,
        min_relation_types=args.min_rels,
        turns_per_dialogue=args.turns,
        max_dialogues=args.max,
    )
    print("\nDone.")
