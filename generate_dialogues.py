"""
Generate conversational dialogues (Capa 4) from Layer 1 triplets.
Output: JSONL with multi-turn dialogues grounded in real entities + relations.

Generates diverse question-answer pairs from the graph:
  - Chain dialogues: walk consecutive steps (e.g. IS_A chain of 2-3 steps)
  - Mixed dialogues: ask different relations for the same entity
  - Single-turn dialogues: bulk reinforcement for coverage

Usage:
    python generate_dialogues.py --layer1 data/processed/layer1_triplets.json
                                 --output data/processed/layer4_dialogues.jsonl
                                 --max 5000
"""
import json
import os
import random
from collections import defaultdict

random.seed(42)

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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(triplets):
    out_edges = defaultdict(list)
    for t in triplets:
        frm = t.get("from", "").strip()
        to = t.get("to", "").strip()
        rel = t.get("relation", "").strip()
        if frm and to and rel:
            out_edges[frm].append((to, rel))
    return out_edges


def find_chains(entity, out_edges, max_depth=3, max_chains=5000):
    chains = []
    visited = set()
    queue = [(entity, [entity], [])]
    while queue and len(chains) < max_chains:
        current, path, rels = queue.pop(0)
        if len(path) >= max_depth:
            continue
        for to, rel in out_edges.get(current, []):
            if rel not in ("IS_A", "CAUSE", "LOCATED_IN", "PART_OF"):
                continue
            if to in visited:
                continue
            new_path = path + [to]
            new_rels = rels + [rel]
            chains.append((new_path, new_rels))
            visited.add(to)
            queue.append((to, new_path, new_rels))
    return chains


def generate_dialogues(triplets_path, output_path, max_dialogues=5000,
                       turns_per_dialogue=3, chain_proportion=0.3):
    data = load_json(triplets_path)
    triplets = data.get("triplets", [])
    print(f"Loaded {len(triplets)} triplets")

    out_edges = build_graph(triplets)
    print(f"Entities with outgoing edges: {len(out_edges)}")

    entity_rel_types = defaultdict(set)
    for t in triplets:
        frm = t.get("from", "").strip()
        rel = t.get("relation", "").strip()
        if frm and rel:
            entity_rel_types[frm].add(rel)

    multi_rel_entities = [e for e, rels in entity_rel_types.items() if len(rels) >= 2]
    random.shuffle(multi_rel_entities)
    print(f"Entities with 2+ rel types: {len(multi_rel_entities)}")

    all_entities = list(out_edges.keys())
    random.shuffle(all_entities)
    print(f"All entities with edges: {len(all_entities)}")

    dialogues = []

    # Type A: Chain dialogues
    n_chain = int(max_dialogues * chain_proportion)
    for entity in all_entities:
        if len(dialogues) >= n_chain:
            break
        chains = find_chains(entity, out_edges, max_depth=3, max_chains=3)
        for path, rels in chains:
            if len(path) < 2 or len(dialogues) >= n_chain:
                break
            turns = []
            for i in range(len(path) - 1):
                src, tgt = path[i], path[i + 1]
                rel = rels[i]
                if i == 0:
                    q = random.choice(QUESTION_TEMPLATES.get(rel, ["What is {e}?"])).format(e=src)
                else:
                    q = random.choice([
                        f"And what is {src}?",
                        f"What about {src}?",
                        f"Tell me about {src}.",
                    ])
                turns.append({"user": q, "topic": src, "expected": tgt, "relation": rel})
            if len(turns) >= 2:
                dialogues.append({"turns": turns, "type": "chain"})

    # Type B: Mixed-relation dialogues
    for entity in multi_rel_entities:
        if len(dialogues) >= max_dialogues:
            break
        rels = list(entity_rel_types[entity])
        random.shuffle(rels)
        selected_rels = rels[:turns_per_dialogue]
        if len(selected_rels) < 2:
            continue
        turns = []
        for rel in selected_rels:
            targets = [t for t, r in out_edges[entity] if r == rel]
            if not targets:
                continue
            target = random.choice(targets)
            q = random.choice(QUESTION_TEMPLATES.get(rel, ["Tell me about {e}."])).format(e=entity)
            turns.append({"user": q, "topic": entity, "expected": target, "relation": rel})
        if len(turns) >= 2:
            dialogues.append({"turns": turns, "type": "mixed"})

    # Type C: Single-turn dialogues (bulk)
    for entity in all_entities:
        if len(dialogues) >= max_dialogues:
            break
        for tgt, rel in out_edges[entity]:
            q = random.choice(QUESTION_TEMPLATES.get(rel, ["Tell me about {e}?"])).format(e=entity)
            turns = [{"user": q, "topic": entity, "expected": tgt, "relation": rel}]
            dialogues.append({"turns": turns, "type": "single"})
            if len(dialogues) >= max_dialogues:
                break

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for d in dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    types = defaultdict(int)
    for d in dialogues:
        types[d.get("type", "unknown")] += 1
    print(f"\nGenerated {len(dialogues)} dialogues: {dict(types)}")
    total_turns = sum(len(d["turns"]) for d in dialogues)
    print(f"Total turns: {total_turns}")

    if dialogues:
        sample = [d for d in dialogues if d.get("type") != "single"]
        if sample:
            print(f"\nSample ({sample[0]['type']}):")
            for turn in sample[0]["turns"]:
                resp = RESPONSE_TEMPLATES.get(turn["relation"], "{E} -> {t}")
                text = resp.format(E=turn["topic"].capitalize(), e=turn["topic"], t=turn["expected"])
                print(f"  User: {turn['user']}")
                print(f"  Bot:  {text}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dialogues from triplets")
    parser.add_argument("--layer1", default="data/processed/layer1_triplets_curated.json",
                        help="Layer 1 triplets JSON (curated)")
    parser.add_argument("--output", "-o", default="data/processed/layer4_dialogues.jsonl",
                        help="Output dialogues JSONL")
    parser.add_argument("--max", type=int, default=5000,
                        help="Max dialogues (default: 5000)")
    parser.add_argument("--turns", type=int, default=3,
                        help="Turns per mixed dialogue (default: 3)")
    parser.add_argument("--chain-prop", type=float, default=0.3,
                        help="Proportion chain dialogues (default: 0.3)")
    args = parser.parse_args()

    print("=== Generate Dialogues v2 (from triplets) ===\n")
    generate_dialogues(args.layer1, args.output,
                       max_dialogues=args.max,
                       turns_per_dialogue=args.turns,
                       chain_proportion=args.chain_prop)
    print("\nDone.")
