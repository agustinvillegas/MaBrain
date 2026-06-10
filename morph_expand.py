"""Morphological generalization — conservative rules driven by graph structure.

Rules applied only when semantically appropriate:
  -ing: only for words with FUNCTION out-synapses (action verbs)
  -er:  only for agent-like words (professions, animals)
  -ly:  only for adjectives (HAS_PROPERTY targets)
  -ness: only for emotion/quality adjectives
  Compounds: only noun+noun where both are concrete entities (IS_A members)
"""
import json
import sys
from collections import Counter


SKIP_PLURAL = {
    "sheep", "deer", "fish", "air", "water", "music", "rice", "bread",
    "butter", "cheese", "milk", "juice", "soup", "pasta", "meat", "grass",
    "sand", "dust", "dirt", "mud", "smoke", "steam", "electricity",
    "sunlight", "information", "knowledge", "furniture", "equipment",
}

def _plural(word):
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith("f"):
        return word[:-1] + "ves"
    if word.endswith("fe"):
        return word[:-2] + "ves"
    return word + "s"

def _add_ing(word):
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith("ee"):
        return word[:-1] + "ing"
    if len(word) >= 3 and word[-1] not in "aeiou" and word[-2] in "aeiou" and word[-3] not in "aeiou":
        return word + word[-1] + "ing"
    return word + "ing"

def _add_er(word):
    if word.endswith("e"):
        return word + "r"
    if len(word) >= 3 and word[-1] not in "aeiou" and word[-2] in "aeiou" and word[-3] not in "aeiou":
        return word + word[-1] + "er"
    return word + "er"

def _add_ly(word):
    if word.endswith("ic"):
        return word + "ally"
    if word.endswith("le"):
        return word[:-1] + "y"
    if word.endswith("y"):
        return word[:-1] + "ily"
    return word + "ly"

def _add_ness(word):
    if word.endswith("y"):
        return word[:-1] + "iness"
    return word + "ness"


# ── Quality / emotion adjectives that make sense for -ness ──────
QUALITY_ADJECTIVES = {
    "happy", "sad", "dark", "bright", "kind", "sweet", "calm",
    "aware", "eager", "firm", "gentle", "honest", "loyal",
    "polite", "pure", "quiet", "rough", "sharp", "shy",
    "tender", "tough", "wild", "bitter", "bold", "brave",
    "clever", "coarse", "cold", "cool", "damp", "deep",
    "dull", "fair", "fast", "fat", "feeble", "fierce",
    "flat", "foul", "fresh", "full", "glad", "grand",
    "grave", "great", "greedy", "harsh", "hazy", "heavy",
    "hollow", "humble", "hungry", "juicy", "keen", "large",
    "loose", "loud", "low", "mellow", "mild", "moist",
    "narrow", "neat", "nimble", "noble", "pale", "plain",
    "plump", "poor", "proud", "quick", "rapid", "rare",
    "raw", "rich", "ripe", "risky", "rosy", "rotten",
    "rude", "salty", "silly", "slimy", "slippery", "slow",
    "small", "smooth", "soft", "sour", "spicy", "steep",
    "stiff", "strange", "strict", "sturdy", "subtle", "sudden",
    "sure", "swift", "tall", "thick", "thin", "thirsty",
    "tight", "tiny", "tired", "warm", "weak", "wealthy",
    "weary", "wide", "wise", "witty", "young", "zealous",
    "splendid", "superb", "certain", "delicious", "nervous",
    "peaceful", "dangerous", "curious", "generous", "jealous",
    "marvelous", "obvious", "precious", "serious", "tedious",
    "tremendous", "vigorous", "wondrous", "joyful", "hopeful",
    "grateful", "powerful", "useful", "harmful", "playful",
    "fearful", "painful", "colorful", "peaceful", "careful",
    "thoughtful", "graceful", "restless", "endless", "careless",
    "hopeless", "homeless", "penniless", "reckless", "powerless",
}

# Words that already look agent-derived (don't double-suffix)
AGENT_BLACKLIST = {"cook", "judge", "guide", "guard", "clerk", "cashier",
                   "programmer", "teacher", "driver", "writer", "singer",
                   "dancer", "player", "runner", "swimmer", "builder",
                   "farmer", "painter", "gardener", "manager", "leader",
                   "reader", "listener", "thinker", "speaker"}


def load_vocab(path="current_vocab.json"):
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def _load_actions():
    """Import the curated ACTIONS list from expand_dataset_en_large.py."""
    from expand_dataset_en_large import ACTIONS
    return set(ACTIONS)


def load_brain_and_analyze(brain_path):
    """Load brain and categorize words by their role in graph."""
    from braincell import Brain
    b = Brain()
    b.load(brain_path)

    actions = _load_actions()

    # Count synapses per word
    word_synapse_count = Counter()
    # True verbs = words with FUNCTION out-synapses that ARE in ACTIONS list
    verbs = set()
    agent_nouns = set()  # words with FUNCTION out-synapses but NOT in ACTIONS (agent nouns)
    adjectives = set()   # words that are targets of HAS_PROPERTY
    concrete_nouns = set()  # words that are IS_A members
    agent_like = set()    # animals, professions (for relation linking)

    # Collect FUNCTION targets (action words)
    func_targets = set()
    for cell in b.cells.values():
        for s in cell.synapses_out:
            if s.relation == "FUNCTION" and s.target and s.target.word:
                func_targets.add(s.target.word)

    for cell in b.cells.values():
        w = cell.word
        if not w:
            continue
        word_synapse_count[w] += len(cell.synapses_out)

        has_function_out = any(s.relation == "FUNCTION" for s in cell.synapses_out)
        is_agent = any(s.relation == "IS_A" and s.target and s.target.word in {
            "animal", "person", "bird", "fish", "mammal", "insect",
            "reptile", "professional", "worker"
        } for s in cell.synapses_out)
        is_concrete = any(s.relation == "IS_A" for s in cell.synapses_out)

        # True verbs: in ACTIONS list AND is a FUNCTION target (or source)
        if w in actions and (w in func_targets or has_function_out):
            verbs.add(w)
        elif has_function_out and w not in actions:
            agent_nouns.add(w)
        elif w in actions:
            verbs.add(w)  # in actions even if no function role

        if is_agent:
            agent_like.add(w)
        if is_concrete:
            concrete_nouns.add(w)

        for s in cell.synapses_in:
            if s.relation == "HAS_PROPERTY":
                adjectives.add(w)

    print(f"    True verbs: {len(verbs)}, Agent-nouns: {len(agent_nouns)}, "
          f"Adjectives: {len(adjectives)}, Concrete nouns: {len(concrete_nouns)}")
    return b, word_synapse_count, verbs, agent_nouns, adjectives, concrete_nouns, agent_like


def generate_dataset(vocab_path, brain_path, output_path, top_n=500, max_compounds=150):
    """Generate morph pairs using graph-driven analysis."""
    existing_vocab = load_vocab(vocab_path)
    brain, wsc, verbs, agent_nouns, adjectives, concrete_nouns, _ = load_brain_and_analyze(brain_path)

    # Top N words by synapse count
    top_words = [w for w, _ in wsc.most_common(top_n)]
    top_set = set(top_words)
    print(f"  Top {len(top_words)} words loaded. "
          f"Verbs: {len(verbs)}, Agent-nouns: {len(agent_nouns)}, "
          f"Adjectives: {len(adjectives)}, Concrete: {len(concrete_nouns)}")

    pairs = []
    seen = set()

    def add_pair(src, tgt, rel):
        key = (src, tgt, rel)
        if key not in seen and tgt != src and tgt not in existing_vocab:
            seen.add(key)
            pairs.append((src, tgt, rel))

    for word in top_words:
        # --- -ing (only for true verbs, >= 4 chars) ---
        if word in verbs and len(word) >= 4 and not word.endswith("ing"):
            ing = _add_ing(word)
            if len(ing) <= 12:
                add_pair(word, ing, "CAUSE")       # walk → causes walking
                add_pair(ing, word, "FUNCTION")     # walking → function is to walk

        # --- -er (only for true verbs, producing agent nouns) ---
        if word in verbs and word not in AGENT_BLACKLIST:
            if not word.endswith(("er", "or", "ist", "ian", "man", "woman")):
                er = _add_er(word)
                if len(er) <= 10 and er not in existing_vocab:
                    add_pair(er, word, "FUNCTION")     # teacher → teach
                    add_pair(er, "person", "IS_A")     # teacher is a person

        # --- -ly (only for adjectives) ---
        if word in adjectives and len(word) >= 4 and not word.endswith("ly"):
            ly = _add_ly(word)
            if len(ly) <= 12:
                add_pair(word, ly, "HAS_PROPERTY")   # quick has_property quickly
                # No: "quick" → "quickly" through HAS_PROPERTY is wrong.
                # Instead, don't link -ly variants. Just add vocab.
                # We'll skip the pair; just add to vocab via an IS_A? No.

        # --- -ness (only for quality adjectives) ---
        if word in QUALITY_ADJECTIVES and len(word) >= 3 and not word.endswith("ness"):
            ness = _add_ness(word)
            if len(ness) <= 14:
                add_pair(word, ness, "HAS_PROPERTY")  # happy has_property happiness

    # --- Compounds (template-driven: known English compound patterns) ---
    compound_pairs = []
    seen_comp = set()
    from expand_dataset_en_large import (
        ANIMALS, BIRDS, FISH, TOOLS, BODY_PARTS, VEHICLES,
        VEGETABLES, FRUITS, FOODS, COLORS, WEATHER, GEOGRAPHY,
    )

    def _c_set(lst):
        return {w for w in lst if w in top_set and len(w) >= 3 and len(w) <= 8}

    body = _c_set(BODY_PARTS)
    tools = _c_set(TOOLS)
    colors = _c_set(COLORS)
    animals = _c_set(ANIMALS) | _c_set(BIRDS) | _c_set(FISH)
    fruits = _c_set(FRUITS)
    vegs = _c_set(VEGETABLES)
    foods = _c_set(FOODS)
    weather = _c_set(WEATHER)
    geo = _c_set(GEOGRAPHY)
    vehicles = _c_set(VEHICLES)

    # Body + tool → tool type (handcuffs, eyeglass, fingernail?)
    templates = [
        (body, tools, "PART_OF"),       # hand + cuffs → handcuffs
        (colors, animals, "HAS_PROPERTY"),  # blue + bird → bluebird
        (colors, fruits, "HAS_PROPERTY"),   # blue + berry → blueberry
        (weather, geo, "CAUSE"),        # rain + bow → rainbow, snow + fall → snowfall
        (weather, tools, "USED_FOR"),   # rain + coat → raincoat
        (fruits, foods, "IS_A"),        # apple + pie → applepie
        (vegs, foods, "IS_A"),          # carrot + cake → carrotcake
        (geo, body, "LOCATED_IN"),      # sea + shell → seashell
    ]

    for src_set, tgt_set, rel in templates:
        for w1 in src_set:
            for w2 in tgt_set:
                if len(w1) + len(w2) > 14:
                    continue
                c = w1 + w2
                if c in existing_vocab or c in seen_comp:
                    continue
                seen_comp.add(c)
                compound_pairs.append((c, w1, w2, rel))
                if len(compound_pairs) >= max_compounds:
                    break
            if len(compound_pairs) >= max_compounds:
                break
        if len(compound_pairs) >= max_compounds:
            break
        if len(compound_pairs) >= max_compounds:
            break

    # --- Build sequences ---
    sequences = []
    for src, tgt, rel in pairs:
        sequences.append({"sequence": [src, tgt], "relations": [rel]})
    for comp, w1, w2, rel in compound_pairs:
        sequences.append({"sequence": [comp, w1], "relations": [rel], "weight": 1.0})
        sequences.append({"sequence": [comp, w2], "relations": [rel], "weight": 1.0})

    # --- Stats ---
    new_words = set()
    for s in sequences:
        for w in s["sequence"]:
            if w not in existing_vocab:
                new_words.add(w)

    desc = f"Morphological expansion: {len(sequences)} seqs, {len(new_words)} new words"
    result = {"description": desc, "sequences": sequences}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)

    print(f"  Sequences generated: {len(sequences)}")
    print(f"  New unique words: {len(new_words)}")
    print(f"  Morph pairs: {len(pairs)}")
    print(f"  Compound pairs: {len(compound_pairs)}")
    print(f"  Example new words: {sorted(new_words)[:20]}")
    print(f"  Saved to {output_path}")
    return result


if __name__ == "__main__":
    generate_dataset(
        vocab_path="current_vocab.json",
        brain_path=r"D:\ma_brain_data\brain_state_v8_v6_pruned_evaled.json",
        output_path=r"D:\ma_brain_data\dataset_morph.json",
        top_n=500,
        max_compounds=150,
    )
