"""
Generation module: graph traversal -> surface text.
Transforms raw graph paths into grammatical sentences.
"""
import re


# Relation -> (verb_phrase, obj_prefix, verb_from_obj, obj_suffix)
#   verb_phrase: literal verb string to use (e.g. "is", "is used for")
#   obj_prefix: article or preposition before obj (e.g. "a", "in")
#   verb_from_obj: if True, verb = obj + "s" instead of verb_phrase
#   obj_suffix: suffix to add to obj (e.g. "ing" for USED_FOR)
RELATION_FRAMES = {
    "IS_A":        ("is",      "a", False, ""),
    "HAS_PROPERTY": ("is",     "",  False, ""),
    "FUNCTION":    ("",        "",  True,  ""),
    "CAUSE":       ("causes",  "",  False, ""),
    "USED_FOR":    ("is used for", "", False, "ing"),
    "INSTRUMENT":  ("uses",    "",  False, ""),
    "PART_OF":     ("is part of", "a", False, ""),
    "LOCATED_IN":  ("is in",   "",  False, ""),
    "OPPOSITE":    ("is opposite of", "", False, ""),
    "NEXT":        ("",        "",  False, ""),
}

# Words that are already adjectives/properties (don't add article)
ADJECTIVES = {
    "hot", "cold", "big", "small", "fast", "slow", "hard", "soft",
    "heavy", "light", "sweet", "sour", "bitter", "salty", "spicy",
    "smooth", "rough", "wet", "dry", "clean", "dirty", "old", "new",
    "good", "bad", "happy", "sad", "angry", "calm", "brave", "shy",
    "smart", "strong", "weak", "tall", "short", "long", "wide",
    "thick", "thin", "deep", "shallow", "sharp", "dull", "bright",
    "dark", "loud", "quiet", "rich", "poor", "young", "beautiful",
    "ugly", "kind", "cruel", "honest", "loyal", "wise", "useful",
    "valuable", "precious", "dangerous", "peaceful", "gentle",
    "stubborn", "curious", "innocent", "creative", "majestic",
    "uplifting", "certain", "delicious", "fragile", "cozy",
    "elegant", "fierce", "playful", "powerful", "durable",
    "conductive", "explosive", "rocky", "paralyzing", "soft",
    "warm", "blue", "green", "red", "yellow", "white", "black",
    "pink", "orange", "purple", "brown", "grey", "gray",
}


def _article(word):
    """Choose a/an based on word starting sound."""
    if not word:
        return ""
    word = word.lstrip()
    if word[0] in "aeiou":
        return "an"
    return "a"


def _add_ing(word):
    """Add -ing with English spelling rules."""
    if word.endswith("ie"):
        return word[:-2] + "ying"
    if word.endswith("e") and not word.endswith("ee"):
        return word[:-1] + "ing"
    # Double final consonant if CVC pattern
    if len(word) >= 3 and word[-1] not in "aeiou" and word[-2] in "aeiou" and word[-3] not in "aeiou":
        return word + word[-1] + "ing"
    return word + "ing"


def _verb_form(word):
    """Convert a noun/concept to a verb-ish form.
    E.g., 'pump' -> 'pumps', 'teach' -> 'teaches', 'fly' -> 'flies'."""
    IRREGULAR = {"life": "gives life", "death": "causes death", "love": "loves",
                 "hate": "hates", "anger": "angers", "sadness": "saddens",
                 "fear": "frightens", "joy": "brings joy"}
    if word in IRREGULAR:
        return IRREGULAR[word]
    if word.endswith("fe"):
        return word[:-2] + "ves"
    if word.endswith("s") or word.endswith("sh") or word.endswith("ch") or word.endswith("x"):
        return word + "es"
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _needs_article(word):
    """Check if a word needs an article (not adjective, not plural-looking)."""
    if word in ADJECTIVES:
        return False
    if word.endswith("s") and word not in ADJECTIVES:
        # Could be plural but many nouns end in s
        return True
    return True


def build_traced_thought(brain, start_cell, steps=10, temperature=None, epsilon=0.0,
                         relation_filter=None, select_config=None):
    """
    Like think() but returns list of (word, relation) tuples.
    If relation_filter is a set/list, only follow synapses with those relations.
    select_config: optional dict passed to _select_synapse (min_strength, min_usage, char_top_k)
    """
    current = start_cell
    result = []
    for _ in range(steps):
        if not current.synapses_out:
            break
        options = current.synapses_out
        if relation_filter:
            options = [s for s in options if s.relation in relation_filter]
            if not options:
                break
        choice = brain._select_synapse(options, temperature=temperature, epsilon=epsilon,
                                       context=current.word, config=select_config)
        result.append({
            "word": choice.concept,
            "relation": choice.relation,
            "target_word": choice.target.word if choice.target else None,
        })
        choice.inference_usage += 1
        choice.target.activation += choice.strength
        current = choice.target
    return result


def linearize_path(path, start_word):
    """
    Convert a traced path into a grammatical sentence.
    """
    if not path:
        return start_word or ""

    current_subj = start_word
    clauses = []

    for step in path:
        rel = step.get("relation")
        obj = step.get("word") or step.get("target_word", "")
        if not obj:
            continue

        frame = RELATION_FRAMES.get(rel)
        if frame is None:
            clauses.append(f"{current_subj} -> {obj}")
            current_subj = obj
            continue

        verb_phrase, obj_prefix, verb_from_obj, obj_suffix = frame

        # Determine verb and object
        if verb_from_obj:
            # Verb IS the object's action form; don't repeat object separately
            verb = _verb_form(obj)
            obj_text = ""
        else:
            verb = verb_phrase
            obj_text = obj
            if obj_suffix == "ing":
                obj_text = _add_ing(obj)
            elif obj_suffix and not obj.endswith(obj_suffix):
                obj_text = obj + obj_suffix
            if obj_prefix == "a" and _needs_article(obj):
                obj_text = f"{_article(obj)} {obj_text}"

        if rel == "NEXT":
            clause = obj_text or verb or obj
            if clause:
                clauses.append(clause)
            current_subj = obj
            continue

        # Build clause
        if verb and obj_text:
            clause = f"{current_subj} {verb} {obj_text}"
        elif verb:
            clause = f"{current_subj} {verb}"
        else:
            clause = f"{current_subj} {obj_text}"

        clauses.append(clause)
        current_subj = obj

    # Join clauses
    if len(clauses) == 1:
        sentence = clauses[0]
    elif len(clauses) == 2:
        sentence = f"{clauses[0]} and {clauses[1]}"
    else:
        sentence = ", ".join(clauses[:-1]) + f", and {clauses[-1]}"

    # Normalize: replace underscores with spaces
    sentence = sentence.replace("_", " ")

    # Capitalize, add period
    sentence = sentence[0].upper() + sentence[1:] if sentence else ""
    if sentence and not sentence.endswith((".", "!", "?")):
        sentence += "."

    return sentence


def generate(brain, start_word, steps=8, temperature=None, epsilon=0.0):
    """High-level: take a start word, return a grammatical sentence."""
    start_cell = brain.get_or_create_cell(start_word, fuzzy=True)
    if not start_cell or not start_cell.word:
        return start_word or ""

    path = build_traced_thought(brain, start_cell, steps=steps, temperature=temperature, epsilon=epsilon)
    return linearize_path(path, start_word)


def strategy_generate(brain, start_word, strategy_config):
    """Generate a response following a specific strategy.

    strategy_config: dict with keys:
        - temp: temperature
        - ew: embedding_weight (set on brain)
        - steps: max steps
        - relations: set of relations to follow (or None for all)
        - use_analogy: if True, use analogy instead of walk
        - ctx_bias: context_bias_weight (set on brain, optional)
        - min_strength: filter synapses below this strength (optional)
        - min_usage: filter synapses below this usage count (optional)
        - char_top_k: keep top-k by char ngram sim before MiniLM (optional)
    """
    ew = strategy_config.get("ew", 0.0)
    temp = strategy_config.get("temp", None)
    steps = strategy_config.get("steps", 6)
    relations = strategy_config.get("relations", None)
    use_analogy = strategy_config.get("use_analogy", False)
    ctx_bias = strategy_config.get("ctx_bias", None)
    select_config = {
        k: strategy_config[k]
        for k in ("min_strength", "min_usage", "char_top_k")
        if k in strategy_config
    } or None

    # Set params temporarily
    old_ew = brain.embedding_weight
    brain.embedding_weight = ew
    old_ctx = brain.context_bias_weight
    if ctx_bias is not None:
        brain.context_bias_weight = ctx_bias

    try:
        if use_analogy and start_word:
            # Walk a short path then use analogy
            start_cell = brain.get_or_create_cell(start_word, fuzzy=True)
            if start_cell and start_cell.synapses_out:
                path = build_traced_thought(brain, start_cell, steps=2,
                                            temperature=0.1, relation_filter={"IS_A"},
                                            select_config=select_config)
                if path:
                    analogies = brain.analogy(start_word, path[-1]["word"],
                                              start_word, top_k=3)
                    if analogies:
                        result = "Analogy: " + start_word + " is to " + path[-1]["word"] + " as " + start_word + " is to " + analogies[0]["d"]
                        return result

        # Standard walk
        path = build_traced_thought(brain,
                                    brain.get_or_create_cell(start_word, fuzzy=True),
                                    steps=steps, temperature=temp,
                                    relation_filter=relations,
                                    select_config=select_config)
        return linearize_path(path, start_word)
    finally:
        brain.embedding_weight = old_ew
        if ctx_bias is not None:
            brain.context_bias_weight = old_ctx
