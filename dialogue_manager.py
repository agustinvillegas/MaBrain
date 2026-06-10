"""
Dialogue Manager — intent classification, strategy selection, response composition.

Integrates with WorkingMemory to produce context-aware grammatical responses.
"""
import re


# ── Intent patterns (priority-ordered) ───────────────────────────
# (intent_name, regex_pattern, extract_entity_group)
INTENT_PATTERNS = [
    ("GREETING",   re.compile(r"^(hi|hello|hey|hola|good\s+(morning|afternoon|evening))\b", re.I), None),
    ("PROPERTY",   re.compile(r"what\s+is\s+(a|an|the)\s+(?P<entity>\w+)\s+like", re.I), "entity"),
    ("ANALOGY",    re.compile(r"how\s+is\s+(a|an|the)\s+(?P<entity>\w+)\s+similar\s+to\s+(?P<other>\w+)", re.I), "entity"),
    ("ANALOGY",    re.compile(r"what.?\s+is\s+(a|an|the)\s+(?P<entity>\w+)\s+similar\s+to\s+(?P<other>\w+)", re.I), "entity"),
    ("ANALOGY",    re.compile(r"how\s+is\s+(a|an|the)\s+(?P<entity>\w+)\s+like\s+(a|an|the)\s+(?P<other>\w+)", re.I), "entity"),
    ("FUNCTION",   re.compile(r"what\s+does\s+(a|an|the)\s+(?P<entity>\w+)\s+do", re.I), "entity"),
    ("FUNCTION",   re.compile(r"how\s+does\s+(a|an|the)\s+(?P<entity>\w+)\s+work", re.I), "entity"),
    ("FUNCTION",   re.compile(r"what\s+is\s+(a|an|the)\s+(?P<entity>\w+)\s+used\s+for", re.I), "entity"),
    ("DEFINITION", re.compile(r"what\s+(is|are|was|were)\s+(a|an|the)\s+(?P<entity>\w+)", re.I), "entity"),
    ("DEFINITION", re.compile(r"tell\s+me\s+about\s+(a|an|the)\s+(?P<entity>\w+)", re.I), "entity"),
    ("DEFINITION", re.compile(r"what.?\s+is\s+(a|an|the)\s+(?P<entity>\w+)", re.I), "entity"),
    ("PROPERTY",   re.compile(r"is\s+(a|an|the)\s+(?P<entity>\w+)\s+(?P<property>\w+)", re.I), "entity"),
    ("LOCATION",   re.compile(r"where\s+(is|are|do)\s+(a|an|the)\s+(?P<entity>\w+)", re.I), "entity"),
    ("CAUSE",      re.compile(r"what\s+causes\s+((a|an|the)\s+)?(?P<entity>\w+)", re.I), "entity"),
    ("CAUSE",      re.compile(r"why\s+does\s+((a|an|the)\s+)?(?P<entity>\w+)", re.I), "entity"),
    ("FOLLOWUP",   re.compile(r"^what\s+about\b", re.I), None),
    ("FOLLOWUP",   re.compile(r"^(tell\s+me\s+more|what\s+else|and\b)", re.I), None),
    ("FOLLOWUP",   re.compile(r"^(yes|ok|sure|yeah|yep|okay)$", re.I), None),
]


# ── Strategy map ─────────────────────────────────────────────────
STRATEGY_MAP = {
    "DEFINITION": "EXPLAIN_IS_A",
    "FUNCTION":   "EXPLAIN_FUNCTION",
    "PROPERTY":   "EXPLAIN_PROPERTY",
    "LOCATION":   "EXPLAIN_LOCATION",
    "CAUSE":      "EXPLAIN_CAUSE",
    "ANALOGY":    "ANALOGY_SEARCH",
    "FOLLOWUP":   "CONTINUE_TOPIC",
    "GREETING":   "GREET",
}

# Per-strategy configuration
#   temp: temperature for graph walk (None = dynamic)
#   ew: embedding_weight to use during this response
#   steps: max steps for graph walk
#   relations: filter synapses by these relations (None = all)
#   use_analogy: use analogy method instead of walk
#   template: use a hardcoded template
STRATEGY_CONFIG = {
    "EXPLAIN_IS_A":     {"temp": 0.1, "ew": 0.0, "steps": 6, "relations": {"IS_A"}, "use_analogy": False, "template": None},
    "EXPLAIN_FUNCTION": {"temp": 0.2, "ew": 0.0, "steps": 6, "relations": {"FUNCTION", "USED_FOR"}, "use_analogy": False, "template": None},
    "EXPLAIN_PROPERTY": {"temp": 0.1, "ew": 0.0, "steps": 4, "relations": {"HAS_PROPERTY"}, "use_analogy": False, "template": None},
    "EXPLAIN_LOCATION": {"temp": 0.1, "ew": 0.0, "steps": 4, "relations": {"LOCATED_IN"}, "use_analogy": False, "template": None},
    "EXPLAIN_CAUSE":    {"temp": 0.2, "ew": 0.0, "steps": 5, "relations": {"CAUSE"}, "use_analogy": False, "template": None},
    "ANALOGY_SEARCH":   {"temp": 0.0, "ew": 0.0, "steps": 0, "relations": None, "use_analogy": True, "template": None},
    "CONTINUE_TOPIC":   {"temp": 0.3, "ew": 0.1, "steps": 8, "relations": None, "use_analogy": False, "template": None},
    "GREET":            {"temp": 0.0, "ew": 0.0, "steps": 0, "relations": None, "use_analogy": False, "template": "hello"},
    "EXPLORE":          {"temp": 0.5, "ew": 0.1, "steps": 8, "relations": None, "use_analogy": False, "template": None},
}

# Greeting templates
GREETINGS = [
    "Hello! I'm MaBrain. Ask me about things, and I'll explore my knowledge graph.",
    "Hi there! I can explain concepts, their functions, properties, and relationships.",
    "Hey! I'm ready to think. What do you want to know?",
]


class IntentClassifier:
    """Classify user intent from text + working memory context."""

    def classify(self, text, working_memory=None):
        """Return (intent, extracted_entity) tuple."""
        clean = text.strip()
        for intent, pattern, entity_group in INTENT_PATTERNS:
            m = pattern.search(clean)
            if m:
                entity = m.group(entity_group) if entity_group else None
                if entity:
                    return intent, entity
                # For FOLLOWUP without entity, use WM's current topic
                if intent == "FOLLOWUP" and working_memory:
                    topic = working_memory.current_topic()
                    if topic:
                        return intent, topic
                return intent, None
        return "UNKNOWN", None


class StrategySelector:
    """Select response strategy based on intent and WM state."""

    def select(self, intent, working_memory=None):
        """Return (strategy_name, config_dict)."""
        strategy = STRATEGY_MAP.get(intent, "EXPLORE")
        config = dict(STRATEGY_CONFIG.get(strategy, STRATEGY_CONFIG["EXPLORE"]))

        # Special handling for CONTINUE_TOPIC: use active entities
        if strategy == "CONTINUE_TOPIC" and working_memory:
            active = working_memory.get_active_entities(min_salience=0.1)
            if active:
                config["_fallback_entity"] = active[0]

        return strategy, config


class ResponseComposer:
    """Assemble the final response using generation + strategy config."""

    def compose(self, brain, start_word, strategy, config):
        """Generate a response using the strategy config."""
        from generation import strategy_generate

        if strategy == "GREET":
            import random
            return random.choice(GREETINGS)

        if not start_word:
            fb = config.get("_fallback_entity")
            if fb:
                start_word = fb
            else:
                if brain.cells:
                    import random
                    for _ in range(50):
                        cell = random.choice(list(brain.cells.values()))
                        if cell.word and cell.synapses_out:
                            start_word = cell.word
                            break
                if not start_word:
                    return "I don't know what to say."

        # Fuzzy resolve
        if start_word and start_word not in brain.concept_registry:
            fuzzy = brain.get_or_create_cell(start_word, fuzzy=True)
            if fuzzy and fuzzy.word:
                start_word = fuzzy.word
            else:
                return "I don't know what '%s' is." % start_word

        # For ANALOGY_SEARCH, use the analogy methods
        if strategy == "ANALOGY_SEARCH" and start_word:
            cell_a = brain.get_or_create_cell(start_word, fuzzy=True)
            if cell_a and cell_a.synapses_out:
                # Pick the strongest IS_A target as B
                is_a_targets = [(s.target.word, s.strength) for s in cell_a.synapses_out
                                if s.relation == "IS_A" and s.target and s.target.word]
                if is_a_targets:
                    b = max(is_a_targets, key=lambda x: x[1])[0]
                    # Use analogy_structural for cross-domain
                    results = brain.analogy_structural(start_word, b, start_word, top_k=3)
                    if results:
                        candidates = [r["d"] for r in results if r["d"] != b]
                        if candidates:
                            return "%s is a %s, like %s." % (start_word.capitalize(), b, candidates[0])
                # Fallback: use standard analogy
                results = brain.analogy(start_word, start_word, start_word, top_k=3)
                if results:
                    similar = [r["d"] for r in results if r["d"] != start_word]
                    if similar:
                        return "%s is similar to %s." % (start_word.capitalize(), similar[0])

        return strategy_generate(brain, start_word, config)


class DialogueManager:
    """Top-level dialogue manager. Integrates IntentClassifier, StrategySelector,
    and ResponseComposer with Brain.get_response()."""

    def __init__(self, brain):
        self.brain = brain
        self.classifier = IntentClassifier()
        self.selector = StrategySelector()
        self.composer = ResponseComposer()

    def respond(self, user_text):
        """Full pipeline: classify -> select -> compose.
        Returns response string."""
        wm = self.brain.wm

        # 1. Extract entities (already done by brain._extract_entities)
        #    but we need quick entity detection for intent patterns too
        entities = self.brain._extract_entities(user_text)

        # 2. Classify intent
        intent, intent_entity = self.classifier.classify(user_text, wm)

        # 3. Determine start word: intent entity > first extracted entity > WM active
        start_word = intent_entity
        if not start_word and entities:
            start_word = entities[0]
        if not start_word:
            active = wm.get_active_entities(min_salience=0.1)
            if active:
                start_word = active[0]

        # 4. Select strategy
        strategy, config = self.selector.select(intent, wm)

        # 5. Compose response
        response = self.composer.compose(self.brain, start_word, strategy, config)

        # 6. Update topic stack if we have a clear entity
        if start_word:
            wm.push_topic(start_word)

        return response
