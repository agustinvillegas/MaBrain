"""English dataset v7 generator — structural relations + deep hierarchies."""

import json
import itertools

# ── Relation constants ──────────────────────────────────────────
IS_A = "IS_A"
NEXT = "NEXT"
HAS = "HAS"
HAS_PROPERTY = "HAS_PROPERTY"
FUNCTION = "FUNCTION"
CAUSE = "CAUSE"
PART_OF = "PART_OF"
INSTRUMENT = "INSTRUMENT"
LOCATED_IN = "LOCATED_IN"
OPPOSITE = "OPPOSITE"
SIMILAR_TO = "SIMILAR_TO"
MADE_OF = "MADE_OF"
SYMBOL_OF = "SYMBOL_OF"
DESIRES = "DESIRES"
AVOIDS = "AVOIDS"

# Inverses
HAS_INV = "HAS_INVERSE"
PROPERTY_OF = "PROPERTY_OF"
FUNCTION_OF = "FUNCTION_OF"
EFFECT_OF = "EFFECT_OF"
CONTAINS = "CONTAINS"
USED_FOR = "USED_FOR"
CONTAINED_IN = "CONTAINED_IN"
SYMBOLIZED_BY = "SYMBOLIZED_BY"

INVERSE_MAP = {
    IS_A: HAS_INV,
    NEXT: HAS_INV,
    HAS: HAS_INV,
    HAS_PROPERTY: PROPERTY_OF,
    FUNCTION: FUNCTION_OF,
    CAUSE: EFFECT_OF,
    PART_OF: CONTAINS,
    INSTRUMENT: USED_FOR,
    LOCATED_IN: CONTAINED_IN,
    SYMBOL_OF: SYMBOLIZED_BY,
    OPPOSITE: OPPOSITE,        # symmetric
    SIMILAR_TO: SIMILAR_TO,    # symmetric
    DESIRES: DESIRES,          # simplified
    AVOIDS: AVOIDS,
}


# ── Helper templates ────────────────────────────────────────────
def seq(words, relations=None):
    return {"sequence": words, "relations": relations} if relations else {"sequence": words}

def chain(head, *tail, rel=NEXT):
    return seq([head] + list(tail), [rel] * len(tail))

def isa(member, *categories):
    return seq([member] + list(categories), [IS_A] * len(categories))

def cause(cause_word, effect_word):
    return seq([cause_word, effect_word], [CAUSE])

def function(agent, action):
    return seq([agent, action], [FUNCTION])

def part_of(part, whole):
    return seq([part, whole], [PART_OF])

def instrument(agent, tool, action):
    return seq([agent, tool, action], [INSTRUMENT, NEXT])

def located_in(entity, location):
    return seq([entity, location], [LOCATED_IN])

def opposite(a, b):
    return seq([a, b], [OPPOSITE])

def similar(a, b):
    return seq([a, b], [SIMILAR_TO])

def has_property(entity, prop):
    return seq([entity, prop], [HAS_PROPERTY])


# ════════════════════════════════════════════════════════════════
# DOMAIN GENERATORS
# ════════════════════════════════════════════════════════════════

# ── ANIMALS ──────────────────────────────────────────────────────
SUBGROUPS = {
    "dog": ["canine", "carnivore"],
    "wolf": ["canine", "carnivore"],
    "fox": ["canine", "carnivore"],
    "cat": ["feline", "carnivore"],
    "lion": ["feline", "carnivore"],
    "tiger": ["feline", "carnivore"],
    "horse": ["equine", "herbivore"],
    "cow": ["bovine", "herbivore"],
    "sheep": ["ovine", "herbivore"],
    "pig": ["suid", "omnivore"],
}

GROUP_ANCESTORS = {
    "mammal": ["vertebrate", "animal", "living_thing"],
    "bird": ["vertebrate", "animal", "living_thing"],
    "fish": ["vertebrate", "animal", "living_thing"],
    "reptile": ["vertebrate", "animal", "living_thing"],
    "insect": ["invertebrate", "animal", "living_thing"],
}

ANIMALS = {
    "mammals": {
        "members": {
            "dog": {"sound": "bark", "action": "run", "body": "paw"},
            "cat": {"sound": "meow", "action": "jump", "body": "claw"},
            "lion": {"sound": "roar", "action": "hunt", "body": "mane"},
            "tiger": {"sound": "growl", "action": "stalk", "body": "stripe"},
            "elephant": {"sound": "trumpet", "action": "carry", "body": "trunk"},
            "horse": {"sound": "neigh", "action": "gallop", "body": "mane"},
            "cow": {"sound": "moo", "action": "graze", "body": "udder"},
            "sheep": {"sound": "baa", "action": "graze", "body": "wool"},
            "pig": {"sound": "grunt", "action": "root", "body": "snout"},
            "monkey": {"sound": "screech", "action": "climb", "body": "tail"},
            "bear": {"sound": "growl", "action": "fish", "body": "fur"},
            "dolphin": {"sound": "click", "action": "jump", "body": "fin"},
            "whale": {"sound": "sing", "action": "dive", "body": "blowhole"},
            "giraffe": {"sound": "snort", "action": "reach", "body": "neck"},
            "zebra": {"sound": "neigh", "action": "run", "body": "stripe"},
            "rabbit": {"sound": "squeak", "action": "hop", "body": "ear"},
            "squirrel": {"sound": "chatter", "action": "gather", "body": "tail"},
            "fox": {"sound": "howl", "action": "hunt", "body": "tail"},
            "wolf": {"sound": "howl", "action": "hunt", "body": "fang"},
            "hamster": {"sound": "squeak", "action": "hoard", "body": "pouch"},
        },
        "base": "mammal",
        "properties": ["domestic", "wild", "herbivore", "carnivore"],
    },
    "birds": {
        "members": {
            "eagle": {"sound": "screech", "action": "soar", "body": "wing"},
            "sparrow": {"sound": "chirp", "action": "fly", "body": "feather"},
            "owl": {"sound": "hoot", "action": "hunt", "body": "feather"},
            "duck": {"sound": "quack", "action": "swim", "body": "beak"},
            "rooster": {"sound": "crow", "action": "call", "body": "crest"},
            "hen": {"sound": "cluck", "action": "peck", "body": "beak"},
            "parrot": {"sound": "talk", "action": "fly", "body": "feather"},
            "penguin": {"sound": "honk", "action": "swim", "body": "flipper"},
            "swan": {"sound": "hiss", "action": "glide", "body": "feather"},
            "pigeon": {"sound": "coo", "action": "walk", "body": "feather"},
        },
        "base": "bird",
        "properties": ["flying", "nonflying", "migratory", "aquatic"],
    },
    "fish": {
        "members": {
            "shark": {"sound": None, "action": "hunt", "body": "fin"},
            "salmon": {"sound": None, "action": "swim", "body": "scale"},
            "tuna": {"sound": None, "action": "swim", "body": "fin"},
            "goldfish": {"sound": None, "action": "swim", "body": "scale"},
            "eel": {"sound": None, "action": "slither", "body": "scale"},
        },
        "base": "fish",
        "properties": ["marine", "freshwater", "predator"],
    },
    "reptiles": {
        "members": {
            "snake": {"sound": "hiss", "action": "slither", "body": "scale"},
            "crocodile": {"sound": "bellow", "action": "stalk", "body": "scale"},
            "lizard": {"sound": "hiss", "action": "climb", "body": "scale"},
            "turtle": {"sound": "hiss", "action": "swim", "body": "shell"},
            "iguana": {"sound": "hiss", "action": "climb", "body": "scale"},
        },
        "base": "reptile",
        "properties": ["venomous", "nonvenomous", "terrestrial", "aquatic"],
    },
    "insects": {
        "members": {
            "ant": {"sound": None, "action": "work", "body": "antenna"},
            "bee": {"sound": "buzz", "action": "pollinate", "body": "wing"},
            "butterfly": {"sound": None, "action": "flutter", "body": "wing"},
            "fly": {"sound": "buzz", "action": "buzz", "body": "wing"},
            "spider": {"sound": None, "action": "spin", "body": "web"},
        },
        "base": "insect",
        "properties": ["flying", "nonflying", "social"],
    },
}

def generate_animal_sequences():
    s = []
    for group_name, group in ANIMALS.items():
        base = group["base"]
        ancestors = GROUP_ANCESTORS.get(base, ["animal", "living_thing"])
        for name, traits in group["members"].items():
            subgroups = SUBGROUPS.get(name, [])
            s.append(isa(name, *subgroups, base, *ancestors))
            if traits.get("sound"):
                s.append(function(name, traits["sound"]))
            if traits.get("action"):
                s.append(function(name, traits["action"]))
            if traits.get("body"):
                s.append(has_property(name, traits["body"]))
                s.append(part_of(traits["body"], name))
        for prop in group["properties"]:
            for name in group["members"]:
                s.append(has_property(name, prop))
            s.append(isa(prop, "attribute"))
    # Cross-group links
    for g1, g2 in itertools.combinations(ANIMALS.values(), 2):
        for m1 in list(g1["members"])[:2]:
            for m2 in list(g2["members"])[:2]:
                s.append(chain(m1, m2, rel=NEXT))
    return s


# ── PLANTS ──────────────────────────────────────────────────────
def generate_plant_sequences():
    s = []
    plants = {
        "trees": {"base": "tree", "ancestors": ["plant", "living_thing"],
                  "members": ["oak", "pine", "willow", "cherry", "apple"]},
        "flowers": {"base": "flower", "ancestors": ["plant", "living_thing"],
                    "members": ["rose", "sunflower", "daisy", "orchid", "tulip"]},
        "fruits": {"base": "fruit", "ancestors": ["produce", "plant", "living_thing"],
                   "members": ["apple", "orange", "banana", "grape", "strawberry"]},
        "vegetables": {"base": "vegetable", "ancestors": ["produce", "plant", "living_thing"],
                       "members": ["carrot", "broccoli", "tomato", "lettuce", "onion"]},
    }
    for cat, group in plants.items():
        base = group["base"]
        ancestors = group["ancestors"]
        for member in group["members"]:
            s.append(isa(member, base, *ancestors))
    return s


# ── BODY ─────────────────────────────────────────────────────────
BODY_PARTS = {
    "head": ["eye", "ear", "nose", "mouth", "tongue", "tooth", "brain",
             "hair", "forehead", "cheek", "chin", "eyebrow", "eyelash"],
    "torso": ["heart", "lung", "liver", "stomach", "kidney", "bone", "skin", "muscle", "blood"],
    "limbs": ["hand", "arm", "elbow", "shoulder", "finger", "leg", "knee", "foot", "heel"],
}

def generate_body_sequences():
    s = []
    for group, parts in BODY_PARTS.items():
        group_word = group.rstrip("s")
        for part in parts:
            s.append(isa(part, group_word, "body_part", "body", "human"))
            s.append(part_of(part, group_word))
    # Functions
    body_funcs = [
        ("eye", "see"), ("ear", "hear"), ("nose", "smell"),
        ("tongue", "taste"), ("skin", "touch"), ("heart", "pump"),
        ("lung", "breathe"), ("brain", "think"), ("hand", "grasp"),
        ("foot", "walk"), ("mouth", "speak"), ("tooth", "chew"),
        ("stomach", "digest"), ("liver", "filter"), ("kidney", "filter"),
        ("muscle", "contract"), ("blood", "oxygenate"), ("bone", "support"),
        ("finger", "point"), ("arm", "reach"),
    ]
    for organ, func in body_funcs:
        s.append(function(organ, func))
    s.append(part_of("head", "body"))
    s.append(part_of("torso", "body"))
    s.append(part_of("limbs", "body"))
    return s


# ── NATURE ───────────────────────────────────────────────────────
NATURE = {
    "elements": {"water": ["liquid", "flows", "wets", "life"],
                 "fire": ["heat", "burns", "flame", "light"],
                 "earth": ["solid", "soil", "growth", "life"],
                 "air": ["invisible", "wind", "breathe", "life"]},
    "weather": {"rain": ["water", "cloud", "wet", "storm"],
                "snow": ["cold", "white", "ice", "winter"],
                "wind": ["air", "strong", "storm", "pushes"],
                "sun": ["light", "heat", "day", "sky"],
                "cloud": ["water", "white", "sky", "rain"]},
    "geography": {"river": ["water", "flows", "long", "fish"],
                  "sea": ["water", "salt", "wave", "deep"],
                  "jungle": ["tree", "green", "animal", "wet"],
                  "desert": ["sand", "dry", "heat", "cactus"]},
}

def generate_nature_sequences():
    s = []
    for cat, items in NATURE.items():
        for name, props in items.items():
            for p in props:
                s.append(chain(name, p, rel=NEXT))
            s.append(isa(name, cat.rstrip("s"), "nature", "world"))
    # Causal
    cause_pairs = [
        ("rain", "flood"), ("rain", "grow"),
        ("fire", "heat"), ("fire", "ash"),
        ("wind", "erosion"), ("wind", "wave"),
        ("sun", "light"), ("sun", "heat"),
        ("cloud", "rain"), ("snow", "cold"),
        ("river", "fertility"),
    ]
    for a, b in cause_pairs:
        s.append(cause(a, b))
    # Body-nature connections
    s.append(similar("river", "vein"))
    s.append(similar("lung", "leaf"))
    return s


# ── TECHNOLOGY ───────────────────────────────────────────────────
TECH = {
    "devices": {
        "base": "digital_device", "ancestors": ["tool", "technology"],
        "members": ["computer", "phone", "tablet", "tv", "smartwatch"],
    },
    "internet": {
        "base": "internet", "ancestors": ["network", "technology"],
        "members": ["browser", "email", "social_media", "search", "download"],
    },
    "software": {
        "base": "software", "ancestors": ["program", "technology"],
        "members": ["app", "os", "database", "algorithm"],
    },
    "hardware": {
        "base": "component", "ancestors": ["hardware", "technology"],
        "members": ["cpu", "memory", "hard_drive", "screen", "keyboard", "mouse"],
    },
}

def generate_tech_sequences():
    s = []
    for cat, group in TECH.items():
        base = group["base"]
        ancestors = group["ancestors"]
        for member in group["members"]:
            s.append(isa(member, base, *ancestors))
    # Functions
    func_pairs = [
        ("computer", "process"), ("computer", "calculate"),
        ("phone", "communicate"), ("browser", "browse"),
        ("app", "automate"), ("algorithm", "solve"),
        ("database", "organize"), ("os", "manage"),
        ("cpu", "compute"), ("memory", "store"),
        ("hard_drive", "save"), ("screen", "display"),
        ("keyboard", "type"), ("mouse", "navigate"),
    ]
    for device, func in func_pairs:
        s.append(function(device, func))
    # Instruments
    inst_pairs = [
        ("user", "keyboard", "type"),
        ("user", "mouse", "navigate"),
        ("programmer", "computer", "code"),
        ("designer", "app", "create"),
    ]
    for agent, tool, act in inst_pairs:
        s.append(instrument(agent, tool, act))
    s.append(part_of("cpu", "computer"))
    s.append(part_of("memory", "computer"))
    s.append(part_of("screen", "tv"))
    s.append(part_of("keyboard", "computer"))
    return s


# ── EMOTIONS ─────────────────────────────────────────────────────
EMOTIONS = {
    "positive": {"joy": "smile", "love": "hug", "hope": "dream",
                 "gratitude": "thank", "peace": "meditate", "excitement": "cheer"},
    "negative": {"sadness": "cry", "fear": "flee", "anger": "shout",
                 "anxiety": "worry", "loneliness": "isolate", "disgust": "avoid"},
}

def generate_emotion_sequences():
    s = []
    for cat, items in EMOTIONS.items():
        for emotion, action in items.items():
            s.append(isa(emotion, "emotion_{}".format(cat), "emotion", "human"))
            s.append(function(emotion, action))
    # Opposite pairs
    pairs = [("joy", "sadness"), ("love", "hate"), ("hope", "despair"),
             ("peace", "anger"), ("fear", "courage")]
    for a, b in pairs:
        a_in = a in EMOTIONS["positive"] or a in EMOTIONS["negative"]
        b_in = b in EMOTIONS["positive"] or b in EMOTIONS["negative"]
        if a_in and b_in:
            s.append(opposite(a, b))
    return s


# ── FOOD ─────────────────────────────────────────────────────────
FOOD = {
    "beverages": ["water", "juice", "coffee", "tea", "milk"],
    "meals": ["bread", "rice", "pasta", "meat", "cheese", "egg"],
    "cooking": ["cook", "boil", "fry", "bake", "chop"],
    "tastes": ["sweet", "salty", "bitter", "sour", "umami"],
}

def generate_food_sequences():
    s = []
    for cat, items in FOOD.items():
        for item in items:
            s.append(isa(item, cat.rstrip("s"), "food", "need"))
    s.append(function("chef", "cook"))
    s.append(instrument("chef", "knife", "chop"))
    s.append(part_of("flour", "bread"))
    s.append(part_of("water", "coffee"))
    s.append(cause("heat", "cook"))
    return s


# ── SPORTS ───────────────────────────────────────────────────────
SPORTS = {
    "ball": ["soccer", "tennis", "basketball", "volleyball", "baseball"],
    "water": ["swimming", "diving", "surfing", "rowing"],
    "combat": ["boxing", "karate", "judo", "fencing"],
}

def generate_sport_sequences():
    s = []
    for cat, sports in SPORTS.items():
        for sport in sports:
            s.append(isa(sport, "{}_sport".format(cat), "sport", "activity", "human"))
            s.append(function("player", "play"))
    s.append(function("soccer", "kick"))
    s.append(function("tennis", "hit"))
    s.append(function("swimming", "swim"))
    s.append(function("boxing", "punch"))
    s.append(instrument("swimmer", "pool", "swim"))
    s.append(instrument("tennis_player", "racket", "hit"))
    return s


# ── SPACE ────────────────────────────────────────────────────────
SPACE = {
    "celestial": {"sun": "star", "moon": "satellite", "earth": "planet",
                  "mars": "planet", "jupiter": "planet", "venus": "planet"},
    "phenomena": ["gravity", "light", "radiation", "magnetism", "vacuum"],
}

def generate_space_sequences():
    s = []
    for name, kind in SPACE["celestial"].items():
        s.append(isa(name, kind, "celestial_body", "universe"))
    for phenom in SPACE["phenomena"]:
        s.append(isa(phenom, "physical_phenomenon", "universe"))
    s.append(function("sun", "shine"))
    s.append(function("moon", "reflect"))
    s.append(cause("gravity", "attract"))
    s.append(cause("light", "illuminate"))
    s.append(part_of("solar_system", "galaxy"))
    s.append(part_of("galaxy", "universe"))
    s.append(part_of("earth", "solar_system"))
    s.append(part_of("moon", "solar_system"))
    return s


# ── SOCIETY ──────────────────────────────────────────────────────
SOCIETY_PROFESSIONS = {
    "doctor": {"function": "heal", "location": "hospital", "tool": "scalpel"},
    "teacher": {"function": "teach", "location": "school", "tool": "book"},
    "engineer": {"function": "design", "location": "office", "tool": "computer"},
    "lawyer": {"function": "defend", "location": "office", "tool": "document"},
    "artist": {"function": "create", "location": "studio", "tool": "brush"},
    "firefighter": {"function": "extinguish", "location": "fire_station", "tool": "hose"},
    "police": {"function": "protect", "location": "station", "tool": "badge"},
    "chef": {"function": "cook", "location": "restaurant", "tool": "knife"},
    "pilot": {"function": "fly", "location": "airport", "tool": "controls"},
    "writer": {"function": "write", "location": "library", "tool": "computer"},
}

SOCIETY_PLACES = ["school", "hospital", "library", "park", "museum",
                  "theater", "restaurant", "bank", "store", "office"]

def generate_society_sequences():
    s = []
    for prof, info in SOCIETY_PROFESSIONS.items():
        s.append(isa(prof, "professional", "occupation", "human"))
        s.append(function(prof, info["function"]))
        s.append(located_in(prof, info["location"]))
        s.append(instrument(prof, info["tool"], info["function"]))
    for place in SOCIETY_PLACES:
        s.append(isa(place, "place", "society"))
    # Place functions
    place_funcs = [
        ("school", "educate"), ("hospital", "heal"), ("library", "lend"),
        ("park", "recreate"), ("museum", "exhibit"), ("theater", "perform"),
        ("restaurant", "feed"), ("bank", "store_money"), ("office", "work"),
        ("store", "sell"),
    ]
    for place, func in place_funcs:
        s.append(function(place, func))
    s.append(part_of("classroom", "school"))
    s.append(part_of("operating_room", "hospital"))
    s.append(part_of("reading_room", "library"))
    return s


# ── COLORS ───────────────────────────────────────────────────────
def generate_color_sequences():
    s = []
    colors = ["red", "blue", "green", "yellow", "black", "white",
              "orange", "purple", "pink", "brown", "gray", "gold"]
    associations = {"red": "passion", "blue": "calm", "green": "nature",
                    "yellow": "joy", "black": "darkness", "white": "peace"}
    for color in colors:
        s.append(isa(color, "color", "property", "concept"))
    for color, meaning in associations.items():
        s.append(similar(color, meaning))
        s.append(symbol_of(color, meaning))
    return s

def symbol_of(a, b):
    return seq([a, b], [SYMBOL_OF])


# ── MATH ─────────────────────────────────────────────────────────
def generate_math_sequences():
    s = []
    ops = {"add": "sum", "subtract": "difference", "multiply": "product", "divide": "quotient"}
    shapes = ["circle", "square", "triangle", "rectangle", "diamond", "oval"]
    for op, result in ops.items():
        s.append(isa(op, "operation", "math", "science"))
        s.append(function(op, result))
    for shape in shapes:
        s.append(isa(shape, "shape", "geometry", "math"))
    s.append(cause("divide", "split"))
    s.append(cause("add", "increase"))
    s.append(cause("subtract", "decrease"))
    return s


# ── ABSTRACT ─────────────────────────────────────────────────────
ABSTRACT = {
    "concepts": ["time", "space", "energy", "matter", "information",
                 "truth", "beauty", "justice", "freedom", "knowledge"],
    "pairs": {"time": "clock", "space": "distance", "energy": "force",
              "matter": "mass", "information": "data",
              "truth": "fact", "beauty": "art", "justice": "law",
              "freedom": "right", "knowledge": "wisdom"},
}

def generate_abstract_sequences():
    s = []
    for concept in ABSTRACT["concepts"]:
        s.append(isa(concept, "abstract_concept", "idea", "mind"))
    for concept, related in ABSTRACT["pairs"].items():
        s.append(similar(concept, related))
    s.append(opposite("love", "hate"))
    s.append(opposite("truth", "lie"))
    s.append(opposite("freedom", "prison"))
    s.append(opposite("knowledge", "ignorance"))
    s.append(cause("knowledge", "power"))
    s.append(cause("time", "change"))
    s.append(cause("energy", "motion"))
    return s


# ── PHYSICS / SCIENCE ────────────────────────────────────────────
def generate_physics_sequences():
    s = []
    forces = {
        "gravity": ["attract", "mass", "weight"],
        "magnetism": ["attract", "metal", "magnet"],
        "electricity": ["charge", "light", "circuit"],
        "friction": ["heat", "resistance", "wear"],
    }
    processes = {
        "photosynthesis": ["light", "chlorophyll", "glucose"],
        "combustion": ["fire", "heat", "oxygen"],
        "evaporation": ["water", "vapor", "heat"],
        "condensation": ["vapor", "water", "cold"],
        "melting": ["solid", "liquid", "heat"],
        "freezing": ["liquid", "solid", "cold"],
    }
    materials = {
        "water": ("liquid", "transparent", "life"),
        "iron": ("metal", "hard", "conductor", "heavy"),
        "gold": ("metal", "shiny", "valuable", "soft"),
        "plastic": ("synthetic", "moldable", "insulator", "light"),
    }
    for cat, items in forces.items():
        for p in items:
            s.append(chain(cat, p, rel=NEXT))
        s.append(isa(cat, "force", "physical_phenomenon", "science", "knowledge"))
    for process, props in processes.items():
        for p in props:
            s.append(chain(process, p, rel=NEXT))
        s.append(isa(process, "process", "science", "knowledge"))
    for mat, props in materials.items():
        for p in props:
            s.append(chain(mat, p, rel=NEXT))
        s.append(isa(mat, "material", "science", "knowledge"))
    # Causality
    for a, b in [("gravity", "fall"), ("gravity", "weight"),
                 ("magnetism", "attraction"), ("electricity", "light"),
                 ("friction", "heat"), ("photosynthesis", "oxygen"),
                 ("combustion", "heat"), ("evaporation", "vapor"),
                 ("condensation", "rain"), ("melting", "liquid"),
                 ("freezing", "ice"), ("heat", "evaporation"),
                 ("cold", "freezing")]:
        s.append(cause(a, b))
    s.append(part_of("electricity", "physics"))
    s.append(part_of("magnetism", "physics"))
    s.append(part_of("gravity", "physics"))
    s.append(part_of("photosynthesis", "biology"))
    s.append(part_of("combustion", "chemistry"))
    s.append(function("magnet", "attract"))
    s.append(function("lamp", "illuminate"))
    return s


# ── ACTIONS / DAILY LIFE ─────────────────────────────────────────
ACTIONS = {
    "verbs": ["run", "jump", "swim", "fly", "walk", "roll",
              "push", "pull", "lift", "carry"],
    "results": {"run": "speed", "jump": "height", "swim": "water",
                "fly": "air", "walk": "movement", "roll": "rotation"},
}

def generate_action_sequences():
    s = []
    for verb in ACTIONS["verbs"]:
        s.append(isa(verb, "physical_action", "movement", "concept"))
    for verb, result in ACTIONS["results"].items():
        s.append(cause(verb, result))
    # Causality
    for a, b in [("push", "move"), ("pull", "move"), ("lift", "raise"),
                 ("carry", "transport"), ("hit", "break"), ("cut", "separate"),
                 ("eat", "nourish"), ("drink", "hydrate")]:
        s.append(cause(a, b))
    # Instruments
    for agent, tool, act in [("person", "hand", "grasp"), ("person", "foot", "walk"),
                              ("person", "knife", "cut"), ("person", "hammer", "hit"),
                              ("person", "key", "open"), ("person", "pencil", "write")]:
        s.append(instrument(agent.replace(" ", "_"), tool, act))
    # Opposite actions
    s.append(opposite("push", "pull"))
    s.append(opposite("open", "close"))
    s.append(opposite("enter", "exit"))
    s.append(opposite("create", "destroy"))
    return s


# ════════════════════════════════════════════════════════════════
# INVERSE GENERATION
# ════════════════════════════════════════════════════════════════
def generate_inverses(sequences, inverse_map):
    inverses = []
    seen = set()
    for entry in sequences:
        tokens = entry.get("sequence", [])
        rels = entry.get("relations")
        if rels is None or len(tokens) < 2:
            continue
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            rel = rels[i]
            inv_rel = inverse_map.get(rel)
            if inv_rel is None:
                continue
            inv_pair = (b, a, inv_rel)
            if inv_pair not in seen:
                seen.add(inv_pair)
                inverses.append({"sequence": [b, a], "relations": [inv_rel]})
    return sequences + inverses


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
def main():
    generators = [
        ("animals", generate_animal_sequences),
        ("plants", generate_plant_sequences),
        ("body", generate_body_sequences),
        ("nature", generate_nature_sequences),
        ("technology", generate_tech_sequences),
        ("emotions", generate_emotion_sequences),
        ("food", generate_food_sequences),
        ("sports", generate_sport_sequences),
        ("space", generate_space_sequences),
        ("society", generate_society_sequences),
        ("colors", generate_color_sequences),
        ("math", generate_math_sequences),
        ("abstract", generate_abstract_sequences),
        ("physics", generate_physics_sequences),
        ("actions", generate_action_sequences),
    ]
    all_sequences = []
    for name, gen in generators:
        seqs = gen()
        all_sequences.extend(seqs)

    # Deduplicate
    seen = set()
    deduped = []
    for s in all_sequences:
        key = tuple(s["sequence"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    # Add inverse relations
    final = generate_inverses(deduped, INVERSE_MAP)

    # Stats
    vocab = set()
    for s in final:
        vocab.update(s["sequence"])

    dataset = {
        "name": "conceptos_rutas_v7_en",
        "description": "Dataset v7 EN: {} sequences, {} words, structural relations".format(
            len(final), len(vocab)),
        "sequences": final,
    }

    out_path = "datasets/dataset_v7_en.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("  Generators: {}".format(len(generators)))
    print("  Raw sequences: {}".format(len(all_sequences)))
    print("  After dedup: {}".format(len(deduped)))
    print("  After inverses: {}".format(len(final)))
    print("  Unique words: {}".format(len(vocab)))
    print("  Saved: {}".format(out_path))


if __name__ == "__main__":
    main()
