import json
from collections import defaultdict

# Load schemas
with open("data/processed/layer3_schemas.json") as f:
    data = json.load(f)

# Map: entity -> set of relation types
entity_relations = defaultdict(set)
entity_connections = defaultdict(list)  # entity -> [(target, relation)]

for schema in data["schemas"]:
    rel_seq = schema["relation_sequence"]
    primary_rel = rel_seq[0]  # first relation
    roles = schema["roles"]
    for inst in schema["instances"]:
        vals = list(inst.values())
        if len(vals) >= 2:
            source = vals[0]
            target = vals[1]
            entity_relations[source].add(primary_rel)
            entity_connections[source].append((target, primary_rel))

# Show entities with most relation variety
sorted_entities = sorted(entity_relations.items(), key=lambda x: -len(x[1]))
print("Entities with most relation types:")
for entity, rels in sorted_entities[:20]:
    counts = defaultdict(int)
    for t, r in entity_connections[entity]:
        counts[r] += 1
    rel_detail = ", ".join(f"{r}={counts[r]}" for r in sorted(counts))
    print(f"  {entity}: {len(rels)} types [{rel_detail}]")
