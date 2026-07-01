import json
with open("data/processed/layer3_schemas.json") as f:
    s = json.load(f)
for schema in s["schemas"][:10]:
    seq = " -> ".join(schema["relation_sequence"])
    print(f"{schema['schema_id']}: [{seq}] ({schema['num_instances']} instances)")
    for inst in schema["instances"][:2]:
        print(f"  {list(inst.values())}")
print(f"\nTotal schemas: {len(s['schemas'])}")
