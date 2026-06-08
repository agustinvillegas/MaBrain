import sys; sys.path.insert(0, '.')
from braincell import Brain
b = Brain()
b.load(r'D:\ma_brain_data\brain_state_v8_v3_pruned.json')
print(f'Brain: {len(b.cells)} cells, {len(b.synapses)} synapses')
for msg in ['dog cat', 'doctor hospital', 'water fire']:
    resp = b.get_response(msg, steps=8)
    print(f'  "{msg}" -> "{resp}"')
    active = b.wm.get_active_entities()
    print(f'  WM turns: {len(b.wm.turns)}, active: {active}')
