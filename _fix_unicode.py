import os, glob
for fp in glob.glob('*.py'):
    if fp in ('_fix_unicode.py',):
        continue
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    bad = set()
    for c in content:
        if 127 < ord(c) < 160:
            bad.add(c)
        if ord(c) > 255:
            bad.add(c)
    if bad:
        for c in bad:
            print(f'{fp}: U+{ord(c):04X} = {repr(c)}')
    else:
        print(f'{fp}: OK')
