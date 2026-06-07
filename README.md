# MaBrain — Motor de razonamiento por grafo eficiente

Sistema de razonamiento bio-inspirado que representa conocimiento como un **grafo de conceptos con costos** y piensa eligiendo **caminos de bajo costo energético**.

No multiplica matrices. No necesita GPU para funcionar.

## Arquitectura

Tres entidades:

- **Braincell**: nodo del grafo. Cada uno puede representar un concepto (`.word`).
- **Synapse**: conexión dirigida entre dos células. Contiene un `concept` (la palabra que la etiqueta), `strength` (fuerza), `cost` (costo de usarla), y `activation_trace` (memoria temporal de participación).
- **Brain**: el grafo completo. Gestiona células, sinapsis, aprendizaje, pensamiento, persistencia.

### Principios operativos

- **Los conceptos viven en las conexiones.** Las sinapsis llevan la etiqueta semántica.
- **Pensar = caminar el grafo con costo.** `think()` selecciona sinapsis por `efficiency = cost / strength` vía softmax con temperatura + epsilon-greedy.
- **Refuerzo Hebbiano.** `reward_thought()` fortalece las sinapsis activas y abarata su costo futuro.
- **ConceptRegistry.** Cada concepto canónico se asigna a una célula. `learn()` reusa células para el mismo concepto.

## Uso

```python
from braincell import Brain

brain = Brain()
brain.learn("ser vivo animal perro ladra corre")

start = brain.get_or_create_cell("ser_vivo")
pensamiento = brain.think(start, steps=5, temperature=1.0, epsilon=0.1)
print(pensamiento)  # ej: "animal perro ladra corre"

brain.save()
```

### Entrenar desde dataset

```bash
python train.py
```

### Exportar estado del grafo

```bash
python brain_graph.py
```

### Tests

```bash
python -m unittest tests.test_core -v
```

## Roadmap

Ver [docs/article.md](docs/article.md) y el plan completo en este README (fases):

1. Fase 0 ✅ — Saneamiento de base (ConceptRegistry, softmax, reward_thought, tests)
2. Fase 1 — Substrato de lenguaje (n-gramas, sinapsis tipadas, poda)
3. Fase 2 — Algoritmo de razonamiento (analogías por relación)
4. Fase 3 — Robustez (embeddings externos, refuerzo negativo, temperatura dinámica)
5. Fase 4 — Demo presentable (CLI, visualización, benchmark)

## Licencia

MIT
