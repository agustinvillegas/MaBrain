# MaBrain — Diseño de Entrenamiento Masivo

> Paradigma: **razonamiento bio-inspirado sobre grafo relacional**, no next-token prediction.

## Tabla de Contenidos

1. [Principios de Diseño](#1-principios-de-diseño)
2. [Estructura del Dataset: 3 Capas](#2-estructura-del-dataset-3-capas)
   - [Capa 1: Knowledge Graph Canónico](#capa-1-knowledge-graph-canónico-hechos-atómicos)
   - [Capa 2: Caminos de Razonamiento](#capa-2-caminos-de-razonamiento-secuencias-multi-salto)
   - [Capa 3: Esquemas Abstractos](#capa-3-esquemas-abstractos-para-analogía-estructural)
3. [Pipeline de Construcción](#3-pipeline-de-construcción)
4. [Tamaños Objetivo](#4-tamaños-objetivo)
5. [Loop de Entrenamiento Efectivo](#5-loop-de-entrenamiento-efectivo)
6. [Diferencias vs. Entrenamiento Tradicional (LLM)](#6-diferencias-vs-entrenamiento-tradicional-llm)
7. [Estimación de Esfuerzo Humano](#7-estimación-de-esfuerzo-humano)
8. [Próximos Pasos Accionables](#8-próximos-pasos-accionables)

---

## 1. Principios de Diseño

**Next-token prediction sobre texto crudo es el objetivo equivocado** para MaBrain. La "función de pérdida" implícita de este paradigma es:

> **Estructura relacional correcta + costos bajos en caminos útiles.**

| Objetivo del paradigma | Formato de dato requerido |
|------------------------|---------------------------|
| `ConceptRegistry` canónico | 1 célula por concepto (deduplicado) |
| Sinapsis tipadas semánticamente | Tripletas `(origen, relación, destino)` con `relation` ∈ `{IS_A, CAUSE, FUNCTION, PART_OF, HAS_PROPERTY, LOCATED_IN, USED_FOR, INSTRUMENT, OPPOSITE, SIMILAR_TO}` |
| Costos/fuerzas hebbianos | **Secuencias de razonamiento** (caminos multi-salto) que se recorren y recompensan |
| Analogía estructural | **Patrones relacionales recurrentes** (mismo esquema en dominios distintos) |
| Generalización composicional | **Esquemas abstractos** instanciados en múltiples ejemplos |

### ¿Por qué no sirve texto crudo?

| Problema | Consecuencia |
|----------|--------------|
| Sin relaciones explícitas | `learn()` crea solo `NEXT` — no hay IS_A, CAUSE, FUNCTION |
| Ruido estadístico | Palabras sin contenido semántico se registran como conceptos |
| Sin coherencia causal | "perro ataca gato por instinto" se vuelve secuencia plana, no cadena causal |
| Fragmentación | Cada oración crea células duplicadas (singular/plural, formas conjugadas) |

---

## 2. Estructura del Dataset: 3 Capas

### Capa 1: Knowledge Graph Canónico (Hechos Atómicos)

Tripletas sujeto-relación-objeto deduplicadas. Es la base de todo el conocimiento factual.

```json
{
  "version": "v1.0",
  "triplets": [
    {
      "from": "perro",
      "relation": "IS_A",
      "to": "mamifero",
      "source": "wikidata",
      "confidence": 0.99,
      "domain": "biologia"
    },
    {
      "from": "perro",
      "relation": "FUNCTION",
      "to": "ladrar",
      "source": "conceptnet",
      "confidence": 0.9,
      "domain": "etologia"
    },
    {
      "from": "perro",
      "relation": "CAUSE",
      "to": "instinto_presa",
      "source": "atomic",
      "confidence": 0.8,
      "domain": "etologia"
    },
    {
      "from": "instinto_presa",
      "relation": "CAUSE",
      "to": "perseguir_gato",
      "source": "atomic",
      "confidence": 0.75,
      "domain": "etologia"
    }
  ]
}
```

**Fuentes recomendadas**:

| Fuente | Relaciones que aporta | Tamaño raw | Calidad |
|--------|----------------------|------------|---------|
| **Wikidata** | IS_A (P31/P279), PART_OF (P361), LOCATED_IN (P131) | ~10M tripletas | Alta (curada) |
| **ConceptNet 5.7** | IS_A, FUNCTION, CAUSE, USED_FOR, HAS_PROPERTY, PART_OF, LOCATED_IN | ~3M tripletas | Media-alta (multilingüe) |
| **ATOMIC 2020** | CAUSE, DESIRES, INSTRUMENT, EFFECT_OF | ~1M cadenas | Media-alta (social+eventos) |
| **WordNet** | IS_A (hiperonimia), PART_OF (meronimia), PROPERTY_OF | ~200K | Alta (lingüística) |
| **ATOMIC10X** | CAUSE, EFFECT, INTENT, ATTRIBUTE | ~10M | Media (LLM-generado) |
| **Specialized KGs** (biomed, legal, code) | Según dominio | Variable | Alta |

**Procesamiento**:
- Embedding-based entity linking (MiniLM + clustering, umbral 0.85)
- Filtro: mínimo 2 fuentes o confidence > 0.9
- Deduplicación por `ConceptRegistry` (merge de sinónimos)

### Capa 2: Caminos de Razonamiento (Secuencias Multi-Salto)

Secuencias encadenadas de relaciones que forman argumentos, explicaciones o cadenas causales. Son la señal de supervisión para `reward_thought()`.

```json
{
  "reasoning_paths": [
    {
      "path": ["fiebre", "respuesta_inmune", "muerte_patogeno", "recuperacion"],
      "relations": ["CAUSE", "CAUSE", "CAUSE"],
      "type": "causal_chain",
      "domain": "biologia",
      "difficulty": 1
    },
    {
      "path": ["corazon", "bombear", "sangre", "oxigeno", "celulas"],
      "relations": ["FUNCTION", "TRANSPORTS", "CONTAINS", "USES"],
      "type": "functional_chain",
      "domain": "fisiologia",
      "difficulty": 1
    },
    {
      "path": ["falta_sueño", "fatiga", "bajo_rendimiento", "error_trabajo", "estres"],
      "relations": ["CAUSE", "CAUSE", "CAUSE", "CAUSE"],
      "type": "causal_chain",
      "domain": "salud",
      "difficulty": 2
    },
    {
      "path": ["perro", "instinto_presa", "perseguir", "gato", "huir"],
      "relations": ["CAUSE", "CAUSE", "CAUSE", "CAUSE"],
      "type": "behavioral_chain",
      "domain": "etologia",
      "difficulty": 2
    }
  ]
}
```

**Tipos de caminos**:
- `causal_chain`: A → B → C → D (causas)
- `functional_chain`: A → B → C → D (funciones)
- `partonomic_chain`: A → B → C → D (partes de)
- `analogical_pair`: A:B :: C:D (para entrenar analogía directa)
- `problem_solution`: problema → causa → acción → efecto
- `hierarchical`: concepto → IS_A → categoría → IS_A → supercategoría

**Generación**:
- BFS sobre grafo Capa 1 (depth 3-6) desde nodos semilla
- Filtrado: solo caminos con relaciones coherentes (CAUSE\*, FUNCTION\*, PART_OF\*)
- Balanceo por dominio (biología, física, social, cotidiano, técnico)

### Capa 3: Esquemas Abstractos (Para Analogía Estructural)

Patrones relacionales reutilizables en distintos dominios. Entrenan directamente `analogy_structural()`.

```json
{
  "schemas": [
    {
      "schema_id": "ORGAN_FUNCTION",
      "description": "Un órgano realiza una función específica",
      "pattern": [
        {"role": "organ", "relations": ["FUNCTION"]},
        {"role": "action"}
      ],
      "instances": [
        {"organ": "corazon", "action": "bombear"},
        {"organ": "pulmon", "action": "respirar"},
        {"organ": "rinon", "action": "filtrar"},
        {"organ": "higado", "action": "detoxificar"},
        {"organ": "cerebro", "action": "procesar"}
      ],
      "domain": "biologia",
      "difficulty": 1
    },
    {
      "schema_id": "PREDATOR_PREY",
      "description": "Depredador → instinto → persigue → presa → huye",
      "pattern": [
        {"role": "predator", "relations": ["CAUSE"]},
        {"role": "instinct"},
        {"role": "chase", "relations": ["CAUSE"]},
        {"role": "prey"},
        {"role": "flee", "relations": ["CAUSE"]}
      ],
      "instances": [
        {"predator": "leon", "instinct": "instinto_cazar", "chase": "perseguir", "prey": "cebra", "flee": "huir"},
        {"predator": "aguila", "instinct": "instinto_cazar", "chase": "picar", "prey": "raton", "flee": "escapar"},
        {"predator": "perro", "instinct": "instinto_presa", "chase": "perseguir", "prey": "gato", "flee": "huir"}
      ],
      "domain": "etologia",
      "difficulty": 2
    },
    {
      "schema_id": "TOOL_FUNCTION",
      "description": "Una herramienta sirve para una acción",
      "pattern": [
        {"role": "tool", "relations": ["USED_FOR"]},
        {"role": "action"}
      ],
      "instances": [
        {"tool": "martillo", "action": "clavar"},
        {"tool": "sierra", "action": "cortar"},
        {"tool": "destornillador", "action": "atornillar"},
        {"tool": "llave", "action": "ajustar"}
      ],
      "domain": "tecnico",
      "difficulty": 1
    },
    {
      "schema_id": "HEAT_CAUSE_EFFECT",
      "description": "Fuente de calor → causa efecto",
      "pattern": [
        {"role": "source", "relations": ["CAUSE"]},
        {"role": "object"},
        {"role": "effect", "relations": ["CAUSE"]}
      ],
      "instances": [
        {"source": "fuego", "object": "agua", "effect": "hervir"},
        {"source": "sol", "object": "hielo", "effect": "derretir"},
        {"source": "calor_ corporal", "object": "metal", "effect": "calentar"},
        {"source": "electricidad", "object": "bombilla", "effect": "iluminar"}
      ],
      "domain": "fisica",
      "difficulty": 2
    }
  ]
}
```

**Extracción**:
1. Mining de patrones relacionales frecuentes (Apriori sobre caminos de Capa 2)
2. Clustering de patrones isomórficos (misma secuencia de relaciones)
3. Validación humana de 200-500 esquemas clave
4. Expansión automática de instancias por similitud de embeddings + verificación

---

## 3. Pipeline de Construcción

```
FASE 1: EXTRACCIÓN KG BRUTO (1-2 semanas)
├── 1a. Wikidata dump → SPARQL queries (IS_A, PART_OF, LOCATED_IN)
│     Output: ~10M tripletas raw
├── 1b. ConceptNet 5.7 → python API (FUNCTION, CAUSE, USED_FOR, HAS_PROPERTY)
│     Output: ~3M tripletas raw
├── 1c. ATOMIC 2020/ATOMIC10X → cadenas causales
│     Output: ~1-10M cadenas raw
└── 1d. WordNet → taxonomía IS_A/HYPERNYM limpia
      Output: ~200K tripletas

FASE 2: LIMPIEZA + CANONIZACIÓN (1-2 semanas)
├── 2a. Embedding-based entity linking (MiniLM + clustering, umbral 0.85)
│     Agrupa: "perro", "can", "dog", "perro_domestico" → "perro"
├── 2b. Filtro de confianza (≥2 fuentes o confidence > 0.9)
├── 2c. Deduplicación por ConceptRegistry (merge sinónimos)
├── 2d. Balanceo por dominio y relación
└── 2e. Salida: Capa 1 (500K-2M tripletas únicas)

FASE 3: GENERACIÓN DE CAMINOS (1 semana)
├── 3a. BFS sobre grafo limpio (depth 3-6) desde 10K nodos semilla
├── 3b. Filtrado: solo caminos con relaciones coherentes
├── 3c. Anotación automática de tipo (causal, funcional, partonómico, etc.)
├── 3d. Balanceo por dominio + dificultad
└── 3e. Salida: Capa 2 (50K-200K caminos)

FASE 4: EXTRACCIÓN DE ESQUEMAS (1-2 semanas con validación humana)
├── 4a. Mining de secuencias de relaciones frecuentes (Apriori)
├── 4b. Clustering de patrones isomórficos
├── 4c. Validación humana de 200-500 esquemas clave (20-40h)
├── 4d. Expansión automática de instancias (embeddings + verificación)
└── 4e. Salida: Capa 3 (200-500 esquemas × 5-30 instancias)

FASE 5: GENERACIÓN DATASET FINAL (2-3 días)
├── 5a. Integrar 3 capas en un único archivo estructurado
├── 5b. Estadísticas de cobertura y distribución
├── 5c. Validación cruzada (holdout 10% para evaluación)
└── 5d. Salida: dataset_final_v1.json
```

**Duración total estimada**: 5-8 semanas (1 persona tiempo completo).
**Dependencias externas**: MiniLM offline, SPARQL endpoint, 16-32 GB RAM, 1 GPU modesta para embeddings (opcional si se usa CPU + char-ngrams).

---

## 4. Tamaños Objetivo

| Componente | Mínimo Viable | Target Robusto | Ambicioso |
|------------|---------------|----------------|-----------|
| **Capa 1** | | | |
| Conceptos únicos | 50K | 150K | 300K |
| Tripletas totales | 200K | 800K | 2M |
| Tipos de relación | 6 | 10 | 14 |
| **Capa 2** | | | |
| Caminos razonamiento | 20K | 100K | 300K |
| Tipos de camino | 3 | 5 | 7 |
| Longitud promedio | 3 | 4 | 5 |
| **Capa 3** | | | |
| Esquemas abstractos | 100 | 350 | 600 |
| Instancias por esquema | 5 | 15 | 30 |
| **RAM estimada (grafo cargado)** | **~1.5 GB** | **~4 GB** | **~10 GB** |

> Con 150K conceptos + 800K tripletas + embeddings MiniLM en int8 ≈ **2.5-3 GB RAM**. Cabe en target 1-2 GB si se poda agresivamente (top-k sinapsis por nodo, quantización de embeddings).

---

## 5. Loop de Entrenamiento Efectivo

```python
# train_massive.py — Pseudo-código del loop correcto

def train_massive(brain, dataset, epochs=3):
    """Entrenamiento masivo para razonamiento bio-inspirado.

    NO es: next-token prediction.
    ES: construcción de estructura relacional + refuerzo de caminos útiles.
    """

    # ── FASE 1: Carga de Hechos Atómicos (Capa 1) ──
    # Crea la estructura base del grafo: conceptos + sinapsis tipadas.
    # No hay "pérdida" — cada tripleta es un hecho que se registra.
    for triplet in dataset["triplets"]:
        cell_a = brain.get_or_create_cell(triplet["from"])
        cell_b = brain.get_or_create_cell(triplet["to"])
        brain.connect(cell_a, cell_b, triplet["to"], relation=triplet["relation"])

    brain.auto_infer_relations()  # Completa IS_A/PROPERTY/NEXT por topología

    # ── FASE 2: Refuerzo de Caminos (Capa 2) ──
    # Recorre caminos expertos y aplica reward_thought.
    # Esto entrena costos/fuerzas: caminos útiles se abaratan.
    for epoch in range(epochs):
        for path_data in dataset["reasoning_paths"]:
            # Walk forzado por el camino experto
            current = brain.get_or_create_cell(path_data["path"][0])
            for i in range(1, len(path_data["path"])):
                target = brain.get_or_create_cell(path_data["path"][i])
                rel = path_data["relations"][i-1] if i-1 < len(path_data["relations"]) else None
                # Busca o crea la sinapsis
                syn = brain.connect(current, target, path_data["path"][i], relation=rel)
                syn.activation_trace += syn.strength
                current = target

            # Recompensa global por el camino completo
            brain.reward_thought(score=1.0 / len(path_data["path"]))

    # ── FASE 3: Entrenamiento de Esquemas (Capa 3) ──
    # No hay "backprop" — se registran las instancias en el grafo.
    # La generalización emerge vía analogy_structural() durante inferencia.
    for schema in dataset["schemas"]:
        for instance in schema["instances"]:
            # Registrar cada instancia como caminos en el grafo
            # El patrón queda implicitamente en la topología relacional
            pass  # Las instancias ya están en Capa 1/Capa 2 normalmente

    # ── FASE 4: Poda + Consolidación ──
    removed = brain.prune(min_usage=2)  # Elimina sinapsis ruidosas
    print(f"Podadas {removed} sinapsis")

    # Normalización global de costos
    total_cost = sum(s.cost for s in brain.synapses.values())
    avg_cost = total_cost / max(len(brain.synapses), 1)
    for s in brain.synapses.values():
        s.cost = s.cost / avg_cost  # Normalizar

    brain.save("states/brain_massive_v1.json")
```

### ¿Qué NO hay aquí?

| Ausencia | Por qué |
|----------|---------|
| **Función de pérdida** | No hay gradiente global. La señal es recompensa local (`reward_thought`). |
| **Batches aleatorios** | El currículo importa: hechos → caminos → esquemas. |
| **Backpropagation** | El aprendizaje es Hebbiano (local, co-ocurrencia). |
| **Validación por perplexity** | Se evalúa por precisión de analogías y recall de caminos. |
| **Fine-tuning por LoRA** | El aprendizaje continuo es nativo (`reward_thought` en vivo). |

---

## 6. Diferencias vs. Entrenamiento Tradicional (LLM)

| Dimensión | Entrenamiento Tradicional (Transformer) | Entrenamiento MaBrain (Grafo Bio-Razonamiento) |
|-----------|----------------------------------------|------------------------------------------------|
| **Objetivo de pérdida** | Next-token prediction (cross-entropy). Minimizar perplexity. | Estructura relacional correcta + costos bajos en caminos útiles. |
| **Señal de supervisión** | Gradiente global (backprop) calculado sobre toda la red. | Recompensa local Hebbiana (`reward_thought`) sobre sinapsis activas. |
| **Unidad de aprendizaje** | Token (subpalabra, ≈4 caracteres). | Concepto + Relación (tripletas, paths, esquemas). |
| **Generalización** | Distribuida: cada peso participa en millones de inferencias. Emergente, opaca. | Composicional: isomorfismo relacional vía `analogy_structural()`. Explícita, trazable. |
| **Currículo** | Aleatorio (shuffle global + batches i.i.d.). | Estructurado: hechos atómicos → caminos → esquemas abstractos. |
| **Memoria / Conocimiento** | Implícita en pesos (caja negra, ineditable). | Explícita en grafo (inspeccionable, editable, auditable). |
| **Aprendizaje continuo** | Catastrophic forgetting. Requiere LoRA/FT con datos nuevos. | Online nativo: `reward_thought` + `punish_thought` en milisegundos, sin olvido catastrófico. |
| **Escalabilidad de parámetros** | Leyes de potencia empíricas: más parámetros → mejor rendimiento. | Leyes de cobertura: más tripletas verificadas → mejor precisión. |
| **Hardware de inferencia** | GPU obligatoria (matmul denso, 8B FLOPs/token). | CPU suficiente (walk disperso, O(steps × degree) por inferencia). |
| **Hardware de entrenamiento** | GPU masiva (miles de horas, cientos de miles de USD). | CPU (horas-días). GPU solo para embeddings offline (opcional). |
| **Razonamiento** | Emergente con escala (no garantizado, no trazable). | Explícito, trazable, verificable (camino en grafo). |
| **Entrada de datos** | Texto crudo masivo (1T+ tokens, cualquier fuente). | Conocimiento estructurado curado (500K-2M tripletas de fuentes confiables). |
| **Evaluación** | Perplexity (proxy). Benchmarks downstream (MMLU, GSM8K, etc.). | Precisión de analogías (top-1/top-5). Recall de caminos multi-hop. Coherencia de diálogo humano. |
| **Alucinaciones** | Frecuentes (el modelo genera texto plausible pero falso). | Bajas (el grafo solo sabe lo que tiene; no "inventa" hechos nuevos sin supervisión). |
| **Interpretabilidad** | Baja (pesos distribuidos, atención como proxy). | Alta (cada inferencia es un camino trazable de conceptos + relaciones). |
| **Edición de conocimiento** | Difícil (editar un hecho = re-entrenar o cirugía de pesos). | Trivial (eliminar/agregar sinapsis directamente). |

### Diferencia Filosófica Central

```
TRADICIONAL:  "Aprende a predecir la siguiente palabra → emerge razonamiento"
TU PARADIGMA: "Codifica razonamiento estructurado explícitamente → generaliza por isomorfismo relacional"
```

El LLM tradicional depende de que la escala bruta de datos + parámetros produzca razonamiento como *subproducto emergente*. MaBrain invierte la causalidad: el razonamiento (caminos, analogías, esquemas) es la *unidad fundamental de aprendizaje*, y la generación de lenguaje es solo la *superficie de salida*.

### Implicaciones Prácticas

| Aspecto | Gana Tradicional | Gana MaBrain |
|---------|-----------------|--------------|
| Fluidez lingüística | ✅ | ❌ |
| Cobertura de conocimiento masivo | ✅ | ❌ |
| Aprendizaje de 1 ejemplo (online) | ❌ | ✅ |
| Razonamiento verificable | ❌ | ✅ |
| Edición / corrección de conocimiento | ❌ | ✅ |
| Sin GPU (inferencia CPU) | ❌ | ✅ |
| Sin olvido catastrófico | ❌ | ✅ |
| Generalización composicional (analogías) | ❌ | ✅ |

---

## 7. Estimación de Esfuerzo Humano

| Fase | Tarea | Esfuerzo (horas) | Herramientas | Perfil requerido |
|------|-------|-------------------|--------------|------------------|
| **F1** | Extracción Wikidata (SPARQL + parse) | 20-30 | Python + SPARQL | Ingeniero de datos |
| **F1** | Extracción ConceptNet | 10-15 | conceptnet python lib | Ingeniero de datos |
| **F1** | Extracción ATOMIC | 10-20 | python + NLP básico | Ingeniero de datos |
| **F1** | Extracción WordNet | 5-10 | nltk | Ingeniero de datos |
| **F2** | Entity linking + clustering embeddings | 20-30 | MiniLM, sklearn, FAISS | ML Engineer |
| **F2** | Deduplicación + fusión de sinónimos | 10-20 | python | Ingeniero de datos |
| **F2** | Filtro de confianza + balanceo | 10-15 | python, pandas | Ingeniero de datos |
| **F3** | BFS sobre grafo (generación de caminos) | 15-20 | python, networkx | Ingeniero de datos |
| **F3** | Filtrado + anotación de tipos | 10-15 | python | Ingeniero de datos |
| **F4** | Mining de patrones relacionales | 15-25 | python, apriori/fpgrowth | ML Engineer |
| **F4** | Validación humana de esquemas | **20-40** | revisión manual | **Experto de dominio** |
| **F4** | Expansión de instancias | 10-15 | embedding similarity | Ingeniero de datos |
| **F5** | Integración + validación cruzada | 10-15 | python, pytest | ML Engineer |
| **F5** | Evaluación de analogías (holdout) | 10-15 | `eval/evaluate.py` | ML Engineer |
| **Total** | **Pipeline completo** | **~175-285h** **(≈5-8 semanas 1 persona FT)** | | |

- **Sin GPU disponible**: Añadir +40h para implementar char-ngram clustering (menos preciso, más lento).
- **Con GPU para embeddings**: Reducir F2 en ~15h.
- **Crowdsourcing para validación de esquemas**: Reducir a 5-10h de supervisión + ~100h pagadas.

---

## 8. Próximos Pasos Accionables

### Prioridad 1: Scripts de Extracción (Capa 1)

| Script | Descripción | Dependencias | Tiempo |
|--------|-------------|--------------|--------|
| `extract_wikidata.py` | SPARQL queries → tripletas IS_A, PART_OF, LOCATED_IN | `requests`, `sparqlwrapper` | 2-3 días |
| `extract_conceptnet.py` | API ConceptNet → tripletas FUNCTION, CAUSE, USED_FOR | `conceptnet5` | 1-2 días |
| `extract_atomic.py` | Parse ATOMIC → cadenas causales | python stdlib | 2-3 días |
| `canonicalize.py` | Entity linking + dedup + fusion | `sentence-transformers`, `sklearn` | 2-4 días |
| `build_triplets.py` | Integra fuentes → Capa 1 final | pandas | 1 día |

### Prioridad 2: Scripts de Caminos y Esquemas (Capas 2-3)

| Script | Descripción | Dependencias | Tiempo |
|--------|-------------|--------------|--------|
| `generate_paths.py` | BFS desde semilla → caminos multi-salto | networkx | 2 días |
| `filter_paths.py` | Filtra por coherencia + tipo + dominio | python | 1 día |
| `mine_schemas.py` | Mining de patrones relacionales frecuentes | `mlxtend` o custom | 2-3 días |
| `expand_schema_instances.py` | Embedding-similarity + verif | `sentence-transformers` | 1-2 días |

### Prioridad 3: Entrenamiento Masivo

| Script | Descripción | Tiempo |
|--------|-------------|--------|
| `train_massive.py` | Loop completo: carga → refuerzo → poda → save | 2-4 días |
| `eval_analogy.py` | Evaluación automática en holdout schemas (top-1/top-5) | 1 día |
| `eval_dialogue.py` | Evaluación humana: coherencia, factualidad, razonamiento | 2-3 días |

### Prioridad 4: Demo Conversacional

| Script | Descripción | Tiempo |
|--------|-------------|--------|
| `demo/chat.py` | CLI interactivo con `Brain.get_response()` | 1 día |
| `demo/streamlit_app.py` | UI web simple con Streamlit | 2-3 días |

---

## Resumen Ejecutivo

1. **Objetivo**: Construir dataset masivo estructurado para MaBrain en 3 capas (tripletas, caminos, esquemas).
2. **Plazo**: 5-8 semanas para pipeline completo (1 persona FT).
3. **Tamaño target**: 150K conceptos + 800K tripletas + 100K caminos + 350 esquemas ≈ **2.5-3 GB RAM**.
4. **Diferenciador clave vs LLM**: Razonamiento explícito trazable, aprendizaje online, sin GPU.
5. **Riesgo principal**: Calidad de validación humana de esquemas (20-40h de experto de dominio).
6. **Primer milestone**: Script `extract_wikidata.py` + `canonicalize.py` → 200K tripletas en 2 semanas.

---

*Documento de diseño v1.0 — Junio 2026.*
