import unittest
from braincell import Brain, Braincell


def _make_brain():
    """Crea un brain pequeño con relaciones estructurales para test."""
    brain = Brain()
    # FUNCION relations
    brain.learn("corazon latir", relations=["FUNCION"])
    brain.learn("pulmon respirar", relations=["FUNCION"])
    brain.learn("cerebro pensar", relations=["FUNCION"])
    brain.learn("mano agarrar", relations=["FUNCION"])
    brain.learn("medico curar", relations=["FUNCION"])
    brain.learn("maestro enseñar", relations=["FUNCION"])

    # CAUSA relations
    brain.learn("fuego calor", relations=["CAUSA"])
    brain.learn("lluvia rio_crece", relations=["CAUSA"])
    brain.learn("sol luz", relations=["CAUSA"])
    brain.learn("viento erosion", relations=["CAUSA"])

    # PARTE_DE relations
    brain.learn("corazon cuerpo", relations=["PARTE_DE"])
    brain.learn("cabeza cuerpo", relations=["PARTE_DE"])
    brain.learn("mano brazo", relations=["PARTE_DE"])
    brain.learn("aula escuela", relations=["PARTE_DE"])
    brain.learn("sala_operaciones hospital", relations=["PARTE_DE"])

    # IS_A + HAS for context
    brain.learn("corazon organo", relations=["IS_A"])
    brain.learn("pulmon organo", relations=["IS_A"])
    brain.learn("cerebro organo", relations=["IS_A"])

    return brain


class TestExtractNeighborhood(unittest.TestCase):

    def test_neighborhood_1hop(self):
        brain = _make_brain()
        cell = brain.get_or_create_cell("corazon")
        hood = brain._extract_neighborhood(cell, hops=1)
        self.assertIn(cell.id, hood["nodes"])
        self.assertGreaterEqual(len(hood["edges"]), 2)  # corazon -> latir, corazon -> cuerpo

    def test_neighborhood_2hops(self):
        brain = _make_brain()
        cell = brain.get_or_create_cell("corazon")
        hood = brain._extract_neighborhood(cell, hops=2)
        self.assertGreater(len(hood["nodes"]), 1)
        self.assertGreater(len(hood["edges"]), 1)


class TestAnalogyStructural(unittest.TestCase):

    def test_funcion_analogy_cross_domain(self):
        """corazon:latir :: pulmon:?  →  respirar (misma FUNCION relation)"""
        brain = _make_brain()
        candidates = brain.analogy_structural("corazon", "latir", "pulmon", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "respirar")
        self.assertEqual(candidates[0]["relation"], "FUNCION")

    def test_funcion_analogy_profession(self):
        """medico:curar :: maestro:?  →  enseñar"""
        brain = _make_brain()
        candidates = brain.analogy_structural("medico", "curar", "maestro", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "enseñar")

    def test_causa_analogy(self):
        """fuego:calor :: sol:?  →  luz"""
        brain = _make_brain()
        candidates = brain.analogy_structural("fuego", "calor", "sol", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "luz")

    def test_causa_analogy_lluvia(self):
        """lluvia:rio_crece :: viento:?  →  erosion"""
        brain = _make_brain()
        candidates = brain.analogy_structural("lluvia", "rio_crece", "viento", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "erosion")

    def test_parte_de_analogy(self):
        """corazon:cuerpo :: cabeza:?  →  cuerpo (misma relacion PARTE_DE)"""
        brain = _make_brain()
        candidates = brain.analogy_structural("corazon", "cuerpo", "cabeza", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "cuerpo")

    def test_multiple_candidates_top5(self):
        """top_k=5 devuelve varios candidatos ordenados"""
        brain = _make_brain()
        candidates = brain.analogy_structural("corazon", "latir", "pulmon", top_k=5, min_sim=0.0)
        self.assertLessEqual(len(candidates), 5)
        self.assertGreaterEqual(len(candidates), 1)

    def test_unknown_concept_returns_empty(self):
        brain = _make_brain()
        candidates = brain.analogy_structural("xyz", "abc", "pulmon")
        self.assertEqual(len(candidates), 0)

    def test_no_relation_between_ab(self):
        """Si A-B no tiene relacion directa, devuelve vacio"""
        brain = _make_brain()
        candidates = brain.analogy_structural("corazon", "cuerpo", "pulmon", min_sim=0.0)
        # corazon->cuerpo is PARTE_DE, pulmon->organo is IS_A, no match
        # but pulmon->respirar is FUNCION, so the direct relation IS_A won't match PARTE_DE
        self.assertEqual(len(candidates), 0)

    def test_structural_score_fields(self):
        """Cada candidato incluye structural_score y embedding_score"""
        brain = _make_brain()
        candidates = brain.analogy_structural("corazon", "latir", "pulmon", min_sim=0.0)
        self.assertIn("structural_score", candidates[0])
        self.assertIn("embedding_score", candidates[0])
        self.assertGreater(candidates[0]["structural_score"], 0)

    def test_analogy_structural_increments_usage(self):
        """Inference usage se incrementa en sinapsis recorridas"""
        brain = _make_brain()
        syn_before = sum(s.inference_usage for s in brain.synapses.values())
        candidates = brain.analogy_structural("corazon", "latir", "pulmon", min_sim=0.0)
        syn_after = sum(s.inference_usage for s in brain.synapses.values())
        self.assertGreater(syn_after, syn_before)


class TestStructuralBenchmark(unittest.TestCase):

    def test_build_benchmark_structural(self):
        from eval.evaluate import build_benchmark_structural, STRUCTURAL_RELATIONS
        brain = _make_brain()
        analogies = build_benchmark_structural(brain)
        self.assertGreater(len(analogies), 0)
        for a in analogies:
            self.assertIn(a["relation"], STRUCTURAL_RELATIONS)

    def test_run_ma_brain_structural(self):
        from eval.evaluate import build_benchmark_structural, run_ma_brain_structural
        brain = _make_brain()
        analogies = build_benchmark_structural(brain)
        stats = run_ma_brain_structural(brain, analogies, hops=2, min_sim=0.0)
        self.assertGreater(stats["total"], 0)
        self.assertGreaterEqual(stats["accuracy_top1"], 0)
        self.assertGreaterEqual(stats["accuracy_top5"], 0)


if __name__ == "__main__":
    unittest.main()
