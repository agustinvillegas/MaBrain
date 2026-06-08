import unittest
from braincell import Brain, Braincell


def _make_brain():
    """Crea un brain pequeño con relaciones estructurales para test."""
    brain = Brain()
    # FUNCTION relations
    brain.learn("heart pump", relations=["FUNCTION"])
    brain.learn("lung breathe", relations=["FUNCTION"])
    brain.learn("brain think", relations=["FUNCTION"])
    brain.learn("hand grasp", relations=["FUNCTION"])
    brain.learn("doctor heal", relations=["FUNCTION"])
    brain.learn("teacher teach", relations=["FUNCTION"])

    # CAUSE relations
    brain.learn("fire heat", relations=["CAUSE"])
    brain.learn("rain grow", relations=["CAUSE"])
    brain.learn("sun light", relations=["CAUSE"])
    brain.learn("wind erosion", relations=["CAUSE"])

    # PART_OF relations
    brain.learn("heart body", relations=["PART_OF"])
    brain.learn("head body", relations=["PART_OF"])
    brain.learn("hand arm", relations=["PART_OF"])
    brain.learn("classroom school", relations=["PART_OF"])
    brain.learn("operating_room hospital", relations=["PART_OF"])

    # IS_A + HAS for context
    brain.learn("heart organ", relations=["IS_A"])
    brain.learn("lung organ", relations=["IS_A"])
    brain.learn("brain organ", relations=["IS_A"])

    return brain


class TestExtractNeighborhood(unittest.TestCase):

    def test_neighborhood_1hop(self):
        brain = _make_brain()
        cell = brain.get_or_create_cell("heart")
        hood = brain._extract_neighborhood(cell, hops=1)
        self.assertIn(cell.id, hood["nodes"])
        self.assertGreaterEqual(len(hood["edges"]), 2)  # heart->pump, heart->body

    def test_neighborhood_2hops(self):
        brain = _make_brain()
        cell = brain.get_or_create_cell("heart")
        hood = brain._extract_neighborhood(cell, hops=2)
        self.assertGreater(len(hood["nodes"]), 1)
        self.assertGreater(len(hood["edges"]), 1)


class TestAnalogyStructural(unittest.TestCase):

    def test_funcion_analogy_cross_domain(self):
        """heart:pump :: lung:?  →  breathe (same FUNCTION relation)"""
        brain = _make_brain()
        candidates = brain.analogy_structural("heart", "pump", "lung", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "breathe")
        self.assertEqual(candidates[0]["relation"], "FUNCTION")

    def test_funcion_analogy_profession(self):
        """doctor:heal :: teacher:?  →  teach"""
        brain = _make_brain()
        candidates = brain.analogy_structural("doctor", "heal", "teacher", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "teach")

    def test_causa_analogy(self):
        """fire:heat :: sun:?  →  light"""
        brain = _make_brain()
        candidates = brain.analogy_structural("fire", "heat", "sun", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "light")

    def test_causa_analogy_lluvia(self):
        """rain:grow :: wind:?  →  erosion"""
        brain = _make_brain()
        candidates = brain.analogy_structural("rain", "grow", "wind", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "erosion")

    def test_parte_de_analogy(self):
        """heart:body :: head:?  →  body (same PART_OF relation)"""
        brain = _make_brain()
        candidates = brain.analogy_structural("heart", "body", "head", min_sim=0.0)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "body")

    def test_multiple_candidates_top5(self):
        """top_k=5 returns several ordered candidates"""
        brain = _make_brain()
        candidates = brain.analogy_structural("heart", "pump", "lung", top_k=5, min_sim=0.0)
        self.assertLessEqual(len(candidates), 5)
        self.assertGreaterEqual(len(candidates), 1)

    def test_unknown_concept_returns_empty(self):
        brain = _make_brain()
        candidates = brain.analogy_structural("xyz", "abc", "lung")
        self.assertEqual(len(candidates), 0)

    def test_no_relation_between_ab(self):
        """If A-B has no direct relation, returns empty"""
        brain = _make_brain()
        candidates = brain.analogy_structural("heart", "body", "lung", min_sim=0.0)
        # heart->body is PART_OF, lung->organ is IS_A, no match
        self.assertEqual(len(candidates), 0)

    def test_structural_score_fields(self):
        """Each candidate includes structural_score and embedding_score"""
        brain = _make_brain()
        candidates = brain.analogy_structural("heart", "pump", "lung", min_sim=0.0)
        self.assertIn("structural_score", candidates[0])
        self.assertIn("embedding_score", candidates[0])
        self.assertGreater(candidates[0]["structural_score"], 0)

    def test_analogy_structural_increments_usage(self):
        """Inference usage incremented on traversed synapses"""
        brain = _make_brain()
        syn_before = sum(s.inference_usage for s in brain.synapses.values())
        candidates = brain.analogy_structural("heart", "pump", "lung", min_sim=0.0)
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
