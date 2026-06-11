import json
import os
import tempfile
import unittest
from braincell import Brain, Braincell, Synapse


class TestBraincell(unittest.TestCase):

    def test_cell_creation(self):
        cell = Braincell()
        self.assertIsNotNone(cell.id)
        self.assertIsNone(cell.word)
        self.assertEqual(cell.energy, 0.0)
        self.assertEqual(cell.activation, 0.0)

    def test_cell_with_word(self):
        cell = Braincell(word="perro")
        self.assertEqual(cell.word, "perro")

    def test_cell_with_id(self):
        cell = Braincell(cell_id="test-123")
        self.assertEqual(cell.id, "test-123")


class TestSynapse(unittest.TestCase):

    def test_synapse_creation(self):
        a = Braincell()
        b = Braincell()
        s = Synapse(a, b, "animal")
        self.assertEqual(s.origin, a)
        self.assertEqual(s.target, b)
        self.assertEqual(s.concept, "animal")
        self.assertEqual(s.strength, 1.0)
        self.assertEqual(s.cost, 1.0)
        self.assertEqual(s.inference_usage, 0)
        self.assertEqual(s.reward, 0.0)
        self.assertEqual(s.activation_trace, 0.0)

    def test_efficiency(self):
        a = Braincell()
        b = Braincell()
        s = Synapse(a, b, "test")
        s.strength = 2.0
        s.cost = 1.0
        self.assertEqual(s.efficiency(), 0.5)

    def test_efficiency_zero_strength(self):
        a = Braincell()
        b = Braincell()
        s = Synapse(a, b, "test")
        s.strength = 0
        self.assertEqual(s.efficiency(), 999)


class TestBrainCore(unittest.TestCase):

    def test_brain_creation(self):
        brain = Brain()
        self.assertEqual(len(brain.cells), 0)
        self.assertEqual(len(brain.synapses), 0)
        self.assertEqual(len(brain.thoughts), 0)
        self.assertEqual(len(brain.concept_registry), 0)

    def test_create_cell(self):
        brain = Brain()
        cell = brain.create_cell()
        self.assertEqual(len(brain.cells), 1)
        self.assertIn(cell.id, brain.cells)

    def test_get_or_create_cell_new(self):
        brain = Brain()
        cell = brain.get_or_create_cell("perro")
        self.assertEqual(cell.word, "perro")
        self.assertIn("perro", brain.concept_registry)
        self.assertEqual(len(brain.cells), 1)

    def test_get_or_create_cell_reuses(self):
        brain = Brain()
        cell1 = brain.get_or_create_cell("perro")
        cell2 = brain.get_or_create_cell("perro")
        self.assertIs(cell1, cell2)
        self.assertEqual(len(brain.cells), 1)

    def test_connect(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        s = brain.connect(a, b, "animal")
        self.assertEqual(len(brain.synapses), 1)
        self.assertIn(s.id, brain.synapses)
        self.assertIn(s, a.synapses_out)
        self.assertIn(s, b.synapses_in)

    def test_connect_dedup(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        s1 = brain.connect(a, b, "animal")
        s2 = brain.connect(a, b, "animal")
        self.assertIs(s1, s2)
        self.assertEqual(len(brain.synapses), 1)


class TestLearnAndThink(unittest.TestCase):

    def test_learn_creates_graph(self):
        brain = Brain()
        brain.learn("ser vivo animal perro")
        self.assertGreaterEqual(len(brain.cells), 2)

    def test_learn_reuses_concepts(self):
        brain = Brain()
        brain.learn("ser vivo animal perro")
        brain.learn("ser vivo animal gato")
        self.assertEqual(len(brain.cells), 5)

    def test_think_returns_string(self):
        brain = Brain()
        brain.learn("ser vivo animal perro")
        start = brain.get_or_create_cell("ser")
        thought = brain.think(start, steps=5, temperature=0)
        self.assertIsInstance(thought, str)
        self.assertGreater(len(thought), 0)

    def test_think_greedy_vs_probabilistic(self):
        brain = Brain()
        brain.learn("a b c")
        brain.learn("a b d")
        start = brain.get_or_create_cell("a")
        out_greedy = brain.think(start, steps=3, temperature=0)
        self.assertEqual(out_greedy, "b c")
        out_random = brain.think(start, steps=3, temperature=5.0, epsilon=0.5)
        self.assertIn(out_random, ["b c", "b d"])


class TestRewardThought(unittest.TestCase):

    def test_reward_strengthens(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        old_strength = s.strength
        s.activation_trace = 1.0
        brain.reward_thought(1.0)
        self.assertGreater(s.strength, old_strength)
        self.assertGreater(s.reward, 0)
        self.assertLess(s.cost, 1.0)

    def test_reward_clears_trace(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        s.activation_trace = 1.0
        brain.reward_thought(1.0)
        self.assertEqual(s.activation_trace, 0)

    def test_reward_empty_trace_noop(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        old_strength = s.strength
        brain.reward_thought(1.0)
        self.assertEqual(s.strength, old_strength)


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.mktemp(suffix=".json")

    def tearDown(self):
        if os.path.exists(self.tmpfile):
            os.remove(self.tmpfile)

    def test_save_load_roundtrip(self):
        brain = Brain()
        brain.learn("ser vivo animal perro ladra")
        start = brain.get_or_create_cell("ser")
        brain.think(start, steps=3, temperature=0)

        brain.save(self.tmpfile)

        brain2 = Brain()
        brain2.load(self.tmpfile)

        self.assertEqual(len(brain2.cells), len(brain.cells))
        self.assertEqual(len(brain2.synapses), len(brain.synapses))
        self.assertEqual(len(brain2.thoughts), len(brain.thoughts))
        self.assertEqual(len(brain2.concept_registry), len(brain.concept_registry))

    def test_load_rebuilds_registry(self):
        brain = Brain()
        brain.learn("hola mundo")
        brain.save(self.tmpfile)

        brain2 = Brain()
        brain2.load(self.tmpfile)
        cell = brain2.get_or_create_cell("hola")
        self.assertIsNotNone(cell)
        self.assertEqual(cell.word, "hola")

    def test_load_missing_file(self):
        brain = Brain()
        brain.load("nonexistent_file_12345.json")
        self.assertEqual(len(brain.cells), 0)


class TestRelation(unittest.TestCase):

    def test_synapse_with_relation(self):
        a = Braincell()
        b = Braincell()
        s = Synapse(a, b, "ladra", relation="SOUND_OF")
        self.assertEqual(s.relation, "SOUND_OF")

    def test_connect_with_relation(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        s = brain.connect(a, b, "ladra", relation="SOUND_OF")
        self.assertEqual(s.relation, "SOUND_OF")

    def test_connect_dedup_with_relation(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        s1 = brain.connect(a, b, "ladra", relation="SOUND_OF")
        s2 = brain.connect(a, b, "ladra", relation="SOUND_OF")
        self.assertIs(s1, s2)

    def test_connect_different_relation_no_dedup(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        s1 = brain.connect(a, b, "ladra", relation="SOUND_OF")
        s2 = brain.connect(a, b, "ladra", relation="ACTION")
        self.assertIsNot(s1, s2)
        self.assertEqual(len(brain.synapses), 2)

    def test_query_relation(self):
        brain = Brain()
        brain.learn("a b", relations=["NEXT"])
        rel = brain.query_relation("a", "b")
        self.assertEqual(rel, "NEXT")

    def test_query_relation_missing(self):
        brain = Brain()
        brain.learn("a b")
        rel = brain.query_relation("a", "c")
        self.assertIsNone(rel)

    def test_find_by_relation(self):
        brain = Brain()
        brain.learn("a b", relations=["SOUND_OF"])
        brain.learn("a c", relations=["SOUND_OF"])
        candidates = brain.find_by_relation("a", "SOUND_OF")
        self.assertEqual(len(candidates), 2)
        words = [c[0] for c in candidates]
        self.assertIn("b", words)
        self.assertIn("c", words)

    def test_analogy(self):
        brain = Brain()
        brain.learn("perro ladra", relations=["SOUND_OF"])
        brain.learn("gato maulla", relations=["SOUND_OF"])
        brain.learn("perro corre", relations=["ACTION"])
        candidates = brain.analogy("perro", "ladra", "gato")
        self.assertGreater(len(candidates), 0)
        best = candidates[0]["d"]
        self.assertEqual(best, "maulla")
        self.assertIn("trace", candidates[0])
        self.assertIn("score", candidates[0])
        self.assertIn("relation", candidates[0])

    def test_analogy_multi_relation(self):
        brain = Brain()
        brain.learn("perro ladra", relations=["SOUND_OF"])
        brain.learn("gato maulla", relations=["SOUND_OF"])
        brain.learn("gato ronronea", relations=["SOUND_OF"])
        candidates = brain.analogy("perro", "ladra", "gato")
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["d"], "maulla")

    def test_analogy_no_match(self):
        brain = Brain()
        brain.learn("perro ladra", relations=["SOUND_OF"])
        brain.learn("gato come", relations=["ACTION"])
        candidates = brain.analogy("perro", "ladra", "gato")
        self.assertEqual(len(candidates), 0)


class TestPrune(unittest.TestCase):

    def test_prune_unused_synapses(self):
        brain = Brain()
        brain.learn("a b c")
        self.assertGreater(len(brain.synapses), 0)
        n = brain.prune(min_usage=1)
        self.assertGreater(n, 0)
        self.assertEqual(len(brain.synapses), 0)

    def test_prune_keeps_used_synapses(self):
        brain = Brain()
        brain.learn("a b c")
        total = len(brain.synapses)
        s = list(brain.synapses.values())[0]
        s.inference_usage = 5
        n = brain.prune(min_usage=1)
        self.assertEqual(n, total - 1)
        self.assertEqual(len(brain.synapses), 1)
        self.assertIn(s.id, brain.synapses)


class TestWorkingMemory(unittest.TestCase):

    def test_think_populates_working_memory(self):
        brain = Brain()
        brain.learn("a b c d")
        start = brain.get_or_create_cell("a")
        brain.think(start, steps=3, temperature=0, track_in_wm=True)
        active = brain.wm.get_active_entities(min_salience=0.01)
        self.assertGreater(len(active), 0)

    def test_working_memory_has_concepts(self):
        brain = Brain()
        brain.learn("a b c")
        start = brain.get_or_create_cell("a")
        brain.think(start, steps=3, temperature=0, track_in_wm=True)
        active = brain.wm.get_active_entities(min_salience=0.01)
        self.assertIn("b", active)
        self.assertIn("c", active)


class TestLearnWithRelations(unittest.TestCase):

    def test_learn_with_relations(self):
        brain = Brain()
        brain.learn("a b c", relations=["NEXT", "NEXT"])
        rel = brain.query_relation("a", "b")
        self.assertEqual(rel, "NEXT")

    def test_learn_partial_relations(self):
        brain = Brain()
        brain.learn("a b c d", relations=["NEXT"])
        rel_ab = brain.query_relation("a", "b")
        rel_bc = brain.query_relation("b", "c")
        self.assertEqual(rel_ab, "NEXT")
        self.assertIsNone(rel_bc)


class TestPunishThought(unittest.TestCase):

    def test_punish_reduces_strength(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        old_strength = s.strength
        s.activation_trace = 1.0
        brain.punish_thought(1.0)
        self.assertLess(s.strength, old_strength)
        self.assertLess(s.reward, 0)

    def test_punish_increases_cost(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        old_cost = s.cost
        s.activation_trace = 1.0
        brain.punish_thought(1.0)
        self.assertGreater(s.cost, old_cost)

    def test_punish_clears_trace(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        s.activation_trace = 1.0
        brain.punish_thought(1.0)
        self.assertEqual(s.activation_trace, 0)

    def test_punish_noop_without_trace(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        old_strength = s.strength
        brain.punish_thought(1.0)
        self.assertEqual(s.strength, old_strength)

    def test_punish_reduces_cell_energy(self):
        brain = Brain()
        brain.learn("a b")
        s = list(brain.synapses.values())[0]
        cell = s.target
        old_energy = cell.energy
        cell.thought_trace = 1.0
        brain.punish_thought(1.0)
        self.assertLess(cell.energy, old_energy)


class TestDynamicTemperature(unittest.TestCase):

    def test_single_option_returns_zero(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        s = brain.connect(a, b, "test")
        temp = brain._dynamic_temperature([s])
        self.assertEqual(temp, 0.0)

    def test_equal_options_returns_base_temp(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        c = brain.create_cell()
        s1 = brain.connect(a, b, "x")
        s2 = brain.connect(a, c, "y")
        s1.strength = 1.0
        s2.strength = 1.0
        temp = brain._dynamic_temperature([s1, s2], base_temp=2.0)
        self.assertAlmostEqual(temp, 2.0, places=4)

    def test_dominant_option_low_temperature(self):
        brain = Brain()
        a = brain.create_cell()
        b = brain.create_cell()
        c = brain.create_cell()
        s1 = brain.connect(a, b, "x")
        s2 = brain.connect(a, c, "y")
        s1.strength = 100.0
        s2.strength = 1.0
        temp = brain._dynamic_temperature([s1, s2], base_temp=1.0)
        self.assertLess(temp, 0.5)

    def test_think_with_none_temperature_no_error(self):
        brain = Brain()
        brain.learn("a b c")
        start = brain.get_or_create_cell("a")
        thought = brain.think(start, steps=2, temperature=None)
        self.assertIsInstance(thought, str)
        self.assertGreater(len(thought), 0)


class TestPredictNext(unittest.TestCase):

    def test_predict_basic(self):
        brain = Brain()
        brain.learn("a b c")
        preds = brain.predict_next("a")
        self.assertGreater(len(preds), 0)
        self.assertEqual(preds[0][0], "b")

    def test_predict_returns_probabilities(self):
        brain = Brain()
        brain.learn("a b c")
        brain.learn("a d e")
        preds = brain.predict_next("a", top_k=10)
        self.assertEqual(len(preds), 2)
        total_p = sum(p for _, p in preds)
        self.assertAlmostEqual(total_p, 1.0, places=5)

    def test_predict_unknown_context(self):
        brain = Brain()
        brain.learn("a b c")
        preds = brain.predict_next("x")
        self.assertEqual(len(preds), 0)

    def test_predict_empty_context(self):
        brain = Brain()
        preds = brain.predict_next("")
        self.assertEqual(len(preds), 0)

    def test_predict_top_k(self):
        brain = Brain()
        brain.learn("a b c d e f")
        brain.learn("a g h i j")
        preds = brain.predict_next("a", top_k=2)
        self.assertLessEqual(len(preds), 2)

    def test_predict_multi_word_context(self):
        brain = Brain()
        brain.learn("a b c")
        preds = brain.predict_next("x y z a b")
        self.assertEqual(preds[0][0], "c")


class TestEmbeddingBridge(unittest.TestCase):

    def test_char_similarity_identical(self):
        from embedding_bridge import EmbeddingBridge
        b = EmbeddingBridge()
        self.assertEqual(b.similarity("perro", "perro"), 1.0)

    def test_char_similarity_morphological(self):
        from embedding_bridge import EmbeddingBridge
        b = EmbeddingBridge()
        sim = b.similarity("corriendo", "corre")
        self.assertGreater(sim, 0.1)

    def test_char_similarity_different(self):
        from embedding_bridge import EmbeddingBridge
        b = EmbeddingBridge()
        sim = b.similarity("a", "b")
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_closest_returns_sorted(self):
        from embedding_bridge import EmbeddingBridge
        b = EmbeddingBridge()
        candidates = ["hola", "ola", "casa"]
        matches = b.closest("ola", candidates, min_score=0.1)
        self.assertGreater(matches[0][1], matches[1][1])

    def test_closest_empty_candidates(self):
        from embedding_bridge import EmbeddingBridge
        b = EmbeddingBridge()
        self.assertEqual(b.closest("test", []), [])

    def test_closest_cell_word_exact(self):
        from braincell import Brain
        brain = Brain()
        brain.get_or_create_cell("perro")
        match = brain.embedding_bridge.closest_cell_word("perro", brain)
        self.assertIsNotNone(match)
        self.assertEqual(match[0], "perro")

    def test_closest_cell_word_fuzzy(self):
        from braincell import Brain
        brain = Brain()
        brain.get_or_create_cell("perro")
        match = brain.embedding_bridge.closest_cell_word("pero", brain, min_score=0.1)
        self.assertIsNotNone(match)


class TestFuzzyMatching(unittest.TestCase):

    def test_get_or_create_cell_fuzzy_reuses(self):
        brain = Brain()
        orig = brain.get_or_create_cell("corriendo")
        fuzzy = brain.get_or_create_cell("corre", fuzzy=True)
        self.assertIs(fuzzy, orig)

    def test_get_or_create_cell_fuzzy_no_match_creates_new(self):
        brain = Brain()
        brain.get_or_create_cell("perro")
        cell = brain.get_or_create_cell("sol", fuzzy=True)
        self.assertIsNotNone(cell)
        self.assertEqual(cell.word, "sol")

    def test_analogy_finds_fuzzy_match(self):
        brain = Brain()
        brain.learn("perro ladra", relations=["SOUND"])
        brain.learn("gato maulla", relations=["SOUND"])
        results = brain.analogy("perro", "ladra", "gat")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["d"], "maulla")

    def test_predict_next_fuzzy(self):
        brain = Brain()
        brain.learn("a b c")
        preds = brain.predict_next("hola a b")
        self.assertGreater(len(preds), 0)
        self.assertEqual(preds[0][0], "c")

    def test_query_relation_fuzzy(self):
        brain = Brain()
        brain.learn("a b", relations=["NEXT"])
        rel = brain.query_relation("a", "b")
        self.assertEqual(rel, "NEXT")

    def test_find_by_relation_fuzzy(self):
        brain = Brain()
        brain.learn("a b", relations=["SOUND_OF"])
        brain.learn("a c", relations=["SOUND_OF"])
        candidates = brain.find_by_relation("a", "SOUND_OF")
        self.assertEqual(len(candidates), 2)


class TestEmbeddingScoring(unittest.TestCase):

    def test_analogy_embedding_weight_boosts(self):
        brain = Brain(embedding_weight=0.5)
        brain.learn("dog bark", relations=["SOUND"])
        brain.learn("cat meow", relations=["SOUND"])
        brain.learn("cat purr", relations=["SOUND"])
        results = brain.analogy("dog", "bark", "cat")
        self.assertGreater(len(results), 0)
        self.assertIn(results[0]["d"], ["meow", "purr"])
        self.assertGreater(results[0]["embedding_score"], 0)

    def test_analogy_embedding_prefers_semantic_match(self):
        brain = Brain(embedding_weight=0.5)
        brain.learn("dog bark", relations=["SOUND"])
        brain.learn("dog animal", relations=["IS_A"])
        brain.learn("cat meow", relations=["SOUND"])
        brain.learn("cat feline", relations=["IS_A"])
        results = brain.analogy("dog", "bark", "cat")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["d"], "meow")
        self.assertGreater(results[0]["embedding_score"], 0)

    def test_think_prefers_semantically_close(self):
        brain = Brain(embedding_weight=0.5)
        brain.learn("perro ladra")
        brain.learn("perro animal")
        brain.learn("gato maulla")
        start = brain.get_or_create_cell("gato")
        thought = brain.think(start, steps=2, temperature=0.5)
        self.assertEqual(thought, "maulla")

    def test_predict_embedding_boost(self):
        brain = Brain(embedding_weight=0.3)
        brain.learn("sol luz calor")
        brain.learn("sol energia vida")
        preds = brain.predict_next("sol")
        words = [p[0] for p in preds]
        self.assertIn("luz", words)
        self.assertIn("energia", words)

    def test_synapse_score_increases_for_similar(self):
        brain = Brain(embedding_weight=0.5)
        brain.learn("perro ladra")
        brain.learn("perro animal")
        cell = brain.get_or_create_cell("perro")
        score_normal = brain._synapse_score(cell.synapses_out[0])
        score_context = brain._synapse_score(cell.synapses_out[0], context="perro")
        self.assertGreaterEqual(score_context, score_normal)

    def test_embedding_weight_zero_by_default(self):
        brain = Brain()
        self.assertEqual(brain.embedding_weight, 0.0)


if __name__ == "__main__":
    unittest.main()
