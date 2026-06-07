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
        self.assertEqual(s.usage, 0)
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
        best = candidates[0][0]
        self.assertEqual(best, "maulla")


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
        s.usage = 5
        n = brain.prune(min_usage=1)
        self.assertEqual(n, total - 1)
        self.assertEqual(len(brain.synapses), 1)
        self.assertIn(s.id, brain.synapses)


class TestWorkingMemory(unittest.TestCase):

    def test_think_clears_and_populates_working_memory(self):
        brain = Brain()
        brain.learn("a b c d")
        start = brain.get_or_create_cell("a")
        brain.think(start, steps=3, temperature=0, use_working_memory=True)
        self.assertGreater(len(brain.working_memory), 0)

    def test_working_memory_has_concepts(self):
        brain = Brain()
        brain.learn("a b c")
        start = brain.get_or_create_cell("a")
        brain.think(start, steps=3, temperature=0, use_working_memory=True)
        self.assertIn("b", brain.working_memory)
        self.assertIn("c", brain.working_memory)


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


if __name__ == "__main__":
    unittest.main()
