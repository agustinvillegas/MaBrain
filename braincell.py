import json
import math
import random
import uuid
from embedding_bridge import EmbeddingBridge
from working_memory import WorkingMemory


class Braincell:

    def __init__(self, cell_id=None, word=None):
        self.id = cell_id or str(uuid.uuid4())
        self.word = word

        self.energy = 0.0
        self.activation = 0.0

        # nueva memoria de participación
        self.thought_trace = 0.0

        self.synapses_out = []
        self.synapses_in = []


class Synapse:

    def __init__(self, origin, target, concept, relation=None):

        self.id = str(uuid.uuid4())

        self.origin = origin
        self.target = target
        self.concept = concept
        self.relation = relation

        self.strength = 1.0
        self.cost = 1.0

        self.inference_usage = 0
        self.reward = 0.0

        # nueva
        self.activation_trace = 0.0


    def efficiency(self):

        if self.strength <= 0:
            return 999

        return self.cost / self.strength



class Brain:


    def __init__(self, embedding_bridge=None, embedding_weight=0.0, structural_weight=0.0,
                 context_bias_weight=0.0):

        self.cells = {}
        self.synapses = {}
        self.concept_registry = {}
        self.wm = WorkingMemory()
        self.thoughts = []
        self.embedding_bridge = embedding_bridge or EmbeddingBridge()
        self.embedding_weight = embedding_weight
        self.structural_weight = structural_weight
        self.context_bias_weight = context_bias_weight


    def _resolve_concept(self, concept, min_score=0.1):

        cell_id = self.concept_registry.get(concept)
        if cell_id:
            return cell_id

        match = self.embedding_bridge.closest_cell_word(concept, self, min_score=min_score)
        if match:
            matched_concept, score = match
            if self.embedding_bridge.model_loaded or score >= 0.1:
                return self.concept_registry.get(matched_concept)

        return None


    def create_cell(self):

        cell = Braincell()

        self.cells[cell.id] = cell

        return cell


    def get_or_create_cell(self, concept, fuzzy=False):

        if concept in self.concept_registry:
            cell_id = self.concept_registry[concept]
            return self.cells[cell_id]

        if fuzzy:
            match_id = self._resolve_concept(concept)
            if match_id:
                return self.cells[match_id]

        cell = Braincell(word=concept)
        self.cells[cell.id] = cell
        self.concept_registry[concept] = cell.id

        return cell


    def reward_thought(self, score):

        for syn in self.synapses.values():

            if syn.activation_trace > 0:

            # recompensa proporcional al aporte
                gain = syn.activation_trace * score

                syn.reward += gain
                syn.strength += gain * 0.1

            # abarata caminos útiles
                syn.cost *= (1 - min(gain * 0.01, 0.1))

            # limpiar memoria temporal
                syn.activation_trace = 0


        for cell in self.cells.values():

            if cell.thought_trace > 0:

                cell.energy += cell.thought_trace * score

                cell.thought_trace = 0


    def punish_thought(self, score):

        for syn in self.synapses.values():

            if syn.activation_trace > 0:

                penalty = syn.activation_trace * score

                syn.reward -= penalty
                syn.strength = max(syn.strength - penalty * 0.1, 0.01)
                syn.cost *= (1 + min(penalty * 0.01, 0.2))

                syn.activation_trace = 0


        for cell in self.cells.values():

            if cell.thought_trace > 0:

                cell.energy -= cell.thought_trace * score

                cell.thought_trace = 0


    def connect(self, a, b, concept, relation=None):

        # evitar duplicados del mismo concepto
        for syn in a.synapses_out:

            if (
                syn.target == b and
                syn.concept == concept and
                syn.relation == relation
            ):
                return syn


        syn = Synapse(
            a,
            b,
            concept,
            relation=relation
        )

        self.synapses[syn.id] = syn


        a.synapses_out.append(syn)
        b.synapses_in.append(syn)


        return syn


    def learn(self, text, relations=None):

        words = text.split()

        if len(words) < 2:
            return


        previous = None
        rel_idx = 0


        for word in words:

            cell = self.get_or_create_cell(word)


            if previous:

                rel = relations[rel_idx] if relations and rel_idx < len(relations) else None

                self.connect(
                    previous,
                    cell,
                    word,
                    relation=rel
                )

                rel_idx += 1


            previous = cell



    def _dynamic_temperature(self, options, base_temp=1.0):

        if len(options) <= 1:
            return 0.0

        strengths = [s.strength for s in options]
        total = sum(strengths)
        if total <= 0:
            return base_temp

        probs = [s / total for s in strengths]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(options))

        if max_entropy <= 0:
            return 0.0

        return (entropy / max_entropy) * base_temp


    def _select_synapse(self, options, temperature=1.0, epsilon=0.0, context=None):

        if temperature is None:
            temperature = self._dynamic_temperature(options)

        if temperature <= 0:
            return max(options, key=lambda s: self._synapse_score(s, context))

        if random.random() < epsilon:
            return random.choice(options)

        scores = [self._synapse_score(s, context) for s in options]
        max_score = max(scores)
        weights = [math.exp((sc - max_score) / temperature) for sc in scores]

        total = sum(weights)
        if total <= 0:
            return random.choice(options)

        r = random.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return options[i]
        return options[-1]

    def _context_bias(self, synapse):
        """Bias contextual de WM: entidades activas + topic stack.
        Retorna ≥ 0. Escalado por context_bias_weight."""
        if self.context_bias_weight <= 0 or not synapse.target.word:
            return 0.0
        bias = 0.0
        wm_salience = self.wm.active_entities.get(synapse.target.word, 0.0)
        bias += wm_salience
        for i, topic in enumerate(reversed(self.wm.topic_stack)):
            if topic == synapse.target.word:
                bias += 1.0 / (i + 1)
                break
        return self.context_bias_weight * bias

    def _synapse_score(self, synapse, context=None):
        """Score combinado: estructural + embedding + contexto (mayor = mejor)."""
        structural = synapse.strength / max(synapse.cost, 0.001)
        emb = 0.0
        if context and self.embedding_weight > 0 and synapse.target.word:
            emb = self.embedding_weight * self.embedding_bridge.similarity(synapse.target.word, context)
        ctx = self._context_bias(synapse)
        return structural + emb + ctx


    def think(self, start_cell, steps=10, temperature=None, epsilon=0.0, track_in_wm=False):

        current = start_cell
        result = []

        for i in range(steps):

            if len(current.synapses_out) == 0:
                break

            options = current.synapses_out

            choice = self._select_synapse(
                options,
                temperature=temperature,
                epsilon=epsilon,
                context=current.word
            )

            result.append(choice.concept)

            choice.inference_usage += 1
            choice.activation_trace += choice.strength
            choice.target.activation += choice.strength
            choice.target.thought_trace += choice.strength
            current = choice.target

            if track_in_wm and current.word:
                self.wm.activate_entity(current.word, salience=0.5)

        thought = " ".join(result)

        self.thoughts.append(thought)

        return thought


    def query_relation(self, concept_a, concept_b):

        cell_a_id = self.concept_registry.get(concept_a) or self._resolve_concept(concept_a)
        cell_b_id = self.concept_registry.get(concept_b) or self._resolve_concept(concept_b)

        if not cell_a_id or not cell_b_id:
            return None

        cell_a = self.cells[cell_a_id]

        for syn in cell_a.synapses_out:
            if syn.target.id == cell_b_id:
                return syn.relation

        return None


    def find_by_relation(self, concept, relation):

        cell_id = self.concept_registry.get(concept) or self._resolve_concept(concept)

        if not cell_id:
            return []

        cell = self.cells[cell_id]
        results = []

        for syn in cell.synapses_out:
            if syn.relation == relation:
                target_word = syn.target.word
                if target_word:
                    results.append((target_word, syn.strength, syn.cost))

        return sorted(results, key=lambda x: (x[2], -x[1]))


    def analogy(self, a, b, c, top_k=5):

        results = []

        cell_a_id = self.concept_registry.get(a) or self._resolve_concept(a)
        cell_b_id = self.concept_registry.get(b) or self._resolve_concept(b)
        cell_c_id = self.concept_registry.get(c) or self._resolve_concept(c)

        if not cell_a_id or not cell_b_id or not cell_c_id:
            return results

        cell_a = self.cells[cell_a_id]
        cell_c = self.cells[cell_c_id]

        relations_ab = set()

        for syn in cell_a.synapses_out:
            if syn.target.id == cell_b_id:
                relations_ab.add(syn.relation)
                syn.inference_usage += 1

        for relation in relations_ab:

            for csyn in cell_c.synapses_out:
                if csyn.relation != relation:
                    continue
                if not csyn.target.word:
                    continue
                csyn.inference_usage += 1
                score = csyn.strength / max(csyn.cost, 0.001)
                emb_score = 0.0
                if self.embedding_weight > 0 and csyn.target.word:
                    sim = self.embedding_bridge.similarity(csyn.target.word, b)
                    emb_score = self.embedding_weight * sim
                    score += emb_score
                results.append({
                    "d": csyn.target.word,
                    "relation": relation,
                    "score": score,
                    "embedding_score": emb_score,
                    "strength": csyn.strength,
                    "cost": csyn.cost,
                    "trace": {
                        "a_b": {"from": a, "to": b, "relation": relation},
                        "c_d": {"from": c, "to": csyn.target.word, "relation": relation}
                    }
                })

        results.sort(key=lambda x: (-x["score"], x["cost"]))
        return results[:top_k]


    def analogy_structural(self, a, b, c, top_k=5, hops=2, min_sim=0.1):

        results = []

        cell_a_id = self.concept_registry.get(a) or self._resolve_concept(a)
        cell_b_id = self.concept_registry.get(b) or self._resolve_concept(b)
        cell_c_id = self.concept_registry.get(c) or self._resolve_concept(c)

        if not cell_a_id or not cell_b_id or not cell_c_id:
            return results

        cell_a = self.cells[cell_a_id]
        cell_b = self.cells[cell_b_id]
        cell_c = self.cells[cell_c_id]

        # 1. Find direct relations from A -> B
        ab_rels = set()
        for syn in cell_a.synapses_out:
            if syn.target.id == cell_b_id:
                ab_rels.add(syn.relation)
                syn.inference_usage += 1

        if not ab_rels:
            return results

        # 2. Pre-compute outgoing relation patterns for A, B, C (don't depend on D)
        def _out_pattern(cell):
            pat = {}
            for syn in cell.synapses_out:
                if syn.relation:
                    pat[syn.relation] = pat.get(syn.relation, 0) + 1
            return pat

        a_pattern = _out_pattern(cell_a)
        b_pattern = _out_pattern(cell_b)
        c_pattern = _out_pattern(cell_c)

        match_ac = sum(1 for r in a_pattern if r in c_pattern)
        total_ac_patterns = max(len(a_pattern) + len(c_pattern), 1)

        # 3. For each candidate D reachable from C with matching relation
        for csyn in cell_c.synapses_out:
            if csyn.relation not in ab_rels:
                continue
            if not csyn.target.word:
                continue

            d_cell = csyn.target
            csyn.inference_usage += 1

            # 3a. B vs D outgoing pattern overlap
            d_pattern = _out_pattern(d_cell)
            match_bd = sum(1 for r in b_pattern if r in d_pattern)
            total_bd_patterns = max(len(b_pattern) + len(d_pattern), 1)
            structural_score = (match_ac / total_ac_patterns + match_bd / total_bd_patterns) / 2.0

            # 3b. Embedding similarity
            emb_b_d = 0.0
            emb_a_c = 0.0
            if self.embedding_bridge is not None:
                emb_b_d = self.embedding_bridge.similarity(b, d_cell.word) if b and d_cell.word else 0.0
                emb_a_c = self.embedding_bridge.similarity(a, c) if a and c else 0.0
            emb_score = (emb_b_d + emb_a_c) / 2.0

            if emb_score < min_sim:
                continue

            # 3c. Final score: weighted combination
            score = structural_score * 0.5 + emb_score * 0.3
            score += (csyn.strength / max(csyn.cost, 0.001)) * 0.2

            results.append({
                "d": d_cell.word,
                "relation": csyn.relation,
                "score": score,
                "structural_score": structural_score,
                "embedding_score": emb_score,
                "strength": csyn.strength,
                "cost": csyn.cost,
                "trace": {
                    "a_b": {"from": a, "to": b, "relation": csyn.relation},
                    "c_d": {"from": c, "to": d_cell.word, "relation": csyn.relation},
                    "a_pattern": list(a_pattern.keys()),
                    "c_pattern": list(c_pattern.keys()),
                    "b_pattern": list(b_pattern.keys()),
                    "d_pattern": list(d_pattern.keys()),
                }
            })

        results.sort(key=lambda x: (-x["score"]))
        return results[:top_k]


    def _extract_neighborhood(self, cell, hops=2):

        nodes = {cell.id}
        edges = []
        frontier = {cell}

        for _ in range(hops):
            next_frontier = set()
            for c in frontier:
                for syn in c.synapses_out:
                    if syn.target.id not in nodes:
                        nodes.add(syn.target.id)
                        next_frontier.add(syn.target)
                    if syn.relation:
                        edges.append({
                            "from": syn.origin.word,
                            "relation": syn.relation,
                            "to": syn.target.word,
                            "strength": syn.strength,
                        })
                for syn in c.synapses_in:
                    if syn.origin.id not in nodes:
                        nodes.add(syn.origin.id)
                        next_frontier.add(syn.origin)
            frontier = next_frontier

        return {"nodes": nodes, "edges": edges}


    def prune(self, min_usage=1):

        to_remove = []

        for sid, syn in self.synapses.items():
            if syn.inference_usage < min_usage:
                to_remove.append(sid)

        for sid in to_remove:
            syn = self.synapses[sid]

            if syn in syn.origin.synapses_out:
                syn.origin.synapses_out.remove(syn)
            if syn in syn.target.synapses_in:
                syn.target.synapses_in.remove(syn)

            del self.synapses[sid]

        return len(to_remove)


    def predict_next(self, context, top_k=5, temperature=1.0):

        words = context.split()
        if not words:
            return []

        last_word = words[-1]
        cell_id = self.concept_registry.get(last_word) or self._resolve_concept(last_word)

        if not cell_id:
            return []

        cell = self.cells[cell_id]
        if not cell.synapses_out:
            return []

        scores = []
        for s in cell.synapses_out:
            sc = self._synapse_score(s, context=last_word)
            scores.append(sc / temperature)
        # Numerically stable softmax: subtract max to avoid overflow
        max_score = max(scores) if scores else 0
        scores = [math.exp(s - max_score) for s in scores]
        total = sum(scores)

        if total <= 0:
            return [(s.concept, 0.0) for s in cell.synapses_out[:top_k]]

        probs = [s / total for s in scores]
        results = [
            (syn.concept, prob)
            for syn, prob in zip(cell.synapses_out, probs)
        ]
        results.sort(key=lambda x: -x[1])

        return results[:top_k]


    def auto_infer_relations(self):

        for syn in self.synapses.values():
            if syn.relation is not None:
                continue
            outgoing = len(syn.origin.synapses_out)
            incoming = len(syn.target.synapses_in)
            outgoing_target = len(syn.target.synapses_out)
            if outgoing >= 3:
                syn.relation = "IS_A"
            elif outgoing_target == 0:
                syn.relation = "PROPERTY"
            else:
                syn.relation = "NEXT"


    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "can",
        "could", "shall", "should", "may", "might", "must", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "about",
        "up", "down", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "some", "any",
        "no", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "also", "but", "and", "or", "if", "while", "what",
        "which", "who", "whom", "this", "that", "these", "those", "it", "its",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their", "tell", "ask", "please", "yes",
    })

    def _extract_entities(self, text):
        words = text.lower().split()
        entities = []
        seen = set()
        for w in words:
            w_clean = w.strip(".,!?;:'\"()[]{}")
            if w_clean in self.STOP_WORDS or len(w_clean) <= 2:
                continue
            if w_clean in self.concept_registry and w_clean not in seen:
                entities.append(w_clean)
                seen.add(w_clean)
                continue
            # Try fuzzy match (read-only) for singular/plural etc.
            found = self.embedding_bridge.closest_cell_word(
                w_clean, self, top_k=1, min_score=0.5
            )
            if found:
                fw = found[0] if isinstance(found, (list, tuple)) else found
                if fw not in seen and abs(len(fw) - len(w_clean)) <= 2:
                    # Prevent false positives like 'hello' -> 'cello':
                    # require a common prefix of at least min_len-1 chars
                    min_len = min(len(fw), len(w_clean))
                    common_prefix = 0
                    for a, b in zip(fw, w_clean):
                        if a == b:
                            common_prefix += 1
                        else:
                            break
                    if common_prefix >= min_len - 1:
                        entities.append(fw)
                        seen.add(fw)
        return entities

    def get_response(self, user_text, temperature=None, steps=10, epsilon=0.0):
        # 1. Extract entities, ingest user turn
        entities = self._extract_entities(user_text)
        self.wm.ingest(role="user", message=user_text, entities=entities)

        # 2. Activate entities in graph + WM
        for ent in entities:
            cell_id = self.concept_registry.get(ent)
            if cell_id:
                self.cells[cell_id].activation += 1.0
            self.wm.activate_entity(ent, salience=1.0)

        # 3. Delegate to DialogueManager for intent/strategy/response
        from dialogue_manager import DialogueManager
        dm = DialogueManager(self)
        response = dm.respond(user_text)

        # 4. Ingest bot turn
        resp_entities = self._extract_entities(response)
        self.wm.ingest(role="bot", message=response, entities=resp_entities, thought=response)

        # 5. Decay activations
        self.wm.decay_entities()
        for cell in self.cells.values():
            cell.activation *= 0.85

        return response

    def save(self, filename="states/brain_state.json"):


        data={

            "cells": {},

            "synapses": {},

            "thoughts": self.thoughts,

            "working_memory": self.wm.to_dict(),

            "embedding_weight": self.embedding_weight,

            "structural_weight": self.structural_weight,

            "context_bias_weight": self.context_bias_weight,

        }


        for cid,c in self.cells.items():

            data["cells"][cid]={

                "word": c.word,
                "energy":c.energy,
                "activation":c.activation

            }



        for sid,s in self.synapses.items():

            data["synapses"][sid]={

                "origin":s.origin.id,
                "target":s.target.id,

                "concept":s.concept,
                "relation":s.relation,

                "strength":s.strength,
                "cost":s.cost,

                "inference_usage":s.inference_usage,
                "reward":s.reward

            }


        with open(filename,"w",encoding="utf8") as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )



    def load(self, filename="states/brain_state.json"):

        try:

            with open(filename,"r",encoding="utf8") as f:

                data=json.load(f)

        except:

            return



        for cid,c in data["cells"].items():

            cell=Braincell(cid, word=c.get("word"))

            cell.energy=c["energy"]
            cell.activation=c["activation"]

            self.cells[cid]=cell

            if cell.word:
                self.concept_registry[cell.word] = cid



        for sid,s in data["synapses"].items():

            origin=self.cells[s["origin"]]
            target=self.cells[s["target"]]


            syn=Synapse(
                origin,
                target,
                s["concept"],
                relation=s.get("relation")
            )


            syn.id=sid
            syn.strength=s["strength"]
            syn.cost=s["cost"]
            syn.inference_usage=s.get("inference_usage", 0)
            syn.reward=s["reward"]


            self.synapses[sid]=syn


            origin.synapses_out.append(syn)
            target.synapses_in.append(syn)

        self.thoughts=data.get(
            "thoughts",
            []
        )

        wm_data = data.get("working_memory")
        if wm_data:
            self.wm.from_dict(wm_data)

        self.embedding_weight = data.get("embedding_weight", self.embedding_weight)
        self.structural_weight = data.get("structural_weight", self.structural_weight)
        self.context_bias_weight = data.get("context_bias_weight", self.context_bias_weight)