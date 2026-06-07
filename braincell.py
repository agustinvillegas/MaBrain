import json
import math
import random
import uuid
from collections import deque


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

        self.usage = 0
        self.reward = 0.0

        # nueva
        self.activation_trace = 0.0


    def efficiency(self):

        if self.strength <= 0:
            return 999

        return self.cost / self.strength



class Brain:


    def __init__(self):

        self.cells = {}
        self.synapses = {}
        self.concept_registry = {}
        self.working_memory = deque(maxlen=5)

        self.thoughts = []


    def create_cell(self):

        cell = Braincell()

        self.cells[cell.id] = cell

        return cell


    def get_or_create_cell(self, concept):

        if concept in self.concept_registry:
            cell_id = self.concept_registry[concept]
            return self.cells[cell_id]

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



    def _select_synapse(self, options, temperature=1.0, epsilon=0.0):

        if temperature <= 0:
            return min(options, key=lambda x: x.efficiency())

        if random.random() < epsilon:
            return random.choice(options)

        efficiencies = [s.efficiency() for s in options]
        max_eff = max(efficiencies)
        weights = [math.exp(-(e - max_eff) / temperature) for e in efficiencies]

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


    def think(self, start_cell, steps=10, temperature=1.0, epsilon=0.0, use_working_memory=False):

        current = start_cell

        self.working_memory.clear()

        result = []


        for i in range(steps):

            if len(current.synapses_out) == 0:
                break


            options = current.synapses_out


            choice = self._select_synapse(
                options,
                temperature=temperature,
                epsilon=epsilon
            )


            result.append(choice.concept)

            choice.usage += 1
            choice.activation_trace += choice.strength
            choice.target.activation += choice.strength
            choice.target.thought_trace += choice.strength
            current = choice.target

            if use_working_memory:
                self.working_memory.append(current.word)

        thought = " ".join(result)

        self.thoughts.append(thought)

        return thought


    def query_relation(self, concept_a, concept_b):

        cell_a_id = self.concept_registry.get(concept_a)
        cell_b_id = self.concept_registry.get(concept_b)

        if not cell_a_id or not cell_b_id:
            return None

        cell_a = self.cells[cell_a_id]

        for syn in cell_a.synapses_out:
            if syn.target.id == cell_b_id:
                return syn.relation

        return None


    def find_by_relation(self, concept, relation):

        cell_id = self.concept_registry.get(concept)

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

        cell_a_id = self.concept_registry.get(a)
        cell_b_id = self.concept_registry.get(b)
        cell_c_id = self.concept_registry.get(c)

        if not cell_a_id or not cell_b_id or not cell_c_id:
            return results

        cell_a = self.cells[cell_a_id]
        cell_c = self.cells[cell_c_id]

        relations_ab = set()

        for syn in cell_a.synapses_out:
            if syn.target.id == cell_b_id:
                relations_ab.add(syn.relation)

        for relation in relations_ab:

            for csyn in cell_c.synapses_out:
                if csyn.relation != relation:
                    continue
                if not csyn.target.word:
                    continue
                score = csyn.strength / max(csyn.cost, 0.001)
                results.append({
                    "d": csyn.target.word,
                    "relation": relation,
                    "score": score,
                    "strength": csyn.strength,
                    "cost": csyn.cost,
                    "trace": {
                        "a_b": {"from": a, "to": b, "relation": relation},
                        "c_d": {"from": c, "to": csyn.target.word, "relation": relation}
                    }
                })

        results.sort(key=lambda x: (-x["score"], x["cost"]))
        return results[:top_k]


    def prune(self, min_usage=1):

        to_remove = []

        for sid, syn in self.synapses.items():
            if syn.usage < min_usage:
                to_remove.append(sid)

        for sid in to_remove:
            syn = self.synapses[sid]

            if syn in syn.origin.synapses_out:
                syn.origin.synapses_out.remove(syn)
            if syn in syn.target.synapses_in:
                syn.target.synapses_in.remove(syn)

            del self.synapses[sid]

        return len(to_remove)



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


    def save(self, filename="brain_state.json"):


        data={

            "cells": {},

            "synapses": {},

            "thoughts": self.thoughts

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

                "usage":s.usage,
                "reward":s.reward

            }


        with open(filename,"w",encoding="utf8") as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )



    def load(self, filename="brain_state.json"):

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
            syn.usage=s["usage"]
            syn.reward=s["reward"]


            self.synapses[sid]=syn


            origin.synapses_out.append(syn)
            target.synapses_in.append(syn)



        self.thoughts=data.get(
            "thoughts",
            []
        )