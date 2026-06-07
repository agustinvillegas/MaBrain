import json
import random
import uuid


class Braincell:

    def __init__(self, cell_id=None):
        self.id = cell_id or str(uuid.uuid4())

        self.energy = 0.0
        self.activation = 0.0

        # nueva memoria de participación
        self.thought_trace = 0.0

        self.synapses_out = []
        self.synapses_in = []


class Synapse:

    def __init__(self, origin, target, concept):

        self.id = str(uuid.uuid4())

        self.origin = origin
        self.target = target
        self.concept = concept

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

        self.thoughts = []


    def create_cell(self):

        cell = Braincell()

        self.cells[cell.id] = cell

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

    def connect(self, a, b, concept):

        # evitar duplicados del mismo concepto
        for syn in a.synapses_out:

            if (
                syn.target == b and
                syn.concept == concept
            ):
                return syn


        syn = Synapse(
            a,
            b,
            concept
        )

        self.synapses[syn.id] = syn


        a.synapses_out.append(syn)
        b.synapses_in.append(syn)


        return syn


    def learn(self, text):

        words = text.split()

        if len(words) < 2:
            return


        previous = None


        for word in words:

            # neurona temporal de procesamiento
            cell = self.create_cell()


            if previous:

                self.connect(
                    previous,
                    cell,
                    word
                )


            previous = cell



    def think(self, start_cell, steps=10):

        current = start_cell

        result = []


        for i in range(steps):

            if len(current.synapses_out) == 0:
                break


            options = current.synapses_out


            # economía energética
            best = min(
                options,
                key=lambda x:x.efficiency()
            )


            result.append(best.concept)

            best.usage += 1
            best.activation_trace += best.strength
            best.target.activation += best.strength
            best.target.thought_trace += best.strength
            current = best.target

        thought = " ".join(result)

        self.thoughts.append(thought)

        return thought



    def save(self, filename="brain_state.json"):


        data={

            "cells": {},

            "synapses": {},

            "thoughts": self.thoughts

        }


        for cid,c in self.cells.items():

            data["cells"][cid]={

                "energy":c.energy,
                "activation":c.activation

            }



        for sid,s in self.synapses.items():

            data["synapses"][sid]={

                "origin":s.origin.id,
                "target":s.target.id,

                "concept":s.concept,

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

            cell=Braincell(cid)

            cell.energy=c["energy"]
            cell.activation=c["activation"]

            self.cells[cid]=cell



        for sid,s in data["synapses"].items():

            origin=self.cells[s["origin"]]
            target=self.cells[s["target"]]


            syn=Synapse(
                origin,
                target,
                s["concept"]
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