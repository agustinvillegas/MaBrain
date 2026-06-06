from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional
import json
import re


@dataclass
class Thought:
    input_tokens: List[str] # inp tok
    output_tokens: List[str]
    input_values: List[float]
    output_values: List[float]
    active_synapses: List[int]
    synapse_merit: Dict[int, float]
    energy_used: float
    budget: float
    budget_used: float
    economic_trace: List[dict] = field(default_factory=list)
    block_analysis: List[dict] = field(default_factory=list)
    result: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class Braincell:
    next_id = 0

    def __init__(self, label: Optional[str] = None, energy: float = 0.0):
        self.id = Braincell.next_id
        Braincell.next_id += 1

        self.label = label
        self.energy = energy
        self.synapses_in: List[Synapse] = []
        self.synapses_out: List[Synapse] = []
        self.active = False

    def receive(self, signal: float) -> None:
        self.energy += signal
        self.active = self.energy > 0

    def fire(self) -> float:
        if self.energy <= 0 or not self.synapses_out:
            self.active = False
            return 0.0

        self.active = True
        output_signal = self.energy
        self.energy = 0.0
        return output_signal

    def create_synapse(self, target: "Braincell", cost: float = 1.0, strength: float = 1.0) -> "Synapse":
        synapse = Synapse(origin=self, target=target, cost=cost, strength=strength)
        self.synapses_out.append(synapse)
        target.synapses_in.append(synapse)
        return synapse

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "energy": self.energy,
            "active": self.active,
            "synapses_in": [syn.id for syn in self.synapses_in],
            "synapses_out": [syn.id for syn in self.synapses_out],
        }


class Synapse:
    next_id = 0

    def __init__(self, origin: Braincell, target: Braincell, cost: float = 1.0, strength: float = 1.0):
        self.id = Synapse.next_id
        Synapse.next_id += 1

        self.origin = origin
        self.target = target
        self.strength = strength
        self.cost = cost
        self.usage = 0
        self.reward = 0.0

    def effective_cost(self) -> float:
        usage_discount = 1.0 + (self.usage ** 0.5) * 0.07
        reward_discount = 1.0 + min(0.8, max(0.0, self.reward) * 0.015)
        rarity_penalty = 1.0 + (1.0 / (self.usage + 1.0)) * 1.15
        return max(0.05, (self.cost * rarity_penalty) / (usage_discount * reward_discount))

    def transmit(self, input_signal: float) -> float:
        self.usage += 1
        return (input_signal * self.strength) / max(self.effective_cost(), 0.0001)

    def reward_route(self, amount: float = 1.0) -> None:
        self.reward += amount
        self.strength += 0.02 * amount
        self.cost = max(0.08, self.cost - 0.06 * amount)

    def punish_route(self, amount: float = 1.0) -> None:
        self.reward -= amount
        self.strength = max(0.1, self.strength - 0.02 * amount)
        self.cost += 0.11 * amount

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "origin": self.origin.id,
            "target": self.target.id,
            "strength": self.strength,
            "cost": self.cost,
            "usage": self.usage,
            "reward": self.reward,
        }


class Brain:
    MIN_SYNAPSE_SCORE = 0.82
    MAX_OUTPUTS_PER_TOKEN = 3
    PHRASE_BOOST = 1.25
    TOKEN_PENALTY_WITH_PHRASE = 0.90
    STOPWORDS = {"y", "o", "e", "u"}
    BUDGET_PER_INPUT_TOKEN = 1.75
    BUDGET_PER_OUTPUT_TOKEN = 1.0

    def __init__(self):
        self.cells: Dict[str, Braincell] = {}
        self.synapses: Dict[int, Synapse] = {}
        self.thoughts: List[Thought] = []
        self.memory_path = Path("brain_state.json")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())

    def get_or_create_cell(self, label: str) -> Braincell:
        if label not in self.cells:
            self.cells[label] = Braincell(label=label)
        return self.cells[label]

    @staticmethod
    def build_phrases(tokens: List[str]) -> List[str]:
        phrases: List[str] = []
        filtered = [token for token in tokens if token not in Brain.STOPWORDS]
        for index in range(len(filtered) - 1):
            phrases.append(f"{filtered[index]} {filtered[index + 1]}")
        return phrases

    @staticmethod
    def split_output_blocks(text: str) -> List[List[str]]:
        blocks: List[List[str]] = []
        for segment in text.split(","):
            tokens = [token for token in Brain.tokenize(segment) if token not in Brain.STOPWORDS]
            if tokens:
                blocks.append(tokens)
        return blocks

    def connect(self, origin_label: str, target_label: str, cost: float = 1.0, strength: float = 1.0) -> Synapse:
        origin = self.get_or_create_cell(origin_label)
        target = self.get_or_create_cell(target_label)
        for synapse in origin.synapses_out:
            if synapse.target is target:
                return synapse
        synapse = origin.create_synapse(target, cost=cost, strength=strength)
        self.synapses[synapse.id] = synapse
        return synapse

    def infer(self, input_text: str) -> Thought:
        tokens = self.tokenize(input_text)
        output_tokens: List[str] = []
        output_values: List[float] = []
        active_synapses: List[int] = []
        synapse_energy: Dict[int, float] = {}
        economic_trace: List[dict] = []
        energy_used = 0.0
        candidate_fanout = sum(len(self.cells[token].synapses_out) for token in tokens if token in self.cells)
        budget = max(
            1.0,
            (len(tokens) * self.BUDGET_PER_INPUT_TOKEN)
            + (candidate_fanout * 0.08)
            + 0.5,
        )
        budget_used = 0.0

        for token in tokens:
            cell = self.cells.get(token)
            if not cell:
                continue

            signal = 1.0
            ranked_synapses = sorted(
                cell.synapses_out,
                key=lambda syn: ((syn.strength / max(syn.effective_cost(), 0.0001)), syn.reward, -syn.usage),
                reverse=True,
            )
            for synapse in ranked_synapses[: self.MAX_OUTPUTS_PER_TOKEN]:
                effective_cost = synapse.effective_cost()
                synapse_score = synapse.strength / max(effective_cost, 0.0001)
                if synapse_score < self.MIN_SYNAPSE_SCORE:
                    continue
                projected_cost = effective_cost
                if budget_used + projected_cost > budget:
                    continue
                transmitted = synapse.transmit(signal)
                synapse.target.receive(transmitted)
                active_synapses.append(synapse.id)
                synapse_energy[synapse.id] = synapse_energy.get(synapse.id, 0.0) + transmitted
                economic_trace.append(
                    {
                        "synapse_id": synapse.id,
                        "origin": synapse.origin.label,
                        "target": synapse.target.label,
                        "effective_cost": effective_cost,
                        "transmitted": transmitted,
                    }
                )
                energy_used += transmitted
                budget_used += projected_cost
                if synapse.target.label:
                    output_tokens.append(synapse.target.label)
                    output_values.append(transmitted)

        output_tokens = list(dict.fromkeys(output_tokens))
        if output_values:
            max_output = max(output_values)
            output_values = [value / max_output if max_output > 0 else 0.0 for value in output_values]
        input_values = [1.0 for _ in tokens]

        if active_synapses:
            if energy_used > 0:
                synapse_merit = {syn_id: value / energy_used for syn_id, value in synapse_energy.items()}
            else:
                uniform = 1.0 / len(set(active_synapses))
                synapse_merit = {syn_id: uniform for syn_id in set(active_synapses)}
        else:
            synapse_merit = {}

        thought = Thought(
            input_tokens=tokens,
            output_tokens=output_tokens,
            input_values=input_values,
            output_values=output_values,
            active_synapses=active_synapses,
            synapse_merit=synapse_merit,
            energy_used=energy_used,
            budget=budget,
            budget_used=budget_used,
            economic_trace=economic_trace,
        )
        self.thoughts.append(thought)
        return thought

    def explain_thought(self, thought: Thought) -> List[tuple[str, float]]:
        ranked_outputs: List[tuple[str, float]] = []
        for synapse_id, merit in thought.synapse_merit.items():
            synapse = self.synapses.get(synapse_id)
            if not synapse or not synapse.target.label:
                continue
            score = (synapse.strength / max(synapse.cost, 0.0001)) * merit
            ranked_outputs.append((synapse.target.label, score))

        ranked_outputs.sort(key=lambda item: item[1], reverse=True)
        return ranked_outputs

    def debug_thought(self, thought: Thought) -> List[dict]:
        details: List[dict] = []
        for synapse_id, merit in thought.synapse_merit.items():
            synapse = self.synapses.get(synapse_id)
            if not synapse:
                continue
            score = (synapse.strength / max(synapse.cost, 0.0001)) * merit
            details.append(
                {
                    "synapse_id": synapse.id,
                    "origin": synapse.origin.label,
                    "target": synapse.target.label,
                    "strength": synapse.strength,
                    "cost": synapse.cost,
                    "usage": synapse.usage,
                    "reward": synapse.reward,
                    "merit": merit,
                    "score": score,
                }
            )

        details.sort(key=lambda item: item["score"], reverse=True)
        return details

    def reward(self, thought: Thought, amount: float = 1.0) -> None:
        thought.result = "correct"
        for synapse_id, merit in thought.synapse_merit.items():
            self.synapses[synapse_id].reward_route(amount * merit)

    def punish(self, thought: Thought, amount: float = 1.0) -> None:
        thought.result = "incorrect"
        for synapse_id, merit in thought.synapse_merit.items():
            self.synapses[synapse_id].punish_route(amount * merit)

    def train_pair(self, input_text: str, expected_output: str) -> Thought:
        input_tokens = self.tokenize(input_text)
        output_blocks = self.split_output_blocks(expected_output)
        output_tokens: List[str] = []
        output_phrases: List[str] = []
        for block in output_blocks:
            output_tokens.extend(block)
            output_phrases.extend(self.build_phrases(block))

        for token in input_tokens:
            self.get_or_create_cell(token)
        for token in output_tokens:
            self.get_or_create_cell(token)
        for phrase in output_phrases:
            self.get_or_create_cell(phrase)

        created_synapses: List[int] = []
        for source in input_tokens:
            for target in output_tokens:
                synapse = self.connect(source, target)
                created_synapses.append(synapse.id)
            for phrase in output_phrases:
                synapse = self.connect(source, phrase)
                synapse.strength *= self.PHRASE_BOOST
                synapse.cost = max(0.1, synapse.cost * 0.98)
                created_synapses.append(synapse.id)
                for token in output_tokens:
                    if token in phrase.split():
                        token_synapse = self.connect(source, token)
                        token_synapse.strength *= self.TOKEN_PENALTY_WITH_PHRASE
                        token_synapse.cost *= 1.01
                        created_synapses.append(token_synapse.id)

        thought = self.infer(input_text)
        thought.active_synapses = list(dict.fromkeys(created_synapses + thought.active_synapses))
        for synapse_id in created_synapses:
            thought.synapse_merit.setdefault(synapse_id, 0.0)
        return thought

    def save(self, path: Optional[str] = None) -> None:
        save_path = Path(path) if path else self.memory_path
        data = {
            "cells": [cell.to_dict() for cell in self.cells.values()],
            "synapses": {syn_id: synapse.to_dict() for syn_id, synapse in self.synapses.items()},
            "thoughts": [thought.to_dict() for thought in self.thoughts],
            "next_cell_id": Braincell.next_id,
            "next_synapse_id": Synapse.next_id,
        }
        save_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")

    def reset_memory(self, path: Optional[str] = None) -> None:
        self.cells = {}
        self.synapses = {}
        self.thoughts = []
        Braincell.next_id = 0
        Synapse.next_id = 0

        target_path = Path(path) if path else self.memory_path
        if target_path.exists():
            target_path.unlink()

    def load(self, path: Optional[str] = None) -> None:
        load_path = Path(path) if path else self.memory_path
        if not load_path.exists():
            return

        data = json.loads(load_path.read_text(encoding="utf-8"))
        self.cells = {}
        self.synapses = {}
        self.thoughts = []

        cells_data = data.get("cells", [])
        if isinstance(cells_data, dict):
            cells_data = list(cells_data.values())

        cell_lookup: Dict[int, Braincell] = {}
        max_cell_id = -1
        for cell_data in cells_data:
            cell = Braincell(label=cell_data.get("label"), energy=cell_data.get("energy", 0.0))
            cell.id = cell_data["id"]
            cell.active = cell_data.get("active", False)
            self.cells[cell.label] = cell
            cell_lookup[cell.id] = cell
            max_cell_id = max(max_cell_id, cell.id)

        synapses_data = data.get("synapses", {})
        if isinstance(synapses_data, list):
            synapses_iter = ((str(item["id"]), item) for item in synapses_data)
        else:
            synapses_iter = synapses_data.items()

        synapse_lookup: Dict[int, Synapse] = {}
        for syn_id_str, syn_data in synapses_iter:
            origin = cell_lookup[syn_data["origin"]]
            target = cell_lookup[syn_data["target"]]
            synapse = Synapse(origin=origin, target=target, cost=syn_data["cost"], strength=syn_data["strength"])
            synapse.id = int(syn_id_str)
            synapse.usage = syn_data["usage"]
            synapse.reward = syn_data["reward"]
            origin.synapses_out.append(synapse)
            target.synapses_in.append(synapse)
            synapse_lookup[synapse.id] = synapse

        self.synapses = synapse_lookup
        Braincell.next_id = max(max_cell_id + 1, data.get("next_cell_id", 0))
        Synapse.next_id = max((max(synapse_lookup) + 1) if synapse_lookup else 0, data.get("next_synapse_id", 0))

        for thought_data in data.get("thoughts", []):
            synapse_merit_data = thought_data.get("synapse_merit", {})
            synapse_merit = {int(k): float(v) for k, v in synapse_merit_data.items()}
            input_values = thought_data.get("input_values")
            if input_values is None:
                input_values = [1.0 for _ in thought_data.get("input_tokens", [])]
            output_values = thought_data.get("output_values")
            if output_values is None:
                output_values = [1.0 for _ in thought_data.get("output_tokens", [])]
            block_analysis = thought_data.get("block_analysis")
            if block_analysis is None:
                block_analysis = []
            economic_trace = thought_data.get("economic_trace")
            if economic_trace is None:
                economic_trace = []
            self.thoughts.append(
                Thought(
                    input_tokens=thought_data["input_tokens"],
                    output_tokens=thought_data["output_tokens"],
                    input_values=input_values,
                    output_values=output_values,
                    active_synapses=thought_data["active_synapses"],
                    synapse_merit=synapse_merit,
                    energy_used=thought_data["energy_used"],
                    budget=thought_data.get(
                        "budget",
                        max(1.0, len(thought_data.get("input_tokens", [])) * self.BUDGET_PER_INPUT_TOKEN),
                    ),
                    budget_used=thought_data.get("budget_used", 0.0),
                    economic_trace=economic_trace,
                    block_analysis=block_analysis,
                    result=thought_data.get("result"),
                )
            )
