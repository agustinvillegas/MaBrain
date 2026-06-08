import time
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    role: str  # "user" | "bot" | "system"
    message: str
    entities: list = field(default_factory=list)
    thought: str = None
    timestamp: float = field(default_factory=time.time)


class WorkingMemory:

    def __init__(self, max_turns=50, decay_rate=0.85):
        self.turns = []
        self.active_entities = {}
        self.topic_stack = []
        self.max_turns = max_turns
        self.decay_rate = decay_rate

    # -- Turn management --

    def ingest(self, role, message, entities=None, thought=None):
        turn = ConversationTurn(
            role=role,
            message=message,
            entities=entities or [],
            thought=thought,
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
        return turn

    def last_turns(self, n=5):
        return self.turns[-n:]

    def context_str(self, n=5, include_thoughts=False):
        parts = []
        for t in self.last_turns(n):
            line = f"{t.role}: {t.message}"
            if include_thoughts and t.thought:
                line += f" [{t.thought}]"
            parts.append(line)
        return "\n".join(parts)

    # -- Entity management --

    def activate_entity(self, entity, salience=1.0):
        self.active_entities[entity] = self.active_entities.get(entity, 0) + salience

    def get_active_entities(self, min_salience=0.1):
        return [e for e, s in self.active_entities.items() if s >= min_salience]

    def decay_entities(self):
        to_del = []
        for e, s in self.active_entities.items():
            s *= self.decay_rate
            if s < 0.01:
                to_del.append(e)
            else:
                self.active_entities[e] = s
        for e in to_del:
            del self.active_entities[e]

    # -- Topic management --

    def push_topic(self, topic):
        self.topic_stack.append(topic)

    def pop_topic(self):
        if self.topic_stack:
            return self.topic_stack.pop()
        return None

    def current_topic(self):
        return self.topic_stack[-1] if self.topic_stack else None

    # -- Serialization --

    def to_dict(self):
        return {
            "turns": [
                {"role": t.role, "message": t.message, "entities": t.entities,
                 "thought": t.thought, "timestamp": t.timestamp}
                for t in self.turns
            ],
            "active_entities": dict(self.active_entities),
            "topic_stack": list(self.topic_stack),
        }

    def from_dict(self, data):
        self.turns = [
            ConversationTurn(**t) for t in data.get("turns", [])
        ]
        self.active_entities = dict(data.get("active_entities", {}))
        self.topic_stack = list(data.get("topic_stack", []))
