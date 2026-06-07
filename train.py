import json
from braincell import Brain


brain = Brain()


def train_sequence(sequence, relations=None):

    previous_neuron = None
    rel_idx = 0

    for concept in sequence:

        current_neuron = brain.get_or_create_cell(concept)

        if previous_neuron:

            rel = relations[rel_idx] if relations and rel_idx < len(relations) else None

            synapse = brain.connect(
                previous_neuron,
                current_neuron,
                concept,
                relation=rel
            )

            # marca el recorrido para reward_thought
            synapse.activation_trace = 1.0
            rel_idx += 1

        previous_neuron = current_neuron

    # refuerzo hebbiano del recorrido completo
    brain.reward_thought(0.1)



def main():

    with open(
        "dataset.json",
        "r",
        encoding="utf8"
    ) as f:

        dataset = json.load(f)


    for item in dataset["sequences"]:

        train_sequence(
            item["sequence"],
            relations=item.get("relations")
        )


    brain.save()


    print(
        "Entrenamiento terminado"
    )

    print(
        "Neuronas:",
        len(brain.cells)
    )

    print(
        "Sinapsis:",
        len(brain.synapses)
    )



if __name__ == "__main__":
    main()