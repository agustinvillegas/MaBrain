import json
from braincell import Brain


brain = Brain()


def train_sequence(sequence):

    previous_neuron = None

    for concept in sequence:

        # crea una neurona que procesa estado,
        # no representa el concepto
        current_neuron = brain.create_cell()

        if previous_neuron:

            synapse = brain.connect(
                previous_neuron,
                current_neuron,
                concept
            )

            # entrenamiento inicial
            synapse.usage += 1
            synapse.reward += 0.1

            # fortalecer ruta usada
            synapse.strength += 0.05

            # abaratar camino
            synapse.cost *= 0.999


        previous_neuron = current_neuron



def main():

    with open(
        "dataset.json",
        "r",
        encoding="utf8"
    ) as f:

        dataset = json.load(f)


    for item in dataset["sequences"]:

        train_sequence(
            item["sequence"]
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