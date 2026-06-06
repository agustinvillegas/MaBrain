from braincell import Brain


def save_brain(brain: Brain, filename="brain_dump.txt"):
    with open(filename, "w", encoding="utf-8") as f:

        f.write("========== ESTADO DEL CEREBRO ==========\n\n")

        f.write(
            f"Neuronas: {len(brain.cells)}\n"
            f"Sinapsis: {len(brain.synapses)}\n\n"
        )

        f.write("\n========== NEURONAS ==========\n\n")

        for label, cell in brain.cells.items():
            f.write(
                f"[NEURONA {cell.id}] {label}\n"
                f"  energia: {cell.energy:.3f}\n"
                f"  activa: {cell.active}\n"
                f"  conexiones salida: {len(cell.synapses_out)}\n\n"
            )


        f.write("\n========== CONEXIONES ==========\n\n")

        for syn_id, syn in brain.synapses.items():

            f.write(
                f"[SINAPSIS {syn_id}]\n"
                f"  {syn.origin.label} ---> {syn.target.label}\n"
                f"  fuerza: {syn.strength:.3f}\n"
                f"  costo: {syn.cost:.3f}\n"
                f"  uso: {syn.usage}\n"
                f"  recompensa: {syn.reward:.3f}\n\n"
            )


def main():
    brain = Brain()
    brain.load()

    save_brain(brain)

    print("Exportado correctamente a brain_dump.txt")


if __name__ == "__main__":
    main()