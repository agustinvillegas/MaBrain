from braincell import Brain


def save_brain(brain: Brain, filename="brain_dump.txt"):
    with open(filename, "w", encoding="utf-8") as f:

        f.write("========== ESTADO DEL CEREBRO ==========\n\n")

        f.write(
            f"Neuronas: {len(brain.cells)}\n"
            f"Sinapsis: {len(brain.synapses)}\n\n"
        )

        f.write("\n========== NEURONAS ==========\n\n")

        for cell_id, cell in brain.cells.items():
            label = cell.word or cell.id[:8]
            f.write(
                f"[NEURONA {cell_id[:8]}] {label}\n"
                f"  energia: {cell.energy:.3f}\n"
                f"  activacion: {cell.activation:.3f}\n"
                f"  conexiones salida: {len(cell.synapses_out)}\n\n"
            )


        f.write("\n========== CONEXIONES ==========\n\n")

        for syn_id, syn in brain.synapses.items():

            origin_label = syn.origin.word or syn.origin.id[:8]
            target_label = syn.target.word or syn.target.id[:8]

            f.write(
                f"[SINAPSIS {syn_id[:8]}]\n"
                f"  {origin_label} ---> {target_label}\n"
                f"  concepto: {syn.concept}\n"
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