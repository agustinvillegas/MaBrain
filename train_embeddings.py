import os
import sys
import json

DATASET_PATH = "datasets/dataset.json"
MODEL_OUTPUT = "embeddings.bin"


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    vocab = set()

    for item in data["sequences"]:
        seq = item["sequence"]
        sentences.append(seq)
        vocab.update(seq)

    return sentences, sorted(vocab)


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_OUTPUT

    if not os.path.exists(dataset_path):
        print(f"  X dataset no encontrado: {dataset_path}")
        sys.exit(1)

    try:
        from gensim.models.fasttext import FastText
    except ImportError:
        print("  X gensim no instalado. Corre: pip install gensim")
        sys.exit(1)

    print(f"  Cargando: {dataset_path}")
    sentences, vocab = load_dataset(dataset_path)
    print(f"  {len(sentences)} secuencias, {len(vocab)} palabras unicas")

    dims = min(len(vocab) * 2, 100)
    epochs = max(50, 500 // len(sentences) + 1)

    print(f"  Entrenando FastText (vector_size={dims}, epochs={epochs})...")
    model = FastText(
        sentences=sentences,
        vector_size=dims,
        window=3,
        min_count=1,
        sg=1,
        epochs=epochs,
        seed=42,
    )

    model.save(output_path)
    print(f"  Modelo guardado: {output_path} ({os.path.getsize(output_path)} bytes)")

    print()
    print("  Usalo asi:")
    print(f"    from braincell import Brain")
    print(f"    from embedding_bridge import EmbeddingBridge")
    print(f"    brain = Brain(embedding_bridge=EmbeddingBridge('{output_path}'))")
    print()


if __name__ == "__main__":
    main()
