import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
CHUNKS_DIR = Path("data/chunks")
DB_DIR = Path("data/chroma_db")

DB_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 128

# Chargement du modèle d'embeddings
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# Initialisation ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))

strategies = ["fixed", "recursive", "semantic"]


def build_collection(strategy):

    print(f"\nProcessing {strategy}...")

    file_path = CHUNKS_DIR / f"{strategy}_chunks.json"

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    collection_name = f"fitness_{strategy}"

    # Recréation de la collection pour éviter les doublons
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        texts = [item["content"] for item in batch]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        ids = [item["chunk_id"] for item in batch]

        metadatas = [
            {
                "source": item["source"],
                "page": int(item["page"]),
                "chunk_id": item["chunk_id"]
            }
            for item in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"  Indexed {min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)}"
        )

    print(f"Collection {collection_name}: {collection.count()} vectors")

    return collection


# Construction des trois collections
collections = {}

for strategy in strategies:
    collections[strategy] = build_collection(strategy)

print("\nAll vector databases created successfully!")


# Fonction de recherche Top-K
def search(query, strategy="fixed", top_k=5):

    collection = client.get_collection(f"fitness_{strategy}")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    output = []

    for i in range(len(results["documents"][0])):

        output.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })

    return output


if __name__ == "__main__":

    question = "What are the benefits of protein intake for muscle growth?"

    for strategy in strategies:

        print(f"\n===== {strategy.upper()} SEARCH =====")

        results = search(question, strategy)

        for result in results:
            print(f"\nSource: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Content: {result['content'][:300]}...")
