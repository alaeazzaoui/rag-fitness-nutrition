import json
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = Path("data/processed/documents.json")
OUTPUT_DIR = Path("data/chunks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


# 1. FIXED-SIZE CHUNKING
def fixed_chunking(text, size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + size].strip()

        if chunk:
            chunks.append(chunk)

        start += size - overlap

    return chunks


# 2. RECURSIVE CHUNKING
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# 3. SEMANTIC CHUNKING
model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunking(text, threshold=0.65, max_size=500, min_size=150):

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    embeddings = model.encode(sentences)

    chunks = []
    current_sentences = [sentences[0]]

    for i in range(1, len(sentences)):

        current_text = " ".join(current_sentences)
        candidate = current_text + " " + sentences[i]

        similarity = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if similarity >= threshold and len(candidate) <= max_size:
            current_sentences.append(sentences[i])

        else:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentences[i]]

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    # Fusionner les petits chunks avec leurs voisins
    optimized_chunks = []

    for chunk in chunks:

        if optimized_chunks and len(chunk) < min_size:
            candidate = optimized_chunks[-1] + " " + chunk

            if len(candidate) <= max_size:
                optimized_chunks[-1] = candidate
            else:
                optimized_chunks.append(chunk)
        else:
            optimized_chunks.append(chunk)
    
    final_chunks = []

    for chunk in optimized_chunks:
        if len(chunk) > max_size:
            final_chunks.extend(
                fixed_chunking(chunk, size=max_size, overlap=50)
            )
        else:
            final_chunks.append(chunk)

    return final_chunks

# EXECUTION
strategies = {
    "fixed": fixed_chunking,
    "recursive": lambda text: recursive_splitter.split_text(text),
    "semantic": semantic_chunking
}

for strategy_name, strategy_function in strategies.items():

    all_chunks = []

    for doc_index, document in enumerate(documents):

        text_chunks = strategy_function(document["content"])

        for chunk_index, chunk in enumerate(text_chunks):

            all_chunks.append({
                "chunk_id": f"{strategy_name}_{doc_index}_{chunk_index}",
                "source": document["source"],
                "page": document["page"],
                "content": chunk
            })

    output_file = OUTPUT_DIR / f"{strategy_name}_chunks.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"✅ {strategy_name}: {len(all_chunks)} chunks")

print("\n🎉 Les trois stratégies ont été exécutées !")