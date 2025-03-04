import faiss
import numpy as np
import json

def retrieve_context(query, embedder, documents, doc_embeddings, k=1):
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)  # Retrieve top-k similar docs
    
    return [documents[i] for i in indices[0]]

def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data