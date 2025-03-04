from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def retrieve_context(query, k=1):
    with open("data\Main\openbook.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
    documents = [line.strip() for line in lines]

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(doc_embeddings))

    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)  # Retrieve top-k similar docs
    
    return [documents[i] for i in indices[0]]