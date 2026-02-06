import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []

    def build_index(self, chunks):
        embeddings = self.model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.text_chunks = chunks

    def search(self, query, k=3):
        query_embedding = self.model.encode([query]).astype("float32")
        _, indices = self.index.search(query_embedding, k)
        return [self.text_chunks[i] for i in indices[0]]

