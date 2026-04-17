import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RecyclingRAG:
    def __init__(self, knowledge_dir="knowledge_base"):
        self.knowledge_dir = knowledge_dir
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.documents = []
        self.doc_classes = []
        self.index = None

        self._load_documents()
        self._build_index()

    def _chunk_text(self, text, chunk_size=120):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _load_documents(self):
        for file_name in os.listdir(self.knowledge_dir):
            if file_name.endswith(".txt"):
                class_name = file_name.replace(".txt", "").upper()
                path = os.path.join(self.knowledge_dir, file_name)

                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                chunks = self._chunk_text(text)

                for chunk in chunks:
                    self.documents.append(chunk)
                    self.doc_classes.append(class_name)

    def _build_index(self):
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve_context(self, waste_class, top_k=3):
        query = f"How to recycle {waste_class} waste and why it is useful"
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype("float32")

        distances, indices = self.index.search(query_embedding, 10)

        matched_chunks = []
        for idx in indices[0]:
            if 0 <= idx < len(self.doc_classes) and self.doc_classes[idx] == waste_class.upper():
                chunk = self.documents[idx]
                if chunk not in matched_chunks:
                    matched_chunks.append(chunk)
            if len(matched_chunks) == top_k:
                break

        return "\n\n".join(matched_chunks)