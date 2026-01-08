# retrievers/sapbert_faiss.py

from retrievers.base import BaseRetriever
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils.text import normalize_text


class SapBERTFaissRetriever(BaseRetriever):
    def __init__(
        self,
        concept_path,
        emb_path,
        faiss_path,
        model_name,
        top_k=64,
        minimum_score=0.7,
    ):
        super().__init__(name="sapbert_faiss")

        self.top_k = top_k
        self.name = "sapbert_faiss"
        self.concepts = pd.read_parquet(concept_path)
        self.embeddings = np.load(emb_path).astype("float32")
        self.index = faiss.read_index(faiss_path)
        self.encoder = SentenceTransformer(model_name)
        self.minimum_score = minimum_score


    def retrieve(self, text, top_k=None):
        # query = text if not context else f"{text} {context}"
        query = normalize_text(text)

        q_emb = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        distances, idxs = self.index.search(
            q_emb,
            top_k or self.top_k,
        )

        results = []
        for dist, idx in zip(distances[0], idxs[0]):
            row = self.concepts.iloc[idx]
            score = float(1 - dist)
            if score >= self.minimum_score:
                results.append({
                    "concept_id": str(row["concept_id"]),
                    "concept_name": row["concept_name"],
                    "hierarchy": row.get("hierarchy"),
                    "parent_name":row["parent_name"],
                    "parent_hierarchy":row["parent_hierarchy"],
                    "score": score,
                    "source": self.name,
                })
        
        # query_context = f"{text} {context}"
        # q_emb_con = self.encoder.encode([query_context], normalize_embeddings=True).astype("float32")
        # distances_con, idxs_con = self.index.search(
        #     q_emb_con,
        #     top_k or self.top_k,
        # )

        # for dist, idx in zip(distances_con[0], idxs_con[0]):
        #     row = self.concepts.iloc[idx]
        #     score = float(1 - dist)
        #     if score >= self.minimum_score:
        #         results.append({
        #             "concept_id": str(row["concept_id"]),
        #             "concept_name": row["concept_name"],
        #             "hierarchy": row.get("hierarchy"),
        #             "score": score,
        #             "source": f"{self.name}_context",
        #         })

        return results
