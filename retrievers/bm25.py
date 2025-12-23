# retrievers/bm25.py

from typing import List, Dict
import pandas as pd
from rank_bm25 import BM25Okapi
import re


def simple_tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer for clinical text.
    Keeps alphanumerics, splits on whitespace.
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


class BM25Retriever:
    """
    Sparse lexical retriever using BM25.

    Interface matches SapBERTFaissRetriever:
        retrieve(text, context) -> List[candidates]
    """

    def __init__(
        self,
        concept_path: str,
        top_k: int = 32,
        text_field: str = "concept_name",
        synonym_field: str = "synonyms",
        minimum_score: float = 3.0,
    ):
        """
        Args:
            concept_path: parquet with SNOMED / ontology concepts
            top_k: number of candidates to return
            text_field: preferred term column
            synonym_field: optional synonyms column (list or string)
        """

        self.top_k = top_k
        self.minimum_score = minimum_score
        self.concepts = pd.read_parquet(concept_path).reset_index(drop=True)

        # Build corpus
        self.corpus = []
        self.rows = []

        for _, row in self.concepts.iterrows():
            pieces = []

            # Preferred term
            if text_field in row and isinstance(row[text_field], str):
                pieces.append(row[text_field])

            # Synonyms (optional)
            if synonym_field in row:
                syn = row[synonym_field]
                if isinstance(syn, list):
                    pieces.extend(syn)
                elif isinstance(syn, str):
                    pieces.append(syn)

            joined = " ".join(pieces)
            tokens = simple_tokenize(joined)

            if tokens:
                self.corpus.append(tokens)
                self.rows.append(row)

        # Initialize BM25
        self.bm25 = BM25Okapi(self.corpus)

        print(f"[BM25Retriever] Indexed {len(self.corpus)} concepts.")

    def retrieve(self, text: str, context: str = "") -> List[Dict]:
        """
        Retrieve BM25 candidates for an entity.

        Args:
            text: entity text (possibly abbreviation-expanded)
            context: unused for now (kept for interface compatibility)

        Returns:
            List of candidate dicts
        """

        query_tokens = simple_tokenize(text)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[: self.top_k]

        candidates = []
        for idx in top_indices:
            score = scores[idx]
            if score < self.minimum_score:
                continue

            row = self.rows[idx]

            candidates.append({
                "concept_id": str(row["concept_id"]),
                "concept_name": row["concept_name"],
                "hierarchy": row.get("hierarchy"),
                "score": float(score),
                "source": "bm25",
            })

        return candidates
