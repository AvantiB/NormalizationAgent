# retrievers/lexical.py
from rapidfuzz import fuzz
from utils.text import normalize_text, token_overlap_score

class LexicalRetriever:
    def __init__(self, concepts_df, top_k=32, minimum_score=0.7, lex_weight=0.7, tok_weight=0.3):
        self.df = concepts_df
        self.top_k = top_k
        self.minimum_score = minimum_score
        self.lex_weight = lex_weight
        self.tok_weight = tok_weight
        # exact string map
        self.exact_map = {
            normalize_text(row["concept_name"]): row
            for _, row in self.df.iterrows()
        }

    def retrieve(self, text: str, context: str | None = None):
        norm = normalize_text(text)

        candidates = []

        # 1) exact string match
        if norm in self.exact_map:
            row = self.exact_map[norm]
            return [{
                "concept_id": row["concept_id"],
                "concept_name": row["concept_name"],
                "hierarchy": row["hierarchy"],
                "score": 1.0,
                "source": "exact"
            }]

        # 2) fuzzy + token overlap
        for _, row in self.df.iterrows():
            pt_norm = normalize_text(row["concept_name"])
            lex = fuzz.WRatio(norm, pt_norm) / 100.0
            tok = token_overlap_score(norm, pt_norm)

            combined = self.lex_weight * lex + self.tok_weight * tok
            if combined > self.minimum_score:
                candidates.append({
                    "concept_id": row["concept_id"],
                    "concept_name": row["concept_name"],
                    "hierarchy": row["hierarchy"],
                    "score": combined,
                    "source": "lexical"
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[: self.top_k]
