# retrievers/lexical.py
from collections import defaultdict
from rapidfuzz import fuzz
from utils.text import normalize_text, token_overlap_score

class LexicalRetriever:
    def __init__(self, concepts_df, top_k=32, minimum_score=0.7, lex_weight=0.7, tok_weight=0.3, use_synonyms=True):

        self.df = concepts_df
        self.name = "lexical"
        self.top_k = top_k
        self.minimum_score = minimum_score
        self.lex_weight = lex_weight
        self.tok_weight = tok_weight
        self.use_synonyms = use_synonyms
        # exact string map
        # self.exact_map = {
        #     normalize_text(row["concept_name"]): row
        #     for _, row in self.df.iterrows()
        # }
        self.exact_map = defaultdict(list)
        for _, row in self.df.iterrows():
            # canonical name
            self.exact_map[normalize_text(row["concept_name"])].append(row)

            # synonyms
            # synonyms (ONLY if enabled)
            if self.use_synonyms:
                for syn in row.get("synonyms", []):
                    self.exact_map[normalize_text(syn)].append(row)

    def retrieve(self, text: str):
        norm = normalize_text(text)

        candidates = []

        # 1) exact string match
        if norm in self.exact_map:
            rows = self.exact_map[norm]
            out = []
            for row in rows:
                out.append({
                    "concept_id": row["concept_id"],
                    "concept_name": row["concept_name"],
                    "hierarchy": row["hierarchy"],
                    "parent_name": row.get("parent_name"),
                    "parent_hierarchy": row.get("parent_hierarchy"),
                    "score": 1.0,
                    "source": "exact",
                })
            return out[: self.top_k]

        # 2) fuzzy + token overlap
        for _, row in self.df.iterrows():
            if self.use_synonyms:
                names = [row["concept_name"]] + row.get("synonyms", [])

                best_score = 0.0
                for name in names:
                    name_norm = normalize_text(name)
                    lex = fuzz.WRatio(norm, name_norm) / 100.0
                    tok = token_overlap_score(norm, name_norm)
                    score = self.lex_weight * lex + self.tok_weight * tok
                    best_score = max(best_score, score)

                combined = best_score
            else:
                pt_norm = normalize_text(row["concept_name"])
                lex = fuzz.WRatio(norm, pt_norm) / 100.0
                tok = token_overlap_score(norm, pt_norm)
                combined = self.lex_weight * lex + self.tok_weight * tok

            if combined > self.minimum_score:
                candidates.append({
                    "concept_id": row["concept_id"],
                    "concept_name": row["concept_name"],
                    "hierarchy": row["hierarchy"],
                    "parent_name":row["parent_name"],
                    "parent_hierarchy":row["parent_hierarchy"],
                    "score": combined,
                    "source": "lexical"
                })

        # candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates.sort(key=lambda x: (-x["score"], x["concept_id"]))
        return candidates[: self.top_k]
