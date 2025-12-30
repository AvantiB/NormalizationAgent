# agents/fusion.py

class CandidateFusionAgent:
    def fuse(self, candidates):
        """
        candidates: List[Dict] from multiple retrievers
        """

        if not candidates:
            return []

        # If someone accidentally passed a dict, fix it safely
        if isinstance(candidates, dict):
            candidates = list(candidates.values())

        fused = {}

        for c in candidates:
            if not isinstance(c, dict):
                raise TypeError(f"Fusion received non-dict candidate: {type(c)}")

            cid = c["concept_id"]

            if cid not in fused:
                fused[cid] = c
            else:
                # Keep the higher score
                fused[cid]["score"] = max(
                    fused[cid]["score"],
                    c.get("score", 0.0)
                )

        return list(fused.values())


# agents/fusion.py

# from collections import defaultdict

# class CandidateFusionAgent:
#     """
#     Source-aware, rank-aware candidate fusion.
#     """

#     SOURCE_WEIGHTS = {
#         "lexical": 1.0,
#         "bm25": 0.85,
#         "sapbert": 0.75,
#     }

#     def fuse(self, candidates, top_k=32):
#         if not candidates:
#             return []

#         grouped = defaultdict(list)

#         # Group by concept_id
#         for c in candidates:
#             if not isinstance(c, dict):
#                 raise TypeError(f"Fusion received non-dict candidate: {type(c)}")

#             grouped[c["concept_id"]].append(c)

#         fused = []

#         for cid, group in grouped.items():
#             # Best candidate per source
#             best_per_source = {}

#             for c in group:
#                 src = c.get("source", "unknown")
#                 score = c.get("score", 0.0)

#                 if src not in best_per_source or score > best_per_source[src]["score"]:
#                     best_per_source[src] = c

#             # Aggregate score
#             final_score = 0.0
#             evidence = []

#             for src, c in best_per_source.items():
#                 weight = self.SOURCE_WEIGHTS.get(src, 0.5)
#                 final_score += weight * c.get("score", 0.0)
#                 evidence.append(src)

#             # Use representative candidate
#             rep = max(best_per_source.values(), key=lambda x: x.get("score", 0.0))

#             fused.append({
#                 "concept_id": cid,
#                 "concept_name": rep.get("concept_name"),
#                 "hierarchy": rep.get("hierarchy"),
#                 "score": final_score,
#                 "sources": evidence,
#             })

#         # Sort by fused score
#         fused.sort(key=lambda x: x["score"], reverse=True)

#         return fused[:top_k]
