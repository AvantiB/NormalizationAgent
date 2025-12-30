# agents/fusion.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

@dataclass
class FusionConfig:
    # final candidates passed to LLM
    top_k_final: int = 20

    # per-retriever cap before fusion (prevents huge unions)
    per_source_cap: int = 32

    # RRF constant; larger makes ranks matter slightly less
    rrf_k: int = 60

    # require candidate to be supported by >= N sources OR be top_n in any one source
    require_min_sources: int = 1
    allow_if_top_n_any_source: int = 8

    # optional: slightly favor some sources in RRF
    source_weights: Optional[Dict[str, float]] = None

    # optional: enforce small diversity across hierarchies
    enforce_hierarchy_diversity: bool = False
    min_per_hierarchy: int = 2


class CandidateFusionAgent:
    """
    Robust hybrid fusion using Reciprocal Rank Fusion (RRF).
    Expects candidates as a flat list of dicts from multiple retrievers:
        {
          "concept_id": str,
          "concept_name": str,
          "hierarchy": str|None,
          "score": float,
          "source": "lexical"|"bm25"|"sapbert_faiss"|...
        }
    Returns a ranked list of fused candidates with:
        - fused_score
        - sources (list)
        - best_per_source (dict)
    """

    def __init__(self, config: FusionConfig | None = None):
        self.cfg = config or FusionConfig()

    def _cap_per_source(self, candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for c in candidates:
            if not isinstance(c, dict):
                continue
            src = c.get("source", "unknown")
            by_source[src].append(c)

        for src, lst in by_source.items():
            lst.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            by_source[src] = lst[: self.cfg.per_source_cap]
        return by_source

    def _rrf(self, by_source: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        fused_scores: Dict[str, float] = defaultdict(float)
        weights = self.cfg.source_weights or {}

        for src, lst in by_source.items():
            w = float(weights.get(src, 1.0))
            for rank, c in enumerate(lst, start=1):
                cid = str(c["concept_id"])
                fused_scores[cid] += w * (1.0 / (self.cfg.rrf_k + rank))
        return fused_scores

    def _support_stats(self, by_source: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]] , Dict[str, int]]:
        """
        Returns:
          - support_count[cid] = number of sources containing cid
          - best_score_per_source[cid][src] = best raw score of cid from that source
          - best_rank_any_source[cid] = best rank position across sources
        """
        support_count: Dict[str, int] = defaultdict(int)
        best_score_per_source: Dict[str, Dict[str, float]] = defaultdict(dict)
        best_rank_any_source: Dict[str, int] = defaultdict(lambda: 10**9)

        for src, lst in by_source.items():
            seen = set()
            for rank, c in enumerate(lst, start=1):
                cid = str(c["concept_id"])
                if cid not in seen:
                    support_count[cid] += 1
                    seen.add(cid)

                raw = float(c.get("score", 0.0))
                prev = best_score_per_source[cid].get(src, None)
                if prev is None or raw > prev:
                    best_score_per_source[cid][src] = raw

                if rank < best_rank_any_source[cid]:
                    best_rank_any_source[cid] = rank

        return support_count, best_score_per_source, best_rank_any_source

    def _pick_representative(self, candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        For each concept_id pick a representative candidate dict (prefer highest raw score).
        """
        best: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            cid = str(c.get("concept_id"))
            if not cid:
                continue
            if cid not in best:
                best[cid] = c
            else:
                if float(c.get("score", 0.0)) > float(best[cid].get("score", 0.0)):
                    best[cid] = c
        return best

    def fuse(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # Defensive: if someone passed a dict of candidates
        if isinstance(candidates, dict):
            candidates = list(candidates.values())

        # 1) cap per source
        by_source = self._cap_per_source(candidates)

        # 2) compute RRF scores
        rrf_scores = self._rrf(by_source)

        # 3) support stats for pruning
        support_count, best_score_per_source, best_rank_any_source = self._support_stats(by_source)

        # 4) representative dict per concept_id
        rep = self._pick_representative(candidates)

        # 5) build fused list
        fused_list: List[Dict[str, Any]] = []
        for cid, rrf in rrf_scores.items():
            base = rep.get(cid, {"concept_id": cid, "concept_name": None, "hierarchy": None})
            fused_list.append({
                "concept_id": cid,
                "concept_name": base.get("concept_name"),
                "hierarchy": base.get("hierarchy"),
                "source": "fused",
                "fused_score": float(rrf),
                "support": int(support_count.get(cid, 0)),
                "best_rank_any_source": int(best_rank_any_source.get(cid, 10**9)),
                "best_per_source": best_score_per_source.get(cid, {}),
                # keep raw representative score too (debug only)
                "rep_score": float(base.get("score", 0.0)),
                "rep_source": base.get("source", "unknown"),
            })

        # 6) prune: require evidence or top-N in any source
        pruned = []
        for c in fused_list:
            support = c["support"]
            best_rank = c["best_rank_any_source"]
            if support >= self.cfg.require_min_sources or best_rank <= self.cfg.allow_if_top_n_any_source:
                pruned.append(c)

        # 7) sort by fused score
        pruned.sort(key=lambda x: x["fused_score"], reverse=True)

        # 8) optional hierarchy diversity (useful when huge candidate sets)
        if self.cfg.enforce_hierarchy_diversity:
            out: List[Dict[str, Any]] = []
            by_h = defaultdict(list)
            for c in pruned:
                by_h[c.get("hierarchy")].append(c)

            # take min_per_hierarchy first
            for h, lst in by_h.items():
                out.extend(lst[: self.cfg.min_per_hierarchy])

            # fill remainder by global rank
            seen = set(x["concept_id"] for x in out)
            for c in pruned:
                if c["concept_id"] in seen:
                    continue
                out.append(c)
                if len(out) >= self.cfg.top_k_final:
                    break
            return out[: self.cfg.top_k_final]

        return pruned[: self.cfg.top_k_final]
