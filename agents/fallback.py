# agents/fallback.py

from typing import List, Dict, Optional


def ontology_guided_fallback(
    candidates: List[Dict],
    span_semantic_type: Optional[str] = None,
):
    """
    Returns a FULL decision dict, consistent with decision_llm output.
    Assumes candidates are sorted by fusion_score DESC.
    """

    if not candidates:
        return {
            "status": "hard_fail",
            "concept_id": None,
            "concept_name": None,
            "hierarchy": None,
            "confidence": 0.0,
            "source": "fallback_no_candidates",
        }

    top = candidates[0]

    # ---------- Rule 1: Parent backoff ----------
    if top.get("parent_id"):
        return {
            "status": "uncertain_fallback",
            "concept_id": top["parent_id"],
            "concept_name": top.get("parent_name"),
            "hierarchy": top.get("parent_hierarchy"),
            "confidence": 0.25,
            "source": "fallback_parent",
        }

    # ---------- Rule 2: Shallow semantic-compatible ----------
    if span_semantic_type:
        compatible = [
            c for c in candidates
            if c.get("hierarchy") == span_semantic_type
            and c.get("depth") is not None
        ]
        if compatible:
            compatible.sort(key=lambda c: c["depth"])
            c = compatible[0]
            return {
                "status": "uncertain_fallback",
                "concept_id": c["concept_id"],
                "concept_name": c.get("concept_name"),
                "hierarchy": c.get("hierarchy"),
                "confidence": 0.30,
                "source": "fallback_shallow_semantic",
            }

    # ---------- Rule 3: Trust retrieval ----------
    return {
        "status": "uncertain_fallback",
        "concept_id": top["concept_id"],
        "concept_name": top.get("concept_name"),
        "hierarchy": top.get("hierarchy"),
        "confidence": 0.35,
        "source": "fallback_top_retrieval",
    }
