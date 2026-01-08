from config import TAU_HIGH, TAU_LOW, FUSION_TOP_K_FOR_LLM

# -----------------------------
# Utility
# -----------------------------
def recall_at_k(cands, gold_id, k: int) -> int:
    gold = str(gold_id)
    return int(any(str(c.get("concept_id")) == gold for c in cands[:k]))


# -----------------------------
# Controller
# -----------------------------
class ControllerAgent:
    def __init__(self, retrievers, abbrev_resolver, fusion_agent,
                 decision_agent, entity_checker):
        self.retrievers = retrievers
        self.abbrev_resolver = abbrev_resolver
        self.fusion_agent = fusion_agent
        self.decision_agent = decision_agent
        self.entity_checker = entity_checker

    # -----------------------------
    def _sort(self, cands):
        return sorted(
            cands,
            key=lambda c: c.get("score", c.get("fused_score", 0.0)),
            reverse=True,
        )

    # -----------------------------
    def retrieve(self, text, gold_id=None):
        all_cands = []
        outputs = {}
        evals = {}

        for r in self.retrievers:
            cands = self._sort(r.retrieve(text=text))
            outputs[r.name] = cands
            all_cands.extend(cands)

        if gold_id is not None:
            for name, cands in outputs.items():
                evals[name] = {
                    "recall@1": recall_at_k(cands, gold_id, 1),
                    "recall@5": recall_at_k(cands, gold_id, 5),
                    "recall@10": recall_at_k(cands, gold_id, 10),
                }

        return all_cands, outputs, evals

    # -----------------------------
    def normalize(self, entity):
        orig = entity["text"]
        context = entity.get("context", "")
        gold_id = entity.get("gold_concept_id")

        # 0) Abbreviation expansion
        query = self.abbrev_resolver.expand(orig) if self.abbrev_resolver else orig

        # 1) Initial retrieval
        all_cands, outputs, retrieval_eval = self.retrieve(query, gold_id)
        sapbert = outputs.get("sapbert_faiss", [])
        top = sapbert[0] if sapbert else None
        score = float(top["score"]) if top else 0.0

        llm_entity = {
            "verbatim": orig,
            "expanded": query,
            "context": context,
            "semantic_tag_hint": "",
        }

        # --------------------------------------------------
        # CASE 1: High-confidence SapBERT → LLM Verifier
        # --------------------------------------------------
        if top and score >= TAU_HIGH:
            verdict = self.decision_agent.verify(
                entity=llm_entity,
                candidate={
                    "concept_id": top["concept_id"],
                    "concept_name": top["concept_name"],
                    "hierarchy": top["hierarchy"],
                },
            )
            if verdict.get("accept", False):
                return {
                    "concept_id": top["concept_id"],
                    "concept_name": top["concept_name"],
                    "hierarchy": top["hierarchy"],
                    "confidence": verdict.get("confidence", score),
                    "status": "verified",
                    "source": "sapbert_then_llm_verify",
                    "retrieval_eval": retrieval_eval,
                }
            # else → escalate

        # --------------------------------------------------
        # CASE 2: Mid-confidence SapBERT → TRUST SapBERT
        # --------------------------------------------------
        if top and score >= TAU_LOW:
            return {
                "concept_id": top["concept_id"],
                "concept_name": top["concept_name"],
                "hierarchy": top["hierarchy"],
                "confidence": score,
                "status": "sapbert_mid_confidence",
                "source": "sapbert_at_1",
                "retrieval_eval": retrieval_eval,
            }

        # --------------------------------------------------
        # CASE 3: Low confidence or empty → EntityChecker
        # --------------------------------------------------
        check = self.entity_checker.run(orig, context)

        if check["needs_rewrite"] and check["normalized_entity"]:
            new_query = check["normalized_entity"]
            all_cands, outputs, retrieval_eval = self.retrieve(new_query, gold_id)
            sapbert = outputs.get("sapbert_faiss", [])
            top = sapbert[0] if sapbert else None
            score = float(top["score"]) if top else 0.0

            llm_entity["expanded"] = new_query
            llm_entity["semantic_tag_hint"] = check.get("semantic_tag_hint", "")

            if top and score >= TAU_HIGH:
                verdict = self.decision_agent.verify(
                    entity=llm_entity,
                    candidate={
                        "concept_id": top["concept_id"],
                        "concept_name": top["concept_name"],
                        "hierarchy": top["hierarchy"],
                    },
                )
                if verdict.get("accept", False):
                    return {
                        "concept_id": top["concept_id"],
                        "concept_name": top["concept_name"],
                        "hierarchy": top["hierarchy"],
                        "confidence": verdict.get("confidence", score),
                        "status": "verified_after_entity_check",
                        "source": "entitychecker_then_sapbert_verify",
                        "retrieval_eval": retrieval_eval,
                    }

            if top:
                return {
                    "concept_id": top["concept_id"],
                    "concept_name": top["concept_name"],
                    "hierarchy": top["hierarchy"],
                    "confidence": score,
                    "status": "entitycheck_sapbert_fallback",
                    "source": "entitychecker_sapbert_at_1",
                    "retrieval_eval": retrieval_eval,
                }

        # --------------------------------------------------
        # CASE 4: TRUE FAILURE → Fusion → LLM Selector
        # --------------------------------------------------
        fused = self._sort(self.fusion_agent.fuse(all_cands))
        fused_topk = fused[:FUSION_TOP_K_FOR_LLM]

        if fused_topk:
            selection = self.decision_agent.select(
                entity=llm_entity,
                candidates=[
                    {
                        "concept_id": c["concept_id"],
                        "concept_name": c["concept_name"],
                        "hierarchy": c["hierarchy"],
                    }
                    for c in fused_topk
                ],
            )
            if selection.get("concept_id"):
                selection["status"] = "selected"
                selection["source"] = "llm_selector"
                selection["retrieval_eval"] = retrieval_eval
                return selection

        # --------------------------------------------------
        # FINAL FALLBACK
        # --------------------------------------------------
        if fused:
            top_fused = fused[0]
            return {
                "concept_id": top_fused["concept_id"],
                "concept_name": top_fused["concept_name"],
                "hierarchy": top_fused["hierarchy"],
                "confidence": top_fused.get("fused_score", 0.0),
                "status": "fallback_fusion",
                "source": "fallback_fusion_at_1",
                "retrieval_eval": retrieval_eval,
            }

        return {
            "concept_id": None,
            "concept_name": None,
            "hierarchy": None,
            "confidence": 0.0,
            "status": "hard_fail",
            "source": "no_candidates",
            "retrieval_eval": retrieval_eval,
        }
