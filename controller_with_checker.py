from config import TAU_HIGH, TAU_LOW, FUSION_TOP_K_FOR_LLM


# -----------------------------
# Utilities
# -----------------------------
def recall_at_k(cands, gold_id, k: int) -> int:
    gold = str(gold_id)
    return int(any(str(c.get("concept_id")) == gold for c in cands[:k]))


# -----------------------------
# Controller Agent
# -----------------------------
class ControllerAgent:
    def __init__(
        self,
        retrievers,
        abbrev_resolver,
        fusion_agent,
        decision_agent,
        entity_checker,
    ):
        self.retrievers = retrievers
        self.abbrev_resolver = abbrev_resolver
        self.fusion_agent = fusion_agent
        self.decision_agent = decision_agent
        self.entity_checker = entity_checker

    # ---------------------------------
    # Internal helpers
    # ---------------------------------
    def _sort_by_score(self, cands):
        return sorted(
            cands,
            key=lambda c: c.get("score", c.get("fused_score", 0.0)),
            reverse=True,
        )

    def _run_retrievers(self, query_text, gold_id=None):
        all_candidates = []
        retriever_outputs = {}

        for r in self.retrievers:
            cands = self._sort_by_score(r.retrieve(text=query_text))
            retriever_outputs[r.name] = cands
            all_candidates.extend(cands)

        retrieval_eval = {}
        if gold_id is not None:
            for name, cands in retriever_outputs.items():
                retrieval_eval[name] = {
                    "recall@1": recall_at_k(cands, gold_id, 1),
                    "recall@5": recall_at_k(cands, gold_id, 5),
                    "recall@10": recall_at_k(cands, gold_id, 10),
                }

        return all_candidates, retriever_outputs, retrieval_eval

    # ---------------------------------
    # Main normalization entrypoint
    # ---------------------------------
    def normalize(self, entity):
        orig_text = entity["text"]
        context = entity.get("context", "")
        gold_id = entity.get("gold_concept_id")

        # ---------------------------------
        # Step 0: Abbreviation resolver
        # ---------------------------------
        expanded = (
            self.abbrev_resolver.expand(orig_text)
            if self.abbrev_resolver
            else orig_text
        )

        llm_entity = {
            "verbatim": orig_text,
            "expanded": expanded,
            "context": context,
            "semantic_tag_hint": "",
        }

        # ---------------------------------
        # Step 1: Initial retrieval
        # ---------------------------------
        all_cands, outputs, retrieval_eval = self._run_retrievers(
            expanded, gold_id
        )

        sapbert_cands = outputs.get("sapbert_faiss", [])
        top_sapbert = sapbert_cands[0] if sapbert_cands else None
        top_score = float(top_sapbert["score"]) if top_sapbert else 0.0

        # ---------------------------------
        # Step 2: High-confidence SapBERT → LLM verifier
        # ---------------------------------
        if top_sapbert and top_score >= TAU_HIGH:
            verdict = self.decision_agent.verify(
                entity=llm_entity,
                candidate={
                    "concept_id": top_sapbert["concept_id"],
                    "concept_name": top_sapbert["concept_name"],
                    "hierarchy": top_sapbert["hierarchy"],
                },
            )

            if verdict.get("accept", False):
                return {
                    "concept_id": top_sapbert["concept_id"],
                    "concept_name": top_sapbert["concept_name"],
                    "hierarchy": top_sapbert["hierarchy"],
                    "confidence": verdict.get("confidence", top_score),
                    "status": "verified",
                    "source": "sapbert_then_llm_verify",
                    "retrieval_eval": retrieval_eval,
                }

        # ---------------------------------
        # Step 3: EntityCheckerAgent (ALWAYS if not accepted)
        # ---------------------------------
        check = self.entity_checker.run(orig_text, context)

        if check["needs_rewrite"] and check["normalized_entity"]:
            rewritten = check["normalized_entity"]

            llm_entity["expanded"] = rewritten
            llm_entity["semantic_tag_hint"] = check.get("semantic_tag_hint", "")

            all_cands, outputs, retrieval_eval = self._run_retrievers(
                rewritten, gold_id
            )

            sapbert_cands = outputs.get("sapbert_faiss", [])
            top_sapbert = sapbert_cands[0] if sapbert_cands else None
            top_score = float(top_sapbert["score"]) if top_sapbert else 0.0

            # Re-verify if now high confidence
            if top_sapbert and top_score >= TAU_HIGH:
                verdict = self.decision_agent.verify(
                    entity=llm_entity,
                    candidate={
                        "concept_id": top_sapbert["concept_id"],
                        "concept_name": top_sapbert["concept_name"],
                        "hierarchy": top_sapbert["hierarchy"],
                    },
                )

                if verdict.get("accept", False):
                    return {
                        "concept_id": top_sapbert["concept_id"],
                        "concept_name": top_sapbert["concept_name"],
                        "hierarchy": top_sapbert["hierarchy"],
                        "confidence": verdict.get("confidence", top_score),
                        "status": "verified_after_entity_check",
                        "source": "entitychecker_then_sapbert_verify",
                        "retrieval_eval": retrieval_eval,
                    }

        # ---------------------------------
        # Step 4: Fusion + LLM selector (mid confidence OR rejected)
        # ---------------------------------
        fused = self._sort_by_score(self.fusion_agent.fuse(all_cands))
        fused_topk = fused[:FUSION_TOP_K_FOR_LLM]

        if gold_id is not None:
            retrieval_eval["fusion"] = {
                "recall@1": recall_at_k(fused, gold_id, 1),
                "recall@5": recall_at_k(fused, gold_id, 5),
                "recall@10": recall_at_k(fused, gold_id, 10),
            }

        # Optional semantic filtering hook (can be expanded later)
        if llm_entity["semantic_tag_hint"]:
            fused_topk = [
                c for c in fused_topk
                if c.get("hierarchy") == llm_entity["semantic_tag_hint"]
            ] or fused_topk

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

        if selection.get("concept_id") is not None:
            selection["status"] = "selected"
            selection["source"] = "llm_selector"
            selection["retrieval_eval"] = retrieval_eval
            return selection

        # ---------------------------------
        # Step 5: Deterministic fallback
        # ---------------------------------
        if top_sapbert:
            return {
                "concept_id": top_sapbert["concept_id"],
                "concept_name": top_sapbert["concept_name"],
                "hierarchy": top_sapbert["hierarchy"],
                "confidence": top_score,
                "status": "fallback",
                "source": "fallback_sapbert_at_1",
                "retrieval_eval": retrieval_eval,
            }

        if fused:
            return {
                "concept_id": fused[0]["concept_id"],
                "concept_name": fused[0]["concept_name"],
                "hierarchy": fused[0]["hierarchy"],
                "confidence": fused[0].get("fused_score", fused[0].get("score", 0.0)),
                "status": "fallback",
                "source": "fallback_fusion_at_1",
                "retrieval_eval": retrieval_eval,
            }

        # ---------------------------------
        # Step 6: Hard failure
        # ---------------------------------
        return {
            "concept_id": None,
            "concept_name": None,
            "hierarchy": None,
            "confidence": 0.0,
            "status": "hard_fail",
            "source": "no_candidates_all_retrievers",
            "retrieval_eval": retrieval_eval,
        }
