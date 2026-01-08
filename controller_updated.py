from config import TAU_HIGH, TAU_LOW, FUSION_TOP_K_FOR_LLM
from agents.fallback import ontology_guided_fallback  # (optional; not used yet)

def recall_at_k(cands, gold_id, k: int) -> int:
    gold = str(gold_id)
    topk = cands[:k]
    return int(any(str(c.get("concept_id")) == gold for c in topk))

def needs_entity_check(sapbert_cands, tau_low, tau_high):
    if not sapbert_cands:
        return True
    score = sapbert_cands[0]["score"]
    # low confidence OR mid confidence
    return score < tau_high

class ControllerAgent:
    def __init__(self, retrievers, abbrev_resolver, fusion_agent, decision_agent, entity_checker):
        self.retrievers = retrievers
        self.abbrev_resolver = abbrev_resolver
        self.fusion_agent = fusion_agent
        self.decision_agent = decision_agent
        self.entity_checker = entity_checker

    def _sort_by_score(self, cands):
        # defensively sort by either score or fused_score
        return sorted(
            cands,
            key=lambda c: c.get("score", c.get("fused_score", 0.0)),
            reverse=True,
        )
    
    def retrieve_candidates(self, query_text, gold_id):
        all_candidates = []
        retriever_outputs = {}

        for r in self.retrievers:
            cands = r.retrieve(text=query_text)
            cands = self._sort_by_score(cands)
            retriever_outputs[r.name] = cands
            all_candidates.extend(cands)

        retrieval_eval = {}
        
        if gold_id is not None:
            for name, cand_list in retriever_outputs.items():
                cand_list = self._sort_by_score(cand_list)
                retrieval_eval[name] = {
                    "recall@1": recall_at_k(cand_list, gold_id, 1),
                    "recall@5": recall_at_k(cand_list, gold_id, 5),
                    "recall@10": recall_at_k(cand_list, gold_id, 10),
                }
        return all_candidates, retriever_outputs, retrieval_eval


    def normalize(self, entity):
        # print("Entity Information: ", entity)
        orig_text = entity["text"]
        context = entity.get("context", "")
        gold_id = entity.get("gold_concept_id")

        sapbert_verified = False
        sapbert_rejected = False

        expanded = orig_text
        if self.abbrev_resolver:
            expanded = self.abbrev_resolver.expand(orig_text)

        query_text = expanded

        # ---------------------------
        # 1) Run retrievers, keep per-retriever outputs
        # ---------------------------
        
        all_candidates, retriever_outputs, retrieval_eval = self.retrieve_candidates(query_text, gold_id)
        sapbert_cands = retriever_outputs.get("sapbert_faiss", [])
        top_sapbert = sapbert_cands[0] if sapbert_cands else None
        top_sapbert_score = float(top_sapbert.get("score", 0.0)) if top_sapbert else 0.0

        llm_entity = {
            "verbatim": orig_text,
            "expanded": expanded,
            "context": context,
            "semantic_tag_hint":"",
        }

        # ---------------------------
        # 2) High-confidence SapBERT -> LLM Verifier (single candidate)
        # ---------------------------
        if top_sapbert and top_sapbert_score >= TAU_HIGH:
            # Build ONE candidate for verifier (include ontology info; ok if None)
            candidate_for_verify = {
                "concept_id": top_sapbert.get("concept_id"),
                "concept_name": top_sapbert.get("concept_name"),
                "hierarchy": top_sapbert.get("hierarchy"),
                # "parent_name": top_sapbert.get("parent_name"),
                # "parent_hierarchy": top_sapbert.get("parent_hierarchy"),
                # "score": top_sapbert.get("score"),
                # "source": top_sapbert.get("source", "sapbert_faiss"),
            }

            verdict = self.decision_agent.verify(entity=llm_entity, candidate=candidate_for_verify)

            if verdict.get("accept", False):
                # Return SapBERT@1 as final decision
                out =  {
                    "concept_id": candidate_for_verify["concept_id"],
                    "concept_name": candidate_for_verify["concept_name"],
                    "hierarchy": candidate_for_verify["hierarchy"],
                    "confidence": float(verdict.get("confidence", 0.0)),
                    "status": "verified",
                    "source": "sapbert_then_llm_verify",
                    "llm_notes": verdict.get("notes", ""),
                    "retrieval_eval": retrieval_eval,
                }
                # print("SapBERT candidate as final decision: ",out)
                return out
            else: 
                sapbert_rejected=True

        # if needs_entity_check(sapbert_cands, TAU_LOW, TAU_HIGH):
        else:
            check = self.entity_checker.run(orig_text, context)

            if check["needs_rewrite"] and check["normalized_entity"]:
                all_candidates, retriever_outputs, retrieval_eval = self.retrieve_candidates(check["normalized_entity"], gold_id)
                sapbert_cands = retriever_outputs.get("sapbert_faiss", [])
                top_sapbert = sapbert_cands[0] if sapbert_cands else None
                top_sapbert_score = float(top_sapbert.get("score", 0.0)) if top_sapbert else 0.0

                llm_entity = {
                    "verbatim": orig_text,
                    "expanded": check["normalized_entity"],
                    "context": context,
                    "semantic_tag_hint":check["semantic_tag_hint"],
                }

                # ---------------------------
                # 2) High-confidence SapBERT -> LLM Verifier (single candidate)
                # ---------------------------
                if top_sapbert and top_sapbert_score >= TAU_HIGH:
                    # Build ONE candidate for verifier (include ontology info; ok if None)
                    candidate_for_verify = {
                        "concept_id": top_sapbert.get("concept_id"),
                        "concept_name": top_sapbert.get("concept_name"),
                        "hierarchy": top_sapbert.get("hierarchy"),
                        # "parent_name": top_sapbert.get("parent_name"),
                        # "parent_hierarchy": top_sapbert.get("parent_hierarchy"),
                        # "score": top_sapbert.get("score"),
                        # "source": top_sapbert.get("source", "sapbert_faiss"),
                    }

                    verdict = self.decision_agent.verify(entity=llm_entity, candidate=candidate_for_verify)
                    if verdict.get("accept", False):
                        out =  {
                            "concept_id": candidate_for_verify["concept_id"],
                            "concept_name": candidate_for_verify["concept_name"],
                            "hierarchy": candidate_for_verify["hierarchy"],
                            "confidence": float(verdict.get("confidence", 0.0)),
                            "status": "verified_after_entity_check",
                            "source": "entitychecker_then_sapbert_verify",
                            "llm_notes": verdict.get("notes", ""),
                            "retrieval_eval": retrieval_eval,
                        }
                        # print("Entity Information: ", entity)
                        # print("SapBERT + EntityCheckerAgent: ",out)
                        return out
                    else:
                        sapbert_rejected = True

        # ---------------------------
        # 3) If SapBERT mid-confidence, OR verifier rejected, OR SapBERT empty:
        #    -> Fusion(top-K) -> LLM Selector
        # ---------------------------
            fused = self.fusion_agent.fuse(all_candidates)
            fused = self._sort_by_score(fused)
            fused_topk = fused[:FUSION_TOP_K_FOR_LLM]

            if gold_id is not None:
                fused_sorted = self._sort_by_score(fused)
                retrieval_eval["fusion"] = {
                    "recall@1": recall_at_k(fused_sorted, gold_id, 1),
                    "recall@5": recall_at_k(fused_sorted, gold_id, 5),
                    "recall@10": recall_at_k(fused_sorted, gold_id, 10),
                }

            candidates_for_llm = [
                {
                    "concept_id": c.get("concept_id"),
                    "concept_name": c.get("concept_name"),
                    "hierarchy": c.get("hierarchy"),
                    # "parent_name": c.get("parent_name"),
                    # "parent_hierarchy": c.get("parent_hierarchy"),
                    # "fused_score": c.get("fused_score", c.get("score", None)),
                    # "support": c.get("support", None),
                    # "retriever_source": c.get("source", None),
                }
                for c in fused_topk
            ]
        # print("Fused candidates for consideration to LLM selector: ",candidates_for_llm)
        
        # Only invoke selector if SapBERT score is in [TAU_LOW, TAU_HIGH) OR SapBERT was rejected OR SapBERT empty.
            if sapbert_rejected or (TAU_LOW<=top_sapbert_score<=TAU_HIGH):
                # Note: if top_sapbert_score >= TAU_HIGH and verifier rejected, we also come here
                selection = self.decision_agent.select(entity=llm_entity, candidates=candidates_for_llm)

                if selection.get("concept_id") is not None:
                    selection["status"] = "selected"
                    # keep a clean source tag (your CSV saving expects these fields)
                    selection["source"] = selection.get("source", "llm_selector")
                    # print("Selected concept by LLM: ", selection)
                    return selection

        # ---------------------------
        # 4) Selector returned null -> fallback (for now)
        #    (We will add EntityChecker/Paraphraser later)
        # ---------------------------
        # Fallback policy (your choice): SapBERT@1 if exists else fusion@1 else null
        if top_sapbert:
            fallback_sapbert= {
                "concept_id": top_sapbert.get("concept_id"),
                "concept_name": top_sapbert.get("concept_name"),
                "hierarchy": top_sapbert.get("hierarchy"),
                "confidence": float(top_sapbert.get("score", 0.0)),
                "status": "fallback_sapbert",
                "source": "fallback_sapbert_at_1",
                "retrieval_eval": retrieval_eval,
            }
            # print("Fallback to SAPBERT@1: ",fallback_sapbert)
            return fallback_sapbert

        if fused:
            top_fused = fused[0]
            fallback_fused =  {
                "concept_id": top_fused.get("concept_id"),
                "concept_name": top_fused.get("concept_name"),
                "hierarchy": top_fused.get("hierarchy"),
                "confidence": float(top_fused.get("fused_score", top_fused.get("score", 0.0))),
                "status": "fallback_fusion",
                "source": "fallback_fusion_at_1",
                "retrieval_eval": retrieval_eval,
            }
            # print("Fallback to FUSED@1: ",fallback_fused)
            return fallback_fused

        return {
            "concept_id": None,
            "concept_name": None,
            "hierarchy": None,
            "confidence": 0.0,
            "status": "hard_fail",
            "source": "no_candidates_all_retrievers",
            "retrieval_eval": retrieval_eval,
        }
