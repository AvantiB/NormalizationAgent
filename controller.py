from config import STRICT_AUTO_THRESHOLD, LLM_THRESHOLD
import random
import numpy as np
from agents.fallback import ontology_guided_fallback

def recall_at_k(candidates, gold_id, k):
    if not gold_id:
        return None
    gold_id = str(gold_id)
    topk = [str(c["concept_id"]) for c in candidates[:k]]
    return int(gold_id in topk)

class ControllerAgent:
    def __init__(self, retrievers, abbrev_resolver, fusion_agent, decision_agent):
        self.retrievers = retrievers
        self.abbrev_resolver = abbrev_resolver
        self.fusion_agent = fusion_agent
        self.decision_agent = decision_agent

    def normalize(self, entity):
        orig_text = entity["text"]
        context = entity.get("context", "")

        expanded = orig_text
        if self.abbrev_resolver:
            expanded = self.abbrev_resolver.expand(orig_text)

        # retrieval uses expanded (better recall)
        query_text = expanded

        all_candidates = []
        retriever_outputs = {}
        for r in self.retrievers:
            # all_candidates.extend(r.retrieve(text=query_text, context=context))
            cands = r.retrieve(text=query_text, context=context)
            retriever_outputs[r.name] = cands   # retriever must have .name
            all_candidates.extend(cands)
        
        gold_id = entity.get("gold_concept_id") ##need this from the dataframe
        retrieval_eval = {}

        if gold_id:
            for name, cand_list in retriever_outputs.items():
                # enforce ordering defensively
                cand_list = sorted(
                    cand_list,
                    key=lambda c: c.get("score", c.get("fused_score", 0.0)),
                    reverse=True,
                )
                retrieval_eval[name] = {
                    "recall@1": recall_at_k(cand_list, gold_id, 1),
                    "recall@5": recall_at_k(cand_list, gold_id, 5),
                    "recall@10": recall_at_k(cand_list, gold_id, 10),
                }

        # print(all_candidates[:5])

        fused = self.fusion_agent.fuse(all_candidates)
        if gold_id:
            fused_sorted = sorted(
                fused,
                key=lambda c: c.get("fused_score", 0.0),
                reverse=True,
            )
            retrieval_eval["fusion"] = {
                "recall@1": recall_at_k(fused_sorted, gold_id, 1),
                "recall@5": recall_at_k(fused_sorted, gold_id, 5),
                "recall@10": recall_at_k(fused_sorted, gold_id, 10),
            }

            ##debug (begin)
            # gold_id = str(entity.get("gold_concept_id"))
            # fused_ids = [str(c.get("concept_id")) for c in fused]
            # print("gold in fused anywhere?", gold_id in fused_ids)
            # if gold_id in fused_ids:
            #     print("gold position in fused:", fused_ids.index(gold_id))
            # print("top10 fused ids:", fused_ids[:10])
            ## debug (end)

        # LLM sees BOTH
        llm_entity = {
            "verbatim": orig_text,
            "expanded": expanded,
            "context": context,
        }

        candidates_for_llm = [
            {
                "concept_id": c["concept_id"],
                "concept_name": c["concept_name"],
                # "hierarchy": c["hierarchy"], #temporarily commented

                # "parent_name":c["parent_name"],
                # "parent_hierarchy":c["parent_hierarchy"],
                # optional debug: include fused_score/support
                "fused_score": c.get("fused_score", None),
                "support": c.get("support", None),
            }
            for c in fused
        ]
        # print(candidates_for_llm)

        final_decision = self.decision_agent.decide(entity=llm_entity, candidates=candidates_for_llm)

        ## Fallback logic##
        if final_decision["concept_id"] == None or final_decision["concept_id"]=="":
            # print("Fallback activated..")
            # Fallback tier 1: SapBERT
            sapbert_cands = retriever_outputs.get("sapbert_faiss", [])
            if sapbert_cands:
                # final_prediction = sapbert_cands[0]
                # final_source = "sapbert_fallback"
                final_decision = sapbert_cands[0]
                final_decision["source"] = "sapbert_fallback"
                # print("selecting SAPBERT_FAISS retriever...")
                # print(final_decision)
            else:
                # Fallback tier 2: Fusion
                if fused:
                    # final_prediction = fusion_cands[0]
                    # final_source = "fusion_fallback"
                    final_decision = fused[0]
                    final_decision["source"] = "fusion_fallback"
                    # print("selecting Fusion retriever...")
                    # print(final_decision)
                else:
                    print("No viable candidates retrieved...")
                    print("entity: ",entity)
                    print("retriever results: ",retriever_outputs)

        final_decision["retrieval_eval"] = retrieval_eval

        return final_decision






        # # 1. Preprocess
        # original_text = entity["text"]
        # context = entity.get("context", "")

        # # Expanded form (used for retrieval + exposed to LLM)
        # if self.abbrev_resolver:
        #     expanded_text = self.abbrev_resolver.expand(original_text)
        # else:
        #     expanded_text = original_text

        # # 2. Retrieve candidates
        # all_candidates = []
        # for r in self.retrievers:
        #     candidates = r.retrieve(
        #         text=expanded_text,
        #         context=context,
        #     )
        #     print(f"Retriever {r.__class__.__name__} returned {len(candidates)} candidates.")
        #     # print("DEBUG retrieve output:", candidates)
        #     all_candidates.extend(candidates)
        # # print(f"Retrieved {len(all_candidates)} total candidates.")
        # # random.shuffle(all_candidates)

        # # 3. Fuse candidates
        # fused = self.fusion_agent.fuse(all_candidates)
        # print("Total fused candidates:", len(fused))
        # # print("FINAL candidates passed to LLM:", fused)
       
        # # 4. Decide (LLM or rule-based)
        # candidates_for_LLM = [
        #         {"concept_id":c["concept_id"],
        #          "concept_name":c["concept_name"],
        #          "hierarchy":c.get("hierarchy"),
        #         }
        #         for c in fused
        #     ]
        
        # entity={
        #         "verbatim": original_text,
        #         "expanded": expanded_text,
        #         "context": context,
        #     }
        # print("entity:", entity)
        # print("candidates for LLM:", candidates_for_LLM)

        # result = self.decision_agent.decide(
        #     entity=entity,
        #     candidates=candidates_for_LLM,
        # )
        # print(result)

        # return result