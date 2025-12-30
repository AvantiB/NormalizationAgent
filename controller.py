from config import STRICT_AUTO_THRESHOLD, LLM_THRESHOLD
import random

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
        for r in self.retrievers:
            all_candidates.extend(r.retrieve(text=query_text, context=context))

        fused = self.fusion_agent.fuse(all_candidates)

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
                "hierarchy": c.get("hierarchy"),
                # optional debug: include fused_score/support
                "fused_score": c.get("fused_score", None),
                "support": c.get("support", None),
            }
            for c in fused
        ]

        result = self.decision_agent.decide(entity=llm_entity, candidates=candidates_for_llm)
        return result

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