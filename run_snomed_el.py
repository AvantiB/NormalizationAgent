"""
Entry point for SNOMED-CT Entity Linking (Agentic Pipeline)
"""

import pandas as pd
import os
from tqdm import tqdm
import json

# -----------------------------
# SapBERT+FAISS vectorstore builder
# -----------------------------
from vectorstore.snomed_builder_new import SnomedVectorStoreBuilder

# -----------------------------
# Core controller
# -----------------------------
from controller_with_checker1 import ControllerAgent

# -----------------------------
# Agents
# -----------------------------
from agents.abbreviations import AbbreviationResolver
from agents.fusion import CandidateFusionAgent
from agents.decision_llm_updated import DecisionAgentLLM
from agents.entity_checker import EntityCheckerAgent

# -----------------------------
# Retrievers
# -----------------------------
from retrievers.sapbert_faiss import SapBERTFaissRetriever
from retrievers.lexical import LexicalRetriever
from retrievers.bm25 import BM25Retriever

# -----------------------------
# Evaluation
# -----------------------------
from evaluation.snomed_eval import evaluate_normalization, print_eval


# ============================================================
# CONFIG
# ============================================================
# USE_SYNONYMS = True ##change this when needed
DATA_DIR = "/dgx1data/aii/tao/m338824/ADRD/SNOMED_EL"
CACHE_DIR = "/dgx1data/aii/tao/m338824/ADRD/code/normalization_agent/cache"

SNOMED_CONCEPTS_SYN = f"{CACHE_DIR}/snomedEL_with_synonyms_concepts.parquet"
SNOMED_CONCEPTS_NoSYN = f"{CACHE_DIR}/snomedEL_nosyn_concepts.parquet"
SNOMED_EMB_NoSYN     = f"{CACHE_DIR}/snomedEL_nosyn_embeddings.npy"
SNOMED_FAISS_NoSYN   = f"{CACHE_DIR}/snomedEL_nosyn_faiss.index"

if not os.path.exists(SNOMED_FAISS_NoSYN):
    print("Building vector store...")
    RF2_PATH = "/dgx1data/aii/tao/m338824/ADRD/code/SNOMED/SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z"
    # if USE_SYNONYMS:
    #     # WITH synonyms
    #     # print("Setting Synonym=True")
    #     builder_syn = SnomedVectorStoreBuilder(
    #         rf2_path=RF2_PATH,
    #         output_dir=CACHE_DIR,
    #         use_synonyms=True,
    #     )
    #     builder_syn.build()
    # else:
        # WITHOUT synonyms
        # print("Setting Synonym=False")
    builder_nosyn = SnomedVectorStoreBuilder(
        rf2_path=RF2_PATH,
        output_dir=CACHE_DIR,
        use_synonyms=False,
    )
    builder_nosyn.build()

ABBREV_CSV = f"{DATA_DIR}/medical_abbreviations.csv"

MODEL_NAME = "google/medgemma-27b-text-it"
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
concepts_df = pd.read_parquet(SNOMED_CONCEPTS_SYN)

# ============================================================
# Debugging and Sanity checks
# ============================================================
# #sanity check
# # print(concepts_df[["concept_name","parent_name"]].sample(10))
# print("LEXICAL DF SHAPE:", concepts_df.shape)
# print("LEXICAL DF COLS:", concepts_df.columns.tolist())

# # concept_name sanity
# print("concept_name sample:", concepts_df["concept_name"].head(3).tolist())

# # synonyms sanity (even if lexical doesn't use them)
# if "synonyms" in concepts_df.columns:
#     nonempty = concepts_df["synonyms"].apply(lambda x: isinstance(x, list) and len(x) > 0).mean()
#     print("synonyms non-empty fraction:", nonempty)

# def fingerprint(df):
#     # df_fp = df.copy()
#     # df_fp["synonyms"] = df_fp["synonyms"].apply(tuple)
#     # fingerprint = pd.util.hash_pandas_object(df_fp, index=True).sum()
#     fingerprint = pd.util.hash_pandas_object(
#                 df[["concept_id", "concept_name", "hierarchy", "parent_id", "depth"]],
#                 index=True
#             ).sum()
#     return fingerprint

# print("Fingerprint:", fingerprint(concepts_df))


# ============================================================
# BUILD CONTROLLER
# ============================================================

controller = ControllerAgent(
    retrievers=[
        LexicalRetriever(concepts_df=concepts_df, 
                         top_k=32,
                         minimum_score=0.7,
                         use_synonyms=True
                         ),
        BM25Retriever(
                        concept_path=SNOMED_CONCEPTS_SYN,
                        top_k=32,
                        minimum_score=3.0
                    ),
        SapBERTFaissRetriever(
            concept_path=SNOMED_CONCEPTS_NoSYN, ##be sure to point these to No_synonym version
            emb_path=SNOMED_EMB_NoSYN,
            faiss_path=SNOMED_FAISS_NoSYN,
            model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            top_k=64,
            minimum_score=0.6
        )
    ],
    abbrev_resolver=AbbreviationResolver(
        ABBREV_CSV,
        expand_only_uppercase=True,
        standalone_only=True,
    ),
    fusion_agent=CandidateFusionAgent(),
    decision_agent=DecisionAgentLLM(
        model_name=MODEL_NAME,
        api_url=VLLM_API_URL,
    ),
    entity_checker=EntityCheckerAgent(
        model_name=MODEL_NAME,
        api_url=VLLM_API_URL,
    )
)

def save_jsonl_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============================================================
# RUN SNOMED-EL
# ============================================================

def main():
    print("Loading SNOMED-CT EL dataset...")

    df = pd.read_csv(f"{DATA_DIR}/train_df.csv").sample(frac=0.005, random_state=1)

    all_results=[]

    for i, row in tqdm(df.iterrows(),total=len(df), desc="Processing..."):
        entity = {
            "entity_id": f"{row['note_id']}_{row.name}",
            "gold_concept_id":row["concept_id"],
            "text": row["verbatim"],
            "context": row.get("context", ""),
        }

        result = controller.normalize(entity)
        result["entity_id"] = entity["entity_id"]
        result["gold_concept_id"] = entity["gold_concept_id"]
        result["gold_concept_name"] = row["gold_pt"]
        result["gold_hierarchy"] = row["gold_tag"]

        all_results.append(result)

        df.loc[i, "pred_concept_id"] = result["concept_id"]
        df.loc[i, "pred_concept_name"] = result["concept_name"]
        df.loc[i,"pred_hierarchy"] = result.get("hierarchy","")
        df.loc[i, "pred_confidence"] = result.get("confidence", 0.0)
        df.loc[i, "source"] = result.get("source","")
   
    save_jsonl_results(all_results, f"{DATA_DIR}/snomed_predictions_detailed_mixed.jsonl")

    df = df.drop(["text"], axis=1)
    print("\nEvaluation:")
    metrics = evaluate_normalization(df)
    print_eval(metrics)


if __name__ == "__main__":
    main()
