"""
Entry point for SNOMED-CT Entity Linking (Agentic Pipeline)
"""

import pandas as pd
from tqdm import tqdm

# -----------------------------
# Core controller
# -----------------------------
from controller import ControllerAgent

# -----------------------------
# Agents
# -----------------------------
from agents.abbreviations import AbbreviationResolver
from agents.fusion import CandidateFusionAgent
from agents.decision_llm import DecisionAgentLLM

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

DATA_DIR = "/dgx1data/aii/tao/m338824/ADRD/SNOMED_EL"
ONTOLOGY_DIR = "/dgx1data/aii/tao/m338824/ADRD/code/cluster_induction_taxonomy/ontology_caches"

SNOMED_CONCEPTS = f"{ONTOLOGY_DIR}/snomedEL_subset_with_synonyms.parquet"
SNOMED_EMB      = f"{ONTOLOGY_DIR}/snomedEL_subset_embeddings.npy"
SNOMED_FAISS   = f"{ONTOLOGY_DIR}/snomedEL_subset_faiss.index"

ABBREV_CSV = f"{DATA_DIR}/medical_abbreviations.csv"

MODEL_NAME = "google/medgemma-27b-text-it"
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
concepts_df = pd.read_parquet(SNOMED_CONCEPTS)


# ============================================================
# BUILD CONTROLLER
# ============================================================

controller = ControllerAgent(
    retrievers=[
        LexicalRetriever(concepts_df=concepts_df, top_k=32,minimum_score=0.7),
        BM25Retriever(
            concept_path=SNOMED_CONCEPTS,
            top_k=32,
            minimum_score=3.0
        ),
        SapBERTFaissRetriever(
            concept_path=SNOMED_CONCEPTS,
            emb_path=SNOMED_EMB,
            faiss_path=SNOMED_FAISS,
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
    )
)


# ============================================================
# RUN SNOMED-EL
# ============================================================

def main():
    print("Loading SNOMED-CT EL dataset...")

    df = pd.read_csv(f"{DATA_DIR}/train_df.csv").sample(frac=0.010, random_state=1)

    # predictions = []

    for i, row in tqdm(df.iterrows(),total=len(df), desc="Processing..."):
        entity = {
            "entity_id": f"{row['note_id']}_{row.name}",
            "text": row["verbatim"],
            "context": row.get("context", ""),
        }
        # print(entity)

        result = controller.normalize(entity)
        # predictions.append(result)
        df.loc[i, "pred_concept_id"] = result["concept_id"]
        df.loc[i, "pred_concept_name"] = result["concept_name"]
        df.loc[i,"pred_hierarchy"] = result.get("hierarchy","")
        df.loc[i, "pred_confidence"] = result.get("confidence", 0.0)

    # pred_df = pd.DataFrame(predictions)
    # df = pd.concat([df.reset_index(drop=True),], axis=1)

    # df.to_csv(f"{DATA_DIR}/snomed_el_predictions.csv", index=False)
    df = df.drop(["text"], axis=1)
    print(df)

    print("\nEvaluation:")
    metrics = evaluate_normalization(df)
    print_eval(metrics)



if __name__ == "__main__":
    main()
