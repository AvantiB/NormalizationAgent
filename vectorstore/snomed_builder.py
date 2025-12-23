import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import faiss
import torch
from sentence_transformers import SentenceTransformer


class SnomedVectorStoreBuilder:
    """
    Builds a FAISS vector store for SNOMED CT or SNOMED-EL subsets.
    This is ontology-specific but pipeline-agnostic.
    """

    def __init__(
        self,
        rf2_path: Path,
        output_dir: Path,
        embedding_model_id: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str | None = None,
    ):
        self.rf2_path = rf2_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model_id = embedding_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SentenceTransformer(embedding_model_id, device=self.device)

    # ------------------------------------------------------------------
    # 1. Load SNOMED RF2 (adapted from challenge organizers)
    # ------------------------------------------------------------------
    def load_active_snomed(self) -> pd.DataFrame:
        def _read_active(file):
            df = pd.read_csv(file, sep="\t", dtype=str)
            return df[df["active"] == "1"]

        concepts = _read_active(
            self.rf2_path / "Snapshot/Terminology/sct2_Concept_Snapshot_INT_20230531.txt"
        )
        descs = _read_active(
            self.rf2_path / "Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20230531.txt"
        )

        merged = concepts.merge(
            descs, left_on="id", right_on="conceptId", how="inner"
        )

        merged = merged[["id_x", "term", "typeId"]].rename(
            columns={"id_x": "concept_id", "term": "name", "typeId": "name_type"}
        )

        # Preferred Term (P) or Synonym (A)
        merged["name_type"] = merged["name_type"].replace(
            {
                "900000000000003001": "P",
                "900000000000013009": "A",
            }
        )

        # Extract hierarchy from FSN
        merged["hierarchy"] = merged["name"].str.extract(r"\(([^)]+)\)$")

        return merged.dropna(subset=["hierarchy"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Filter to SNOMED-EL subset
    # ------------------------------------------------------------------
    def filter_to_el_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        allowed = {
            "finding",
            "disorder",
            "procedure",
            "body structure",
            "morphologic abnormality",
            "regime/therapy",
            "cell structure",
        }

        df = df[df["hierarchy"].isin(allowed)]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Attach synonyms per concept
    # ------------------------------------------------------------------
    def attach_synonyms(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = []

        for cid, g in df.groupby("concept_id"):
            pt = g[g["name_type"] == "P"]["name"].iloc[0]
            synonyms = g[g["name_type"] == "A"]["name"].tolist()

            grouped.append({
                "concept_id": cid,
                "concept_name": pt,
                "hierarchy": g["hierarchy"].iloc[0],
                "synonyms": synonyms,
            })

        return pd.DataFrame(grouped)

    # ------------------------------------------------------------------
    # 4. Build embeddings + FAISS
    # ------------------------------------------------------------------
    def build_vector_store(self, df: pd.DataFrame, prefix: str):
        df = df.reset_index(drop=True)

        def build_text(row):
            parts = [row["concept_name"]]
            parts.extend(row["synonyms"])
            return " ; ".join(parts)

        texts = df.apply(build_text, axis=1).tolist()

        embeddings = []
        for i in tqdm(range(0, len(texts), 512), desc="Embedding concepts"):
            batch = texts[i:i+512]
            emb = self.model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            embeddings.append(emb)

        embeddings = np.vstack(embeddings).astype("float32")

        # Save artifacts
        concept_path = self.output_dir / f"{prefix}_concepts.parquet"
        emb_path = self.output_dir / f"{prefix}_embeddings.npy"
        faiss_path = self.output_dir / f"{prefix}_faiss.index"

        df.to_parquet(concept_path, index=False)
        np.save(emb_path, embeddings)

        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(embeddings)

        faiss.write_index(index, str(faiss_path))

        return {
            "concept_path": concept_path,
            "embedding_path": emb_path,
            "faiss_path": faiss_path,
        }

    # ------------------------------------------------------------------
    # 5. One-call build (what you’ll usually use)
    # ------------------------------------------------------------------
    def build_snomed_el(self):
        df = self.load_active_snomed()
        df = self.filter_to_el_subset(df)
        df = self.attach_synonyms(df)

        return self.build_vector_store(df, prefix="snomedEL_subset")
