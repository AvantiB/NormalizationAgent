import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import faiss
import torch
from sentence_transformers import SentenceTransformer

from utils.validate_snomed import validate_snomed_concepts


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
        use_synonyms: bool = True,   
    ):
        self.rf2_path = Path(rf2_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model_id = embedding_model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SentenceTransformer(embedding_model_id, device=self.device)
        self.use_synonyms = use_synonyms

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
            columns={"id_x": "concept_id", "term": "name", "typeId": "desc_type"}
        )

        # Preferred Term (P) or Synonym (A)
        merged["desc_type"] = merged["desc_type"].replace(
                {
                    "900000000000003001": "FSN",
                    "900000000000013009": "SYN",
                }
            )

        # Extract hierarchy from FSN
        # merged["hierarchy"] = merged["name"].str.extract(r"\(([^)]+)\)$")
        merged["hierarchy"] = None
        fsn_mask = merged["desc_type"] == "FSN"
        # Extract hierarchy tag from FSN (return Series via expand=False)
        merged.loc[fsn_mask, "hierarchy"] = (
            merged.loc[fsn_mask, "name"]
            .astype(str)
            .str.strip()
            .str.extract(r"\(([^)]+)\)$", expand=False)
        )

        # Fail fast if extraction is broken
        if merged.loc[fsn_mask, "hierarchy"].isna().all():
            # print a few raw FSNs to debug formatting quickly
            sample_fsns = merged.loc[fsn_mask, "name"].head(5).tolist()
            raise RuntimeError(
                "Hierarchy extraction failed for all FSNs. "
                f"Sample FSNs: {sample_fsns}"
            )

        # Build concept_id -> hierarchy map from FSN rows ONLY
        hier_map = (
            merged.loc[fsn_mask, ["concept_id", "hierarchy"]]
            .dropna(subset=["hierarchy"])
            .drop_duplicates("concept_id")
            .set_index("concept_id")["hierarchy"]
            .to_dict()
        )

        merged["hierarchy"] = merged["concept_id"].map(hier_map)

        if merged.loc[fsn_mask, "hierarchy"].isna().all():
            raise RuntimeError(
                "Hierarchy extraction failed for all FSNs. "
                "Check FSN regex or RF2 formatting."
            )
        print("After load_active_snomed:")
        print("  rows:", len(merged))
        print("  desc_type counts:\n", merged["desc_type"].value_counts())
        print("  hierarchy non-null:", merged["hierarchy"].notna().sum())

        return merged.dropna(subset=["hierarchy"]).reset_index(drop=True)
    
    # ------------------------------------------------------------------
    # 1b. Load SNOMED ISA relationships (parent mapping)
    # ------------------------------------------------------------------
    def load_parent_relationships(self) -> dict:
        """
        Load one-hop IS-A relationships:
        child_concept_id -> parent_concept_id
        """
        rel_file = (
            self.rf2_path
            / "Snapshot/Terminology/sct2_Relationship_Snapshot_INT_20230531.txt"
        )

        df = pd.read_csv(rel_file, sep="\t", dtype=str)

        # Active IS-A relationships only
        df = df[
            (df["active"] == "1") &
            (df["typeId"] == "116680003")  # IS-A
        ]

        # sourceId = child, destinationId = parent
        parent_map = {}

        for _, row in df.iterrows():
            child = row["sourceId"]
            parent = row["destinationId"]

            # Keep first parent only (intentional simplification)
            if child not in parent_map:
                parent_map[child] = parent

        return parent_map
    
    def compute_depth_map(self, parent_map: dict) -> dict:
        """
        Compute depth for each concept based on parent_map.
        """
        depth_map = {}

        def depth(cid):
            d = 0
            while cid in parent_map:
                cid = parent_map[cid]
                d += 1
            return d

        for cid in parent_map.keys():
            depth_map[cid] = depth(cid)

        return depth_map

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

        print("After filter_to_el_subset:")
        print("  rows:", len(df))
        print("  hierarchy counts:\n", df["hierarchy"].value_counts())

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Attach synonyms per concept
    # ------------------------------------------------------------------
    #ToDo: The synonyms returned are empty.. check the KIRI implementation on how to attach synonyms
    def attach_synonyms(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = []

        for cid, g in df.groupby("concept_id"):
            fsn = g[g["desc_type"] == "FSN"]["name"].iloc[0]
            fsn_clean = fsn.split(" (")[0]

            synonyms = (
                g[g["desc_type"] == "SYN"]["name"]
                .str.strip()
                .drop_duplicates()
                .tolist()
            )

            # Remove canonical name from synonyms (normal + lowercase safe)
            synonyms = [
                        s for s in synonyms
                        if s.lower() != fsn_clean.lower()
                    ]

            grouped.append({
                "concept_id": cid,
                "concept_name": fsn_clean,   # use FSN base as canonical
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
    # def build_snomed_el(self):
    #     df = self.load_active_snomed()
    #     df = self.filter_to_el_subset(df)
    #     df = self.attach_synonyms(df)

    #     return self.build_vector_store(df, prefix="snomedEL_subset")
    def build_snomed_el(self):
        df = self.load_active_snomed()
        df = self.filter_to_el_subset(df)
        print("DF COLUMNS BEFORE VALIDATION:", df.columns.tolist())

        if self.use_synonyms:
            df = self.attach_synonyms(df)
            issues = validate_snomed_concepts(df,sample_size=30)
        else:
            # Minimal concept table: FSN-only
            # FSN-only, but MUST be one row per concept_id (same canonicalization as attach_synonyms)
            fsn_df = df[df["desc_type"] == "FSN"].copy()

            # Keep exactly one FSN per concept_id (mirror attach_synonyms behavior)
            fsn_df = (
                fsn_df.sort_values(["concept_id", "name"])  # stable deterministic tie-break
                .drop_duplicates("concept_id", keep="first")
                .assign(
                    concept_name=lambda x: x["name"].astype(str).str.split(" (", n=1).str[0].str.strip(),
                    synonyms=lambda x: [[] for _ in range(len(x))],
                )[["concept_id", "concept_name", "hierarchy", "synonyms"]]
                .reset_index(drop=True)
            )

            df = fsn_df
        # df = self.attach_synonyms(df)
        
        # ---------- ontology metadata ----------
        parent_map = self.load_parent_relationships()
        depth_map = self.compute_depth_map(parent_map)

        # Map concept_id → name
        # concept_id → (name, hierarchy)
        concept_name_map = dict(zip(df["concept_id"], df["concept_name"]))
        concept_hierarchy_map = dict(zip(df["concept_id"], df["hierarchy"]))

        # Attach parent + depth + parent hierarchy
        df["parent_id"] = df["concept_id"].map(parent_map)
        df["parent_name"] = df["parent_id"].map(concept_name_map)
        df["parent_hierarchy"] = df["parent_id"].map(concept_hierarchy_map)
        df["depth"] = df["concept_id"].map(depth_map)
        suffix = "_with_synonyms" if self.use_synonyms else ""
        return self.build_vector_store(df, prefix=f"snomedEL_subset{suffix}")

