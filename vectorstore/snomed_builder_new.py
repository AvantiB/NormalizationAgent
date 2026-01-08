import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import faiss
import torch
from sentence_transformers import SentenceTransformer


class SnomedVectorStoreBuilder:
    """
    Deterministic SNOMED-EL vector store builder.

    Invariant:
      - concept_id set is identical regardless of use_synonyms
      - concept_name and hierarchy are identical
      - only the 'synonyms' column changes
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

        self.use_synonyms = use_synonyms
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(embedding_model_id, device=self.device)

    # ------------------------------------------------------------------
    # 1. Load RF2 (raw, no logic)
    # ------------------------------------------------------------------
    def _read_active(self, file: Path) -> pd.DataFrame:
        df = pd.read_csv(file, sep="\t", dtype=str)
        return df[df["active"] == "1"]

    def load_rf2(self) -> pd.DataFrame:
        concepts = self._read_active(
            self.rf2_path / "Snapshot/Terminology/sct2_Concept_Snapshot_INT_20230531.txt"
        )
        descs = self._read_active(
            self.rf2_path / "Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20230531.txt"
        )

        df = concepts.merge(descs, left_on="id", right_on="conceptId", how="inner")

        df = df[["id_x", "term", "typeId"]].rename(
            columns={"id_x": "concept_id", "term": "name", "typeId": "desc_type"}
        )

        df["desc_type"] = df["desc_type"].replace(
            {
                "900000000000003001": "FSN",
                "900000000000013009": "SYN",
            }
        )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Canonical ontology construction (THIS NEVER CHANGES)
    # ------------------------------------------------------------------
    def build_ontology(self, rf2: pd.DataFrame) -> pd.DataFrame:
        fsn = rf2[rf2["desc_type"] == "FSN"].copy()

        fsn["hierarchy"] = (
            fsn["name"]
            .astype(str)
            .str.extract(r"\(([^)]+)\)$", expand=False)
        )

        fsn = (
            fsn
            .dropna(subset=["hierarchy"])
            .sort_values(["concept_id", "name"], kind="mergesort")
            .drop_duplicates("concept_id", keep="first")
            .assign(
                concept_name=lambda x: (
                                        x["name"]
                                        .astype(str)
                                        .str.split(" (", n=1, regex=False)
                                        .str[0]
                                        .str.strip()
                                    )
            )
            [["concept_id", "concept_name", "hierarchy"]]
            .reset_index(drop=True)
        )

        allowed = {
            "finding",
            "disorder",
            "procedure",
            "body structure",
            "morphologic abnormality",
            "regime/therapy",
            "cell structure",
        }

        fsn = fsn[fsn["hierarchy"].isin(allowed)].reset_index(drop=True)
        return fsn

    # ------------------------------------------------------------------
    # 3. Synonyms (OPTIONAL, REPRESENTATION ONLY)
    # ------------------------------------------------------------------
    def attach_synonyms(self, ontology: pd.DataFrame, rf2: pd.DataFrame) -> pd.DataFrame:
        if not self.use_synonyms:
            ontology["synonyms"] = [[] for _ in range(len(ontology))]
            return ontology

        syn = rf2[rf2["desc_type"] == "SYN"][["concept_id", "name"]].copy()

        syn_map = (
            syn.groupby("concept_id")["name"]
            .apply(
                lambda s: sorted(
                    {
                        n.strip()
                        for n in s
                        if isinstance(n, str)
                    }
                )
            )
        )

        canonical_map = (
            ontology.set_index("concept_id")["concept_name"]
            .str.lower()
            .to_dict()
        )

        ontology["synonyms"] = ontology["concept_id"].map(
            lambda cid: [
                s for s in syn_map.get(cid, [])
                if s.lower() != canonical_map.get(cid, "")
            ]
        )

        return ontology


    # def attach_synonyms(self, ontology: pd.DataFrame, rf2: pd.DataFrame) -> pd.DataFrame:
    #     if not self.use_synonyms:
    #         ontology["synonyms"] = [[] for _ in range(len(ontology))]
    #         return ontology

    #     syn = rf2[rf2["desc_type"] == "SYN"].copy()

    #     syn_map = (
    #         syn.groupby("concept_id")["name"]
    #         .apply(
    #             lambda s: sorted(
    #                 {
    #                     n.strip()
    #                     for n in s
    #                     if isinstance(n, str)
    #                 }
    #             )
    #         )
    #         .to_dict()
    #     )

    #     ontology["synonyms"] = ontology["concept_id"].map(
    #         lambda cid: [
    #             s for s in syn_map.get(cid, [])
    #             if s.lower() != ontology.loc[
    #                 ontology["concept_id"] == cid, "concept_name"
    #             ].iloc[0].lower()
    #         ]
    #     )

    #     return ontology

    # ------------------------------------------------------------------
    # 4. Parent + depth (ontology metadata)
    # ------------------------------------------------------------------
    def load_parent_relationships(self) -> dict:
        rel = pd.read_csv(
            self.rf2_path
            / "Snapshot/Terminology/sct2_Relationship_Snapshot_INT_20230531.txt",
            sep="\t",
            dtype=str,
        )

        rel = rel[(rel["active"] == "1") & (rel["typeId"] == "116680003")]

        parent_map = {}
        for _, r in rel.iterrows():
            parent_map.setdefault(r["sourceId"], r["destinationId"])

        return parent_map

    def attach_parent_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        parent_map = self.load_parent_relationships()

        def depth(cid):
            d = 0
            while cid in parent_map:
                cid = parent_map[cid]
                d += 1
            return d

        name_map = dict(zip(df["concept_id"], df["concept_name"]))
        hier_map = dict(zip(df["concept_id"], df["hierarchy"]))

        df["parent_id"] = df["concept_id"].map(parent_map)
        df["parent_name"] = df["parent_id"].map(name_map)
        df["parent_hierarchy"] = df["parent_id"].map(hier_map)
        df["depth"] = df["concept_id"].map(depth)

        return df

    # ------------------------------------------------------------------
    # 5. Vector store (representation only)
    # ------------------------------------------------------------------
    def build_vector_store(self, df: pd.DataFrame, prefix: str):
        def text(row):
            parts = [row["concept_name"]]
            parts.extend(row["synonyms"])
            return " ; ".join(parts)

        texts = df.apply(text, axis=1).tolist()

        embeddings = []
        for i in tqdm(range(0, len(texts), 512), desc="Embedding"):
            embeddings.append(
                self.model.encode(
                    texts[i:i+512],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
            )

        emb = np.vstack(embeddings).astype("float32")

        concept_path = self.output_dir / f"{prefix}_concepts.parquet"
        emb_path = self.output_dir / f"{prefix}_embeddings.npy"
        index_path = self.output_dir / f"{prefix}_faiss.index"

        df.to_parquet(concept_path, index=False)
        np.save(emb_path, emb)

        index = faiss.IndexHNSWFlat(emb.shape[1], 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        index.add(emb)

        faiss.write_index(index, str(index_path))

        return {
            "concepts": concept_path,
            "embeddings": emb_path,
            "index": index_path,
        }

    # ------------------------------------------------------------------
    # 6. One-call API
    # ------------------------------------------------------------------
    def build(self):
        rf2 = self.load_rf2()
        ontology = self.build_ontology(rf2)
        ontology = self.attach_synonyms(ontology, rf2)
        ontology = self.attach_parent_metadata(ontology)
        ontology = ontology.sort_values("concept_id").reset_index(drop=True)

        suffix = "_with_synonyms" if self.use_synonyms else "_nosyn"
        return self.build_vector_store(ontology, f"snomedEL{suffix}")
