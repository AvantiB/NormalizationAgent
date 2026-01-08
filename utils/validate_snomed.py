import random
from typing import List
import pandas as pd


def validate_snomed_concepts(
    df: pd.DataFrame,
    sample_size: int = 20,
    min_synonyms: int = 1,
    seed: int = 42,
):
    """
    Validate SNOMED concept_name, hierarchy, and synonyms.

    This does NOT require rebuilding the vector DB.
    """

    random.seed(seed)

    print("=" * 80)
    print(f"SNOMED CONCEPT VALIDATION (sample_size={sample_size})")
    print("=" * 80)

    # Basic schema checks
    required_cols = {"concept_id", "concept_name", "hierarchy", "synonyms"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sample concepts
    sample = df.sample(min(sample_size, len(df)), random_state=seed)

    issues = []

    for _, row in sample.iterrows():
        cid = row["concept_id"]
        name = row["concept_name"]
        hierarchy = row["hierarchy"]
        synonyms: List[str] = row["synonyms"]

        print(f"\nConcept ID: {cid}")
        print(f"  concept_name : {name}")
        print(f"  hierarchy    : {hierarchy}")
        print(f"  synonyms ({len(synonyms)}): {synonyms[:10]}")

        # ---------- Checks ----------
        if "(" in name or ")" in name:
            issues.append((cid, "FSN not stripped from concept_name"))

        if not isinstance(synonyms, list):
            issues.append((cid, "synonyms is not a list"))

        if len(synonyms) < min_synonyms:
            issues.append((cid, f"too few synonyms (<{min_synonyms})"))

        if name in synonyms:
            issues.append((cid, "concept_name appears in synonyms"))

        if hierarchy not in {
            "finding",
            "disorder",
            "procedure",
            "body structure",
            "morphologic abnormality",
            "regime/therapy",
            "cell structure",
        }:
            issues.append((cid, f"unexpected hierarchy: {hierarchy}"))

    # ---------- Summary ----------
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if not issues:
        print("✅ No issues detected in sampled concepts.")
    else:
        print(f"⚠️  Detected {len(issues)} potential issues:")
        for cid, msg in issues:
            print(f"  - {cid}: {msg}")

    return issues
