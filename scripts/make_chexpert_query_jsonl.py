"""
Make a CheXpert-only query JSONL for LLaVA-style evaluation.

This replicates the CheXpert preprocessing logic used for LLaVA-Rad training,
but without any GPT augmentation. It:
  - Loads CheXpert Plus and the preprocessed splits CSV
  - Filters/merges using `CheXpertPreprocessing` (same as in finetune scripts)
  - Builds LLaVA-style conversation prompts per study
  - Writes one JSON line per example with keys:
        reason, findings, impressions, image, generate_method,
        chexpert_labels, split, conversations

The output JSONL can be used as `--query_file` for `llava.eval.model_mimic_cxr`.
"""

import json
from pathlib import Path

import pandas as pd

from ChexpertPreprocessing import CheXpertPreprocessing


CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def create_labels_dict(row: pd.Series) -> dict:
    """Convert CheXpert labels to dictionary for a single row."""
    return {label: row[label] for label in CHEXPERT_LABELS if label in row}


def create_conversations_list(row: pd.Series) -> list[dict]:
    """Build LLaVA-style conversations list from CheXpert sections."""
    conversations: list[dict] = []
    output = ""
    if row.get("section_findings") and row["section_findings"] != "nan":
        output = (
            output + str(row["section_findings"]).replace("\n", "").strip() + " "
        )
    if row.get("section_impression") and row["section_impression"] != "nan":
        output = (
            output + str(row["section_impression"]).replace("\n", "").strip()
        )
    dict_findings = {"from": "gpt", "value": output}

    reason_text = row.get("section_indication", "")
    if isinstance(reason_text, str) and reason_text.strip():
        reason = reason_text.replace("\n", "")
        dict_reason = {
            "from": "human",
            "value": (
                "<image>\nProvide a description of the findings in the "
                f"radiology image given the following indication: {reason}"
            ),
        }
    else:
        dict_reason = {
            "from": "human",
            "value": (
                "<image>\nProvide a description of the findings in the "
                "radiology image."
            ),
        }
    conversations.append(dict_reason)
    conversations.append(dict_findings)
    return conversations


def main() -> None:
    root = Path("/lfs/skampere2/0/mahmedc/CUMC_Radiology")

    # Paths consistent with updated evaluation.py
    chexpert_plus_path = (
        root
        / "data/raw_data/chexpert/chexpertplus"
        / "df_chexpert_plus_240401.csv"
    )
    preprocessed_path = (
        root
        / "LLaVA-Rad/evaluation_results"
        / "preprocessed_chexpert.csv"
    )

    if not chexpert_plus_path.exists():
        raise FileNotFoundError(f"CheXpert Plus CSV not found at {chexpert_plus_path}")
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"preprocessed_chexpert.csv not found at {preprocessed_path}"
        )

    print(f"Reading splits from: {preprocessed_path}")
    splits = pd.read_csv(preprocessed_path)

    print(f"Reading CheXpert Plus from: {chexpert_plus_path}")
    chexpert_obj = CheXpertPreprocessing(str(chexpert_plus_path), splits)
    df = chexpert_obj.read_data()

    # Build indication text from history + clinical history (as in finetune_chexpert.py)
    df["section_indication"] = (
        df["section_history"].fillna("") + " " + df["section_clinical_history"].fillna("")
    )
    df["chexpert_labels"] = df.apply(create_labels_dict, axis=1)
    df["conversations"] = df.apply(create_conversations_list, axis=1)

    print("Sample path_to_image values:")
    print(df["path_to_image"].head())
    print("Split value counts:")
    print(df["split"].value_counts())
    print("Number of null findings:", df["section_findings"].isnull().sum())
    print("Number of null impressions:", df["section_impression"].isnull().sum())
    print("Total length after preprocessing:", len(df))

    # Build records in the same shape as process_input_json_file
    records: list[dict] = []
    for _, row in df.iterrows():
        rec = {
            "reason": row["section_indication"],
            "findings": row["section_findings"],
            "impressions": row["section_impression"],
            "examination": row.get("section_narrative", ""),
            "image": row["path_to_image"],
            "generate_method": "chexpert",
            "chexpert_labels": row["chexpert_labels"],
            "split": row["split"],
            "conversations": row["conversations"],
        }
        records.append(rec)

    out_path = root / "LLaVA-Rad/scripts/chexpert_query.jsonl"
    print(f"\nWriting {len(records)} records to: {out_path}")
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()

