# MedGemma Evaluation: Step-by-Step Process

## Objective
Evaluate MedGemma-1.5 4B on CheXpert dataset using the **same methodology** as LLaVA-Rad to ensure fair comparison.

**Results to Report**:
- Macro-F1-14: **0.2800** (CI: 0.2461 - 0.3211)
- Micro-F1-14: **0.5005** (CI: 0.4638 - 0.5367)
- Macro-F1-5: **0.3185** (CI: 0.2486 - 0.3710)
- Micro-F1-5: **0.4036** (CI: 0.3293 - 0.4731)


---
## Step 1: Data Preparation

### 1.1 Create Preprocessed Splits File
**Script**: `evaluation_results/create_preprocessed_chexpert.py`
- **Purpose**: Generate `preprocessed_chexpert.csv` that defines train/validation/test splits
- **Input**: CheXpert `train.csv` and `valid.csv`
- **Output**: `preprocessed_chexpert.csv` with columns: `path`, `split`
- **Why needed**: Used by multiple scripts to filter data by split

### 1.2 Generate Query JSONL File
**Script**: `scripts/make_chexpert_query_jsonl.py`
- **Purpose**: Create `chexpert_query.jsonl` in LLaVA-Rad format for evaluation
- **Input**: 
  - CheXpert Plus CSV (`df_chexpert_plus_240401.csv`)
  - Preprocessed splits CSV (`preprocessed_chexpert.csv`)
- **Process**:
  - Uses `CheXpertPreprocessing` class (same as LLaVA-Rad training)
  - Extracts findings, impressions, and indications from CheXpert Plus
  - Creates LLaVA-style conversation format: `[{"from": "human", "value": "<image>\n...indication..."}, {"from": "gpt", "value": "findings + impressions"}]`
- **Output**: `scripts/chexpert_query.jsonl` with keys: `image`, `conversations`, `split`, etc.
- **Why needed**: Standardized input format for model inference

---

## Step 2: Model Inference

### 2.1 Run MedGemma Inference
**Script**: `scripts/run_medgemma_inference.py`
**Orchestration**: `scripts/eval_medgemma.sh`

- **Purpose**: Generate predictions from MedGemma model on CheXpert validation set
- **Input**: 
  - `chexpert_query.jsonl` (from Step 1.2)
  - CheXpert images from `CheXpert-v1.0/` directory
- **Process**:
  1. Load MedGemma model (`google/medgemma-1.5-4b-it`) and processor
  2. Filter queries to `split == "valid"` only
  3. For each image:
     - Load and preprocess image
     - Format prompt using MedGemma's chat template (`processor.apply_chat_template`)
     - Generate prediction with `model.generate()`
     - Extract generated text (remove chat template artifacts)
  4. Write to `predictions.jsonl` in LLaVA-Rad format:
     ```json
     {
       "image": "valid/patient64620/study1/view1_frontal.jpg",
       "query": "Provide a description...",
       "reference": "ground truth report",
       "prediction": "generated report",
       "generation loss": 0.0
     }
     ```
- **Output**: `evaluation_results/medgemma_base/predictions.jsonl`
- **Key details**:
  - Uses MedGemma's specific chat template format
  - Handles image token replacement (`<image>` → `<image_soft_token>`)
  - Removes chat template prefixes from generated text

---

## Step 3: Run LLaVA-Rad Native Evaluation

### 3.1 Run Official LLaVA-Rad Evaluation Pipeline
**Script**: `evaluation_results/run_llavarad_evaluation.py`
- **Purpose**: Use LLaVA-Rad's official evaluation script (`llava/eval/rrg_eval/run.py`)
- **Input**: `predictions.jsonl`
- **Process**:
  1. Load predictions and references from JSONL
  2. Call `rrg_eval.chexbert.evaluate(predictions, references, ...)`
  3. **Critical Methodology: CheXbert Labels BOTH Predictions AND References**
     - **Why this matters**: CheXbert is a BERT-based labeler that extracts 14 medical conditions from radiology reports
     - **Key implementation**: `model(preds + refs)` in `chexbert.py` (line 300)
       - Both predictions and references are passed together to CheXbert
       - CheXbert labels all reports in a single batch
       - Then splits: `binary_rets[:len(preds)]` for predictions, `binary_rets[len(preds):]` for references
     - **Why this ensures fair comparison**:
       - Both predictions and ground truth references are labeled using the same CheXbert model
       - This eliminates bias from using different labeling methods
       - Original CheXpert labels (from `valid.csv`) are NOT used - only CheXbert-labeled references
     - **Label mapping**: CheXbert outputs 0=blank, 1=negative, 2=uncertain, 3=positive
       - Converted to binary: 3 (positive) → 1, all others → 0 (mode='rrg')
       - For "+" variants: uncertain (2) is also mapped to 1
  4. Compute F1 scores:
     - Micro-F1-14, Macro-F1-14 (14 conditions)
     - Micro-F1-5, Macro-F1-5 (5 key conditions)
     - Both "uncertain as negative" (-) and "uncertain as positive" (+) variants
  5. Bootstrap confidence intervals (500 resamples)
- **Output**: `evaluation_results/medgemma_base/eval/`
  - `main.csv` - Main results with median and CI
  - `breakdown_p.csv` - Per-condition breakdown (uncertain as positive)
  - `breakdown_n.csv` - Per-condition breakdown (uncertain as negative)
- **Results** (from `main.csv`):
  - Macro-F1-14: **0.2800** (CI: 0.2461 - 0.3211)
  - Micro-F1-14: **0.5005** (CI: 0.4638 - 0.5367)
  - Macro-F1-5: **0.3185** (CI: 0.2486 - 0.3710)
  - Micro-F1-5: **0.4036** (CI: 0.3293 - 0.4731)
- **Why this is the method**: This is the exact methodology used in LLaVA-Rad paper

---

## Step 4: Model Comparison

### 4.1 Compare All Models
**Script**: `evaluation_results/compare_all_models.py`
- **Purpose**: Compare MedGemma vs LLaVA-Rad Base vs LLaVA-Rad Finetuned
- **Input**: 
  - `medgemma_base/eval/main.csv`
  - `llavarad_base/eval/main.csv`
  - `llavarad_finetuned/eval/main.csv`
- **Output**:
  - Console output with comparison table
  - `model_comparison_detailed.csv` with confidence intervals
- **Features**:
  - Shows all 8 F1 metrics (Micro/Macro F1-14, F1-5, with +/- variants)
  - Calculates differences (Δ) between models
  - Shows confidence intervals and statistical significance (CI overlap)

---

## Critical Methodology Points

1. **CheXbert labels BOTH predictions and references**: This is the key alignment point for fair comparison.
   - **Implementation**: In `rrg_eval/rrg_eval/chexbert.py`, the `evaluate()` function calls `model(preds + refs)` (line 300)
   - **Process**: 
     - All reports (predictions + references) are passed to CheXbert in a single batch
     - CheXbert extracts 14 medical conditions from each report
     - Results are split: first `len(preds)` are predictions, remaining are references
   - **Why critical**: 
     - Both predictions and ground truth are labeled using the same model (CheXbert)
     - Original CheXpert labels from `valid.csv` are NOT used as ground truth
     - This ensures the evaluation compares "apples to apples" - both labeled by CheXbert
   - **Label conversion**: CheXbert outputs are mapped to binary (positive=1, others=0) for F1 calculation

2. **Same evaluation script**: Both use `llava/eval/rrg_eval/run.py` with same parameters.

3. **Same metrics**: Micro-F1-14, Macro-F1-14, Micro-F1-5, Macro-F1-5 (with +/- variants).

4. **Same bootstrap CI**: 500 resamples for confidence intervals.

5. **Same data**: CheXpert validation split (234 samples).

---

## Pipeline Summary

```
1. Data Prep
   ├─ create_preprocessed_chexpert.py → preprocessed_chexpert.csv
   └─ make_chexpert_query_jsonl.py → chexpert_query.jsonl

2. Inference
   └─ run_medgemma_inference.py → predictions.jsonl

3. Evaluation (Primary)
   └─ run_llavarad_evaluation.py → eval/main.csv, breakdown_*.csv

4. Comparison
   └─ compare_all_models.py → model_comparison_detailed.csv
```

---