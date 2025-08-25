# df-deploy

A command-line tool for deploying and evaluating machine learning models created with 
[`df-analyze`](https://github.com/stfxecutables/df-analyze). Supports optional integration with 
[`DeepTune`](https://github.com/moayadeldin/deeptune-beta) for embedding extraction using deep learning 
models trained using the `DeepTune` software.

## Overview

`df-deploy` lets you:

- Train ready-to-use machine learning models using the optimal hyperparameters discovered by `df-analyze`.
- Automatically reproduce `df-analyze`'s internal data cleaning and feature selection pipeline.
- Optionally integrate a pretrained `DeepTune` embedding model.
- Generate predictions and evaluation metrics on unseen datasets.

Use:
- `df-deploy.py` to create a deployable model directory.
- `df-predict.py` to generate predictions and/or evaluate on new data.



## Installation

Install the required packages (editable installs recommended if developing locally):

**NOTE** - Only tested using editable installs so far because unpublished local changes were required for 
packaging and functionality.

```shell
pip install -e /path/to/df-analyze
pip install -e /path/to/DeepTune   # Optional
```


## Quickstart

```bash
# Deploy models based on df-analyze output
python df-deploy.py --df data.csv --results-dir df_analyze_output --out deployment/

# Generate predictions on a test set
python df-predict.py --df test.csv --model-dir deployment/ --out predictions/
```

### Really Quick Start
To produce deployable ML models and generate predictions + evaluation on a test set, run:
```bash
python df-deploy.py \
    --df data.csv \
    --holdout-df test.csv \
    --results-dir df_analyze_output \
    --out deployment/
```


## Usage

### `df-deploy`: Model Deployment

For full documentation of `df-deploy` run:
```shell
python df-deploy.py -h
```

#### Required Arguments

- `--df DF`  
  - Path to the dataset file originally used with df-analyze.
- `--results-dir RESULTS_DIR`  
  - Directory containing the output of df-analyze, used to reconstruct and retrain the best-performing models.
- `--out OUT`  
  - Output directory where the deployment bundle (models, processors, metadata) will be saved.

#### Optional Arguments

- `--cls-metric {acc, auroc, sens, spec, ppv, npv, f1, bal-acc}`  
  - Metric to select the best classification model(s). Default: `acc`. **Only needed for classification tasks**.
- `--reg-metric {mae, msqe, mdae, r2, var-exp}`  
  - Metric to select the best regression model(s). Default: `mae`. **Only needed for regression tasks**.
- `--top TOP`  
  - Number of top-performing models to deploy. Use an integer (e.g., 3) or "all"; Default is `1`.
- `--ranking-method {holdout, 5-fold, holdout_average}`  
  - Specify how models were ranked in the df-analyze results. Default is `holdout_average`.
- `--exclude [{dummy, elastic, knn, lgbm, lr, mlp, rf, sgd} ...]`  
  - Exclude specific model types from deployment. This is useful for skipping slower or undesirable models.

#### Optional Inputs for Enhanced Deployment

- `--holdout-df HOLDOUT_DF`  
  - Optional dataset used for evaluating the final model(s). Must match the original structure used in `df-analyze`.
- `--embedding-model EMBEDDING_MODEL`  
  - Path to training directory produced by DeepTune for the model trained on the same data from the same dataset (excluding holdout/test) and used to embed the data given to `df-analyze`.
  - When enabled, the embedding model will be automatically applied to raw input features (e.g., a `text` or `image` column). The output of the model is used as input features for the downstream tabular ML models.

    **Expected structure**
    ```
    📂 deeptune_train_dir/
    ├── cli_arguments.json
    ├── model_weights.pth
    └── ...
    ```


#### Example

```shell
python df-deploy.py \
  --df example_data.parquet \
  --results-dir results_directory \
  --cls-metric acc \
  --reg-metric r2 \
  --ranking-method holdout_average \
  --out test_deployment_output \
  --holdout-df test_data.parquet \
  --embedding-model deeptune_train_output
```

#### Output Structure

The output produced is primarily to facilitate loading the deployed models for prediction using `df-predict`.
```
📂 deployment_output/
└── 📂 Month_dd_yyyy_hhmmss/
    ├── meta.json
    ├── feature_selections.json
    ├── 📂 (optional) embedding_model/
    │   ├── cli_arguments.json
    │   ├── model_weights.pth
    │   └── ...
    ├── 📂 processor/
    │   ├── processor.pkl
    │   ├── processor_dict.json       # Fallback if loading pickled processor fails
    │   ├── imputer.pkl               # The only attribute required to be pickled.
    │   └── ...
    ├── 📂 trained_models/
    │   ├── model_info.json
    │   ├── model_selection_embed.pkl
    │   └── ...
    └── 📂 (optional) predictions/
        ├── metrics.csv
        ├── metrics.json
        ├── performance_deltas.json
        ├── predictions_with_inputs.csv
        ├── predictions.csv
        └── 📂 predict_proba/
            ├── model_selection_embed_predict_proba.csv
            └── ...
```

---

### `df-predict`: Use Deployed Models for Prediction or Evaluation

Use `df-predict` to generate predictions on an input dataset, and produce evaluation metrics 
if a ground-truth target column is present. The input dataset **must** have the same format 
(column names and types) as the dataset given to `df-analyze` and used to train the deployed models.

If a dataset requires tabular embedding and an embedding model directory was specified when running `df-deploy`, 
the embedding model is automatically used. The input dataset must be therefore be in raw un-embedded format. 
This means the dataset contains an `image` column of image bytes or a `text` column of NLP data.

For full documentation of `df-predict` run:
```shell
python df-predict.py -h
```

#### Required Arguments

- `--df DF`  
  - Path to the dataset file used for final evaluation or inference. This must contain the same features (and target column, if available) used during training.
- `--model-dir MODEL_DIR`  
  - Directory containing the output from `df-deploy.py`.

#### Optional Arguments

- `--model-name MODEL_NAME`  
  - Specific model to use for prediction. This must match a file in `trained_models/`. If omitted, the top-performing model from `models_info.json` is used.

    **Expected format**:
    `[model_shortname]_[feature_selection]`  
    **Examples**:
    - `lgbm_embed_lgbm`
    - `rf_embed_linear`
    - `lr_no_select`

- `--out OUT`  
  - Path to the output directory where predictions and metrics will be saved. If the directory does not exist, it will be created.
  - If omitted, the `predictions` subdirectory in `model-dir` will be used or created.

#### Example

```shell
python df-predict.py \
    --df test_dataset.parquet \
    --model-dir test_deployment_output \
    --model-name rf_embed_linear \
    --out test_predict_output
```

<!-- #### Output Structure -->

#### 🧾 Evaluation Output

Evaluation output is produced when the input data contains the target column.

The output structure is the same as the evaluation output produced using `df-deploy` on a holdout set, with the addition of a timestamp directory for organisation. 

```
📂 evaluation_output/
└── 📂 Month_dd_yyyy_hhmmss/
    ├── metrics.csv                   # Tabular summary of model performance (e.g., acc, f1, r2)
    ├── metrics.json                  # Same metrics in machine-readable JSON format
    ├── performance_deltas.json       # Difference in performance compared to df-analyze selection
    ├── predictions.csv               # Final predictions and true targets for each sample
    ├── predictions_with_inputs.csv   # All non-embedding features + predicted and true targets
    └── 📂 predict_proba/
        ├── model_selection_embed_predict_proba.csv
        └── ...
```

#### 🔮 Inference Output (No ground-truth provided)

For **pure inference** (no target column is present), evaluation files are **not** produced and the prediction files do not contain `y_true` column.

```
📂 prediction_output/
└── 📂 Month_dd_yyyy_hhmmss/
    ├── predictions.csv
    ├── predictions_with_inputs.csv
    └── 📂 predict_proba/
        ├── model_selection_embed_predict_proba.csv
        └── ...
```


## Notes on Data Splits and Deployment Strategy

To ensure consistency between feature selection, hyperparameter optimization, and model ranking, it is recommended to deploy models trained on the **same dataset** used as input to `df-analyze`.

New data should be reserved for testing the generalisability of the deployed model(s). If performance on the new data degrades significantly, a new pipeline run should be initiated:

    DeepTune (optional) → df-analyze → df-deploy

You can then evaluate the final model(s) on the held-out dataset using `df-predict`.

---

### Recommended Data Split Strategy

#### 📊 Tabular Data

```
full_dataset → training | holdout 
```
- **training**: used as input to `df-analyze`, then to `df-deploy`.
- **holdout**: used exclusively for evaluating deployed model(s) via `df-predict`.


#### 🧠 Embedding Text/Image Data using `DeepTune`

```
full_dataset → training | holdout
training     → train | valid | test
```

- **train + valid**: used to train the `DeepTune` model.
- **test**: passed through the trained `DeepTune` model to obtain tabular embeddings.
  - These embeddings (and any other tabular features) are used as input to `df-analyze`, then `df-deploy`.
- **holdout**: remains untouched until final model evaluation (`df-predict`).

---

#### ⚠️ Why not use `DeepTune` training data for `df-analyze`?

Using the same data that trained the `DeepTune` model for embedding may introduce **information leakage** and overly optimistic performance metrics.

Instead, only the **`DeepTune` test split** should be embedded and used for downstream model development in `df-analyze` and `df-deploy`. This ensures evaluation metrics reflect **true generalization performance**.


**Note**: Although `df-deploy` uses `DeepTune`'s embedding functionality internally, there may be some discrepency between ebeddings obtained from directly using the `DeepTune` software and the embeddings obtained when using `df-deploy` for prediction.



## Citation

```bibtex
@software{df-deploy,
  author = {John Kendall},
  title = {df-deploy},
  year = {2025},
  url = {https://github.com/[repo-TBD]},
  version = {0.0.1}
}
```