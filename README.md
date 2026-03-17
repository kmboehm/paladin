<center><img style="width: 50%" src="paladin.jpeg" alt="paladin"></center>

# paladin

<p align="center">
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-0.4.0-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

H&E WSI biomarker inference to predict treatment effects.

## Workshop Quickstart

Follow these steps to get paladin running on synthetic data in minutes.

### 1. Install

Requires **Python 3.11**. We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible installs.

```bash
# Install uv (if not already installed)
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and set up the project
git clone git@github.com:kmboehm/paladin.git
cd paladin
# --seed includes pip & setuptools in the venv (required: pytorch-lightning
# uses pkg_resources namespace packages to provide 'import lightning.pytorch')
uv venv --seed --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch (the --index-url replaces PyPI, so only torch packages go here)
# CPU-only (works everywhere):
uv pip install torch==2.4.1+cpu torchvision==0.19.1+cpu --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.1):
# uv pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install Lightning stack from PyPI
# setuptools<81 is required: lightning 2.3 uses pkg_resources (removed in 81+)
uv pip install lightning==2.3.0 torchmetrics==1.4.2 numpy==1.26.4 "setuptools<81"

# Install nn-template-core (provides nn_core module, installed without its
# transitive deps to avoid a lightning version conflict)
uv pip install --no-deps nn-template-core==0.4.0

# Install paladin and all remaining dependencies
uv pip install -e ".[dev]"
```

### 2. Set up Weights & Biases

Training logs to [Weights & Biases](https://wandb.ai). Create a free account at [wandb.ai](https://wandb.ai), then:

```bash
echo "YOUR_WANDB_API_KEY" > wandb.key
```

If no `wandb.key` file is found, paladin automatically falls back to **offline mode** so you can still train without a W&B account.

### 3. Generate synthetic data

```bash
python scripts/synthesize_workshop_data.py
```

This creates a `workshop_data/` directory containing:
- **`synthetic_data.parquet`** — sample DataFrame with 120 samples (60% train / 20% val / 20% test)
- **`tensors/`** — fake tile embedding `.pt` files (25 tiles x 1536 dims each)
- **`h5/`** — fake coordinate `.h5` files

Each sample has targets for all three task types (classification, regression, survival), so the same dataset works with all three configs below.

### 4. Train

The simple configs automatically use CPU if no GPU is available.

```bash
# Classification (binary biomarker prediction, beta-binomial loss)
python src/paladin/run.py --config-name simple-clf

# Regression (continuous value prediction, MSE loss)
python src/paladin/run.py --config-name simple-reg

# Survival analysis (Cox proportional hazards)
python src/paladin/run.py --config-name simple-surv
```

### 5. View results

Open your [W&B dashboard](https://wandb.ai) to see training curves, metrics, and logged artifacts. Each run creates a project named `paladin-workshop-clf`, `paladin-workshop-reg`, or `paladin-workshop-surv`.

---

## Bringing Your Own Data

To train on your own data, prepare a **parquet DataFrame** and tile embedding files, then point the config at them.

### Required DataFrame Columns

| Column | Type | Description |
|---|---|---|
| `sample_id` | str | Unique identifier for the tissue sample |
| `patient_id` | str | Patient identifier |
| `image_id` | str | Unique identifier for the H&E image |
| `oncotree_code` | str | Histology code (e.g., `"LUAD"`, `"BRCA"`) |
| `site` | str | `"Primary"`, `"Metastasis"`, `"Local Recurrence"`, or `"Unknown"` |
| `split` | str | `"train"`, `"val"`, or `"test"` |
| `filtered_tiles_h5_path` | str | Path to the H5 file with filtered tile coordinates |
| `{tile_tensor_url_col}` | str | Path to the tile embedding tensor (`.pt`) file. Column name is configured in the data config (default: `tile_tensor_path`) |

### Target Columns (task-dependent)

**Classification** (`task: classification`):

| Column | Type | Description |
|---|---|---|
| `{target_name}` | float | Binary target (0.0 or 1.0). NaN values exclude the sample. |

**Regression** (`task: regression`):

| Column | Type | Description |
|---|---|---|
| `{target_name}` | float | Continuous target value. NaN values exclude the sample. |

**Survival** (`task: survival`):

| Column | Type | Description |
|---|---|---|
| `{time_col}` | float | Time-to-event (e.g., months). Specified as the part before `:` in the target field. |
| `{event_col}` | float | Event indicator (1.0 = event, 0.0 = censored). Specified as the part after `:`. |

### Example: custom classification task

1. Create a data config (e.g., `conf/nn/data/my_clf.yaml`):

```yaml
_target_: paladin.data.joint_datamodule.JointDataModule
accelerator: ${train.trainer.accelerator}
num_workers:
  train: 0
  val: 0
  test: 0
batch_size:
  train: 4
  val: 4
  test: 4
dataset:
  _target_: paladin.utils.setup_simple_dataset.setup_dataset
  tile_emb_dim: 1536
  tile_tensor_url_col: my_embedding_col
  sample_df_path: /path/to/my_data.parquet
  max_seq_len: 5000
  tasks:
  - histologies:
    - LUAD
    sites:
    - Primary
    target:
    - TP53_MUTANT
    target_type:
    - binary
    task:
    - classification
defaults:
  - _self_
```

2. Create an nn config (e.g., `conf/nn/my_clf.yaml`):

```yaml
defaults:
  - data: my_clf
  - module: jointbb
  - _self_
```

3. Create a top-level config (e.g., `conf/my-clf.yaml`):

```yaml
core:
  project_name: my-project
  storage_dir: ${oc.env:PROJECT_ROOT}/storage
  version: 0.0.1
  tags: null
conventions:
  y_key: 'y'
defaults:
  - hydra: default
  - nn: my_clf
  - train: jointbb
  - _self_
```

4. Run:

```bash
python src/paladin/run.py --config-name my-clf
```

## Configuration

Paladin uses [Hydra](https://hydra.cc/) for hierarchical configuration. All config files live in `conf/`.

### Key Config Files

| File | Purpose |
|---|---|
| `conf/simple-clf.yaml` | Workshop: classification (beta-binomial) |
| `conf/simple-reg.yaml` | Workshop: regression (MSE) |
| `conf/simple-surv.yaml` | Workshop: survival (Cox) |
| `conf/default.yaml` | Default entry point (multi-cohort) |
| `conf/nn/module/jointbb.yaml` | Beta-binomial lightning module |
| `conf/nn/module/joint.yaml` | BCE + MSE lightning module |
| `conf/nn/module/cox.yaml` | Cox survival lightning module |
| `conf/nn/module/model/acontextual.yaml` | Standard aggregator model |
| `conf/train/jointbb.yaml` | Training config for classification |
| `conf/train/joint.yaml` | Training config for regression |
| `conf/train/cox.yaml` | Training config for survival |

## Task Types

### Classification (BCE / Beta-Binomial)

**Module**: `JointBetaBinomialLightningModule` (default) or `JointLightningModule`
**Loss**: Beta-binomial loss (default) or BCE with logits
**Metrics**: AUROC with bootstrapped 95% CIs, calibration plots, decision curves
**Config**: `--config-name simple-clf`

### Regression (MSE)

**Module**: `JointLightningModule`
**Loss**: Mean squared error (MSE)
**Metrics**: Pearson correlation with bootstrapped 95% CIs, MSE
**Config**: `--config-name simple-reg`

### Survival (Cox Proportional Hazards)

**Module**: `JointCoxLightningModule`
**Loss**: Negative Cox partial log-likelihood (Breslow approximation)
**Metrics**: Harrell's concordance index (c-index) with bootstrapped 95% CIs
**Callback**: `CoxTestMetricsCallback` — computes c-index per histology and split
**Config**: `--config-name simple-surv`


## Development

```bash
# Run tests
pytest -v

# Run linting
pre-commit run --all-files

# Update dependencies
uv pip install -e ".[dev]"
```
