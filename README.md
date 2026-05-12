# ConRetroBert: EMA-Stabilized Dual Encoders for Template-Based Single-Step Retrosynthesis

Public release of the ConRetroBert codebase for template-based single-step retrosynthesis.

ConRetroBert is a template-based retrosynthesis framework trained in two stages:
1. Contrastive product-template representation learning.
2. Candidate-set listwise ranking with hard negatives.

## 1. Released Artifacts

- Model checkpoints: <https://drive.google.com/drive/folders/1iSKIUiZZ8uA7KdYR62S6YjhV2eVBKKNj?usp=drive_link>
- Model dataset: <https://drive.google.com/file/d/1ahCJe1e594Meaxm7xJoU30O6idc0EY_q/view?usp=drive_link>

## Abstract

Template-based single step retrosynthesis predicts reactants by selecting and applying an explicit reaction template, making each prediction traceable to a chemical transformation rule. This interpretability is useful for synthesis planning, but template-based methods are often viewed as less competitive than template free models because template prediction is commonly formulated as global classification over a long tailed rule library. We argue that this weakness is not inherent to templates, but to the learning formulation. We present \textbf{ConRetroBert}, a dual encoder framework that reframes template-based retrosynthesis as dense product template retrieval followed by candidate set listwise ranking. In Stage 1, contrastive pretraining learns a shared embedding space between products and reaction templates. In Stage 2, a multi positive listwise objective refines template ranking over mined hard negative candidate sets, matching the inference time decision problem more closely than full vocabulary classification. To enable template side adaptation without destabilizing hard negative mining, ConRetroBert uses a slow moving exponential moving average, or EMA, template encoder for retrieval bank construction while updating the live template encoder through the ranking loss. On the local USPTO-50k benchmark, the main gain comes from Stage 2 candidate set ranking, which improves top-1 reaction accuracy from 50.5% to about 61.3%, while EMA stabilized template adaptation provides a further improvement to about 62.4%. The local model reaches 81.6%, 85.3%, and 87.8% at top-3, top-5, and top-10. We further show that retrieval based template prediction is especially strong in the long tail of rare templates, and that Stage 2 ranking improves the applicability of retrieved templates relative to contrastive retrieval alone. As a separate scaling result, fine tuning from a leakage controlled USPTO-Full checkpoint reaches 75.4% top-1 accuracy on USPTO-50k. These results show that template-based retrosynthesis can combine strong predictive performance with chemically inspectable predictions, challenging the common assumption that high accuracy requires abandoning explicit reaction templates.

## 2. Repository Scope

- Public runnable code for training and evaluation.
- Packaged local dataset under `data/uspto-50k/`.
- Training, evaluation, and batch-evaluation scripts for single-step retrosynthesis.

## 3. Environment

Requirements:
- Python >= 3.10
- Install dependencies:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install rdkit==2024.3.5
pip install faiss-gpu-cu12
pip install pytorch_lightning==2.4.0
pip install tensorboard
```

## 4. Packaged Data

The repository includes:

- `data/uspto-50k/raw_train.jsonl`
- `data/uspto-50k/raw_val.jsonl`
- `data/uspto-50k/raw_test.jsonl`
- `data/uspto-50k/merged_dataset_full.jsonl`
- `data/uspto-50k/merged_dataset_filtered.jsonl`

All five JSONL files use the same row schema (8 fields):

```json
{
  "reactants": "string (atom-mapped reactant SMILES, '.'-joined when multi-reactant)",
  "reagents": "string (currently empty string in packaged data)",
  "product": "string (atom-mapped product SMILES)",
  "template": "string (reaction SMARTS/transform, format: reactant_pattern>>product_pattern)",
  "id": "string (source reaction identifier; not unique across rows)",
  "class": "string (reaction class label; currently 'UNK' in packaged data)",
  "rxn_smiles": "string (reaction SMILES, format: reactants>>product)",
  "split": "string enum: train | val | test"
}
```

Schema and integrity notes inferred from packaged data:

- Required fields for training/retrieval: `product`, `template`, `split`.
- Required fields for reaction-level evaluation: `reactants` and `rxn_smiles`.
- In all packaged files, `rxn_smiles` is consistent with `reactants>>product`.

Per-file split semantics:

- `raw_train.jsonl`: all rows have `split = "train"` (39,992 rows).
- `raw_val.jsonl`: all rows have `split = "val"` (5,001 rows).
- `raw_test.jsonl`: all rows have `split = "test"` (5,005 rows).
- `merged_dataset_full.jsonl`: union of all three splits (49,998 rows).
- `merged_dataset_filtered.jsonl`: filtered subset of the merged dataset (48,755 rows).

## 5. Quickstart

Build tokenizer:

```bash
python scripts/prepare_tokenizer.py \
  --input data/uspto-50k/raw_train.jsonl \
  --input data/uspto-50k/raw_val.jsonl \
  --output configs/tokenizer.json \
  --fields product,template
```

## 6. Training (Manuscript Pipeline)

Stage 1 contrastive training:

```bash
python train_lightning.py --config configs/uspto_50k/stage1.yaml
```

Stage 2 listwise ranking (frozen template encoder variant):

```bash
python train_lightning.py \
  --config configs/uspto_50k/stage2_frozen.yaml \
  --stage 2
```
Set `training.stage2.init_checkpoint` in the selected config before running Stage 2.

Stage 2 EMA (trainable template encoder + EMA stabilization):

```bash
python train_lightning.py \
  --config configs/uspto_50k/stage2_ema.yaml \
  --stage 2
```
Set `training.stage2.init_checkpoint` in the selected config before running Stage 2.

Resume any run:

```bash
python train_lightning.py \
  --config configs/uspto_50k/stage2_ema.yaml \
  --stage 2 \
  --resume /path/to/last.ckpt
```

## 7. Evaluation

Build template cache:

```bash
python scripts/build_template_cache.py \
  --config configs/uspto_50k/stage2_frozen.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --templates data/uspto-50k/merged_dataset_full.jsonl \
  --output artifacts/template_cache.pt \
  --device cuda
```

Evaluate on packaged test split:

```bash
python onestep_retrosynthesis_evaluation.py \
  --config configs/uspto_50k/stage2_frozen.yaml \
  --checkpoint /path/to/checkpoint.ckpt \
  --cache artifacts/template_cache.pt \
  --eval_jsonl data/uspto-50k/raw_test.jsonl \
  --eval-k 1,3,5,10 \
  --summary \
  --summary-out artifacts/summary.json
```

Batch-evaluate many checkpoints:

```bash
python onestep_batch_retrosynthesis_evaluation.py /path/to/experiment_root \
  --templates data/uspto-50k/merged_dataset_full.jsonl \
  --eval-jsonl data/uspto-50k/raw_test.jsonl
```

## 8. Config Mapping

Relevant config files:

- `configs/uspto_50k/stage1.yaml`: Stage 1 contrastive training.
- `configs/uspto_50k/stage2_frozen.yaml`: Stage 2 listwise ranking baseline.
- `configs/uspto_50k/stage2_ema.yaml`: Stage 2 EMA-stabilized training.
- `configs/uspto_50k/base.yaml`: shared schema and defaults.

Note: some Stage-2 configs contain local output/checkpoint paths from internal runs.
Before long runs, verify these fields in your selected config:
- `checkpoint.dirpath`
- `training.output_dir`
- `training.stage2.init_checkpoint` (or pass `--pretrained`)

## 9. Public Release Notes

- Some Stage-2 configs may contain local output/checkpoint paths from internal runs.
- Verify config paths before training or evaluation in a new environment.
- For reproducibility, prefer storing generated artifacts (checkpoints, caches, summaries) under a dedicated `artifacts/` directory.

## 10. Manuscript Alignment

The implementation follows the manuscript structure:

- Dual-encoder product/template architecture.
- Stage 1 contrastive pretraining.
- Stage 2 candidate-set listwise ranking with mined hard negatives.
- EMA-based stabilization for trainable template encoder retrieval updates.

## 11. Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{conretrobert2026,
  title        = {ConRetroBert: EMA-Stabilized Dual Encoders for Template-Based Single-Step Retrosynthesis},
  author       = {Mohammad Jahid Ibna Basher and Ali Khodabandeh Yalabadi and Ivan Garibay and Ozlem Ozmen Garibay},
  year         = {2026},
  note         = {Public release},
  institution  = {University of Central Florida},
  address      = {Orlando, FL, USA}
}
```
