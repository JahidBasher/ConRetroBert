# ConRetroBert: EMA-Stabilized Dual Encoders for Template-Based Single-Step Retrosynthesis

This repository contains the anonymous-review code package for the NeurIPS 2026 submission.

ConRetroBert is a template-based retrosynthesis framework trained in two stages:
1. Contrastive product-template representation learning.
2. Candidate-set listwise ranking with hard negatives.

The manuscript reports that Stage 2 provides the main gain over Stage 1, and EMA-stabilized
template adaptation further improves top-k performance on USPTO-50k.

## Abstract

Template-based single-step retrosynthesis predicts reactants by selecting and applying an explicit
reaction template, making each prediction traceable to a chemical transformation rule. This
interpretability is useful for synthesis planning, but template-based methods are often viewed as
less competitive than template-free models because template prediction is commonly formulated as
global classification over a long-tailed rule library. We argue that this weakness is not inherent to
templates, but to the learning formulation.

We present ConRetroBert, a dual-encoder framework that reframes template-based retrosynthesis as
dense product-template retrieval followed by candidate-set listwise ranking. In Stage 1, contrastive
pretraining learns a shared embedding space between products and reaction templates. In Stage 2, a
multi-positive listwise objective refines template ranking over mined hard-negative candidate sets,
matching the inference-time decision problem more closely than full-vocabulary classification. To
enable template-side adaptation without destabilizing hard-negative mining, ConRetroBert uses a
slow-moving exponential moving average (EMA) template encoder for retrieval bank construction
while updating the live template encoder through the ranking loss.

On the local USPTO-50k benchmark, the main gain comes from Stage 2 candidate-set ranking, which
improves top-1 reaction accuracy from 50.5% to about 61.3%, while EMA-stabilized template
adaptation provides a further improvement to about 62.4%. The local model reaches 81.6%, 85.3%,
and 87.8% at top-3, top-5, and top-10. We further show that retrieval-based template prediction is
especially strong in the long tail of rare templates, and that Stage 2 ranking improves applicability
of retrieved templates relative to contrastive retrieval alone. As a separate scaling result, fine-tuning
from a leakage-controlled USPTO-Full checkpoint reaches 75.4% top-1 accuracy on USPTO-50k.
These results show that template-based retrosynthesis can combine strong predictive performance
with chemically inspectable predictions.

## 1. Repository Scope

- Anonymous-review package (runnable code).
- Packaged local dataset under `data/uspto-50k/`.
- Training, evaluation, and batch-evaluation scripts for single-step retrosynthesis.

## 2. Environment

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

## 3. Packaged Data

The repository includes:

- `data/uspto-50k/raw_train.jsonl`
- `data/uspto-50k/raw_val.jsonl`
- `data/uspto-50k/raw_test.jsonl`
- `data/uspto-50k/merged_dataset_full.jsonl`
- `data/uspto-50k/merged_dataset_filtered.jsonl`

Minimum JSONL row format:

```json
{"product":"...", "template":"...", "split":"train"}
```

For evaluation, rows should include `reactants`.

## 4. Quickstart

Build tokenizer:

```bash
python scripts/prepare_tokenizer.py \
  --input data/uspto-50k/raw_train.jsonl \
  --input data/uspto-50k/raw_val.jsonl \
  --output configs/tokenizer.json \
  --fields product,template
```

## 5. Training (Manuscript Pipeline)

Stage 1 contrastive training:

```bash
python train_lightning.py --config configs/uspto_50k/stage1.yaml
```

Stage 2 listwise ranking (frozen template encoder variant):

```bash
python train_lightning.py \
  --config configs/uspto_50k/stage2_frozen.yaml \ # provide training.stage2.init_checkpoint
  --stage 2
```

Stage 2 EMA (trainable template encoder + EMA stabilization):

```bash
python train_lightning.py \
  --config configs/uspto_50k/stage2_ema.yaml \  # provide training.stage2.init_checkpoint
  --stage 2
```

Resume any run:

```bash
python train_lightning.py \
  --config configs/uspto_50k/stage2_ema.yaml \
  --stage 2 \
  --resume /path/to/last.ckpt
```

## 6. Evaluation

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

## 7. Config Mapping

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

## 8. Manuscript Alignment

The implementation follows the manuscript structure:

- Dual-encoder product/template architecture.
- Stage 1 contrastive pretraining.
- Stage 2 candidate-set listwise ranking with mined hard negatives.
- EMA-based stabilization for trainable template encoder retrieval updates.

