# Consumer Complaint Classification with Local LLMs

Code, dataset splits and experimental results for five-category consumer
financial complaint classification. Experiments cover local zero-shot
inference, QLoRA adaptation, supervised classifiers and thinking-on/off controls.

## Repository contents

| Directory | Contents |
| --- | --- |
| `utils/` | Data preparation, training, inference and statistical analysis |
| `dataset/` | Source CSV, fixed test subsets and training/validation splits |
| `preds/` | Predictions from the eleven-model zero-shot benchmark |
| `results/sadhana/` | Metrics, configurations, comparisons and run metadata |
| `results/sadhana/validation/` | Data integrity, overlap and split verification |
| `figures/` | Experimental plots |

## Environment

The recorded environment is provided in `environment-consumercomplaint.yml`;
Python package versions are recorded in `requirements-lock.txt`.

```bash
conda env create -f environment-consumercomplaint.yml
conda activate ConsumerComplaint
python -m pip check
```

GPU training requires a compatible CUDA/PyTorch installation. Zero-shot GGUF
inference and the thinking control require a local Ollama server and the
corresponding models. Commands below run from the repository root.

## Data and evaluation protocol

The source file is `dataset/complaints.csv`, obtained from the
[Consumer Complaints Dataset for NLP](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp).
The five categories are `credit_reporting`, `debt_collection`, `credit_card`,
`mortgages_and_loans` and `retail_banking`.

The benchmark samples complaints whose original narratives contain at most
500 characters. Paired 250-character views use the same records and labels.
Balanced and reduced-imbalance conditions partly overlap and are not
independent replications.

`dataset/finetuning/` contains nested balanced training sets of 500, 2,000,
10,000 and 20,000 records, a separate fixed 2,000-record validation set and a
SHA-256 manifest. These training files being available does not mean every
model was trained at every budget. Completed LLM adaptation uses 500 training
records and three training seeds; SVM covers all four training budgets.

Split preparation excludes normalized exact matches, matching 250-character
prefixes and detected near duplicates across training, validation and test
data. Near-duplicate retrieval uses MinHash LSH followed by word-3-shingle
Jaccard checks and is approximate; it cannot rule out all paraphrases.

Verify the saved splits without changing data or running a model:

```bash
python utils/audit_split_integrity.py --output results/sadhana/validation/split_integrity.json
```

Keep the supplied test CSVs unchanged when analysing the saved predictions.
`utils/build_experiment_subsets.py` constructs test sets and can overwrite them.
`utils/prepare_finetuning_data.py` constructs training splits and refuses to
overwrite an existing split directory.

## Training and inference

Examples for one training seed:

```bash
python utils/run_classical_baseline.py
python utils/train_encoder_baseline.py --train-size 20000 --seed 1
python utils/train_qlora.py --model gemma4 --train-size 500 --seed 1
python utils/train_qlora.py --model qwen3 --train-size 500 --seed 1
python utils/train_qlora.py --model phi4 --train-size 500 --seed 1
```

The QLoRA model identifiers are `google/gemma-4-12B-it`, `Qwen/Qwen3-14B` and
`microsoft/phi-4`. The protocol uses NF4 double quantization, BF16 computation,
LoRA rank 8, batch size 1, gradient accumulation 16, a 512-token sequence
limit and three epochs. Loss is computed on the assistant completion;
checkpoint selection uses validation loss. Model revisions, settings, resource
measurements and adapter hashes are recorded with each run. Use `--revision`
with the recorded HF commit when reproducing a run and `--resume` for an
interrupted compatible training run.

Evaluate a matched frozen-base/adapter pair:

```bash
python utils/evaluate_qlora_pair.py --help
```

This evaluator uses the same HF model revision, inputs and settings in both
modes. Serial latency is measured separately from batched test inference.
The Ollama thinking control is available through
`python utils/run_thinking_control.py --help`. It uses fixed paired inputs,
model artifacts and output-token budgets.

## Analysis and saved results

```bash
python utils/sadhana_audit.py --bootstrap 5000
python utils/summarize_sadhana_baselines.py
python utils/summarize_qlora_comparison.py --help
python utils/verify_qlora_evaluation.py --help
python utils/summarize_replications.py --help
python utils/finetuning_status.py
```

Multi-seed summaries are in `results/sadhana/replication_20261008/`, including
`qlora_seed_summary.csv`, `encoder_summary.csv` and
`label_matched_comparisons.csv`. Thinking-control results are in
`results/sadhana/thinking_control_20261009/`. Invalid outputs count as errors.
Between-fit variation and paired test-set uncertainty are reported separately.

Source data, fixed splits, zero-shot predictions, compact metrics and method
records are versioned. Model checkpoints, large generated HF prediction files,
raw JSONL responses and runtime logs are excluded. Full HF recomputation and
verification require those outputs to be regenerated or supplied separately;
the tracked metric summaries do not replace per-example predictions.
