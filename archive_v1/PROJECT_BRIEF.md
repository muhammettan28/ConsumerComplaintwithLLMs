# Project Brief — Zero-Shot Consumer Complaint Classification with Open-Source LLMs

> **For the coding agent:** This document is self-contained. Read it fully before
> writing code. It describes the research goal, the methodology to replicate, the
> compute environment (a Turkish HPC cluster + a local PC), and the exact scripts
> you should help build. Ask before deviating from the methodology — comparability
> with the reference paper matters.

---

## 1. Objective

Adapt and extend the methodology of:

> Roumeliotis, K.I.; Tselikas, N.D.; Nasiopoulos, D.K. *Think Before You Classify:
> The Rise of Reasoning Large Language Models for Consumer Complaint Detection and
> Classification.* Electronics 2025, 14, 1070.

**End product:** a journal article targeting **TR Dizin** (Turkish national academic
index). The paper will be written in Turkish; the dataset and models are English.

**Novelty / contribution (the angle that makes this publishable):**
The reference paper benchmarked mostly **proprietary, paid-API models** (OpenAI GPT-4o/4.5,
Anthropic Claude 3.5/3.7, Google Gemini) plus a few reasoning models (o1, o3-mini,
DeepSeek-R1). We instead benchmark **open-source / open-weight models that run locally
on GPU at zero API cost**, including **open reasoning models**, and answer:

1. How close do free, self-hostable open models get to the proprietary models the paper used?
2. Among open models, how do **reasoning** models (chain-of-thought) compare to **standard** ones — in accuracy *and* latency?
3. *(Optional sub-study)* How does **quantization** (q4 vs q8 vs fp16) affect classification accuracy? (Easy to do on the cluster; not in the original paper.)

We deliberately reuse the **same dataset** as the paper so our numbers are comparable.

---

## 2. Task definition (identical to the paper)

- **Zero-shot** classification (no fine-tuning) of consumer-complaint free text.
- Five target categories:
  `retail_banking`, `credit_reporting`, `credit_card`, `mortgages_and_loans`, `debt_collection`.
- Input = complaint **narrative text only**. Output = exactly one category as JSON.
- The prompt is **model-agnostic** (same prompt for every model). Use this prompt:

```
You are an AI assistant specializing in consumer complaint classification.
Your task is to analyze a consumer complaint and classify it into the most
appropriate category from the predefined list:
["retail_banking", "credit_reporting", "credit_card", "mortgages_and_loans", "debt_collection"]
Provide your final classification in the following JSON format without explanations:
{"product": "chosen_category_name"}.
Complaint: <NARRATIVE TEXT HERE>
```

---

## 3. Dataset

- Source: Kaggle "Consumer Complaints Dataset for NLP" (by shashwatwork), derived from
  the US CFPB Consumer Complaint Database. Period: **March 2020 – March 2021** (~162k rows).
- File: `complaints_processed.csv` (~95 MB). **Already downloaded to the local PC.**
- Columns: an unnamed index column, `product` (the true label), `narrative` (complaint text).
  Header line is literally: `,product,narrative`
- The `product` column appears to already be consolidated to the 5 target categories.
  **First action: verify** with `value_counts()` and confirm the label strings match the
  5 categories above (adjust a small mapping dict only if they differ). Distribution is
  highly imbalanced (~56% credit reporting in the original).

### Preprocessing (replicate the paper exactly) → `preprocess.py` (runs LOCALLY)
1. Drop rows with empty `product` or `narrative`.
2. Remove narratives **longer than 500 characters**.
3. Standardize narrative text: lowercase; strip special characters, excessive whitespace,
   and inconsistent formatting.
4. **Balanced subset by undersampling**: select **200 rows per category × 5 = 1000 rows**.
   Use a **fixed random seed** (e.g., 42) and record it for reproducibility.
5. Output `balanced_1000.csv` with at least columns: `narrative`, `true_label`.

This 1000-row file (~1 MB) is the only data that gets sent to the cluster.

---

## 4. Compute environment & division of labor

| Stage | Where | Notes |
|---|---|---|
| Data download, preprocessing, balanced sampling | **Local PC** | conda env `kangal` (has pandas, scikit-learn, matplotlib, seaborn) |
| Model inference (the heavy GPU work) | **TRUBA ARF cluster** | via Ollama on GPU nodes |
| Metrics, confusion-matrix heatmaps, write-up | **Local PC** | prediction CSVs are small |

**Data flow:** local `preprocess.py` → `balanced_1000.csv` → `scp` to cluster scratch →
run inference on GPU → `scp` per-model prediction CSVs back → local `analyze.py`.

### TRUBA ARF cluster — environment facts the agent MUST respect
These are environment-specific and cannot be guessed; honor them in any cluster script.

- **Login / interface node:** `arf-ui1` at `172.16.6.11`. Access = OpenVPN, then
  `ssh muhtan@172.16.6.11`. This node is for editing, transfers, submitting jobs — **NOT**
  for running computation.
- **Scheduler:** SLURM. Submit jobs with `sbatch` (never `bash`/`./`).
- **Jobs MUST be submitted from `/arf/scratch/$USER`.** A submit filter REJECTS jobs run
  from `/arf/home`. All job working dirs and output files go under `/arf/scratch/$USER`.
- **Compute nodes have NO internet.** Therefore Ollama models must be **pulled in advance
  on the login node** (`arf-ui1`), which does have internet. Inference then runs offline.
- **Ollama is already installed** at `~/ollama/bin/ollama` (static binary). Models live in
  `~/ollama_models`. Required env in any script that uses it:
  ```bash
  export PATH=$HOME/ollama/bin:$PATH
  export OLLAMA_MODELS=$HOME/ollama_models
  export OLLAMA_HOST=127.0.0.1:11434
  ```
  `$HOME` = `/arf/home/muhtan`. The model files are read from the shared `/arf/home`
  filesystem, which is visible from the compute nodes (no manual copy to the node needed).
- **GPU queues open to all users** (pick one; both have 16 GB VRAM per GPU):
  - `akya-cuda` — 4× NVIDIA Tesla **V100 16 GB** per node. Request **`-c 10` per GPU**.
  - `barbun-cuda` — 2× NVIDIA Tesla **P100 16 GB** per node. Request **`-c 20` per GPU**.
  - `palamut-cuda` (8× A100 80 GB) and `kolyoz-cuda` (H100/H200) exist but are **restricted**
    to specific projects (separate `cuda-ui` login at `172.16.6.16`). Only use if access is confirmed.
  - GPU jobs **must** request a GPU: `#SBATCH --gres=gpu:1` (or `:N`), else rejected.
  - Always set `#SBATCH --time=...` (jobs with no time limit are killed after 2 minutes;
    max walltime is 3 days).
  - Do **not** add `-C weka` or CPU-queue constraints to GPU jobs.

---

## 5. Model roster (open-weight, available via Ollama)

Models below fit a single 16 GB V100/P100 at 4-bit (q4) quantization. Pin the exact tag
and record the pull date for the paper.

**Standard (non-reasoning) open models:**
- `qwen2.5:7b` (already pulled), `llama3.1:8b`, `gemma2:9b`, `mistral:7b`, `phi-4` (14B)

**Open reasoning models (the key comparison group):**
- `gpt-oss:20b` (fits 16 GB, adjustable reasoning, ~o3-mini level)
- `deepseek-r1:8b` (distilled R1)
- `qwen3:8b` (supports a thinking mode)

**Larger reasoning models** (need >16 GB → use `--gres=gpu:4` on akya = 64 GB, or A100 if
access): `deepseek-r1:32b`, `qwq:32b`. Treat as optional/stretch.

For each model the agent should make it trivial to pull (on the login node) and to run
inference (in a SLURM job).

---

## 6. Scripts to build (deliverables)

### 6.1 `preprocess.py` — LOCAL, uses pandas
- Input: `complaints_processed.csv`. Output: `balanced_1000.csv`.
- Implements Section 3 preprocessing. Reproducible (fixed seed). Prints the label
  distribution before/after and asserts 200 per class.

### 6.2 `classify.py` — runs on TRUBA, **standard library only** (`csv`, `json`, `urllib`, `time`, `argparse`)
**Do NOT use pandas or requests here** so that nothing needs to be pip-installed on the cluster.
- CLI args: `--model <ollama_tag>`, `--input balanced_1000.csv`, `--output preds_<model>.csv`,
  optional `--limit N` (for quick pilots), `--host 127.0.0.1:11434`.
- Assumes an Ollama server is reachable at `--host` (the SLURM script starts it).
- For each complaint row:
  - Build the Section 2 prompt with the narrative inserted.
  - `POST` to `http://<host>/api/generate` with body:
    `{"model": <tag>, "prompt": <prompt>, "stream": false, "options": {"temperature": 0}}`
  - Parse the response. **Robust extraction required:** the returned text may contain
    extra prose, markdown code fences, or (for reasoning models) a long chain-of-thought
    *before* the final answer. Extract the JSON object `{"product": "..."}` (search for the
    last valid JSON object / regex fallback). Lowercase + strip the predicted label.
  - Record per row: `index`, `true_label`, `predicted_label`, `raw_response`,
    `latency_seconds`. Flag/normalize **off-label** predictions (anything not in the 5
    categories — the paper found models inventing labels like `auto_loan`).
- Write results to a CSV. Handle request errors gracefully (retry once, then mark as error).

### 6.3 `run_ollama.slurm` — TRUBA SLURM submission script
- Header for `akya-cuda` (default), parametrizable to `barbun-cuda`:
  ```bash
  #SBATCH -p akya-cuda
  #SBATCH -A muhtan
  #SBATCH -J cls-<model>
  #SBATCH -o cls-%j.out          # relative -> written under /arf/scratch submit dir
  #SBATCH --gres=gpu:1
  #SBATCH -N 1
  #SBATCH -n 1
  #SBATCH -c 10                  # 10 for akya-cuda; 20 for barbun-cuda
  #SBATCH --time=02:00:00
  ```
- Body: export the Ollama env (Section 4), `nvidia-smi` (sanity), start `ollama serve`
  in background, `sleep 10`, run `python3 classify.py --model ... --input balanced_1000.csv
  --output preds_<model>.csv`, then `kill` the server.
- Must be submitted from `/arf/scratch/$USER`. Support looping over a list of models (one
  job per model, or a loop inside one job — one job per model is cleaner for the queue).

### 6.4 `analyze.py` — LOCAL, uses pandas + scikit-learn + seaborn/matplotlib
- Reads all `preds_*.csv`. Computes per model: **exact-match accuracy**, **weighted
  precision**, **weighted recall**, **weighted F1**.
- Builds a **summary table** (model × metrics) like the paper's Table 1, plus mean and
  total prediction time per model (latency). Cost is $0 locally — note that as a finding
  and replace the paper's "cost" axis with latency.
- Generates **confusion-matrix heatmaps** per model (true = rows, predicted = columns),
  seaborn, like the paper's Figure 1. Also an accuracy-vs-latency comparison (paper Fig 2/3 style).
- Counts off-label predictions per model (qualitative behavior, as the paper did).

---

## 7. Reproducibility & rigor

- `temperature = 0` for all inference; fixed random seed for the balanced sample.
- Pin exact Ollama model tags + pull date + Ollama version + quantization level + GPU model
  (V100 16 GB) in the methods section.
- Same prompt for every model (model-agnostic); JSON output enforced.
- Data is public (CFPB). Cite the dataset and the original paper. No IRB needed.
- Save raw model responses (not just parsed labels) so parsing decisions are auditable.

---

## 8. Suggested order of work

1. `preprocess.py` locally → `balanced_1000.csv` (verify 200/class). 
2. `scp balanced_1000.csv muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/`
3. Write `classify.py` (stdlib) + `run_ollama.slurm`. **Pilot first:** `--limit 50` with the
   already-pulled `qwen2.5:3b` to confirm the pipeline end-to-end.
4. On the login node, pull the full model roster into `~/ollama_models`.
5. Submit one job per model from `/arf/scratch/muhtan/ccc/`.
6. `scp preds_*.csv` back to the local PC; run `analyze.py` for tables + heatmaps.
7. Iterate (more models, quantization sub-study), then write the article.

---

## 9. Gotchas (learned during setup)

- The cluster submit filter forces `/arf/scratch`; running from home fails with a QOS error.
- Compute nodes are offline → pull models on `arf-ui1` first; inference reads them offline.
- Reasoning models emit long chain-of-thought before the JSON — the parser must tolerate this.
- Some models invent categories outside the 5 — count these as errors, don't silently drop.
- Use `sbatch` (not `bash`) or the job runs on the forbidden login node.
- Quote signed download URLs; this step is already done (data is local).
