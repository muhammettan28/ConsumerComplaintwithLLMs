#!/bin/bash
#SBATCH --job-name=ccc_inference
#SBATCH --partition=akya-cuda
#SBATCH --account=muhtan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=1-96%8
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# 8 model × 12 dataset = 96 iş
# %8 → aynı anda en fazla 8 iş (kuyruğu yormamak için)

MODELS=(
    "deepseek-r1:8b"
    "deepseek-r1:14b"
    "gemma2:9b"
    "gpt-oss:20b"
    "llama3.1:8b"
    "mistral:7b"
    "phi4:14b"
    "qwen3:8b"
)

DATASETS=(
    "1_subset_balanced_500"
    "2_subset_balanced_500"
    "3_subset_balanced_500"
    "1_subset_imbalanced_500"
    "2_subset_imbalanced_500"
    "3_subset_imbalanced_500"
    "1_subset_balanced_250"
    "2_subset_balanced_250"
    "3_subset_balanced_250"
    "1_subset_imbalanced_250"
    "2_subset_imbalanced_250"
    "3_subset_imbalanced_250"
)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
MODEL_IDX=$((IDX / 12))
FILE_IDX=$((IDX % 12))
MODEL="${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$FILE_IDX]}"

# Dosya adı için : → _ (örn. qwen3:8b → qwen3_8b)
MODEL_SAFE="${MODEL//:/_}"
MODEL_SAFE="${MODEL_SAFE//\//_}"

WORKDIR="/arf/scratch/muhtan/ccc"
INPUT="$WORKDIR/dataset/${DATASET}.csv"
OUTPUT="$WORKDIR/preds/preds_${MODEL_SAFE}_${DATASET}.csv"

mkdir -p "$WORKDIR/preds"
mkdir -p "$WORKDIR/logs"

echo "Job $SLURM_ARRAY_TASK_ID  |  Model: $MODEL  |  Dataset: $DATASET"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Node:   $(hostname)  |  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Çıktı zaten varsa atla
if [ -f "$OUTPUT" ]; then
    echo "ATLANDI: $OUTPUT zaten mevcut."
    exit 0
fi

# Ollama ortamını kur
export PATH="$HOME/ollama/bin:$PATH"
export OLLAMA_MODELS="$HOME/ollama_models"
export OLLAMA_HOST="127.0.0.1:11434"

# Ollama sunucusunu başlat
ollama serve &
OLLAMA_PID=$!
sleep 8   # sunucunun ayağa kalkması için bekle

echo "Ollama PID: $OLLAMA_PID"

# Inference
python3 "$WORKDIR/run_inference.py" \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --host "127.0.0.1:11434"

EXIT_CODE=$?

# Ollama'yı kapat
kill $OLLAMA_PID 2>/dev/null

echo "Çıkış kodu: $EXIT_CODE"
exit $EXIT_CODE
