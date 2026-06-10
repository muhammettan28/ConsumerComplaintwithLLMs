#!/bin/bash
# run_ollama.slurm — TRUBA ARF kümesi için SLURM iş betiği
#
# KULLANIM (login node arf-ui1, /arf/scratch/muhtan/ccc/ dizininden):
#
# Deney A — dengeli veri (varsayılan):
#   sbatch --export=MODEL=mistral:7b,OUTFILE=preds_mistral_7b.csv run_ollama.slurm
#
# Deney B — dengesiz veri:
#   sbatch --export=MODEL=mistral:7b,OUTFILE=preds_imb_mistral_7b.csv,INPUT=imbalanced_1000.csv run_ollama.slurm
#
# gpt-oss:20b için (uzun süre):
#   sbatch --time=04:00:00 --export=MODEL=gpt-oss:20b,OUTFILE=preds_gpt-oss_20b.csv run_ollama.slurm

#SBATCH -p akya-cuda
#SBATCH -A muhtan
#SBATCH -J cls-ollama
#SBATCH -o cls-%j.out
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --time=02:00:00

# --- Ollama ortam değişkenleri ---
export PATH=$HOME/ollama/bin:$PATH
export OLLAMA_MODELS=$HOME/ollama_models
export OLLAMA_HOST=127.0.0.1:11434

echo "========================================"
INPUT_FILE="${INPUT:-balanced_1000.csv}"
echo "İş ID   : $SLURM_JOB_ID"
echo "Düğüm   : $(hostname)"
echo "Model   : ${MODEL}"
echo "Girdi   : ${INPUT_FILE}"
echo "Çıktı   : ${OUTFILE}"
echo "Başlangıç: $(date)"
echo "========================================"

# GPU doğrulama
nvidia-smi

# Ollama sunucusunu arka planda başlat
ollama serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Sunucunun hazır olmasını bekle
sleep 15

# Sunucunun gerçekten ayakta olup olmadığını kontrol et
if ! curl -s http://$OLLAMA_HOST/api/tags > /dev/null 2>&1; then
    echo "HATA: Ollama sunucusu başlamadı, 30 saniye daha bekleniyor..."
    sleep 30
fi

echo "Sınıflandırma başlıyor: $(date)"

python3 classify.py \
    --model "${MODEL}" \
    --input  "${INPUT_FILE}" \
    --output "${OUTFILE}" \
    --host   $OLLAMA_HOST

EXIT_CODE=$?

echo "Sınıflandırma bitti: $(date)  (çıkış kodu: $EXIT_CODE)"

# Ollama sunucusunu kapat
kill $OLLAMA_PID 2>/dev/null
wait $OLLAMA_PID 2>/dev/null

echo "========================================"
echo "Tamamlandı: $(date)"
echo "Çıktı dosyası: ${OUTFILE}"
echo "========================================"

exit $EXIT_CODE
