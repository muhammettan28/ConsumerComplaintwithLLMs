"""
utils/run_inference.py — TRUBA ARF kümesinde çalışır (stdlib only — pandas/requests yok)
Ollama sunucusuna POST ederek zero-shot sınıflandırma yapar.

Örnek kullanım:
  python3 utils/run_inference.py --model qwen3:8b --input dataset/1_subset_balanced_500.csv --output preds/preds_qwen3_8b_1_balanced_500.csv
  python3 utils/run_inference.py --model deepseek-r1:8b --input dataset/1_subset_balanced_500.csv --output preds/out.csv --limit 50
"""

import argparse
import csv
import json
import re
import socket
import time
import urllib.request
import urllib.error

CATEGORIES = {
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
}

PROMPT_TEMPLATE = (
    "You are an AI assistant specializing in consumer complaint classification.\n"
    "Your task is to analyze a consumer complaint and classify it into the most\n"
    "appropriate category from the predefined list:\n"
    '["retail_banking", "credit_reporting", "credit_card", "mortgages_and_loans", "debt_collection"]\n'
    "Provide your final classification in the following JSON format without explanations:\n"
    '{{"product": "chosen_category_name"}}.\n'
    "Complaint: {narrative}"
)


def build_prompt(narrative: str) -> str:
    return PROMPT_TEMPLATE.format(narrative=narrative)


def extract_label(text: str) -> tuple[str, str]:
    """
    Ham yanıttan {"product": "..."} JSON nesnesini çıkarır.
    Reasoning modelleri uzun CoT zinciri sonrasında JSON döndürebilir;
    bu yüzden son geçerli eşleşmeyi alırız.
    Döndürür: (tahmin_etiket, durum)  durum ∈ {"ok", "off_label", "parse_error"}
    """
    # Tüm {"product": "..."} eşleşmelerini bul, sonu al
    pattern = r'\{[^{}]*"product"\s*:\s*"([^"]+)"[^{}]*\}'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    if matches:
        label = matches[-1].lower().strip()
        status = "ok" if label in CATEGORIES else "off_label"
        return label, status

    # Geri dönüş: tırnak içinde geçen kategori adını ara
    for cat in CATEGORIES:
        if cat in text.lower():
            return cat, "ok"

    return "parse_error", "parse_error"


def call_ollama(host: str, model: str, prompt: str, retries: int = 1) -> tuple[str, float]:
    """
    Ollama /api/generate'e POST atar.
    Döndürür: (raw_response_text, latency_seconds)
    Hata durumunda ("ERROR:...", latency) döndürür.
    """
    url = f"http://{host}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")

    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                latency = time.time() - t0
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", ""), latency
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, socket.timeout) as exc:
            latency = time.time() - t0
            if attempt < retries:
                time.sleep(3)
            else:
                return f"ERROR:{exc}", latency


def read_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_done_indices(path: str) -> set:
    """Yarım kalmış çıktı dosyasındaki tamamlanmış index'leri okur."""
    done = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["index"])
    except FileNotFoundError:
        pass
    return done


def main():
    parser = argparse.ArgumentParser(description="Zero-shot sınıflandırma (Ollama)")
    parser.add_argument("--model", required=True, help="Ollama model etiketi (ör. qwen2.5:7b)")
    parser.add_argument("--input", default="balanced_1000.csv", help="Girdi CSV")
    parser.add_argument("--output", required=True, help="Çıktı CSV (preds_<model>.csv)")
    parser.add_argument("--limit", type=int, default=None, help="İlk N satır (pilot için)")
    parser.add_argument("--host", default="127.0.0.1:11434", help="Ollama sunucu adresi")
    args = parser.parse_args()

    rows = read_csv(args.input)
    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    print(f"Model: {args.model}  |  Satır: {total}  |  Sunucu: {args.host}")

    # Kaldığı yerden devam: daha önce işlenmiş satırları atla
    done_indices = load_done_indices(args.output)
    if done_indices:
        print(f"  Kaldığı yerden devam: {len(done_indices)} satır zaten tamamlanmış, atlanıyor.")

    fieldnames = ["index", "true_label", "predicted_label", "status",
                  "latency_seconds", "raw_response"]
    import os
    write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
    out_file = open(args.output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    processed = len(done_indices)
    errors = 0

    try:
        for i, row in enumerate(rows):
            idx = str(row.get("", row.get("Unnamed: 0", i)))
            if idx in done_indices:
                continue

            narrative = row["narrative"]
            true_label = row["true_label"]

            prompt = build_prompt(narrative)
            raw, latency = call_ollama(args.host, args.model, prompt)

            if raw.startswith("ERROR:"):
                predicted, status = "error", "error"
                errors += 1
            else:
                predicted, status = extract_label(raw)

            writer.writerow({
                "index": idx,
                "true_label": true_label,
                "predicted_label": predicted,
                "status": status,
                "latency_seconds": f"{latency:.3f}",
                "raw_response": raw.replace("\n", " ")[:2000],
            })
            out_file.flush()
            processed += 1

            if processed % 50 == 0 or processed == total:
                print(
                    f"  [{processed:>4}/{total}]  hatalar={errors}  son_gecikme={latency:.1f}s"
                )
    finally:
        out_file.close()

    print(f"\nTamamlandı → {args.output}")
    print(f"  İşlenen : {processed}/{total}")
    print(f"  Hata    : {errors}")


if __name__ == "__main__":
    main()
