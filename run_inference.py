"""
run_inference.py — TRUBA ARF kümesinde çalışır (stdlib only — pandas/requests yok)
Ollama sunucusuna POST ederek zero-shot sınıflandırma yapar.

Örnek kullanım:
  python3 run_inference.py --model qwen3:8b --input dataset/1_subset_balanced_500.csv --output preds/preds_qwen3_8b_1_balanced_500.csv
  python3 run_inference.py --model deepseek-r1:8b --input dataset/1_subset_balanced_500.csv --output preds/out.csv --limit 50
"""

import argparse
import csv
import json
import re
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                latency = time.time() - t0
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("response", ""), latency
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
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

    results = []
    errors = 0

    for i, row in enumerate(rows):
        idx = row.get("", row.get("Unnamed: 0", i))
        narrative = row["narrative"]
        true_label = row["true_label"]

        prompt = build_prompt(narrative)
        raw, latency = call_ollama(args.host, args.model, prompt)

        if raw.startswith("ERROR:"):
            predicted, status = "error", "error"
            errors += 1
        else:
            predicted, status = extract_label(raw)

        results.append({
            "index": idx,
            "true_label": true_label,
            "predicted_label": predicted,
            "status": status,
            "latency_seconds": f"{latency:.3f}",
            "raw_response": raw.replace("\n", " ")[:2000],  # CSV'yi okunabilir tut
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            correct = sum(
                1 for r in results
                if r["true_label"] == r["predicted_label"]
            )
            print(
                f"  [{i+1:>4}/{total}]  anlık doğruluk={correct/(i+1):.3f}"
                f"  hatalar={errors}  son_gecikme={latency:.1f}s"
            )

    # Sonuçları yaz
    fieldnames = ["index", "true_label", "predicted_label", "status",
                  "latency_seconds", "raw_response"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    correct = sum(1 for r in results if r["true_label"] == r["predicted_label"])
    off_label = sum(1 for r in results if r["status"] == "off_label")
    print(f"\nTamamlandı → {args.output}")
    print(f"  Doğruluk   : {correct}/{total} = {correct/total:.4f}")
    print(f"  Off-label  : {off_label}")
    print(f"  Hata       : {errors}")


if __name__ == "__main__":
    main()
