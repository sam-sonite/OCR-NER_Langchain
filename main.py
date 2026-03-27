import csv
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from config import DATA_CONFIG, EVAL_CONFIG, MODEL_CONFIG
from data.funsd_loader import FUNSDLoader
from evaluation.evaluator import Evaluator
from pipeline.ner_model import NERModel
from pipeline.ocr_model import OCRModel


def average(values):
    return sum(values) / len(values) if values else 0.0


def corrupt_word(word, rng, probability):
    if not word or rng.random() >= probability:
        return word
    mode = rng.choice(["drop", "swap", "mask"])
    if mode == "drop" and len(word) > 1:
        index = rng.randrange(len(word))
        return word[:index] + word[index + 1 :]
    if mode == "swap" and len(word) > 2:
        index = rng.randrange(len(word) - 1)
        chars = list(word)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return "".join(chars)
    return "*" * min(len(word), 3)


def noisy_words(words, rng, probability):
    return [corrupt_word(word, rng, probability) for word in words]


def aggregate_boundary(boundary_reports):
    total = Counter()
    confusion = Counter()
    for report in boundary_reports:
        total["boundary_errors"] += report["boundary_errors"]
        total["missed_entities"] += report["missed_entities"]
        total["spurious_entities"] += report["spurious_entities"]
        confusion.update(report["label_confusion"])
    total["label_confusion"] = dict(confusion)
    return dict(total)


def summarize_ocr(results):
    return {
        "avg_cer": average([item["metrics"]["cer"] for item in results]),
        "avg_wer": average([item["metrics"]["wer"] for item in results]),
        "avg_word_accuracy": average([item["metrics"]["word_accuracy"] for item in results]),
        "exact_match_rate": average([1.0 if item["metrics"]["exact_match"] else 0.0 for item in results]),
        "avg_layout_grounding_iou": average(
            [item["metrics"]["layout_grounding_iou"] for item in results]
        ),
        "avg_latency_ms": average([item["latency_ms"] for item in results]),
        "throughput_docs_per_sec": (
            len(results) / (sum(item["latency_ms"] for item in results) / 1000.0)
            if results else 0.0
        ),
    }


def summarize_ner(results, success_threshold, evaluator):
    document_metrics = [
        {
            "labeled_f1": item["metrics"]["labeled"]["f1"],
            "ocr_has_text": bool(item["ocr_text"].strip()),
            "has_correct_entity": item["metrics"]["labeled"]["true_positives"] > 0,
        }
        for item in results
    ]
    return {
        "span_accuracy": average([item["metrics"]["span_accuracy"] for item in results]),
        "label_consistency": average([item["metrics"]["label_consistency"] for item in results]),
        "labeled_precision": average([item["metrics"]["labeled"]["precision"] for item in results]),
        "labeled_recall": average([item["metrics"]["labeled"]["recall"] for item in results]),
        "labeled_f1": average([item["metrics"]["labeled"]["f1"] for item in results]),
        "unlabeled_precision": average([item["metrics"]["unlabeled"]["precision"] for item in results]),
        "unlabeled_recall": average([item["metrics"]["unlabeled"]["recall"] for item in results]),
        "unlabeled_f1": average([item["metrics"]["unlabeled"]["f1"] for item in results]),
        "avg_latency_ms": average([item["latency_ms"] for item in results]),
        "throughput_docs_per_sec": (
            len(results) / (sum(item["latency_ms"] for item in results) / 1000.0)
            if results else 0.0
        ),
        "boundary_analysis": aggregate_boundary(
            [item["metrics"]["boundary_analysis"] for item in results]
        ),
        "downstream": evaluator.summarize_document_metrics(document_metrics, success_threshold),
    }


def run_ocr_benchmark(samples, ocr_model, evaluator):
    results = []
    for sample in samples:
        start = time.perf_counter()
        prediction = ocr_model.extract(sample)
        latency_ms = (time.perf_counter() - start) * 1000
        metrics = evaluator.evaluate_ocr(prediction, sample)
        results.append({
            "sample_id": sample["id"],
            "split": sample["split"],
            "latency_ms": latency_ms,
            "metrics": metrics,
            "text": prediction["text"],
        })
    return results


def run_ocr_noise_benchmark(samples, evaluator, probability, seed):
    rng = random.Random(seed)
    results = []
    for sample in samples:
        start = time.perf_counter()
        words = noisy_words(sample["words"], rng, probability)
        prediction = {
            "words": words,
            "text": " ".join(words),
            "bboxes": sample["bboxes"],
        }
        latency_ms = (time.perf_counter() - start) * 1000
        metrics = evaluator.evaluate_ocr(prediction, sample)
        results.append({
            "sample_id": sample["id"],
            "split": sample["split"],
            "latency_ms": latency_ms,
            "metrics": metrics,
            "text": prediction["text"],
        })
    return results


def run_ner_benchmark(samples, ner_model, evaluator):
    results = []
    for sample in samples:
        start = time.perf_counter()
        prediction = ner_model.predict(sample)
        latency_ms = (time.perf_counter() - start) * 1000
        metrics = evaluator.evaluate_ner(prediction, sample)
        results.append({
            "sample_id": sample["id"],
            "split": sample["split"],
            "latency_ms": latency_ms,
            "metrics": metrics,
            "predicted_entities": prediction["entities"],
            "ocr_text": sample["text"],
        })
    return results


def run_ner_noise_benchmark(samples, ner_model, evaluator, probability, seed):
    rng = random.Random(seed)
    results = []
    for sample in samples:
        noisy = noisy_words(sample["words"], rng, probability)
        start = time.perf_counter()
        prediction = ner_model.predict(sample, words=noisy)
        latency_ms = (time.perf_counter() - start) * 1000
        metrics = evaluator.evaluate_ner(prediction, sample)
        results.append({
            "sample_id": sample["id"],
            "split": sample["split"],
            "latency_ms": latency_ms,
            "metrics": metrics,
            "predicted_entities": prediction["entities"],
            "ocr_text": " ".join(noisy),
        })
    return results


def build_report(summary):
    part1 = summary["part_1"]
    part2 = summary["part_2"]
    return f"""# Evaluation Report

## Part 1: Evaluating Information Extraction LLMs (NER)

### HEADING 1: Understanding of Information Extraction LLMs
- Model: {part1["understanding"]["model_name"]}
- Train samples: {part1["understanding"]["train_samples"]}
- Test samples: {part1["understanding"]["test_samples"]}
- Labels: {", ".join(part1["understanding"]["labels"])}

### HEADING 2: Training Evaluation for Span Accuracy and Label Consistency
- Span accuracy: {part1["training_eval"]["span_accuracy"]:.4f}
- Label consistency: {part1["training_eval"]["label_consistency"]:.4f}
- Labeled precision / recall / F1: {part1["training_eval"]["labeled_precision"]:.4f} / {part1["training_eval"]["labeled_recall"]:.4f} / {part1["training_eval"]["labeled_f1"]:.4f}
- Unlabeled precision / recall / F1: {part1["training_eval"]["unlabeled_precision"]:.4f} / {part1["training_eval"]["unlabeled_recall"]:.4f} / {part1["training_eval"]["unlabeled_f1"]:.4f}

### HEADING 3: Boundary Errors and Entity Confusion Analysis
- Boundary errors: {part1["boundary_analysis"]["boundary_errors"]}
- Missed entities: {part1["boundary_analysis"]["missed_entities"]}
- Spurious entities: {part1["boundary_analysis"]["spurious_entities"]}
- Label confusion: {json.dumps(part1["boundary_analysis"]["label_confusion"])}

### HEADING 4: Robustness to Noisy and Adversarial Text at Inference
- Noisy labeled F1: {part1["robustness"]["noisy_labeled_f1"]:.4f}
- Noisy unlabeled F1: {part1["robustness"]["noisy_unlabeled_f1"]:.4f}
- Adversarial labeled F1: {part1["robustness"]["adversarial_labeled_f1"]:.4f}
- Adversarial unlabeled F1: {part1["robustness"]["adversarial_unlabeled_f1"]:.4f}

### HEADING 5: Downstream Extraction Accuracy and Error Propagation
- Clean document success rate: {part1["downstream"]["clean_document_success_rate"]:.4f}
- Clean automation success rate: {part1["downstream"]["clean_automation_success_rate"]:.4f}
- Noisy document success rate: {part1["downstream"]["noisy_document_success_rate"]:.4f}
- Noisy automation success rate: {part1["downstream"]["noisy_automation_success_rate"]:.4f}

## Part 2: Evaluating Document Understanding LLMs (OCR)

### HEADING 6: OCR Based Document Understanding LLMs
- OCR benchmark source: {part2["understanding"]["model_name"]}
- Test samples: {part2["understanding"]["test_samples"]}
- Benchmark note: {part2["understanding"]["benchmark_note"]}

### HEADING 7: Training Evaluation for Text Layout Grounding and Transcription Accuracy
- CER: {part2["training_eval"]["avg_cer"]:.4f}
- WER: {part2["training_eval"]["avg_wer"]:.4f}
- Word accuracy: {part2["training_eval"]["avg_word_accuracy"]:.4f}
- Exact match rate: {part2["training_eval"]["exact_match_rate"]:.4f}
- Layout grounding IoU: {part2["training_eval"]["avg_layout_grounding_iou"]:.4f}

### HEADING 8: Sensitivity to Image Quality and Resolution at Inference
- Mild synthetic noise CER / WER: {part2["sensitivity"]["mild_noise_cer"]:.4f} / {part2["sensitivity"]["mild_noise_wer"]:.4f}
- Adversarial synthetic noise CER / WER: {part2["sensitivity"]["adversarial_noise_cer"]:.4f} / {part2["sensitivity"]["adversarial_noise_wer"]:.4f}

### HEADING 9: Latency and Throughput in Document Processing
- Avg OCR latency (ms): {part2["latency"]["avg_latency_ms"]:.4f}
- OCR throughput (docs/sec): {part2["latency"]["throughput_docs_per_sec"]:.4f}
- Avg NER latency (ms): {part1["latency"]["avg_latency_ms"]:.4f}
- NER throughput (docs/sec): {part1["latency"]["throughput_docs_per_sec"]:.4f}

### HEADING 10: Document Processing Accuracy and Automation Success
- OCR exact match rate: {part2["automation"]["ocr_exact_match_rate"]:.4f}
- OCR word accuracy: {part2["automation"]["ocr_word_accuracy"]:.4f}
- End-to-end clean automation success: {part2["automation"]["end_to_end_clean_success"]:.4f}
- End-to-end noisy automation success: {part2["automation"]["end_to_end_noisy_success"]:.4f}
"""


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    evaluator = Evaluator()
    ocr_model = OCRModel(MODEL_CONFIG["ocr_model"])
    ner_model = NERModel(MODEL_CONFIG["ner_model"])

    train_samples = FUNSDLoader("train", DATA_CONFIG["train_samples"]).get_samples()
    test_samples = FUNSDLoader("test", DATA_CONFIG["test_samples"]).get_samples()

    clean_ocr_results = run_ocr_benchmark(test_samples, ocr_model, evaluator)
    noisy_ocr_results = run_ocr_noise_benchmark(test_samples, evaluator, EVAL_CONFIG["ocr_noise_probability"], EVAL_CONFIG["random_seed"])
    adversarial_ocr_results = run_ocr_noise_benchmark(test_samples, evaluator, EVAL_CONFIG["adversarial_noise_probability"], EVAL_CONFIG["random_seed"] + 1)

    train_ner_results = run_ner_benchmark(train_samples, ner_model, evaluator)
    clean_test_ner_results = run_ner_benchmark(test_samples, ner_model, evaluator)
    noisy_test_ner_results = run_ner_noise_benchmark(test_samples, ner_model, evaluator, EVAL_CONFIG["ocr_noise_probability"], EVAL_CONFIG["random_seed"])
    adversarial_test_ner_results = run_ner_noise_benchmark(test_samples, ner_model, evaluator, EVAL_CONFIG["adversarial_noise_probability"], EVAL_CONFIG["random_seed"] + 1)

    train_ner_summary = summarize_ner(train_ner_results, EVAL_CONFIG["document_success_f1_threshold"], evaluator)
    clean_test_ner_summary = summarize_ner(clean_test_ner_results, EVAL_CONFIG["document_success_f1_threshold"], evaluator)
    noisy_test_ner_summary = summarize_ner(noisy_test_ner_results, EVAL_CONFIG["document_success_f1_threshold"], evaluator)
    adversarial_test_ner_summary = summarize_ner(adversarial_test_ner_results, EVAL_CONFIG["document_success_f1_threshold"], evaluator)

    clean_ocr_summary = summarize_ocr(clean_ocr_results)
    noisy_ocr_summary = summarize_ocr(noisy_ocr_results)
    adversarial_ocr_summary = summarize_ocr(adversarial_ocr_results)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "part_1": {
            "understanding": {
                "model_name": MODEL_CONFIG["ner_model"],
                "train_samples": len(train_samples),
                "test_samples": len(test_samples),
                "labels": ["header", "question", "answer"],
            },
            "training_eval": {
                "span_accuracy": train_ner_summary["span_accuracy"],
                "label_consistency": train_ner_summary["label_consistency"],
                "labeled_precision": train_ner_summary["labeled_precision"],
                "labeled_recall": train_ner_summary["labeled_recall"],
                "labeled_f1": train_ner_summary["labeled_f1"],
                "unlabeled_precision": train_ner_summary["unlabeled_precision"],
                "unlabeled_recall": train_ner_summary["unlabeled_recall"],
                "unlabeled_f1": train_ner_summary["unlabeled_f1"],
            },
            "boundary_analysis": train_ner_summary["boundary_analysis"],
            "robustness": {
                "noisy_labeled_f1": noisy_test_ner_summary["labeled_f1"],
                "noisy_unlabeled_f1": noisy_test_ner_summary["unlabeled_f1"],
                "adversarial_labeled_f1": adversarial_test_ner_summary["labeled_f1"],
                "adversarial_unlabeled_f1": adversarial_test_ner_summary["unlabeled_f1"],
            },
            "downstream": {
                "clean_document_success_rate": clean_test_ner_summary["downstream"]["document_success_rate"],
                "clean_automation_success_rate": clean_test_ner_summary["downstream"]["automation_success_rate"],
                "noisy_document_success_rate": noisy_test_ner_summary["downstream"]["document_success_rate"],
                "noisy_automation_success_rate": noisy_test_ner_summary["downstream"]["automation_success_rate"],
            },
            "latency": {
                "avg_latency_ms": clean_test_ner_summary["avg_latency_ms"],
                "throughput_docs_per_sec": clean_test_ner_summary["throughput_docs_per_sec"],
            },
        },
        "part_2": {
            "understanding": {
                "model_name": MODEL_CONFIG["ocr_model"],
                "test_samples": len(test_samples),
                "benchmark_note": "FUNSD extracted OCR words are used as the clean OCR benchmark reference.",
            },
            "training_eval": clean_ocr_summary,
            "sensitivity": {
                "mild_noise_cer": noisy_ocr_summary["avg_cer"],
                "mild_noise_wer": noisy_ocr_summary["avg_wer"],
                "adversarial_noise_cer": adversarial_ocr_summary["avg_cer"],
                "adversarial_noise_wer": adversarial_ocr_summary["avg_wer"],
            },
            "latency": {
                "avg_latency_ms": clean_ocr_summary["avg_latency_ms"],
                "throughput_docs_per_sec": clean_ocr_summary["throughput_docs_per_sec"],
            },
            "automation": {
                "ocr_exact_match_rate": clean_ocr_summary["exact_match_rate"],
                "ocr_word_accuracy": clean_ocr_summary["avg_word_accuracy"],
                "end_to_end_clean_success": clean_test_ner_summary["downstream"]["automation_success_rate"],
                "end_to_end_noisy_success": noisy_test_ner_summary["downstream"]["automation_success_rate"],
            },
        },
    }

    artifact_dir = Path(DATA_CONFIG["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_path = artifact_dir / "evaluation_summary.json"
    report_path = artifact_dir / "evaluation_report.md"
    ner_csv_path = artifact_dir / "ner_metrics.csv"
    ocr_csv_path = artifact_dir / "ocr_metrics.csv"
    details_path = artifact_dir / "evaluation_details.json"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(build_report(summary), encoding="utf-8")

    details = {
        "train_ner_results": train_ner_results,
        "clean_test_ner_results": clean_test_ner_results,
        "noisy_test_ner_results": noisy_test_ner_results,
        "adversarial_test_ner_results": adversarial_test_ner_results,
        "clean_ocr_results": clean_ocr_results,
        "noisy_ocr_results": noisy_ocr_results,
        "adversarial_ocr_results": adversarial_ocr_results,
    }
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    ner_rows = []
    for bucket_name, bucket in [("train_clean", train_ner_results), ("test_clean", clean_test_ner_results), ("test_noisy", noisy_test_ner_results), ("test_adversarial", adversarial_test_ner_results)]:
        for item in bucket:
            ner_rows.append({
                "bucket": bucket_name,
                "sample_id": item["sample_id"],
                "latency_ms": item["latency_ms"],
                "span_accuracy": item["metrics"]["span_accuracy"],
                "label_consistency": item["metrics"]["label_consistency"],
                "labeled_precision": item["metrics"]["labeled"]["precision"],
                "labeled_recall": item["metrics"]["labeled"]["recall"],
                "labeled_f1": item["metrics"]["labeled"]["f1"],
                "unlabeled_precision": item["metrics"]["unlabeled"]["precision"],
                "unlabeled_recall": item["metrics"]["unlabeled"]["recall"],
                "unlabeled_f1": item["metrics"]["unlabeled"]["f1"],
            })

    write_csv(ner_csv_path, ner_rows, ["bucket", "sample_id", "latency_ms", "span_accuracy", "label_consistency", "labeled_precision", "labeled_recall", "labeled_f1", "unlabeled_precision", "unlabeled_recall", "unlabeled_f1"])

    ocr_rows = []
    for bucket_name, bucket in [("clean", clean_ocr_results), ("noisy", noisy_ocr_results), ("adversarial", adversarial_ocr_results)]:
        for item in bucket:
            ocr_rows.append({
                "bucket": bucket_name,
                "sample_id": item["sample_id"],
                "latency_ms": item["latency_ms"],
                "cer": item["metrics"]["cer"],
                "wer": item["metrics"]["wer"],
                "word_accuracy": item["metrics"]["word_accuracy"],
                "exact_match": item["metrics"]["exact_match"],
                "layout_grounding_iou": item["metrics"]["layout_grounding_iou"],
            })

    write_csv(ocr_csv_path, ocr_rows, ["bucket", "sample_id", "latency_ms", "cer", "wer", "word_accuracy", "exact_match", "layout_grounding_iou"])

    print("Artifacts saved:")
    print(summary_path.resolve())
    print(report_path.resolve())
    print(ner_csv_path.resolve())
    print(ocr_csv_path.resolve())
    print(details_path.resolve())
    print("\nNER train labeled F1:", round(summary["part_1"]["training_eval"]["labeled_f1"], 4))
    print("NER test noisy labeled F1:", round(summary["part_1"]["robustness"]["noisy_labeled_f1"], 4))
    print("OCR exact match rate:", round(summary["part_2"]["training_eval"]["exact_match_rate"], 4))


if __name__ == "__main__":
    main()
