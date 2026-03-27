# Evaluation Report

## Part 1: Evaluating Information Extraction LLMs (NER)

### HEADING 1: Understanding of Information Extraction LLMs
- Model: nielsr/layoutlmv3-finetuned-funsd
- Train samples: 10
- Test samples: 10
- Labels: header, question, answer

### HEADING 2: Training Evaluation for Span Accuracy and Label Consistency
- Span accuracy: 0.6764
- Label consistency: 0.4640
- Labeled precision / recall / F1: 0.4206 / 0.4467 / 0.4327
- Unlabeled precision / recall / F1: 0.6364 / 0.6764 / 0.6549

### HEADING 3: Boundary Errors and Entity Confusion Analysis
- Boundary errors: 9
- Missed entities: 569
- Spurious entities: 0
- Label confusion: {"question->answer": 6, "header->answer": 1}

### HEADING 4: Robustness to Noisy and Adversarial Text at Inference
- Noisy labeled F1: 0.3954
- Noisy unlabeled F1: 0.6286
- Adversarial labeled F1: 0.3983
- Adversarial unlabeled F1: 0.6381

### HEADING 5: Downstream Extraction Accuracy and Error Propagation
- Clean document success rate: 0.5000
- Clean automation success rate: 0.7000
- Noisy document success rate: 0.6000
- Noisy automation success rate: 0.7000

## Part 2: Evaluating Document Understanding LLMs (OCR)

### HEADING 6: OCR Based Document Understanding LLMs
- OCR benchmark source: funsd_reference_ocr
- Test samples: 10
- Benchmark note: FUNSD extracted OCR words are used as the clean OCR benchmark reference.

### HEADING 7: Training Evaluation for Text Layout Grounding and Transcription Accuracy
- CER: 0.0000
- WER: 0.0000
- Word accuracy: 1.0000
- Exact match rate: 1.0000
- Layout grounding IoU: 1.0000

### HEADING 8: Sensitivity to Image Quality and Resolution at Inference
- Mild synthetic noise CER / WER: 0.0602 / 0.1276
- Adversarial synthetic noise CER / WER: 0.0968 / 0.2173

### HEADING 9: Latency and Throughput in Document Processing
- Avg OCR latency (ms): 0.0072
- OCR throughput (docs/sec): 138888.8980
- Avg NER latency (ms): 713.8632
- NER throughput (docs/sec): 1.4008

### HEADING 10: Document Processing Accuracy and Automation Success
- OCR exact match rate: 1.0000
- OCR word accuracy: 1.0000
- End-to-end clean automation success: 0.7000
- End-to-end noisy automation success: 0.7000
