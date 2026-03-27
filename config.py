MODEL_CONFIG = {
    "ocr_model": "funsd_reference_ocr",
    "ner_model": "nielsr/layoutlmv3-finetuned-funsd"
}

DATA_CONFIG = {
    "train_samples": 10,
    "test_samples": 10,
    "artifact_dir": "artifacts"
}

EVAL_CONFIG = {
    "random_seed": 7,
    "ocr_noise_probability": 0.12,
    "adversarial_noise_probability": 0.22,
    "document_success_f1_threshold": 0.5
}
