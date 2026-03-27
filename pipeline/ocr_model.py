class OCRModel:
    def __init__(self, model_name="funsd_reference_ocr"):
        self.model_name = model_name

    def extract(self, sample):
        words = sample["words"]
        return {
            "model_name": self.model_name,
            "words": words,
            "text": " ".join(words),
            "bboxes": sample["bboxes"],
        }
