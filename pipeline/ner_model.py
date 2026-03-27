import torch
from transformers import AutoModelForTokenClassification, AutoProcessor


class NERModel:
    def __init__(self, model_name="nielsr/layoutlmv3-finetuned-funsd"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, sample, words=None):
        words = words or sample["words"]
        encoding = self.processor(
            sample["image"],
            words,
            boxes=sample["bboxes"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
        )
        batch_word_ids = encoding.word_ids(batch_index=0)
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)

        predicted_ids = outputs.logits.argmax(-1)[0].tolist()
        word_labels = self._collapse_to_words(predicted_ids, batch_word_ids, len(words))
        entities = self._build_entities(words, word_labels)

        return {
            "word_labels": word_labels,
            "entities": entities,
        }

    def _collapse_to_words(self, predicted_ids, word_ids, num_words):
        word_labels = ["O"] * num_words
        seen = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen or word_id >= num_words:
                continue
            seen.add(word_id)
            raw_label = self.id2label[predicted_ids[token_index]]
            if raw_label == "O":
                word_labels[word_id] = "O"
            else:
                word_labels[word_id] = raw_label.split("-", 1)[-1].lower()
        return word_labels

    def _build_entities(self, words, word_labels):
        entities = []
        current = None

        for word_index, (word, tag_name) in enumerate(zip(words, word_labels)):
            if tag_name == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            label_name = tag_name

            if not current or current["label"] != label_name:
                if current:
                    entities.append(current)
                current = {
                    "text": word,
                    "label": label_name,
                    "word_start": word_index,
                    "word_end": word_index + 1,
                }
            else:
                current["text"] = f'{current["text"]} {word}'
                current["word_end"] = word_index + 1

        if current:
            entities.append(current)

        return entities
