from datasets import load_dataset


class FUNSDLoader:
    def __init__(self, split="train", num_samples=10):
        self.dataset = load_dataset("nielsr/funsd", split=split)
        self.split = split
        self.num_samples = min(num_samples, len(self.dataset))
        self.label_names = self.dataset.features["ner_tags"].feature.names

    def get_samples(self):
        return [self._normalize_item(self.dataset[i]) for i in range(self.num_samples)]

    def _normalize_item(self, item):
        words = item["words"]
        entities = self._build_entities(words, item["ner_tags"])
        return {
            "id": item["id"],
            "split": self.split,
            "image": item["image"].convert("RGB"),
            "words": words,
            "bboxes": item["bboxes"],
            "ner_tags": item["ner_tags"],
            "word_labels": [self._coarse_label(tag_id) for tag_id in item["ner_tags"]],
            "text": " ".join(words),
            "entities": entities,
        }

    def _build_entities(self, words, ner_tags):
        entities = []
        current = None

        for word_index, (word, tag_id) in enumerate(zip(words, ner_tags)):
            tag_name = self.label_names[tag_id]
            if tag_name == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            prefix, label_name = tag_name.split("-", 1)
            label_name = label_name.lower()

            if prefix == "B" or not current or current["label"] != label_name:
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

    def _coarse_label(self, tag_id):
        tag_name = self.label_names[tag_id]
        if tag_name == "O":
            return "O"
        return tag_name.split("-", 1)[-1].lower()
