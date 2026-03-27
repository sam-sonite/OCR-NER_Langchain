from collections import Counter


def normalize_text(text):
    return " ".join(text.split()).strip()


def levenshtein_distance(seq1, seq2):
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    previous_row = list(range(len(seq2) + 1))
    for i, item1 in enumerate(seq1, start=1):
        current_row = [i]
        for j, item2 in enumerate(seq2, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (item1 != item2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def char_error_rate(pred_text, true_text):
    pred = normalize_text(pred_text)
    true = normalize_text(true_text)
    if not true:
        return 0.0 if not pred else 1.0
    return levenshtein_distance(list(pred), list(true)) / len(true)


def word_error_rate(pred_words, true_words):
    if not true_words:
        return 0.0 if not pred_words else 1.0
    return levenshtein_distance(pred_words, true_words) / len(true_words)


def word_accuracy(pred_words, true_words):
    total = max(len(true_words), 1)
    matches = sum(1 for pred, true in zip(pred_words, true_words) if pred == true)
    return matches / total


def exact_text_match(pred_text, true_text):
    return normalize_text(pred_text) == normalize_text(true_text)


def bbox_iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def average_layout_iou(pred_bboxes, true_bboxes):
    if not true_bboxes:
        return 0.0
    scores = [bbox_iou(pred, true) for pred, true in zip(pred_bboxes, true_bboxes)]
    return sum(scores) / len(scores) if scores else 0.0


def _entity_key(entity, include_label=True):
    if include_label:
        return (entity["word_start"], entity["word_end"], entity["label"])
    return (entity["word_start"], entity["word_end"])


def _overlap(pred_entity, true_entity):
    return not (
        pred_entity["word_end"] <= true_entity["word_start"]
        or pred_entity["word_start"] >= true_entity["word_end"]
    )


def entity_prf(pred_entities, true_entities, include_label=True):
    pred_set = {_entity_key(entity, include_label) for entity in pred_entities}
    true_set = {_entity_key(entity, include_label) for entity in true_entities}
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "predicted": len(pred_set),
        "ground_truth": len(true_set),
    }


def span_accuracy(pred_entities, true_entities):
    return entity_prf(pred_entities, true_entities, include_label=False)["recall"]


def label_consistency(pred_entities, true_entities):
    overlap_pairs = []
    for true_entity in true_entities:
        overlaps = [pred for pred in pred_entities if _overlap(pred, true_entity)]
        if overlaps:
            overlap_pairs.append((overlaps[0], true_entity))
    if not overlap_pairs:
        return 0.0
    consistent = sum(1 for pred, true in overlap_pairs if pred["label"] == true["label"])
    return consistent / len(overlap_pairs)


def boundary_and_confusion_analysis(pred_entities, true_entities):
    boundary_errors = 0
    confusion = Counter()
    matched_true = set()
    matched_pred = set()

    for pred_index, pred_entity in enumerate(pred_entities):
        overlapping = [
            (true_index, true_entity)
            for true_index, true_entity in enumerate(true_entities)
            if _overlap(pred_entity, true_entity)
        ]
        if not overlapping:
            continue

        true_index, true_entity = overlapping[0]
        matched_pred.add(pred_index)
        matched_true.add(true_index)

        same_boundary = (
            pred_entity["word_start"] == true_entity["word_start"]
            and pred_entity["word_end"] == true_entity["word_end"]
        )
        same_label = pred_entity["label"] == true_entity["label"]

        if not same_boundary:
            boundary_errors += 1
        if not same_label:
            confusion[(true_entity["label"], pred_entity["label"])] += 1

    return {
        "boundary_errors": boundary_errors,
        "missed_entities": len(true_entities) - len(matched_true),
        "spurious_entities": len(pred_entities) - len(matched_pred),
        "label_confusion": {
            f"{true_label}->{pred_label}": count
            for (true_label, pred_label), count in confusion.items()
        },
    }


def document_success_rate(doc_metrics, f1_threshold):
    if not doc_metrics:
        return 0.0
    successes = sum(1 for item in doc_metrics if item["labeled_f1"] >= f1_threshold)
    return successes / len(doc_metrics)


def automation_success_rate(doc_metrics):
    if not doc_metrics:
        return 0.0
    successes = sum(1 for item in doc_metrics if item["ocr_has_text"] and item["has_correct_entity"])
    return successes / len(doc_metrics)


def token_prf(pred_word_labels, true_word_labels, include_label=True):
    tp = 0
    predicted = 0
    ground_truth = 0

    for pred_label, true_label in zip(pred_word_labels, true_word_labels):
        pred_is_entity = pred_label != "O"
        true_is_entity = true_label != "O"

        if pred_is_entity:
            predicted += 1
        if true_is_entity:
            ground_truth += 1

        if include_label:
            if pred_label == true_label and true_is_entity:
                tp += 1
        else:
            if pred_is_entity and true_is_entity:
                tp += 1

    precision = tp / predicted if predicted else 0.0
    recall = tp / ground_truth if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "predicted": predicted,
        "ground_truth": ground_truth,
    }


def token_span_accuracy(pred_word_labels, true_word_labels):
    return token_prf(pred_word_labels, true_word_labels, include_label=False)["recall"]


def token_label_consistency(pred_word_labels, true_word_labels):
    overlaps = [
        (pred_label, true_label)
        for pred_label, true_label in zip(pred_word_labels, true_word_labels)
        if pred_label != "O" and true_label != "O"
    ]
    if not overlaps:
        return 0.0
    matches = sum(1 for pred_label, true_label in overlaps if pred_label == true_label)
    return matches / len(overlaps)
