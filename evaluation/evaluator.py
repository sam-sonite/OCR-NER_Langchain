from evaluation.metrics import (
    automation_success_rate,
    average_layout_iou,
    boundary_and_confusion_analysis,
    char_error_rate,
    document_success_rate,
    entity_prf,
    exact_text_match,
    label_consistency,
    span_accuracy,
    token_label_consistency,
    token_prf,
    token_span_accuracy,
    word_accuracy,
    word_error_rate,
)


class Evaluator:
    def evaluate_ocr(self, prediction, ground_truth):
        return {
            "cer": char_error_rate(prediction["text"], ground_truth["text"]),
            "wer": word_error_rate(prediction["words"], ground_truth["words"]),
            "word_accuracy": word_accuracy(prediction["words"], ground_truth["words"]),
            "exact_match": exact_text_match(prediction["text"], ground_truth["text"]),
            "layout_grounding_iou": average_layout_iou(prediction["bboxes"], ground_truth["bboxes"]),
        }

    def evaluate_ner(self, prediction, ground_truth):
        labeled = token_prf(prediction["word_labels"], ground_truth["word_labels"], include_label=True)
        unlabeled = token_prf(prediction["word_labels"], ground_truth["word_labels"], include_label=False)
        return {
            "span_accuracy": token_span_accuracy(prediction["word_labels"], ground_truth["word_labels"]),
            "label_consistency": token_label_consistency(prediction["word_labels"], ground_truth["word_labels"]),
            "labeled": labeled,
            "unlabeled": unlabeled,
            "boundary_analysis": boundary_and_confusion_analysis(
                prediction["entities"], ground_truth["entities"]
            ),
        }

    def summarize_document_metrics(self, doc_metrics, success_threshold):
        return {
            "document_success_rate": document_success_rate(doc_metrics, success_threshold),
            "automation_success_rate": automation_success_rate(doc_metrics),
        }
