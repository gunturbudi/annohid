"""
Utility functions for calculating annotation quality metrics
Including precision, recall, F1-score against gold standard
"""

from typing import List, Dict, Tuple
from collections import defaultdict


def normalize_entity(entity: Dict) -> Tuple[int, int, str, str]:
    """
    Normalize an entity to a comparable format
    Returns: (start, end, label, text) tuple
    """
    start = entity.get('start', entity.get('start_char', 0))
    end = entity.get('end', entity.get('end_char', 0))
    label = entity.get('label', entity.get('type', 'UNKNOWN'))
    text = entity.get('text', '')
    return (start, end, label, text)


def entities_match(pred_entity: Dict, gold_entity: Dict, match_mode='exact') -> bool:
    """
    Check if predicted entity matches gold standard entity

    Args:
        pred_entity: Predicted entity dictionary
        gold_entity: Gold standard entity dictionary
        match_mode: 'exact' (exact span and label), 'overlap' (overlapping span and same label),
                   'label_only' (same label, any overlap), 'span_only' (exact span, any label)

    Returns:
        True if entities match according to the mode
    """
    start_pred, end_pred, label_pred, _ = normalize_entity(pred_entity)
    start_gold, end_gold, label_gold, _ = normalize_entity(gold_entity)

    if match_mode == 'exact':
        return (start_pred == start_gold and
                end_pred == end_gold and
                label_pred == label_gold)

    elif match_mode == 'overlap':
        # Check if spans overlap and labels match
        overlap = max(0, min(end_pred, end_gold) - max(start_pred, start_gold))
        return overlap > 0 and label_pred == label_gold

    elif match_mode == 'label_only':
        # Same label with any span overlap
        overlap = max(0, min(end_pred, end_gold) - max(start_pred, start_gold))
        return overlap > 0 and label_pred == label_gold

    elif match_mode == 'span_only':
        # Exact span match, any label
        return start_pred == start_gold and end_pred == end_gold

    return False


def calculate_precision_recall_f1(predicted_entities: List[Dict],
                                   gold_entities: List[Dict],
                                   match_mode='exact') -> Dict:
    """
    Calculate precision, recall, and F1-score for entity recognition

    Precision = TP / (TP + FP) = correct predictions / total predictions
    Recall = TP / (TP + FN) = correct predictions / total gold entities
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        predicted_entities: List of predicted entities
        gold_entities: List of gold standard entities
        match_mode: How to match entities

    Returns:
        Dictionary with precision, recall, F1, and confusion matrix elements
    """
    if not gold_entities and not predicted_entities:
        # Perfect score if both are empty
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0
        }

    # Find true positives
    true_positives = []
    remaining_gold = list(gold_entities)  # Track unmatched gold entities

    for pred_entity in predicted_entities:
        for i, gold_entity in enumerate(remaining_gold):
            if entities_match(pred_entity, gold_entity, match_mode):
                true_positives.append((pred_entity, gold_entity))
                remaining_gold.pop(i)
                break

    # Calculate metrics
    tp = len(true_positives)
    fp = len(predicted_entities) - tp  # Predicted but not in gold
    fn = len(remaining_gold)  # In gold but not predicted
    tn = 0  # Not applicable for entity recognition

    # Calculate precision
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    # Calculate recall
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    # Calculate F1
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn,
        'support': len(gold_entities)  # Total gold entities
    }


def calculate_entity_level_metrics(predicted_entities: List[Dict],
                                   gold_entities: List[Dict],
                                   match_mode='exact') -> Dict:
    """
    Calculate precision, recall, F1 for each entity type

    Args:
        predicted_entities: List of predicted entities
        gold_entities: List of gold standard entities
        match_mode: How to match entities

    Returns:
        Dictionary mapping entity_type -> metrics
    """
    # Group entities by type
    pred_by_type = defaultdict(list)
    gold_by_type = defaultdict(list)

    for entity in predicted_entities:
        label = normalize_entity(entity)[2]
        pred_by_type[label].append(entity)

    for entity in gold_entities:
        label = normalize_entity(entity)[2]
        gold_by_type[label].append(entity)

    # Get all entity types
    all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())

    # Calculate metrics for each type
    metrics_by_type = {}

    for entity_type in all_types:
        pred = pred_by_type.get(entity_type, [])
        gold = gold_by_type.get(entity_type, [])

        metrics = calculate_precision_recall_f1(pred, gold, match_mode)
        metrics_by_type[entity_type] = metrics

    # Calculate macro-averaged metrics (average across types)
    if metrics_by_type:
        macro_precision = sum(m['precision'] for m in metrics_by_type.values()) / len(metrics_by_type)
        macro_recall = sum(m['recall'] for m in metrics_by_type.values()) / len(metrics_by_type)
        macro_f1 = sum(m['f1_score'] for m in metrics_by_type.values()) / len(metrics_by_type)
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    # Calculate micro-averaged metrics (aggregate TP/FP/FN across all types)
    total_tp = sum(m['true_positives'] for m in metrics_by_type.values())
    total_fp = sum(m['false_positives'] for m in metrics_by_type.values())
    total_fn = sum(m['false_negatives'] for m in metrics_by_type.values())

    if total_tp + total_fp > 0:
        micro_precision = total_tp / (total_tp + total_fp)
    else:
        micro_precision = 0.0

    if total_tp + total_fn > 0:
        micro_recall = total_tp / (total_tp + total_fn)
    else:
        micro_recall = 0.0

    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_f1 = 0.0

    return {
        'by_type': metrics_by_type,
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        },
        'micro': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1_score': micro_f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
    }


def calculate_boundary_metrics(predicted_entities: List[Dict],
                               gold_entities: List[Dict]) -> Dict:
    """
    Calculate boundary detection metrics (relaxed matching)

    Measures how well annotators identify entity boundaries,
    allowing for small boundary errors

    Args:
        predicted_entities: List of predicted entities
        gold_entities: List of gold standard entities

    Returns:
        Dictionary with boundary-level metrics
    """
    def boundary_overlap(pred, gold, threshold=0.5):
        """Check if boundaries overlap with at least threshold IoU"""
        start_pred, end_pred, _, _ = normalize_entity(pred)
        start_gold, end_gold, _, _ = normalize_entity(gold)

        # Calculate intersection over union (IoU)
        intersection = max(0, min(end_pred, end_gold) - max(start_pred, start_gold))
        union = max(end_pred, end_gold) - min(start_pred, start_gold)

        if union == 0:
            return False

        iou = intersection / union
        return iou >= threshold

    # Count boundary matches
    boundary_matches = []
    remaining_gold = list(gold_entities)

    for pred_entity in predicted_entities:
        for i, gold_entity in enumerate(remaining_gold):
            if boundary_overlap(pred_entity, gold_entity):
                boundary_matches.append((pred_entity, gold_entity))
                remaining_gold.pop(i)
                break

    boundary_tp = len(boundary_matches)
    boundary_fp = len(predicted_entities) - boundary_tp
    boundary_fn = len(remaining_gold)

    # Calculate boundary metrics
    if boundary_tp + boundary_fp > 0:
        boundary_precision = boundary_tp / (boundary_tp + boundary_fp)
    else:
        boundary_precision = 0.0

    if boundary_tp + boundary_fn > 0:
        boundary_recall = boundary_tp / (boundary_tp + boundary_fn)
    else:
        boundary_recall = 0.0

    if boundary_precision + boundary_recall > 0:
        boundary_f1 = 2 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall)
    else:
        boundary_f1 = 0.0

    return {
        'boundary_precision': boundary_precision,
        'boundary_recall': boundary_recall,
        'boundary_f1': boundary_f1,
        'boundary_matches': boundary_tp,
        'boundary_errors': boundary_fp + boundary_fn
    }


def calculate_annotation_efficiency(annotation_time_seconds: float,
                                    n_entities: int,
                                    document_length: int) -> Dict:
    """
    Calculate efficiency metrics for annotation

    Args:
        annotation_time_seconds: Total time spent on annotation
        n_entities: Number of entities annotated
        document_length: Length of document (words or characters)

    Returns:
        Dictionary with efficiency metrics
    """
    if annotation_time_seconds <= 0:
        return {
            'time_per_entity_seconds': 0.0,
            'time_per_entity_minutes': 0.0,
            'entities_per_minute': 0.0,
            'time_per_token_seconds': 0.0
        }

    time_per_entity_seconds = annotation_time_seconds / n_entities if n_entities > 0 else 0.0
    time_per_entity_minutes = time_per_entity_seconds / 60

    entities_per_minute = (n_entities / annotation_time_seconds) * 60 if annotation_time_seconds > 0 else 0.0

    time_per_token = annotation_time_seconds / document_length if document_length > 0 else 0.0

    return {
        'total_time_seconds': annotation_time_seconds,
        'total_time_minutes': annotation_time_seconds / 60,
        'time_per_entity_seconds': time_per_entity_seconds,
        'time_per_entity_minutes': time_per_entity_minutes,
        'entities_per_minute': entities_per_minute,
        'time_per_token_seconds': time_per_token,
        'throughput': entities_per_minute  # Entities annotated per minute
    }


def calculate_confusion_matrix_by_type(predicted_entities: List[Dict],
                                       gold_entities: List[Dict]) -> Dict:
    """
    Calculate confusion matrix for entity type classification

    Shows which entity types are confused with each other

    Args:
        predicted_entities: List of predicted entities
        gold_entities: List of gold standard entities

    Returns:
        Confusion matrix as nested dictionary
    """
    # Match entities by span (regardless of type)
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    # Track which gold entities have been matched
    matched_gold_indices = set()

    for pred_entity in predicted_entities:
        start_pred, end_pred, label_pred, _ = normalize_entity(pred_entity)

        # Find matching gold entity by span
        matched = False
        for i, gold_entity in enumerate(gold_entities):
            if i in matched_gold_indices:
                continue

            start_gold, end_gold, label_gold, _ = normalize_entity(gold_entity)

            # Check for exact span match
            if start_pred == start_gold and end_pred == end_gold:
                confusion_matrix[label_gold][label_pred] += 1
                matched_gold_indices.add(i)
                matched = True
                break

        if not matched:
            # False positive - predicted entity with no gold match
            confusion_matrix['O'][label_pred] += 1  # 'O' represents no entity

    # Add false negatives - gold entities not predicted
    for i, gold_entity in enumerate(gold_entities):
        if i not in matched_gold_indices:
            _, _, label_gold, _ = normalize_entity(gold_entity)
            confusion_matrix[label_gold]['O'] += 1

    return dict(confusion_matrix)
