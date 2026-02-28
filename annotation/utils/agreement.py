"""
Utility functions for calculating inter-annotator agreement metrics
Including Cohen's kappa and entity-level agreement
"""

import numpy as np
from typing import List, Dict, Tuple


def normalize_entity(entity: Dict) -> Tuple[int, int, str]:
    """
    Normalize an entity to a comparable format
    Returns: (start, end, label) tuple
    """
    start = entity.get('start', entity.get('start_char', 0))
    end = entity.get('end', entity.get('end_char', 0))
    label = entity.get('label', entity.get('type', 'UNKNOWN'))
    return (start, end, label)


def entities_match(entity1: Dict, entity2: Dict, match_mode='exact') -> bool:
    """
    Check if two entities match based on the specified mode

    Args:
        entity1: First entity dictionary
        entity2: Second entity dictionary
        match_mode: 'exact' (exact span and label), 'overlap' (overlapping span and same label),
                   'partial' (any overlap regardless of label)

    Returns:
        True if entities match according to the mode
    """
    start1, end1, label1 = normalize_entity(entity1)
    start2, end2, label2 = normalize_entity(entity2)

    if match_mode == 'exact':
        return start1 == start2 and end1 == end2 and label1 == label2

    elif match_mode == 'overlap':
        # Check if spans overlap and labels match
        overlap = max(0, min(end1, end2) - max(start1, start2))
        return overlap > 0 and label1 == label2

    elif match_mode == 'partial':
        # Check if spans overlap at all
        overlap = max(0, min(end1, end2) - max(start1, start2))
        return overlap > 0

    return False


def calculate_entity_agreement(entities1: List[Dict], entities2: List[Dict],
                               match_mode='exact') -> Dict:
    """
    Calculate entity-level agreement between two annotators

    Args:
        entities1: List of entities from annotator 1
        entities2: List of entities from annotator 2
        match_mode: How to match entities ('exact', 'overlap', 'partial')

    Returns:
        Dictionary with agreement statistics
    """
    # Find matching entities
    matched_entities = []
    unmatched_entities1 = []
    unmatched_entities2 = list(entities2)  # Copy to track unmatched

    for entity1 in entities1:
        found_match = False
        for i, entity2 in enumerate(unmatched_entities2):
            if entities_match(entity1, entity2, match_mode):
                matched_entities.append((entity1, entity2))
                unmatched_entities2.pop(i)
                found_match = True
                break

        if not found_match:
            unmatched_entities1.append(entity1)

    # Calculate agreement statistics
    total_entities1 = len(entities1)
    total_entities2 = len(entities2)
    agreed = len(matched_entities)
    disagreed = len(unmatched_entities1) + len(unmatched_entities2)

    # Calculate percent agreement
    total_annotations = total_entities1 + total_entities2
    if total_annotations == 0:
        percent_agreement = 100.0
    else:
        percent_agreement = (2 * agreed / total_annotations) * 100

    # Calculate agreement by entity type
    agreement_by_type = {}
    for entity1, entity2 in matched_entities:
        label = normalize_entity(entity1)[2]
        if label not in agreement_by_type:
            agreement_by_type[label] = {'agreed': 0, 'total1': 0, 'total2': 0}
        agreement_by_type[label]['agreed'] += 1

    # Count totals by type
    for entity in entities1:
        label = normalize_entity(entity)[2]
        if label not in agreement_by_type:
            agreement_by_type[label] = {'agreed': 0, 'total1': 0, 'total2': 0}
        agreement_by_type[label]['total1'] += 1

    for entity in entities2:
        label = normalize_entity(entity)[2]
        if label not in agreement_by_type:
            agreement_by_type[label] = {'agreed': 0, 'total1': 0, 'total2': 0}
        agreement_by_type[label]['total2'] += 1

    return {
        'agreed_entities': agreed,
        'disagreed_entities': disagreed,
        'percent_agreement': percent_agreement,
        'total_entities1': total_entities1,
        'total_entities2': total_entities2,
        'details': {
            'matched_entities': matched_entities,
            'unmatched_annotator1': unmatched_entities1,
            'unmatched_annotator2': unmatched_entities2,
            'agreement_by_type': agreement_by_type
        }
    }


def create_agreement_matrix(entities1: List[Dict], entities2: List[Dict],
                            document_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary agreement matrices for Cohen's kappa calculation

    Args:
        entities1: Entities from annotator 1
        entities2: Entities from annotator 2
        document_length: Length of the document in characters

    Returns:
        Tuple of (annotator1_matrix, annotator2_matrix)
    """
    # Create binary vectors representing annotations at each character position
    matrix1 = np.zeros(document_length, dtype=int)
    matrix2 = np.zeros(document_length, dtype=int)

    # Mark entity positions for annotator 1
    for entity in entities1:
        start, end, label = normalize_entity(entity)
        if 0 <= start < document_length and 0 <= end <= document_length:
            matrix1[start:end] = 1

    # Mark entity positions for annotator 2
    for entity in entities2:
        start, end, label = normalize_entity(entity)
        if 0 <= start < document_length and 0 <= end <= document_length:
            matrix2[start:end] = 1

    return matrix1, matrix2


def calculate_cohens_kappa(entities1: List[Dict], entities2: List[Dict],
                           document_length: int = 10000) -> float:
    """
    Calculate Cohen's kappa coefficient for inter-annotator agreement

    Cohen's kappa measures agreement between two raters who classify items,
    correcting for agreement that would occur by chance.

    Formula: κ = (p_o - p_e) / (1 - p_e)
    where:
        p_o = observed agreement
        p_e = expected agreement by chance

    Args:
        entities1: List of entities from annotator 1
        entities2: List of entities from annotator 2
        document_length: Length of document (default 10000 for estimation)

    Returns:
        Cohen's kappa coefficient (-1 to 1)
        1 = perfect agreement
        0 = agreement by chance
        -1 = perfect disagreement
    """
    if not entities1 and not entities2:
        # Both annotators found nothing - perfect agreement
        return 1.0

    # Create agreement matrices
    matrix1, matrix2 = create_agreement_matrix(entities1, entities2, document_length)

    # Calculate observed agreement
    agreements = (matrix1 == matrix2)
    p_o = np.mean(agreements)  # Proportion of agreement

    # Calculate expected agreement by chance
    p_entity1 = np.mean(matrix1 == 1)  # Proportion annotator 1 marked as entity
    p_entity2 = np.mean(matrix2 == 1)  # Proportion annotator 2 marked as entity
    p_nonentity1 = 1 - p_entity1
    p_nonentity2 = 1 - p_entity2

    # Expected agreement = P(both say entity) + P(both say non-entity)
    p_e = (p_entity1 * p_entity2) + (p_nonentity1 * p_nonentity2)

    # Calculate kappa
    if p_e == 1.0:
        # Perfect expected agreement (degenerate case)
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)

    return kappa


def calculate_multi_annotator_kappa(annotations: Dict[str, List[Dict]],
                                    document_length: int = 10000) -> Dict:
    """
    Calculate pairwise Cohen's kappa for multiple annotators

    Args:
        annotations: Dictionary mapping annotator_id -> list of entities
        document_length: Length of document

    Returns:
        Dictionary with pairwise kappa scores and average
    """
    annotators = list(annotations.keys())
    n_annotators = len(annotators)

    if n_annotators < 2:
        return {'error': 'Need at least 2 annotators'}

    pairwise_kappas = {}
    kappa_values = []

    # Calculate pairwise kappa for all pairs
    for i in range(n_annotators):
        for j in range(i + 1, n_annotators):
            annotator_i = annotators[i]
            annotator_j = annotators[j]

            kappa = calculate_cohens_kappa(
                annotations[annotator_i],
                annotations[annotator_j],
                document_length
            )

            pair_key = f"{annotator_i}_vs_{annotator_j}"
            pairwise_kappas[pair_key] = kappa
            kappa_values.append(kappa)

    # Calculate average kappa
    avg_kappa = np.mean(kappa_values) if kappa_values else 0.0

    return {
        'pairwise_kappas': pairwise_kappas,
        'average_kappa': avg_kappa,
        'min_kappa': min(kappa_values) if kappa_values else 0.0,
        'max_kappa': max(kappa_values) if kappa_values else 0.0,
        'n_pairs': len(kappa_values)
    }


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Cohen's kappa value according to Landis & Koch (1977)

    Args:
        kappa: Cohen's kappa coefficient

    Returns:
        String interpretation of the kappa value
    """
    if kappa < 0:
        return "Poor (worse than chance)"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"
