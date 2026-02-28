"""
Advanced Partial Matching Algorithms for Overlapping Entity Evaluation

This module provides sophisticated algorithms for evaluating LLM performance on overlapping
medical entity annotations, including:

1. Multi-level matching strategies (exact, partial, relaxed)
2. Overlap-aware evaluation metrics
3. Partial credit systems for near-misses
4. Complex performance analytics for LLM improvement
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from itertools import combinations
import math


@dataclass
class EntityMatch:
    """Represents a match between predicted and gold entities"""
    predicted_entity: object
    gold_entity: object
    match_score: float
    match_type: str  # 'exact', 'partial', 'type_mismatch', 'boundary_error'
    overlap_ratio: float
    iou_score: float
    type_match: bool
    boundary_error_type: str  # 'none', 'start_error', 'end_error', 'both_error'

    # Detailed scoring breakdown
    position_score: float = 0.0
    type_score: float = 0.0
    text_similarity_score: float = 0.0
    confidence_penalty: float = 0.0


@dataclass
class MatchingResult:
    """Complete matching result for an annotation comparison"""
    matches: List[EntityMatch] = field(default_factory=list)
    unmatched_predicted: List[object] = field(default_factory=list)
    unmatched_gold: List[object] = field(default_factory=list)

    # Aggregate scores
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Detailed metrics
    exact_matches: int = 0
    partial_matches: int = 0
    type_errors: int = 0
    boundary_errors: int = 0

    # Overlap-specific metrics
    overlap_precision: float = 0.0
    overlap_recall: float = 0.0
    overlap_f1: float = 0.0

    # Performance breakdown
    performance_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_by_overlap_level: Dict[str, Dict[str, float]] = field(default_factory=dict)


class AdvancedPartialMatcher:
    """Advanced matching algorithm for overlapping entities with sophisticated scoring"""

    def __init__(self,
                 exact_threshold: float = 1.0,
                 partial_threshold: float = 0.5,
                 relaxed_threshold: float = 0.3,
                 type_weight: float = 0.3,
                 position_weight: float = 0.5,
                 text_weight: float = 0.2,
                 allow_one_to_many: bool = True,
                 allow_many_to_one: bool = True):
        """
        Initialize the advanced partial matcher

        Args:
            exact_threshold: Threshold for exact matches (default 1.0)
            partial_threshold: Threshold for partial matches (default 0.5)
            relaxed_threshold: Minimum threshold for any match (default 0.3)
            type_weight: Weight for entity type matching (default 0.3)
            position_weight: Weight for position/boundary matching (default 0.5)
            text_weight: Weight for text similarity (default 0.2)
            allow_one_to_many: Allow one gold entity to match multiple predicted
            allow_many_to_one: Allow multiple gold entities to match one predicted
        """
        self.exact_threshold = exact_threshold
        self.partial_threshold = partial_threshold
        self.relaxed_threshold = relaxed_threshold
        self.type_weight = type_weight
        self.position_weight = position_weight
        self.text_weight = text_weight
        self.allow_one_to_many = allow_one_to_many
        self.allow_many_to_one = allow_many_to_one

    def compare_annotations(self,
                          predicted_entities: List[object],
                          gold_entities: List[object]) -> MatchingResult:
        """
        Compare predicted and gold annotations with sophisticated matching

        Args:
            predicted_entities: List of predicted entities
            gold_entities: List of gold standard entities

        Returns:
            MatchingResult with detailed comparison metrics
        """
        if not predicted_entities and not gold_entities:
            return MatchingResult(precision=1.0, recall=1.0, f1_score=1.0)

        if not predicted_entities:
            return MatchingResult(
                unmatched_gold=gold_entities.copy(),
                precision=0.0, recall=0.0, f1_score=0.0
            )

        if not gold_entities:
            return MatchingResult(
                unmatched_predicted=predicted_entities.copy(),
                precision=0.0, recall=0.0, f1_score=0.0
            )

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(predicted_entities, gold_entities)

        # Find optimal matching using Hungarian algorithm variant
        matches = self._find_optimal_matching(predicted_entities, gold_entities, similarity_matrix)

        # Calculate detailed metrics
        result = self._calculate_detailed_metrics(predicted_entities, gold_entities, matches)

        # Add overlap-specific analysis
        result = self._add_overlap_analysis(result, predicted_entities, gold_entities)

        return result

    def _calculate_similarity_matrix(self,
                                   predicted_entities: List[object],
                                   gold_entities: List[object]) -> np.ndarray:
        """Calculate similarity matrix between all predicted and gold entities"""
        matrix = np.zeros((len(predicted_entities), len(gold_entities)))

        for i, pred_entity in enumerate(predicted_entities):
            for j, gold_entity in enumerate(gold_entities):
                similarity = self._calculate_entity_similarity(pred_entity, gold_entity)
                matrix[i, j] = similarity.match_score

        return matrix

    def _calculate_entity_similarity(self, pred_entity: object, gold_entity: object) -> EntityMatch:
        """Calculate detailed similarity between two entities"""

        # Position-based similarity (IoU)
        iou_score = self._calculate_iou(pred_entity, gold_entity)
        overlap_ratio = self._calculate_overlap_ratio(pred_entity, gold_entity)

        # Type matching
        type_match = pred_entity.entity_type == gold_entity.entity_type
        type_score = 1.0 if type_match else 0.0

        # Text similarity
        text_similarity = self._calculate_text_similarity(pred_entity.text, gold_entity.text)

        # Boundary error analysis
        boundary_error_type = self._analyze_boundary_errors(pred_entity, gold_entity)

        # Position score with boundary error penalty
        position_score = iou_score
        if boundary_error_type != 'none':
            # Apply penalty for boundary errors
            if boundary_error_type == 'both_error':
                position_score *= 0.7
            else:  # start_error or end_error
                position_score *= 0.85

        # Confidence penalty (if available)
        confidence_penalty = 0.0
        if hasattr(pred_entity, 'confidence_score') and pred_entity.confidence_score is not None:
            # Penalty for low confidence predictions
            if pred_entity.confidence_score < 0.5:
                confidence_penalty = (0.5 - pred_entity.confidence_score) * 0.2

        # Calculate weighted score
        weighted_score = (
            self.position_weight * position_score +
            self.type_weight * type_score +
            self.text_weight * text_similarity
        ) - confidence_penalty

        # Determine match type
        match_type = self._determine_match_type(weighted_score, type_match, iou_score)

        return EntityMatch(
            predicted_entity=pred_entity,
            gold_entity=gold_entity,
            match_score=max(0.0, weighted_score),
            match_type=match_type,
            overlap_ratio=overlap_ratio,
            iou_score=iou_score,
            type_match=type_match,
            boundary_error_type=boundary_error_type,
            position_score=position_score,
            type_score=type_score,
            text_similarity_score=text_similarity,
            confidence_penalty=confidence_penalty
        )

    def _calculate_iou(self, entity1: object, entity2: object) -> float:
        """Calculate Intersection over Union (IoU) for entity positions"""
        if not hasattr(entity1, 'start_offset') or not hasattr(entity1, 'end_offset'):
            return 0.0

        # Calculate intersection
        intersection_start = max(entity1.start_offset, entity2.start_offset)
        intersection_end = min(entity1.end_offset, entity2.end_offset)

        if intersection_start >= intersection_end:
            return 0.0  # No overlap

        intersection_length = intersection_end - intersection_start

        # Calculate union
        union_start = min(entity1.start_offset, entity2.start_offset)
        union_end = max(entity1.end_offset, entity2.end_offset)
        union_length = union_end - union_start

        return intersection_length / union_length if union_length > 0 else 0.0

    def _calculate_overlap_ratio(self, entity1: object, entity2: object) -> float:
        """Calculate overlap as ratio of the smaller entity"""
        if not hasattr(entity1, 'start_offset') or not hasattr(entity1, 'end_offset'):
            return 0.0

        intersection_start = max(entity1.start_offset, entity2.start_offset)
        intersection_end = min(entity1.end_offset, entity2.end_offset)

        if intersection_start >= intersection_end:
            return 0.0

        intersection_length = intersection_end - intersection_start

        entity1_length = entity1.end_offset - entity1.start_offset
        entity2_length = entity2.end_offset - entity2.start_offset

        min_length = min(entity1_length, entity2_length)
        return intersection_length / min_length if min_length > 0 else 0.0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using token-based Jaccard similarity"""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

    def _analyze_boundary_errors(self, pred_entity: object, gold_entity: object) -> str:
        """Analyze boundary error patterns"""
        if not hasattr(pred_entity, 'start_offset'):
            return 'none'

        start_match = pred_entity.start_offset == gold_entity.start_offset
        end_match = pred_entity.end_offset == gold_entity.end_offset

        if start_match and end_match:
            return 'none'
        elif not start_match and not end_match:
            return 'both_error'
        elif not start_match:
            return 'start_error'
        else:
            return 'end_error'

    def _determine_match_type(self, score: float, type_match: bool, iou_score: float) -> str:
        """Determine the type of match based on scores"""
        if score >= self.exact_threshold and type_match and iou_score >= 0.95:
            return 'exact'
        elif score >= self.partial_threshold and type_match:
            return 'partial'
        elif score >= self.relaxed_threshold and not type_match:
            return 'type_mismatch'
        elif score >= self.relaxed_threshold:
            return 'boundary_error'
        else:
            return 'no_match'

    def _find_optimal_matching(self,
                             predicted_entities: List[object],
                             gold_entities: List[object],
                             similarity_matrix: np.ndarray) -> List[EntityMatch]:
        """Find optimal matching using greedy approach with overlap handling"""

        matches = []
        used_predicted = set()
        used_gold = set()

        # Create list of all possible matches above threshold
        possible_matches = []

        for i, pred_entity in enumerate(predicted_entities):
            for j, gold_entity in enumerate(gold_entities):
                if similarity_matrix[i, j] >= self.relaxed_threshold:
                    match = self._calculate_entity_similarity(pred_entity, gold_entity)
                    possible_matches.append((i, j, match))

        # Sort by match score (descending)
        possible_matches.sort(key=lambda x: x[2].match_score, reverse=True)

        # Greedily select matches
        for pred_idx, gold_idx, match in possible_matches:
            # Check if we can use this match based on our constraints
            can_use_predicted = (pred_idx not in used_predicted or
                               self.allow_many_to_one)
            can_use_gold = (gold_idx not in used_gold or
                          self.allow_one_to_many)

            if can_use_predicted and can_use_gold:
                matches.append(match)
                used_predicted.add(pred_idx)
                used_gold.add(gold_idx)

        return matches

    def _calculate_detailed_metrics(self,
                                  predicted_entities: List[object],
                                  gold_entities: List[object],
                                  matches: List[EntityMatch]) -> MatchingResult:
        """Calculate detailed metrics from matches"""

        result = MatchingResult()
        result.matches = matches

        # Count match types
        exact_matches = sum(1 for m in matches if m.match_type == 'exact')
        partial_matches = sum(1 for m in matches if m.match_type == 'partial')
        type_errors = sum(1 for m in matches if m.match_type == 'type_mismatch')
        boundary_errors = sum(1 for m in matches if m.match_type == 'boundary_error')

        result.exact_matches = exact_matches
        result.partial_matches = partial_matches
        result.type_errors = type_errors
        result.boundary_errors = boundary_errors

        # Calculate precision, recall, F1 with partial credit
        true_positives = 0.0
        for match in matches:
            if match.match_type == 'exact':
                true_positives += 1.0
            elif match.match_type == 'partial':
                true_positives += 0.8
            elif match.match_type == 'boundary_error':
                true_positives += 0.6
            elif match.match_type == 'type_mismatch':
                true_positives += 0.4

        # Identify unmatched entities
        matched_predicted_ids = {id(m.predicted_entity) for m in matches}
        matched_gold_ids = {id(m.gold_entity) for m in matches}

        result.unmatched_predicted = [
            e for e in predicted_entities
            if id(e) not in matched_predicted_ids
        ]
        result.unmatched_gold = [
            e for e in gold_entities
            if id(e) not in matched_gold_ids
        ]

        # Calculate metrics
        total_predicted = len(predicted_entities)
        total_gold = len(gold_entities)

        result.precision = true_positives / total_predicted if total_predicted > 0 else 0.0
        result.recall = true_positives / total_gold if total_gold > 0 else 0.0
        result.f1_score = (2 * result.precision * result.recall /
                          (result.precision + result.recall)
                          if (result.precision + result.recall) > 0 else 0.0)

        # Calculate performance by entity type
        result.performance_by_type = self._calculate_performance_by_type(
            predicted_entities, gold_entities, matches
        )

        return result

    def _add_overlap_analysis(self,
                            result: MatchingResult,
                            predicted_entities: List[object],
                            gold_entities: List[object]) -> MatchingResult:
        """Add overlap-specific analysis to the result"""

        # Identify overlap groups
        pred_overlap_groups = self._identify_overlap_groups(predicted_entities)
        gold_overlap_groups = self._identify_overlap_groups(gold_entities)

        # Calculate overlap-aware metrics
        overlap_metrics = self._calculate_overlap_aware_metrics(
            pred_overlap_groups, gold_overlap_groups
        )

        result.overlap_precision = overlap_metrics['precision']
        result.overlap_recall = overlap_metrics['recall']
        result.overlap_f1 = overlap_metrics['f1']

        # Calculate performance by overlap level
        result.performance_by_overlap_level = self._calculate_performance_by_overlap_level(
            predicted_entities, gold_entities, result.matches
        )

        return result

    def _identify_overlap_groups(self, entities: List[object]) -> List[List[object]]:
        """Identify groups of overlapping entities using Union-Find"""
        if not entities:
            return []

        # Initialize Union-Find structure
        parent = {id(entity): id(entity) for entity in entities}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        # Find overlapping pairs and union them
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                if self._entities_overlap(entity1, entity2):
                    union(id(entity1), id(entity2))

        # Group entities by their root
        groups = defaultdict(list)
        for entity in entities:
            root = find(id(entity))
            groups[root].append(entity)

        return list(groups.values())

    def _entities_overlap(self, entity1: object, entity2: object) -> bool:
        """Check if two entities overlap"""
        if not all(hasattr(e, 'start_offset') and hasattr(e, 'end_offset')
                  for e in [entity1, entity2]):
            return False

        return not (entity1.end_offset <= entity2.start_offset or
                   entity1.start_offset >= entity2.end_offset)

    def _calculate_overlap_aware_metrics(self,
                                       pred_groups: List[List[object]],
                                       gold_groups: List[List[object]]) -> Dict[str, float]:
        """Calculate metrics that account for overlapping entity structures"""

        if not pred_groups and not gold_groups:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}

        if not pred_groups:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        if not gold_groups:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Match groups using group similarity
        matched_groups = 0
        used_gold_groups = set()

        for pred_group in pred_groups:
            best_similarity = 0.0
            best_gold_idx = -1

            for gold_idx, gold_group in enumerate(gold_groups):
                if gold_idx in used_gold_groups:
                    continue

                similarity = self._calculate_group_similarity(pred_group, gold_group)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_gold_idx = gold_idx

            if best_similarity >= self.partial_threshold and best_gold_idx >= 0:
                matched_groups += 1
                used_gold_groups.add(best_gold_idx)

        precision = matched_groups / len(pred_groups)
        recall = matched_groups / len(gold_groups)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _calculate_group_similarity(self, group1: List[object], group2: List[object]) -> float:
        """Calculate similarity between two groups of overlapping entities"""
        if not group1 or not group2:
            return 0.0

        # Calculate pairwise similarities and find best matching
        total_similarity = 0.0
        matched_entities = 0
        used_group2_indices = set()

        for entity1 in group1:
            best_similarity = 0.0
            best_idx = -1

            for idx, entity2 in enumerate(group2):
                if idx in used_group2_indices:
                    continue

                similarity = self._calculate_entity_similarity(entity1, entity2).match_score
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = idx

            if best_similarity >= self.relaxed_threshold and best_idx >= 0:
                total_similarity += best_similarity
                matched_entities += 1
                used_group2_indices.add(best_idx)

        # Normalize by the size of the larger group
        max_group_size = max(len(group1), len(group2))
        return total_similarity / max_group_size if max_group_size > 0 else 0.0

    def _calculate_performance_by_type(self,
                                     predicted_entities: List[object],
                                     gold_entities: List[object],
                                     matches: List[EntityMatch]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics broken down by entity type"""

        # Group entities and matches by type
        pred_by_type = defaultdict(list)
        gold_by_type = defaultdict(list)
        matches_by_type = defaultdict(list)

        for entity in predicted_entities:
            pred_by_type[entity.entity_type].append(entity)

        for entity in gold_entities:
            gold_by_type[entity.entity_type].append(entity)

        for match in matches:
            # Use gold entity type for categorization
            gold_type = match.gold_entity.entity_type
            matches_by_type[gold_type].append(match)

        # Calculate metrics for each type
        performance = {}
        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())

        for entity_type in all_types:
            pred_entities = pred_by_type[entity_type]
            gold_entities = gold_by_type[entity_type]
            type_matches = matches_by_type[entity_type]

            # Calculate true positives with partial credit
            true_positives = 0.0
            for match in type_matches:
                if match.match_type == 'exact':
                    true_positives += 1.0
                elif match.match_type == 'partial':
                    true_positives += 0.8
                elif match.match_type == 'boundary_error':
                    true_positives += 0.6

            precision = true_positives / len(pred_entities) if pred_entities else 0.0
            recall = true_positives / len(gold_entities) if gold_entities else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            performance[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predicted_count': len(pred_entities),
                'gold_count': len(gold_entities),
                'matched_count': len(type_matches)
            }

        return performance

    def _calculate_performance_by_overlap_level(self,
                                              predicted_entities: List[object],
                                              gold_entities: List[object],
                                              matches: List[EntityMatch]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by overlap complexity level"""

        # Categorize entities by overlap level
        pred_by_level = self._categorize_by_overlap_level(predicted_entities)
        gold_by_level = self._categorize_by_overlap_level(gold_entities)

        # Categorize matches by overlap level (using gold entity level)
        matches_by_level = defaultdict(list)
        for match in matches:
            gold_entity = match.gold_entity
            level = self._get_entity_overlap_level(gold_entity, gold_entities)
            matches_by_level[level].append(match)

        # Calculate metrics for each level
        performance = {}
        all_levels = set(pred_by_level.keys()) | set(gold_by_level.keys())

        for level in all_levels:
            pred_entities = pred_by_level.get(level, [])
            gold_entities_level = gold_by_level.get(level, [])
            level_matches = matches_by_level[level]

            # Calculate metrics similar to type-based breakdown
            true_positives = sum(1.0 if m.match_type == 'exact' else
                               0.8 if m.match_type == 'partial' else
                               0.6 if m.match_type == 'boundary_error' else 0.0
                               for m in level_matches)

            precision = true_positives / len(pred_entities) if pred_entities else 0.0
            recall = true_positives / len(gold_entities_level) if gold_entities_level else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            performance[level] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predicted_count': len(pred_entities),
                'gold_count': len(gold_entities_level),
                'matched_count': len(level_matches)
            }

        return performance

    def _categorize_by_overlap_level(self, entities: List[object]) -> Dict[str, List[object]]:
        """Categorize entities by their overlap complexity level"""
        levels = defaultdict(list)

        for entity in entities:
            level = self._get_entity_overlap_level(entity, entities)
            levels[level].append(entity)

        return levels

    def _get_entity_overlap_level(self, entity: object, all_entities: List[object]) -> str:
        """Get the overlap complexity level for an entity"""
        overlap_count = 0

        for other_entity in all_entities:
            if id(entity) != id(other_entity) and self._entities_overlap(entity, other_entity):
                overlap_count += 1

        if overlap_count == 0:
            return 'isolated'
        elif overlap_count <= 2:
            return 'simple'
        elif overlap_count <= 5:
            return 'moderate'
        else:
            return 'complex'


class LLMPerformanceAnalyzer:
    """Analyze LLM performance with detailed insights for model improvement"""

    def __init__(self, matcher: AdvancedPartialMatcher = None):
        self.matcher = matcher or AdvancedPartialMatcher()

    def analyze_performance(self,
                          predicted_annotations: List[object],
                          gold_annotations: List[object]) -> Dict[str, any]:
        """
        Comprehensive performance analysis for LLM improvement

        Returns detailed insights including error patterns, improvement suggestions,
        and performance trends.
        """

        if len(predicted_annotations) != len(gold_annotations):
            raise ValueError("Number of predicted and gold annotations must match")

        results = []

        # Analyze each annotation pair
        for pred_ann, gold_ann in zip(predicted_annotations, gold_annotations):
            pred_entities = list(pred_ann.overlapping_entities.all())
            gold_entities = list(gold_ann.overlapping_entities.all())

            result = self.matcher.compare_annotations(pred_entities, gold_entities)
            results.append(result)

        # Aggregate results and provide insights
        analysis = {
            'overall_performance': self._calculate_overall_performance(results),
            'error_patterns': self._analyze_error_patterns(results),
            'improvement_suggestions': self._generate_improvement_suggestions(results),
            'performance_trends': self._analyze_performance_trends(results),
            'entity_type_insights': self._analyze_entity_type_performance(results),
            'overlap_complexity_insights': self._analyze_overlap_complexity_performance(results)
        }

        return analysis

    def _calculate_overall_performance(self, results: List[MatchingResult]) -> Dict[str, float]:
        """Calculate aggregated performance metrics"""
        if not results:
            return {}

        total_precision = sum(r.precision for r in results)
        total_recall = sum(r.recall for r in results)
        total_f1 = sum(r.f1_score for r in results)

        total_overlap_precision = sum(r.overlap_precision for r in results)
        total_overlap_recall = sum(r.overlap_recall for r in results)
        total_overlap_f1 = sum(r.overlap_f1 for r in results)

        count = len(results)

        return {
            'avg_precision': total_precision / count,
            'avg_recall': total_recall / count,
            'avg_f1': total_f1 / count,
            'avg_overlap_precision': total_overlap_precision / count,
            'avg_overlap_recall': total_overlap_recall / count,
            'avg_overlap_f1': total_overlap_f1 / count,
            'total_annotations': count
        }

    def _analyze_error_patterns(self, results: List[MatchingResult]) -> Dict[str, any]:
        """Analyze common error patterns for model improvement"""

        error_patterns = {
            'boundary_errors': {'count': 0, 'examples': []},
            'type_errors': {'count': 0, 'examples': []},
            'missed_entities': {'count': 0, 'types': Counter()},
            'false_positives': {'count': 0, 'types': Counter()},
            'overlap_handling_errors': {'count': 0, 'examples': []}
        }

        for result in results:
            # Count error types
            error_patterns['boundary_errors']['count'] += result.boundary_errors
            error_patterns['type_errors']['count'] += result.type_errors

            # Analyze unmatched entities
            for entity in result.unmatched_gold:
                error_patterns['missed_entities']['count'] += 1
                error_patterns['missed_entities']['types'][entity.entity_type] += 1

            for entity in result.unmatched_predicted:
                error_patterns['false_positives']['count'] += 1
                error_patterns['false_positives']['types'][entity.entity_type] += 1

            # Collect examples of different error types
            for match in result.matches:
                if match.match_type == 'boundary_error':
                    error_patterns['boundary_errors']['examples'].append({
                        'predicted_text': match.predicted_entity.text,
                        'gold_text': match.gold_entity.text,
                        'boundary_error_type': match.boundary_error_type
                    })
                elif match.match_type == 'type_mismatch':
                    error_patterns['type_errors']['examples'].append({
                        'text': match.predicted_entity.text,
                        'predicted_type': match.predicted_entity.entity_type,
                        'gold_type': match.gold_entity.entity_type
                    })

        return error_patterns

    def _generate_improvement_suggestions(self, results: List[MatchingResult]) -> List[str]:
        """Generate specific suggestions for improving LLM performance"""
        suggestions = []

        # Analyze overall performance
        avg_precision = sum(r.precision for r in results) / len(results) if results else 0
        avg_recall = sum(r.recall for r in results) / len(results) if results else 0

        if avg_precision < 0.7:
            suggestions.append("Consider adding more negative examples to reduce false positives")

        if avg_recall < 0.7:
            suggestions.append("Consider augmenting training data with more diverse entity examples")

        # Analyze boundary errors
        total_boundary_errors = sum(r.boundary_errors for r in results)
        total_entities = sum(len(r.matches) for r in results)

        if total_entities > 0 and (total_boundary_errors / total_entities) > 0.2:
            suggestions.append("Focus on boundary detection training with character-level attention")

        # Analyze overlap handling
        avg_overlap_f1 = sum(r.overlap_f1 for r in results) / len(results) if results else 0
        if avg_overlap_f1 < avg_precision - 0.1:
            suggestions.append("Improve overlapping entity detection with specialized training")

        return suggestions

    def _analyze_performance_trends(self, results: List[MatchingResult]) -> Dict[str, any]:
        """Analyze performance trends across annotations"""

        if len(results) < 2:
            return {'message': 'Insufficient data for trend analysis'}

        # Calculate moving averages
        window_size = min(5, len(results))
        f1_scores = [r.f1_score for r in results]

        moving_avg = []
        for i in range(len(f1_scores) - window_size + 1):
            avg = sum(f1_scores[i:i+window_size]) / window_size
            moving_avg.append(avg)

        # Detect trends
        if len(moving_avg) >= 2:
            trend = "improving" if moving_avg[-1] > moving_avg[0] else "declining"
        else:
            trend = "stable"

        return {
            'trend': trend,
            'f1_scores': f1_scores,
            'moving_average': moving_avg,
            'best_performance': max(f1_scores),
            'worst_performance': min(f1_scores),
            'performance_variance': np.var(f1_scores) if f1_scores else 0
        }

    def _analyze_entity_type_performance(self, results: List[MatchingResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by entity type across all annotations"""

        type_performance = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})

        for result in results:
            for entity_type, metrics in result.performance_by_type.items():
                type_performance[entity_type]['precision'].append(metrics['precision'])
                type_performance[entity_type]['recall'].append(metrics['recall'])
                type_performance[entity_type]['f1'].append(metrics['f1'])

        # Calculate averages
        summary = {}
        for entity_type, metrics in type_performance.items():
            summary[entity_type] = {
                'avg_precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
                'avg_recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
                'avg_f1': np.mean(metrics['f1']) if metrics['f1'] else 0,
                'precision_std': np.std(metrics['precision']) if metrics['precision'] else 0,
                'recall_std': np.std(metrics['recall']) if metrics['recall'] else 0,
                'f1_std': np.std(metrics['f1']) if metrics['f1'] else 0
            }

        return summary

    def _analyze_overlap_complexity_performance(self, results: List[MatchingResult]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by overlap complexity level"""

        complexity_performance = defaultdict(lambda: {'precision': [], 'recall': [], 'f1': []})

        for result in results:
            for level, metrics in result.performance_by_overlap_level.items():
                complexity_performance[level]['precision'].append(metrics['precision'])
                complexity_performance[level]['recall'].append(metrics['recall'])
                complexity_performance[level]['f1'].append(metrics['f1'])

        # Calculate averages
        summary = {}
        for level, metrics in complexity_performance.items():
            summary[level] = {
                'avg_precision': np.mean(metrics['precision']) if metrics['precision'] else 0,
                'avg_recall': np.mean(metrics['recall']) if metrics['recall'] else 0,
                'avg_f1': np.mean(metrics['f1']) if metrics['f1'] else 0,
                'sample_count': len(metrics['f1'])
            }

        return summary