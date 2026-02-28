"""
Models for supporting overlapping entity annotations and sophisticated LLM performance evaluation.

This module provides comprehensive support for:
1. Overlapping entity annotations (multiple entities sharing text spans)
2. Advanced partial matching algorithms
3. Sophisticated LLM performance metrics
4. Multi-level evaluation (exact, partial, overlap-aware)
"""

from django.db import models
from django.utils import timezone
from django.db.models import Q, Avg, Count, Sum
from authentication.models import User
import difflib
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class OverlappingEntitySchema(models.Model):
    """Schema definitions for overlapping entity types and their interaction rules"""

    ENTITY_TYPES = [
        # Basic Medical Entities
        ('SYMPTOM', 'Symptom'),
        ('DISEASE', 'Disease'),
        ('BODY_PART', 'Body Part'),
        ('MEDICATION', 'Medication'),
        ('PROCEDURE', 'Procedure'),

        # Modifiers and Qualifiers
        ('SEVERITY', 'Severity Level'),
        ('FREQUENCY', 'Frequency'),
        ('DURATION', 'Duration'),
        ('DOSAGE', 'Dosage Amount'),
        ('ROUTE', 'Administration Route'),

        # Temporal and Spatial
        ('TIME', 'Temporal Expression'),
        ('LOCATION', 'Anatomical Location'),
        ('LATERALITY', 'Left/Right Specification'),

        # Complex/Compound Entities
        ('MEDICATION_INSTRUCTION', 'Complete Medication Instruction'),
        ('SYMPTOM_COMPLEX', 'Complex Symptom with Qualifiers'),
        ('PROCEDURE_CONTEXT', 'Procedure with Context'),
        ('DIAGNOSIS_STATEMENT', 'Complete Diagnostic Statement'),

        # Clinical Context
        ('NEGATION', 'Negation/Absence'),
        ('UNCERTAINTY', 'Uncertainty/Speculation'),
        ('FAMILY_HISTORY', 'Family History Context'),
        ('PATIENT_HISTORY', 'Patient History Context'),
    ]

    entity_type = models.CharField(max_length=30, choices=ENTITY_TYPES, unique=True,
                                 help_text="Type of medical entity")

    # Overlap behavior configuration
    can_overlap_with = models.JSONField(default=list,
                                       help_text="List of entity types this can overlap with")
    can_contain = models.JSONField(default=list,
                                  help_text="List of entity types this can completely contain")
    can_be_contained_by = models.JSONField(default=list,
                                          help_text="List of entity types that can contain this")

    # Annotation guidelines
    description = models.TextField(help_text="Clear description of this entity type")
    examples = models.TextField(blank=True, help_text="Examples of this entity type")
    overlap_guidelines = models.TextField(blank=True,
                                        help_text="Guidelines for overlapping with other entities")

    # Priority for overlap resolution (higher priority takes precedence in UI)
    display_priority = models.PositiveIntegerField(default=0,
                                                 help_text="Display priority for overlapping entities")

    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_overlap_schemas')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['display_priority', 'entity_type']
        indexes = [
            models.Index(fields=['entity_type']),
            models.Index(fields=['display_priority']),
            models.Index(fields=['is_active']),
        ]

    def __str__(self):
        return f"{self.get_entity_type_display()}"

    def can_overlap_with_entity(self, other_entity_type):
        """Check if this entity type can overlap with another"""
        return other_entity_type in self.can_overlap_with

    def validate_overlap_configuration(self):
        """Validate that overlap configuration is consistent"""
        errors = []

        for entity_type in self.can_contain:
            if entity_type in self.can_be_contained_by:
                errors.append(f"Circular containment detected with {entity_type}")

        return errors


class OverlappingEntity(models.Model):
    """Individual overlapping entity with sophisticated overlap tracking"""

    # Text span information
    text = models.TextField(help_text="The entity text span")
    start_offset = models.PositiveIntegerField(help_text="Start character position in document")
    end_offset = models.PositiveIntegerField(help_text="End character position in document")
    entity_type = models.CharField(max_length=30, help_text="Entity type label")

    # Parent annotation
    annotation_result = models.ForeignKey('OverlappingAnnotationResult', on_delete=models.CASCADE,
                                        related_name='overlapping_entities')

    # Source and confidence
    source = models.CharField(max_length=20, default='llm',
                            choices=[('manual', 'Manual'), ('llm', 'LLM'), ('hybrid', 'Manual+LLM')])
    confidence_score = models.FloatField(null=True, blank=True, help_text="Prediction confidence")

    # Decision tracking for LLM predictions
    status = models.CharField(max_length=20, default='pending',
                            choices=[('pending', 'Pending Review'),
                                   ('accepted', 'Accepted'),
                                   ('modified', 'Modified'),
                                   ('rejected', 'Rejected'),
                                   ('split', 'Split into Multiple'),
                                   ('merged', 'Merged with Others')])

    # Overlap metadata
    overlap_group_id = models.CharField(max_length=50, null=True, blank=True,
                                      help_text="ID for grouping overlapping entities")
    is_primary_in_group = models.BooleanField(default=False,
                                            help_text="Whether this is the primary entity in overlap group")

    # Modification tracking
    original_text = models.TextField(null=True, blank=True, help_text="Original LLM prediction")
    original_start = models.PositiveIntegerField(null=True, blank=True)
    original_end = models.PositiveIntegerField(null=True, blank=True)
    original_type = models.CharField(max_length=30, null=True, blank=True)

    modified_at = models.DateTimeField(null=True, blank=True)
    modified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                                  related_name='modified_overlapping_entities')

    # SNOMED-CT mapping
    snomed_concept = models.JSONField(null=True, blank=True, help_text="SNOMED-CT concept mapping")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['start_offset', 'end_offset', '-confidence_score']
        indexes = [
            models.Index(fields=['annotation_result', 'start_offset', 'end_offset']),
            models.Index(fields=['entity_type']),
            models.Index(fields=['overlap_group_id']),
            models.Index(fields=['status', 'source']),
            models.Index(fields=['start_offset', 'end_offset']),  # For overlap queries
        ]

    def __str__(self):
        return f"{self.entity_type}: '{self.text[:50]}...' ({self.start_offset}-{self.end_offset})"

    def get_span_length(self):
        """Get the length of this entity's text span"""
        return self.end_offset - self.start_offset

    def overlaps_with(self, other_entity):
        """Check if this entity overlaps with another entity"""
        return not (self.end_offset <= other_entity.start_offset or
                   self.start_offset >= other_entity.end_offset)

    def contains(self, other_entity):
        """Check if this entity completely contains another entity"""
        return (self.start_offset <= other_entity.start_offset and
                self.end_offset >= other_entity.end_offset)

    def is_contained_by(self, other_entity):
        """Check if this entity is completely contained by another"""
        return other_entity.contains(self)

    def calculate_overlap_ratio(self, other_entity):
        """Calculate the overlap ratio between this and another entity"""
        if not self.overlaps_with(other_entity):
            return 0.0

        overlap_start = max(self.start_offset, other_entity.start_offset)
        overlap_end = min(self.end_offset, other_entity.end_offset)
        overlap_length = overlap_end - overlap_start

        # Calculate overlap as ratio of the smaller entity
        min_length = min(self.get_span_length(), other_entity.get_span_length())
        return overlap_length / min_length if min_length > 0 else 0.0

    def calculate_iou_score(self, other_entity):
        """Calculate Intersection over Union (IoU) score"""
        if not self.overlaps_with(other_entity):
            return 0.0

        overlap_start = max(self.start_offset, other_entity.start_offset)
        overlap_end = min(self.end_offset, other_entity.end_offset)
        overlap_length = overlap_end - overlap_start

        union_start = min(self.start_offset, other_entity.start_offset)
        union_end = max(self.end_offset, other_entity.end_offset)
        union_length = union_end - union_start

        return overlap_length / union_length if union_length > 0 else 0.0

    def calculate_similarity_score(self, other_entity, use_type_matching=True):
        """Calculate comprehensive similarity score with another entity"""
        # Base IoU score
        iou_score = self.calculate_iou_score(other_entity)

        # Type matching bonus
        type_bonus = 0.0
        if use_type_matching and self.entity_type == other_entity.entity_type:
            type_bonus = 0.2

        # Text similarity bonus (for partial text matches)
        text_similarity = self.calculate_text_similarity(other_entity)
        text_bonus = text_similarity * 0.1

        return min(1.0, iou_score + type_bonus + text_bonus)

    def calculate_text_similarity(self, other_entity):
        """Calculate text-level similarity using token overlap"""
        tokens1 = set(self.text.lower().split())
        tokens2 = set(other_entity.text.lower().split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)


class OverlappingAnnotationResult(models.Model):
    """Annotation result supporting overlapping entities with advanced performance tracking"""

    ANNOTATION_TYPES = [
        ('manual_overlapping', 'Manual Overlapping Annotation'),
        ('llm_overlapping', 'LLM Overlapping Annotation'),
        ('hybrid_overlapping', 'Hybrid Overlapping Annotation'),
        ('gold_standard', 'Gold Standard Reference'),
    ]

    # Basic annotation info
    document = models.ForeignKey('TextDocument', on_delete=models.CASCADE,
                               related_name='overlapping_annotation_results')
    annotator = models.ForeignKey(User, on_delete=models.CASCADE,
                                related_name='overlapping_annotation_results')

    annotation_type = models.CharField(max_length=25, choices=ANNOTATION_TYPES)
    task_type = models.CharField(max_length=10, default='overlapping_ner',
                               help_text="Type of annotation task")

    # LLM information
    model_used = models.CharField(max_length=50, blank=True, help_text="LLM model used")
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")

    # Overlapping entity statistics
    total_entities = models.PositiveIntegerField(default=0)
    overlapping_entity_count = models.PositiveIntegerField(default=0,
                                                         help_text="Number of entities involved in overlaps")
    overlap_groups_count = models.PositiveIntegerField(default=0,
                                                     help_text="Number of distinct overlap groups")
    max_overlap_depth = models.PositiveIntegerField(default=0,
                                                  help_text="Maximum number of entities overlapping at any position")

    # Performance metrics (populated when compared against gold standard)
    exact_match_score = models.FloatField(null=True, blank=True, help_text="Exact match F1 score")
    partial_match_score = models.FloatField(null=True, blank=True, help_text="Partial match F1 score")
    overlap_aware_score = models.FloatField(null=True, blank=True, help_text="Overlap-aware F1 score")

    # Detailed performance breakdown
    performance_breakdown = models.JSONField(default=dict,
                                           help_text="Detailed performance metrics by entity type and overlap level")

    # Status and quality
    review_status = models.CharField(max_length=20, default='pending',
                                   choices=[('pending', 'Pending Review'),
                                          ('in_progress', 'In Progress'),
                                          ('completed', 'Completed'),
                                          ('validated', 'Validated')])

    confidence_score = models.FloatField(null=True, blank=True, help_text="Overall confidence score")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['document', 'annotator']),
            models.Index(fields=['annotation_type']),
            models.Index(fields=['task_type']),
            models.Index(fields=['review_status']),
        ]

    def __str__(self):
        return f"{self.get_annotation_type_display()} by {self.annotator.username} for {self.document.title[:50]}"

    def calculate_overlap_statistics(self):
        """Calculate and update overlap statistics"""
        entities = list(self.overlapping_entities.all())

        if not entities:
            self.total_entities = 0
            self.overlapping_entity_count = 0
            self.overlap_groups_count = 0
            self.max_overlap_depth = 0
            return

        self.total_entities = len(entities)

        # Find all overlapping entities and group them
        overlap_groups = self.identify_overlap_groups(entities)
        self.overlap_groups_count = len(overlap_groups)

        # Count entities involved in overlaps
        overlapping_entities = set()
        for group in overlap_groups:
            if len(group) > 1:  # Only count groups with actual overlaps
                overlapping_entities.update(entity.id for entity in group)

        self.overlapping_entity_count = len(overlapping_entities)

        # Calculate maximum overlap depth
        self.max_overlap_depth = self.calculate_max_overlap_depth(entities)

        self.save()

    def identify_overlap_groups(self, entities):
        """Identify groups of overlapping entities"""
        # Use Union-Find to group overlapping entities
        entity_to_group = {}
        groups = []

        for i, entity1 in enumerate(entities):
            # Check if entity1 already belongs to a group
            group1 = entity_to_group.get(entity1.id)

            for j, entity2 in enumerate(entities[i+1:], i+1):
                if entity1.overlaps_with(entity2):
                    group2 = entity_to_group.get(entity2.id)

                    if group1 is None and group2 is None:
                        # Create new group
                        new_group = [entity1, entity2]
                        groups.append(new_group)
                        entity_to_group[entity1.id] = new_group
                        entity_to_group[entity2.id] = new_group
                        group1 = new_group
                    elif group1 is None:
                        # Add entity1 to entity2's group
                        group2.append(entity1)
                        entity_to_group[entity1.id] = group2
                        group1 = group2
                    elif group2 is None:
                        # Add entity2 to entity1's group
                        group1.append(entity2)
                        entity_to_group[entity2.id] = group1
                    elif group1 != group2:
                        # Merge two different groups
                        group1.extend(group2)
                        for entity in group2:
                            entity_to_group[entity.id] = group1
                        groups.remove(group2)

            # Handle isolated entities
            if entity1.id not in entity_to_group:
                isolated_group = [entity1]
                groups.append(isolated_group)
                entity_to_group[entity1.id] = isolated_group

        return groups

    def calculate_max_overlap_depth(self, entities):
        """Calculate the maximum number of entities overlapping at any character position"""
        if not entities:
            return 0

        # Create events for start and end of each entity
        events = []
        for entity in entities:
            events.append((entity.start_offset, 'start', entity))
            events.append((entity.end_offset, 'end', entity))

        # Sort events by position
        events.sort(key=lambda x: (x[0], x[1] == 'end'))  # Process 'end' events before 'start' at same position

        max_depth = 0
        current_depth = 0

        for position, event_type, entity in events:
            if event_type == 'start':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            else:  # event_type == 'end'
                current_depth -= 1

        return max_depth

    def compare_with_gold_standard(self, gold_annotation, matching_thresholds=None):
        """Compare this annotation with a gold standard annotation"""
        if matching_thresholds is None:
            matching_thresholds = {
                'exact': 1.0,
                'partial': 0.5,
                'overlap_aware': 0.3
            }

        predicted_entities = list(self.overlapping_entities.all())
        gold_entities = list(gold_annotation.overlapping_entities.all())

        # Calculate different types of matching scores
        exact_metrics = self.calculate_matching_metrics(
            predicted_entities, gold_entities, matching_thresholds['exact'], match_type='exact'
        )

        partial_metrics = self.calculate_matching_metrics(
            predicted_entities, gold_entities, matching_thresholds['partial'], match_type='partial'
        )

        overlap_aware_metrics = self.calculate_overlap_aware_metrics(
            predicted_entities, gold_entities, matching_thresholds['overlap_aware']
        )

        # Store results
        self.exact_match_score = exact_metrics['f1']
        self.partial_match_score = partial_metrics['f1']
        self.overlap_aware_score = overlap_aware_metrics['f1']

        # Store detailed breakdown
        self.performance_breakdown = {
            'exact_matching': exact_metrics,
            'partial_matching': partial_metrics,
            'overlap_aware': overlap_aware_metrics,
            'entity_type_breakdown': self.calculate_entity_type_breakdown(predicted_entities, gold_entities),
            'overlap_level_breakdown': self.calculate_overlap_level_breakdown(predicted_entities, gold_entities)
        }

        self.save()

        return self.performance_breakdown

    def calculate_matching_metrics(self, predicted_entities, gold_entities, threshold, match_type='partial'):
        """Calculate precision, recall, and F1 for entity matching"""
        if not predicted_entities and not gold_entities:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'matches': 0}

        if not predicted_entities:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matches': 0}

        if not gold_entities:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matches': 0}

        # Find best matches for each predicted entity
        matched_gold = set()
        total_matches = 0

        for pred_entity in predicted_entities:
            best_score = 0.0
            best_match = None

            for idx, gold_entity in enumerate(gold_entities):
                if idx in matched_gold:
                    continue

                if match_type == 'exact':
                    score = self.calculate_exact_match_score(pred_entity, gold_entity)
                else:  # partial
                    score = pred_entity.calculate_similarity_score(gold_entity)

                if score > best_score:
                    best_score = score
                    best_match = idx

            if best_score >= threshold and best_match is not None:
                total_matches += 1
                matched_gold.add(best_match)

        # Calculate metrics
        precision = total_matches / len(predicted_entities)
        recall = total_matches / len(gold_entities)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matches': total_matches,
            'predicted_count': len(predicted_entities),
            'gold_count': len(gold_entities)
        }

    def calculate_exact_match_score(self, pred_entity, gold_entity):
        """Calculate exact match score (1.0 if perfect match, 0.0 otherwise)"""
        return 1.0 if (pred_entity.start_offset == gold_entity.start_offset and
                      pred_entity.end_offset == gold_entity.end_offset and
                      pred_entity.entity_type == gold_entity.entity_type) else 0.0

    def calculate_overlap_aware_metrics(self, predicted_entities, gold_entities, threshold):
        """Calculate overlap-aware metrics that account for overlapping entity structures"""
        # Group entities by overlap groups
        pred_groups = self.identify_overlap_groups(predicted_entities)
        gold_groups = self.identify_overlap_groups(gold_entities)

        # Match overlap groups
        matched_groups = 0
        total_group_similarity = 0.0

        matched_gold_groups = set()

        for pred_group in pred_groups:
            best_score = 0.0
            best_match_idx = -1

            for idx, gold_group in enumerate(gold_groups):
                if idx in matched_gold_groups:
                    continue

                group_score = self.calculate_group_similarity(pred_group, gold_group)

                if group_score > best_score:
                    best_score = group_score
                    best_match_idx = idx

            if best_score >= threshold and best_match_idx >= 0:
                matched_groups += 1
                matched_gold_groups.add(best_match_idx)
                total_group_similarity += best_score

        # Calculate group-level metrics
        precision = matched_groups / len(pred_groups) if pred_groups else 0.0
        recall = matched_groups / len(gold_groups) if gold_groups else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'matched_groups': matched_groups,
            'predicted_groups': len(pred_groups),
            'gold_groups': len(gold_groups),
            'avg_group_similarity': total_group_similarity / matched_groups if matched_groups > 0 else 0.0
        }

    def calculate_group_similarity(self, group1, group2):
        """Calculate similarity between two groups of overlapping entities"""
        if not group1 or not group2:
            return 0.0

        # Create bipartite matching between entities in the two groups
        total_similarity = 0.0
        matched_entities = 0

        matched_group2 = set()

        for entity1 in group1:
            best_score = 0.0
            best_match_idx = -1

            for idx, entity2 in enumerate(group2):
                if idx in matched_group2:
                    continue

                score = entity1.calculate_similarity_score(entity2)
                if score > best_score:
                    best_score = score
                    best_match_idx = idx

            if best_score > 0.3:  # Minimum threshold for considering a match
                total_similarity += best_score
                matched_entities += 1
                if best_match_idx >= 0:
                    matched_group2.add(best_match_idx)

        # Normalize by the size of the larger group
        max_group_size = max(len(group1), len(group2))
        return total_similarity / max_group_size if max_group_size > 0 else 0.0

    def calculate_entity_type_breakdown(self, predicted_entities, gold_entities):
        """Calculate performance breakdown by entity type"""
        breakdown = {}

        # Group entities by type
        pred_by_type = defaultdict(list)
        gold_by_type = defaultdict(list)

        for entity in predicted_entities:
            pred_by_type[entity.entity_type].append(entity)

        for entity in gold_entities:
            gold_by_type[entity.entity_type].append(entity)

        # Calculate metrics for each entity type
        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())

        for entity_type in all_types:
            pred_entities = pred_by_type[entity_type]
            gold_entities = gold_by_type[entity_type]

            metrics = self.calculate_matching_metrics(pred_entities, gold_entities, 0.5)
            breakdown[entity_type] = metrics

        return breakdown

    def calculate_overlap_level_breakdown(self, predicted_entities, gold_entities):
        """Calculate performance breakdown by overlap complexity level"""
        breakdown = {}

        # Categorize entities by overlap level
        pred_overlap_levels = self.categorize_by_overlap_level(predicted_entities)
        gold_overlap_levels = self.categorize_by_overlap_level(gold_entities)

        all_levels = set(pred_overlap_levels.keys()) | set(gold_overlap_levels.keys())

        for level in all_levels:
            pred_entities = pred_overlap_levels.get(level, [])
            gold_entities = gold_overlap_levels.get(level, [])

            metrics = self.calculate_matching_metrics(pred_entities, gold_entities, 0.5)
            breakdown[f'overlap_level_{level}'] = metrics

        return breakdown

    def categorize_by_overlap_level(self, entities):
        """Categorize entities by their overlap complexity level"""
        overlap_levels = defaultdict(list)

        for entity in entities:
            # Count how many other entities this entity overlaps with
            overlap_count = 0
            for other_entity in entities:
                if entity.id != other_entity.id and entity.overlaps_with(other_entity):
                    overlap_count += 1

            # Categorize by overlap count
            if overlap_count == 0:
                level = 'isolated'
            elif overlap_count <= 2:
                level = 'simple'
            elif overlap_count <= 5:
                level = 'moderate'
            else:
                level = 'complex'

            overlap_levels[level].append(entity)

        return overlap_levels

    def save(self, *args, **kwargs):
        """Auto-calculate statistics before saving"""
        super().save(*args, **kwargs)
        self.calculate_overlap_statistics()


class OverlappingPerformanceStatistics(models.Model):
    """Comprehensive statistics for overlapping entity annotation performance"""

    # Identifiers
    annotator = models.ForeignKey(User, on_delete=models.CASCADE,
                                related_name='overlapping_performance_stats',
                                null=True, blank=True)
    period_start = models.DateTimeField(null=True, blank=True)
    period_end = models.DateTimeField(null=True, blank=True)

    # Basic counts
    total_annotations = models.PositiveIntegerField(default=0)
    total_entities_predicted = models.PositiveIntegerField(default=0)
    total_entities_gold = models.PositiveIntegerField(default=0)

    # Overlap statistics
    avg_entities_per_annotation = models.FloatField(default=0.0)
    avg_overlap_groups_per_annotation = models.FloatField(default=0.0)
    avg_max_overlap_depth = models.FloatField(default=0.0)
    overlap_complexity_distribution = models.JSONField(default=dict,
                                                      help_text="Distribution of annotations by overlap complexity")

    # Performance metrics
    exact_match_precision = models.FloatField(default=0.0)
    exact_match_recall = models.FloatField(default=0.0)
    exact_match_f1 = models.FloatField(default=0.0)

    partial_match_precision = models.FloatField(default=0.0)
    partial_match_recall = models.FloatField(default=0.0)
    partial_match_f1 = models.FloatField(default=0.0)

    overlap_aware_precision = models.FloatField(default=0.0)
    overlap_aware_recall = models.FloatField(default=0.0)
    overlap_aware_f1 = models.FloatField(default=0.0)

    # Entity type performance
    entity_type_performance = models.JSONField(default=dict,
                                             help_text="Performance breakdown by entity type")

    # LLM interaction statistics
    avg_llm_confidence = models.FloatField(default=0.0)
    llm_decision_breakdown = models.JSONField(default=dict,
                                            help_text="Breakdown of accept/modify/reject decisions")

    # Timing information
    avg_annotation_time_minutes = models.FloatField(default=0.0)
    avg_entities_per_minute = models.FloatField(default=0.0)

    # Metadata
    calculated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-calculated_at']
        indexes = [
            models.Index(fields=['annotator']),
            models.Index(fields=['period_start', 'period_end']),
        ]

    def __str__(self):
        annotator_name = self.annotator.username if self.annotator else "All Annotators"
        return f"Overlapping Performance: {annotator_name} ({self.period_start} to {self.period_end})"

    @classmethod
    def calculate_and_store_statistics(cls, annotator=None, period_start=None, period_end=None):
        """Calculate and store comprehensive overlapping entity statistics"""

        # Build queryset for annotations
        queryset = OverlappingAnnotationResult.objects.all()

        if annotator:
            queryset = queryset.filter(annotator=annotator)
        if period_start:
            queryset = queryset.filter(created_at__gte=period_start)
        if period_end:
            queryset = queryset.filter(created_at__lte=period_end)

        # Calculate basic statistics
        annotations = list(queryset.prefetch_related('overlapping_entities'))

        if not annotations:
            return cls.objects.create(
                annotator=annotator,
                period_start=period_start,
                period_end=period_end
            )

        # Basic counts
        total_annotations = len(annotations)
        total_entities_predicted = sum(a.total_entities for a in annotations)

        # Overlap statistics
        total_overlap_groups = sum(a.overlap_groups_count for a in annotations)
        total_max_overlap_depth = sum(a.max_overlap_depth for a in annotations)

        # Performance metrics (only for annotations with performance data)
        annotations_with_performance = [a for a in annotations if a.performance_breakdown]

        exact_f1_scores = []
        partial_f1_scores = []
        overlap_aware_f1_scores = []

        for annotation in annotations_with_performance:
            perf = annotation.performance_breakdown
            if 'exact_matching' in perf:
                exact_f1_scores.append(perf['exact_matching']['f1'])
            if 'partial_matching' in perf:
                partial_f1_scores.append(perf['partial_matching']['f1'])
            if 'overlap_aware' in perf:
                overlap_aware_f1_scores.append(perf['overlap_aware']['f1'])

        # Create statistics record
        stats = cls.objects.create(
            annotator=annotator,
            period_start=period_start,
            period_end=period_end,
            total_annotations=total_annotations,
            total_entities_predicted=total_entities_predicted,
            avg_entities_per_annotation=total_entities_predicted / total_annotations,
            avg_overlap_groups_per_annotation=total_overlap_groups / total_annotations,
            avg_max_overlap_depth=total_max_overlap_depth / total_annotations,
            exact_match_f1=sum(exact_f1_scores) / len(exact_f1_scores) if exact_f1_scores else 0.0,
            partial_match_f1=sum(partial_f1_scores) / len(partial_f1_scores) if partial_f1_scores else 0.0,
            overlap_aware_f1=sum(overlap_aware_f1_scores) / len(overlap_aware_f1_scores) if overlap_aware_f1_scores else 0.0,
        )

        return stats