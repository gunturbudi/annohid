"""
Multi-LLM NER Runner
Executes NER with multiple LLM providers and stores results for comparison
"""

import logging
from typing import List, Dict, Optional, Set
from collections import defaultdict, Counter
from django.db import transaction
from django.utils import timezone

from .models import TextDocument, AnnotationResult, AnnotationTask
from .llm_providers import get_ner_provider, get_default_models, get_all_available_providers
from .llm_providers.base import Entity, NERResult
from authentication.models import User


logger = logging.getLogger(__name__)


class MultiLLMNERRunner:
    """
    Run NER with multiple LLM models and store results separately
    Enables model comparison and consensus analysis
    """

    def __init__(self, model_ids: Optional[List[str]] = None):
        """
        Initialize multi-LLM runner

        Args:
            model_ids: List of model IDs to use. If None, uses default models.
        """
        if model_ids is None:
            model_ids = get_default_models()

        self.model_ids = model_ids
        self.providers = {}

        # Initialize providers
        for model_id in model_ids:
            provider = get_ner_provider(model_id)
            if provider and provider.is_available():
                self.providers[model_id] = provider
            else:
                logger.warning(f"Provider for {model_id} not available")

    def run_all_models(
        self,
        document: TextDocument,
        annotator: User,
        enable_overlapping: bool = True,
        confidence_threshold: float = 0.3
    ) -> Dict[str, AnnotationResult]:
        """
        Run NER with all configured models and store results

        Args:
            document: The document to annotate
            annotator: The user running the annotation
            enable_overlapping: Whether to detect overlapping entities
            confidence_threshold: Minimum confidence threshold

        Returns:
            Dict mapping model_id to AnnotationResult
        """
        results = {}

        for model_id, provider in self.providers.items():
            try:
                logger.info(f"Running NER with {model_id} on document {document.id}")

                # Run annotation
                ner_result = provider.annotate_text(
                    text=document.content,
                    enable_overlapping=enable_overlapping,
                    confidence_threshold=confidence_threshold
                )

                if not ner_result.success:
                    logger.error(f"NER failed for {model_id}: {ner_result.error_message}")
                    continue

                # Convert entities to JSON format
                entities_json = [entity.to_dict() for entity in ner_result.entities]

                # Store as AnnotationResult
                annotation_result = AnnotationResult.objects.create(
                    document=document,
                    annotator=annotator,
                    task_type='ner',
                    annotation_type='llm_comparison',  # Special type for multi-LLM
                    model_used=model_id,
                    entities_json=entities_json,
                    total_entities=ner_result.total_entities,
                    processing_time=ner_result.processing_time,
                    confidence_score=ner_result.avg_confidence,
                    review_status='pending'
                )

                results[model_id] = annotation_result
                logger.info(f"Stored {ner_result.total_entities} entities from {model_id}")

            except Exception as e:
                logger.error(f"Error running {model_id}: {e}", exc_info=True)

        return results

    def get_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        return list(self.providers.keys())

    def get_model_count(self) -> int:
        """Get number of available models"""
        return len(self.providers)


class EntityMerger:
    """
    Merge entities from multiple LLMs with consensus tracking
    Uses overlapping entity detection
    """

    @staticmethod
    def entities_overlap(e1: Dict, e2: Dict) -> bool:
        """Check if two entities overlap"""
        return not (e1['end_offset'] <= e2['start_offset'] or
                   e1['start_offset'] >= e2['end_offset'])

    @staticmethod
    def entities_exact_match(e1: Dict, e2: Dict) -> bool:
        """Check if two entities are exactly the same"""
        return (e1['start_offset'] == e2['start_offset'] and
                e1['end_offset'] == e2['end_offset'] and
                e1['label'] == e2['label'])

    @classmethod
    def merge_results(
        cls,
        annotation_results: List[AnnotationResult],
        merge_strategy: str = 'overlapping'
    ) -> List[Dict]:
        """
        Merge entities from multiple AnnotationResults

        Args:
            annotation_results: List of AnnotationResult objects to merge
            merge_strategy: 'exact' (only exact matches) or 'overlapping' (group overlaps)

        Returns:
            List of merged entities with consensus information
        """
        if merge_strategy == 'exact':
            return cls._merge_exact_match(annotation_results)
        else:
            return cls._merge_overlapping(annotation_results)

    @classmethod
    def _merge_exact_match(cls, annotation_results: List[AnnotationResult]) -> List[Dict]:
        """Merge entities that match exactly"""
        # Group by exact span and label
        entity_groups = defaultdict(list)

        for result in annotation_results:
            model_name = result.model_used

            for entity in result.entities_json:
                key = (
                    entity['start_offset'],
                    entity['end_offset'],
                    entity['label']
                )

                entity_groups[key].append({
                    'model': model_name,
                    'confidence': entity.get('confidence', 0.5),
                    'reasoning': entity.get('reasoning', ''),
                    'entity': entity
                })

        # Convert to merged entities
        merged = []
        total_models = len(annotation_results)

        for (start, end, label), predictions in entity_groups.items():
            # Get text from first prediction
            text = predictions[0]['entity']['text']

            merged_entity = {
                'text': text,
                'start_offset': start,
                'end_offset': end,
                'label': label,
                'source': 'multi_llm',
                'model_predictions': predictions,
                'consensus_score': len(predictions),
                'consensus_percentage': (len(predictions) / total_models) * 100,
                'avg_confidence': sum(p['confidence'] for p in predictions) / len(predictions),
                'unanimous': len(predictions) == total_models,
                'status': cls._determine_status(len(predictions), total_models)
            }

            merged.append(merged_entity)

        # Sort by consensus (highest first), then by position
        merged.sort(key=lambda x: (-x['consensus_score'], x['start_offset']))

        return merged

    @classmethod
    def _merge_overlapping(cls, annotation_results: List[AnnotationResult]) -> List[Dict]:
        """Merge overlapping entities (more sophisticated)"""
        all_entities = []

        # Collect all entities with model information
        for result in annotation_results:
            model_name = result.model_used

            for entity in result.entities_json:
                entity_with_model = entity.copy()
                entity_with_model['_model'] = model_name
                all_entities.append(entity_with_model)

        # Group overlapping entities
        groups = []
        processed = set()

        for i, entity in enumerate(all_entities):
            if i in processed:
                continue

            group = [entity]
            processed.add(i)

            # Find all overlapping entities
            for j, other in enumerate(all_entities):
                if j not in processed and cls.entities_overlap(entity, other):
                    group.append(other)
                    processed.add(j)

            groups.append(group)

        # Create merged entities from groups
        merged = []
        total_models = len(annotation_results)

        for group in groups:
            # Find most common span
            spans = Counter((e['start_offset'], e['end_offset']) for e in group)
            most_common_span, span_count = spans.most_common(1)[0]

            # Find most common label
            labels = Counter(e['label'] for e in group)
            most_common_label, label_count = labels.most_common(1)[0]

            # Check for label disagreement
            has_label_disagreement = len(labels) > 1

            # Get text (from most common span)
            text = next(e['text'] for e in group
                       if e['start_offset'] == most_common_span[0]
                       and e['end_offset'] == most_common_span[1])

            # Build model predictions
            model_predictions = []
            for entity in group:
                model_predictions.append({
                    'model': entity['_model'],
                    'confidence': entity.get('confidence', 0.5),
                    'predicted_span': (entity['start_offset'], entity['end_offset']),
                    'predicted_label': entity['label'],
                    'reasoning': entity.get('reasoning', ''),
                    'exact_match': (entity['start_offset'] == most_common_span[0] and
                                  entity['end_offset'] == most_common_span[1] and
                                  entity['label'] == most_common_label)
                })

            merged_entity = {
                'text': text,
                'start_offset': most_common_span[0],
                'end_offset': most_common_span[1],
                'label': most_common_label,
                'source': 'multi_llm',
                'model_predictions': model_predictions,
                'consensus_score': len(group),
                'consensus_percentage': (len(group) / total_models) * 100,
                'exact_consensus_score': span_count if not has_label_disagreement else 0,
                'avg_confidence': sum(p['confidence'] for p in model_predictions) / len(model_predictions),
                'has_span_disagreement': len(spans) > 1,
                'has_label_disagreement': has_label_disagreement,
                'unanimous': len(group) == total_models and not has_label_disagreement,
                'status': cls._determine_status(len(group), total_models, has_label_disagreement),
                'all_labels': list(labels.keys()),
                'all_spans': list(spans.keys())
            }

            merged.append(merged_entity)

        # Sort by consensus (highest first), then by position
        merged.sort(key=lambda x: (-x['consensus_score'], x['start_offset']))

        return merged

    @staticmethod
    def _determine_status(consensus_count: int, total_models: int, has_disagreement: bool = False) -> str:
        """Determine entity status based on consensus"""
        percentage = (consensus_count / total_models) * 100

        if has_disagreement:
            return 'needs_review'
        elif percentage == 100:
            return 'high_consensus'
        elif percentage >= 75:
            return 'moderate_consensus'
        elif percentage >= 50:
            return 'low_consensus'
        else:
            return 'needs_review'


class AgreementCalculator:
    """Calculate inter-model agreement metrics"""

    @staticmethod
    def calculate_agreement(annotation_results: List[AnnotationResult]) -> Dict:
        """
        Calculate comprehensive agreement metrics

        Args:
            annotation_results: List of AnnotationResult objects

        Returns:
            Dict with agreement statistics
        """
        if len(annotation_results) < 2:
            return {'error': 'Need at least 2 models for agreement calculation'}

        # Merge with exact match strategy
        merged_exact = EntityMerger.merge_results(annotation_results, merge_strategy='exact')

        # Count consensus levels
        total_models = len(annotation_results)
        consensus_levels = Counter(e['consensus_score'] for e in merged_exact)

        unanimous = sum(1 for e in merged_exact if e['unanimous'])
        high_consensus = sum(1 for e in merged_exact if e['consensus_score'] >= max(2, total_models * 0.75))
        moderate_consensus = sum(1 for e in merged_exact if total_models * 0.5 <= e['consensus_score'] < total_models * 0.75)
        low_consensus = sum(1 for e in merged_exact if e['consensus_score'] < total_models * 0.5)

        # Calculate per-model statistics
        model_stats = {}
        for result in annotation_results:
            model_stats[result.model_used] = {
                'total_entities': result.total_entities,
                'avg_confidence': result.confidence_score,
                'processing_time': result.processing_time,
            }

        # Agreement rate (what percentage of entities have unanimous agreement)
        agreement_rate = (unanimous / len(merged_exact) * 100) if merged_exact else 0

        return {
            'total_unique_entities': len(merged_exact),
            'unanimous_consensus': unanimous,
            'high_consensus': high_consensus,
            'moderate_consensus': moderate_consensus,
            'low_consensus': low_consensus,
            'agreement_rate': agreement_rate,
            'consensus_distribution': dict(consensus_levels),
            'model_statistics': model_stats,
            'total_models': total_models,
        }
