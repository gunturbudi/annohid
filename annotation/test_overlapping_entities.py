"""
Comprehensive Testing Suite for Overlapping Medical Entity Annotation System

This test suite covers all aspects of the overlapping entity annotation system:
1. Model validation and data integrity
2. Partial matching algorithms
3. LLM integration
4. Performance analytics
5. User interface interactions
6. End-to-end workflows
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import time
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta

from .overlapping_models import (
    OverlappingEntitySchema, OverlappingEntity, OverlappingAnnotationResult,
    OverlappingPerformanceStatistics
)
from .partial_matching_algorithms import (
    AdvancedPartialMatcher, EntityMatch, MatchingResult, LLMPerformanceAnalyzer
)
from .overlapping_llm_annotator import (
    OverlappingLLMAnnotator, OverlappingEntity as LLMOverlappingEntity,
    OverlappingAnnotationResult as LLMAnnotationResult
)
from .models import TextDocument, AnnotationTask

User = get_user_model()


class OverlappingEntityModelTests(TestCase):
    """Test overlapping entity models and their relationships"""

    def setUp(self):
        self.admin_user = User.objects.create_user(
            username='admin',
            password='testpass',
            role='admin'
        )
        self.annotator_user = User.objects.create_user(
            username='annotator',
            password='testpass',
            role='annotator'
        )
        self.document = TextDocument.objects.create(
            content="Patient has severe chest pain with shortness of breath.",
            uploaded_by=self.admin_user
        )

    def test_overlapping_entity_schema_creation(self):
        """Test creation of overlapping entity schemas"""
        schema = OverlappingEntitySchema.objects.create(
            entity_type='COMPLEX_SYMPTOM',
            is_complex=True,
            max_nesting_depth=3,
            required_children=['SYMPTOM', 'SEVERITY'],
            optional_children=['BODY_PART', 'DURATION'],
            created_by=self.admin_user
        )

        self.assertEqual(schema.entity_type, 'COMPLEX_SYMPTOM')
        self.assertTrue(schema.is_complex)
        self.assertEqual(schema.max_nesting_depth, 3)
        self.assertIn('SYMPTOM', schema.required_children)
        self.assertIn('BODY_PART', schema.optional_children)

    def test_entity_schema_validation(self):
        """Test entity schema validation methods"""
        schema = OverlappingEntitySchema.objects.create(
            entity_type='MEDICATION_INSTRUCTION',
            is_complex=True,
            required_children=['MEDICATION', 'DOSAGE'],
            optional_children=['FREQUENCY', 'DURATION'],
            created_by=self.admin_user
        )

        # Test valid entity structure
        valid_entity = {
            'children': [
                {'label': 'MEDICATION'},
                {'label': 'DOSAGE'},
                {'label': 'FREQUENCY'}
            ]
        }
        errors = schema.validate_nested_entity(valid_entity)
        self.assertEqual(len(errors), 0)

        # Test invalid entity structure (missing required child)
        invalid_entity = {
            'children': [
                {'label': 'MEDICATION'}
                # Missing DOSAGE
            ]
        }
        errors = schema.validate_nested_entity(invalid_entity)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any('Missing required child entity: DOSAGE' in error for error in errors))

    def test_overlapping_entity_creation_and_relationships(self):
        """Test creation of overlapping entities with proper relationships"""
        annotation_result = OverlappingAnnotationResult.objects.create(
            document=self.document,
            annotator=self.annotator_user,
            annotation_type='manual_overlapping'
        )

        # Create overlapping entities
        entity1 = OverlappingEntity.objects.create(
            text="severe",
            start_offset=12,
            end_offset=18,
            entity_type="SEVERITY",
            annotation_result=annotation_result,
            confidence_score=0.9
        )

        entity2 = OverlappingEntity.objects.create(
            text="chest pain",
            start_offset=19,
            end_offset=29,
            entity_type="SYMPTOM",
            annotation_result=annotation_result,
            confidence_score=0.95
        )

        entity3 = OverlappingEntity.objects.create(
            text="severe chest pain",
            start_offset=12,
            end_offset=29,
            entity_type="SYMPTOM",
            annotation_result=annotation_result,
            confidence_score=0.85
        )

        # Test overlapping relationships
        self.assertTrue(entity1.overlaps_with(entity3))
        self.assertTrue(entity2.overlaps_with(entity3))
        self.assertFalse(entity1.overlaps_with(entity2))

        # Test containment relationships
        self.assertTrue(entity3.contains(entity1))
        self.assertTrue(entity3.contains(entity2))
        self.assertFalse(entity1.contains(entity3))

    def test_entity_overlap_calculations(self):
        """Test overlap ratio and IoU calculations"""
        annotation_result = OverlappingAnnotationResult.objects.create(
            document=self.document,
            annotator=self.annotator_user,
            annotation_type='manual_overlapping'
        )

        entity1 = OverlappingEntity.objects.create(
            text="chest pain",
            start_offset=19,
            end_offset=29,
            entity_type="SYMPTOM",
            annotation_result=annotation_result
        )

        entity2 = OverlappingEntity.objects.create(
            text="severe chest pain",
            start_offset=12,
            end_offset=29,
            entity_type="SYMPTOM",
            annotation_result=annotation_result
        )

        # Test overlap ratio
        overlap_ratio = entity1.calculate_overlap_ratio(entity2)
        self.assertGreater(overlap_ratio, 0.5)  # Should have significant overlap

        # Test IoU score
        iou_score = entity1.calculate_iou_score(entity2)
        self.assertGreater(iou_score, 0.3)  # Should have reasonable IoU

        # Test similarity score
        similarity = entity1.calculate_similarity_score(entity2)
        self.assertGreater(similarity, 0.7)  # Should be quite similar

    def test_annotation_result_overlap_statistics(self):
        """Test calculation of overlap statistics"""
        annotation_result = OverlappingAnnotationResult.objects.create(
            document=self.document,
            annotator=self.annotator_user,
            annotation_type='manual_overlapping'
        )

        # Create multiple overlapping entities
        entities_data = [
            ("severe", 12, 18, "SEVERITY"),
            ("chest", 19, 24, "BODY_PART"),
            ("pain", 25, 29, "SYMPTOM"),
            ("chest pain", 19, 29, "SYMPTOM"),
            ("severe chest pain", 12, 29, "SYMPTOM")
        ]

        for text, start, end, entity_type in entities_data:
            OverlappingEntity.objects.create(
                text=text,
                start_offset=start,
                end_offset=end,
                entity_type=entity_type,
                annotation_result=annotation_result,
                confidence_score=0.8
            )

        # Trigger statistics calculation
        annotation_result.calculate_overlap_statistics()

        self.assertEqual(annotation_result.total_entities, 5)
        self.assertGreater(annotation_result.overlapping_entity_count, 0)
        self.assertGreater(annotation_result.overlap_groups_count, 0)
        self.assertGreater(annotation_result.max_overlap_depth, 1)


class PartialMatchingAlgorithmTests(TestCase):
    """Test the advanced partial matching algorithms"""

    def setUp(self):
        self.matcher = AdvancedPartialMatcher(
            exact_threshold=1.0,
            partial_threshold=0.5,
            relaxed_threshold=0.3
        )

    def create_mock_entity(self, text, start, end, entity_type, confidence=0.8):
        """Create a mock entity for testing"""
        entity = Mock()
        entity.text = text
        entity.start_offset = start
        entity.end_offset = end
        entity.entity_type = entity_type
        entity.confidence_score = confidence
        return entity

    def test_iou_calculation(self):
        """Test Intersection over Union calculation"""
        entity1 = self.create_mock_entity("chest pain", 10, 20, "SYMPTOM")
        entity2 = self.create_mock_entity("severe chest pain", 5, 20, "SYMPTOM")

        iou = self.matcher._calculate_iou(entity1, entity2)

        # Expected: intersection = 10, union = 15, IoU = 10/15 = 0.67
        self.assertAlmostEqual(iou, 0.67, places=2)

    def test_text_similarity_calculation(self):
        """Test text similarity using token-based Jaccard"""
        text1 = "chest pain"
        text2 = "severe chest pain"

        similarity = self.matcher._calculate_text_similarity(text1, text2)

        # Expected: intersection = {"chest", "pain"}, union = {"chest", "pain", "severe"}
        # Similarity = 2/3 = 0.67
        self.assertAlmostEqual(similarity, 0.67, places=2)

    def test_entity_similarity_calculation(self):
        """Test comprehensive entity similarity scoring"""
        entity1 = self.create_mock_entity("chest pain", 10, 20, "SYMPTOM", 0.9)
        entity2 = self.create_mock_entity("severe chest pain", 5, 20, "SYMPTOM", 0.8)

        match = self.matcher._calculate_entity_similarity(entity1, entity2)

        self.assertIsInstance(match, EntityMatch)
        self.assertEqual(match.predicted_entity, entity1)
        self.assertEqual(match.gold_entity, entity2)
        self.assertGreater(match.match_score, 0.5)
        self.assertTrue(match.type_match)
        self.assertGreater(match.iou_score, 0.0)

    def test_overlapping_entity_matching(self):
        """Test matching of overlapping entity sets"""
        predicted_entities = [
            self.create_mock_entity("severe", 0, 6, "SEVERITY"),
            self.create_mock_entity("chest pain", 7, 17, "SYMPTOM"),
            self.create_mock_entity("acute", 20, 25, "SEVERITY")
        ]

        gold_entities = [
            self.create_mock_entity("severe", 0, 6, "SEVERITY"),
            self.create_mock_entity("chest", 7, 12, "BODY_PART"),
            self.create_mock_entity("pain", 13, 17, "SYMPTOM"),
            self.create_mock_entity("severe chest pain", 0, 17, "SYMPTOM")
        ]

        result = self.matcher.compare_annotations(predicted_entities, gold_entities)

        self.assertIsInstance(result, MatchingResult)
        self.assertGreater(len(result.matches), 0)
        self.assertGreater(result.precision, 0.0)
        self.assertGreater(result.recall, 0.0)
        self.assertGreater(result.f1_score, 0.0)

    def test_overlap_group_identification(self):
        """Test identification of overlapping entity groups"""
        entities = [
            self.create_mock_entity("severe", 0, 6, "SEVERITY"),
            self.create_mock_entity("chest pain", 7, 17, "SYMPTOM"),
            self.create_mock_entity("severe chest pain", 0, 17, "SYMPTOM"),
            self.create_mock_entity("shortness", 25, 34, "SYMPTOM"),  # Isolated
        ]

        groups = self.matcher._identify_overlap_groups(entities)

        # Should have 2 groups: one with 3 overlapping entities, one isolated
        self.assertEqual(len(groups), 2)
        group_sizes = [len(group) for group in groups]
        self.assertIn(3, group_sizes)  # Group with overlapping entities
        self.assertIn(1, group_sizes)  # Isolated entity

    def test_performance_by_entity_type(self):
        """Test performance calculation breakdown by entity type"""
        predicted_entities = [
            self.create_mock_entity("chest pain", 0, 10, "SYMPTOM"),
            self.create_mock_entity("ibuprofen", 15, 24, "MEDICATION"),
            self.create_mock_entity("severe", 25, 31, "SEVERITY")
        ]

        gold_entities = [
            self.create_mock_entity("chest pain", 0, 10, "SYMPTOM"),
            self.create_mock_entity("ibuprofen", 15, 24, "MEDICATION"),
            self.create_mock_entity("mild", 25, 29, "SEVERITY")  # Different text, same type
        ]

        result = self.matcher.compare_annotations(predicted_entities, gold_entities)

        self.assertIn('SYMPTOM', result.performance_by_type)
        self.assertIn('MEDICATION', result.performance_by_type)
        self.assertIn('SEVERITY', result.performance_by_type)

        # SYMPTOM and MEDICATION should have perfect scores
        self.assertEqual(result.performance_by_type['SYMPTOM']['f1'], 1.0)
        self.assertEqual(result.performance_by_type['MEDICATION']['f1'], 1.0)


class LLMIntegrationTests(TestCase):
    """Test LLM integration for overlapping entity detection"""

    def setUp(self):
        self.annotator = OverlappingLLMAnnotator(model_id="test-model")
        # Mock the LLM model
        self.annotator.model = Mock()

    def test_overlapping_prompt_creation(self):
        """Test creation of overlapping entity detection prompt"""
        text = "Patient has severe chest pain."
        prompt = self.annotator._create_overlapping_prompt(text)

        self.assertIn("OVERLAPPING", prompt.upper())
        self.assertIn("severe chest pain", prompt)
        self.assertIn("JSON", prompt)
        self.assertIn("start_offset", prompt)
        self.assertIn("confidence", prompt)

    def test_entity_validation(self):
        """Test entity prediction validation"""
        original_text = "Patient has severe chest pain."

        # Valid entity
        valid_entity = {
            'text': 'severe',
            'start_offset': 12,
            'end_offset': 18,
            'entity_type': 'SEVERITY',
            'confidence': 0.9
        }

        validated = self.annotator._validate_entity(valid_entity, original_text)
        self.assertIsNotNone(validated)
        self.assertEqual(validated['text'], 'severe')
        self.assertEqual(validated['entity_type'], 'SEVERITY')

        # Invalid entity (out of bounds)
        invalid_entity = {
            'text': 'severe',
            'start_offset': 100,  # Out of bounds
            'end_offset': 106,
            'entity_type': 'SEVERITY',
            'confidence': 0.9
        }

        validated = self.annotator._validate_entity(invalid_entity, original_text)
        self.assertIsNone(validated)

    def test_overlap_information_update(self):
        """Test updating overlap information between entities"""
        entities = [
            LLMOverlappingEntity(
                text="severe",
                start_offset=0,
                end_offset=6,
                entity_type="SEVERITY",
                confidence=0.9
            ),
            LLMOverlappingEntity(
                text="chest pain",
                start_offset=7,
                end_offset=17,
                entity_type="SYMPTOM",
                confidence=0.9
            ),
            LLMOverlappingEntity(
                text="severe chest pain",
                start_offset=0,
                end_offset=17,
                entity_type="SYMPTOM",
                confidence=0.85
            )
        ]

        self.annotator._update_overlap_information(entities)

        # The compound entity should overlap with both component entities
        compound_entity = entities[2]
        self.assertIn("severe", compound_entity.overlaps_with)
        self.assertIn("chest pain", compound_entity.overlaps_with)

    def test_overlap_complexity_estimation(self):
        """Test estimation of text overlap complexity"""
        simple_text = "Patient has pain."
        complex_text = "Patient has severe chronic chest pain with acute shortness of breath after taking 500mg ibuprofen twice daily."

        simple_complexity = self.annotator.estimate_overlap_complexity(simple_text)
        complex_complexity = self.annotator.estimate_overlap_complexity(complex_text)

        self.assertEqual(simple_complexity['complexity'], 'low')
        self.assertIn(complex_complexity['complexity'], ['moderate', 'high'])
        self.assertGreater(complex_complexity['estimated_entities'], simple_complexity['estimated_entities'])

    @patch('google.generativeai.GenerativeModel')
    def test_annotation_with_mock_llm(self, mock_model_class):
        """Test annotation workflow with mocked LLM responses"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.text = '''
        [
            {
                "text": "severe",
                "start_offset": 12,
                "end_offset": 18,
                "entity_type": "SEVERITY",
                "confidence": 0.9,
                "reasoning": "Indicates intensity of symptom"
            },
            {
                "text": "chest pain",
                "start_offset": 19,
                "end_offset": 29,
                "entity_type": "SYMPTOM",
                "confidence": 0.95,
                "reasoning": "Primary symptom complaint"
            }
        ]
        '''

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        # Create annotator with mocked model
        annotator = OverlappingLLMAnnotator()
        annotator.model = mock_model

        text = "Patient has severe chest pain."
        result = annotator.annotate_document(text)

        self.assertIsInstance(result, LLMAnnotationResult)
        self.assertTrue(result.success)
        self.assertEqual(len(result.entities), 2)
        self.assertGreater(result.total_entities, 0)


class PerformanceAnalysisTests(TestCase):
    """Test the LLM performance analysis system"""

    def setUp(self):
        self.analyzer = LLMPerformanceAnalyzer()

    def create_mock_annotation_with_entities(self, entities_data):
        """Create mock annotation with entities"""
        annotation = Mock()
        annotation.overlapping_entities.all.return_value = [
            self.create_mock_entity(text, start, end, entity_type)
            for text, start, end, entity_type in entities_data
        ]
        return annotation

    def create_mock_entity(self, text, start, end, entity_type):
        """Create mock entity"""
        entity = Mock()
        entity.text = text
        entity.start_offset = start
        entity.end_offset = end
        entity.entity_type = entity_type
        entity.calculate_hierarchical_similarity.return_value = 0.8
        entity.id = hash(f"{text}_{start}_{end}")
        return entity

    def test_performance_analysis_workflow(self):
        """Test complete performance analysis workflow"""
        # Create mock predicted and gold annotations
        predicted_annotations = [
            self.create_mock_annotation_with_entities([
                ("severe", 0, 6, "SEVERITY"),
                ("chest pain", 7, 17, "SYMPTOM")
            ])
        ]

        gold_annotations = [
            self.create_mock_annotation_with_entities([
                ("severe", 0, 6, "SEVERITY"),
                ("chest", 7, 12, "BODY_PART"),
                ("pain", 13, 17, "SYMPTOM"),
                ("chest pain", 7, 17, "SYMPTOM")
            ])
        ]

        analysis = self.analyzer.analyze_performance(predicted_annotations, gold_annotations)

        self.assertIn('overall_performance', analysis)
        self.assertIn('error_patterns', analysis)
        self.assertIn('improvement_suggestions', analysis)
        self.assertIn('entity_type_insights', analysis)

        # Check overall performance metrics
        overall = analysis['overall_performance']
        self.assertIn('avg_precision', overall)
        self.assertIn('avg_recall', overall)
        self.assertIn('avg_f1', overall)

    def test_error_pattern_analysis(self):
        """Test error pattern identification"""
        # Create mock matching results with various error types
        results = []

        # Result with boundary errors
        result1 = MatchingResult()
        result1.boundary_errors = 5
        result1.type_errors = 2
        result1.unmatched_gold = [Mock(entity_type="SYMPTOM")]
        result1.unmatched_predicted = [Mock(entity_type="MEDICATION")]
        results.append(result1)

        error_patterns = self.analyzer._analyze_error_patterns(results)

        self.assertIn('boundary_errors', error_patterns)
        self.assertIn('type_errors', error_patterns)
        self.assertIn('missed_entities', error_patterns)
        self.assertIn('false_positives', error_patterns)

        self.assertEqual(error_patterns['boundary_errors']['count'], 5)
        self.assertEqual(error_patterns['type_errors']['count'], 2)

    def test_improvement_suggestions_generation(self):
        """Test generation of improvement suggestions"""
        # Create results with various performance issues
        results = []

        # Poor precision result
        poor_precision_result = MatchingResult()
        poor_precision_result.precision = 0.6  # Below threshold
        poor_precision_result.recall = 0.8
        poor_precision_result.f1_score = 0.69
        poor_precision_result.boundary_errors = 10
        poor_precision_result.matches = [Mock()] * 20  # 20 total matches
        results.append(poor_precision_result)

        suggestions = self.analyzer._generate_improvement_suggestions(results)

        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)

        # Should suggest reducing false positives
        precision_suggestion = any("false positives" in suggestion.lower() for suggestion in suggestions)
        self.assertTrue(precision_suggestion)


class WebInterfaceTests(TestCase):
    """Test web interface for overlapping entity annotation"""

    def setUp(self):
        self.client = Client()
        self.admin_user = User.objects.create_user(
            username='admin',
            password='testpass',
            role='admin'
        )
        self.annotator_user = User.objects.create_user(
            username='annotator',
            password='testpass',
            role='annotator'
        )
        self.document = TextDocument.objects.create(
            content="Patient has severe chest pain with shortness of breath.",
            uploaded_by=self.admin_user
        )

    def test_overlapping_annotation_view_access(self):
        """Test access to overlapping annotation view"""
        self.client.login(username='annotator', password='testpass')

        # Assign document to annotator
        self.document.assigned_to.add(self.annotator_user)

        url = reverse('annotation:overlapping_annotate', args=[self.document.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Overlapping Entity Annotation')
        self.assertContains(response, self.document.content)

    def test_annotation_saving_workflow(self):
        """Test saving overlapping annotations via AJAX"""
        self.client.login(username='annotator', password='testpass')
        self.document.assigned_to.add(self.annotator_user)

        # Mock annotation data
        annotation_data = {
            'entities': [
                {
                    'text': 'severe',
                    'start_offset': 12,
                    'end_offset': 18,
                    'entity_type': 'SEVERITY',
                    'confidence': 0.9,
                    'source': 'manual'
                },
                {
                    'text': 'chest pain',
                    'start_offset': 19,
                    'end_offset': 29,
                    'entity_type': 'SYMPTOM',
                    'confidence': 0.95,
                    'source': 'manual'
                }
            ],
            'statistics': {
                'total_entities': 2,
                'overlap_groups': 0,
                'max_overlap_depth': 1
            }
        }

        url = reverse('annotation:save_overlapping_annotation', args=[self.document.id])
        response = self.client.post(
            url,
            data=json.dumps(annotation_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertTrue(response_data.get('success', False))

    def test_analytics_dashboard_access(self):
        """Test access to analytics dashboard"""
        self.client.login(username='admin', password='testpass')

        url = reverse('annotation:overlapping_analytics_dashboard')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Overlapping Entity Performance Analytics')
        self.assertContains(response, 'Overall F1 Score')

    def test_unauthorized_access_protection(self):
        """Test that unauthorized users cannot access admin features"""
        # Try to access as annotator
        self.client.login(username='annotator', password='testpass')

        url = reverse('annotation:overlapping_analytics_dashboard')
        response = self.client.get(url)

        # Should be redirected or forbidden
        self.assertIn(response.status_code, [302, 403])


class DataIntegrityTests(TestCase):
    """Test data integrity and validation for overlapping entities"""

    def setUp(self):
        self.admin_user = User.objects.create_user(
            username='admin',
            password='testpass',
            role='admin'
        )
        self.document = TextDocument.objects.create(
            content="Patient has severe chest pain.",
            uploaded_by=self.admin_user
        )

    def test_entity_offset_validation(self):
        """Test that entity offsets are properly validated"""
        annotation_result = OverlappingAnnotationResult.objects.create(
            document=self.document,
            annotator=self.admin_user,
            annotation_type='manual_overlapping'
        )

        # Valid entity
        valid_entity = OverlappingEntity.objects.create(
            text="severe",
            start_offset=12,
            end_offset=18,
            entity_type="SEVERITY",
            annotation_result=annotation_result
        )

        self.assertEqual(valid_entity.get_span_length(), 6)

        # Test that the text matches the document content
        document_text = self.document.content[valid_entity.start_offset:valid_entity.end_offset]
        self.assertEqual(document_text, "severe")

    def test_overlap_consistency(self):
        """Test consistency of overlap relationships"""
        annotation_result = OverlappingAnnotationResult.objects.create(
            document=self.document,
            annotator=self.admin_user,
            annotation_type='manual_overlapping'
        )

        entity1 = OverlappingEntity.objects.create(
            text="severe",
            start_offset=12,
            end_offset=18,
            entity_type="SEVERITY",
            annotation_result=annotation_result
        )

        entity2 = OverlappingEntity.objects.create(
            text="severe chest",
            start_offset=12,
            end_offset=24,
            entity_type="SYMPTOM",
            annotation_result=annotation_result
        )

        # Test symmetric overlap relationship
        self.assertTrue(entity1.overlaps_with(entity2))
        self.assertTrue(entity2.overlaps_with(entity1))

        # Test containment relationship
        self.assertTrue(entity2.contains(entity1))
        self.assertTrue(entity1.is_contained_by(entity2))

    def test_statistics_accuracy(self):
        """Test accuracy of calculated statistics"""
        annotation_result = OverlappingAnnotationResult.objects.create(
            document=self.document,
            annotator=self.admin_user,
            annotation_type='manual_overlapping'
        )

        # Create entities with known overlaps
        entities_data = [
            ("severe", 12, 18, "SEVERITY"),
            ("chest", 19, 24, "BODY_PART"),
            ("severe chest", 12, 24, "SYMPTOM")
        ]

        for text, start, end, entity_type in entities_data:
            OverlappingEntity.objects.create(
                text=text,
                start_offset=start,
                end_offset=end,
                entity_type=entity_type,
                annotation_result=annotation_result
            )

        annotation_result.calculate_overlap_statistics()

        # Verify statistics
        self.assertEqual(annotation_result.total_entities, 3)
        self.assertGreater(annotation_result.overlapping_entity_count, 0)
        self.assertGreaterEqual(annotation_result.max_overlap_depth, 2)


class PerformanceTests(TestCase):
    """Test performance of overlapping entity algorithms"""

    def test_matching_algorithm_performance(self):
        """Test performance of matching algorithms with large entity sets"""
        matcher = AdvancedPartialMatcher()

        # Create large sets of entities for performance testing
        def create_entity_set(size):
            entities = []
            for i in range(size):
                entity = Mock()
                entity.text = f"entity_{i}"
                entity.start_offset = i * 10
                entity.end_offset = i * 10 + 5
                entity.entity_type = "SYMPTOM"
                entity.confidence_score = 0.8
                return entities

        # Test with 100 entities
        predicted_entities = create_entity_set(100)
        gold_entities = create_entity_set(100)

        start_time = time.time()
        result = matcher.compare_annotations(predicted_entities, gold_entities)
        processing_time = time.time() - start_time

        # Should complete within reasonable time (< 5 seconds)
        self.assertLess(processing_time, 5.0)
        self.assertIsInstance(result, MatchingResult)

    def test_overlap_group_identification_performance(self):
        """Test performance of overlap group identification"""
        matcher = AdvancedPartialMatcher()

        # Create entities with complex overlaps
        entities = []
        for i in range(50):
            entity = Mock()
            entity.start_offset = i * 2  # Overlapping entities
            entity.end_offset = i * 2 + 10
            entity.entity_type = "SYMPTOM"
            entity.id = i
            entities.append(entity)

        start_time = time.time()
        groups = matcher._identify_overlap_groups(entities)
        processing_time = time.time() - start_time

        # Should complete quickly even with complex overlaps
        self.assertLess(processing_time, 2.0)
        self.assertIsInstance(groups, list)


# Test Suite Configuration
class OverlappingEntityTestSuite:
    """Complete test suite for overlapping entity annotation system"""

    @staticmethod
    def get_test_suite():
        """Get complete test suite"""
        suite = unittest.TestSuite()

        # Add all test cases
        test_cases = [
            OverlappingEntityModelTests,
            PartialMatchingAlgorithmTests,
            LLMIntegrationTests,
            PerformanceAnalysisTests,
            WebInterfaceTests,
            DataIntegrityTests,
            PerformanceTests
        ]

        for test_case in test_cases:
            tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
            suite.addTests(tests)

        return suite

    @staticmethod
    def run_all_tests():
        """Run all tests and return results"""
        suite = OverlappingEntityTestSuite.get_test_suite()
        runner = unittest.TextTestRunner(verbosity=2)
        return runner.run(suite)


# Command-line test runner
if __name__ == '__main__':
    # Run specific test or all tests
    import sys

    if len(sys.argv) > 1:
        # Run specific test case
        test_name = sys.argv[1]
        suite = unittest.TestLoader().loadTestsFromName(test_name)
    else:
        # Run all tests
        suite = OverlappingEntityTestSuite.get_test_suite()

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)