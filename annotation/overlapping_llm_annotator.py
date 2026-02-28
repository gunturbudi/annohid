"""
Advanced LLM Integration for Overlapping Medical Entity Detection

This module provides sophisticated LLM-based annotation for overlapping medical entities,
including specialized prompting strategies, confidence estimation, and performance tracking.
"""

import json
import time
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import google.generativeai as genai
import os
import numpy as np


@dataclass
class OverlappingEntity:
    """Represents an overlapping entity prediction from LLM"""
    text: str
    start_offset: int
    end_offset: int
    entity_type: str
    confidence: float
    reasoning: str = ""
    overlaps_with: List[str] = field(default_factory=list)
    source: str = "llm"
    raw_prediction: Dict = field(default_factory=dict)


@dataclass
class OverlappingAnnotationResult:
    """Complete annotation result with overlapping entities"""
    entities: List[OverlappingEntity]
    overlap_groups: List[List[OverlappingEntity]]
    processing_time: float
    model_used: str
    total_entities: int
    overlapping_count: int
    max_overlap_depth: int
    confidence_distribution: Dict[str, int]
    error_messages: List[str] = field(default_factory=list)
    success: bool = True


class OverlappingLLMAnnotator:
    """Advanced LLM annotator specialized for overlapping medical entity detection"""

    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        """
        Initialize the overlapping LLM annotator

        Args:
            model_id: The LLM model to use
            api_key: API key for the model (if not set in environment)
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')

        # Initialize the model
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_id)
        else:
            self.model = None

        # Entity type definitions (matching database schema)
        self.entity_types = {
            'SYMPTOM': 'Medical symptoms, complaints, or clinical presentations (subjective complaints, feelings, sensations)',
            'DISEASE': 'Diseases, disorders, conditions, or diagnoses (specific diseases, conditions, syndromes)',
            'BODY_PART': 'Anatomical structures, organs, or body regions (anatomical structures, organs, body regions)',
            'MEDICATION': 'Drugs, medications, pharmaceuticals, or therapeutic substances (drugs, medicines, treatments)',
            'PROCEDURE': 'Medical procedures, treatments, operations, or interventions (medical procedures, examinations, therapies)',
        }

        # Confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }

    def annotate_document(self, text: str,
                         enable_overlapping: bool = True,
                         confidence_threshold: float = 0.3,
                         max_retries: int = 3) -> OverlappingAnnotationResult:
        """
        Annotate a medical document with overlapping entity detection

        Args:
            text: The medical text to annotate
            enable_overlapping: Whether to detect overlapping entities
            confidence_threshold: Minimum confidence for entity inclusion
            max_retries: Maximum retry attempts for failed predictions

        Returns:
            OverlappingAnnotationResult with detected entities
        """
        start_time = time.time()

        if not self.model:
            return OverlappingAnnotationResult(
                entities=[],
                overlap_groups=[],
                processing_time=time.time() - start_time,
                model_used=self.model_id,
                total_entities=0,
                overlapping_count=0,
                max_overlap_depth=0,
                confidence_distribution={},
                error_messages=["API key not configured"],
                success=False
            )

        try:
            # Generate overlapping entity predictions
            if enable_overlapping:
                raw_entities = self._extract_overlapping_entities(text, max_retries)
            else:
                raw_entities = self._extract_standard_entities(text, max_retries)

            # Process and validate entities
            processed_entities = self._process_raw_entities(raw_entities, text, confidence_threshold)

            # Identify overlap groups
            overlap_groups = self._identify_overlap_groups(processed_entities)

            # Calculate statistics
            overlapping_count = sum(len(group) for group in overlap_groups if len(group) > 1)
            max_overlap_depth = self._calculate_max_overlap_depth(processed_entities)
            confidence_distribution = self._calculate_confidence_distribution(processed_entities)

            processing_time = time.time() - start_time

            return OverlappingAnnotationResult(
                entities=processed_entities,
                overlap_groups=overlap_groups,
                processing_time=processing_time,
                model_used=self.model_id,
                total_entities=len(processed_entities),
                overlapping_count=overlapping_count,
                max_overlap_depth=max_overlap_depth,
                confidence_distribution=confidence_distribution,
                success=True
            )

        except Exception as e:
            return OverlappingAnnotationResult(
                entities=[],
                overlap_groups=[],
                processing_time=time.time() - start_time,
                model_used=self.model_id,
                total_entities=0,
                overlapping_count=0,
                max_overlap_depth=0,
                confidence_distribution={},
                error_messages=[str(e)],
                success=False
            )

    def _extract_overlapping_entities(self, text: str, max_retries: int) -> List[Dict]:
        """Extract overlapping entities using specialized prompting"""

        prompt = self._create_overlapping_prompt(text)

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)

                if response.text:
                    # Parse the structured response
                    entities = self._parse_overlapping_response(response.text, text)
                    return entities

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)  # Brief pause before retry

        return []

    def _create_overlapping_prompt(self, text: str) -> str:
        """Create specialized prompt for overlapping entity detection"""

        entity_definitions = "\n".join([
            f"- {entity_type}: {definition}"
            for entity_type, definition in self.entity_types.items()
        ])

        examples = self._get_overlapping_examples()

        prompt = f"""
You are an expert medical text annotator specializing in detecting OVERLAPPING medical entities.
Your task is to identify ALL medical entities in the text, including entities that OVERLAP or SHARE the same text spans.

ENTITY TYPES:
{entity_definitions}

CRITICAL INSTRUCTIONS FOR OVERLAPPING ENTITIES:
1. Entities can OVERLAP - the same text can belong to multiple entities
2. Look for modifier-noun combinations (e.g., "severe chest pain" contains BODY_PART + SYMPTOM with modifier)
3. Look for compound entities (e.g., "metformin 500mg" is MEDICATION with dosage included)
4. Identify both atomic entities (single concept) and compound entities (multiple overlapping concepts)
5. Include all meaningful medical entities, even if they overlap
6. Include modifiers (severity, frequency, dosage) as PART of the entity text, not as separate entities
7. For body parts within larger medical terms, extract both the body part and the full term

EXAMPLES OF OVERLAPPING ENTITIES:
{examples}

RESPONSE FORMAT:
Return a JSON array where each entity has:
{{
    "text": "exact text span",
    "start_offset": character_position,
    "end_offset": character_position,
    "entity_type": "TYPE",
    "confidence": confidence_score_0_to_1,
    "reasoning": "brief explanation for detection",
    "overlaps_with": ["list of other entity texts that overlap with this one"]
}}

TEXT TO ANNOTATE:
"{text}"

IMPORTANT:
- Be exhaustive in finding overlapping entities
- Include confidence scores based on how certain you are
- Provide reasoning for complex or ambiguous cases
- Ensure character positions are accurate
- Look for nested and overlapping patterns

JSON RESPONSE:
"""
        return prompt

    def _get_overlapping_examples(self) -> str:
        """Get examples of overlapping entity patterns"""
        return """
Example 1: "severe chest pain"
- "chest" (BODY_PART, 7-12)
- "pain" (SYMPTOM, 13-17)
- "chest pain" (SYMPTOM, 7-17) [overlaps with "chest" and "pain"]
- "severe chest pain" (SYMPTOM, 0-17) [overlaps with all above, includes severity modifier]

Example 2: "Patient has diabetes and takes metformin 500mg daily"
- "diabetes" (DISEASE, 12-20)
- "metformin 500mg" (MEDICATION, 31-45) [includes dosage]
- "metformin" (MEDICATION, 31-40) [overlaps with full medication span]

Example 3: "acute abdominal pain"
- "abdominal" (BODY_PART, 6-15)
- "pain" (SYMPTOM, 16-20)
- "abdominal pain" (SYMPTOM, 6-20) [overlaps with body part and symptom]
- "acute abdominal pain" (SYMPTOM, 0-20) [overlaps with all, includes severity modifier]

Example 4: "underwent cardiac catheterization"
- "cardiac" (BODY_PART, 10-17) [can be body part reference]
- "cardiac catheterization" (PROCEDURE, 10-33) [full procedure name]
"""

    def _extract_standard_entities(self, text: str, max_retries: int) -> List[Dict]:
        """Extract entities using standard (non-overlapping) approach for comparison"""

        prompt = f"""
Extract medical entities from the following text. Return a JSON array with:
- text: exact text span
- start_offset: character position
- end_offset: character position
- entity_type: one of {list(self.entity_types.keys())}
- confidence: score from 0 to 1

Text: "{text}"

JSON Response:
"""

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return self._parse_standard_response(response.text, text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)

        return []

    def _parse_overlapping_response(self, response_text: str, original_text: str) -> List[Dict]:
        """Parse LLM response for overlapping entities"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                # Try to find JSON in code blocks
                json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    raise ValueError("No JSON array found in response")
            else:
                json_text = json_match.group()

            entities = json.loads(json_text)

            # Validate and clean entities
            validated_entities = []
            for entity in entities:
                validated_entity = self._validate_entity(entity, original_text)
                if validated_entity:
                    validated_entities.append(validated_entity)

            return validated_entities

        except Exception as e:
            print(f"Error parsing overlapping response: {e}")
            return []

    def _parse_standard_response(self, response_text: str, original_text: str) -> List[Dict]:
        """Parse LLM response for standard entities"""
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                validated_entities = []
                for entity in entities:
                    validated_entity = self._validate_entity(entity, original_text)
                    if validated_entity:
                        validated_entities.append(validated_entity)
                return validated_entities
        except Exception as e:
            print(f"Error parsing standard response: {e}")

        return []

    def _validate_entity(self, entity: Dict, original_text: str) -> Optional[Dict]:
        """Validate and clean an entity prediction"""
        try:
            # Required fields
            text = entity.get('text', '').strip()
            entity_type = entity.get('entity_type', '').upper()

            if not text or entity_type not in self.entity_types:
                return None

            # Calculate or validate offsets
            start_offset = entity.get('start_offset')
            end_offset = entity.get('end_offset')

            if start_offset is None or end_offset is None:
                # Try to find the text in the original
                start_offset = original_text.find(text)
                if start_offset == -1:
                    return None
                end_offset = start_offset + len(text)
            else:
                # Validate existing offsets
                if (start_offset < 0 or end_offset > len(original_text) or
                    start_offset >= end_offset):
                    return None

                # Check if text matches
                actual_text = original_text[start_offset:end_offset]
                if actual_text.strip() != text.strip():
                    # Try to find correct position
                    start_offset = original_text.find(text)
                    if start_offset == -1:
                        return None
                    end_offset = start_offset + len(text)

            # Confidence score
            confidence = entity.get('confidence', 0.5)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                confidence = 0.5

            # Additional fields
            reasoning = entity.get('reasoning', '')
            overlaps_with = entity.get('overlaps_with', [])

            return {
                'text': text,
                'start_offset': int(start_offset),
                'end_offset': int(end_offset),
                'entity_type': entity_type,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'overlaps_with': overlaps_with,
                'raw_prediction': entity
            }

        except Exception as e:
            print(f"Error validating entity: {e}")
            return None

    def _process_raw_entities(self, raw_entities: List[Dict],
                            original_text: str,
                            confidence_threshold: float) -> List[OverlappingEntity]:
        """Process raw entity predictions into OverlappingEntity objects"""

        processed_entities = []

        for entity_data in raw_entities:
            if entity_data['confidence'] >= confidence_threshold:
                entity = OverlappingEntity(
                    text=entity_data['text'],
                    start_offset=entity_data['start_offset'],
                    end_offset=entity_data['end_offset'],
                    entity_type=entity_data['entity_type'],
                    confidence=entity_data['confidence'],
                    reasoning=entity_data.get('reasoning', ''),
                    overlaps_with=entity_data.get('overlaps_with', []),
                    raw_prediction=entity_data
                )
                processed_entities.append(entity)

        # Post-process to identify actual overlaps
        self._update_overlap_information(processed_entities)

        return processed_entities

    def _update_overlap_information(self, entities: List[OverlappingEntity]):
        """Update overlap information based on actual entity positions"""

        for i, entity1 in enumerate(entities):
            actual_overlaps = []

            for j, entity2 in enumerate(entities):
                if i != j and self._entities_overlap(entity1, entity2):
                    actual_overlaps.append(entity2.text)

            entity1.overlaps_with = actual_overlaps

    def _entities_overlap(self, entity1: OverlappingEntity, entity2: OverlappingEntity) -> bool:
        """Check if two entities overlap"""
        return not (entity1.end_offset <= entity2.start_offset or
                   entity1.start_offset >= entity2.end_offset)

    def _identify_overlap_groups(self, entities: List[OverlappingEntity]) -> List[List[OverlappingEntity]]:
        """Identify groups of mutually overlapping entities"""

        groups = []
        processed = set()

        for i, entity in enumerate(entities):
            if id(entity) in processed:
                continue

            # Find all entities that overlap with this one
            group = [entity]
            processed.add(id(entity))

            # Use iterative approach to find all connected overlaps
            to_check = [entity]

            while to_check:
                current = to_check.pop()

                for other_entity in entities:
                    if (id(other_entity) not in processed and
                        self._entities_overlap(current, other_entity)):
                        group.append(other_entity)
                        processed.add(id(other_entity))
                        to_check.append(other_entity)

            groups.append(group)

        return groups

    def _calculate_max_overlap_depth(self, entities: List[OverlappingEntity]) -> int:
        """Calculate maximum number of entities overlapping at any position"""

        if not entities:
            return 0

        # Create events for start and end positions
        events = []
        for entity in entities:
            events.append((entity.start_offset, 'start'))
            events.append((entity.end_offset, 'end'))

        # Sort events
        events.sort(key=lambda x: (x[0], x[1] == 'end'))

        max_depth = 0
        current_depth = 0

        for position, event_type in events:
            if event_type == 'start':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            else:
                current_depth -= 1

        return max_depth

    def _calculate_confidence_distribution(self, entities: List[OverlappingEntity]) -> Dict[str, int]:
        """Calculate distribution of entities by confidence level"""

        distribution = {'high': 0, 'medium': 0, 'low': 0}

        for entity in entities:
            if entity.confidence >= self.confidence_thresholds['high']:
                distribution['high'] += 1
            elif entity.confidence >= self.confidence_thresholds['medium']:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1

        return distribution

    def estimate_overlap_complexity(self, text: str) -> Dict[str, Any]:
        """Estimate the overlap complexity of a text before annotation"""

        # Simple heuristics for overlap complexity
        word_count = len(text.split())
        medical_terms = self._count_medical_terms(text)
        modifier_patterns = self._count_modifier_patterns(text)

        complexity_score = (medical_terms * 0.4 +
                          modifier_patterns * 0.3 +
                          min(word_count / 20, 1.0) * 0.3)

        if complexity_score > 0.7:
            complexity = 'high'
        elif complexity_score > 0.4:
            complexity = 'medium'
        else:
            complexity = 'low'

        return {
            'complexity': complexity,
            'score': complexity_score,
            'estimated_entities': int(medical_terms * 1.5),
            'estimated_overlaps': int(modifier_patterns * 0.8),
            'medical_term_density': medical_terms / word_count if word_count > 0 else 0
        }

    def _count_medical_terms(self, text: str) -> int:
        """Count potential medical terms using simple patterns"""

        medical_patterns = [
            r'\b\w*(pain|ache|itis|osis|emia|pathy|therapy|gram|scopy)\b',
            r'\b(mg|ml|tablet|pill|dose|daily|twice|three times)\b',
            r'\b(severe|mild|acute|chronic|persistent|intermittent)\b',
            r'\b(left|right|upper|lower|anterior|posterior)\b'
        ]

        count = 0
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)

        return count

    def _count_modifier_patterns(self, text: str) -> int:
        """Count modifier-noun patterns that suggest overlapping entities"""

        # Simple pattern for adjective-noun combinations
        modifier_pattern = r'\b(severe|mild|acute|chronic|persistent|intermittent|sudden|gradual)\s+\w+\s*(pain|symptoms?|condition)\b'
        matches = re.findall(modifier_pattern, text, re.IGNORECASE)

        return len(matches)

    def validate_api_setup(self) -> Dict[str, Any]:
        """Validate that the API is properly configured"""

        if not self.api_key:
            return {
                'ready_for_annotation': False,
                'error': 'API key not configured',
                'model_available': False
            }

        try:
            # Test with a simple prompt
            test_response = self.model.generate_content("Say 'API working'")

            return {
                'ready_for_annotation': True,
                'model_available': True,
                'model_id': self.model_id,
                'api_configured': True
            }

        except Exception as e:
            return {
                'ready_for_annotation': False,
                'error': str(e),
                'model_available': False,
                'api_configured': False
            }

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return [
            'gemini-2.5-flash',
            'gemini-1.5-pro',
            'gemini-1.5-flash',
            'gemini-pro'
        ]


# Factory function for easy instantiation
def create_overlapping_medical_annotator(model_id: str = "gemini-2.5-flash") -> OverlappingLLMAnnotator:
    """Create an overlapping medical annotator instance"""
    return OverlappingLLMAnnotator(model_id=model_id)