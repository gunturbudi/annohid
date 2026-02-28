"""
Google Gemini NER Provider
Adapts the existing overlapping_llm_annotator.py for the multi-provider architecture
"""

import json
import time
import re
import os
from typing import List, Dict, Optional
import google.generativeai as genai

from .base import BaseLLMProvider, NERResult, Entity


class GeminiNERProvider(BaseLLMProvider):
    """Google Gemini provider for NER with overlapping entity support"""

    def __init__(self, model_id: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        super().__init__(model_id, api_key)
        self.client = None

    def initialize(self) -> bool:
        """Initialize the Gemini client"""
        try:
            api_key = self.api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                return False

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_id)
            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize Gemini provider: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Gemini is available"""
        if not self.is_initialized:
            self.initialize()
        return self.client is not None

    def annotate_text(
        self,
        text: str,
        enable_overlapping: bool = True,
        confidence_threshold: float = 0.3,
        max_retries: int = 3
    ) -> NERResult:
        """Annotate text using Gemini"""
        start_time = time.time()

        if not self.is_available():
            return self.create_error_result("Gemini API not configured or unavailable")

        try:
            # Generate entity predictions
            if enable_overlapping:
                raw_entities = self._extract_overlapping_entities(text, max_retries)
            else:
                raw_entities = self._extract_standard_entities(text, max_retries)

            # Process and validate entities
            processed_entities = self._process_entities(raw_entities, text, confidence_threshold)

            processing_time = time.time() - start_time

            return NERResult(
                entities=processed_entities,
                processing_time=processing_time,
                model_used=self.model_id,
                success=True
            )

        except Exception as e:
            return self.create_error_result(str(e))

    def _extract_overlapping_entities(self, text: str, max_retries: int) -> List[Dict]:
        """Extract overlapping entities using specialized prompting"""
        prompt = self._create_overlapping_prompt(text)

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(prompt)

                if response.text:
                    entities = self._parse_response(response.text, text)
                    return entities

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)

        return []

    def _extract_standard_entities(self, text: str, max_retries: int) -> List[Dict]:
        """Extract entities using standard approach"""
        prompt = f"""
Extract medical entities from the following text. Return a JSON array with:
- text: exact text span
- start_offset: character position
- end_offset: character position
- label: one of {list(self.ENTITY_TYPES.keys())}
- confidence: score from 0 to 1

Text: "{text}"

JSON Response:
"""

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content(prompt)
                if response.text:
                    return self._parse_response(response.text, text)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)

        return []

    def _create_overlapping_prompt(self, text: str) -> str:
        """Create specialized prompt for overlapping entity detection"""
        entity_definitions = "\n".join([
            f"- {entity_type}: {definition}"
            for entity_type, definition in self.ENTITY_TYPES.items()
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
    "label": "TYPE",
    "confidence": confidence_score_0_to_1,
    "reasoning": "brief explanation for detection"
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
"""

    def _parse_response(self, response_text: str, original_text: str) -> List[Dict]:
        """Parse LLM response for entities"""
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

            # Validate entities
            validated_entities = []
            for entity in entities:
                # Ensure entity has required fields
                if 'text' in entity and ('label' in entity or 'entity_type' in entity):
                    validated_entities.append(entity)

            return validated_entities

        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return []

    def _process_entities(
        self,
        raw_entities: List[Dict],
        original_text: str,
        confidence_threshold: float
    ) -> List[Entity]:
        """Process raw entity predictions into Entity objects"""
        processed_entities = []

        for entity_data in raw_entities:
            entity = self.validate_entity(entity_data, original_text)

            if entity and entity.confidence >= confidence_threshold:
                processed_entities.append(entity)

        return processed_entities

    def get_model_display_name(self) -> str:
        """Get human-readable model name"""
        display_names = {
            'gemini-2.5-flash': 'Gemini 2.5 Flash',
            'gemini-1.5-pro': 'Gemini 1.5 Pro',
            'gemini-1.5-flash': 'Gemini 1.5 Flash',
            'gemini-pro': 'Gemini Pro',
        }
        return display_names.get(self.model_id, self.model_id)
