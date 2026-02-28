"""
OpenAI NER Provider
Supports GPT-4o and other OpenAI models for medical NER
"""

import json
import time
import re
import os
from typing import List, Dict, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseLLMProvider, NERResult, Entity


class OpenAINERProvider(BaseLLMProvider):
    """OpenAI provider for NER with overlapping entity support"""

    def __init__(self, model_id: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model_id, api_key)
        self.client = None

    def initialize(self) -> bool:
        """Initialize the OpenAI client"""
        if not OPENAI_AVAILABLE:
            print("OpenAI library not installed. Install with: pip install openai")
            return False

        try:
            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                return False

            self.client = OpenAI(api_key=api_key)
            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize OpenAI provider: {e}")
            return False

    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        if not OPENAI_AVAILABLE:
            return False

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
        """Annotate text using OpenAI"""
        start_time = time.time()

        if not self.is_available():
            return self.create_error_result("OpenAI API not configured or unavailable")

        try:
            # Create the prompt
            if enable_overlapping:
                prompt = self._create_overlapping_prompt(text)
            else:
                prompt = self._create_standard_prompt(text)

            # Call OpenAI API
            raw_entities = self._call_openai(prompt, max_retries)

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

    def _call_openai(self, prompt: str, max_retries: int) -> List[Dict]:
        """Call OpenAI API with retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert medical text annotator specializing in named entity recognition. You always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0,  # Deterministic for medical annotations
                    response_format={"type": "json_object"} if self.model_id in ["gpt-4o", "gpt-4-turbo"] else None
                )

                response_text = response.choices[0].message.content
                return self._parse_response(response_text)

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

        prompt = f"""
You are an expert medical text annotator specializing in detecting OVERLAPPING medical entities.

ENTITY TYPES:
{entity_definitions}

INSTRUCTIONS:
1. Identify ALL medical entities, including overlapping entities
2. The same text span can belong to multiple entities
3. Look for modifier-noun combinations (e.g., "severe chest pain" can be both "chest pain" and include "chest" as BODY_PART)
4. Include modifiers as PART of the entity text
5. Provide accurate character offsets (start_offset, end_offset)
6. Assign confidence scores (0.0 to 1.0) based on certainty

TEXT TO ANNOTATE:
"{text}"

RESPONSE FORMAT:
Return a JSON object with an "entities" array:
{{
    "entities": [
        {{
            "text": "exact text span",
            "start_offset": <character position>,
            "end_offset": <character position>,
            "label": "<ENTITY_TYPE>",
            "confidence": <0.0 to 1.0>,
            "reasoning": "brief explanation"
        }}
    ]
}}

Provide ONLY the JSON response, no additional text.
"""
        return prompt

    def _create_standard_prompt(self, text: str) -> str:
        """Create standard prompt for non-overlapping entities"""
        entity_types = list(self.ENTITY_TYPES.keys())

        prompt = f"""
Extract medical named entities from the following text.

Entity types: {entity_types}

Text: "{text}"

Return a JSON object with an "entities" array:
{{
    "entities": [
        {{
            "text": "exact text span",
            "start_offset": <character position>,
            "end_offset": <character position>,
            "label": "<ENTITY_TYPE>",
            "confidence": <0.0 to 1.0>
        }}
    ]
}}

Provide ONLY the JSON response.
"""
        return prompt

    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse OpenAI response for entities"""
        try:
            # Parse JSON response
            response_data = json.loads(response_text)

            # Extract entities array
            if isinstance(response_data, dict) and 'entities' in response_data:
                entities = response_data['entities']
            elif isinstance(response_data, list):
                entities = response_data
            else:
                return []

            # Validate entities
            validated_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'text' in entity and 'label' in entity:
                    validated_entities.append(entity)

            return validated_entities

        except json.JSONDecodeError as e:
            print(f"Error parsing OpenAI response: {e}")
            # Try to extract JSON from text
            try:
                json_match = re.search(r'\{.*"entities".*\}', response_text, re.DOTALL)
                if json_match:
                    return self._parse_response(json_match.group())
            except:
                pass
            return []
        except Exception as e:
            print(f"Error processing OpenAI response: {e}")
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
            'gpt-4o': 'GPT-4o',
            'gpt-4-turbo': 'GPT-4 Turbo',
            'gpt-4': 'GPT-4',
            'gpt-3.5-turbo': 'GPT-3.5 Turbo',
        }
        return display_names.get(self.model_id, self.model_id)
