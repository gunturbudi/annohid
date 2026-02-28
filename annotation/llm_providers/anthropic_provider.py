"""
Anthropic Claude NER Provider
Supports Claude 3.5 Sonnet and other Claude models for medical NER
"""

import json
import time
import re
import os
from typing import List, Dict, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .base import BaseLLMProvider, NERResult, Entity


class AnthropicNERProvider(BaseLLMProvider):
    """Anthropic Claude provider for NER with overlapping entity support"""

    def __init__(self, model_id: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        super().__init__(model_id, api_key)
        self.client = None

    def initialize(self) -> bool:
        """Initialize the Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            print("Anthropic library not installed. Install with: pip install anthropic")
            return False

        try:
            api_key = self.api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return False

            self.client = anthropic.Anthropic(api_key=api_key)
            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize Anthropic provider: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        if not ANTHROPIC_AVAILABLE:
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
        """Annotate text using Claude"""
        start_time = time.time()

        if not self.is_available():
            return self.create_error_result("Anthropic API not configured or unavailable")

        try:
            # Create the prompt
            if enable_overlapping:
                prompt = self._create_overlapping_prompt(text)
            else:
                prompt = self._create_standard_prompt(text)

            # Call Anthropic API
            raw_entities = self._call_anthropic(prompt, max_retries)

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

    def _call_anthropic(self, prompt: str, max_retries: int) -> List[Dict]:
        """Call Anthropic API with retry logic"""
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=4096,
                    temperature=0,  # Deterministic for medical annotations
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                response_text = message.content[0].text
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

        prompt = f"""You are an expert medical text annotator specializing in detecting OVERLAPPING medical entities.

ENTITY TYPES:
{entity_definitions}

CRITICAL INSTRUCTIONS:
1. Identify ALL medical entities, including those that overlap
2. The same text span can belong to multiple entities (e.g., "chest" in "chest pain" can be both BODY_PART and part of SYMPTOM "chest pain")
3. Look for both atomic entities and compound entities
4. Include modifiers (severity, frequency) as PART of the entity text
5. Provide accurate character offsets (start_offset, end_offset)
6. Assign confidence scores (0.0 to 1.0) based on your certainty
7. Provide brief reasoning for complex or ambiguous cases

EXAMPLES:
Text: "severe chest pain"
Entities:
- "chest" (BODY_PART, offset 7-12, confidence 0.9)
- "pain" (SYMPTOM, offset 13-17, confidence 0.85)
- "chest pain" (SYMPTOM, offset 7-17, confidence 0.95)
- "severe chest pain" (SYMPTOM, offset 0-17, confidence 0.92)

TEXT TO ANNOTATE:
"{text}"

RESPONSE FORMAT:
Return ONLY a JSON array of entities (no markdown formatting):
[
    {{
        "text": "exact text span from the input",
        "start_offset": <character position where entity starts>,
        "end_offset": <character position where entity ends>,
        "label": "<one of: SYMPTOM, DISEASE, BODY_PART, MEDICATION, PROCEDURE>",
        "confidence": <number between 0.0 and 1.0>,
        "reasoning": "brief explanation if needed"
    }}
]

Important: Ensure character offsets are accurate. The text[start_offset:end_offset] should exactly match the "text" field.
Provide ONLY the JSON array, no additional text or formatting."""
        return prompt

    def _create_standard_prompt(self, text: str) -> str:
        """Create standard prompt for non-overlapping entities"""
        entity_types = list(self.ENTITY_TYPES.keys())

        prompt = f"""Extract medical named entities from the following text.

Entity types: {entity_types}

Text: "{text}"

Return ONLY a JSON array (no markdown):
[
    {{
        "text": "exact text span",
        "start_offset": <character position>,
        "end_offset": <character position>,
        "label": "<ENTITY_TYPE>",
        "confidence": <0.0 to 1.0>
    }}
]

Provide ONLY the JSON array, no additional text."""
        return prompt

    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse Claude response for entities"""
        try:
            # Remove markdown code blocks if present
            text = response_text.strip()
            if '```json' in text:
                start = text.find('```json') + 7
                end = text.find('```', start)
                text = text[start:end].strip()
            elif text.startswith('```'):
                lines = text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or not line.strip().startswith('```'):
                        json_lines.append(line)
                text = '\n'.join(json_lines).strip()

            # Extract JSON array
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                text = json_match.group()

            # Parse JSON
            entities = json.loads(text)

            # Validate entities
            validated_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'text' in entity and 'label' in entity:
                    validated_entities.append(entity)

            return validated_entities

        except json.JSONDecodeError as e:
            print(f"Error parsing Claude response: {e}")
            print(f"Response text: {response_text[:500]}")
            return []
        except Exception as e:
            print(f"Error processing Claude response: {e}")
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
            'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
            'claude-3-5-sonnet-20240620': 'Claude 3.5 Sonnet (June)',
            'claude-3-opus-20240229': 'Claude 3 Opus',
            'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
            'claude-3-haiku-20240307': 'Claude 3 Haiku',
        }
        return display_names.get(self.model_id, self.model_id)
