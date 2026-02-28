"""
OpenRouter NER Provider
Universal gateway to 100+ LLM models through a single API
Supports OpenAI, Anthropic, Google, Meta, and many more providers
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


class OpenRouterNERProvider(BaseLLMProvider):
    """
    OpenRouter provider for NER - access 100+ models with one API key
    Uses OpenAI-compatible API interface
    """

    # Popular models available through OpenRouter
    # Format: "provider/model-name"
    SUPPORTED_MODELS = {
        # OpenAI models
        'openai/gpt-4o': 'GPT-4o',
        'openai/gpt-4o-mini': 'GPT-4o Mini',
        'openai/gpt-4-turbo': 'GPT-4 Turbo',
        'openai/o1': 'OpenAI o1',
        'openai/o1-mini': 'OpenAI o1 Mini',

        # Anthropic Claude models
        'anthropic/claude-3.5-sonnet': 'Claude 3.5 Sonnet',
        'anthropic/claude-3-opus': 'Claude 3 Opus',
        'anthropic/claude-3-haiku': 'Claude 3 Haiku',
        'anthropic/claude-sonnet-4.5': 'Claude Sonnet 4.5',
        'anthropic/claude-opus-4': 'Claude Opus 4',

        # Google models
        'google/gemini-2.5-flash': 'Gemini 2.5 Flash',
        'google/gemini-2.0-pro-exp': 'Gemini 2.0 Pro Exp',
        'google/gemini-pro-1.5': 'Gemini Pro 1.5',
        'google/gemini-flash-1.5': 'Gemini Flash 1.5',

        # Meta models
        'meta-llama/llama-3.3-70b-instruct': 'Llama 3.3 70B',
        'meta-llama/llama-3.1-405b-instruct': 'Llama 3.1 405B',
        'meta-llama/llama-3.1-70b-instruct': 'Llama 3.1 70B',

        # Mistral models
        'mistralai/mistral-large': 'Mistral Large',
        'mistralai/mistral-medium': 'Mistral Medium',
        'mistralai/mistral-small': 'Mistral Small',

        # DeepSeek models
        'deepseek/deepseek-chat': 'DeepSeek Chat',
        'deepseek/deepseek-coder': 'DeepSeek Coder',

        # Cohere models
        'cohere/command-r-plus': 'Command R+',
        'cohere/command-r': 'Command R',

        # Free models (great for testing!)
        'meta-llama/llama-3.1-8b-instruct:free': 'Llama 3.1 8B (Free)',
        'google/gemma-2-9b-it:free': 'Gemma 2 9B (Free)',
        'mistralai/mistral-7b-instruct:free': 'Mistral 7B (Free)',
    }

    def __init__(self, model_id: str, api_key: Optional[str] = None):
        super().__init__(model_id, api_key)
        self.client = None
        self.base_url = "https://openrouter.ai/api/v1"

    def initialize(self) -> bool:
        """Initialize the OpenRouter client"""
        if not OPENAI_AVAILABLE:
            print("OpenAI library not installed. Install with: pip install openai")
            return False

        try:
            api_key = self.api_key or os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                return False

            self.client = OpenAI(
                base_url=self.base_url,
                api_key=api_key
            )
            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Failed to initialize OpenRouter provider: {e}")
            return False

    def is_available(self) -> bool:
        """Check if OpenRouter is available"""
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
        """Annotate text using OpenRouter"""
        start_time = time.time()

        if not self.is_available():
            return self.create_error_result("OpenRouter API not configured or unavailable")

        try:
            # Create the prompt
            if enable_overlapping:
                prompt = self._create_overlapping_prompt(text)
            else:
                prompt = self._create_standard_prompt(text)

            # Call OpenRouter API
            raw_entities = self._call_openrouter(prompt, max_retries)

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

    def _call_openrouter(self, prompt: str, max_retries: int) -> List[Dict]:
        """Call OpenRouter API with retry logic"""
        for attempt in range(max_retries):
            try:
                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert medical text annotator specializing in named entity recognition. You always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                # Call OpenRouter
                # OpenRouter uses OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0,  # Deterministic for medical annotations
                    extra_headers={
                        "HTTP-Referer": "https://medical-annotation-system",
                        "X-Title": "Medical Annotation System",
                    }
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
3. Look for modifier-noun combinations (e.g., "severe chest pain" can include "chest" as BODY_PART)
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
        """Parse OpenRouter response for entities"""
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

            # Parse JSON response
            response_data = json.loads(text)

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
            print(f"Error parsing OpenRouter response: {e}")
            # Try to extract JSON from text
            try:
                json_match = re.search(r'\{.*"entities".*\}', response_text, re.DOTALL)
                if json_match:
                    return self._parse_response(json_match.group())
            except:
                pass
            return []
        except Exception as e:
            print(f"Error processing OpenRouter response: {e}")
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
        return self.SUPPORTED_MODELS.get(self.model_id, self.model_id)

    def get_provider_name(self) -> str:
        """Get provider name"""
        return "OpenRouter"

    @classmethod
    def get_all_supported_models(cls) -> Dict[str, str]:
        """Get all supported models"""
        return cls.SUPPORTED_MODELS

    @classmethod
    def get_free_models(cls) -> List[str]:
        """Get list of free models (great for testing!)"""
        return [
            model_id for model_id in cls.SUPPORTED_MODELS.keys()
            if ':free' in model_id
        ]

    @classmethod
    def get_recommended_models(cls) -> List[str]:
        """Get recommended models for medical NER"""
        return [
            'openai/gpt-4o',
            'anthropic/claude-3.5-sonnet',
            'google/gemini-2.5-flash',
            'meta-llama/llama-3.3-70b-instruct',
            'anthropic/claude-sonnet-4.5',
        ]
