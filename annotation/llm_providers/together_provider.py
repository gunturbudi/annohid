"""
Together AI Provider for NER
Supports 100+ open-source models via Together AI API
"""

import os
import json
import logging
import time
from typing import List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseLLMProvider, Entity, NERResult


logger = logging.getLogger(__name__)


class TogetherNERProvider(BaseLLMProvider):
    """
    Together AI provider using OpenAI-compatible API
    Supports 100+ open-source models with credits-based pricing
    """

    # Popular models available on Together AI
    SUPPORTED_MODELS = {
        # Meta Llama models (FREE models marked with 🆓)
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': 'Llama 3.1 8B Turbo',
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo': 'Llama 3.1 70B Turbo',
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo': 'Llama 3.1 405B Turbo',
        'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'Llama 3.3 70B Turbo',
        'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free': 'Llama 3.3 70B Turbo 🆓 FREE',
        'meta-llama/Llama-3.2-3B-Instruct-Turbo': 'Llama 3.2 3B Turbo',
        'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo': 'Llama 3.2 11B Vision',
        'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo': 'Llama 3.2 90B Vision',

        # DeepSeek models (reasoning)
        'deepseek-ai/DeepSeek-R1': 'DeepSeek R1 (Reasoning)',
        'deepseek-ai/DeepSeek-R1-Distill-Llama-70B': 'DeepSeek R1 Distill 70B',
        'deepseek-ai/DeepSeek-V3': 'DeepSeek V3',

        # Qwen models
        'Qwen/Qwen2.5-72B-Instruct-Turbo': 'Qwen 2.5 72B Turbo',
        'Qwen/Qwen2.5-7B-Instruct-Turbo': 'Qwen 2.5 7B Turbo',
        'Qwen/QwQ-32B-Preview': 'QwQ 32B (Reasoning)',
        'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8': 'Qwen 3 Coder 480B',

        # Mistral models
        'mistralai/Mixtral-8x7B-Instruct-v0.1': 'Mixtral 8x7B',
        'mistralai/Mistral-7B-Instruct-v0.1': 'Mistral 7B',
        'mistralai/Mistral-7B-Instruct-v0.3': 'Mistral 7B v0.3',

        # Google models
        'google/gemma-2-9b-it': 'Gemma 2 9B',
        'google/gemma-2-27b-it': 'Gemma 2 27B',

        # OpenAI GPT models via Together
        'openai/gpt-oss-20b': 'GPT-OSS 20B',
        'openai/gpt-oss-120b': 'GPT-OSS 120B',

        # Other popular models
        'databricks/dbrx-instruct': 'DBRX Instruct',
        'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO': 'Nous Hermes 2 Mixtral',
        'Gryphe/MythoMax-L2-13b': 'MythoMax L2 13B',
        'upstage/SOLAR-10.7B-Instruct-v1.0': 'SOLAR 10.7B',
    }

    def __init__(self, model_id: str, api_key: Optional[str] = None):
        """Initialize Together AI provider"""
        super().__init__(model_id, api_key)
        self.client = None
        self.base_url = "https://api.together.xyz/v1"

    def get_provider_name(self) -> str:
        return "Together AI"

    def get_model_display_name(self) -> str:
        return self.SUPPORTED_MODELS.get(self.model_id, self.model_id)

    def initialize(self) -> bool:
        """Initialize Together AI client"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI library not available (required for Together AI). Install: pip install openai>=1.0.0")
            return False

        try:
            # Check for API key
            api_key = self.api_key or os.getenv('TOGETHER_API_KEY')

            if not api_key:
                logger.warning("Together AI API key not found")
                return False

            # Initialize OpenAI client with Together base URL
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.base_url
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Together AI provider: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Together AI is available"""
        if not OPENAI_AVAILABLE:
            return False

        if not self.client:
            self.initialize()
        return self.client is not None

    def annotate_text(
        self,
        text: str,
        enable_overlapping: bool = True,
        confidence_threshold: float = 0.3,
        max_retries: int = 3
    ) -> NERResult:
        """
        Perform NER on text using Together AI model
        """
        if not self.client:
            return NERResult(
                entities=[],
                processing_time=0.0,
                model_used=self.model_id,
                success=False,
                error_message="Together AI client not initialized"
            )

        prompt = self._build_prompt(text, enable_overlapping)
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical NER expert. Extract medical entities and return them as JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )

                response_text = response.choices[0].message.content
                entities = self._parse_response(response_text, text)

                # Filter by confidence
                entities = [e for e in entities if e.confidence >= confidence_threshold]

                processing_time = time.time() - start_time

                return NERResult(
                    entities=entities,
                    processing_time=processing_time,
                    model_used=self.model_id,
                    success=True
                )

            except Exception as e:
                logger.warning(f"Together AI API attempt {attempt + 1} failed: {e}")

                if attempt == max_retries - 1:
                    processing_time = time.time() - start_time
                    return NERResult(
                        entities=[],
                        processing_time=processing_time,
                        model_used=self.model_id,
                        success=False,
                        error_message=f"All {max_retries} attempts failed: {str(e)}"
                    )

        processing_time = time.time() - start_time
        return NERResult(
            entities=[],
            processing_time=processing_time,
            model_used=self.model_id,
            success=False,
            error_message="Max retries exceeded"
        )

    def _build_prompt(self, text: str, enable_overlapping: bool) -> str:
        """Build prompt for NER task"""
        overlap_instruction = ""
        if enable_overlapping:
            overlap_instruction = """
IMPORTANT: Entities CAN overlap. For example:
- "type 2 diabetes mellitus" contains both "diabetes mellitus" (condition) and "type 2 diabetes mellitus" (condition)
- "left ventricular hypertrophy" contains both "left ventricular" (anatomy) and "ventricular hypertrophy" (condition)

Extract ALL entities, including overlapping ones."""

        prompt = f"""Extract ALL medical entities from the following text.

Medical entity types to extract:
1. CONDITION: Diseases, disorders, symptoms, injuries, abnormalities
2. MEDICATION: Drugs, treatments, vaccines, supplements
3. ANATOMY: Body parts, organs, tissues, anatomical locations
4. PROCEDURE: Medical procedures, tests, surgeries, examinations
5. MEASUREMENT: Lab values, vital signs, measurements with units

{overlap_instruction}

Text to analyze:
{text}

Return the results in this EXACT JSON format:
{{
  "entities": [
    {{
      "text": "exact entity text from original",
      "label": "CONDITION|MEDICATION|ANATOMY|PROCEDURE|MEASUREMENT",
      "start_char": integer position,
      "end_char": integer position,
      "confidence": float between 0.0-1.0
    }}
  ]
}}

Rules:
1. Extract the EXACT text as it appears (preserve spelling, capitalization)
2. Provide accurate character positions (start_char, end_char)
3. Assign appropriate confidence scores
4. Return valid JSON only (no markdown, no explanations)"""

        return prompt

    def _parse_response(self, response_text: str, original_text: str) -> List[Entity]:
        """Parse Together AI response and extract entities"""
        entities = []

        try:
            # Clean markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            data = json.loads(response_text)

            if "entities" not in data:
                logger.warning("No 'entities' key in response")
                return entities

            for entity_data in data["entities"]:
                entity = Entity(
                    text=entity_data.get("text", ""),
                    label=entity_data.get("label", "UNKNOWN"),
                    start_offset=int(entity_data.get("start_char", 0)),
                    end_offset=int(entity_data.get("end_char", 0)),
                    confidence=float(entity_data.get("confidence", 0.5))
                )

                # Validate entity
                if self.validate_entity(entity, original_text):
                    entities.append(entity)
                else:
                    logger.debug(f"Invalid entity filtered out: {entity.text}")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}\nResponse: {response_text}")
        except Exception as e:
            logger.error(f"Error parsing Together AI response: {e}")

        return entities

    @classmethod
    def get_all_supported_models(cls) -> dict:
        """Get all supported models"""
        return cls.SUPPORTED_MODELS.copy()

    @classmethod
    def get_popular_models(cls) -> List[str]:
        """Get list of popular/recommended models"""
        return [
            'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            'deepseek-ai/DeepSeek-V3',
            'Qwen/Qwen2.5-72B-Instruct-Turbo',
            'mistralai/Mixtral-8x7B-Instruct-v0.1',
        ]

    @classmethod
    def get_fast_models(cls) -> List[str]:
        """Get list of fast/small models for quick testing"""
        return [
            'meta-llama/Llama-3.2-3B-Instruct-Turbo',
            'Qwen/Qwen2.5-7B-Instruct-Turbo',
            'mistralai/Mistral-7B-Instruct-v0.3',
        ]

    @classmethod
    def get_reasoning_models(cls) -> List[str]:
        """Get list of reasoning models (better for complex NER)"""
        return [
            'deepseek-ai/DeepSeek-R1',
            'Qwen/QwQ-32B-Preview',
            'openai/gpt-oss-120b',
        ]
