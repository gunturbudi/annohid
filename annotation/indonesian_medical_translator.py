"""
Indonesian Medical Term Translator with 3-Step MCN Workflow
Translates Indonesian → English → Formal Medical Term → SNOMED-CT

This service provides intelligent translation and normalization for Indonesian medical texts.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from django.conf import settings
import google.generativeai as genai
try:
    from google.generativeai.types import RequestOptions
except ImportError:
    RequestOptions = None

# Timeout for individual Gemini API calls (seconds)
GEMINI_REQUEST_TIMEOUT = 30


class IndonesianMedicalTranslator:
    """
    3-Step Medical Concept Normalization for Indonesian terms:
    1. Indonesian → English Translation
    2. English → Formal Medical Term Normalization
    3. Formal Term → SNOMED-CT Code Mapping
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.model_name = "gemini-2.5-flash"

        # Initialize Google Gemini client
        api_key = getattr(settings, 'GOOGLE_API_KEY', None)
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(
                    self.model_name,
                    generation_config={
                        "temperature": 0.3,  # Lower temperature for more consistent medical translations
                        "top_p": 0.95,
                        "top_k": 40,
                    }
                )
                self.logger.info(f"Indonesian Medical Translator initialized with {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
        else:
            self.logger.warning("GOOGLE_API_KEY not configured. Translation service disabled.")

    def is_available(self) -> bool:
        """Check if translator is available"""
        return self.client is not None

    def process_entity_complete(self, text: str, entity_type: str, context: str = "") -> Dict[str, Any]:
        """
        Complete 3-step processing for a single entity.

        Args:
            text: Indonesian medical term
            entity_type: Entity type (SYMPTOM, DISEASE, BODY_PART, etc.)
            context: Surrounding document context

        Returns:
            Dictionary with all 3 steps' results
        """
        if not self.is_available():
            return self._fallback_processing(text, entity_type)

        max_retries = 1  # Reduced from 2 to avoid compounding timeouts
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                prompt = self._create_three_step_prompt(text, entity_type, context)
                gen_kwargs = {}
                if RequestOptions:
                    gen_kwargs['request_options'] = RequestOptions(timeout=GEMINI_REQUEST_TIMEOUT)
                response = self.client.generate_content(prompt, **gen_kwargs)

                # Parse JSON response
                response_text = response.text.strip()

                # Extract JSON from markdown code blocks if present
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()

                # Detect HTML responses (LLM API error pages)
                if response_text.startswith('<') or response_text.startswith('<!'):
                    raise ValueError(f"LLM returned HTML instead of JSON (attempt {attempt + 1})")

                # Try to extract JSON object if there's extra text around it
                if not response_text.startswith('{'):
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}')
                    if json_start != -1 and json_end != -1:
                        response_text = response_text[json_start:json_end + 1]

                result = json.loads(response_text)

                # Add metadata
                result['processing_status'] = 'success'
                result['model_used'] = self.model_name

                return result

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for '{text}': {e}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))  # Backoff: 1s, 2s
                    continue
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} error for '{text}': {e}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue

        self.logger.error(f"All {max_retries + 1} attempts failed for '{text}': {last_error}")
        return self._fallback_processing(text, entity_type, error=str(last_error))

    def process_entities_batch(self, entities: List[Dict[str, Any]], context: str = "") -> List[Dict[str, Any]]:
        """
        Process multiple entities in batch (legacy: one API call per entity).
        """
        results = []

        for entity in entities:
            text = entity.get('text', '')
            entity_type = entity.get('label', 'UNKNOWN')

            processed = self.process_entity_complete(text, entity_type, context)

            result = {
                **entity,
                'translation': processed.get('step1_translation', {}),
                'normalization': processed.get('step2_normalization', {}),
                'snomed_search': processed.get('step3_snomed_search', {}),
                'processing_status': processed.get('processing_status', 'unknown')
            }

            results.append(result)

        return results

    def process_entities_batch_single_call(self, entities: List[Dict[str, Any]], context: str = "") -> List[Dict[str, Any]]:
        """
        Process ALL entities in a SINGLE Gemini API call (~90% token reduction).

        Args:
            entities: List of entity dicts with 'text' and 'label' keys
            context: Document context

        Returns:
            List of result dicts with step1_translation, step2_normalization, step3_snomed_search
        """
        if not self.is_available():
            return [self._fallback_processing(e.get('text', ''), e.get('label', 'UNKNOWN')) for e in entities]

        if len(entities) == 0:
            return []

        # Single entity → use existing method
        if len(entities) == 1:
            result = self.process_entity_complete(
                entities[0].get('text', ''),
                entities[0].get('label', 'UNKNOWN'),
                context
            )
            return [result]

        prompt = self._create_batch_prompt(entities, context)

        max_retries = 1
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                self.logger.info(f"Batch translation: {len(entities)} entities in 1 API call (attempt {attempt + 1})")
                gen_kwargs = {}
                if RequestOptions:
                    gen_kwargs['request_options'] = RequestOptions(timeout=GEMINI_REQUEST_TIMEOUT * 2)
                response = self.client.generate_content(prompt, **gen_kwargs)

                response_text = response.text.strip()

                # Extract JSON from markdown code blocks
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()

                if response_text.startswith('<') or response_text.startswith('<!'):
                    raise ValueError("LLM returned HTML instead of JSON")

                # Try to extract JSON array
                if not response_text.startswith('['):
                    arr_start = response_text.find('[')
                    arr_end = response_text.rfind(']')
                    if arr_start != -1 and arr_end != -1:
                        response_text = response_text[arr_start:arr_end + 1]

                batch_results = json.loads(response_text)

                if not isinstance(batch_results, list):
                    raise ValueError(f"Expected JSON array, got {type(batch_results).__name__}")

                # Map results back to entities (handle length mismatch)
                results = []
                for i, entity in enumerate(entities):
                    if i < len(batch_results):
                        r = batch_results[i]
                        r['processing_status'] = 'success'
                        r['model_used'] = self.model_name
                        results.append(r)
                    else:
                        # Missing result → fallback for this entity
                        self.logger.warning(f"Batch result missing for entity {i}: '{entity.get('text')}'")
                        results.append(self._fallback_processing(
                            entity.get('text', ''), entity.get('label', 'UNKNOWN'),
                            error="Missing from batch response"
                        ))

                self.logger.info(f"Batch translation successful: {len(batch_results)}/{len(entities)} entities")
                return results

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                self.logger.warning(f"Batch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue
            except Exception as e:
                last_error = e
                self.logger.warning(f"Batch attempt {attempt + 1} error: {e}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue

        # All batch attempts failed → fall back to per-entity processing
        self.logger.warning(f"Batch failed after {max_retries + 1} attempts: {last_error}. Falling back to per-entity.")
        return [
            self.process_entity_complete(e.get('text', ''), e.get('label', 'UNKNOWN'), context)
            for e in entities
        ]

    def _create_batch_prompt(self, entities: List[Dict[str, Any]], context: str) -> str:
        """Create a single prompt for processing all entities at once."""

        entity_list = "\n".join(
            f'  {i}: "{e.get("text", "")}" (type: {e.get("label", "UNKNOWN")})'
            for i, e in enumerate(entities)
        )

        return f"""You are a medical AI assistant specializing in Indonesian medical terminology.

Process ALL entities below through a 3-step pipeline. Document context: "{context[:200]}"

Entities:
{entity_list}

For EACH entity, perform:
1. Indonesian → English translation (if already English, mark is_already_english=true)
2. Normalize to formal medical terminology
3. Generate optimal SNOMED-CT search term

Return a JSON array (one object per entity, same order). Each object:
{{
  "step1_translation": {{
    "original": "<term>",
    "english": "<translation>",
    "is_already_english": false,
    "confidence": 0.95,
    "notes": "<brief note>"
  }},
  "step2_normalization": {{
    "informal": "<from step1>",
    "formal": "<formal medical term>",
    "confidence": 0.90,
    "reasoning": "<brief explanation>"
  }},
  "step3_snomed_search": {{
    "search_term": "<optimized SNOMED search term>",
    "alternative_terms": [],
    "semantic_tag": "<disorder|finding|procedure|body structure|substance>"
  }}
}}

Entity type guidelines:
- SYMPTOM: clinical findings/complaints
- DISEASE: pathological conditions/diagnoses
- BODY_PART: anatomical terminology
- MEDICATION: generic pharmaceutical names
- PROCEDURE: standard procedure terminology

Return ONLY the JSON array with {len(entities)} objects. No markdown, no extra text."""

    def _create_three_step_prompt(self, text: str, entity_type: str, context: str) -> str:
        """Create comprehensive 3-step prompt for Gemini"""

        return f"""You are a medical AI assistant specializing in Indonesian medical terminology translation and normalization.

**Task**: Process the following Indonesian medical term through a 3-step pipeline.

**Input:**
- Indonesian Term: "{text}"
- Entity Type: {entity_type}
- Document Context: "{context[:200]}..."

**Instructions:**

**STEP 1 - Indonesian to English Translation:**
- If the term is already in English, mark it as such
- Provide the most accurate medical English translation
- Consider the entity type and context
- Provide confidence score (0.0-1.0)
- List alternative translations if applicable

**STEP 2 - Normalize to Formal Medical English:**
- Convert informal/colloquial English to formal medical terminology
- Use standard medical nomenclature
- For example: "belly" → "abdominal region", "headache" → "cephalgia"
- Provide confidence score (0.0-1.0)

**STEP 3 - SNOMED-CT Search Term:**
- Provide the best search term for SNOMED-CT lookup
- Should be the most standardized medical term
- Optimize for SNOMED-CT database search

**Output Format (JSON only, no markdown):**
{{
    "step1_translation": {{
        "original": "{text}",
        "english": "<translated term>",
        "is_already_english": false,
        "confidence": 0.95,
        "alternatives": ["<alt1>", "<alt2>"],
        "notes": "<any relevant notes>"
    }},
    "step2_normalization": {{
        "informal": "<from step 1>",
        "formal": "<formal medical term>",
        "confidence": 0.90,
        "reasoning": "<brief explanation>"
    }},
    "step3_snomed_search": {{
        "search_term": "<optimized term for SNOMED search>",
        "alternative_terms": ["<alt1>", "<alt2>"],
        "semantic_tag": "<disorder|finding|procedure|body structure|etc>"
    }}
}}

**Entity Type Guidelines:**
- SYMPTOM: Focus on clinical findings and patient complaints
- DISEASE: Focus on pathological conditions and diagnoses
- BODY_PART: Use anatomical terminology
- MEDICATION: Use generic pharmaceutical names
- PROCEDURE: Use standard medical procedure terminology

Return ONLY the JSON object, no additional text or markdown formatting."""

    def _fallback_processing(self, text: str, entity_type: str, error: str = None) -> Dict[str, Any]:
        """Fallback processing when LLM is unavailable"""

        # Simple heuristic: check if term looks like English
        is_english = all(ord(c) < 128 for c in text)

        return {
            "step1_translation": {
                "original": text,
                "english": text if is_english else text,
                "is_already_english": is_english,
                "confidence": 0.50,
                "alternatives": [],
                "notes": "Fallback processing - LLM unavailable" + (f": {error}" if error else "")
            },
            "step2_normalization": {
                "informal": text,
                "formal": text,
                "confidence": 0.50,
                "reasoning": "No normalization applied (fallback mode)"
            },
            "step3_snomed_search": {
                "search_term": text,
                "alternative_terms": [],
                "semantic_tag": entity_type.lower().replace('_', ' ')
            },
            "processing_status": "fallback",
            "error": error
        }

    def translate_only(self, text: str, entity_type: str) -> Dict[str, Any]:
        """
        Perform only Step 1 - Indonesian to English translation.
        Useful for manual workflows where user wants more control.
        """
        if not self.is_available():
            return self._fallback_processing(text, entity_type)['step1_translation']

        try:
            prompt = f"""Translate this Indonesian medical term to English:

Term: "{text}"
Type: {entity_type}

Return JSON:
{{
    "original": "{text}",
    "english": "<translation>",
    "is_already_english": <boolean>,
    "confidence": <0.0-1.0>,
    "alternatives": ["<alt1>", "<alt2>"]
}}"""

            response = self.client.generate_content(prompt)
            response_text = response.text.strip()

            # Clean response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            return json.loads(response_text)

        except Exception as e:
            self.logger.error(f"Translation error for '{text}': {e}")
            return self._fallback_processing(text, entity_type)['step1_translation']


# Singleton instance
_translator_instance = None

def get_indonesian_translator() -> IndonesianMedicalTranslator:
    """Get singleton instance of Indonesian medical translator"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = IndonesianMedicalTranslator()
    return _translator_instance


# Convenience functions
def translate_and_normalize_entity(text: str, entity_type: str, context: str = "") -> Dict[str, Any]:
    """Process a single entity through all 3 steps"""
    translator = get_indonesian_translator()
    return translator.process_entity_complete(text, entity_type, context)


def translate_and_normalize_entities(entities: List[Dict[str, Any]], context: str = "") -> List[Dict[str, Any]]:
    """Process multiple entities through all 3 steps"""
    translator = get_indonesian_translator()
    return translator.process_entities_batch(entities, context)
