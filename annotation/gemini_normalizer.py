"""
Gemini 2.5 Flash Medical Term Translation & Normalization Service

Translates Indonesian medical entities to English and normalizes them to formal
medical terminology in a SINGLE batch Gemini call per document (for performance).
Falls back to dictionary-based translation if Gemini is unavailable.
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional

from django.conf import settings

try:
    from google.generativeai.types import RequestOptions
except ImportError:
    RequestOptions = None

# Timeout for individual Gemini API calls (seconds)
GEMINI_REQUEST_TIMEOUT = 30

logger = logging.getLogger(__name__)


class GeminiMedicalNormalizer:
    """
    Medical term normalizer using Gemini 2.5 Flash model.
    Performs Indonesian→English translation + formal medical normalization
    in a single batch LLM call for all entities in a document.
    """

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.client = None

        # Initialize Gemini client
        api_key = getattr(settings, 'GOOGLE_API_KEY', None)
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(
                    self.model_name,
                    generation_config={
                        "temperature": 0.2,
                        "top_p": 0.95,
                        "top_k": 40,
                    }
                )
                logger.info(f"GeminiMedicalNormalizer initialized with {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
        else:
            logger.warning("GOOGLE_API_KEY not configured. Using dictionary fallback.")

        # Fallback dictionary: Indonesian → English
        self.indo_to_english = {
            'sakit kepala': 'headache', 'sakit kepala hebat': 'severe headache',
            'demam': 'fever', 'demam tinggi': 'high fever',
            'batuk': 'cough', 'batuk berdahak': 'productive cough',
            'batuk kering': 'dry cough',
            'nyeri': 'pain', 'nyeri dada': 'chest pain',
            'nyeri perut': 'abdominal pain', 'nyeri sendi': 'joint pain',
            'mual': 'nausea', 'muntah': 'vomiting',
            'diare': 'diarrhea', 'sembelit': 'constipation',
            'pusing': 'dizziness', 'sesak napas': 'shortness of breath',
            'sesak': 'dyspnea', 'lelah': 'fatigue', 'lemas': 'weakness',
            'pandangan buram': 'blurred vision',
            'penglihatan buram': 'blurred vision',
            'haus berlebihan': 'excessive thirst',
            'sering buang air kecil': 'frequent urination',
            'sering kencing': 'frequent urination',
            'sering pipis': 'frequent urination',
            'kesemutan': 'tingling', 'mati rasa': 'numbness',
            'berat badan turun': 'weight loss',
            'gula darah tinggi': 'hyperglycemia',
            'gula darah': 'blood sugar',
            'tekanan darah tinggi': 'high blood pressure',
            'tekanan darah rendah': 'low blood pressure',
            'bengkak': 'swelling', 'gatal': 'itching', 'ruam': 'rash',
            'memar': 'bruising', 'perdarahan': 'bleeding',
            'keringat dingin': 'cold sweat', 'menggigil': 'chills',
            # Diseases
            'diabetes melitus': 'diabetes mellitus',
            'kencing manis': 'diabetes mellitus',
            'hipertensi': 'hypertension', 'asma': 'asthma',
            'gastritis': 'gastritis', 'maag': 'gastric ulcer',
            'tukak lambung': 'gastric ulcer', 'radang paru': 'pneumonia',
            'bronkitis': 'bronchitis', 'stroke': 'stroke',
            'serangan jantung': 'myocardial infarction',
            'infeksi': 'infection', 'alergi': 'allergy',
            'anemia': 'anemia', 'osteoporosis': 'osteoporosis',
            # Body parts
            'kepala': 'head', 'dada': 'chest', 'perut': 'abdomen',
            'lambung': 'stomach', 'jantung': 'heart',
            'paru-paru': 'lung', 'paru': 'lung',
            'ginjal': 'kidney', 'hati': 'liver', 'otak': 'brain',
            'tulang': 'bone', 'sendi': 'joint',
            'mata': 'eye', 'telinga': 'ear', 'hidung': 'nose',
            'mulut': 'mouth', 'tenggorokan': 'throat',
            'lengan': 'arm', 'kaki': 'leg', 'tangan': 'hand',
            'punggung': 'back', 'leher': 'neck', 'kulit': 'skin',
            # Medications
            'parasetamol': 'paracetamol', 'ibuprofen': 'ibuprofen',
            'aspirin': 'aspirin', 'insulin': 'insulin',
            'antibiotik': 'antibiotic', 'antasida': 'antacid',
            'obat': 'medication', 'obat nyeri': 'analgesic',
            'obat batuk': 'cough medicine', 'obat demam': 'antipyretic',
            'metformin': 'metformin', 'amoksisilin': 'amoxicillin',
            # Procedures
            'operasi': 'surgery', 'pemeriksaan': 'examination',
            'pemeriksaan darah': 'blood test', 'tes darah': 'blood test',
            'rontgen': 'X-ray', 'USG': 'ultrasound',
            'terapi': 'therapy', 'rawat inap': 'hospitalization',
            'cek gula darah': 'blood sugar test',
        }

        # English → formal medical term normalization
        self.english_to_formal = {
            'headache': 'cephalgia', 'severe headache': 'severe cephalgia',
            'stomach ache': 'gastralgia', 'stomach pain': 'gastralgia',
            'abdominal pain': 'abdominal pain', 'back pain': 'dorsalgia',
            'chest pain': 'chest pain', 'joint pain': 'arthralgia',
            'sore throat': 'pharyngitis', 'runny nose': 'rhinorrhea',
            'stuffy nose': 'nasal congestion',
            'shortness of breath': 'dyspnea', 'difficulty breathing': 'dyspnea',
            'vomiting': 'emesis', 'constipation': 'obstipation',
            'fever': 'pyrexia', 'high fever': 'hyperpyrexia',
            'chills': 'rigors', 'fatigue': 'asthenia', 'weakness': 'asthenia',
            'dizziness': 'vertigo', 'rash': 'cutaneous eruption',
            'itching': 'pruritus', 'swelling': 'edema',
            'bruising': 'ecchymosis', 'bleeding': 'hemorrhage',
            'high blood pressure': 'hypertension',
            'low blood pressure': 'hypotension',
            'heart attack': 'myocardial infarction',
            'stroke': 'cerebrovascular accident',
        }

    def is_available(self) -> bool:
        return self.client is not None

    def normalize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize multiple entities. Uses a single Gemini batch call for all
        entities, falling back to dictionary if Gemini is unavailable.
        """
        if not entities:
            return entities

        if self.is_available():
            try:
                return self._normalize_with_gemini_batch(entities)
            except Exception as e:
                logger.error(f"Gemini batch normalization failed, using fallback: {e}")
                return self._normalize_with_dictionary(entities)
        else:
            return self._normalize_with_dictionary(entities)

    def _normalize_with_gemini_batch(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Single Gemini call to translate + normalize ALL entities at once.
        Much faster than per-entity calls.
        """
        # Build entity list for the prompt
        entity_list = []
        for i, entity in enumerate(entities):
            text = entity.get('text', '')
            label = entity.get('label', entity.get('entity_type', 'UNKNOWN'))
            entity_list.append(f'{i}: "{text}" [{label}]')

        entities_text = '\n'.join(entity_list)

        prompt = f"""You are a medical terminology expert specializing in Indonesian medical text.

For each entity below, provide:
1. **english**: The English translation (if already English, keep as-is)
2. **normalized**: The formal/standard medical term for SNOMED-CT lookup

Entities:
{entities_text}

Return ONLY a JSON array, one object per entity, in the SAME order:
[
  {{"index": 0, "english": "<english translation>", "normalized": "<formal medical term>"}},
  ...
]

Rules:
- If the term is already in English, set english to that term
- For normalized, use standard medical nomenclature suitable for SNOMED-CT search
- Examples: "sakit kepala" → english: "headache", normalized: "headache"
- Examples: "demam tinggi" → english: "high fever", normalized: "pyrexia"
- Examples: "jantung" → english: "heart", normalized: "heart"
- Examples: "diabetes melitus" → english: "diabetes mellitus", normalized: "diabetes mellitus"
- Return ONLY the JSON array, no markdown formatting or extra text."""

        max_retries = 2
        gemini_results = None

        for attempt in range(max_retries + 1):
            try:
                request_opts = {}
                if RequestOptions:
                    request_opts['request_options'] = RequestOptions(timeout=GEMINI_REQUEST_TIMEOUT)
                response = self.client.generate_content(prompt, **request_opts)
                response_text = response.text.strip()

                # Clean response - extract JSON
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()

                # Detect HTML responses (API error pages)
                if response_text.startswith('<') or response_text.startswith('<!'):
                    raise ValueError(f"Gemini returned HTML instead of JSON (attempt {attempt + 1})")

                # Try to extract JSON array if there's extra text
                if not response_text.startswith('['):
                    arr_start = response_text.find('[')
                    arr_end = response_text.rfind(']')
                    if arr_start != -1 and arr_end != -1:
                        response_text = response_text[arr_start:arr_end + 1]

                gemini_results = json.loads(response_text)
                break  # Success

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Gemini batch normalization attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue
                raise  # Let outer handler catch and fallback to dictionary

            except Exception as e:
                logger.warning(f"Gemini batch normalization attempt {attempt + 1}/{max_retries + 1} error: {e}")
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue
                raise

        # Merge Gemini results back into entities
        result_map = {}
        for item in gemini_results:
            idx = item.get('index', -1)
            result_map[idx] = item

        normalized_entities = []
        for i, entity in enumerate(entities):
            gemini_data = result_map.get(i, {})
            english_term = gemini_data.get('english', entity.get('text', ''))
            normalized_term = gemini_data.get('normalized', english_term)

            normalized_entity = {
                **entity,
                'english_term': english_term,
                'normalized_term': normalized_term,
                'normalization_source': self.model_name,
            }
            normalized_entities.append(normalized_entity)

        return normalized_entities

    def _normalize_with_dictionary(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback: dictionary-based Indonesian→English + normalization."""
        normalized_entities = []

        for entity in entities:
            text = entity.get('text', '').strip()
            text_lower = text.lower()

            # Step 1: Indonesian → English
            english_term = self._dict_translate(text_lower)

            # Step 2: English → formal medical term
            normalized_term = self.english_to_formal.get(
                english_term.lower(), english_term
            )

            normalized_entity = {
                **entity,
                'english_term': english_term,
                'normalized_term': normalized_term,
                'normalization_source': 'dictionary_fallback',
            }
            normalized_entities.append(normalized_entity)

        return normalized_entities

    def _dict_translate(self, text_lower: str) -> str:
        """Translate Indonesian text to English using dictionary."""
        # Exact match
        if text_lower in self.indo_to_english:
            return self.indo_to_english[text_lower]

        # Partial match (longest match first)
        best_match = ''
        best_english = text_lower
        for indo, eng in self.indo_to_english.items():
            if indo in text_lower and len(indo) > len(best_match):
                best_match = indo
                best_english = eng

        if best_match:
            return best_english

        # Assume already English if no Indonesian match found
        return text_lower


# Singleton instance
_normalizer_instance = None


def get_medical_normalizer() -> GeminiMedicalNormalizer:
    """Get singleton instance of medical normalizer."""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = GeminiMedicalNormalizer()
    return _normalizer_instance


def normalize_medical_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize multiple medical entities (convenience function)."""
    normalizer = get_medical_normalizer()
    return normalizer.normalize_entities(entities)
