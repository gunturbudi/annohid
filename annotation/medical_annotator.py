#!/usr/bin/env python3
"""
Medical Annotator for Django Annotation System
=============================================

Adapted from the Flask simple_annotation_system to work with Django.
Uses LangExtract for medical entity extraction with Gemini models.
Now uses the EXACT same LangExtract approach as simple_annotation_system.
"""

import os
import textwrap
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Import real LangExtract - NEVER use mock in production!
try:
    import langextract as lx
    LANGEXTRACT_AVAILABLE = True
    print("[OK] LangExtract loaded successfully")
except ImportError:
    # CRITICAL: Mock implementation is DISABLED for security
    # Install real LangExtract: pip install langextract
    LANGEXTRACT_AVAILABLE = False
    lx = None
    print("ERROR: LangExtract library not found. Install with: pip install langextract")
    print("CRITICAL: Mock implementations are disabled for security reasons.")

# Load environment variables
load_dotenv()

# Ensure API keys are available to LangExtract (exact copy from simple_annotation_system)
google_api_key = os.getenv("GOOGLE_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set LangExtract API key (prioritize Google, then OpenAI, fallback to DeepSeek)
if google_api_key:
    # LangExtract expects LANGEXTRACT_API_KEY for Gemini models
    os.environ["LANGEXTRACT_API_KEY"] = google_api_key
if openai_api_key:
    # Set OPENAI_API_KEY for OpenAI models
    os.environ["OPENAI_API_KEY"] = openai_api_key
if deepseek_api_key:
    # Set DEEPSEEK_API_KEY for DeepSeek models
    os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key


class MedicalAnnotator:
    """
    Medical text annotator using LangExtract for entity extraction
    Adapted for Django annotation system
    """

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        """
        Initialize the medical annotator

        Args:
            model_id: The LLM model to use for extraction
        """
        self.model_id = model_id
        self.prompt = self._create_medical_prompt()
        self.examples = self._create_medical_examples()

    def _create_medical_prompt(self) -> str:
        """Create the prompt for medical entity extraction"""
        return textwrap.dedent("""\
        You are an expert medical text annotator. Extract ALL medical entities from patient text (English or Indonesian).

        Entity schema (use these 5 specific types):
        - SYMPTOM: subjective complaints, feelings, sensations
          Examples: chest pain, headache, nausea, shortness of breath, dizziness, fatigue, mual, nyeri, pusing, sakit kepala
        - DISEASE: specific diseases, conditions, syndromes
          Examples: diabetes, hypertension, gastritis, migraine, asthma, maag, hipertensi, migrain
        - BODY_PART: anatomical structures, organs, body regions
          Examples: chest, head, stomach, heart, lungs, arm, leg, kepala, perut, dada, jantung
        - MEDICATION: drugs, medicines, treatments
          Examples: aspirin, paracetamol, insulin, antibiotics, metformin, lisinopril, antasida
        - PROCEDURE: medical procedures, examinations, therapies
          Examples: surgery, X-ray, ultrasound, blood test, therapy, operasi, USG, rontgen, pemeriksaan

        CRITICAL - Extract overlapping entities for maximum recall:
        - Extract BOTH atomic entities AND compound phrases
        - "severe chest pain" should yield: "chest" (BODY_PART), "chest pain" (SYMPTOM), "severe chest pain" (SYMPTOM)
        - "sakit kepala hebat" should yield: "kepala" (BODY_PART), "sakit kepala" (SYMPTOM), "sakit kepala hebat" (SYMPTOM)
        - "metformin 500mg" should yield: "metformin" (MEDICATION), "metformin 500mg" (MEDICATION with dosage)
        - Include severity modifiers, dosages, and anatomical qualifiers as part of larger entity spans

        Medical annotation guidelines:
        - Extract ALL clinically relevant medical concepts, even if they overlap
        - Focus on maximizing recall - it's better to extract more entities than to miss important ones
        - Extract complete medication names with dosages when mentioned
        - Don't extract common non-medical words like "patient", "has", "taking"
        - Be exhaustive in identifying body parts, symptoms, diseases, medications, and procedures

        Use exact text from the input. Do not paraphrase.
        Provide meaningful medical attributes for each entity to add clinical context.
        """)

    def _create_medical_examples(self) -> List:
        """Create few-shot examples for medical entity extraction with overlapping entities"""
        if not LANGEXTRACT_AVAILABLE or lx is None:
            return []

        examples = [
            # Example 1: Indonesian with overlapping entities (body part + symptom)
            lx.data.ExampleData(
                text="Saya mengalami sakit kepala hebat selama 3 hari.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="BODY_PART",
                        extraction_text="kepala",
                        attributes={
                            "anatomical_region": "head",
                            "system": "nervous"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="sakit kepala",
                        attributes={
                            "type": "headache",
                            "body_part": "head"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="sakit kepala hebat",
                        attributes={
                            "type": "headache",
                            "body_part": "head",
                            "severity": "severe"
                        }
                    )
                ]
            ),
            # Example 2: Multiple entities with gastrointestinal overlaps
            lx.data.ExampleData(
                text="Dok, saya sering mual dan kembung di perut sejak 2 minggu, mungkin maag.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="mual",
                        attributes={
                            "type": "nausea",
                            "system": "gastrointestinal"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="kembung",
                        attributes={
                            "type": "bloating",
                            "system": "gastrointestinal"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="kembung di perut",
                        attributes={
                            "type": "abdominal_bloating",
                            "system": "gastrointestinal",
                            "location": "abdomen"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="BODY_PART",
                        extraction_text="perut",
                        attributes={
                            "anatomical_region": "abdomen",
                            "system": "gastrointestinal"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="DISEASE",
                        extraction_text="maag",
                        attributes={
                            "condition": "gastritis",
                            "certainty": "possible"
                        }
                    )
                ]
            ),
            # Example 3: Medication with dosage (overlapping medication spans)
            lx.data.ExampleData(
                text="Minum paracetamol 500mg 3x sehari untuk demam tinggi.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="MEDICATION",
                        extraction_text="paracetamol",
                        attributes={
                            "generic_name": "acetaminophen",
                            "category": "analgesic_antipyretic"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="MEDICATION",
                        extraction_text="paracetamol 500mg",
                        attributes={
                            "generic_name": "acetaminophen",
                            "dosage": "500mg",
                            "category": "analgesic_antipyretic"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="demam",
                        attributes={
                            "type": "fever"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="demam tinggi",
                        attributes={
                            "type": "fever",
                            "severity": "high"
                        }
                    )
                ]
            ),
            # Example 4: English with overlapping body part + symptom
            lx.data.ExampleData(
                text="Patient has severe chest pain and shortness of breath. Takes aspirin 100mg daily.",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="BODY_PART",
                        extraction_text="chest",
                        attributes={
                            "anatomical_region": "thorax",
                            "system": "cardiovascular"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="chest pain",
                        attributes={
                            "type": "pain",
                            "location": "chest"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="severe chest pain",
                        attributes={
                            "type": "pain",
                            "location": "chest",
                            "severity": "severe"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="SYMPTOM",
                        extraction_text="shortness of breath",
                        attributes={
                            "type": "dyspnea",
                            "system": "respiratory"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="MEDICATION",
                        extraction_text="aspirin",
                        attributes={
                            "drug_class": "antiplatelet"
                        }
                    ),
                    lx.data.Extraction(
                        extraction_class="MEDICATION",
                        extraction_text="aspirin 100mg",
                        attributes={
                            "drug_class": "antiplatelet",
                            "dosage": "100mg"
                        }
                    )
                ]
            )
        ]
        return examples

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text

        Args:
            text: Medical text to annotate

        Returns:
            List of extracted entities in Django-compatible format

        Raises:
            Exception: Re-raises exceptions with user-friendly messages
        """
        try:
            # Perform extraction using LangExtract
            result = lx.extract(
                text_or_documents=text,
                prompt_description=self.prompt,
                examples=self.examples,
                model_id=self.model_id
            )

            # Convert to Django-compatible format
            # Allow overlapping entities for better recall
            entities = []

            # Track (text, start_pos) tuples to avoid exact duplicates only
            seen_entities = set()

            # Track how many times each text has been used to handle multiple occurrences
            text_occurrence_count = {}

            # Diagnostic counters
            total_extractions = len(result.extractions)
            skipped_not_found = 0
            skipped_duplicates = 0

            for extraction in result.extractions:
                search_text = extraction.extraction_text
                search_text_lower = search_text.lower()

                # Track which occurrence of this text we're looking for
                occurrence_index = text_occurrence_count.get(search_text_lower, 0)
                text_occurrence_count[search_text_lower] = occurrence_index + 1

                # Find the Nth occurrence of this text
                search_start = 0
                start_pos = -1
                current_occurrence = 0

                while True:
                    pos = text.lower().find(search_text_lower, search_start)
                    if pos == -1:
                        break

                    if current_occurrence == occurrence_index:
                        start_pos = pos
                        break

                    current_occurrence += 1
                    search_start = pos + 1

                # If specific occurrence not found, try to find any valid position
                if start_pos == -1:
                    # Fall back to first occurrence
                    start_pos = text.lower().find(search_text_lower)

                if start_pos == -1:
                    skipped_not_found += 1
                    print(f"[LangExtract] Entity not found in text: '{extraction.extraction_text}'")
                    continue

                end_pos = start_pos + len(extraction.extraction_text)

                # Create unique key for exact duplicate detection (same text at same position)
                entity_key = (search_text_lower, start_pos)

                if entity_key in seen_entities:
                    skipped_duplicates += 1
                    continue

                seen_entities.add(entity_key)

                entity = {
                    'text': extraction.extraction_text,
                    'label': extraction.extraction_class,
                    'start': start_pos,
                    'end': end_pos,
                    'attributes': extraction.attributes if hasattr(extraction, 'attributes') else {},
                    'confidence': extraction.attributes.get('confidence', 0.85) if hasattr(extraction, 'attributes') else 0.85
                }
                entities.append(entity)

            # Log diagnostic info
            if skipped_not_found > 0 or skipped_duplicates > 0:
                print(f"[LangExtract] Extraction stats: {total_extractions} total, "
                      f"{len(entities)} accepted, {skipped_not_found} not found, "
                      f"{skipped_duplicates} duplicates")

            return entities

        except Exception as e:
            error_str = str(e).lower()

            # Provide user-friendly error messages for common API issues
            if '429' in error_str or 'resource_exhausted' in error_str or 'quota' in error_str:
                user_msg = "The Gemini API quota has been exhausted. Please try again later or check your Google Cloud quota limits."
            elif '401' in error_str or 'unauthorized' in error_str or 'api key' in error_str:
                user_msg = "Invalid or missing API key. Please check your GOOGLE_API_KEY configuration."
            elif '403' in error_str or 'forbidden' in error_str:
                user_msg = "Access denied. Please verify your API key has the correct permissions."
            elif 'timeout' in error_str or 'timed out' in error_str:
                user_msg = "Request timed out. The API service may be slow or unavailable. Please try again."
            elif 'connection' in error_str or 'network' in error_str:
                user_msg = "Network error. Please check your internet connection and try again."
            elif '500' in error_str or '503' in error_str:
                user_msg = "The Gemini API service is currently unavailable. Please try again later."
            else:
                user_msg = f"Failed to extract entities: {str(e)}"

            print(f"Error extracting entities: {e}")
            raise Exception(user_msg)

    def annotate_document(self, document_text: str) -> Dict[str, Any]:
        """
        Annotate a single document and return structured result

        Args:
            document_text: Text content to annotate

        Returns:
            Annotation result with entities and metadata
        """
        entities = self.extract_entities(document_text)

        # Calculate statistics
        entity_stats = {}
        for entity in entities:
            label = entity['label']
            entity_stats[label] = entity_stats.get(label, 0) + 1

        return {
            'text': document_text,
            'entities': entities,
            'model': self.model_id,
            'total_entities': len(entities),
            'entity_types': list(entity_stats.keys()),
            'entity_stats': entity_stats,
            'success': True
        }

    def get_available_models(self) -> List[str]:
        """Get list of available models for annotation"""
        return [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-5-mini-2025-08-07",
            "deepseek-chat"
        ]

    def validate_api_setup(self) -> Dict[str, Any]:
        """Validate API key setup - adapted from simple_annotation_system"""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        langextract_api_key = os.getenv("LANGEXTRACT_API_KEY")

        # Check if we have the API keys needed for real annotation
        has_api_keys = bool(google_api_key) or bool(deepseek_api_key) or bool(openai_api_key)

        return {
            'langextract_available': LANGEXTRACT_AVAILABLE,
            'google_api_key_set': bool(google_api_key),
            'deepseek_api_key_set': bool(deepseek_api_key),
            'openai_api_key_set': bool(openai_api_key),
            'langextract_api_key_set': bool(langextract_api_key),
            'ready_for_annotation': has_api_keys  # Ready if we have API keys, regardless of LangExtract availability
        }


def create_medical_annotator(model_id: str = "gemini-2.5-flash") -> MedicalAnnotator:
    """Factory function to create a medical annotator instance"""
    return MedicalAnnotator(model_id=model_id)