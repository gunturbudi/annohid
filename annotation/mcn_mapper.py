"""
Medical Concept Normalization (MCN) Mapper Service
Automatically maps Indonesian medical terms to SNOMED-CT concepts using Google Gemini
"""

import json
import logging
import re
from typing import List, Dict, Any
from django.conf import settings
import google.generativeai as genai


class MCNMapper:
    """
    Service to automatically extract and map Indonesian medical terms to SNOMED-CT
    Uses Google Gemini with specialized prompt for medical concept normalization
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.model_name = "gemini-2.5-flash"  # Using latest Gemini model

        # Initialize Google Gemini client if API key is available
        api_key = getattr(settings, 'GOOGLE_API_KEY', None)
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model_name)
                self.logger.info(f"Google Gemini MCN Mapper initialized with model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None
        else:
            self.logger.warning("GOOGLE_API_KEY not configured. MCN mapping will be disabled.")

    def is_available(self) -> bool:
        """Check if MCN mapper is available (API key configured)"""
        return self.client is not None

    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract potential medical terms from Indonesian text
        Uses simple heuristics and pattern matching

        Args:
            text: Indonesian medical text

        Returns:
            List of potential medical terms
        """
        # Common Indonesian medical term patterns
        medical_patterns = [
            r'\b(?:sakit|nyeri|rasa)\s+\w+',  # Pain/ache patterns
            r'\b(?:demam|panas|pusing)\b',     # Fever/dizziness
            r'\b(?:batuk|pilek|flu|mual|muntah|diare)\b',  # Common symptoms
            r'\b(?:diabetes|hipertensi|asma|maag|gastritis)\b',  # Diseases
            r'\b(?:gatal|bengkak|lemas|lelah|capek)\b',  # General symptoms
            r'\b(?:kepala|dada|perut|lambung|jantung)\b',  # Body parts
            r'\b(?:gatal-gatal|sakit\s+gigi|pusing\s+kepala)\b',  # Compound terms
        ]

        terms = set()
        text_lower = text.lower()

        # Extract using patterns
        for pattern in medical_patterns:
            matches = re.findall(pattern, text_lower)
            terms.update(matches)

        # Also split text and look for common medical words
        words = text_lower.split()
        medical_keywords = {
            'sakit', 'nyeri', 'demam', 'batuk', 'pusing', 'mual', 'muntah',
            'diare', 'lemas', 'lelah', 'capek', 'gatal', 'bengkak',
            'diabetes', 'hipertensi', 'asma', 'maag', 'gastritis',
            'kepala', 'dada', 'perut', 'jantung', 'gatal-gatal'
        }

        # Look for compound terms (two consecutive medical words)
        for i in range(len(words) - 1):
            if words[i] in medical_keywords or words[i+1] in medical_keywords:
                compound = f"{words[i]} {words[i+1]}"
                terms.add(compound)

        for word in words:
            if word in medical_keywords:
                terms.add(word)

        return sorted(list(terms))[:10]  # Limit to 10 terms max per document

    def map_terms_to_snomed(self, terms: List[str], context: str = "") -> Dict[str, Any]:
        """
        Map Indonesian medical terms to SNOMED-CT using Google Gemini

        Args:
            terms: List of Indonesian medical terms
            context: Optional context from the original medical text

        Returns:
            Dictionary with mapping results and metadata
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'MCN mapper not available (API key not configured)',
                'mappings': []
            }

        if not terms:
            return {
                'success': True,
                'mappings': [],
                'total_terms': 0
            }

        try:
            # Build the MCN mapping prompt
            prompt = self._build_mcn_prompt(terms, context)

            # Call Google Gemini API
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,  # Deterministic for medical terminology
                    top_p=1,
                    top_k=1,
                    max_output_tokens=4096,
                )
            )

            # Extract JSON response
            response_text = response.text
            mappings = self._parse_mcn_response(response_text)

            return {
                'success': True,
                'mappings': mappings,
                'total_terms': len(terms),
                'mapped_terms': len(mappings),
                'model_used': self.model_name,
                'context': context[:200] if context else ""
            }

        except Exception as e:
            self.logger.error(f"Error mapping terms to SNOMED: {e}")
            return {
                'success': False,
                'error': str(e),
                'mappings': []
            }

    def _build_mcn_prompt(self, terms: List[str], context: str = "") -> str:
        """
        Build the MCN mapping prompt for Gemini
        Uses the specialized medical terminology mapping instructions
        """
        terms_json = json.dumps(terms, ensure_ascii=False)

        context_section = ""
        if context:
            context_section = f"\n\n*Context from original text:*\n{context[:500]}"

        prompt = f"""You are an expert health informatics and medical terminology assistant, specializing in mapping clinical language to international standards.

*Your Task:*
Take a list of health-related terms in Indonesian (which may be colloquial, technical, or abbreviations) and map each term to the most appropriate SNOMED-CT concept.

*Output Format:*
Always provide your answer in a JSON format. Create an array of objects, where each object represents one mapped term and has the following keys:
1. input_term_id: The original Indonesian term I provided.
2. preferred_term_en: Candidate list of the SNOMED-CT Preferred Term in English.
3. notes: A brief explanation if the term is ambiguous, too general, or a colloquial term with no direct match. If no match is found, set preferred_term_en to an empty list and explain in notes.

*Important Rules:*
1. *Specificity:* Always find the most specific and clinically relevant concept.
2. *Ambiguity:* If a term can mean multiple things (e.g., "sakit" can mean "pain" or "illness"), choose the most common clinical meaning. For example, "sakit kepala" is clearly Headache, but "sakit" by itself is most likely Pain. List all possible terms and use the notes field to explain this.
3. *Colloquial Terms:* For colloquial terms that do not exist in SNOMED-CT (e.g., "masuk angin"), use notes to explain it's a cultural term, then (if possible) list related mappable symptoms (e.g., Flatulence or Common cold) in preferred_term_en.

*Example:*
* *My Input:* ["pusing kepala", "sakit perut", "masuk angin"]
* *Your Output (Example):*
```json
[
  {{
    "input_term_id": "pusing kepala",
    "preferred_term_en": ["Headache", "Dizziness"],
    "notes": "Could refer to either headache (pain in head) or dizziness (vertigo). Most commonly refers to headache in Indonesian context."
  }},
  {{
    "input_term_id": "sakit perut",
    "preferred_term_en": ["Abdominal pain"],
    "notes": "Direct mapping for pain in the abdominal area."
  }},
  {{
    "input_term_id": "masuk angin",
    "preferred_term_en": ["Flatulence", "Common cold", "Malaise"],
    "notes": "Indonesian colloquial/cultural term, not a clinical diagnosis. Associated symptoms may include 'Flatulence', 'Common cold', or 'Malaise'."
  }}
]
```
{context_section}

---

*Begin Task:*

Map the following Indonesian terms to SNOMED-CT according to the JSON format and rules established above:
{terms_json}

Please provide ONLY the JSON array response, without any additional explanation or markdown formatting."""

        return prompt

    def _parse_mcn_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from Gemini

        Args:
            response_text: Raw response text from Gemini

        Returns:
            List of mapping dictionaries
        """
        try:
            # Remove markdown code blocks if present
            text = response_text.strip()
            if '```json' in text:
                # Extract JSON from code blocks
                start = text.find('```json') + 7
                end = text.find('```', start)
                text = text[start:end].strip()
            elif text.startswith('```'):
                # Generic code block
                lines = text.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                text = '\n'.join(json_lines)

            # Parse JSON
            mappings = json.loads(text)

            # Validate structure
            if not isinstance(mappings, list):
                self.logger.error("MCN response is not a list")
                return []

            # Validate each mapping has required fields
            valid_mappings = []
            for mapping in mappings:
                if isinstance(mapping, dict) and 'input_term_id' in mapping:
                    # Ensure preferred_term_en is a list
                    if 'preferred_term_en' in mapping:
                        if isinstance(mapping['preferred_term_en'], str):
                            mapping['preferred_term_en'] = [mapping['preferred_term_en']]
                        elif not isinstance(mapping['preferred_term_en'], list):
                            mapping['preferred_term_en'] = []
                    else:
                        mapping['preferred_term_en'] = []

                    # Ensure notes field exists
                    if 'notes' not in mapping:
                        mapping['notes'] = ""

                    valid_mappings.append(mapping)

            return valid_mappings

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse MCN response as JSON: {e}")
            self.logger.debug(f"Response text: {response_text}")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing MCN response: {e}")
            return []

    def process_document(self, document_content: str) -> Dict[str, Any]:
        """
        Process a complete medical document for MCN mapping

        Args:
            document_content: Full text of medical document

        Returns:
            Dictionary with extracted terms and mappings
        """
        # Extract medical terms
        terms = self.extract_medical_terms(document_content)

        if not terms:
            return {
                'success': True,
                'extracted_terms': [],
                'mappings': [],
                'message': 'No medical terms found in document'
            }

        # Map terms to SNOMED
        result = self.map_terms_to_snomed(terms, context=document_content[:500])
        result['extracted_terms'] = terms

        return result

    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch

        Args:
            documents: List of document dictionaries with 'id' and 'content'

        Returns:
            List of results for each document
        """
        results = []

        for doc in documents:
            doc_id = doc.get('id')
            content = doc.get('content', '')

            result = self.process_document(content)
            result['document_id'] = doc_id
            results.append(result)

        return results


# Global MCN mapper instance
_mcn_mapper = None

def get_mcn_mapper():
    """Get global MCN mapper instance (singleton pattern)"""
    global _mcn_mapper
    if _mcn_mapper is None:
        _mcn_mapper = MCNMapper()
    return _mcn_mapper
