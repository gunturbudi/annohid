"""
SNOMED-CT service for Django Medical Annotation System
Provides functionality to search SNOMED-CT terms with fallback mechanisms
Adapted from simple_annotation_system for Django compatibility
"""

import requests
import json
import os
import time
import logging
from django.conf import settings

class SNOMEDService:
    """Service to handle SNOMED-CT term lookups with fallback data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Use NHS Term Browser API as primary, fallback to Snowstorm
        self.nhs_api_url = 'https://termbrowser.nhs.uk/sct-browser-api/snomed/uk-edition/v20251119/descriptions'
        self.snowstorm_api_url = os.environ.get('SNOMED_API_URL', 'https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct')
        self.cache = {}
        self.cache_timeout = 86400  # 24 hours
        self.request_timeout = 10
        self.max_retries = 1
        self.fallback_mode = False
        self.use_nhs_api = True
        self._connectivity_tested = False

    def _ensure_connectivity_tested(self):
        """Lazy connectivity test — only runs once on first search call."""
        if not self._connectivity_tested:
            self._connectivity_tested = True
            self._test_api_connectivity()

    def search_term(self, term, max_results=5, context=""):
        """
        Search for SNOMED-CT terms with context awareness

        Args:
            term (str): Search query
            max_results (int): Maximum results
            context (str): Context from the original text

        Returns:
            list: SNOMED-CT candidates with relevance scores
        """
        if not term or not term.strip():
            return []

        self._ensure_connectivity_tested()

        term = term.strip()
        cache_key = f"search_{term}_{max_results}"

        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return self._enhance_with_context(cached_data, context)

        # Try NHS API first, then Snowstorm as fallback, then local fallback data
        candidates = None
        if self.use_nhs_api:
            candidates = self._search_nhs_api(term, max_results)

        # Always try Snowstorm if NHS returned nothing
        if not candidates:
            candidates = self._search_snowstorm_api(term, max_results)

        # Use local fallback data if both APIs failed
        if not candidates:
            candidates = self._get_fallback_data(term)
            if candidates:
                self.logger.info(f"Using fallback data for term: {term}")

        if not candidates:
            candidates = []

        # Cache results
        self.cache[cache_key] = (candidates, time.time())

        return self._enhance_with_context(candidates, context)

    def _search_nhs_api(self, term, max_results):
        """Search NHS Term Browser API for SNOMED-CT terms"""
        try:
            params = {
                'query': term,
                'limit': min(max_results, 50),  # NHS API supports up to 50
                'searchMode': 'partialMatching',
                'lang': 'english',
                'statusFilter': 'activeOnly',
                'skipTo': 0,
                'returnLimit': min(max_results * 2, 100),  # Get more for better filtering
                'normalize': True
            }

            response = requests.get(
                self.nhs_api_url,
                params=params,
                timeout=self.request_timeout,
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'DjangoMedicalAnnotationSystem/1.0'
                }
            )

            if response.status_code == 200:
                data = response.json()
                candidates = []

                for match in data.get('matches', [])[:max_results]:
                    if match.get('active', True) and match.get('conceptActive', True):
                        # Extract semantic tag from FSN (e.g., "Influenza (disorder)" -> "disorder")
                        fsn = match.get('fsn', '')
                        semantic_tag = ''
                        if '(' in fsn and ')' in fsn:
                            semantic_tag = fsn[fsn.rfind('(')+1:fsn.rfind(')')]

                        candidate = {
                            'preferredTerm': match.get('term', ''),
                            'conceptId': match.get('conceptId', ''),
                            'fsn': fsn,
                            'semantic_tag': semantic_tag,
                            'definition_status': match.get('definitionStatus', ''),
                            'module': match.get('module', ''),
                            'indonesian_translation': self._get_indonesian_translation(match.get('term', '')),
                            'relevance': self._calculate_relevance(term, match.get('term', '')),
                            'source': 'nhs_api'
                        }
                        candidates.append(candidate)

                # Sort by relevance
                return sorted(candidates, key=lambda x: x['relevance'], reverse=True)
            else:
                self.logger.warning(f"NHS API returned status {response.status_code}: {response.text}")

        except Exception as e:
            self.logger.error(f"NHS SNOMED-CT API error: {e}")

        return None

    def _search_snowstorm_api(self, term, max_results):
        """Search Snowstorm SNOMED-CT API (fallback)"""
        for attempt in range(self.max_retries + 1):
            try:
                params = {
                    'term': term,
                    'limit': max_results,
                    'activeFilter': 'true',
                }

                response = requests.get(
                    f"{self.snowstorm_api_url}/MAIN/concepts",
                    params=params,
                    timeout=self.request_timeout,
                    headers={'Accept': 'application/json'}
                )

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Snowstorm API rate limited, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code == 200:
                    data = response.json()
                    candidates = []

                    for item in data.get('items', []):
                        preferred_term = item.get('pt', {}).get('term', '')
                        if preferred_term:
                            fsn = item.get('fsn', {}).get('term', preferred_term)
                            semantic_tag = ''
                            if '(' in fsn and ')' in fsn:
                                semantic_tag = fsn[fsn.rfind('(')+1:fsn.rfind(')')]

                            candidates.append({
                                'preferredTerm': preferred_term,
                                'conceptId': item.get('conceptId', ''),
                                'fsn': fsn,
                                'semantic_tag': semantic_tag,
                                'definition_status': item.get('definitionStatus', ''),
                                'module': item.get('module', ''),
                                'indonesian_translation': self._get_indonesian_translation(preferred_term),
                                'relevance': self._calculate_relevance(term, preferred_term),
                                'source': 'snowstorm_api'
                            })

                    return sorted(candidates, key=lambda x: x['relevance'], reverse=True)
                else:
                    self.logger.warning(f"Snowstorm API returned status {response.status_code}")

            except Exception as e:
                self.logger.error(f"SNOMED-CT API error: {e}")

        return None

    def _get_indonesian_translation(self, english_term):
        """Get Indonesian translation of English medical term"""
        # Core translation mappings (key medical terms)
        translations = {
            # Symptoms
            'headache': 'sakit kepala',
            'fever': 'demam',
            'cough': 'batuk',
            'pain': 'nyeri',
            'nausea': 'mual',
            'vomiting': 'muntah',
            'diarrhea': 'diare',
            'dizziness': 'pusing',
            'fatigue': 'lelah',
            'shortness of breath': 'sesak napas',
            'chest pain': 'nyeri dada',
            'abdominal pain': 'nyeri perut',
            'blurred vision': 'pandangan buram',
            'excessive thirst': 'haus berlebihan',
            'frequent urination': 'sering buang air kecil',
            'tingling': 'kesemutan',
            'numbness': 'mati rasa',
            'weight loss': 'berat badan turun',

            # Diseases/Conditions
            'diabetes mellitus': 'diabetes melitus',
            'hypertension': 'hipertensi',
            'asthma': 'asma',
            'gastritis': 'gastritis',
            'ulcer': 'maag',
            'hyperglycemia': 'gula darah tinggi',

            # Body parts
            'head': 'kepala',
            'chest': 'dada',
            'abdomen': 'perut',
            'stomach': 'lambung',
            'heart': 'jantung',
            'lung': 'paru-paru',
            'kidney': 'ginjal',

            # Medications
            'paracetamol': 'parasetamol',
            'acetaminophen': 'parasetamol',
            'ibuprofen': 'ibuprofen',
            'aspirin': 'aspirin',
            'insulin': 'insulin',
        }

        term_lower = english_term.lower()
        for eng, indo in translations.items():
            if eng in term_lower:
                return indo

        return ''

    def translate_indonesian_to_english(self, indonesian_term):
        """Translate Indonesian medical term to English for better SNOMED-CT searching"""
        # Key Indonesian to English mappings
        indo_to_eng = {
            # Symptoms
            'sakit kepala': 'headache',
            'demam': 'fever',
            'batuk': 'cough',
            'nyeri': 'pain',
            'mual': 'nausea',
            'muntah': 'vomiting',
            'diare': 'diarrhea',
            'pusing': 'dizziness',
            'sesak napas': 'shortness of breath',
            'nyeri dada': 'chest pain',
            'nyeri perut': 'abdominal pain',
            'pandangan buram': 'blurred vision',
            'penglihatan buram': 'blurred vision',
            'haus berlebihan': 'excessive thirst',
            'sering haus': 'excessive thirst',
            'sering buang air kecil': 'frequent urination',
            'sering pipis': 'frequent urination',
            'sering kencing': 'frequent urination',
            'kesemutan': 'tingling',
            'mati rasa': 'numbness',
            'berat badan turun': 'weight loss',
            'gula darah': 'blood sugar',
            'gula darah tinggi': 'hyperglycemia',

            # Diseases/Conditions
            'diabetes melitus': 'diabetes mellitus',
            'kencing manis': 'diabetes mellitus',
            'hipertensi': 'hypertension',
            'asma': 'asthma',
            'gastritis': 'gastritis',
            'maag': 'ulcer',

            # Body parts
            'kepala': 'head',
            'dada': 'chest',
            'perut': 'abdomen',
            'lambung': 'stomach',
            'jantung': 'heart',
            'paru-paru': 'lung',
            'ginjal': 'kidney',

            # Medications
            'parasetamol': 'paracetamol',
            'ibuprofen': 'ibuprofen',
            'aspirin': 'aspirin',
            'insulin': 'insulin',
        }

        term_lower = indonesian_term.lower().strip()

        # Direct match
        if term_lower in indo_to_eng:
            return indo_to_eng[term_lower]

        # Partial matches
        for indo, eng in indo_to_eng.items():
            if indo in term_lower or term_lower in indo:
                return eng

        # If no translation found, return original term
        return indonesian_term

    def search_with_auto_translation(self, term, max_results=5, context=""):
        """
        Enhanced search that automatically translates Indonesian terms to English
        for better SNOMED-CT matching
        """
        # First, try to translate the term to English if it appears to be Indonesian
        english_term = self.translate_indonesian_to_english(term)

        # Search with both original and translated terms
        results = []

        # Search with original term
        original_results = self.search_term(term, max_results, context)
        if original_results:
            results.extend(original_results)

        # If translation is different, also search with English term
        if english_term.lower() != term.lower():
            english_results = self.search_term(english_term, max_results, context)
            if english_results:
                # Add English results, but mark them as translated searches
                for result in english_results:
                    result['translated_from'] = term
                    result['english_search_term'] = english_term
                results.extend(english_results)

        # Remove duplicates based on conceptId and sort by relevance
        seen_concept_ids = set()
        unique_results = []
        for result in results:
            concept_id = result.get('conceptId', '')
            if concept_id not in seen_concept_ids:
                unique_results.append(result)
                seen_concept_ids.add(concept_id)

        # Sort by relevance and limit results
        unique_results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
        return unique_results[:max_results]

    def _calculate_relevance(self, search_term, result_term):
        """Calculate relevance between search and result terms"""
        if not search_term or not result_term:
            return 0.0

        search_lower = search_term.lower()
        result_lower = result_term.lower()

        if search_lower == result_lower:
            return 1.0
        if search_lower in result_lower:
            return 0.9
        if result_lower in search_lower:
            return 0.8

        # Word overlap
        search_words = set(search_lower.split())
        result_words = set(result_lower.split())

        if search_words and result_words:
            overlap = len(search_words.intersection(result_words))
            total = len(search_words.union(result_words))
            if overlap > 0:
                return max(0.5, overlap / total)

        return 0.7

    def _enhance_with_context(self, candidates, context):
        """Enhance relevance scores based on context"""
        if not context:
            return candidates

        context_words = set(context.lower().split())

        for candidate in candidates:
            term_words = set(candidate.get('preferredTerm', '').lower().split())
            context_bonus = len(term_words.intersection(context_words)) * 0.1
            candidate['relevance'] = min(1.0, candidate['relevance'] + context_bonus)

        return sorted(candidates, key=lambda x: x['relevance'], reverse=True)

    def _test_api_connectivity(self):
        """Test SNOMED-CT API connectivity (NHS first, then Snowstorm fallback)"""
        # Test NHS API first
        try:
            response = requests.get(
                self.nhs_api_url,
                params={
                    'query': 'headache',
                    'limit': 1,
                    'searchMode': 'partialMatching',
                    'lang': 'english',
                    'statusFilter': 'activeOnly'
                },
                timeout=5,
                headers={'User-Agent': 'DjangoMedicalAnnotationSystem/1.0'}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('matches'):
                    self.use_nhs_api = True
                    self.fallback_mode = False
                    self.logger.info("NHS Term Browser API is available")
                    return
                else:
                    self.logger.warning("NHS API returned 200 but no results - treating as unavailable")
        except Exception as e:
            self.logger.warning(f"NHS API unavailable: {e}")

        # Test Snowstorm API as fallback
        try:
            response = requests.get(
                f"{self.snowstorm_api_url}/MAIN/concepts",
                params={'term': 'test', 'limit': 1},
                timeout=5
            )
            if response.status_code == 200:
                self.use_nhs_api = False
                self.fallback_mode = False
                self.logger.info("Snowstorm API is available as fallback")
                return
        except Exception as e:
            self.logger.warning(f"Snowstorm API also unavailable: {e}")

        # Both APIs failed - set fallback mode but we won't use fallback data
        self.fallback_mode = True
        self.logger.warning("Both NHS and Snowstorm APIs unavailable - searches will return empty results")

    def _get_fallback_data(self, term):
        """Enhanced fallback data with key medical mappings"""
        term_lower = term.lower().strip()

        # Core medical term mappings for fallback
        medical_mappings = {
            # Symptoms
            'headache': [
                {'preferredTerm': 'Headache', 'conceptId': '25064002', 'fsn': 'Headache (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Sakit kepala', 'relevance': 0.95, 'source': 'fallback'},
                {'preferredTerm': 'Migraine', 'conceptId': '37796009', 'fsn': 'Migraine (disorder)', 'semantic_tag': 'disorder', 'indonesian_translation': 'Migrain', 'relevance': 0.85, 'source': 'fallback'}
            ],
            'sakit kepala': [
                {'preferredTerm': 'Headache', 'conceptId': '25064002', 'fsn': 'Headache (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Sakit kepala', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'fever': [
                {'preferredTerm': 'Fever', 'conceptId': '386661006', 'fsn': 'Fever (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Demam', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'demam': [
                {'preferredTerm': 'Fever', 'conceptId': '386661006', 'fsn': 'Fever (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Demam', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'cough': [
                {'preferredTerm': 'Cough', 'conceptId': '49727002', 'fsn': 'Cough (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Batuk', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'batuk': [
                {'preferredTerm': 'Cough', 'conceptId': '49727002', 'fsn': 'Cough (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Batuk', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'pain': [
                {'preferredTerm': 'Pain', 'conceptId': '22253000', 'fsn': 'Pain (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Nyeri', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'nyeri': [
                {'preferredTerm': 'Pain', 'conceptId': '22253000', 'fsn': 'Pain (finding)', 'semantic_tag': 'finding', 'indonesian_translation': 'Nyeri', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'diabetes': [
                {'preferredTerm': 'Diabetes mellitus', 'conceptId': '73211009', 'fsn': 'Diabetes mellitus (disorder)', 'semantic_tag': 'disorder', 'indonesian_translation': 'Diabetes melitus', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'kencing manis': [
                {'preferredTerm': 'Diabetes mellitus', 'conceptId': '73211009', 'fsn': 'Diabetes mellitus (disorder)', 'semantic_tag': 'disorder', 'indonesian_translation': 'Kencing manis', 'relevance': 0.95, 'source': 'fallback'}
            ],
            'paracetamol': [
                {'preferredTerm': 'Paracetamol', 'conceptId': '387517004', 'fsn': 'Paracetamol (substance)', 'semantic_tag': 'substance', 'indonesian_translation': 'Parasetamol', 'relevance': 0.95, 'source': 'fallback'}
            ]
        }

        # Try exact match
        if term_lower in medical_mappings:
            candidates = medical_mappings[term_lower].copy()
            for candidate in candidates:
                self._normalize_fallback_candidate(candidate)
            return candidates

        # Try partial matches
        for key, candidates in medical_mappings.items():
            if key in term_lower or term_lower in key:
                result_candidates = candidates.copy()
                for candidate in result_candidates:
                    self._normalize_fallback_candidate(candidate)
                    candidate['relevance'] = max(0.3, candidate['relevance'] - 0.2)
                return result_candidates

        # No match found - return empty rather than fabricating a concept
        return []

    def _normalize_fallback_candidate(self, candidate):
        """Normalize fallback candidate to include all required fields"""
        # Add missing fields with defaults
        if 'definition_status' not in candidate:
            candidate['definition_status'] = 'Primitive'
        if 'module' not in candidate:
            candidate['module'] = '900000000000207008'  # International module

        return candidate


# Global SNOMED service instance
_snomed_service = None

def get_snomed_service():
    """Get global SNOMED service instance (singleton pattern)"""
    global _snomed_service
    if _snomed_service is None:
        _snomed_service = SNOMEDService()
    return _snomed_service