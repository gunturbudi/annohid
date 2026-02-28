"""
Base classes and interfaces for LLM providers
All NER providers must implement this interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class Entity:
    """Represents a single named entity prediction"""
    text: str
    start_offset: int
    end_offset: int
    label: str  # SYMPTOM, DISEASE, BODY_PART, MEDICATION, PROCEDURE
    confidence: float
    reasoning: str = ""
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON storage"""
        return {
            'text': self.text,
            'start_offset': self.start_offset,
            'end_offset': self.end_offset,
            'label': self.label,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'source': self.source,
        }


@dataclass
class NERResult:
    """Result from NER annotation by an LLM provider"""
    entities: List[Entity]
    processing_time: float
    model_used: str
    success: bool = True
    error_message: str = ""

    # Statistics
    total_entities: int = 0
    avg_confidence: float = 0.0
    entities_by_type: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate statistics after initialization"""
        self.total_entities = len(self.entities)

        if self.entities:
            self.avg_confidence = sum(e.confidence for e in self.entities) / len(self.entities)

            # Count entities by type
            for entity in self.entities:
                self.entities_by_type[entity.label] = self.entities_by_type.get(entity.label, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON storage"""
        return {
            'entities': [e.to_dict() for e in self.entities],
            'processing_time': self.processing_time,
            'model_used': self.model_used,
            'success': self.success,
            'error_message': self.error_message,
            'total_entities': self.total_entities,
            'avg_confidence': self.avg_confidence,
            'entities_by_type': self.entities_by_type,
        }


class BaseLLMProvider(ABC):
    """
    Base class for all LLM providers
    Each provider must implement the annotate_text method
    """

    # Entity types (matching database schema)
    ENTITY_TYPES = {
        'SYMPTOM': 'Medical symptoms, complaints, or clinical presentations',
        'DISEASE': 'Diseases, disorders, conditions, or diagnoses',
        'BODY_PART': 'Anatomical structures, organs, or body regions',
        'MEDICATION': 'Drugs, medications, pharmaceuticals, or therapeutic substances',
        'PROCEDURE': 'Medical procedures, treatments, operations, or interventions',
    }

    def __init__(self, model_id: str, api_key: Optional[str] = None):
        """
        Initialize the LLM provider

        Args:
            model_id: The specific model to use (e.g., 'gemini-2.5-flash', 'gpt-4o')
            api_key: API key for the provider (optional, can use environment variable)
        """
        self.model_id = model_id
        self.api_key = api_key
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the provider (setup API client, test connection, etc.)

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def annotate_text(
        self,
        text: str,
        enable_overlapping: bool = True,
        confidence_threshold: float = 0.3,
        max_retries: int = 3
    ) -> NERResult:
        """
        Annotate text and extract named entities

        Args:
            text: The medical text to annotate
            enable_overlapping: Whether to detect overlapping entities
            confidence_threshold: Minimum confidence for entity inclusion
            max_retries: Maximum retry attempts for failed predictions

        Returns:
            NERResult containing detected entities and metadata
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is available (API key configured, model accessible, etc.)

        Returns:
            True if provider is ready to use, False otherwise
        """
        pass

    def get_provider_name(self) -> str:
        """Get human-readable provider name"""
        return self.__class__.__name__.replace('Provider', '')

    def get_model_display_name(self) -> str:
        """Get human-readable model name for UI display"""
        return self.model_id

    def validate_entity(self, entity: Dict[str, Any], original_text: str) -> Optional[Entity]:
        """
        Validate and convert a raw entity dict to Entity object
        Shared validation logic for all providers

        Args:
            entity: Raw entity dictionary from LLM response
            original_text: Original text being annotated

        Returns:
            Entity object if valid, None otherwise
        """
        try:
            # Required fields
            text = entity.get('text', '').strip()
            label = entity.get('label', entity.get('entity_type', '')).upper()

            if not text or label not in self.ENTITY_TYPES:
                return None

            # Calculate or validate offsets
            start_offset = entity.get('start_offset')
            end_offset = entity.get('end_offset')

            if start_offset is None or end_offset is None:
                # Try to find the text in the original (case-insensitive search with fallback)
                start_offset = original_text.find(text)
                if start_offset == -1:
                    # Try case-insensitive search
                    start_offset = original_text.lower().find(text.lower())
                if start_offset == -1:
                    return None
                end_offset = start_offset + len(text)
            else:
                # Ensure offsets are integers
                start_offset = int(start_offset)
                end_offset = int(end_offset)

                # Validate existing offsets
                if (start_offset < 0 or end_offset > len(original_text) or
                    start_offset >= end_offset):
                    # Try to find correct position instead of rejecting
                    start_offset = original_text.find(text)
                    if start_offset == -1:
                        start_offset = original_text.lower().find(text.lower())
                    if start_offset == -1:
                        return None
                    end_offset = start_offset + len(text)
                else:
                    # Check if text matches at given position
                    actual_text = original_text[start_offset:end_offset]
                    if actual_text.lower().strip() != text.lower().strip():
                        # Try to find correct position (case-insensitive)
                        new_start = original_text.find(text)
                        if new_start == -1:
                            new_start = original_text.lower().find(text.lower())
                        if new_start == -1:
                            return None
                        start_offset = new_start
                        end_offset = start_offset + len(text)

            # Confidence score
            confidence = entity.get('confidence', 0.5)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                confidence = 0.5

            # Additional fields
            reasoning = entity.get('reasoning', '')
            source = entity.get('source', 'llm')

            return Entity(
                text=text,
                start_offset=int(start_offset),
                end_offset=int(end_offset),
                label=label,
                confidence=float(confidence),
                reasoning=reasoning,
                source=source
            )

        except Exception as e:
            # Log error if needed
            print(f"[validate_entity] Error validating entity: {e}")
            return None

    def create_error_result(self, error_message: str) -> NERResult:
        """Create an error result"""
        return NERResult(
            entities=[],
            processing_time=0.0,
            model_used=self.model_id,
            success=False,
            error_message=error_message
        )
