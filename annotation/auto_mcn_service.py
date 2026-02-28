"""
Automatic MCN Service
Integrates MCN mapper with document processing and storage
"""

import logging
from typing import List, Dict, Any
from django.db import transaction
from .models import TextDocument, AutoMCNMapping
from .mcn_mapper import get_mcn_mapper
from authentication.models import User


class AutoMCNService:
    """
    Service to automatically process documents and create MCN mappings
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mcn_mapper = get_mcn_mapper()

    def is_available(self) -> bool:
        """Check if automatic MCN service is available"""
        return self.mcn_mapper.is_available()

    def process_document(self, document: TextDocument, user: User, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a single document for automatic MCN mapping

        Args:
            document: TextDocument instance to process
            user: User who triggered the processing
            force_reprocess: If True, delete existing mappings and reprocess

        Returns:
            Dictionary with processing results
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'Automatic MCN service not available (API key not configured)',
                'document_id': document.id
            }

        try:
            # Check if document already has mappings
            existing_mappings = AutoMCNMapping.objects.filter(document=document)
            if existing_mappings.exists() and not force_reprocess:
                return {
                    'success': False,
                    'error': 'Document already has MCN mappings. Use force_reprocess=True to regenerate.',
                    'document_id': document.id,
                    'existing_mappings_count': existing_mappings.count()
                }

            # Delete existing mappings if force_reprocess
            if force_reprocess and existing_mappings.exists():
                deleted_count = existing_mappings.count()
                existing_mappings.delete()
                self.logger.info(f"Deleted {deleted_count} existing mappings for document {document.id}")

            # Extract terms and get mappings from LLM
            result = self.mcn_mapper.process_document(document.content)

            if not result.get('success'):
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error during MCN mapping'),
                    'document_id': document.id
                }

            # Store mappings in database
            created_mappings = self._store_mappings(
                document=document,
                mappings=result.get('mappings', []),
                user=user,
                model_used=result.get('model_used', '')
            )

            return {
                'success': True,
                'document_id': document.id,
                'extracted_terms': result.get('extracted_terms', []),
                'total_terms_extracted': len(result.get('extracted_terms', [])),
                'total_mappings_created': len(created_mappings),
                'mappings': created_mappings,
                'model_used': result.get('model_used', '')
            }

        except Exception as e:
            self.logger.error(f"Error processing document {document.id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_id': document.id
            }

    def batch_process_documents(
        self,
        documents: List[TextDocument],
        user: User,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Process multiple documents for automatic MCN mapping

        Args:
            documents: List of TextDocument instances
            user: User who triggered the processing
            force_reprocess: If True, regenerate mappings even if they exist

        Returns:
            Dictionary with batch processing results
        """
        results = {
            'success': True,
            'total_documents': len(documents),
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_mappings_created': 0,
            'document_results': [],
            'errors': []
        }

        for document in documents:
            doc_result = self.process_document(document, user, force_reprocess)

            if doc_result.get('success'):
                results['processed'] += 1
                results['total_mappings_created'] += doc_result.get('total_mappings_created', 0)
            elif 'already has MCN mappings' in doc_result.get('error', ''):
                results['skipped'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'document_id': document.id,
                    'error': doc_result.get('error')
                })

            results['document_results'].append(doc_result)

        if results['failed'] > 0:
            results['success'] = False

        return results

    @transaction.atomic
    def _store_mappings(
        self,
        document: TextDocument,
        mappings: List[Dict[str, Any]],
        user: User,
        model_used: str = ""
    ) -> List[AutoMCNMapping]:
        """
        Store MCN mappings in database

        Args:
            document: TextDocument instance
            mappings: List of mapping dictionaries from MCN mapper
            user: User who created the mappings
            model_used: LLM model identifier

        Returns:
            List of created AutoMCNMapping instances
        """
        created_mappings = []

        for mapping in mappings:
            indonesian_term = mapping.get('input_term_id', '')
            if not indonesian_term:
                continue

            # Create suggested_mappings structure
            # The mapping contains: input_term_id, preferred_term_en (list), notes
            suggested_mappings = [{
                'preferred_term_en': mapping.get('preferred_term_en', []),
                'notes': mapping.get('notes', '')
            }]

            # Create AutoMCNMapping instance
            mcn_mapping = AutoMCNMapping.objects.create(
                document=document,
                indonesian_term=indonesian_term,
                suggested_mappings=suggested_mappings,
                mapping_notes=mapping.get('notes', ''),
                created_by=user,
                model_used=model_used,
                status='pending'
            )

            created_mappings.append(mcn_mapping)
            self.logger.info(f"Created MCN mapping: {indonesian_term} for document {document.id}")

        return created_mappings

    def get_document_mappings_summary(self, document: TextDocument) -> Dict[str, Any]:
        """
        Get summary of MCN mappings for a document

        Args:
            document: TextDocument instance

        Returns:
            Dictionary with mapping statistics
        """
        mappings = AutoMCNMapping.get_document_mappings(document)

        return {
            'document_id': document.id,
            'total_mappings': mappings['total'],
            'pending_count': mappings['pending'].count(),
            'approved_count': mappings['approved'].count(),
            'modified_count': mappings['modified'].count(),
            'rejected_count': mappings['rejected'].count(),
            'completion_percentage': self._calculate_completion_percentage(mappings)
        }

    def _calculate_completion_percentage(self, mappings: Dict[str, Any]) -> float:
        """Calculate percentage of reviewed mappings"""
        total = mappings['total']
        if total == 0:
            return 0.0

        reviewed = (
            mappings['approved'].count() +
            mappings['modified'].count() +
            mappings['rejected'].count()
        )

        return round((reviewed / total) * 100, 2)

    def approve_mapping(
        self,
        mapping_id: int,
        snomed_term: str,
        user: User,
        snomed_id: str = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Approve an automatic MCN mapping

        Args:
            mapping_id: AutoMCNMapping ID
            snomed_term: Selected SNOMED-CT term
            user: User approving the mapping
            snomed_id: Optional SNOMED-CT concept ID
            notes: Optional review notes

        Returns:
            Result dictionary
        """
        try:
            mapping = AutoMCNMapping.objects.get(id=mapping_id)
            mapping.approve_mapping(snomed_term, user, snomed_id, notes)

            return {
                'success': True,
                'mapping_id': mapping_id,
                'indonesian_term': mapping.indonesian_term,
                'selected_term': snomed_term,
                'status': 'approved'
            }
        except AutoMCNMapping.DoesNotExist:
            return {
                'success': False,
                'error': f'Mapping with ID {mapping_id} not found'
            }
        except Exception as e:
            self.logger.error(f"Error approving mapping {mapping_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def reject_mapping(self, mapping_id: int, user: User, notes: str = "") -> Dict[str, Any]:
        """Reject an automatic MCN mapping"""
        try:
            mapping = AutoMCNMapping.objects.get(id=mapping_id)
            mapping.reject_mapping(user, notes)

            return {
                'success': True,
                'mapping_id': mapping_id,
                'status': 'rejected'
            }
        except AutoMCNMapping.DoesNotExist:
            return {
                'success': False,
                'error': f'Mapping with ID {mapping_id} not found'
            }
        except Exception as e:
            self.logger.error(f"Error rejecting mapping {mapping_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Global service instance
_auto_mcn_service = None

def get_auto_mcn_service():
    """Get global Auto MCN service instance (singleton pattern)"""
    global _auto_mcn_service
    if _auto_mcn_service is None:
        _auto_mcn_service = AutoMCNService()
    return _auto_mcn_service
