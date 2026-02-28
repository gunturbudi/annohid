from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db import transaction
from django.db.models import Sum, Avg, Count, Q, F
from django.core.paginator import Paginator
from django.utils import timezone
from authentication.models import User
from .models import EntityDefinition, TextDocument, TextImportBatch, AnnotationTask, AnnotationResult, AnnotationStatistics, Guideline, AutoMCNMapping, GoldStandard, AnnotatorAgreement, AnnotationMetrics
from .forms import TextImportForm, DocumentAssignmentForm
from .medical_annotator import create_medical_annotator
from .snomed_service import get_snomed_service
from .auto_mcn_service import get_auto_mcn_service
import io
import time
import json


def _get_gamification_response(task):
    """Extract gamification data from a completed task for JSON response"""
    result = getattr(task, '_gamification_result', None)
    if not result:
        return {}
    return {
        'gamification': {
            'points_earned': result.get('total_points', 0),
            'breakdown': result.get('breakdown', []),
            'new_total': result.get('new_total', 0),
            'level': result.get('level', 1),
            'level_title': result.get('level_title', ''),
            'level_up': result.get('level_up', False),
            'streak': result.get('streak', 0),
            'achievements_earned': result.get('achievements_earned', []),
            'is_status_only': False,
        }
    }


def _get_gamification_status_response(user):
    """Return current gamification status as fallback when no fresh points were awarded."""
    try:
        from authentication.gamification_models import AnnotatorProfile
        profile = AnnotatorProfile.get_or_create_profile(user)
        return {
            'gamification': {
                'points_earned': 0,
                'breakdown': [],
                'new_total': profile.total_points,
                'level': profile.level,
                'level_title': profile.get_level_title(),
                'level_up': False,
                'streak': profile.current_streak,
                'achievements_earned': [],
                'is_status_only': True,
            }
        }
    except Exception:
        return {}


def resolve_overlaps_longest_span(entities):
    """
    Resolve overlapping entities by keeping only the longest span.
    Shorter overlapping entities are removed entirely.
    Rejected entities are preserved separately for statistics tracking.

    Handles both field naming conventions:
    - 'start'/'end' (used by LLM entity JSON)
    - 'start_offset'/'end_offset' (used by annotation storage)

    Returns non-overlapping active entities + all rejected entities,
    sorted by start position.
    """
    if not entities:
        return entities

    def _get_start(e):
        return e.get('start', e.get('start_offset', 0))

    def _get_end(e):
        return e.get('end', e.get('end_offset', 0))

    # Separate rejected entities from active ones
    rejected_entities = [e for e in entities if e.get('status') == 'rejected']
    active_entities = [e for e in entities if e.get('status') != 'rejected']

    # Sort active entities by span length descending (longest first),
    # then by confidence descending as tiebreaker
    sorted_entities = sorted(active_entities, key=lambda e: (
        -(_get_end(e) - _get_start(e)),
        -e.get('confidence', 0)
    ))

    kept = []
    for entity in sorted_entities:
        start = _get_start(entity)
        end = _get_end(entity)

        # Check if this entity overlaps with any already-kept entity
        has_overlap = any(
            not (end <= _get_start(k) or start >= _get_end(k))
            for k in kept
        )

        if not has_overlap:
            kept.append(entity)

    # Combine active kept entities with all rejected entities
    all_entities = kept + rejected_entities
    all_entities.sort(key=lambda e: _get_start(e))
    return all_entities


def _save_annotation_for_annotators(doc, request_user, task_type, annotation_type,
                                    model_used, entities_json, total_entities,
                                    processing_time, review_status='completed'):
    """Save annotation result for all assigned annotators of a document.

    Falls back to request_user if no annotators are assigned.
    Returns the first AnnotationResult created (or existing).
    """
    annotators = list(doc.assigned_to.all())
    if not annotators:
        annotators = [request_user]

    first_result = None
    for annotator in annotators:
        existing = AnnotationResult.objects.filter(
            document=doc, annotator=annotator,
            task_type=task_type, annotation_type=annotation_type,
            review_status='completed'
        ).first()
        if existing:
            if first_result is None:
                first_result = existing
            continue

        result = AnnotationResult.objects.create(
            document=doc, annotator=annotator,
            task_type=task_type, annotation_type=annotation_type,
            model_used=model_used, entities_json=entities_json,
            total_entities=total_entities, processing_time=processing_time,
            review_status=review_status
        )
        if first_result is None:
            first_result = result

        task, _ = AnnotationTask.objects.get_or_create(
            document=doc, annotator=annotator, task_type=task_type
        )
        if not task.is_started:
            task.start_task()
        task.complete_task()

    return first_result


@login_required
def admin_data_management(request):
    """Admin view for data management and import"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    # Get statistics
    total_documents = TextDocument.objects.count()
    pending_documents = TextDocument.objects.filter(status='pending').count()
    assigned_documents = TextDocument.objects.filter(status='assigned').count()
    completed_documents = TextDocument.objects.filter(status='completed').count()

    # Recent imports
    recent_imports = TextImportBatch.objects.all()[:5]

    # Recent documents
    recent_documents = TextDocument.objects.all()[:10]

    # Available annotators
    annotators = User.objects.filter(role='annotator', is_active=True)

    context = {
        'total_documents': total_documents,
        'pending_documents': pending_documents,
        'assigned_documents': assigned_documents,
        'completed_documents': completed_documents,
        'recent_imports': recent_imports,
        'recent_documents': recent_documents,
        'annotators': annotators,
        'import_form': TextImportForm(),
    }

    return render(request, 'annotation/admin_data_management.html', context)


@login_required
@require_http_methods(["POST"])
def import_text_data(request):
    """Handle text file import"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    form = TextImportForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({'success': False, 'error': 'Invalid form data'}, status=400)

    uploaded_file = request.FILES['file']
    auto_assign = form.cleaned_data.get('auto_assign', False)
    skip_empty_lines = form.cleaned_data.get('skip_empty_lines', True)
    assigned_annotator = form.cleaned_data.get('assigned_annotator')
    auto_mcn_mapping = form.cleaned_data.get('auto_mcn_mapping', False)

    try:
        # Read file content
        if uploaded_file.content_type not in ['text/plain', 'application/octet-stream']:
            return JsonResponse({'success': False, 'error': 'Only text files are supported'}, status=400)

        # Decode file content
        try:
            content = uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = uploaded_file.read().decode('latin-1')
            except UnicodeDecodeError:
                return JsonResponse({'success': False, 'error': 'Unable to decode file. Please ensure it\'s a valid text file.'}, status=400)

        # Split into lines
        lines = content.split('\n')

        # Create import batch record
        import_batch = TextImportBatch.objects.create(
            filename=uploaded_file.name,
            imported_by=request.user,
            total_lines=len(lines),
            skip_empty_lines=skip_empty_lines,
            auto_assign=auto_assign
        )

        # Process lines
        successful_imports = 0
        failed_imports = 0
        error_messages = []
        imported_documents = []

        with transaction.atomic():
            for line_number, line in enumerate(lines, 1):
                line_content = line.strip()

                # Skip empty lines if requested
                if skip_empty_lines and not line_content:
                    continue

                try:
                    # Create document
                    document = TextDocument.objects.create(
                        content=line_content,
                        source_file=uploaded_file.name,
                        line_number=line_number,
                        uploaded_by=request.user,
                        status='assigned' if auto_assign and assigned_annotator else 'pending'
                    )

                    # Assign annotator if auto-assigning
                    if auto_assign and assigned_annotator:
                        document.assigned_to.add(assigned_annotator)
                        document.save()

                        # Create annotation task
                        AnnotationTask.objects.create(
                            document=document,
                            annotator=assigned_annotator
                        )

                    successful_imports += 1
                    imported_documents.append(document)

                except Exception as e:
                    failed_imports += 1
                    error_messages.append(f"Line {line_number}: {str(e)}")

        # Update import batch statistics
        import_batch.successful_imports = successful_imports
        import_batch.failed_imports = failed_imports
        if error_messages:
            import_batch.notes = '\n'.join(error_messages[:10])  # Limit error messages
        import_batch.save()

        # Process automatic MCN mapping if requested
        mcn_stats = {'enabled': False, 'processed': 0, 'mappings_created': 0}
        if auto_mcn_mapping and imported_documents:
            try:
                auto_mcn_service = get_auto_mcn_service()
                if auto_mcn_service.is_available():
                    mcn_result = auto_mcn_service.batch_process_documents(
                        documents=imported_documents,
                        user=request.user,
                        force_reprocess=False
                    )
                    mcn_stats = {
                        'enabled': True,
                        'processed': mcn_result.get('processed', 0),
                        'mappings_created': mcn_result.get('total_mappings_created', 0),
                        'failed': mcn_result.get('failed', 0)
                    }
                else:
                    mcn_stats = {
                        'enabled': True,
                        'error': 'MCN service not available (API key not configured)'
                    }
            except Exception as e:
                mcn_stats = {
                    'enabled': True,
                    'error': f'MCN processing failed: {str(e)}'
                }

        return JsonResponse({
            'success': True,
            'message': f'Import completed: {successful_imports} documents imported successfully, {failed_imports} failed.',
            'stats': {
                'successful': successful_imports,
                'failed': failed_imports,
                'total': len(lines),
                'mcn_mapping': mcn_stats
            }
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Import failed: {str(e)}'}, status=500)


@login_required
def document_list(request):
    """List all documents with pagination and filtering"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    # Filtering
    status_filter = request.GET.get('status', 'all')
    annotator_filter = request.GET.get('annotator', 'all')

    documents = TextDocument.objects.all()

    if status_filter != 'all':
        documents = documents.filter(status=status_filter)

    if annotator_filter != 'all':
        documents = documents.filter(assigned_to__id=annotator_filter)

    # Pagination
    paginator = Paginator(documents, 25)  # 25 documents per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Available annotators for filtering
    annotators = User.objects.filter(role='annotator', is_active=True)

    context = {
        'page_obj': page_obj,
        'status_filter': status_filter,
        'annotator_filter': annotator_filter,
        'annotators': annotators,
        'status_choices': TextDocument.STATUS_CHOICES,
    }

    return render(request, 'annotation/document_list.html', context)


@login_required
def document_detail(request, document_id):
    """View individual document details"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    document = get_object_or_404(TextDocument, id=document_id)

    # Get assigned annotator IDs for easy template checking
    assigned_annotator_ids = list(document.assigned_to.values_list('id', flat=True))

    # Get existing NER and MCN annotations
    latest_ner = document.get_latest_ner_annotation()
    latest_mcn = document.get_latest_mcn_annotation()

    # Get all NER and MCN annotations for this document (for history)
    ner_annotations = document.annotation_results.filter(task_type='ner').order_by('-created_at')
    mcn_annotations = document.annotation_results.filter(task_type='mcn').order_by('-created_at')

    context = {
        'document': document,
        'annotators': User.objects.filter(role='annotator', is_active=True),
        'assigned_annotator_ids': assigned_annotator_ids,
        'latest_ner': latest_ner,
        'latest_mcn': latest_mcn,
        'ner_annotations': ner_annotations,
        'mcn_annotations': mcn_annotations,
        'has_ner': document.has_ner_annotation(),
        'has_mcn': document.has_mcn_annotation(),
    }

    return render(request, 'annotation/document_detail.html', context)


@login_required
@require_http_methods(["POST"])
def assign_document(request, document_id):
    """Assign document to annotator (supports multiple annotators)"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    document = get_object_or_404(TextDocument, id=document_id)
    annotator_id = request.POST.get('annotator_id')
    action = request.POST.get('action', 'add')  # 'add', 'remove', or 'clear'

    try:
        if action == 'clear':
            # Remove all annotators
            document.assigned_to.clear()
            document.status = 'pending'
            document.save()

            # Remove all annotation tasks
            AnnotationTask.objects.filter(document=document).delete()

            return JsonResponse({
                'success': True,
                'message': 'All annotators removed from document'
            })

        if not annotator_id:
            return JsonResponse({'success': False, 'error': 'Annotator ID required'}, status=400)

        annotator = User.objects.get(id=annotator_id, role='annotator', is_active=True)

        if action == 'remove':
            # Remove specific annotator
            document.assigned_to.remove(annotator)

            # Remove annotation task for this annotator
            AnnotationTask.objects.filter(document=document, annotator=annotator).delete()

            # Update status if no annotators left
            if document.assigned_to.count() == 0:
                document.status = 'pending'
            document.save()

            return JsonResponse({
                'success': True,
                'message': f'Removed {annotator.username} from document'
            })

        else:  # action == 'add' (default)
            # Check if annotator is already assigned
            if document.assigned_to.filter(id=annotator_id).exists():
                return JsonResponse({
                    'success': False,
                    'error': f'{annotator.username} is already assigned to this document'
                }, status=400)

            # Add annotator to document
            document.assigned_to.add(annotator)
            document.status = 'assigned'
            document.save()

            # Create annotation task
            AnnotationTask.objects.get_or_create(
                document=document,
                annotator=annotator
            )

            assigned_count = document.assigned_to.count()
            return JsonResponse({
                'success': True,
                'message': f'Added {annotator.username} to document ({assigned_count} annotator{"s" if assigned_count != 1 else ""} total)'
            })

    except User.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Invalid annotator'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
@require_http_methods(["POST"])
def bulk_assign_documents(request):
    """Bulk assign multiple documents to annotators (supports multiple annotators per document)"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    document_ids = request.POST.getlist('document_ids')
    annotator_ids = request.POST.getlist('annotator_ids')
    # Backward compat: support single annotator_id param
    if not annotator_ids:
        single_id = request.POST.get('annotator_id')
        if single_id:
            annotator_ids = [single_id]
    action = request.POST.get('action', 'add')  # 'add' or 'replace'

    if not document_ids or not annotator_ids:
        return JsonResponse({'success': False, 'error': 'Document IDs and at least one annotator required'}, status=400)

    try:
        annotators = list(User.objects.filter(id__in=annotator_ids, role='annotator', is_active=True))
        if not annotators:
            return JsonResponse({'success': False, 'error': 'No valid annotators found'}, status=400)

        assigned_count = 0
        skipped_count = 0

        with transaction.atomic():
            for doc_id in document_ids:
                try:
                    document = TextDocument.objects.get(id=doc_id)

                    if action == 'replace':
                        document.assigned_to.clear()
                        AnnotationTask.objects.filter(document=document).delete()

                        for annotator in annotators:
                            document.assigned_to.add(annotator)
                            AnnotationTask.objects.create(
                                document=document,
                                annotator=annotator
                            )
                        document.status = 'assigned'
                        document.save()
                        assigned_count += 1

                    else:  # action == 'add'
                        doc_assigned = False
                        for annotator in annotators:
                            if not document.assigned_to.filter(id=annotator.id).exists():
                                document.assigned_to.add(annotator)
                                AnnotationTask.objects.get_or_create(
                                    document=document,
                                    annotator=annotator
                                )
                                doc_assigned = True
                        if doc_assigned:
                            document.status = 'assigned'
                            document.save()
                            assigned_count += 1
                        else:
                            skipped_count += 1

                except TextDocument.DoesNotExist:
                    continue

        annotator_names = ', '.join(a.username for a in annotators)
        message = f'{assigned_count} documents assigned to {annotator_names}'
        if skipped_count > 0:
            message += f' ({skipped_count} already assigned)'

        return JsonResponse({
            'success': True,
            'message': message
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def auto_assign_documents(request):
    """Automatically assign selected documents evenly across all active annotators using heuristic distribution"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST request required'}, status=405)

    document_ids = request.POST.getlist('document_ids')

    if not document_ids:
        return JsonResponse({'success': False, 'error': 'No documents selected'}, status=400)

    try:
        # Get all active annotators
        annotators = list(User.objects.filter(role='annotator', is_active=True))

        if not annotators:
            return JsonResponse({'success': False, 'error': 'No active annotators available'}, status=400)

        # Get current workload for each annotator (number of assigned but not completed documents)
        annotator_workload = {}
        for annotator in annotators:
            workload = AnnotationTask.objects.filter(
                annotator=annotator,
                is_completed=False
            ).count()
            annotator_workload[annotator.id] = workload

        # Sort annotators by current workload (least busy first)
        sorted_annotators = sorted(annotators, key=lambda a: annotator_workload[a.id])

        # Randomize the order slightly while maintaining relative balance
        # Split into groups and shuffle within groups to add randomness
        import random
        group_size = max(1, len(sorted_annotators) // 3)
        randomized_annotators = []
        for i in range(0, len(sorted_annotators), group_size):
            group = sorted_annotators[i:i+group_size]
            random.shuffle(group)
            randomized_annotators.extend(group)

        assigned_count = 0
        skipped_count = 0
        assignment_summary = {annotator.username: 0 for annotator in annotators}

        with transaction.atomic():
            # Round-robin assignment with the randomized annotator list
            for idx, doc_id in enumerate(document_ids):
                try:
                    document = TextDocument.objects.get(id=doc_id)

                    # Select annotator using round-robin
                    annotator = randomized_annotators[idx % len(randomized_annotators)]

                    # Check if already assigned to this annotator
                    if not document.assigned_to.filter(id=annotator.id).exists():
                        # Clear existing assignments
                        document.assigned_to.clear()
                        AnnotationTask.objects.filter(document=document).delete()

                        # Assign to selected annotator
                        document.assigned_to.add(annotator)
                        document.status = 'assigned'
                        document.save()

                        AnnotationTask.objects.create(
                            document=document,
                            annotator=annotator
                        )

                        assigned_count += 1
                        assignment_summary[annotator.username] += 1
                    else:
                        skipped_count += 1

                except TextDocument.DoesNotExist:
                    continue

        # Create summary message
        message = f'Successfully assigned {assigned_count} documents'
        if skipped_count > 0:
            message += f' ({skipped_count} already assigned)'

        # Add distribution details
        distribution_details = [f"{username}: {count}" for username, count in assignment_summary.items() if count > 0]
        if distribution_details:
            message += f'\n\nDistribution: {", ".join(distribution_details)}'

        return JsonResponse({
            'success': True,
            'message': message,
            'assigned_count': assigned_count,
            'skipped_count': skipped_count,
            'distribution': assignment_summary
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def import_history(request):
    """View import history"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    imports = TextImportBatch.objects.all()
    paginator = Paginator(imports, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
    }

    return render(request, 'annotation/import_history.html', context)


@login_required
def annotate_document(request, document_id):
    """Admin interface for annotating a document using LLM"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    document = get_object_or_404(TextDocument, id=document_id)

    # Get existing annotations for this document
    existing_annotations = AnnotationResult.objects.filter(document=document).order_by('-created_at')

    # Get available models
    annotator = create_medical_annotator()
    available_models = annotator.get_available_models()
    api_status = annotator.validate_api_setup()

    context = {
        'document': document,
        'existing_annotations': existing_annotations,
        'available_models': available_models,
        'api_status': api_status,
        'default_model': 'gemini-2.5-flash',
    }

    return render(request, 'annotation/annotate_document.html', context)


@login_required
@require_http_methods(["POST"])
def perform_annotation(request, document_id):
    """Perform LLM annotation on a document"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    document = get_object_or_404(TextDocument, id=document_id)

    try:
        data = json.loads(request.body)
        model_id = data.get('model', 'gemini-2.5-flash')

        # Create annotator with specified model
        annotator = create_medical_annotator(model_id=model_id)

        # Check API setup
        api_status = annotator.validate_api_setup()
        if not api_status['ready_for_annotation']:
            return JsonResponse({
                'success': False,
                'error': 'API not properly configured. Please check your environment variables.',
                'api_status': api_status
            }, status=400)

        # Perform annotation
        print(f"Starting annotation for document {document_id} with model {model_id}")
        start_time = time.time()
        result = annotator.annotate_document(document.content)
        processing_time = time.time() - start_time
        print(f"Annotation completed in {processing_time:.2f}s, found {len(result.get('entities', []))} entities")

        if result['success']:
            # Save annotation result
            annotation_result = AnnotationResult.objects.create(
                document=document,
                annotator=request.user,
                annotation_type='llm',
                model_used=model_id,
                entities_json=result['entities'],
                processing_time=processing_time
            )

            # Update document status if it was pending
            if document.status == 'pending':
                document.status = 'in_progress'
                document.save()

            return JsonResponse({
                'success': True,
                'annotation_id': annotation_result.id,
                'entities': result['entities'],
                'total_entities': result['total_entities'],
                'entity_stats': result['entity_stats'],
                'processing_time': processing_time,
                'model_used': model_id,
                'confidence_score': annotation_result.confidence_score
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Annotation failed to process'
            }, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Annotation failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def run_langextract(request):
    """Run LangExtract on a document using selected LLM model"""
    document_id = request.POST.get('document_id')
    model_id = request.POST.get('model_id', 'gemini-2.5-flash')  # Default to Gemini 2.5 Flash

    if not document_id:
        return JsonResponse({'success': False, 'error': 'Document ID required'}, status=400)

    document = get_object_or_404(TextDocument, id=document_id)

    try:
        # Import LangExtract annotator
        from .medical_annotator import create_medical_annotator

        # Create annotator with selected model
        print(f"Running LangExtract on document {document_id} with model {model_id}")
        start_time = time.time()

        # MedicalAnnotator uses LangExtract by default
        annotator = create_medical_annotator(model_id=model_id)
        result = annotator.annotate_document(document.content)

        processing_time = time.time() - start_time
        print(f"LangExtract completed in {processing_time:.2f}s, found {len(result.get('entities', []))} entities")

        if result['success']:
            # Save annotation result
            annotation_result = AnnotationResult.objects.create(
                document=document,
                annotator=request.user,
                annotation_type='llm',
                model_used=f'langextract-{model_id}',
                entities_json=result['entities'],
                processing_time=processing_time
            )

            # Update document status if it was pending
            if document.status == 'pending':
                document.status = 'in_progress'
                document.save()

            # Transform entities to match frontend expectations
            # Backend uses: label, start, end
            # Frontend expects: entity_type, start_offset, end_offset
            frontend_entities = []
            for entity in result['entities']:
                frontend_entities.append({
                    'text': entity['text'],
                    'entity_type': entity['label'],  # Rename label -> entity_type
                    'start_offset': entity['start'],  # Rename start -> start_offset
                    'end_offset': entity['end'],      # Rename end -> end_offset
                    'confidence': entity.get('confidence', 0.85),
                    'attributes': entity.get('attributes', {})
                })

            return JsonResponse({
                'success': True,
                'message': f'LangExtract completed successfully! Found {result["total_entities"]} medical entities.',
                'annotation_id': annotation_result.id,
                'entities_count': result['total_entities'],
                'processing_time': round(processing_time, 2),
                'entities': frontend_entities,  # Use transformed entities for visualization
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'LangExtract failed to process the document. Please try again.'
            }, status=500)

    except Exception as e:
        import traceback
        error_message = str(e)
        print(f"LangExtract error: {e}")
        print(traceback.format_exc())

        # The error message from medical_annotator is already user-friendly
        # Just pass it through to the frontend
        return JsonResponse({
            'success': False,
            'error': error_message
        }, status=500)


@login_required
@require_http_methods(["POST"])
def run_three_step_mcn(request):
    """
    Run 3-Step MCN on a single document with model selection
    Steps:
    1. Extract entities using LangExtract (or use existing NER)
    2. Translate Indonesian → English & normalize to medical terms
    3. Map entities to SNOMED-CT codes
    """
    document_id = request.POST.get('document_id')
    model_id = request.POST.get('model_id', 'gemini-2.5-flash')

    if not document_id:
        return JsonResponse({'success': False, 'error': 'Document ID required'}, status=400)

    document = get_object_or_404(TextDocument, id=document_id)

    try:
        # Import necessary services
        from .medical_annotator import create_medical_annotator
        from .gemini_normalizer import normalize_medical_entities
        from .snomed_service import get_snomed_service

        start_time = time.time()

        # STEP 1: Get or create NER entities (check from any user)
        ner_annotation = AnnotationResult.objects.filter(
            document=document,
            task_type='ner',
            review_status='completed'
        ).first()

        if not ner_annotation:
            # Run LangExtract NER with selected model
            annotator = create_medical_annotator(model_id=model_id)
            ner_result = annotator.annotate_document(document.content)

            if not ner_result['success']:
                return JsonResponse({
                    'success': False,
                    'error': 'Failed to extract entities from document'
                }, status=500)

            entities = ner_result['entities']

            # Save NER annotation for all assigned annotators
            ner_annotation = _save_annotation_for_annotators(
                doc=document, request_user=request.user,
                task_type='ner', annotation_type='llm_assisted',
                model_used=f'langextract-{model_id}',
                entities_json=entities,
                total_entities=len(entities),
                processing_time=time.time() - start_time
            )
        else:
            entities = ner_annotation.entities_json

        # Filter only approved entities (exclude rejected ones)
        entities = [
            entity for entity in entities
            if entity.get('status') == 'approved' or 'status' not in entity
        ]

        if not entities:
            return JsonResponse({
                'success': False,
                'error': 'No approved entities found in document'
            }, status=400)

        # STEP 2: Translate and Normalize (single Gemini batch call)
        step2_start = time.time()

        # Only MCN-relevant types need SNOMED mapping
        MCN_TYPES = {'SYMPTOM', 'DISEASE', 'MEDICATION', 'PROCEDURE'}
        mcn_entities = [
            e for e in entities
            if e.get('label', e.get('entity_type', 'UNKNOWN')).upper() in MCN_TYPES
        ]
        skip_entities = [
            e for e in entities
            if e.get('label', e.get('entity_type', 'UNKNOWN')).upper() not in MCN_TYPES
        ]

        normalized_entities = normalize_medical_entities(mcn_entities)

        # STEP 3: Auto-map to SNOMED-CT (parallelized)
        step3_start = time.time()
        snomed_service = get_snomed_service()

        successfully_mapped = 0
        successfully_translated = 0

        def _map_entity(entity):
            normalized_term = entity.get('normalized_term', entity.get('text', ''))
            entity_type = entity.get('entity_type', entity.get('label', 'UNKNOWN'))
            try:
                candidates = snomed_service.search_with_auto_translation(
                    normalized_term, max_results=1, context=entity_type
                )
                if candidates:
                    best = candidates[0]
                    entity.update({
                        'snomed_concept': {
                            'conceptId': best.get('conceptId', ''),
                            'fsn': best.get('fsn', ''),
                            'preferredTerm': best.get('preferredTerm', ''),
                            'definition': best.get('definition', ''),
                            'confidence_score': best.get('confidence_score', 0.8),
                        },
                        'snomed_code': best.get('conceptId', ''),
                        'snomed_display': best.get('preferredTerm', ''),
                        'auto_mapped': True,
                        'is_mapped': True,
                        'mapping_confidence': best.get('confidence_score', 0.8),
                    })
                else:
                    entity.update({
                        'snomed_concept': None,
                        'snomed_code': '',
                        'snomed_display': '',
                        'auto_mapped': False,
                        'is_mapped': False,
                        'mapping_confidence': 0.0,
                    })
            except Exception as err:
                entity.update({
                    'snomed_concept': None,
                    'snomed_code': '',
                    'snomed_display': '',
                    'auto_mapped': False,
                    'is_mapped': False,
                    'mapping_confidence': 0.0,
                    'mapping_error': str(err),
                })
            return entity

        from concurrent.futures import ThreadPoolExecutor, as_completed
        mapped_entities = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(_map_entity, ent): idx
                for idx, ent in enumerate(normalized_entities)
            }
            results_by_idx = {}
            for future in as_completed(futures):
                idx = futures[future]
                results_by_idx[idx] = future.result()
            for idx in range(len(normalized_entities)):
                mapped_entities.append(results_by_idx[idx])

        # Count statistics
        for ent in mapped_entities:
            if ent.get('english_term'):
                successfully_translated += 1
            if ent.get('auto_mapped'):
                successfully_mapped += 1

        # Add back skipped entity types (BODY_PART etc.)
        for entity in skip_entities:
            entity.update({
                'english_term': entity.get('text', ''),
                'normalized_term': entity.get('text', ''),
                'snomed_concept': None,
                'snomed_code': '',
                'snomed_display': '',
                'auto_mapped': False,
                'is_mapped': False,
                'mapping_confidence': 0.0,
                'normalization_source': 'skipped',
            })
            mapped_entities.append(entity)

        # Save MCN annotation result for all assigned annotators
        mcn_annotation = _save_annotation_for_annotators(
            doc=document, request_user=request.user,
            task_type='mcn', annotation_type='mcn_llm_assisted',
            model_used=f'3step-mcn-{model_id}',
            entities_json=mapped_entities,
            total_entities=len(mapped_entities),
            processing_time=time.time() - start_time
        )

        # Calculate statistics
        total_entities = len(entities)
        translation_rate = (successfully_translated / total_entities * 100) if total_entities > 0 else 0
        mapping_rate = (successfully_mapped / total_entities * 100) if total_entities > 0 else 0

        processing_time = time.time() - start_time

        response_data = {
            'success': True,
            'message': f'3-Step MCN completed! Processed {total_entities} entities.',
            'annotation_id': mcn_annotation.id,
            'entities_count': total_entities,
            'processing_time': round(processing_time, 2),
            'statistics': {
                'total_entities': total_entities,
                'successfully_translated': successfully_translated,
                'successfully_mapped': successfully_mapped,
                'translation_rate': round(translation_rate, 2),
                'mapping_rate': round(mapping_rate, 2)
            }
        }
        gamification_data = _get_gamification_response(mcn_task)
        if not gamification_data:
            gamification_data = _get_gamification_status_response(request.user)
        response_data.update(gamification_data)
        return JsonResponse(response_data)

    except Exception as e:
        import traceback
        error_message = str(e)
        print(f"3-Step MCN error: {e}")
        print(traceback.format_exc())

        return JsonResponse({
            'success': False,
            'error': f'3-Step MCN failed: {error_message}'
        }, status=500)


@login_required
def view_annotation_result(request, annotation_id):
    """View detailed annotation result"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    annotation = get_object_or_404(AnnotationResult, id=annotation_id)

    # Group entities by type for better visualization
    entities_by_type = annotation.get_filtered_entities_by_type()
    entity_stats = annotation.get_entity_statistics()

    # Resolve overlapping entities and serialize for JavaScript
    entities_list = resolve_overlaps_longest_span(annotation.entities_json) if annotation.entities_json else []
    entities_json_str = json.dumps(entities_list)

    context = {
        'annotation': annotation,
        'annotation_entities_json': entities_json_str,
        'entities_by_type': entities_by_type,
        'entity_stats': entity_stats,
        'entity_types': ['SYMPTOM', 'DISEASE', 'BODY_PART', 'MEDICATION', 'PROCEDURE']
    }

    return render(request, 'annotation/annotation_result.html', context)


@login_required
def annotation_results_list(request):
    """List all annotation results"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    # Filtering
    model_filter = request.GET.get('model', 'all')
    type_filter = request.GET.get('type', 'all')
    status_filter = request.GET.get('status', 'all')

    annotations = AnnotationResult.objects.all()

    if model_filter != 'all':
        annotations = annotations.filter(model_used=model_filter)

    if type_filter != 'all':
        annotations = annotations.filter(annotation_type=type_filter)

    if status_filter != 'all':
        annotations = annotations.filter(review_status=status_filter)

    # Pagination
    paginator = Paginator(annotations, 25)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Get filter options
    available_models = AnnotationResult.objects.values_list('model_used', flat=True).distinct()

    context = {
        'page_obj': page_obj,
        'model_filter': model_filter,
        'type_filter': type_filter,
        'status_filter': status_filter,
        'available_models': available_models,
        'annotation_types': AnnotationResult.ANNOTATION_TYPES,
        'review_statuses': [('pending', 'Pending Review'), ('approved', 'Approved'), ('rejected', 'Rejected')],
    }

    return render(request, 'annotation/annotation_results_list.html', context)


@login_required
@require_http_methods(["POST"])
def update_annotation_status(request, annotation_id):
    """Update annotation review status"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    annotation = get_object_or_404(AnnotationResult, id=annotation_id)

    try:
        data = json.loads(request.body)
        new_status = data.get('status')
        admin_notes = data.get('notes', '')

        if new_status not in ['pending', 'approved', 'rejected']:
            return JsonResponse({'success': False, 'error': 'Invalid status'}, status=400)

        annotation.review_status = new_status
        annotation.admin_notes = admin_notes
        annotation.save()

        return JsonResponse({
            'success': True,
            'message': f'Annotation status updated to {new_status}'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# Manual Annotation Views
@login_required
def manual_annotate(request, document_id):
    """Manual annotation interface for annotators"""
    document = get_object_or_404(TextDocument, id=document_id)

    # Check if user can access this document
    if not request.user.is_admin() and not document.is_assigned_to_user(request.user):
        messages.error(request, 'You do not have access to this document.')
        return redirect('authentication:dashboard')

    # Check if user can do manual annotation
    if not request.user.can_do_manual_annotation():
        messages.error(request, 'You are not authorized to perform manual annotations.')
        return redirect('annotation:annotator_documents')

    # Get existing manual annotation for this user and document
    existing_annotation = AnnotationResult.objects.filter(
        document=document,
        annotator=request.user,
        annotation_type='manual'
    ).first()

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            annotations = data.get('annotations', [])
            annotation_id = data.get('annotation_id')
            deletion_history = data.get('deletion_history', [])

            if not annotations:
                return JsonResponse({'success': False, 'error': 'No annotations provided'}, status=400)

            # Ensure all annotations are marked as manual
            for annotation in annotations:
                annotation['source'] = 'manual'
                annotation['status'] = 'approved'

            # Build decisions_json with deletion history if present
            decisions = {}
            if deletion_history:
                decisions['deletion_history'] = deletion_history
                decisions['deleted_count'] = len(deletion_history)

            # Update existing annotation or create new one
            if annotation_id and existing_annotation:
                existing_annotation.entities_json = annotations
                existing_annotation.review_status = 'completed'
                existing_annotation.task_type = 'ner'  # Mark as NER task
                if decisions:
                    existing_annotation.decisions_json = decisions
                existing_annotation.save()
                result_annotation = existing_annotation
            else:
                result_annotation = AnnotationResult.objects.create(
                    document=document,
                    annotator=request.user,
                    annotation_type='manual',
                    task_type='ner',
                    entities_json=annotations,
                    decisions_json=decisions if decisions else {},
                    review_status='completed'
                )

            # Mark NER task as completed
            ner_task = AnnotationTask.get_ner_task(document, request.user)
            if ner_task and not ner_task.is_completed:
                ner_task.complete_task()

            # Update document status
            if document.status == 'pending':
                document.status = 'in_progress'
                document.save()

            response_data = {
                'success': True,
                'annotation_id': result_annotation.id,
                'message': 'Manual annotations saved successfully'
            }
            gamification_data = _get_gamification_response(ner_task) if ner_task else {}
            if not gamification_data:
                gamification_data = _get_gamification_status_response(request.user)
            response_data.update(gamification_data)
            return JsonResponse(response_data)

        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Failed to save annotations: {str(e)}'}, status=500)

    # GET request - show manual annotation interface
    # Start the task timer when user opens the annotation page
    ner_task = AnnotationTask.get_ner_task(document, request.user)
    if ner_task and not ner_task.is_started and not ner_task.is_completed:
        ner_task.start_task()

    context = {
        'document': document,
        'existing_annotation': existing_annotation,
    }

    return render(request, 'annotation/manual_annotate.html', context)


@login_required
def llm_assisted_annotate(request, document_id):
    """LLM-assisted annotation interface where user can approve/reject LLM suggestions"""
    document = get_object_or_404(TextDocument, id=document_id)

    # Check if user can access this document
    if not request.user.is_admin() and not document.is_assigned_to_user(request.user):
        messages.error(request, 'You do not have access to this document.')
        return redirect('authentication:dashboard')

    # Check if user can do LLM-assisted annotation
    if not request.user.can_do_llm_assisted_annotation():
        messages.error(request, 'You are not authorized to perform LLM-assisted annotations.')
        return redirect('annotation:annotator_documents')

    # Get or generate LLM suggestions
    # Look for LLM annotations with various types (llm_only, llm, langextract)
    llm_annotation = AnnotationResult.objects.filter(
        document=document,
        annotation_type__in=['llm_only', 'llm']
    ).order_by('-created_at').first()

    # Get existing LLM-assisted annotation for this user
    existing_annotation = AnnotationResult.objects.filter(
        document=document,
        annotator=request.user,
        annotation_type='llm_assisted'
    ).first()

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            annotations = data.get('annotations', [])
            decisions = data.get('decisions', {})

            if not annotations:
                return JsonResponse({'success': False, 'error': 'No annotations provided'}, status=400)

            # Resolve overlapping entities: keep longest span, auto-reject shorter overlaps
            annotations = resolve_overlaps_longest_span(annotations)

            # Calculate decision statistics for analysis
            decision_stats = {
                'total_llm_suggestions': 0,
                'approved': 0,
                'rejected': 0,
                'manual_added': 0,
                'pending': 0
            }

            for annotation in annotations:
                source = annotation.get('source', 'llm')
                status = annotation.get('status', 'pending')

                if source == 'manual':
                    decision_stats['manual_added'] += 1
                elif source == 'llm':
                    decision_stats['total_llm_suggestions'] += 1
                    if status == 'approved':
                        decision_stats['approved'] += 1
                    elif status == 'rejected':
                        decision_stats['rejected'] += 1
                    elif status == 'pending':
                        decision_stats['pending'] += 1

            # Determine review status based on pending entities
            # Only mark as 'completed' if all LLM entities have been reviewed (no pending status)
            if decision_stats['pending'] > 0:
                review_status = 'in_progress'
            else:
                review_status = 'completed'

            # Update existing annotation or create new one
            if existing_annotation:
                existing_annotation.entities_json = annotations
                existing_annotation.decisions_json = {
                    'decision_stats': decision_stats,
                    'decisions': decisions,
                    'timestamp': time.time()
                }
                existing_annotation.review_status = review_status
                existing_annotation.task_type = 'ner'  # Mark as NER task
                existing_annotation.save()
                result_annotation = existing_annotation
            else:
                result_annotation = AnnotationResult.objects.create(
                    document=document,
                    annotator=request.user,
                    annotation_type='llm_assisted',
                    task_type='ner',
                    model_used=llm_annotation.model_used if llm_annotation else 'unknown',
                    entities_json=annotations,
                    decisions_json={
                        'decision_stats': decision_stats,
                        'decisions': decisions,
                        'timestamp': time.time()
                    },
                    review_status=review_status
                )

            # Mark NER task as completed ONLY if review is complete
            ner_task = AnnotationTask.get_ner_task(document, request.user)
            if ner_task:
                # Only mark as completed if all reviews are done
                if review_status == 'completed' and not ner_task.is_completed:
                    ner_task.complete_task()

            # Update document status
            if document.status == 'pending':
                document.status = 'in_progress'
                document.save()

            # Calculate agreement rate
            agreement_rate = 0
            if decision_stats['total_llm_suggestions'] > 0:
                agreement_rate = (decision_stats['approved'] / decision_stats['total_llm_suggestions']) * 100

            # Prepare appropriate message based on review status
            if review_status == 'completed':
                message = f'NER annotation completed! All entities reviewed ({decision_stats["approved"]} approved, {decision_stats["rejected"]} rejected)'
            else:
                message = f'Progress saved! {decision_stats["pending"]} entities still need review'

            response_data = {
                'success': True,
                'annotation_id': result_annotation.id,
                'message': message,
                'review_status': review_status,
                'agreement_rate': agreement_rate,
                'decision_stats': decision_stats,
                'is_completed': review_status == 'completed'
            }
            if review_status != 'completed':
                response_data['completion_hint'] = (
                    f'Review all {decision_stats["pending"]} pending entities to earn points!'
                )
            gamification_data = _get_gamification_response(ner_task) if ner_task and review_status == 'completed' else {}
            if not gamification_data:
                gamification_data = _get_gamification_status_response(request.user)
            response_data.update(gamification_data)
            return JsonResponse(response_data)

        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Failed to save annotations: {str(e)}'}, status=500)

    # GET request - show LLM-assisted annotation interface
    # If no LLM annotation exists, generate one first
    if not llm_annotation:
        # Generate LLM annotation
        annotator = create_medical_annotator()
        result = annotator.annotate_document(document.content)

        if result['success']:
            # Resolve overlapping entities at creation time: keep longest span only
            clean_entities = resolve_overlaps_longest_span(result['entities'])
            llm_annotation = AnnotationResult.objects.create(
                document=document,
                annotator=request.user,
                annotation_type='llm_only',
                model_used=annotator.model_id,
                entities_json=clean_entities
            )

    # Start the task timer when user opens the annotation page
    ner_task = AnnotationTask.get_ner_task(document, request.user)
    if ner_task and not ner_task.is_started and not ner_task.is_completed:
        ner_task.start_task()

    # Resolve overlapping LLM entities: keep only longest span
    llm_entities = llm_annotation.entities_json if llm_annotation and llm_annotation.entities_json else []
    llm_entities = resolve_overlaps_longest_span(llm_entities)

    # Serialize JSON fields properly for JavaScript
    llm_entities_json = json.dumps(llm_entities)
    existing_entities_json = '[]'
    existing_decisions_json = '{}'

    if existing_annotation:
        existing_entities_json = json.dumps(existing_annotation.entities_json) if existing_annotation.entities_json else '[]'
        existing_decisions_json = json.dumps(existing_annotation.decisions_json) if existing_annotation.decisions_json else '{}'

    context = {
        'document': document,
        'llm_annotation': llm_annotation,
        'llm_entities_json': llm_entities_json,
        'existing_annotation': existing_annotation,
        'existing_entities_json': existing_entities_json,
        'existing_decisions_json': existing_decisions_json,
    }

    return render(request, 'annotation/llm_assisted_annotate.html', context)


# SNOMED-CT Search Views
@login_required
@require_http_methods(["POST"])
def search_snomed(request):
    """Search SNOMED-CT terms for entity normalization"""
    try:
        data = json.loads(request.body)
        search_term = data.get('term', '').strip()
        context = data.get('context', '')
        max_results = min(int(data.get('max_results', 5)), 20)  # Limit to 20 max

        if not search_term:
            return JsonResponse({'success': False, 'error': 'Search term is required'}, status=400)

        # Get SNOMED service
        snomed_service = get_snomed_service()

        # Search with auto-translation for Indonesian terms
        candidates = snomed_service.search_with_auto_translation(
            search_term,
            max_results=max_results,
            context=context
        )

        return JsonResponse({
            'success': True,
            'candidates': candidates,
            'search_term': search_term,
            'total_found': len(candidates)
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'SNOMED search failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def map_entity_to_snomed(request):
    """Map an entity to a selected SNOMED-CT concept"""
    try:
        data = json.loads(request.body)
        annotation_id = data.get('annotation_id')
        entity_index = data.get('entity_index')
        snomed_concept = data.get('snomed_concept')

        if not all([annotation_id, entity_index is not None, snomed_concept]):
            return JsonResponse({'success': False, 'error': 'Missing required parameters'}, status=400)

        # Get the annotation result
        annotation = get_object_or_404(AnnotationResult, id=annotation_id)

        # Check permissions
        if not request.user.is_admin() and annotation.annotator != request.user:
            return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

        # Update the entity with SNOMED mapping
        entities = annotation.entities_json
        if 0 <= entity_index < len(entities):
            entities[entity_index]['snomed_concept'] = snomed_concept
            entities[entity_index]['snomed_mapped_at'] = time.time()
            entities[entity_index]['snomed_mapped_by'] = request.user.username

            # Save the updated annotation
            annotation.entities_json = entities
            annotation.save()

            return JsonResponse({
                'success': True,
                'message': 'Entity mapped to SNOMED-CT concept successfully',
                'mapped_concept': snomed_concept
            })
        else:
            return JsonResponse({'success': False, 'error': 'Invalid entity index'}, status=400)

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Mapping failed: {str(e)}'}, status=500)


@login_required
def annotator_documents(request):
    """List documents assigned to the current annotator with NER/MCN task separation"""
    if request.user.role != 'annotator':
        messages.error(request, 'Annotator access required.')
        return redirect('authentication:dashboard')

    # Get documents assigned to this annotator
    assigned_documents = TextDocument.objects.filter(
        assigned_to=request.user
    ).order_by('-created_at')

    # Create NER and MCN tasks for existing documents if they don't exist
    for doc in assigned_documents:
        # Ensure NER task exists
        ner_task, created = AnnotationTask.objects.get_or_create(
            document=doc,
            annotator=request.user,
            task_type='ner',
            defaults={'assigned_at': timezone.now()}
        )

        # Ensure MCN task exists
        mcn_task, created = AnnotationTask.objects.get_or_create(
            document=doc,
            annotator=request.user,
            task_type='mcn',
            defaults={'assigned_at': timezone.now()}
        )

    # Get NER and MCN tasks separately
    ner_tasks = AnnotationTask.objects.filter(
        annotator=request.user,
        task_type='ner',
        document__in=assigned_documents
    ).select_related('document')

    mcn_tasks = AnnotationTask.objects.filter(
        annotator=request.user,
        task_type='mcn',
        document__in=assigned_documents
    ).select_related('document')

    # Build task status information
    ner_task_data = []
    mcn_task_data = []

    for task in ner_tasks:
        # Get NER annotations for this document
        manual_annotation = AnnotationResult.objects.filter(
            document=task.document,
            annotator=request.user,
            task_type='ner',
            annotation_type='manual'
        ).first()

        llm_assisted_annotation = AnnotationResult.objects.filter(
            document=task.document,
            annotator=request.user,
            task_type='ner',
            annotation_type='llm_assisted'
        ).first()

        # Check completion based on user's annotation mode
        task_completed_based_on_role = task.is_completed_based_on_role()

        ner_task_data.append({
            'task': task,
            'document': task.document,
            'manual_completed': manual_annotation is not None,
            'llm_assisted_completed': llm_assisted_annotation is not None,
            'manual_annotation': manual_annotation,
            'llm_assisted_annotation': llm_assisted_annotation,
            'can_access': True,  # NER is always accessible
            'completed_based_on_role': task_completed_based_on_role,
            'can_do_manual': request.user.can_do_manual_annotation(),
            'can_do_llm_assisted': request.user.can_do_llm_assisted_annotation()
        })

    for task in mcn_tasks:
        # Check if NER is completed for this document using role-based logic
        ner_task = AnnotationTask.get_ner_task(task.document, request.user)
        can_access_mcn = ner_task and ner_task.is_completed_based_on_role()

        # Get MCN annotations for this document
        manual_annotation = AnnotationResult.objects.filter(
            document=task.document,
            annotator=request.user,
            task_type='mcn',
            annotation_type='mcn_manual'
        ).first()

        llm_assisted_annotation = AnnotationResult.objects.filter(
            document=task.document,
            annotator=request.user,
            task_type='mcn',
            annotation_type='mcn_llm_assisted'
        ).first()

        # Check completion based on user's annotation mode
        task_completed_based_on_role = task.is_completed_based_on_role()

        mcn_task_data.append({
            'task': task,
            'document': task.document,
            'manual_completed': manual_annotation is not None,
            'llm_assisted_completed': llm_assisted_annotation is not None,
            'manual_annotation': manual_annotation,
            'llm_assisted_annotation': llm_assisted_annotation,
            'can_access': can_access_mcn,
            'ner_completed': ner_task and ner_task.is_completed_based_on_role() if ner_task else False,
            'completed_based_on_role': task_completed_based_on_role,
            'can_do_manual': request.user.can_do_manual_annotation(),
            'can_do_llm_assisted': request.user.can_do_llm_assisted_annotation()
        })

    # Statistics using role-based completion
    total_ner = len(ner_task_data)
    completed_ner = len([t for t in ner_task_data if t['completed_based_on_role']])
    total_mcn = len(mcn_task_data)
    accessible_mcn = len([t for t in mcn_task_data if t['can_access']])
    completed_mcn = len([t for t in mcn_task_data if t['completed_based_on_role']])

    context = {
        'ner_tasks': ner_task_data,
        'mcn_tasks': mcn_task_data,
        'total_assigned': assigned_documents.count(),
        'total_ner': total_ner,
        'completed_ner': completed_ner,
        'total_mcn': total_mcn,
        'accessible_mcn': accessible_mcn,
        'completed_mcn': completed_mcn,
    }

    return render(request, 'annotation/annotator_documents.html', context)


@login_required
def time_analytics(request):
    """View for annotator time analytics"""
    if request.user.role != 'annotator':
        messages.error(request, 'Annotator access required.')
        return redirect('authentication:dashboard')

    # Get time statistics for NER and MCN tasks
    ner_stats = AnnotationTask.get_time_statistics(
        annotator=request.user,
        task_type='ner'
    )

    mcn_stats = AnnotationTask.get_time_statistics(
        annotator=request.user,
        task_type='mcn'
    )

    # Get recent completed tasks for timeline
    recent_tasks = AnnotationTask.objects.filter(
        annotator=request.user,
        is_completed=True
    ).select_related('document').order_by('-completed_at')[:10]

    # Calculate daily/weekly/monthly statistics
    from datetime import datetime, timedelta
    now = timezone.now()

    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    daily_stats = {
        'ner': AnnotationTask.get_time_statistics(
            annotator=request.user,
            task_type='ner',
            date_from=today_start
        ),
        'mcn': AnnotationTask.get_time_statistics(
            annotator=request.user,
            task_type='mcn',
            date_from=today_start
        )
    }

    weekly_stats = {
        'ner': AnnotationTask.get_time_statistics(
            annotator=request.user,
            task_type='ner',
            date_from=week_start
        ),
        'mcn': AnnotationTask.get_time_statistics(
            annotator=request.user,
            task_type='mcn',
            date_from=week_start
        )
    }

    monthly_stats = {
        'ner': AnnotationTask.get_time_statistics(
            annotator=request.user,
            task_type='ner',
            date_from=month_start
        ),
        'mcn': AnnotationTask.get_time_statistics(
            annotator=request.user,
            task_type='mcn',
            date_from=month_start
        )
    }

    context = {
        'ner_stats': ner_stats,
        'mcn_stats': mcn_stats,
        'recent_tasks': recent_tasks,
        'daily_stats': daily_stats,
        'weekly_stats': weekly_stats,
        'monthly_stats': monthly_stats,
    }

    return render(request, 'annotation/time_analytics.html', context)


# MCN Annotation Views
@login_required
def mcn_manual(request, document_id):
    """Manual MCN interface for normalizing entities to SNOMED-CT"""
    document = get_object_or_404(TextDocument, id=document_id)

    # Check if user can access this document
    if not request.user.is_admin() and not document.is_assigned_to_user(request.user):
        messages.error(request, 'You do not have access to this document.')
        return redirect('authentication:dashboard')

    # Check if user can do manual annotation
    if not request.user.can_do_manual_annotation():
        messages.error(request, 'You are not authorized to perform manual annotations.')
        return redirect('annotation:annotator_documents')

    # Check if NER is completed based on user's annotation role
    ner_task = AnnotationTask.get_ner_task(document, request.user)
    if not ner_task or not ner_task.is_completed_based_on_role():
        messages.error(request, 'Please complete NER annotation first before proceeding to MCN.')
        return redirect('annotation:annotator_documents')

    # Get completed NER annotations
    ner_annotation = AnnotationResult.objects.filter(
        document=document,
        annotator=request.user,
        task_type='ner',
        annotation_type__in=['manual', 'llm_assisted'],
        review_status='completed'
    ).first()

    if not ner_annotation:
        messages.error(request, 'No completed NER annotations found. Please complete NER annotation first.')
        return redirect('annotation:annotator_documents')

    # Get existing MCN annotation for this user
    existing_mcn = AnnotationResult.objects.filter(
        document=document,
        annotator=request.user,
        task_type='mcn',
        annotation_type='mcn_manual'
    ).first()

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            normalized_entities = data.get('entities', [])

            if not normalized_entities:
                return JsonResponse({'success': False, 'error': 'No entities to save'}, status=400)

            # Allow saving even without mappings (user may want to save progress)
            # mapped_count = sum(1 for entity in normalized_entities if entity.get('snomed_concept'))
            # if mapped_count == 0:
            #     return JsonResponse({'success': False, 'error': 'Please map at least one entity to SNOMED-CT before saving'}, status=400)

            # Update existing MCN annotation or create new one
            if existing_mcn:
                existing_mcn.entities_json = normalized_entities
                existing_mcn.review_status = 'completed'
                existing_mcn.save()
                result_annotation = existing_mcn
            else:
                result_annotation = AnnotationResult.objects.create(
                    document=document,
                    annotator=request.user,
                    annotation_type='mcn_manual',
                    task_type='mcn',
                    entities_json=normalized_entities,
                    review_status='completed'
                )

            # Mark MCN task as completed
            mcn_task = AnnotationTask.get_mcn_task(document, request.user)
            if mcn_task and not mcn_task.is_completed:
                mcn_task.complete_task()

            response_data = {
                'success': True,
                'annotation_id': result_annotation.id,
                'message': 'MCN normalization saved successfully'
            }
            gamification_data = _get_gamification_response(mcn_task) if mcn_task else {}
            if not gamification_data:
                gamification_data = _get_gamification_status_response(request.user)
            response_data.update(gamification_data)
            return JsonResponse(response_data)

        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Failed to save MCN: {str(e)}'}, status=500)

    # GET request - show MCN interface
    # Start the MCN task timer when user opens the page
    mcn_task = AnnotationTask.get_mcn_task(document, request.user)
    if mcn_task and not mcn_task.is_started and not mcn_task.is_completed:
        mcn_task.start_task()

    # Filter only approved entities from NER annotation
    all_ner_entities = ner_annotation.entities_json if ner_annotation else []
    approved_entities = [
        entity for entity in all_ner_entities
        if entity.get('status') == 'approved' or 'status' not in entity
    ]

    # Serialize entities_json properly for JavaScript
    existing_mcn_entities_json = '[]'
    if existing_mcn and existing_mcn.entities_json:
        existing_mcn_entities_json = json.dumps(existing_mcn.entities_json)

    context = {
        'document': document,
        'ner_annotation': ner_annotation,
        'existing_mcn': existing_mcn,
        'existing_mcn_entities_json': existing_mcn_entities_json,
        'ner_entities': approved_entities
    }

    return render(request, 'annotation/mcn_manual.html', context)


@login_required
def mcn_llm_assisted(request, document_id):
    """LLM-assisted MCN interface with automatic term normalization"""
    document = get_object_or_404(TextDocument, id=document_id)

    # Check if user can access this document
    if not request.user.is_admin() and not document.is_assigned_to_user(request.user):
        messages.error(request, 'You do not have access to this document.')
        return redirect('authentication:dashboard')

    # Check if user can do LLM-assisted annotation
    if not request.user.can_do_llm_assisted_annotation():
        messages.error(request, 'You are not authorized to perform LLM-assisted annotations.')
        return redirect('annotation:annotator_documents')

    # Check if NER is completed based on user's annotation role
    ner_task = AnnotationTask.get_ner_task(document, request.user)
    if not ner_task or not ner_task.is_completed_based_on_role():
        messages.error(request, 'Please complete NER annotation first before proceeding to MCN.')
        return redirect('annotation:annotator_documents')

    # Get completed NER annotations
    ner_annotation = AnnotationResult.objects.filter(
        document=document,
        annotator=request.user,
        task_type='ner',
        annotation_type__in=['manual', 'llm_assisted'],
        review_status='completed'
    ).first()

    if not ner_annotation:
        messages.error(request, 'No completed NER annotations found. Please complete NER annotation first.')
        return redirect('annotation:annotator_documents')

    # Get existing MCN annotation for this user
    existing_mcn = AnnotationResult.objects.filter(
        document=document,
        annotator=request.user,
        task_type='mcn',
        annotation_type='mcn_llm_assisted'
    ).first()

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            normalized_entities = data.get('entities', [])

            if not normalized_entities:
                return JsonResponse({'success': False, 'error': 'No entities to save'}, status=400)

            # Allow saving even without mappings (user may want to save progress)
            # mapped_count = sum(1 for entity in normalized_entities if entity.get('snomed_concept'))
            # if mapped_count == 0:
            #     return JsonResponse({'success': False, 'error': 'Please map at least one entity to SNOMED-CT before saving'}, status=400)

            # Update existing MCN annotation or create new one
            if existing_mcn:
                existing_mcn.entities_json = normalized_entities
                existing_mcn.review_status = 'completed'
                existing_mcn.save()
                result_annotation = existing_mcn
            else:
                result_annotation = AnnotationResult.objects.create(
                    document=document,
                    annotator=request.user,
                    annotation_type='mcn_llm_assisted',
                    task_type='mcn',
                    model_used='gemini-2.5-flash',
                    entities_json=normalized_entities,
                    review_status='completed'
                )

            # Mark MCN task as completed
            mcn_task = AnnotationTask.get_mcn_task(document, request.user)
            if mcn_task and not mcn_task.is_completed:
                mcn_task.complete_task()

            response_data = {
                'success': True,
                'annotation_id': result_annotation.id,
                'message': 'LLM-assisted MCN normalization saved successfully'
            }
            gamification_data = _get_gamification_response(mcn_task) if mcn_task else {}
            if not gamification_data:
                gamification_data = _get_gamification_status_response(request.user)
            response_data.update(gamification_data)
            return JsonResponse(response_data)

        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'Failed to save MCN: {str(e)}'}, status=500)

    # GET request - show LLM-assisted MCN interface
    # Start the MCN task timer when user opens the page
    mcn_task = AnnotationTask.get_mcn_task(document, request.user)
    if mcn_task and not mcn_task.is_started and not mcn_task.is_completed:
        mcn_task.start_task()

    # Filter only approved entities from NER annotation
    all_ner_entities = ner_annotation.entities_json if ner_annotation else []
    approved_entities = [
        entity for entity in all_ner_entities
        if entity.get('status') == 'approved' or 'status' not in entity
    ]

    context = {
        'document': document,
        'ner_annotation': ner_annotation,
        'existing_mcn': existing_mcn,
        'existing_mcn_json': json.dumps(existing_mcn.entities_json) if existing_mcn and existing_mcn.entities_json else '[]',
        'ner_entities': approved_entities
    }

    return render(request, 'annotation/mcn_llm_assisted.html', context)


@login_required
@require_http_methods(["POST"])
def normalize_medical_terms(request):
    """Use Gemini 2.5 Flash to normalize medical terms to formal concepts"""
    try:
        from .gemini_normalizer import normalize_medical_entities

        data = json.loads(request.body)
        entities = data.get('entities', [])

        if not entities:
            return JsonResponse({'success': False, 'error': 'No entities provided'}, status=400)

        # Use enhanced Gemini normalizer
        normalized_entities = normalize_medical_entities(entities)

        # Add processing statistics
        stats = {
            'total_processed': len(normalized_entities),
            'high_confidence': len([e for e in normalized_entities if e.get('confidence', 0) >= 0.9]),
            'medium_confidence': len([e for e in normalized_entities if 0.7 <= e.get('confidence', 0) < 0.9]),
            'low_confidence': len([e for e in normalized_entities if e.get('confidence', 0) < 0.7]),
        }

        return JsonResponse({
            'success': True,
            'normalized_entities': normalized_entities,
            'total_normalized': len(normalized_entities),
            'statistics': stats,
            'model_used': 'gemini-2.5-flash',
            'processing_time': 0.5  # Mock processing time
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Normalization failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def auto_map_to_snomed(request):
    """Automatically normalize terms and map to SNOMED-CT (first result) for LLM-assisted MCN"""
    try:
        from .gemini_normalizer import normalize_medical_entities
        from .snomed_service import get_snomed_service

        data = json.loads(request.body)
        entities = data.get('entities', [])

        if not entities:
            return JsonResponse({'success': False, 'error': 'No entities provided'}, status=400)

        # Step 1: Normalize using Gemini 2.5 Flash
        normalized_entities = normalize_medical_entities(entities)

        # Step 2: Auto-map to SNOMED-CT
        snomed_service = get_snomed_service()
        mapped_entities = []

        for entity in normalized_entities:
            normalized_term = entity.get('normalized_term', entity.get('text', ''))
            entity_type = entity.get('entity_type', entity.get('label', 'UNKNOWN'))

            # Search SNOMED-CT with the normalized term
            try:
                candidates = snomed_service.search_with_auto_translation(
                    normalized_term,
                    max_results=1,  # Only get the first/best result
                    context=entity_type
                )

                if candidates:
                    # Auto-select the first (best) result
                    best_match = candidates[0]
                    entity.update({
                        'snomed_concept': {
                            'conceptId': best_match['conceptId'],
                            'fsn': best_match['fsn'],
                            'term': best_match['term'],
                            'definition': best_match.get('definition', ''),
                            'confidence_score': best_match.get('confidence_score', 0.8)
                        },
                        'auto_mapped': True,
                        'mapping_confidence': best_match.get('confidence_score', 0.8)
                    })
                else:
                    entity.update({
                        'snomed_concept': None,
                        'auto_mapped': False,
                        'mapping_confidence': 0.0,
                        'mapping_error': 'No SNOMED-CT matches found'
                    })

            except Exception as mapping_error:
                entity.update({
                    'snomed_concept': None,
                    'auto_mapped': False,
                    'mapping_confidence': 0.0,
                    'mapping_error': str(mapping_error)
                })

            mapped_entities.append(entity)

        # Calculate mapping statistics
        mapped_count = len([e for e in mapped_entities if e.get('auto_mapped', False)])
        high_confidence_mappings = len([e for e in mapped_entities if e.get('mapping_confidence', 0) >= 0.8])

        stats = {
            'total_entities': len(mapped_entities),
            'successfully_mapped': mapped_count,
            'high_confidence_mappings': high_confidence_mappings,
            'mapping_rate': (mapped_count / len(mapped_entities)) * 100 if mapped_entities else 0,
            'average_confidence': sum([e.get('mapping_confidence', 0) for e in mapped_entities]) / len(mapped_entities) if mapped_entities else 0
        }

        return JsonResponse({
            'success': True,
            'mapped_entities': mapped_entities,
            'statistics': stats,
            'model_used': 'gemini-2.5-flash + snomed-ct',
            'processing_time': len(mapped_entities) * 0.2  # Mock processing time
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Auto-mapping failed: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def translate_indonesian_entities(request):
    """
    3-Step Indonesian Medical Term Processing:
    Step 1: Indonesian → English Translation
    Step 2: English → Formal Medical Term
    Step 3: Generate SNOMED Search Term
    """
    try:
        from .indonesian_medical_translator import get_indonesian_translator
        from .snomed_service import get_snomed_service

        data = json.loads(request.body)
        entities = data.get('entities', [])
        document_context = data.get('context', '')
        step = data.get('step', 'all')  # 'all', 'translate', 'normalize', 'map'

        if not entities:
            return JsonResponse({'success': False, 'error': 'No entities provided'}, status=400)

        translator = get_indonesian_translator()

        if not translator.is_available():
            return JsonResponse({
                'success': False,
                'error': 'Translation service unavailable. Please check GOOGLE_API_KEY configuration.'
            }, status=503)

        # Step 1+2+3: Batch translate ALL entities in a single Gemini API call (~90% token savings)
        batch_results = translator.process_entities_batch_single_call(entities, document_context)

        # Merge batch results with original entity data
        processed_entities = []
        for i, entity in enumerate(entities):
            result = batch_results[i] if i < len(batch_results) else translator._fallback_processing(
                entity.get('text', ''), entity.get('label', 'UNKNOWN')
            )
            processed_entities.append({**entity, **result})

        # SNOMED lookups: parallelize since these are HTTP calls to external API
        if step in ['all', 'map']:
            def _snomed_lookup(idx_entity):
                idx, entity = idx_entity
                search_term = entity.get('step3_snomed_search', {}).get('search_term', '')
                if not search_term:
                    entity['snomed_results'] = []
                    entity['snomed_best_match'] = None
                    return
                try:
                    snomed_service = get_snomed_service()
                    snomed_candidates = snomed_service.search_with_auto_translation(
                        search_term,
                        max_results=5,
                        context=entity.get('label', '')
                    )
                    if snomed_candidates:
                        entity['snomed_results'] = snomed_candidates
                        entity['snomed_best_match'] = snomed_candidates[0]
                    else:
                        entity['snomed_results'] = []
                        entity['snomed_best_match'] = None
                except Exception as snomed_error:
                    entity['snomed_results'] = []
                    entity['snomed_error'] = str(snomed_error)

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=5) as executor:
                list(executor.map(_snomed_lookup, enumerate(processed_entities)))

        # Calculate statistics
        successful = len([e for e in processed_entities if e.get('processing_status') == 'success'])
        translated = len([e for e in processed_entities if e.get('step1_translation', {}).get('confidence', 0) >= 0.7])
        mapped = len([e for e in processed_entities if e.get('snomed_best_match') is not None])

        stats = {
            'total_entities': len(processed_entities),
            'successfully_processed': successful,
            'successfully_translated': translated,
            'successfully_mapped': mapped,
            'translation_rate': (translated / len(processed_entities) * 100) if processed_entities else 0,
            'mapping_rate': (mapped / len(processed_entities) * 100) if processed_entities else 0
        }

        return JsonResponse({
            'success': True,
            'entities': processed_entities,
            'statistics': stats,
            'model_used': 'gemini-2.5-flash',
            'workflow': '3-step: Indonesian → English → Medical → SNOMED'
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Translation workflow failed: {str(e)}'}, status=500)


# Annotation Statistics Views
@login_required
def admin_statistics_dashboard(request):
    """Comprehensive statistics dashboard for admins"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    from datetime import datetime, timedelta
    from django.db.models import Avg, Sum, Count

    # Get filter parameters
    annotator_filter = request.GET.get('annotator', 'all')
    task_type_filter = request.GET.get('task_type', 'all')
    date_range = request.GET.get('date_range', '30')  # days

    # Calculate date range
    end_date = timezone.now()
    if date_range == '7':
        start_date = end_date - timedelta(days=7)
        period_label = "Last 7 Days"
    elif date_range == '30':
        start_date = end_date - timedelta(days=30)
        period_label = "Last 30 Days"
    elif date_range == '90':
        start_date = end_date - timedelta(days=90)
        period_label = "Last 90 Days"
    else:  # all time
        start_date = None
        period_label = "All Time"

    # Get annotators for filter
    annotators = User.objects.filter(role='annotator', is_active=True)

    # Build base querysets
    task_queryset = AnnotationTask.objects.filter(is_completed=True)
    result_queryset = AnnotationResult.objects.filter(
        annotation_type__in=['llm_assisted', 'mcn_llm_assisted']
    )

    if annotator_filter != 'all':
        task_queryset = task_queryset.filter(annotator__id=annotator_filter)
        result_queryset = result_queryset.filter(annotator__id=annotator_filter)

    if task_type_filter != 'all':
        task_queryset = task_queryset.filter(task_type=task_type_filter)
        result_queryset = result_queryset.filter(task_type=task_type_filter)

    if start_date:
        task_queryset = task_queryset.filter(completed_at__gte=start_date)
        result_queryset = result_queryset.filter(created_at__gte=start_date)

    # Calculate comprehensive statistics
    timing_stats = calculate_timing_statistics(task_queryset)
    llm_stats = calculate_llm_statistics(result_queryset)

    # MCN-specific statistics
    mcn_result_queryset = AnnotationResult.objects.filter(
        annotation_type__in=['mcn_llm_assisted', 'mcn_manual']
    )
    if annotator_filter != 'all':
        mcn_result_queryset = mcn_result_queryset.filter(annotator__id=annotator_filter)
    if task_type_filter != 'all':
        mcn_result_queryset = mcn_result_queryset.filter(task_type=task_type_filter)
    if start_date:
        mcn_result_queryset = mcn_result_queryset.filter(created_at__gte=start_date)
    mcn_stats = calculate_mcn_statistics(mcn_result_queryset)

    annotator_stats = calculate_annotator_performance(annotators, start_date, task_type_filter)

    # Generate or update cached statistics
    if annotator_filter != 'all':
        selected_annotator = User.objects.get(id=annotator_filter)
        cached_stats = AnnotationStatistics.calculate_and_store_statistics(
            annotator=selected_annotator,
            task_type=task_type_filter,
            period_type='custom',
            period_start=start_date,
            period_end=end_date
        )
    else:
        cached_stats = AnnotationStatistics.calculate_and_store_statistics(
            annotator=None,
            task_type=task_type_filter,
            period_type='custom',
            period_start=start_date,
            period_end=end_date
        )

    context = {
        'timing_stats': timing_stats,
        'llm_stats': llm_stats,
        'mcn_stats': mcn_stats,
        'annotator_stats': annotator_stats,
        'cached_stats': cached_stats,
        'annotators': annotators,
        'annotator_filter': annotator_filter,
        'task_type_filter': task_type_filter,
        'date_range': date_range,
        'period_label': period_label,
        'start_date': start_date,
        'end_date': end_date,
    }

    return render(request, 'annotation/admin_statistics_dashboard.html', context)


def deduplicate_tasks(task_queryset):
    """
    Deduplicate tasks by keeping only the most recent task per (document, annotator, task_type).
    This prevents double-counting if users accidentally create duplicate tasks.
    """
    unique_tasks = []
    seen_combinations = set()

    for task in task_queryset.order_by('-assigned_at'):
        combo = (task.document_id, task.annotator_id, task.task_type)
        if combo not in seen_combinations:
            unique_tasks.append(task)
            seen_combinations.add(combo)

    # Return deduplicated queryset
    if unique_tasks:
        return AnnotationTask.objects.filter(id__in=[t.id for t in unique_tasks])
    else:
        return task_queryset.none()


def calculate_timing_statistics(task_queryset):
    """Calculate timing statistics from task queryset"""
    # Deduplicate tasks to prevent counting duplicates
    task_queryset = deduplicate_tasks(task_queryset)

    stats = {
        'total_tasks': task_queryset.count(),
        'total_time_minutes': 0,
        'average_time_minutes': 0,
        'min_time_minutes': None,
        'max_time_minutes': None,
        'ner_stats': {'total': 0, 'avg_time': 0, 'total_time': 0},
        'mcn_stats': {'total': 0, 'avg_time': 0, 'total_time': 0}
    }

    if stats['total_tasks'] > 0:
        # Overall statistics
        completed_tasks = list(task_queryset.exclude(actual_duration__isnull=True))
        if completed_tasks:
            durations = [task.get_duration_minutes() for task in completed_tasks]
            stats['total_time_minutes'] = sum(durations)
            stats['average_time_minutes'] = sum(durations) / len(durations)
            stats['min_time_minutes'] = min(durations)
            stats['max_time_minutes'] = max(durations)

        # Task type breakdown
        ner_tasks = task_queryset.filter(task_type='ner').exclude(actual_duration__isnull=True)
        mcn_tasks = task_queryset.filter(task_type='mcn').exclude(actual_duration__isnull=True)

        if ner_tasks.exists():
            ner_durations = [task.get_duration_minutes() for task in ner_tasks]
            stats['ner_stats'] = {
                'total': ner_tasks.count(),
                'avg_time': sum(ner_durations) / len(ner_durations),
                'total_time': sum(ner_durations)
            }

        if mcn_tasks.exists():
            mcn_durations = [task.get_duration_minutes() for task in mcn_tasks]
            stats['mcn_stats'] = {
                'total': mcn_tasks.count(),
                'avg_time': sum(mcn_durations) / len(mcn_durations),
                'total_time': sum(mcn_durations)
            }

    return stats


def calculate_llm_statistics(result_queryset):
    """Calculate NER LLM prediction statistics.
    Filters out MCN results - those are handled by calculate_mcn_statistics()."""
    # Filter to NER-only LLM results (exclude MCN)
    ner_results = result_queryset.exclude(annotation_type__in=['mcn_llm_assisted', 'mcn_manual'])

    stats = {
        'total_results': ner_results.count(),
        'total_predictions': 0,
        'accepted': 0,
        'rejected': 0,
        'manual_added': 0,  # Manual entities added by annotator in LLM mode
        'acceptance_rate': 0,
        'rejection_rate': 0,
    }

    for result in ner_results:
        for entity in result.entities_json:
            source = entity.get('source', 'llm')
            if source == 'llm':
                stats['total_predictions'] += 1
                status = entity.get('status', 'pending')

                if status == 'accepted' or status == 'approved':
                    stats['accepted'] += 1
                elif status == 'rejected':
                    stats['rejected'] += 1
            elif source == 'manual':
                stats['manual_added'] += 1

    # Calculate rates
    if stats['total_predictions'] > 0:
        total_decided = stats['accepted'] + stats['rejected']
        if total_decided > 0:
            stats['acceptance_rate'] = (stats['accepted'] / total_decided) * 100
            stats['rejection_rate'] = (stats['rejected'] / total_decided) * 100

    return stats


def calculate_mcn_statistics(result_queryset):
    """Calculate MCN-specific statistics separate from NER LLM statistics.
    Processes entities with mcn_source/mcn_status fields."""
    stats = {
        'total_results': result_queryset.count(),
        'total_entities': 0,
        # LLM-assisted MCN
        'llm_total': 0,
        'llm_accepted': 0,    # Kept LLM SNOMED suggestion
        'llm_modified': 0,    # Changed to different SNOMED term
        'llm_pending': 0,     # Left unmapped
        'llm_acceptance_rate': 0,
        'llm_modification_rate': 0,
        'llm_mapping_rate': 0,
        # Manual MCN
        'manual_total': 0,
        'manual_mapped': 0,
        'manual_unmapped': 0,
        'manual_mapping_rate': 0,
        # Overall
        'total_mapped': 0,
        'overall_mapping_rate': 0,
    }

    for result in result_queryset:
        for entity in result.entities_json:
            mcn_source = entity.get('mcn_source')
            mcn_status = entity.get('mcn_status')

            # Skip entities without MCN tracking fields (legacy data)
            if not mcn_source:
                # Fallback: infer from annotation_type and is_mapped
                if result.annotation_type == 'mcn_llm_assisted':
                    mcn_source = 'llm'
                    is_mapped = entity.get('is_mapped', False)
                    mcn_status = 'accepted' if is_mapped else 'pending'
                elif result.annotation_type == 'mcn_manual':
                    mcn_source = 'manual'
                    is_mapped = entity.get('is_mapped', False)
                    mcn_status = 'mapped' if is_mapped else 'unmapped'
                else:
                    continue

            stats['total_entities'] += 1

            if mcn_source == 'llm':
                stats['llm_total'] += 1
                if mcn_status == 'accepted':
                    stats['llm_accepted'] += 1
                    stats['total_mapped'] += 1
                elif mcn_status == 'modified':
                    stats['llm_modified'] += 1
                    stats['total_mapped'] += 1
                elif mcn_status == 'pending':
                    stats['llm_pending'] += 1
            elif mcn_source == 'manual':
                stats['manual_total'] += 1
                if mcn_status == 'mapped':
                    stats['manual_mapped'] += 1
                    stats['total_mapped'] += 1
                else:
                    stats['manual_unmapped'] += 1

    # Calculate rates
    if stats['llm_total'] > 0:
        llm_decided = stats['llm_accepted'] + stats['llm_modified']
        stats['llm_mapping_rate'] = (llm_decided / stats['llm_total']) * 100
        if llm_decided > 0:
            stats['llm_acceptance_rate'] = (stats['llm_accepted'] / llm_decided) * 100
            stats['llm_modification_rate'] = (stats['llm_modified'] / llm_decided) * 100

    if stats['manual_total'] > 0:
        stats['manual_mapping_rate'] = (stats['manual_mapped'] / stats['manual_total']) * 100

    if stats['total_entities'] > 0:
        stats['overall_mapping_rate'] = (stats['total_mapped'] / stats['total_entities']) * 100

    return stats


def calculate_annotator_performance(annotators, start_date=None, task_type_filter='all'):
    """Calculate performance statistics for each annotator"""
    annotator_performance = []

    for annotator in annotators:
        task_queryset = AnnotationTask.objects.filter(
            annotator=annotator,
            is_completed=True
        )

        if start_date:
            task_queryset = task_queryset.filter(completed_at__gte=start_date)

        if task_type_filter != 'all':
            task_queryset = task_queryset.filter(task_type=task_type_filter)

        # Calculate statistics for this annotator
        timing_stats = calculate_timing_statistics(task_queryset)

        # Get NER LLM statistics for this annotator
        llm_results = AnnotationResult.objects.filter(
            annotator=annotator,
            annotation_type__in=['llm_assisted', 'mcn_llm_assisted']
        )

        if start_date:
            llm_results = llm_results.filter(created_at__gte=start_date)

        if task_type_filter != 'all':
            llm_results = llm_results.filter(task_type=task_type_filter)

        llm_stats = calculate_llm_statistics(llm_results)

        # Get MCN statistics for this annotator
        mcn_results = AnnotationResult.objects.filter(
            annotator=annotator,
            annotation_type__in=['mcn_llm_assisted', 'mcn_manual']
        )
        if start_date:
            mcn_results = mcn_results.filter(created_at__gte=start_date)
        if task_type_filter != 'all':
            mcn_results = mcn_results.filter(task_type=task_type_filter)
        mcn_stats = calculate_mcn_statistics(mcn_results)

        # Calculate entity source statistics (manual vs LLM)
        entity_source_stats = calculate_entity_source_statistics(annotator, start_date, task_type_filter)

        annotator_performance.append({
            'annotator': annotator,
            'timing': timing_stats,
            'llm': llm_stats,
            'mcn': mcn_stats,
            'entity_source': entity_source_stats,
            'productivity_score': calculate_productivity_score(timing_stats, llm_stats)
        })

    return annotator_performance


def calculate_entity_source_statistics(annotator, start_date=None, task_type_filter='all'):
    """Calculate entity statistics broken down by source (manual vs LLM)"""
    from collections import defaultdict

    # Get all results for this annotator
    results_qs = AnnotationResult.objects.filter(annotator=annotator)

    if start_date:
        results_qs = results_qs.filter(created_at__gte=start_date)

    if task_type_filter != 'all':
        results_qs = results_qs.filter(task_type=task_type_filter)

    stats = {
        'total_entities': 0,
        'active_entities': 0,  # Excludes rejected
        'manual_entities': 0,
        'manual_in_llm_mode': 0,  # Manual entities added during LLM-assisted annotation
        'manual_pure': 0,  # Manual entities in pure manual mode
        'llm_entities_accepted': 0,
        'llm_entities_rejected': 0,
        'llm_entities_pending': 0,
        'llm_entities_added_manually': 0,  # Alias for manual_in_llm_mode
        'manual_percentage': 0,
        'llm_accepted_percentage': 0,
        'llm_rejected_percentage': 0,
        'manual_in_llm_percentage': 0,
        'by_entity_type': defaultdict(lambda: {
            'manual': 0,
            'manual_in_llm': 0,
            'llm_accepted': 0,
            'llm_rejected': 0,
            'total': 0,
            'active': 0
        })
    }

    for result in results_qs:
        is_llm_assisted = result.annotation_type in ['llm_assisted', 'mcn_llm_assisted']
        is_mcn = result.annotation_type in ['mcn_llm_assisted', 'mcn_manual']

        for entity in result.entities_json:
            entity_type = entity.get('label', 'UNKNOWN')

            # MCN entities use mcn_source/mcn_status fields
            if is_mcn:
                mcn_source = entity.get('mcn_source')
                if mcn_source:
                    # MCN entities count toward totals but use MCN-specific tracking
                    stats['total_entities'] += 1
                    stats['by_entity_type'][entity_type]['total'] += 1
                    stats['active_entities'] += 1
                    stats['by_entity_type'][entity_type]['active'] += 1

                    if mcn_source == 'manual':
                        stats['manual_entities'] += 1
                        stats['manual_pure'] += 1
                        stats['by_entity_type'][entity_type]['manual'] += 1
                    elif mcn_source == 'llm':
                        mcn_status = entity.get('mcn_status', 'pending')
                        if mcn_status == 'accepted':
                            stats['llm_entities_accepted'] += 1
                            stats['by_entity_type'][entity_type]['llm_accepted'] += 1
                        elif mcn_status == 'modified':
                            stats['llm_entities_accepted'] += 1  # Modified SNOMED still counts as used
                            stats['by_entity_type'][entity_type]['llm_accepted'] += 1
                    continue
                # Fallback for legacy MCN without mcn_source: count as manual
                stats['total_entities'] += 1
                stats['by_entity_type'][entity_type]['total'] += 1
                stats['manual_entities'] += 1
                stats['manual_pure'] += 1
                stats['by_entity_type'][entity_type]['manual'] += 1
                stats['active_entities'] += 1
                stats['by_entity_type'][entity_type]['active'] += 1
                continue

            # NER entities use source/status fields
            source = entity.get('source', 'manual')
            status = entity.get('status', 'pending')

            stats['total_entities'] += 1
            stats['by_entity_type'][entity_type]['total'] += 1

            if source == 'manual':
                stats['manual_entities'] += 1
                stats['by_entity_type'][entity_type]['manual'] += 1

                # Track whether this manual entity was added in LLM-assisted mode
                if is_llm_assisted:
                    stats['manual_in_llm_mode'] += 1
                    stats['llm_entities_added_manually'] += 1
                    stats['by_entity_type'][entity_type]['manual_in_llm'] += 1
                else:
                    stats['manual_pure'] += 1

                stats['active_entities'] += 1
                stats['by_entity_type'][entity_type]['active'] += 1
            elif source == 'llm':
                if status == 'accepted' or status == 'approved':
                    stats['llm_entities_accepted'] += 1
                    stats['by_entity_type'][entity_type]['llm_accepted'] += 1
                    stats['active_entities'] += 1
                    stats['by_entity_type'][entity_type]['active'] += 1
                elif status == 'rejected':
                    stats['llm_entities_rejected'] += 1
                    stats['by_entity_type'][entity_type]['llm_rejected'] += 1
                elif status == 'pending':
                    stats['llm_entities_pending'] += 1
                    stats['active_entities'] += 1
                    stats['by_entity_type'][entity_type]['active'] += 1

    # Calculate percentages based on total entities (including rejected)
    if stats['total_entities'] > 0:
        stats['manual_percentage'] = (stats['manual_entities'] / stats['total_entities']) * 100
        stats['llm_accepted_percentage'] = (stats['llm_entities_accepted'] / stats['total_entities']) * 100
        stats['llm_rejected_percentage'] = (stats['llm_entities_rejected'] / stats['total_entities']) * 100
        stats['manual_in_llm_percentage'] = (stats['manual_in_llm_mode'] / stats['total_entities']) * 100

    # Count entities actually used (manual + accepted LLM)
    stats['entities_used'] = (stats['manual_entities'] +
                              stats['llm_entities_accepted'])

    # LLM agreement rate: how much the annotator agreed with LLM (accepted / total LLM decided)
    total_llm_decided = (stats['llm_entities_accepted'] +
                         stats['llm_entities_rejected'])
    if total_llm_decided > 0:
        stats['llm_agreement_rate'] = (stats['llm_entities_accepted'] /
                                       total_llm_decided) * 100
        stats['llm_pure_acceptance_rate'] = (stats['llm_entities_accepted'] / total_llm_decided) * 100
    else:
        stats['llm_agreement_rate'] = 0
        stats['llm_pure_acceptance_rate'] = 0

    # Convert defaultdict to regular dict for JSON serialization
    stats['by_entity_type'] = dict(stats['by_entity_type'])

    return stats


def calculate_productivity_score(timing_stats, llm_stats):
    """Calculate a simple productivity score based on timing and accuracy"""
    # Base score on number of completed tasks
    base_score = min(timing_stats['total_tasks'] * 10, 100)

    # Adjust for speed (lower average time = higher score)
    if timing_stats['average_time_minutes'] > 0:
        # Normalize to 0-20 range (assuming 10 minutes is optimal)
        speed_bonus = max(0, 20 - (timing_stats['average_time_minutes'] - 10))
        speed_bonus = min(speed_bonus, 20)
    else:
        speed_bonus = 0

    # Adjust for LLM accuracy (higher acceptance rate = higher score)
    accuracy_bonus = llm_stats['acceptance_rate'] * 0.2  # 0-20 range

    total_score = min(base_score + speed_bonus + accuracy_bonus, 100)
    return round(total_score, 1)


@login_required
@require_http_methods(["POST"])
def refresh_statistics(request):
    """Refresh/recalculate statistics for specified parameters"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    try:
        data = json.loads(request.body)
        annotator_id = data.get('annotator_id')
        task_type = data.get('task_type', 'all')
        period_type = data.get('period_type', 'all_time')

        # Calculate date range based on period_type
        end_date = timezone.now()
        start_date = None

        if period_type == 'daily':
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period_type == 'weekly':
            start_date = end_date - timedelta(days=7)
        elif period_type == 'monthly':
            start_date = end_date - timedelta(days=30)

        # Get annotator if specified
        annotator = None
        if annotator_id and annotator_id != 'all':
            try:
                annotator = User.objects.get(id=annotator_id, role='annotator')
            except User.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Invalid annotator'}, status=400)

        # Calculate and store statistics
        stats = AnnotationStatistics.calculate_and_store_statistics(
            annotator=annotator,
            task_type=task_type,
            period_type=period_type,
            period_start=start_date,
            period_end=end_date
        )

        return JsonResponse({
            'success': True,
            'message': 'Statistics refreshed successfully',
            'stats': {
                'total_tasks': stats.total_tasks_completed,
                'avg_time': stats.average_annotation_time_minutes,
                'acceptance_rate': stats.acceptance_rate,
                'rejection_rate': stats.rejection_rate,
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to refresh statistics: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def batch_llm_annotate(request):
    """Perform LLM annotation on multiple documents"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    try:
        document_ids = request.POST.getlist('document_ids')
        model_id = request.POST.get('model', 'gemini-2.5-flash')
        force_reannotate = request.POST.get('force_reannotate', 'false').lower() == 'true'

        if not document_ids:
            return JsonResponse({'success': False, 'error': 'No documents selected'}, status=400)

        # Get documents
        documents = TextDocument.objects.filter(id__in=document_ids)
        if documents.count() != len(document_ids):
            return JsonResponse({'success': False, 'error': 'Some documents not found'}, status=400)

        # Create annotator
        annotator = create_medical_annotator(model_id=model_id)

        # Check API setup
        api_status = annotator.validate_api_setup()
        if not api_status['ready_for_annotation']:
            return JsonResponse({
                'success': False,
                'error': 'API not properly configured. Please check your environment variables.',
                'api_status': api_status
            }, status=400)

        # Process documents in batch
        results = []
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        start_time = time.time()

        with transaction.atomic():
            for document in documents:
                try:
                    print(f"Processing document {document.id} with model {model_id}")

                    # Check for existing annotation
                    existing_annotation = AnnotationResult.objects.filter(
                        document=document,
                        annotation_type='llm',
                        model_used=model_id
                    ).first()

                    if existing_annotation:
                        if force_reannotate:
                            # Delete existing annotation to re-annotate
                            print(f"Document {document.id} already annotated, but force_reannotate=True, deleting old annotation")
                            existing_annotation.delete()
                        else:
                            # Skip if already annotated and not forcing
                            print(f"Document {document.id} already annotated, skipping")
                            results.append({
                                'document_id': document.id,
                                'status': 'skipped',
                                'message': 'Already annotated with this model',
                                'entities_count': len(existing_annotation.entities_json)
                            })
                            skipped_count += 1
                            continue

                    # Perform annotation
                    doc_start_time = time.time()
                    result = annotator.annotate_document(document.content)
                    processing_time = time.time() - doc_start_time

                    print(f"Document {document.id} annotation result: success={result['success']}, entities={len(result.get('entities', []))}")

                    if result['success']:
                        # Resolve overlapping entities: keep longest span only
                        clean_entities = resolve_overlaps_longest_span(result['entities'])

                        # Save annotation result
                        annotation_result = AnnotationResult.objects.create(
                            document=document,
                            annotator=request.user,
                            annotation_type='llm',
                            model_used=model_id,
                            entities_json=clean_entities,
                            processing_time=processing_time
                        )

                        # Update document status if it was pending
                        if document.status == 'pending':
                            document.status = 'in_progress'
                            document.save()

                        results.append({
                            'document_id': document.id,
                            'status': 'success',
                            'annotation_id': annotation_result.id,
                            'entities_count': len(clean_entities),
                            'processing_time': processing_time
                        })
                        successful_count += 1
                    else:
                        error_msg = result.get('error', 'Annotation processing failed')
                        print(f"Document {document.id} failed: {error_msg}")
                        results.append({
                            'document_id': document.id,
                            'status': 'failed',
                            'message': error_msg
                        })
                        failed_count += 1

                except Exception as e:
                    print(f"Exception processing document {document.id}: {type(e).__name__}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'document_id': document.id,
                        'status': 'error',
                        'message': str(e)
                    })
                    failed_count += 1

        total_time = time.time() - start_time

        return JsonResponse({
            'success': True,
            'message': f'Batch annotation completed: {successful_count} successful, {skipped_count} skipped, {failed_count} failed',
            'results': results,
            'statistics': {
                'total_processed': len(document_ids),
                'successful': successful_count,
                'skipped': skipped_count,
                'failed': failed_count,
                'total_time': total_time,
                'average_time_per_document': total_time / len(document_ids) if document_ids else 0,
                'model_used': model_id
            }
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Batch annotation failed: {str(e)}'}, status=500)


@login_required
def export_statistics(request):
    """Export statistics data as CSV or JSON - supports summary and raw data"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    from django.http import HttpResponse
    import csv
    import json

    export_format = request.GET.get('format', 'csv')
    data_type = request.GET.get('data_type', 'summary')  # summary, raw_annotations, tasks, documents
    annotator_filter = request.GET.get('annotator', 'all')
    task_type_filter = request.GET.get('task_type', 'all')

    # Route to appropriate export handler
    if data_type == 'summary':
        return _export_summary_statistics(request, export_format, annotator_filter, task_type_filter)
    elif data_type == 'raw_annotations':
        return _export_raw_annotations(request, export_format, annotator_filter, task_type_filter)
    elif data_type == 'tasks':
        return _export_tasks(request, export_format, annotator_filter, task_type_filter)
    elif data_type == 'documents':
        return _export_documents(request, export_format, annotator_filter, task_type_filter)
    else:
        messages.error(request, 'Invalid data type specified.')
        return redirect('annotation:admin_statistics_dashboard')


def _export_summary_statistics(request, export_format, annotator_filter, task_type_filter):
    """Export summary statistics from AnnotationStatistics model"""
    from django.http import HttpResponse
    import csv
    import json

    # Get statistics
    queryset = AnnotationStatistics.objects.all()

    if annotator_filter != 'all':
        queryset = queryset.filter(annotator__id=annotator_filter)

    if task_type_filter != 'all':
        queryset = queryset.filter(task_type=task_type_filter)

    queryset = queryset.order_by('-calculated_at')

    if export_format == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="annotation_statistics_summary.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Annotator', 'Task Type', 'Period', 'Total Tasks', 'Avg Time (min)',
            'Total LLM Predictions', 'Acceptance Rate (%)', 'Modification Rate (%)',
            'Rejection Rate (%)', 'Avg Edit Distance', 'Calculated At'
        ])

        for stat in queryset:
            writer.writerow([
                stat.annotator.username if stat.annotator else 'All Annotators',
                stat.get_task_type_display(),
                stat.get_period_type_display(),
                stat.total_tasks_completed,
                round(stat.average_annotation_time_minutes, 2),
                stat.total_llm_predictions,
                round(stat.acceptance_rate, 2),
                round(stat.modification_rate, 2),
                round(stat.rejection_rate, 2),
                round(stat.average_edit_distance, 2),
                stat.calculated_at.strftime('%Y-%m-%d %H:%M:%S')
            ])

        return response

    else:  # JSON format
        data = []
        for stat in queryset:
            data.append({
                'annotator': stat.annotator.username if stat.annotator else 'All Annotators',
                'task_type': stat.task_type,
                'period_type': stat.period_type,
                'total_tasks_completed': stat.total_tasks_completed,
                'average_annotation_time_minutes': stat.average_annotation_time_minutes,
                'total_llm_predictions': stat.total_llm_predictions,
                'acceptance_rate': stat.acceptance_rate,
                'modification_rate': stat.modification_rate,
                'rejection_rate': stat.rejection_rate,
                'average_edit_distance': stat.average_edit_distance,
                'calculated_at': stat.calculated_at.isoformat()
            })

        response = HttpResponse(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        response['Content-Disposition'] = 'attachment; filename="annotation_statistics_summary.json"'
        return response


def _export_raw_annotations(request, export_format, annotator_filter, task_type_filter):
    """Export raw annotation results with detailed entity information"""
    from django.http import HttpResponse
    import csv
    import json

    # Get annotation results
    queryset = AnnotationResult.objects.select_related('document', 'annotator').all()

    if annotator_filter != 'all':
        queryset = queryset.filter(annotator__id=annotator_filter)

    if task_type_filter != 'all':
        queryset = queryset.filter(task_type=task_type_filter)

    queryset = queryset.order_by('-created_at')

    if export_format == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="annotation_raw_data.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Document ID', 'Document Title', 'Annotator', 'Task Type', 'Annotation Type',
            'Entity Text', 'Entity Label', 'Entity Start', 'Entity End', 'Entity Source',
            'Entity Status', 'Original Text', 'Modified Text', 'Edit Distance', 'Confidence',
            'Model Used', 'Processing Time (s)', 'Created At'
        ])

        for result in queryset:
            # Export each entity as a separate row
            if result.entities_json:
                for entity in result.entities_json:
                    writer.writerow([
                        result.document.id,
                        result.document.title or f"Document {result.document.id}",
                        result.annotator.username,
                        result.get_task_type_display(),
                        result.get_annotation_type_display(),
                        entity.get('text', ''),
                        entity.get('label', ''),
                        entity.get('start', ''),
                        entity.get('end', ''),
                        entity.get('source', 'manual'),
                        entity.get('status', 'N/A'),
                        entity.get('original_text', ''),
                        entity.get('modified_text', ''),
                        entity.get('edit_distance', ''),
                        entity.get('confidence', ''),
                        result.model_used,
                        result.processing_time or '',
                        result.created_at.strftime('%Y-%m-%d %H:%M:%S')
                    ])
            else:
                # If no entities, still include a row for the annotation
                writer.writerow([
                    result.document.id,
                    result.document.title or f"Document {result.document.id}",
                    result.annotator.username,
                    result.get_task_type_display(),
                    result.get_annotation_type_display(),
                    '', '', '', '', '', '', '', '', '', '',
                    result.model_used,
                    result.processing_time or '',
                    result.created_at.strftime('%Y-%m-%d %H:%M:%S')
                ])

        return response

    else:  # JSON format
        data = []
        for result in queryset:
            # Separate entities by status
            all_entities = result.entities_json or []
            approved_entities = [e for e in all_entities if e.get('status') in ('approved', 'accepted')]
            rejected_entities = [e for e in all_entities if e.get('status') == 'rejected']
            modified_entities = [e for e in all_entities if e.get('status') == 'modified']
            pending_entities = [e for e in all_entities if e.get('status', 'pending') == 'pending']
            manual_entities = [e for e in all_entities if e.get('source') == 'manual']

            data.append({
                'document_id': result.document.id,
                'document_title': result.document.title or f"Document {result.document.id}",
                'document_content': result.document.content,
                'annotator': result.annotator.username,
                'task_type': result.task_type,
                'annotation_type': result.annotation_type,
                'model_used': result.model_used,
                'processing_time_seconds': result.processing_time,
                'confidence_score': result.confidence_score,
                'review_status': result.review_status,
                'created_at': result.created_at.isoformat(),
                # Entity counts breakdown
                'entity_counts': {
                    'total': len(all_entities),
                    'approved': len(approved_entities),
                    'rejected': len(rejected_entities),
                    'modified': len(modified_entities),
                    'pending': len(pending_entities),
                    'manual_added': len(manual_entities),
                    'active': result.total_entities,  # non-rejected count
                },
                # Entities separated by status
                'approved_entities': approved_entities,
                'rejected_entities': rejected_entities,
                'modified_entities': modified_entities,
                'pending_entities': pending_entities,
                'manual_entities': manual_entities,
                # Full entity list (all statuses combined)
                'all_entities': all_entities,
                'decisions': result.decisions_json,
                'decision_summary': result.get_llm_decision_summary()
            })

        response = HttpResponse(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        response['Content-Disposition'] = 'attachment; filename="annotation_raw_data.json"'
        return response


def _export_tasks(request, export_format, annotator_filter, task_type_filter):
    """Export task assignment and completion data"""
    from django.http import HttpResponse
    import csv
    import json

    # Get tasks
    queryset = AnnotationTask.objects.select_related('document', 'annotator').all()

    if annotator_filter != 'all':
        queryset = queryset.filter(annotator__id=annotator_filter)

    if task_type_filter != 'all':
        queryset = queryset.filter(task_type=task_type_filter)

    queryset = queryset.order_by('-assigned_at')

    if export_format == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="annotation_tasks.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Task ID', 'Document ID', 'Document Title', 'Annotator', 'Task Type',
            'Assigned At', 'Started At', 'Completed At', 'Is Started', 'Is Completed',
            'Duration (minutes)', 'Document Status', 'Notes'
        ])

        for task in queryset:
            writer.writerow([
                task.id,
                task.document.id,
                task.document.title or f"Document {task.document.id}",
                task.annotator.username,
                task.get_task_type_display(),
                task.assigned_at.strftime('%Y-%m-%d %H:%M:%S'),
                task.started_at.strftime('%Y-%m-%d %H:%M:%S') if task.started_at else '',
                task.completed_at.strftime('%Y-%m-%d %H:%M:%S') if task.completed_at else '',
                task.is_started,
                task.is_completed,
                task.get_duration_minutes(),
                task.document.get_status_display(),
                task.notes
            ])

        return response

    else:  # JSON format
        data = []
        for task in queryset:
            data.append({
                'task_id': task.id,
                'document_id': task.document.id,
                'document_title': task.document.title or f"Document {task.document.id}",
                'document_content': task.document.content,
                'document_word_count': task.document.word_count,
                'annotator': task.annotator.username,
                'task_type': task.task_type,
                'assigned_at': task.assigned_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'is_started': task.is_started,
                'is_completed': task.is_completed,
                'duration_minutes': task.get_duration_minutes(),
                'duration_display': task.get_duration_display(),
                'document_status': task.document.status,
                'notes': task.notes
            })

        response = HttpResponse(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        response['Content-Disposition'] = 'attachment; filename="annotation_tasks.json"'
        return response


def _export_documents(request, export_format, annotator_filter, task_type_filter):
    """Export document data with status and assignment information"""
    from django.http import HttpResponse
    import csv
    import json

    # Get documents
    queryset = TextDocument.objects.prefetch_related('assigned_to').all()

    if annotator_filter != 'all':
        queryset = queryset.filter(assigned_to__id=annotator_filter)

    queryset = queryset.order_by('-created_at')

    if export_format == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="documents.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Document ID', 'Title', 'Content Preview', 'Word Count', 'Character Count',
            'Status', 'Assigned To', 'Source File', 'Line Number', 'Uploaded By',
            'Created At', 'Updated At'
        ])

        for doc in queryset:
            writer.writerow([
                doc.id,
                doc.title or '',
                doc.get_content_preview(200),
                doc.word_count,
                doc.character_count,
                doc.get_status_display(),
                doc.get_assigned_annotators_names(),
                doc.source_file,
                doc.line_number or '',
                doc.uploaded_by.username,
                doc.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                doc.updated_at.strftime('%Y-%m-%d %H:%M:%S')
            ])

        return response

    else:  # JSON format
        data = []
        for doc in queryset:
            data.append({
                'document_id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'word_count': doc.word_count,
                'character_count': doc.character_count,
                'status': doc.status,
                'assigned_to': [user.username for user in doc.assigned_to.all()],
                'source_file': doc.source_file,
                'line_number': doc.line_number,
                'uploaded_by': doc.uploaded_by.username,
                'created_at': doc.created_at.isoformat(),
                'updated_at': doc.updated_at.isoformat()
            })

        response = HttpResponse(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        response['Content-Disposition'] = 'attachment; filename="documents.json"'
        return response


# Guideline Management Views
@login_required
def admin_guidelines(request):
    """Admin view for managing guidelines"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    guidelines = Guideline.objects.all().order_by('-created_at')

    # Pagination
    paginator = Paginator(guidelines, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'total_guidelines': guidelines.count(),
        'active_guidelines': guidelines.filter(is_active=True).count(),
    }

    return render(request, 'annotation/admin_guidelines.html', context)


@login_required
def upload_guideline(request):
    """Admin view for uploading new guidelines"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        description = request.POST.get('description', '').strip()
        content = request.POST.get('content', '').strip()
        is_active = request.POST.get('is_active') == 'on'

        if not title or not content:
            messages.error(request, 'Title and content are required.')
            return render(request, 'annotation/upload_guideline.html')

        try:
            guideline = Guideline.objects.create(
                title=title,
                description=description,
                content=content,
                is_active=is_active,
                uploaded_by=request.user
            )

            messages.success(request, f'Guideline "{title}" uploaded successfully.')
            return redirect('annotation:admin_guidelines')

        except Exception as e:
            messages.error(request, f'Failed to upload guideline: {str(e)}')
            return render(request, 'annotation/upload_guideline.html')

    return render(request, 'annotation/upload_guideline.html')


@login_required
def edit_guideline(request, guideline_id):
    """Admin view for editing existing guidelines"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    guideline = get_object_or_404(Guideline, id=guideline_id)

    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        description = request.POST.get('description', '').strip()
        content = request.POST.get('content', '').strip()
        is_active = request.POST.get('is_active') == 'on'

        if not title or not content:
            messages.error(request, 'Title and content are required.')
            return render(request, 'annotation/edit_guideline.html', {'guideline': guideline})

        try:
            guideline.title = title
            guideline.description = description
            guideline.content = content
            guideline.is_active = is_active
            guideline.save()

            messages.success(request, f'Guideline "{title}" updated successfully.')
            return redirect('annotation:admin_guidelines')

        except Exception as e:
            messages.error(request, f'Failed to update guideline: {str(e)}')

    context = {
        'guideline': guideline,
    }

    return render(request, 'annotation/edit_guideline.html', context)


@login_required
@require_http_methods(["POST"])
def delete_guideline(request, guideline_id):
    """Admin endpoint for deleting guidelines"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    guideline = get_object_or_404(Guideline, id=guideline_id)

    try:
        guideline_title = guideline.title
        guideline.delete()

        return JsonResponse({
            'success': True,
            'message': f'Guideline "{guideline_title}" deleted successfully'
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to delete guideline: {str(e)}'}, status=500)


@login_required
@require_http_methods(["POST"])
def toggle_guideline_status(request, guideline_id):
    """Admin endpoint for toggling guideline active status"""
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    guideline = get_object_or_404(Guideline, id=guideline_id)

    try:
        guideline.is_active = not guideline.is_active
        guideline.save()

        status = "activated" if guideline.is_active else "deactivated"
        return JsonResponse({
            'success': True,
            'message': f'Guideline "{guideline.title}" {status} successfully',
            'is_active': guideline.is_active
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': f'Failed to update guideline status: {str(e)}'}, status=500)


@login_required
def view_guidelines(request):
    """Annotator view for viewing active guidelines"""
    if request.user.role not in ['annotator', 'admin']:
        messages.error(request, 'Access denied.')
        return redirect('authentication:dashboard')

    # Get only active guidelines
    guidelines = Guideline.objects.filter(is_active=True).order_by('-created_at')

    # Pagination
    paginator = Paginator(guidelines, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'total_guidelines': guidelines.count(),
    }

    return render(request, 'annotation/view_guidelines.html', context)


@login_required
def guideline_detail(request, guideline_id):
    """Detailed view of a specific guideline"""
    if request.user.role not in ['annotator', 'admin']:
        messages.error(request, 'Access denied.')
        return redirect('authentication:dashboard')

    # For annotators, only show active guidelines
    if request.user.role == 'annotator':
        guideline = get_object_or_404(Guideline, id=guideline_id, is_active=True)
    else:
        guideline = get_object_or_404(Guideline, id=guideline_id)

    context = {
        'guideline': guideline,
    }

    return render(request, 'annotation/guideline_detail.html', context)


@login_required
@require_http_methods(["GET"])
def get_entity_definitions(request):
    """API endpoint to fetch entity definitions for tooltips"""
    try:
        # Get all active entity definitions
        definitions = EntityDefinition.objects.filter(is_active=True).order_by('entity_type')

        # Convert to dictionary format for easy lookup by entity type
        definitions_dict = {}
        for definition in definitions:
            definitions_dict[definition.entity_type] = {
                'entity_type': definition.entity_type,
                'entity_type_display': definition.get_entity_type_display(),
                'definition': definition.definition,
                'examples': definition.examples,
                'guidelines': definition.guidelines,
                'created_at': definition.created_at.isoformat(),
                'updated_at': definition.updated_at.isoformat()
            }

        return JsonResponse({
            'success': True,
            'definitions': definitions_dict,
            'total_definitions': len(definitions_dict)
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Failed to fetch entity definitions: {str(e)}'
        }, status=500)


# ==================== GRANULAR STATISTICS VIEWS ====================

@login_required
def annotator_detail_stats(request, annotator_id):
    """Detailed statistics for a specific annotator with task breakdown"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    try:
        annotator = User.objects.get(id=annotator_id, role='annotator')
    except User.DoesNotExist:
        messages.error(request, 'Annotator not found.')
        return redirect('annotation:admin_statistics_dashboard')

    # Get filters
    task_type_filter = request.GET.get('task_type', 'all')
    date_range = request.GET.get('date_range', '30')

    # Calculate date range
    from datetime import timedelta
    end_date = timezone.now()
    if date_range == '7':
        start_date = end_date - timedelta(days=7)
        period_label = "Last 7 Days"
    elif date_range == '30':
        start_date = end_date - timedelta(days=30)
        period_label = "Last 30 Days"
    elif date_range == '90':
        start_date = end_date - timedelta(days=90)
        period_label = "Last 90 Days"
    else:
        start_date = None
        period_label = "All Time"

    # Build queryset for tasks
    tasks_qs = AnnotationTask.objects.filter(annotator=annotator, is_completed=True)

    if task_type_filter != 'all':
        tasks_qs = tasks_qs.filter(task_type=task_type_filter)

    if start_date:
        tasks_qs = tasks_qs.filter(completed_at__gte=start_date)

    # Calculate overall statistics
    total_tasks = tasks_qs.count()
    completed_tasks = list(tasks_qs.exclude(actual_duration__isnull=True))

    # Time statistics
    time_stats = {
        'total_time_minutes': 0,
        'average_time_minutes': 0,
        'median_time_minutes': 0,
        'min_time_minutes': None,
        'max_time_minutes': None,
        'std_time_minutes': 0,
    }

    if completed_tasks:
        import numpy as np
        durations = [task.get_duration_minutes() for task in completed_tasks]
        time_stats['total_time_minutes'] = sum(durations)
        time_stats['average_time_minutes'] = np.mean(durations)
        time_stats['median_time_minutes'] = np.median(durations)
        time_stats['min_time_minutes'] = min(durations)
        time_stats['max_time_minutes'] = max(durations)
        time_stats['std_time_minutes'] = np.std(durations)

    # Entity statistics
    results_qs = AnnotationResult.objects.filter(annotator=annotator)
    if task_type_filter != 'all':
        results_qs = results_qs.filter(task_type=task_type_filter)
    if start_date:
        results_qs = results_qs.filter(created_at__gte=start_date)

    total_entities = results_qs.aggregate(total=Sum('total_entities'))['total'] or 0

    # Entity type distribution
    from collections import defaultdict
    entity_distribution = defaultdict(int)
    for result in results_qs:
        for entity_type, count in result.get_entity_statistics().items():
            entity_distribution[entity_type] += count

    # NER LLM statistics (if applicable)
    llm_results = results_qs.filter(annotation_type__in=['llm_assisted', 'mcn_llm_assisted'])
    llm_stats = calculate_llm_statistics(llm_results)

    # MCN statistics
    mcn_results = results_qs.filter(annotation_type__in=['mcn_llm_assisted', 'mcn_manual'])
    mcn_stats = calculate_mcn_statistics(mcn_results)

    # Get recent tasks for detailed breakdown
    recent_tasks = tasks_qs.order_by('-completed_at')[:50]  # Last 50 tasks
    task_details = []
    for task in recent_tasks:
        # Get associated result (get the most recent completed one)
        result = AnnotationResult.objects.filter(
            document=task.document,
            annotator=task.annotator,
            task_type=task.task_type,
            review_status='completed'
        ).order_by('-created_at').first()

        if result:
            entities_count = result.total_entities
        else:
            entities_count = 0

        task_details.append({
            'task': task,
            'duration_minutes': task.get_duration_minutes(),
            'entities_count': entities_count,
            'entities_per_minute': entities_count / task.get_duration_minutes() if task.get_duration_minutes() > 0 else 0,
        })

    # Task type breakdown
    ner_tasks = tasks_qs.filter(task_type='ner').exclude(actual_duration__isnull=True)
    mcn_tasks = tasks_qs.filter(task_type='mcn').exclude(actual_duration__isnull=True)

    task_type_breakdown = {
        'ner': {
            'count': ner_tasks.count(),
            'avg_time': np.mean([t.get_duration_minutes() for t in ner_tasks]) if ner_tasks.exists() else 0,
            'total_time': sum([t.get_duration_minutes() for t in ner_tasks]) if ner_tasks.exists() else 0,
        },
        'mcn': {
            'count': mcn_tasks.count(),
            'avg_time': np.mean([t.get_duration_minutes() for t in mcn_tasks]) if mcn_tasks.exists() else 0,
            'total_time': sum([t.get_duration_minutes() for t in mcn_tasks]) if mcn_tasks.exists() else 0,
        }
    }

    # Productivity metrics
    entities_per_hour = (total_entities / (time_stats['total_time_minutes'] / 60)) if time_stats['total_time_minutes'] > 0 else 0

    # Entity source statistics (manual vs LLM)
    entity_source_stats = calculate_entity_source_statistics(annotator, start_date, task_type_filter)

    context = {
        'annotator': annotator,
        'total_tasks': total_tasks,
        'time_stats': time_stats,
        'total_entities': total_entities,
        'entity_distribution': dict(entity_distribution),
        'entity_source_stats': entity_source_stats,
        'llm_stats': llm_stats,
        'mcn_stats': mcn_stats,
        'task_details': task_details,
        'task_type_breakdown': task_type_breakdown,
        'entities_per_hour': entities_per_hour,
        'task_type_filter': task_type_filter,
        'date_range': date_range,
        'period_label': period_label,
    }

    return render(request, 'annotation/annotator_detail_stats.html', context)


@login_required
def annotator_task_breakdown(request, annotator_id):
    """Complete task-by-task breakdown for an annotator"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    try:
        annotator = User.objects.get(id=annotator_id, role='annotator')
    except User.DoesNotExist:
        messages.error(request, 'Annotator not found.')
        return redirect('annotation:admin_statistics_dashboard')

    # Get all completed tasks
    tasks = AnnotationTask.objects.filter(
        annotator=annotator,
        is_completed=True
    ).order_by('-completed_at')

    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(tasks, 25)  # 25 tasks per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Prepare task data with details
    task_data = []
    for task in page_obj:
        # Get associated result (use filter+first to handle duplicates)
        result = AnnotationResult.objects.filter(
            document=task.document,
            annotator=task.annotator,
            task_type=task.task_type,
            review_status='completed'
        ).order_by('-created_at').first()

        if result:
            entities_count = result.total_entities
            entity_stats = result.get_entity_statistics()

            # LLM decision stats if applicable
            if result.annotation_type in ['llm_assisted', 'mcn_llm_assisted']:
                llm_decision_summary = result.get_llm_decision_summary()
            else:
                llm_decision_summary = None
        else:
            entities_count = 0
            entity_stats = {}
            llm_decision_summary = None

        task_data.append({
            'task': task,
            'duration_minutes': task.get_duration_minutes(),
            'entities_count': entities_count,
            'entity_stats': entity_stats,
            'llm_decision_summary': llm_decision_summary,
            'entities_per_minute': entities_count / task.get_duration_minutes() if task.get_duration_minutes() > 0 else 0,
        })

    context = {
        'annotator': annotator,
        'task_data': task_data,
        'page_obj': page_obj,
    }

    return render(request, 'annotation/annotator_task_breakdown.html', context)


@login_required
def document_stats(request, document_id):
    """Statistics for a specific document across all annotators"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    try:
        document = TextDocument.objects.get(id=document_id)
    except TextDocument.DoesNotExist:
        messages.error(request, 'Document not found.')
        return redirect('annotation:admin_statistics_dashboard')

    # Get all tasks for this document
    tasks = AnnotationTask.objects.filter(document=document, is_completed=True)

    # Calculate statistics
    import numpy as np
    annotator_stats = []

    for task in tasks:
        # Get result (use filter+first to handle duplicates)
        result = AnnotationResult.objects.filter(
            document=document,
            annotator=task.annotator,
            task_type=task.task_type,
            review_status='completed'
        ).order_by('-created_at').first()

        if result:
            entities_count = result.total_entities
            entity_stats = result.get_entity_statistics()

            # LLM stats if applicable
            if result.annotation_type in ['llm_assisted', 'mcn_llm_assisted']:
                llm_summary = result.get_llm_decision_summary()
            else:
                llm_summary = None
        else:
            entities_count = 0
            entity_stats = {}
            llm_summary = None

        annotator_stats.append({
            'annotator': task.annotator,
            'task_type': task.get_task_type_display(),
            'duration_minutes': task.get_duration_minutes(),
            'entities_count': entities_count,
            'entity_stats': entity_stats,
            'llm_summary': llm_summary,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
        })

    # Overall document statistics
    if tasks.exists():
        durations = [task.get_duration_minutes() for task in tasks.exclude(actual_duration__isnull=True)]
        doc_stats = {
            'total_annotations': tasks.count(),
            'avg_time_minutes': np.mean(durations) if durations else 0,
            'min_time_minutes': min(durations) if durations else 0,
            'max_time_minutes': max(durations) if durations else 0,
            'total_time_minutes': sum(durations) if durations else 0,
        }
    else:
        doc_stats = {
            'total_annotations': 0,
            'avg_time_minutes': 0,
            'min_time_minutes': 0,
            'max_time_minutes': 0,
            'total_time_minutes': 0,
        }

    context = {
        'document': document,
        'annotator_stats': annotator_stats,
        'doc_stats': doc_stats,
    }

    return render(request, 'annotation/document_stats.html', context)


@login_required
def task_detail_stats(request, task_id):
    """Detailed statistics for a single task"""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    try:
        task = AnnotationTask.objects.get(id=task_id)
    except AnnotationTask.DoesNotExist:
        messages.error(request, 'Task not found.')
        return redirect('annotation:admin_statistics_dashboard')

    # Get associated result (use filter+first to handle duplicates)
    result = AnnotationResult.objects.filter(
        document=task.document,
        annotator=task.annotator,
        task_type=task.task_type,
        review_status='completed'
    ).order_by('-created_at').first()

    # Calculate detailed statistics
    if result:
        entity_stats = result.get_entity_statistics()
        entities_by_type = result.get_filtered_entities_by_type()

        # LLM decision breakdown if applicable
        if result.annotation_type in ['llm_assisted', 'mcn_llm_assisted']:
            llm_summary = result.get_llm_decision_summary()

            # Detailed entity-by-entity breakdown
            llm_entity_details = []
            manual_entity_details = []
            for entity in result.entities_json:
                if entity.get('source') == 'llm':
                    llm_entity_details.append({
                        'text': entity.get('text', ''),
                        'label': entity.get('label', ''),
                        'status': entity.get('status', 'pending'),
                        'original_text': entity.get('original_text', ''),
                        'modified_text': entity.get('modified_text', ''),
                        'edit_distance': entity.get('edit_distance', 0),
                        'token_changes': entity.get('token_changes', 0),
                    })
                elif entity.get('source') == 'manual':
                    manual_entity_details.append({
                        'text': entity.get('text', ''),
                        'label': entity.get('label', ''),
                        'source': 'manual',
                    })
        else:
            llm_summary = None
            llm_entity_details = []
            manual_entity_details = []
    else:
        entity_stats = {}
        entities_by_type = {}
        llm_summary = None
        llm_entity_details = []
        manual_entity_details = []

    # Time breakdown
    duration_minutes = task.get_duration_minutes()
    entities_count = result.total_entities if result else 0
    entities_per_minute = entities_count / duration_minutes if duration_minutes > 0 else 0

    context = {
        'task': task,
        'result': result,
        'duration_minutes': duration_minutes,
        'entities_count': entities_count,
        'entities_per_minute': entities_per_minute,
        'entity_stats': entity_stats,
        'entities_by_type': entities_by_type,
        'llm_summary': llm_summary,
        'llm_entity_details': llm_entity_details,
        'manual_entity_details': manual_entity_details,
    }

    return render(request, 'annotation/task_detail_stats.html', context)


@login_required
@require_http_methods(["POST"])
def bulk_mcn_three_step(request):
    """
    Bulk MCN 3-Step Processing
    Processes multiple documents through all 3 steps of MCN:
    1. Extract entities using LangExtract NER
    2. Translate Indonesian terms to English & normalize
    3. Map entities to SNOMED-CT codes
    """
    try:
        # Get selected document IDs
        document_ids = request.POST.getlist('document_ids')
        skip_if_exists = request.POST.get('skip_if_exists', 'true') == 'true'

        if not document_ids:
            return JsonResponse({
                'success': False,
                'error': 'No documents selected'
            }, status=400)

        # Get documents
        documents = TextDocument.objects.filter(id__in=document_ids)

        # Check permissions
        if not request.user.is_admin():
            # Annotators can only process assigned documents
            documents = documents.filter(assigned_to=request.user)

        if not documents.exists():
            return JsonResponse({
                'success': False,
                'error': 'No accessible documents found'
            }, status=403)

        # Import necessary services
        from .medical_annotator import create_medical_annotator
        from .gemini_normalizer import normalize_medical_entities
        from .snomed_service import get_snomed_service

        # Process results
        results_summary = {
            'total': documents.count(),
            'successful': 0,
            'skipped': 0,
            'failed': 0,
            'failures': [],
            'details': []
        }

        for doc in documents:
            doc_result = {
                'document_id': doc.id,
                'document_title': doc.content[:50] + '...' if len(doc.content) > 50 else doc.content,
                'step_completed': 0,
                'status': 'pending'
            }

            try:
                # Check if MCN already exists and skip if requested
                if skip_if_exists:
                    # Check if MCN exists for ANY assigned annotator (not just admin)
                    target_annotators = list(doc.assigned_to.all()) or [request.user]
                    existing_mcn = AnnotationResult.objects.filter(
                        document=doc,
                        annotator__in=target_annotators,
                        task_type='mcn',
                        review_status='completed'
                    ).exists()

                    if existing_mcn:
                        results_summary['skipped'] += 1
                        doc_result['status'] = 'skipped'
                        doc_result['message'] = 'Already has MCN annotation'
                        results_summary['details'].append(doc_result)
                        continue

                # STEP 1: Extract entities using LangExtract NER
                start_time = time.time()

                # Check if NER already exists (from any user)
                ner_annotation = AnnotationResult.objects.filter(
                    document=doc,
                    task_type='ner',
                    review_status='completed'
                ).first()

                if not ner_annotation:
                    # Run LangExtract NER
                    annotator = create_medical_annotator(model_id="gemini-2.5-flash")
                    ner_result = annotator.annotate_document(doc.content)

                    if not ner_result['success']:
                        raise Exception('NER extraction failed')

                    entities = ner_result['entities']

                    # Resolve overlapping entities: keep only the longest span
                    entities = resolve_overlaps_longest_span(entities)

                    # Save NER annotation for all assigned annotators
                    ner_annotation = _save_annotation_for_annotators(
                        doc=doc, request_user=request.user,
                        task_type='ner', annotation_type='llm_assisted',
                        model_used='langextract-gemini-2.5-flash',
                        entities_json=entities,
                        total_entities=len(entities),
                        processing_time=time.time() - start_time
                    )
                else:
                    entities = ner_annotation.entities_json

                # Filter only approved entities (exclude rejected ones)
                entities = [
                    entity for entity in entities
                    if entity.get('status') == 'approved' or 'status' not in entity
                ]

                doc_result['step_completed'] = 1
                doc_result['ner_entities_count'] = len(entities)

                if not entities:
                    raise Exception('No approved entities found')

                # STEP 2: Translate and Normalize (single Gemini batch call)
                step2_start = time.time()

                # Only MCN-relevant types need SNOMED mapping
                MCN_TYPES = {'SYMPTOM', 'DISEASE', 'MEDICATION', 'PROCEDURE'}
                mcn_entities = [
                    e for e in entities
                    if e.get('label', e.get('entity_type', 'UNKNOWN')).upper() in MCN_TYPES
                ]
                skip_entities = [
                    e for e in entities
                    if e.get('label', e.get('entity_type', 'UNKNOWN')).upper() not in MCN_TYPES
                ]

                normalized_entities = normalize_medical_entities(mcn_entities)
                doc_result['step_completed'] = 2
                doc_result['normalization_time'] = round(time.time() - step2_start, 2)

                # STEP 3: Auto-map to SNOMED-CT (parallelized)
                step3_start = time.time()
                snomed_service = get_snomed_service()

                def _map_single_entity(entity):
                    """Map one entity to SNOMED-CT (runs in thread pool)."""
                    normalized_term = entity.get('normalized_term', entity.get('text', ''))
                    entity_type = entity.get('entity_type', entity.get('label', 'UNKNOWN'))
                    try:
                        candidates = snomed_service.search_with_auto_translation(
                            normalized_term, max_results=1, context=entity_type
                        )
                        if candidates:
                            best = candidates[0]
                            entity.update({
                                'snomed_concept': {
                                    'conceptId': best.get('conceptId', ''),
                                    'fsn': best.get('fsn', ''),
                                    'preferredTerm': best.get('preferredTerm', ''),
                                    'definition': best.get('definition', ''),
                                    'confidence_score': best.get('confidence_score', 0.8),
                                },
                                'snomed_code': best.get('conceptId', ''),
                                'snomed_display': best.get('preferredTerm', ''),
                                'auto_mapped': True,
                                'mapping_confidence': best.get('confidence_score', 0.8),
                            })
                        else:
                            entity.update({
                                'snomed_concept': None,
                                'snomed_code': '',
                                'snomed_display': '',
                                'auto_mapped': False,
                                'mapping_confidence': 0.0,
                                'mapping_error': 'No SNOMED-CT matches found',
                            })
                    except Exception as err:
                        entity.update({
                            'snomed_concept': None,
                            'snomed_code': '',
                            'snomed_display': '',
                            'auto_mapped': False,
                            'mapping_confidence': 0.0,
                            'mapping_error': str(err),
                        })
                    return entity

                # Parallel SNOMED lookups
                from concurrent.futures import ThreadPoolExecutor, as_completed
                mapped_entities = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(_map_single_entity, ent): idx
                        for idx, ent in enumerate(normalized_entities)
                    }
                    results_by_idx = {}
                    for future in as_completed(futures):
                        idx = futures[future]
                        results_by_idx[idx] = future.result()
                    # Preserve original order
                    for idx in range(len(normalized_entities)):
                        mapped_entities.append(results_by_idx[idx])

                # Add back skipped entity types (BODY_PART etc.) without SNOMED mapping
                for entity in skip_entities:
                    entity.update({
                        'english_term': entity.get('text', ''),
                        'normalized_term': entity.get('text', ''),
                        'snomed_concept': None,
                        'snomed_code': '',
                        'snomed_display': '',
                        'auto_mapped': False,
                        'mapping_confidence': 0.0,
                        'normalization_source': 'skipped',
                    })
                    mapped_entities.append(entity)

                doc_result['step_completed'] = 3
                doc_result['mapping_time'] = round(time.time() - step3_start, 2)
                doc_result['mapped_count'] = sum(1 for e in mapped_entities if e.get('auto_mapped'))

                # Save final MCN annotation for all assigned annotators
                total_processing_time = time.time() - start_time
                mcn_annotation = _save_annotation_for_annotators(
                    doc=doc, request_user=request.user,
                    task_type='mcn', annotation_type='mcn_llm_assisted',
                    model_used='gemini-2.5-flash',
                    entities_json=mapped_entities,
                    total_entities=len(mapped_entities),
                    processing_time=total_processing_time
                )

                # Update document status
                if doc.status != 'completed':
                    doc.status = 'completed'
                    doc.save()

                doc_result['status'] = 'success'
                doc_result['total_time'] = round(total_processing_time, 2)
                doc_result['annotation_id'] = mcn_annotation.id
                results_summary['successful'] += 1

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error processing document {doc.id}: {e}")
                print(error_trace)

                doc_result['status'] = 'failed'
                doc_result['error'] = str(e)
                results_summary['failed'] += 1
                results_summary['failures'].append({
                    'document_id': doc.id,
                    'error': str(e),
                    'step': doc_result['step_completed']
                })

            results_summary['details'].append(doc_result)

        # Return results
        return JsonResponse({
            'success': True,
            'message': f'Batch MCN processing complete: {results_summary["successful"]}/{results_summary["total"]} successful, {results_summary["skipped"]} skipped, {results_summary["failed"]} failed',
            'results': results_summary
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in bulk MCN 3-step: {e}")
        print(error_trace)
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@login_required
@require_http_methods(["GET"])
def get_ner_annotation_data(request, document_id):
    """
    Get NER annotation data for admin popup view
    Returns JSON with document content and entities for highlighting
    """
    document = get_object_or_404(TextDocument, id=document_id)

    # Check permissions - admin only
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    # Get latest NER annotation
    ner_annotation = document.get_latest_ner_annotation()

    if not ner_annotation:
        return JsonResponse({
            'success': False,
            'error': 'No NER annotation found for this document'
        }, status=404)

    # Parse entities
    entities = []
    if ner_annotation.entities_json:
        try:
            if isinstance(ner_annotation.entities_json, str):
                entities = json.loads(ner_annotation.entities_json)
            else:
                entities = ner_annotation.entities_json
        except:
            entities = []

    return JsonResponse({
        'success': True,
        'document': {
            'id': document.id,
            'content': document.content,
        },
        'annotation': {
            'id': ner_annotation.id,
            'created_at': ner_annotation.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'annotator': ner_annotation.annotator.get_full_name() if ner_annotation.annotator else 'Unknown',
            'model_used': ner_annotation.model_used or 'Unknown',
            'total_entities': ner_annotation.total_entities or 0,
        },
        'entities': entities
    })


@login_required
@require_http_methods(["GET"])
def get_mcn_annotation_data(request, document_id):
    """
    Get MCN annotation data for admin popup view
    Returns JSON with document content and normalized entities
    """
    document = get_object_or_404(TextDocument, id=document_id)

    # Check permissions - admin only
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    # Get latest MCN annotation
    mcn_annotation = document.get_latest_mcn_annotation()

    if not mcn_annotation:
        return JsonResponse({
            'success': False,
            'error': 'No MCN annotation found for this document'
        }, status=404)

    # Parse entities
    entities = []
    if mcn_annotation.entities_json:
        try:
            if isinstance(mcn_annotation.entities_json, str):
                entities = json.loads(mcn_annotation.entities_json)
            else:
                entities = mcn_annotation.entities_json
        except:
            entities = []

    return JsonResponse({
        'success': True,
        'document': {
            'id': document.id,
            'content': document.content,
        },
        'annotation': {
            'id': mcn_annotation.id,
            'created_at': mcn_annotation.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'annotator': mcn_annotation.annotator.get_full_name() if mcn_annotation.annotator else 'Unknown',
            'model_used': mcn_annotation.model_used or 'Unknown',
            'total_entities': mcn_annotation.total_entities or 0,
        },
        'entities': entities
    })


@require_http_methods(["GET"])
def get_entity_definitions(request):
    """
    Get entity definitions including colors
    Returns JSON with entity types, colors, definitions, and examples
    """
    entity_definitions = EntityDefinition.objects.filter(is_active=True)

    definitions_list = []
    for entity_def in entity_definitions:
        definitions_list.append({
            'entity_type': entity_def.entity_type,
            'color': entity_def.color,
            'definition': entity_def.definition,
            'examples': entity_def.examples,
            'guidelines': entity_def.guidelines,
        })

    # Get all colors as a simple mapping
    colors = EntityDefinition.get_entity_colors()

    return JsonResponse({
        'success': True,
        'colors': colors,
        'definitions': definitions_list
    })


@login_required
def export_sql(request):
    """Export entire database as SQL INSERT statements for migration to another system."""
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    from django.http import HttpResponse
    from django.apps import apps
    from django.db import connection
    import datetime

    tables_param = request.GET.get('tables', 'all')

    # Define which models to export and in what order (respecting FK dependencies)
    export_models = [
        # Authentication models first (no FK dependencies)
        ('authentication', 'User'),
        ('authentication', 'UserSession'),
        # Annotation core models
        ('annotation', 'EntityDefinition'),
        ('annotation', 'Guideline'),
        ('annotation', 'TextImportBatch'),
        ('annotation', 'TextDocument'),
        ('annotation', 'AnnotationTask'),
        ('annotation', 'AnnotationResult'),
        ('annotation', 'GoldStandard'),
        ('annotation', 'AnnotatorAgreement'),
        ('annotation', 'AnnotationMetrics'),
        ('annotation', 'AnnotationStatistics'),
        ('annotation', 'AutoMCNMapping'),
    ]

    # Try to add optional models if they exist
    optional_models = [
        ('annotation', 'LLMProvider'),
        ('annotation', 'APIKey'),
        ('annotation', 'OverlappingEntitySchema'),
        ('annotation', 'OverlappingEntity'),
        ('annotation', 'OverlappingAnnotationResult'),
        ('annotation', 'OverlappingPerformanceStatistics'),
        ('authentication', 'GamificationConfig'),
        ('authentication', 'AnnotatorProfile'),
        ('authentication', 'Achievement'),
        ('authentication', 'UserAchievement'),
        ('authentication', 'PointTransaction'),
    ]

    for app_label, model_name in optional_models:
        try:
            apps.get_model(app_label, model_name)
            export_models.append((app_label, model_name))
        except LookupError:
            pass

    def sql_value(val):
        """Convert a Python value to a SQL literal."""
        if val is None:
            return 'NULL'
        if isinstance(val, bool):
            return '1' if val else '0'
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, (datetime.datetime, datetime.date)):
            return "'{}'".format(val.isoformat())
        # String - escape single quotes
        s = str(val).replace("'", "''")
        return "'{}'".format(s)

    lines = []
    lines.append('-- Medical Annotation System - SQL Export')
    lines.append('-- Generated: {}'.format(timezone.now().strftime('%Y-%m-%d %H:%M:%S')))
    lines.append('-- Django Annotation System Database Dump')
    lines.append('')
    lines.append('-- NOTE: This export uses generic SQL INSERT statements.')
    lines.append('-- For PostgreSQL/MySQL import, you may need to adjust data types.')
    lines.append('-- Auto-increment/sequence values may need resetting after import.')
    lines.append('')

    total_rows = 0

    for app_label, model_name in export_models:
        try:
            Model = apps.get_model(app_label, model_name)
        except LookupError:
            continue

        db_table = Model._meta.db_table
        queryset = Model.objects.all()
        count = queryset.count()

        if count == 0:
            lines.append('-- Table: {} (0 rows - skipped)'.format(db_table))
            lines.append('')
            continue

        # Get column names from model fields
        fields = []
        for field in Model._meta.local_fields:
            fields.append(field)

        col_names = [f.column for f in fields]

        lines.append('-- -----------------------------------------------')
        lines.append('-- Table: {} ({} rows)'.format(db_table, count))
        lines.append('-- -----------------------------------------------')

        # Generate CREATE TABLE (simplified - column definitions from Django)
        lines.append('-- CREATE TABLE IF NOT EXISTS {} ('.format(db_table))
        for i, field in enumerate(fields):
            col_type = field.db_type(connection)
            nullable = 'NULL' if field.null else 'NOT NULL'
            pk = ' PRIMARY KEY' if field.primary_key else ''
            comma = ',' if i < len(fields) - 1 else ''
            lines.append('--   {} {} {}{}{}'.format(field.column, col_type or 'TEXT', nullable, pk, comma))
        lines.append('-- );')
        lines.append('')

        # Generate INSERT statements
        for obj in queryset.iterator():
            values = []
            for field in fields:
                val = field.value_from_object(obj)
                values.append(sql_value(val))

            lines.append('INSERT INTO {} ({}) VALUES ({});'.format(
                db_table,
                ', '.join(col_names),
                ', '.join(values)
            ))
            total_rows += 1

        lines.append('')

    lines.append('-- Export complete: {} total rows across {} tables'.format(
        total_rows, len(export_models)))

    content = '\n'.join(lines)

    response = HttpResponse(content, content_type='application/sql')
    timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
    response['Content-Disposition'] = 'attachment; filename="annotation_export_{}.sql"'.format(timestamp)
    return response

