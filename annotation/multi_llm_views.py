"""
Views for Multi-LLM NER comparison
Allows running NER with multiple models and comparing results
"""

import json
import logging
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages

from .models import TextDocument, AnnotationResult, AnnotationTask
from .multi_llm_ner import MultiLLMNERRunner, EntityMerger, AgreementCalculator
from .llm_providers import get_all_available_providers, get_default_models
from .views import resolve_overlaps_longest_span, _get_gamification_response, _get_gamification_status_response

logger = logging.getLogger(__name__)


@login_required
@require_http_methods(["GET", "POST"])
def multi_llm_ner_run(request, document_id):
    """
    Run NER with multiple LLM models
    GET: Show configuration page
    POST: Execute multi-LLM NER
    """
    document = get_object_or_404(TextDocument, id=document_id)

    # Check permissions
    if not (request.user.is_staff or document.is_assigned_to_user(request.user)):
        messages.error(request, "You don't have permission to annotate this document.")
        return redirect('annotation:annotator_documents')

    if request.method == 'GET':
        # Show configuration page
        available_providers = get_all_available_providers()
        default_models = get_default_models()

        context = {
            'document': document,
            'available_providers': available_providers,
            'default_models': default_models,
        }
        return render(request, 'annotation/multi_llm_ner_config.html', context)

    elif request.method == 'POST':
        # Execute multi-LLM NER
        try:
            # Get selected models from form
            selected_models = request.POST.getlist('models')
            if not selected_models:
                selected_models = get_default_models()

            enable_overlapping = request.POST.get('enable_overlapping', 'true') == 'true'
            confidence_threshold = float(request.POST.get('confidence_threshold', '0.3'))

            # Run multi-LLM NER
            runner = MultiLLMNERRunner(model_ids=selected_models)

            if runner.get_model_count() == 0:
                messages.error(request, "No models available. Please configure API keys.")
                return redirect('annotation:multi_llm_ner_run', document_id=document_id)

            results = runner.run_all_models(
                document=document,
                annotator=request.user,
                enable_overlapping=enable_overlapping,
                confidence_threshold=confidence_threshold
            )

            if not results:
                messages.error(request, "Failed to run NER with any models.")
                return redirect('annotation:multi_llm_ner_run', document_id=document_id)

            messages.success(
                request,
                f"Successfully ran NER with {len(results)} models. "
                f"View comparison to review results."
            )

            return redirect('annotation:multi_llm_ner_comparison', document_id=document_id)

        except Exception as e:
            logger.error(f"Error running multi-LLM NER: {e}", exc_info=True)
            messages.error(request, f"Error running multi-LLM NER: {str(e)}")
            return redirect('annotation:multi_llm_ner_run', document_id=document_id)


@login_required
def multi_llm_ner_comparison(request, document_id):
    """
    Show comparison of NER results from multiple LLMs
    Displays entities grouped by consensus level
    """
    document = get_object_or_404(TextDocument, id=document_id)

    # Check permissions
    if not (request.user.is_staff or document.is_assigned_to_user(request.user)):
        messages.error(request, "You don't have permission to view this comparison.")
        return redirect('annotation:annotator_documents')

    # Get all LLM comparison results for this document and user
    llm_results = AnnotationResult.objects.filter(
        document=document,
        annotator=request.user,
        task_type='ner',
        annotation_type='llm_comparison'
    ).order_by('-created_at')

    # Get unique models (in case of multiple runs)
    # Group by model and take most recent
    latest_results = {}
    for result in llm_results:
        if result.model_used not in latest_results:
            latest_results[result.model_used] = result

    llm_results = list(latest_results.values())

    if not llm_results:
        messages.info(request, "No multi-LLM results found. Please run multi-LLM NER first.")
        return redirect('annotation:multi_llm_ner_run', document_id=document_id)

    # Merge entities for comparison
    merged_entities = EntityMerger.merge_results(
        llm_results,
        merge_strategy='overlapping'
    )

    # Group by consensus level
    grouped_entities = {
        'high_consensus': [],
        'moderate_consensus': [],
        'low_consensus': [],
        'needs_review': [],
    }

    for entity in merged_entities:
        status = entity.get('status', 'needs_review')
        grouped_entities[status].append(entity)

    # Calculate agreement metrics
    agreement_stats = AgreementCalculator.calculate_agreement(llm_results)

    context = {
        'document': document,
        'llm_results': llm_results,
        'merged_entities': merged_entities,
        'grouped_entities': grouped_entities,
        'agreement_stats': agreement_stats,
        'total_models': len(llm_results),
    }

    return render(request, 'annotation/multi_llm_ner_comparison.html', context)


@login_required
@require_http_methods(["POST"])
def multi_llm_ner_approve(request, document_id):
    """
    Approve selected entities from multi-LLM comparison
    Creates final annotation result
    """
    document = get_object_or_404(TextDocument, id=document_id)

    # Check permissions
    if not (request.user.is_staff or document.is_assigned_to_user(request.user)):
        return JsonResponse({'success': False, 'error': 'Permission denied'}, status=403)

    try:
        # Get approved entities from request
        approved_entities_json = request.POST.get('approved_entities')
        if not approved_entities_json:
            return JsonResponse({'success': False, 'error': 'No entities provided'}, status=400)

        approved_entities = json.loads(approved_entities_json)

        # Resolve overlapping entities: keep only the longest span
        approved_entities = resolve_overlaps_longest_span(approved_entities)

        # Create final annotation result
        annotation_result = AnnotationResult.objects.create(
            document=document,
            annotator=request.user,
            task_type='ner',
            annotation_type='llm_assisted',  # Final approved result
            model_used='multi_llm_consensus',
            entities_json=approved_entities,
            total_entities=len(approved_entities),
            review_status='completed'
        )

        # Mark NER task as completed
        try:
            ner_task = AnnotationTask.objects.get(
                document=document,
                annotator=request.user,
                task_type='ner'
            )
            ner_task.complete_task()
        except AnnotationTask.DoesNotExist:
            # Create task if it doesn't exist
            ner_task = AnnotationTask.objects.create(
                document=document,
                annotator=request.user,
                task_type='ner'
            )
            ner_task.start_task()
            ner_task.complete_task()

        response_data = {
            'success': True,
            'message': f'Approved {len(approved_entities)} entities',
            'annotation_id': annotation_result.id
        }
        gamification_data = _get_gamification_response(ner_task)
        if not gamification_data:
            gamification_data = _get_gamification_status_response(request.user)
        response_data.update(gamification_data)
        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Error approving multi-LLM entities: {e}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def multi_llm_ner_stats(request):
    """
    Show statistics dashboard for multi-LLM NER comparisons
    Admin only
    """
    if not request.user.is_staff:
        messages.error(request, "Admin access required.")
        return redirect('authentication:dashboard')

    # Get all multi-LLM comparison results
    all_comparisons = AnnotationResult.objects.filter(
        annotation_type='llm_comparison'
    ).values('document_id', 'annotator_id').distinct()

    comparison_stats = []

    for comp in all_comparisons:
        # Get results for this comparison
        results = AnnotationResult.objects.filter(
            document_id=comp['document_id'],
            annotator_id=comp['annotator_id'],
            annotation_type='llm_comparison'
        )

        if len(results) < 2:
            continue

        # Calculate agreement
        agreement = AgreementCalculator.calculate_agreement(list(results))

        comparison_stats.append({
            'document_id': comp['document_id'],
            'document': results[0].document,
            'annotator': results[0].annotator,
            'models': [r.model_used for r in results],
            'agreement': agreement,
        })

    context = {
        'comparison_stats': comparison_stats,
        'total_comparisons': len(comparison_stats),
    }

    return render(request, 'annotation/multi_llm_ner_stats.html', context)


@login_required
@require_http_methods(["GET", "POST"])
def batch_multi_llm_ner(request):
    """
    Batch Multi-LLM NER processing
    GET: Show configuration page for batch processing
    POST: Execute batch multi-LLM NER on selected documents
    """
    if request.method == 'GET':
        # Get document IDs from query params
        document_ids = request.GET.getlist('documents')

        # Handle comma-separated values if provided as single parameter
        if len(document_ids) == 1 and ',' in document_ids[0]:
            document_ids = [id.strip() for id in document_ids[0].split(',') if id.strip()]

        if not document_ids:
            messages.error(request, "No documents selected for batch processing.")
            return redirect('annotation:annotator_documents')

        # Get documents
        documents = TextDocument.objects.filter(
            id__in=document_ids
        ).order_by('id')

        # Check permissions
        if request.user.is_staff:
            # Admin can process any documents
            pass
        else:
            # Annotator can only process assigned documents
            documents = documents.filter(assigned_to=request.user)

        if not documents.exists():
            messages.error(request, "No accessible documents found.")
            return redirect('annotation:annotator_documents')

        # Get available providers and default models
        available_providers = get_all_available_providers()
        default_models = get_default_models()

        context = {
            'documents': documents,
            'document_count': documents.count(),
            'available_providers': available_providers,
            'default_models': default_models,
        }

        return render(request, 'annotation/batch_multi_llm_ner_config.html', context)

    elif request.method == 'POST':
        # Execute batch multi-LLM NER
        try:
            # Get selected documents
            document_ids = request.POST.getlist('document_ids')
            if not document_ids:
                return JsonResponse({
                    'success': False,
                    'error': 'No documents provided'
                }, status=400)

            # Get selected models
            selected_models = request.POST.getlist('models')
            if not selected_models:
                selected_models = get_default_models()

            enable_overlapping = request.POST.get('enable_overlapping', 'true') == 'true'
            confidence_threshold = float(request.POST.get('confidence_threshold', '0.3'))

            # Get documents
            documents = TextDocument.objects.filter(id__in=document_ids)

            # Check permissions
            if not request.user.is_staff:
                documents = documents.filter(assigned_to=request.user)

            if not documents.exists():
                return JsonResponse({
                    'success': False,
                    'error': 'No accessible documents found'
                }, status=403)

            # Process each document
            results_summary = {
                'total': documents.count(),
                'successful': 0,
                'failed': 0,
                'failures': []
            }

            for doc in documents:
                try:
                    # Run multi-LLM NER for this document
                    runner = MultiLLMNERRunner(model_ids=selected_models)

                    if runner.get_model_count() == 0:
                        results_summary['failed'] += 1
                        results_summary['failures'].append({
                            'document_id': doc.id,
                            'error': 'No models available'
                        })
                        continue

                    results = runner.run_all_models(
                        document=doc,
                        annotator=request.user,
                        enable_overlapping=enable_overlapping,
                        confidence_threshold=confidence_threshold
                    )

                    if results:
                        results_summary['successful'] += 1
                    else:
                        results_summary['failed'] += 1
                        results_summary['failures'].append({
                            'document_id': doc.id,
                            'error': 'Failed to run NER'
                        })

                except Exception as e:
                    logger.error(f"Error processing document {doc.id}: {e}", exc_info=True)
                    results_summary['failed'] += 1
                    results_summary['failures'].append({
                        'document_id': doc.id,
                        'error': str(e)
                    })

            # Return results
            return JsonResponse({
                'success': True,
                'message': f'Batch processing complete: {results_summary["successful"]}/{results_summary["total"]} documents processed successfully',
                'results': results_summary
            })

        except Exception as e:
            logger.error(f"Error in batch multi-LLM NER: {e}", exc_info=True)
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
