"""
Views for evaluation metrics - inter-annotator agreement, accuracy, efficiency
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db.models import Avg, Count
from django.utils import timezone
from authentication.models import User
from .models import (
    TextDocument, AnnotationResult, AnnotationTask,
    GoldStandard, AnnotatorAgreement, AnnotationMetrics
)
import json


@login_required
def evaluation_metrics_dashboard(request):
    """
    Admin dashboard for viewing evaluation metrics
    Shows inter-annotator agreement, accuracy metrics, and efficiency
    """
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    # Get summary statistics
    total_gold_standards = GoldStandard.objects.filter(is_active=True).count()
    total_agreements = AnnotatorAgreement.objects.count()
    total_metrics = AnnotationMetrics.objects.count()

    # Get recent metrics
    recent_agreements = AnnotatorAgreement.objects.all()[:10]
    recent_metrics = AnnotationMetrics.objects.select_related(
        'annotation_result__annotator',
        'annotation_result__document'
    ).all()[:10]

    # Calculate average metrics
    avg_kappa = AnnotatorAgreement.objects.aggregate(
        avg=Avg('cohens_kappa')
    )['avg'] or 0.0

    avg_f1 = AnnotationMetrics.objects.aggregate(
        avg=Avg('f1_score')
    )['avg'] or 0.0

    avg_precision = AnnotationMetrics.objects.aggregate(
        avg=Avg('precision')
    )['avg'] or 0.0

    avg_recall = AnnotationMetrics.objects.aggregate(
        avg=Avg('recall')
    )['avg'] or 0.0

    # Get efficiency metrics
    avg_time_per_annotation = AnnotationTask.objects.filter(
        is_completed=True,
        actual_duration__isnull=False
    ).aggregate(avg=Avg('actual_duration'))['avg']

    if avg_time_per_annotation:
        avg_time_minutes = avg_time_per_annotation.total_seconds() / 60
    else:
        avg_time_minutes = 0.0

    # Get all annotators for filtering
    annotators = User.objects.filter(role='annotator', is_active=True)

    # Get all documents with gold standards
    documents_with_gold = TextDocument.objects.filter(
        gold_standards__is_active=True
    ).distinct()

    context = {
        'total_gold_standards': total_gold_standards,
        'total_agreements': total_agreements,
        'total_metrics': total_metrics,
        'recent_agreements': recent_agreements,
        'recent_metrics': recent_metrics,
        'avg_kappa': avg_kappa,
        'avg_f1': avg_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_time_minutes': avg_time_minutes,
        'annotators': annotators,
        'documents_with_gold': documents_with_gold,
    }

    return render(request, 'annotation/evaluation_metrics.html', context)


@login_required
@require_http_methods(["POST"])
def calculate_agreement(request):
    """
    Calculate inter-annotator agreement for a document
    """
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    document_id = request.POST.get('document_id')
    annotator1_id = request.POST.get('annotator1_id')
    annotator2_id = request.POST.get('annotator2_id')
    task_type = request.POST.get('task_type', 'ner')

    if not all([document_id, annotator1_id, annotator2_id]):
        return JsonResponse({'success': False, 'error': 'Missing required parameters'}, status=400)

    try:
        document = TextDocument.objects.get(id=document_id)
        annotator1 = User.objects.get(id=annotator1_id)
        annotator2 = User.objects.get(id=annotator2_id)
    except (TextDocument.DoesNotExist, User.DoesNotExist):
        return JsonResponse({'success': False, 'error': 'Invalid document or annotator'}, status=404)

    # Calculate agreement
    agreement = AnnotatorAgreement.calculate_agreement(
        document, annotator1, annotator2, task_type
    )

    if not agreement:
        return JsonResponse({
            'success': False,
            'error': 'Could not calculate agreement. Ensure both annotators have completed annotations.'
        }, status=400)

    return JsonResponse({
        'success': True,
        'agreement': {
            'cohens_kappa': round(agreement.cohens_kappa, 3) if agreement.cohens_kappa else 0.0,
            'percent_agreement': round(agreement.percent_agreement, 2),
            'agreement_level': agreement.get_agreement_level(),
            'agreed_entities': agreement.agreed_entities,
            'disagreed_entities': agreement.disagreed_entities,
            'total_entities_annotator1': agreement.total_entities_annotator1,
            'total_entities_annotator2': agreement.total_entities_annotator2,
        }
    })


@login_required
@require_http_methods(["POST"])
def calculate_metrics(request):
    """
    Calculate precision, recall, F1-score for an annotation result
    """
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    annotation_result_id = request.POST.get('annotation_result_id')
    gold_standard_id = request.POST.get('gold_standard_id')

    if not annotation_result_id:
        return JsonResponse({'success': False, 'error': 'Missing annotation result ID'}, status=400)

    try:
        annotation_result = AnnotationResult.objects.get(id=annotation_result_id)
    except AnnotationResult.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Annotation result not found'}, status=404)

    # Get gold standard if provided
    gold_standard = None
    if gold_standard_id:
        try:
            gold_standard = GoldStandard.objects.get(id=gold_standard_id)
        except GoldStandard.DoesNotExist:
            return JsonResponse({'success': False, 'error': 'Gold standard not found'}, status=404)

    # Calculate metrics
    metric = AnnotationMetrics.calculate_metrics(annotation_result, gold_standard)

    return JsonResponse({
        'success': True,
        'metrics': {
            'precision': round(metric.precision, 3) if metric.precision else 0.0,
            'recall': round(metric.recall, 3) if metric.recall else 0.0,
            'f1_score': round(metric.f1_score, 3) if metric.f1_score else 0.0,
            'true_positives': metric.true_positives,
            'false_positives': metric.false_positives,
            'false_negatives': metric.false_negatives,
            'annotation_time': metric.get_annotation_time_display(),
            'average_time_per_entity': round(metric.average_time_per_entity, 2) if metric.average_time_per_entity else 0.0,
        }
    })


@login_required
@require_http_methods(["POST"])
def create_gold_standard(request):
    """
    Create or update a gold standard annotation for a document
    """
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    document_id = request.POST.get('document_id')
    task_type = request.POST.get('task_type', 'ner')
    entities_json = request.POST.get('entities_json')
    validation_notes = request.POST.get('validation_notes', '')

    if not all([document_id, entities_json]):
        return JsonResponse({'success': False, 'error': 'Missing required parameters'}, status=400)

    try:
        document = TextDocument.objects.get(id=document_id)
        entities = json.loads(entities_json)
    except TextDocument.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Document not found'}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid entities JSON'}, status=400)

    # Create or update gold standard
    gold_standard, created = GoldStandard.objects.update_or_create(
        document=document,
        task_type=task_type,
        defaults={
            'entities_json': entities,
            'created_by': request.user,
            'validation_notes': validation_notes,
            'is_active': True
        }
    )

    return JsonResponse({
        'success': True,
        'gold_standard': {
            'id': gold_standard.id,
            'document_id': document.id,
            'task_type': gold_standard.task_type,
            'entity_count': gold_standard.get_entity_count(),
            'created': created
        },
        'message': 'Gold standard created successfully' if created else 'Gold standard updated successfully'
    })


@login_required
def annotator_performance_report(request, annotator_id=None):
    """
    Generate performance report for an annotator or all annotators
    """
    if not request.user.is_admin():
        messages.error(request, 'Admin access required.')
        return redirect('authentication:dashboard')

    # Get annotator if specified
    annotator = None
    if annotator_id:
        annotator = get_object_or_404(User, id=annotator_id, role='annotator')

    # Get all annotators for dropdown
    all_annotators = User.objects.filter(role='annotator', is_active=True)

    # Filter metrics by annotator
    metrics_queryset = AnnotationMetrics.objects.select_related(
        'annotation_result__annotator',
        'annotation_result__document',
        'gold_standard'
    )

    if annotator:
        metrics_queryset = metrics_queryset.filter(
            annotation_result__annotator=annotator
        )

    # Calculate aggregate statistics
    aggregate_stats = metrics_queryset.aggregate(
        avg_precision=Avg('precision'),
        avg_recall=Avg('recall'),
        avg_f1=Avg('f1_score'),
        total_annotations=Count('id'),
        avg_time=Avg('annotation_time_seconds')
    )

    # Get metrics by task type
    ner_metrics = metrics_queryset.filter(
        annotation_result__task_type='ner'
    ).aggregate(
        avg_precision=Avg('precision'),
        avg_recall=Avg('recall'),
        avg_f1=Avg('f1_score')
    )

    mcn_metrics = metrics_queryset.filter(
        annotation_result__task_type='mcn'
    ).aggregate(
        avg_precision=Avg('precision'),
        avg_recall=Avg('recall'),
        avg_f1=Avg('f1_score')
    )

    # Get recent annotations with metrics
    recent_metrics = metrics_queryset.order_by('-calculated_at')[:20]

    context = {
        'annotator': annotator,
        'all_annotators': all_annotators,
        'aggregate_stats': aggregate_stats,
        'ner_metrics': ner_metrics,
        'mcn_metrics': mcn_metrics,
        'recent_metrics': recent_metrics,
    }

    return render(request, 'annotation/annotator_performance.html', context)


@login_required
@require_http_methods(["GET"])
def export_metrics_data(request):
    """
    Export evaluation metrics data as JSON or CSV
    """
    if not request.user.is_admin():
        return JsonResponse({'success': False, 'error': 'Admin access required'}, status=403)

    export_format = request.GET.get('format', 'json')
    metric_type = request.GET.get('type', 'all')  # 'agreement', 'accuracy', 'efficiency', 'all'

    data = {
        'exported_at': timezone.now().isoformat(),
        'exported_by': request.user.username,
    }

    # Export agreement metrics
    if metric_type in ['agreement', 'all']:
        agreements = AnnotatorAgreement.objects.select_related(
            'document', 'annotator1', 'annotator2'
        ).all()

        data['agreement_metrics'] = [{
            'document_id': agreement.document_id,
            'document_title': agreement.document.title,
            'annotator1': agreement.annotator1.username,
            'annotator2': agreement.annotator2.username,
            'task_type': agreement.task_type,
            'cohens_kappa': agreement.cohens_kappa,
            'percent_agreement': agreement.percent_agreement,
            'agreement_level': agreement.get_agreement_level(),
            'agreed_entities': agreement.agreed_entities,
            'disagreed_entities': agreement.disagreed_entities,
        } for agreement in agreements]

    # Export accuracy metrics
    if metric_type in ['accuracy', 'all']:
        metrics = AnnotationMetrics.objects.select_related(
            'annotation_result__annotator',
            'annotation_result__document'
        ).all()

        data['accuracy_metrics'] = [{
            'annotation_id': metric.annotation_result_id,
            'document_id': metric.annotation_result.document_id,
            'document_title': metric.annotation_result.document.title,
            'annotator': metric.annotation_result.annotator.username,
            'task_type': metric.annotation_result.task_type,
            'precision': metric.precision,
            'recall': metric.recall,
            'f1_score': metric.f1_score,
            'true_positives': metric.true_positives,
            'false_positives': metric.false_positives,
            'false_negatives': metric.false_negatives,
        } for metric in metrics]

    # Export efficiency metrics
    if metric_type in ['efficiency', 'all']:
        tasks = AnnotationTask.objects.filter(
            is_completed=True,
            actual_duration__isnull=False
        ).select_related('annotator', 'document').all()

        data['efficiency_metrics'] = [{
            'task_id': task.id,
            'document_id': task.document_id,
            'document_title': task.document.title,
            'annotator': task.annotator.username,
            'task_type': task.task_type,
            'annotation_time_minutes': task.get_duration_minutes(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
        } for task in tasks]

    if export_format == 'json':
        return JsonResponse(data, json_dumps_params={'indent': 2})

    # CSV export would require additional formatting
    return JsonResponse({'success': False, 'error': 'CSV export not yet implemented'}, status=400)
