from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q
from .models import User, UserSession
from .forms import CustomUserCreationForm, CustomUserEditForm, CustomPasswordChangeForm, UserSearchForm
import json


def login_view(request):
    """Handle user login"""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username and password:
            user = authenticate(request, username=username, password=password)
            if user is not None:
                # Update last activity
                user.last_activity = timezone.now()
                user.save()

                # Log in the user
                login(request, user)

                # Track session
                if hasattr(request, 'session'):
                    session_key = request.session.session_key
                    if session_key:
                        UserSession.objects.update_or_create(
                            session_key=session_key,
                            defaults={
                                'user': user,
                                'ip_address': get_client_ip(request),
                                'user_agent': request.META.get('HTTP_USER_AGENT', ''),
                                'is_active': True
                            }
                        )

                messages.success(request, f'Welcome, {user.username}!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please provide both username and password.')

    return render(request, 'authentication/login.html')


@login_required
def logout_view(request):
    """Handle user logout"""
    # Mark session as inactive
    if hasattr(request, 'session') and request.session.session_key:
        UserSession.objects.filter(
            session_key=request.session.session_key
        ).update(is_active=False)

    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('login')


@login_required
def dashboard_view(request):
    """Main dashboard view"""
    user = request.user
    context = {
        'user': user,
        'is_admin': user.is_admin(),
        'is_annotator': user.is_annotator(),
    }

    if user.is_admin():
        # Admin dashboard with user statistics
        users = User.objects.all().order_by('-last_activity')
        active_sessions = UserSession.objects.filter(is_active=True)

        context.update({
            'users': users,
            'active_sessions': active_sessions,
            'total_users': users.count(),
            'total_annotators': users.filter(role='annotator').count(),
            'total_experts': users.filter(role='expert').count(),
            'total_admins': users.filter(role='admin').count(),
        })
        return render(request, 'authentication/admin_dashboard.html', context)
    else:
        # Annotator dashboard with statistics
        from annotation.models import AnnotationTask, AnnotationResult, TextDocument
        from datetime import timedelta

        # Calculate statistics for the current annotator
        today = timezone.now().date()

        # Annotations today
        annotations_today = AnnotationResult.objects.filter(
            annotator=user,
            created_at__date=today
        ).count()

        # Total annotations
        total_annotations = AnnotationResult.objects.filter(annotator=user).count()

        # Documents processed (completed tasks)
        documents_processed = AnnotationTask.objects.filter(
            annotator=user,
            is_completed=True
        ).count()

        # Average time per document (only for completed tasks with actual duration)
        completed_tasks = AnnotationTask.objects.filter(
            annotator=user,
            is_completed=True,
            actual_duration__isnull=False
        )

        avg_time_per_doc = None
        if completed_tasks.exists():
            total_duration = sum([task.actual_duration.total_seconds() for task in completed_tasks], 0)
            avg_seconds = total_duration / completed_tasks.count()
            avg_time_per_doc = f"{int(avg_seconds // 60)}m {int(avg_seconds % 60)}s"

        # Recent annotation results for activity feed
        recent_annotations = AnnotationResult.objects.filter(
            annotator=user
        ).order_by('-created_at')[:5]

        # Activity breakdown: NER LLM accept/reject stats
        ner_llm_results = AnnotationResult.objects.filter(
            annotator=user,
            annotation_type='llm_assisted'
        )
        ner_accepted = 0
        ner_rejected = 0
        ner_manual_added = 0
        for result in ner_llm_results:
            for entity in result.entities_json:
                source = entity.get('source', 'llm')
                status = entity.get('status', 'pending')
                if source == 'llm':
                    if status in ('accepted', 'approved'):
                        ner_accepted += 1
                    elif status == 'rejected':
                        ner_rejected += 1
                elif source == 'manual':
                    ner_manual_added += 1

        # MCN mapping stats
        mcn_results = AnnotationResult.objects.filter(
            annotator=user,
            annotation_type__in=['mcn_llm_assisted', 'mcn_manual']
        )
        mcn_mapped = 0
        mcn_unmapped = 0
        for result in mcn_results:
            for entity in result.entities_json:
                mcn_source = entity.get('mcn_source')
                mcn_status = entity.get('mcn_status')
                if mcn_source:
                    if mcn_status in ('accepted', 'modified', 'mapped'):
                        mcn_mapped += 1
                    else:
                        mcn_unmapped += 1
                else:
                    # Legacy: infer from is_mapped
                    if entity.get('is_mapped', False):
                        mcn_mapped += 1
                    else:
                        mcn_unmapped += 1

        # Entity type distribution across all results
        from collections import Counter
        entity_type_counts = Counter()
        all_results = AnnotationResult.objects.filter(annotator=user)
        for result in all_results:
            for entity in result.entities_json:
                label = entity.get('label', 'UNKNOWN')
                entity_type_counts[label] += 1

        ner_total_decided = ner_accepted + ner_rejected
        mcn_total = mcn_mapped + mcn_unmapped

        activity_breakdown = {
            'ner_accepted': ner_accepted,
            'ner_rejected': ner_rejected,
            'ner_manual_added': ner_manual_added,
            'ner_acceptance_rate': round((ner_accepted / ner_total_decided * 100) if ner_total_decided > 0 else 0, 1),
            'ner_has_data': ner_total_decided > 0,
            'mcn_mapped': mcn_mapped,
            'mcn_unmapped': mcn_unmapped,
            'mcn_mapping_rate': round((mcn_mapped / mcn_total * 100) if mcn_total > 0 else 0, 1),
            'mcn_has_data': mcn_total > 0,
            'entity_types': dict(entity_type_counts.most_common(10)),
            'has_entity_data': bool(entity_type_counts),
        }

        # Add annotator-specific context
        context.update({
            'annotations_today': annotations_today,
            'total_annotations': total_annotations,
            'documents_processed': documents_processed,
            'avg_time_per_doc': avg_time_per_doc,
            'recent_annotations': recent_annotations,
            'activity_breakdown': activity_breakdown,
        })
        
        # Add gamification context
        try:
            from .gamification_service import GamificationService
            
            # Get user stats
            gamification_stats = GamificationService.get_user_stats(user)
            
            # Get leaderboard preview (top 5)
            weekly_leaderboard = GamificationService.get_leaderboard(period='weekly', limit=5)
            
            # Get user's rank
            user_rank = GamificationService.get_user_rank(user, period='weekly')
            
            context.update({
                'gamification': gamification_stats,
                'weekly_leaderboard': weekly_leaderboard,
                'user_rank': user_rank,
                'has_gamification': True,
            })
            
            # Mark achievements as notified after displaying
            if gamification_stats.get('unnotified_achievements'):
                request.session['new_achievements'] = [
                    {
                        'name': ua.achievement.name,
                        'icon': ua.achievement.icon,
                        'description': ua.achievement.description,
                    }
                    for ua in gamification_stats['unnotified_achievements']
                ]
                GamificationService.mark_achievements_notified(user)
        except Exception as e:
            # Gamification not yet set up or error occurred
            context['has_gamification'] = False
            context['gamification_error'] = str(e)

        return render(request, 'authentication/annotator_dashboard.html', context)


@login_required
@require_http_methods(["POST"])
def update_activity(request):
    """Update user's last activity (AJAX endpoint)"""
    if request.user.is_authenticated:
        request.user.last_activity = timezone.now()
        request.user.save()

        # Update session activity
        if hasattr(request, 'session') and request.session.session_key:
            UserSession.objects.filter(
                session_key=request.session.session_key
            ).update(last_activity=timezone.now())

        return JsonResponse({'success': True})

    return JsonResponse({'success': False}, status=401)


def get_client_ip(request):
    """Get client IP address from request"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


@login_required
def profile_view(request):
    """User profile view"""
    user = request.user
    user_sessions = UserSession.objects.filter(user=user).order_by('-last_activity')[:10]

    context = {
        'user': user,
        'recent_sessions': user_sessions,
    }

    return render(request, 'authentication/profile.html', context)


def admin_required(func):
    """Decorator to require admin role"""
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_admin():
            messages.error(request, 'You do not have permission to access this page.')
            return redirect('dashboard')
        return func(request, *args, **kwargs)
    return wrapper


@login_required
@admin_required
def user_management(request):
    """User management dashboard for admins"""
    search_form = UserSearchForm(request.GET)
    users = User.objects.all()

    # Apply search filters
    if search_form.is_valid():
        search_query = search_form.cleaned_data.get('search')
        role_filter = search_form.cleaned_data.get('role')
        is_active_filter = search_form.cleaned_data.get('is_active')

        if search_query:
            users = users.filter(
                Q(username__icontains=search_query) |
                Q(email__icontains=search_query) |
                Q(first_name__icontains=search_query) |
                Q(last_name__icontains=search_query)
            )

        if role_filter:
            users = users.filter(role=role_filter)

        if is_active_filter:
            users = users.filter(is_active=(is_active_filter == '1'))

    users = users.order_by('-created_at')

    # Pagination
    paginator = Paginator(users, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_form': search_form,
        'total_users': User.objects.count(),
        'active_users': User.objects.filter(is_active=True).count(),
        'admin_users': User.objects.filter(role='admin').count(),
        'annotator_users': User.objects.filter(role='annotator').count(),
        'expert_users': User.objects.filter(role='expert').count(),
    }

    return render(request, 'authentication/user_management.html', context)


@login_required
@admin_required
def create_user(request):
    """Create new user"""
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, f'User "{user.username}" has been created successfully.')
            return redirect('authentication:user_management')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserCreationForm()

    context = {
        'form': form,
        'title': 'Create New User'
    }

    return render(request, 'authentication/create_user.html', context)


@login_required
@admin_required
def edit_user(request, user_id):
    """Edit existing user"""
    user = get_object_or_404(User, id=user_id)

    if request.method == 'POST':
        form = CustomUserEditForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, f'User "{user.username}" has been updated successfully.')
            return redirect('authentication:user_management')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomUserEditForm(instance=user)

    context = {
        'form': form,
        'user_obj': user,
        'title': f'Edit User: {user.username}'
    }

    return render(request, 'authentication/edit_user.html', context)


@login_required
@admin_required
def change_user_password(request, user_id):
    """Change user password"""
    user = get_object_or_404(User, id=user_id)

    if request.method == 'POST':
        form = CustomPasswordChangeForm(user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, f'Password for "{user.username}" has been changed successfully.')
            return redirect('authentication:user_management')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CustomPasswordChangeForm(user)

    context = {
        'form': form,
        'user_obj': user,
        'title': f'Change Password: {user.username}'
    }

    return render(request, 'authentication/change_password.html', context)


@login_required
@admin_required
def delete_user(request, user_id):
    """Delete user"""
    user = get_object_or_404(User, id=user_id)

    if user == request.user:
        messages.error(request, 'You cannot delete your own account.')
        return redirect('authentication:user_management')

    if request.method == 'POST':
        username = user.username
        user.delete()
        messages.success(request, f'User "{username}" has been deleted successfully.')
        return redirect('authentication:user_management')

    context = {
        'user_obj': user,
        'title': f'Delete User: {user.username}'
    }

    return render(request, 'authentication/delete_user.html', context)


@login_required
@admin_required
@require_http_methods(["POST"])
def reset_all_data(request):
    """Reset all annotation data - DANGER ZONE"""
    confirmation = request.POST.get('confirmation', '')
    if confirmation != 'RESET ALL DATA':
        return JsonResponse({
            'success': False,
            'error': 'Invalid confirmation phrase.'
        }, status=400)

    try:
        from annotation.models import (
            TextDocument, TextImportBatch, AnnotationTask, AnnotationResult,
            AnnotationStatistics, AutoMCNMapping, GoldStandard,
            AnnotatorAgreement, AnnotationMetrics
        )
        from .gamification_models import AnnotatorProfile, UserAchievement, PointTransaction

        counts = {}

        # Delete annotation data
        counts['annotation_metrics'] = AnnotationMetrics.objects.all().delete()[0]
        counts['annotator_agreement'] = AnnotatorAgreement.objects.all().delete()[0]
        counts['gold_standards'] = GoldStandard.objects.all().delete()[0]
        counts['annotation_statistics'] = AnnotationStatistics.objects.all().delete()[0]
        counts['annotation_results'] = AnnotationResult.objects.all().delete()[0]
        counts['annotation_tasks'] = AnnotationTask.objects.all().delete()[0]
        counts['auto_mcn_mappings'] = AutoMCNMapping.objects.all().delete()[0]
        counts['import_batches'] = TextImportBatch.objects.all().delete()[0]
        counts['documents'] = TextDocument.objects.all().delete()[0]

        # Delete gamification data
        counts['point_transactions'] = PointTransaction.objects.all().delete()[0]
        counts['user_achievements'] = UserAchievement.objects.all().delete()[0]
        counts['annotator_profiles'] = AnnotatorProfile.objects.all().delete()[0]

        total_deleted = sum(counts.values())

        return JsonResponse({
            'success': True,
            'message': f'All data has been reset. {total_deleted} records deleted.',
            'data': counts
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error resetting data: {str(e)}'
        }, status=500)


@login_required
@admin_required
@require_http_methods(["POST"])
def quick_create_annotator(request):
    """Quick create annotator with auto-generated username and standard password"""
    try:
        # Get the next annotator number based on existing annotators
        existing_annotators = User.objects.filter(
            username__startswith='annotator'
        ).order_by('username')

        # Find the next available number
        next_number = 1
        for annotator in existing_annotators:
            username = annotator.username
            if username.startswith('annotator') and username[9:].isdigit():
                num = int(username[9:])
                if num >= next_number:
                    next_number = num + 1

        # Create new annotator
        username = f'annotator{next_number}'
        password = 'ann123'  # Standard password for quick-created annotators

        # Check if username already exists (shouldn't happen but safety check)
        if User.objects.filter(username=username).exists():
            return JsonResponse({
                'success': False,
                'error': f'Username "{username}" already exists.'
            }, status=400)

        # Create the user
        user = User.objects.create_user(
            username=username,
            password=password,
            role='annotator',
            annotation_mode='manual_only',
            is_active=True
        )

        return JsonResponse({
            'success': True,
            'message': f'Annotator created successfully!',
            'data': {
                'username': username,
                'password': password,
                'role': 'Annotator',
                'annotation_mode': user.get_annotation_mode_display(),
            }
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error creating annotator: {str(e)}'
        }, status=500)
