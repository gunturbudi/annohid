"""
Gamification views for leaderboard, achievements, and API endpoints.
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count
import json

from django.core.paginator import Paginator
from .gamification_service import GamificationService
from .gamification_models import Achievement, UserAchievement, AnnotatorProfile, GamificationConfig, PointTransaction
from .models import User


@login_required
def leaderboard_view(request):
    """
    Full leaderboard page with weekly and all-time rankings.
    """
    period = request.GET.get('period', 'weekly')
    
    # Get leaderboards
    weekly_leaderboard = GamificationService.get_leaderboard(period='weekly', limit=20)
    alltime_leaderboard = GamificationService.get_leaderboard(period='all_time', limit=20)
    
    # Get user's ranks
    weekly_rank = GamificationService.get_user_rank(request.user, period='weekly')
    alltime_rank = GamificationService.get_user_rank(request.user, period='all_time')
    
    # Get user's stats
    user_stats = GamificationService.get_user_stats(request.user)
    
    context = {
        'period': period,
        'weekly_leaderboard': weekly_leaderboard,
        'alltime_leaderboard': alltime_leaderboard,
        'weekly_rank': weekly_rank,
        'alltime_rank': alltime_rank,
        'user_stats': user_stats,
        'user': request.user,
    }
    
    return render(request, 'authentication/leaderboard.html', context)


@login_required
def achievements_view(request):
    """
    Full achievements page showing all available badges with progress.
    """
    user = request.user
    profile = AnnotatorProfile.get_or_create_profile(user)
    
    # Get all achievements
    all_achievements = Achievement.objects.filter(is_active=True).order_by('display_order')
    
    # Get user's earned achievements
    earned_ids = UserAchievement.objects.filter(user=user).values_list('achievement_id', flat=True)
    earned_achievements = UserAchievement.objects.filter(user=user).select_related('achievement')
    
    # Build achievements with progress
    achievements_data = []
    for achievement in all_achievements:
        is_earned = achievement.id in earned_ids
        progress = 0
        progress_total = 0
        
        # Calculate progress
        if achievement.required_annotations:
            progress = min(profile.total_annotations, achievement.required_annotations)
            progress_total = achievement.required_annotations
        elif achievement.required_entities:
            progress = min(profile.total_entities, achievement.required_entities)
            progress_total = achievement.required_entities
        elif achievement.required_streak:
            progress = min(profile.longest_streak, achievement.required_streak)
            progress_total = achievement.required_streak
        elif achievement.required_ner_count:
            progress = min(profile.ner_annotations, achievement.required_ner_count)
            progress_total = achievement.required_ner_count
        elif achievement.required_mcn_count:
            progress = min(profile.mcn_annotations, achievement.required_mcn_count)
            progress_total = achievement.required_mcn_count
        
        # Get earned timestamp if earned
        earned_at = None
        if is_earned:
            ua = earned_achievements.filter(achievement=achievement).first()
            if ua:
                earned_at = ua.earned_at
        
        achievements_data.append({
            'achievement': achievement,
            'is_earned': is_earned,
            'earned_at': earned_at,
            'progress': progress,
            'progress_total': progress_total,
            'progress_percent': int((progress / progress_total * 100)) if progress_total > 0 else 0,
        })
    
    # Group by difficulty
    bronze = [a for a in achievements_data if a['achievement'].difficulty == 'bronze']
    silver = [a for a in achievements_data if a['achievement'].difficulty == 'silver']
    gold = [a for a in achievements_data if a['achievement'].difficulty == 'gold']
    platinum = [a for a in achievements_data if a['achievement'].difficulty == 'platinum']
    
    context = {
        'achievements_data': achievements_data,
        'bronze': bronze,
        'silver': silver,
        'gold': gold,
        'platinum': platinum,
        'profile': profile,
        'total_achievements': all_achievements.count(),
        'earned_count': len(earned_ids),
        'user': user,
    }
    
    return render(request, 'authentication/achievements.html', context)


@login_required
@require_http_methods(["POST"])
def award_points_api(request):
    """
    API endpoint to award points after an annotation is completed.
    Called by the annotation views after saving an annotation.
    """
    try:
        data = json.loads(request.body)
        
        task_type = data.get('task_type', 'ner')
        annotation_id = data.get('annotation_id')
        confidence_score = data.get('confidence_score')
        duration_seconds = data.get('duration_seconds')
        
        # Get annotation result if provided
        annotation_result = None
        if annotation_id:
            from annotation.models import AnnotationResult
            try:
                annotation_result = AnnotationResult.objects.get(id=annotation_id)
            except AnnotationResult.DoesNotExist:
                pass
        
        # Award points
        result = GamificationService.award_annotation_points(
            user=request.user,
            task_type=task_type,
            annotation_result=annotation_result,
            confidence_score=confidence_score,
            duration_seconds=duration_seconds
        )
        
        return JsonResponse({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@login_required
def gamification_config_view(request):
    """Admin page to configure gamification rules."""
    if not request.user.is_admin:
        messages.error(request, 'Access denied. Admin only.')
        return redirect('authentication:dashboard')

    config = GamificationConfig.get_config()

    if request.method == 'POST':
        action = request.POST.get('action', 'save')

        if action == 'reset_defaults':
            config.points_ner_complete = 10
            config.points_mcn_complete = 15
            config.points_first_of_day = 5
            config.points_speed_bonus = 3
            config.points_quality_bonus = 2
            config.points_streak_per_day = 1
            config.points_streak_max = 7
            config.points_level_up = 10
            config.speed_threshold_ner = 180
            config.speed_threshold_mcn = 300
            config.quality_threshold = 0.9
            config.streak_enabled = True
            config.speed_bonus_enabled = True
            config.quality_bonus_enabled = True
            config.first_of_day_enabled = True
            config.leaderboard_enabled = True
            config.weekly_reset_day = 0
            config.level_definitions = GamificationConfig.DEFAULT_LEVELS
            config.updated_by = request.user
            config.save()
            messages.success(request, 'Gamification settings reset to defaults.')
            return redirect('authentication:gamification_config')

        # Save config
        try:
            config.points_ner_complete = int(request.POST.get('points_ner_complete', 10))
            config.points_mcn_complete = int(request.POST.get('points_mcn_complete', 15))
            config.points_first_of_day = int(request.POST.get('points_first_of_day', 5))
            config.points_speed_bonus = int(request.POST.get('points_speed_bonus', 3))
            config.points_quality_bonus = int(request.POST.get('points_quality_bonus', 2))
            config.points_streak_per_day = int(request.POST.get('points_streak_per_day', 1))
            config.points_streak_max = int(request.POST.get('points_streak_max', 7))
            config.points_level_up = int(request.POST.get('points_level_up', 10))

            config.speed_threshold_ner = int(request.POST.get('speed_threshold_ner', 180))
            config.speed_threshold_mcn = int(request.POST.get('speed_threshold_mcn', 300))
            config.quality_threshold = float(request.POST.get('quality_threshold', 0.9))
            config.weekly_reset_day = int(request.POST.get('weekly_reset_day', 0))

            config.streak_enabled = request.POST.get('streak_enabled') == 'on'
            config.speed_bonus_enabled = request.POST.get('speed_bonus_enabled') == 'on'
            config.quality_bonus_enabled = request.POST.get('quality_bonus_enabled') == 'on'
            config.first_of_day_enabled = request.POST.get('first_of_day_enabled') == 'on'
            config.leaderboard_enabled = request.POST.get('leaderboard_enabled') == 'on'

            # Parse level definitions from hidden JSON field
            levels_json = request.POST.get('level_definitions', '')
            if levels_json:
                levels = json.loads(levels_json)
                # Validate
                validated = []
                for row in levels:
                    lvl = int(row[0])
                    title = str(row[1]).strip()
                    threshold = int(row[2])
                    if title:
                        validated.append([lvl, title, threshold])
                if validated:
                    validated.sort(key=lambda x: x[0])
                    config.level_definitions = validated

            config.updated_by = request.user
            config.save()
            messages.success(request, 'Gamification settings saved successfully.')
        except (ValueError, json.JSONDecodeError) as e:
            messages.error(request, f'Invalid input: {e}')

        return redirect('authentication:gamification_config')

    # GET: gather summary stats
    annotator_count = User.objects.filter(role__in=['annotator', 'expert'], is_active=True).count()
    active_streaks = AnnotatorProfile.objects.filter(current_streak__gt=0).count()
    achievement_count = UserAchievement.objects.count()
    avg_points = 0
    profiles = AnnotatorProfile.objects.all()
    if profiles.exists():
        from django.db.models import Avg
        avg_points = int(profiles.aggregate(avg=Avg('total_points'))['avg'] or 0)

    # Level distribution
    level_distribution = (
        AnnotatorProfile.objects
        .values('level')
        .annotate(count=Count('id'))
        .order_by('level')
    )
    level_dist_map = {row['level']: row['count'] for row in level_distribution}

    context = {
        'config': config,
        'annotator_count': annotator_count,
        'active_streaks': active_streaks,
        'achievement_count': achievement_count,
        'avg_points': avg_points,
        'level_distribution': level_dist_map,
        'level_definitions_json': json.dumps(config.level_definitions or GamificationConfig.DEFAULT_LEVELS),
        'level_distribution_json': json.dumps(level_dist_map),
    }

    return render(request, 'authentication/gamification_config.html', context)


@login_required
def points_history_view(request):
    """Points history page showing all point transactions for the current user."""
    user = request.user
    profile = AnnotatorProfile.get_or_create_profile(user)

    # Optional filter by action_type
    action_filter = request.GET.get('action', '')
    transactions = PointTransaction.objects.filter(user=user)
    if action_filter:
        transactions = transactions.filter(action_type=action_filter)

    # Get distinct action types for filter tabs
    action_types = (
        PointTransaction.objects.filter(user=user)
        .values_list('action_type', flat=True)
        .distinct()
        .order_by('action_type')
    )
    # Build label map from ACTION_TYPES choices
    action_label_map = dict(PointTransaction.ACTION_TYPES)
    filter_tabs = [
        {'value': at, 'label': action_label_map.get(at, at)}
        for at in action_types
    ]

    paginator = Paginator(transactions, 25)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'profile': profile,
        'page_obj': page_obj,
        'action_filter': action_filter,
        'filter_tabs': filter_tabs,
    }
    return render(request, 'authentication/points_history.html', context)
