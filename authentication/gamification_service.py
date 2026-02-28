"""
Gamification service for the annotation platform.
Handles point calculations, level progression, streak tracking, and achievement unlocks.
"""
from django.utils import timezone
from django.db import transaction
from datetime import timedelta


class GamificationService:
    """
    Service class to handle all gamification logic.
    Reads point values, thresholds, and feature toggles from GamificationConfig.
    """

    @classmethod
    def _get_config(cls):
        from .gamification_models import GamificationConfig
        return GamificationConfig.get_config()

    @classmethod
    def award_annotation_points(cls, user, task_type, annotation_result=None, confidence_score=None, duration_seconds=None):
        """
        Award points for completing an annotation.

        Args:
            user: The user who completed the annotation
            task_type: 'ner' or 'mcn'
            annotation_result: Optional AnnotationResult instance
            confidence_score: Optional average confidence (0-1)
            duration_seconds: Optional time taken

        Returns:
            dict with points breakdown and any achievements earned
        """
        from .gamification_models import AnnotatorProfile, PointTransaction

        config = cls._get_config()
        points = config.get_points_dict()

        profile = AnnotatorProfile.get_or_create_profile(user)
        today = timezone.now().date()

        points_breakdown = []
        total_points = 0
        achievements_earned = []

        # Base points for annotation type
        if task_type == 'ner':
            base_points = points['ner_complete']
            points_breakdown.append(('NER Completed', base_points))
            profile.ner_annotations += 1
        else:  # mcn
            base_points = points['mcn_complete']
            points_breakdown.append(('MCN Completed', base_points))
            profile.mcn_annotations += 1

        total_points += base_points
        profile.total_annotations += 1

        # Track entities if available
        if annotation_result and hasattr(annotation_result, 'total_entities'):
            profile.total_entities += annotation_result.total_entities

        # First annotation of the day bonus
        if config.first_of_day_enabled and profile.last_annotation_date != today:
            bonus = points['first_of_day']
            total_points += bonus
            points_breakdown.append(('First of Day Bonus', bonus))

            PointTransaction.objects.create(
                user=user,
                action_type='first_of_day',
                points=bonus,
                description='First annotation of the day'
            )

        # Streak bonus
        if config.streak_enabled:
            streak_bonus = cls._update_streak(profile, today)
            if streak_bonus > 0:
                total_points += streak_bonus
                points_breakdown.append((f'Streak Bonus (Day {profile.current_streak})', streak_bonus))

                PointTransaction.objects.create(
                    user=user,
                    action_type='streak_bonus',
                    points=streak_bonus,
                    description=f'{profile.current_streak} day streak'
                )
        else:
            # Still update the date tracking even if bonus is disabled
            cls._update_streak(profile, today)

        # Quality bonus
        if config.quality_bonus_enabled and confidence_score and confidence_score > config.quality_threshold:
            bonus = points['quality_bonus']
            total_points += bonus
            points_breakdown.append((f'Quality Bonus (>{config.quality_threshold:.0%} confidence)', bonus))

            PointTransaction.objects.create(
                user=user,
                action_type='quality_bonus',
                points=bonus,
                description=f'High confidence: {confidence_score:.1%}'
            )

        # Speed bonus
        if config.speed_bonus_enabled:
            target_time = config.speed_threshold_ner if task_type == 'ner' else config.speed_threshold_mcn
            if duration_seconds and duration_seconds < target_time:
                bonus = points['speed_bonus']
                total_points += bonus
                points_breakdown.append(('Speed Bonus', bonus))

                PointTransaction.objects.create(
                    user=user,
                    action_type='speed_bonus',
                    points=bonus,
                    description=f'Completed in {duration_seconds:.0f}s'
                )

        # Update profile totals
        profile.total_points += total_points
        profile.weekly_points += total_points
        profile.last_annotation_date = today

        # Create main transaction
        PointTransaction.objects.create(
            user=user,
            action_type='ner_complete' if task_type == 'ner' else 'mcn_complete',
            points=base_points,
            description=f'{task_type.upper()} annotation completed',
            annotation_id=annotation_result.id if annotation_result else None
        )

        # Check for level up
        level_up = cls._check_level_up(profile)
        if level_up:
            level_up_bonus = points['level_up']
            points_breakdown.append(('Level Up Bonus', level_up_bonus))
            total_points += level_up_bonus
            profile.total_points += level_up_bonus
            profile.weekly_points += level_up_bonus

        profile.save()

        # Check achievements
        achievements_earned = cls._check_achievements(user, profile)

        return {
            'total_points': total_points,
            'breakdown': points_breakdown,
            'new_total': profile.total_points,
            'level': profile.level,
            'level_title': profile.get_level_title(),
            'level_up': level_up,
            'streak': profile.current_streak,
            'achievements_earned': achievements_earned,
        }

    @classmethod
    def _update_streak(cls, profile, today):
        """Update streak and return bonus points earned"""
        config = cls._get_config()
        points = config.get_points_dict()
        bonus = 0

        if profile.last_annotation_date is None:
            # First ever annotation
            profile.current_streak = 1
        elif profile.last_annotation_date == today:
            # Already annotated today, no streak change
            pass
        elif profile.last_annotation_date == today - timedelta(days=1):
            # Consecutive day - extend streak
            profile.current_streak += 1
            bonus = min(profile.current_streak, points['streak_bonus_max'])
        else:
            # Streak broken - reset
            profile.current_streak = 1

        # Update longest streak
        if profile.current_streak > profile.longest_streak:
            profile.longest_streak = profile.current_streak

        return bonus

    @classmethod
    def _check_level_up(cls, profile):
        """Check if user should level up, returns True if leveled up"""
        config = cls._get_config()
        thresholds = config.get_level_thresholds()
        current_level = profile.level

        for level, title, threshold in thresholds:
            if level > current_level and profile.total_points >= threshold:
                profile.level = level

        return profile.level > current_level

    @classmethod
    def _check_achievements(cls, user, profile):
        """Check and award any earned achievements"""
        from .gamification_models import Achievement, UserAchievement, PointTransaction

        earned = []

        # Get all active achievements user hasn't earned yet
        earned_codes = UserAchievement.objects.filter(user=user).values_list('achievement__code', flat=True)
        available_achievements = Achievement.objects.filter(is_active=True).exclude(code__in=earned_codes)

        for achievement in available_achievements:
            should_award = False

            # Check requirements
            if achievement.required_annotations and profile.total_annotations >= achievement.required_annotations:
                should_award = True
            elif achievement.required_entities and profile.total_entities >= achievement.required_entities:
                should_award = True
            elif achievement.required_streak and profile.current_streak >= achievement.required_streak:
                should_award = True
            elif achievement.required_ner_count and profile.ner_annotations >= achievement.required_ner_count:
                should_award = True
            elif achievement.required_mcn_count and profile.mcn_annotations >= achievement.required_mcn_count:
                should_award = True

            if should_award:
                # Award achievement
                UserAchievement.objects.create(
                    user=user,
                    achievement=achievement,
                    context={'triggered_by': 'auto_check'}
                )

                # Award bonus points
                if achievement.points_reward > 0:
                    profile.total_points += achievement.points_reward
                    profile.weekly_points += achievement.points_reward
                    profile.save()

                    PointTransaction.objects.create(
                        user=user,
                        action_type='achievement',
                        points=achievement.points_reward,
                        description=f'Achievement: {achievement.name}',
                        achievement_id=achievement.id
                    )

                earned.append({
                    'code': achievement.code,
                    'name': achievement.name,
                    'icon': achievement.icon,
                    'description': achievement.description,
                    'points_reward': achievement.points_reward,
                })

        return earned

    @classmethod
    def get_user_stats(cls, user):
        """Get comprehensive gamification stats for a user"""
        from .gamification_models import AnnotatorProfile, UserAchievement, PointTransaction

        profile = AnnotatorProfile.get_or_create_profile(user)

        # Recent achievements
        recent_achievements = UserAchievement.objects.filter(user=user).select_related('achievement')[:5]

        # Unnotified achievements
        unnotified = UserAchievement.objects.filter(user=user, is_notified=False).select_related('achievement')

        # Recent point history
        recent_transactions = PointTransaction.objects.filter(user=user)[:10]

        return {
            'profile': profile,
            'total_points': profile.total_points,
            'weekly_points': profile.weekly_points,
            'level': profile.level,
            'level_title': profile.get_level_title(),
            'level_progress': profile.get_level_progress_percent(),
            'points_for_next_level': profile.get_points_for_next_level(),
            'current_streak': profile.current_streak,
            'longest_streak': profile.longest_streak,
            'total_annotations': profile.total_annotations,
            'total_entities': profile.total_entities,
            'recent_achievements': recent_achievements,
            'unnotified_achievements': list(unnotified),
            'recent_transactions': recent_transactions,
        }

    @classmethod
    def get_leaderboard(cls, period='weekly', limit=10):
        """
        Get leaderboard data.

        Args:
            period: 'weekly' or 'all_time'
            limit: Number of users to return

        Returns:
            List of user stats sorted by points
        """
        from .gamification_models import AnnotatorProfile
        from .models import User

        config = cls._get_config()
        if not config.leaderboard_enabled:
            return []

        # Only include annotators/experts, not admins
        annotator_users = User.objects.filter(role__in=['annotator', 'expert'], is_active=True)

        if period == 'weekly':
            profiles = AnnotatorProfile.objects.filter(
                user__in=annotator_users
            ).select_related('user').order_by('-weekly_points')[:limit]

            return [{
                'rank': idx + 1,
                'username': p.user.username,
                'points': p.weekly_points,
                'level': p.level,
                'level_title': p.get_level_title(),
                'streak': p.current_streak,
                'user_id': p.user.id,
            } for idx, p in enumerate(profiles)]
        else:
            profiles = AnnotatorProfile.objects.filter(
                user__in=annotator_users
            ).select_related('user').order_by('-total_points')[:limit]

            return [{
                'rank': idx + 1,
                'username': p.user.username,
                'points': p.total_points,
                'level': p.level,
                'level_title': p.get_level_title(),
                'streak': p.current_streak,
                'user_id': p.user.id,
            } for idx, p in enumerate(profiles)]

    @classmethod
    def get_user_rank(cls, user, period='weekly'):
        """Get the user's rank on the leaderboard"""
        from .gamification_models import AnnotatorProfile
        from .models import User

        profile = AnnotatorProfile.get_or_create_profile(user)
        annotator_users = User.objects.filter(role__in=['annotator', 'expert'], is_active=True)

        if period == 'weekly':
            rank = AnnotatorProfile.objects.filter(
                user__in=annotator_users,
                weekly_points__gt=profile.weekly_points
            ).count() + 1
        else:
            rank = AnnotatorProfile.objects.filter(
                user__in=annotator_users,
                total_points__gt=profile.total_points
            ).count() + 1

        total_users = AnnotatorProfile.objects.filter(user__in=annotator_users).count()

        return {
            'rank': rank,
            'total_users': total_users,
            'percentile': int((1 - (rank / max(total_users, 1))) * 100) if total_users > 0 else 0
        }

    @classmethod
    def mark_achievements_notified(cls, user):
        """Mark all unnotified achievements as notified"""
        from .gamification_models import UserAchievement

        UserAchievement.objects.filter(user=user, is_notified=False).update(is_notified=True)


def setup_default_achievements():
    """
    Create default achievements. Call this once during initial setup.
    """
    from authentication.gamification_models import Achievement

    default_achievements = [
        {
            'code': 'first_steps',
            'name': 'First Steps',
            'description': 'Complete your first annotation',
            'icon': '🎯',
            'difficulty': 'bronze',
            'points_reward': 5,
            'required_annotations': 1,
            'display_order': 1,
        },
        {
            'code': 'getting_started',
            'name': 'Getting Started',
            'description': 'Complete 10 annotations',
            'icon': '📝',
            'difficulty': 'bronze',
            'points_reward': 10,
            'required_annotations': 10,
            'display_order': 2,
        },
        {
            'code': 'dedicated_daily',
            'name': 'Daily Dedication',
            'description': 'Maintain a 7-day annotation streak',
            'icon': '🔥',
            'difficulty': 'silver',
            'points_reward': 25,
            'required_streak': 7,
            'display_order': 3,
        },
        {
            'code': 'century_club',
            'name': 'Century Club',
            'description': 'Complete 100 annotations',
            'icon': '💯',
            'difficulty': 'silver',
            'points_reward': 50,
            'required_annotations': 100,
            'display_order': 4,
        },
        {
            'code': 'entity_expert',
            'name': 'Entity Expert',
            'description': 'Annotate 500 entities',
            'icon': '🏷️',
            'difficulty': 'silver',
            'points_reward': 30,
            'required_entities': 500,
            'display_order': 5,
        },
        {
            'code': 'mcn_specialist',
            'name': 'MCN Specialist',
            'description': 'Complete 50 MCN annotations',
            'icon': '🧬',
            'difficulty': 'gold',
            'points_reward': 40,
            'required_mcn_count': 50,
            'display_order': 6,
        },
        {
            'code': 'ner_master',
            'name': 'NER Master',
            'description': 'Complete 100 NER annotations',
            'icon': '🔍',
            'difficulty': 'gold',
            'points_reward': 40,
            'required_ner_count': 100,
            'display_order': 7,
        },
        {
            'code': 'marathon_streak',
            'name': 'Marathon Streak',
            'description': 'Maintain a 30-day annotation streak',
            'icon': '🏃',
            'difficulty': 'gold',
            'points_reward': 100,
            'required_streak': 30,
            'display_order': 8,
        },
        {
            'code': 'annotation_legend',
            'name': 'Annotation Legend',
            'description': 'Complete 500 annotations',
            'icon': '👑',
            'difficulty': 'platinum',
            'points_reward': 200,
            'required_annotations': 500,
            'display_order': 9,
        },
    ]

    created_count = 0
    for achievement_data in default_achievements:
        achievement, created = Achievement.objects.update_or_create(
            code=achievement_data['code'],
            defaults=achievement_data
        )
        if created:
            created_count += 1

    return created_count
