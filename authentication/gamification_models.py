"""
Gamification models for the annotation platform.
Tracks points, levels, streaks, and achievements for annotators.
"""
import json
from django.db import models
from django.core.cache import cache
from django.utils import timezone
from .models import User


class GamificationConfig(models.Model):
    """
    Singleton configuration for gamification rules.
    Only one row (pk=1) should exist. Use get_config() to access.
    """

    WEEKLY_RESET_DAY_CHOICES = [
        (0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'),
        (3, 'Thursday'), (4, 'Friday'), (5, 'Saturday'), (6, 'Sunday'),
    ]

    DEFAULT_LEVELS = [
        [1, 'Novice Annotator', 0],
        [2, 'Junior Annotator', 50],
        [3, 'Annotator', 150],
        [4, 'Senior Annotator', 400],
        [5, 'Expert Annotator', 800],
        [6, 'Master Annotator', 1500],
        [7, 'Grand Master', 3000],
        [8, 'Annotation Legend', 5000],
    ]

    CACHE_KEY = 'gamification_config'
    CACHE_TTL = 300  # 5 minutes

    # Point values
    points_ner_complete = models.PositiveIntegerField(default=10, help_text="Points for completing an NER annotation")
    points_mcn_complete = models.PositiveIntegerField(default=15, help_text="Points for completing an MCN annotation")
    points_first_of_day = models.PositiveIntegerField(default=5, help_text="Bonus for first annotation of the day")
    points_speed_bonus = models.PositiveIntegerField(default=3, help_text="Bonus for fast completion")
    points_quality_bonus = models.PositiveIntegerField(default=2, help_text="Bonus for high confidence")
    points_streak_per_day = models.PositiveIntegerField(default=1, help_text="Streak bonus per consecutive day")
    points_streak_max = models.PositiveIntegerField(default=7, help_text="Maximum streak bonus")
    points_level_up = models.PositiveIntegerField(default=10, help_text="Bonus points on level up")

    # Thresholds
    speed_threshold_ner = models.PositiveIntegerField(default=180, help_text="Seconds under which NER earns speed bonus")
    speed_threshold_mcn = models.PositiveIntegerField(default=300, help_text="Seconds under which MCN earns speed bonus")
    quality_threshold = models.FloatField(default=0.9, help_text="Confidence above which quality bonus is earned (0-1)")

    # Feature toggles
    streak_enabled = models.BooleanField(default=True, help_text="Enable streak tracking and bonus")
    speed_bonus_enabled = models.BooleanField(default=True, help_text="Enable speed bonus")
    quality_bonus_enabled = models.BooleanField(default=True, help_text="Enable quality bonus")
    first_of_day_enabled = models.BooleanField(default=True, help_text="Enable first-of-day bonus")
    leaderboard_enabled = models.BooleanField(default=True, help_text="Enable leaderboard")

    # Level definitions as JSON: [[level, title, threshold], ...]
    level_definitions = models.JSONField(
        default=list,
        blank=True,
        help_text="Level definitions as [[level, title, threshold], ...]"
    )

    # Weekly reset
    weekly_reset_day = models.PositiveIntegerField(
        default=0, choices=WEEKLY_RESET_DAY_CHOICES,
        help_text="Day of week for weekly points reset (0=Monday)"
    )

    # Metadata
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='gamification_config_updates'
    )

    class Meta:
        verbose_name = 'Gamification Configuration'
        verbose_name_plural = 'Gamification Configuration'

    def __str__(self):
        return "Gamification Configuration"

    def save(self, *args, **kwargs):
        self.pk = 1  # Enforce singleton
        if not self.level_definitions:
            self.level_definitions = self.DEFAULT_LEVELS
        super().save(*args, **kwargs)
        self.invalidate_cache()

    def delete(self, *args, **kwargs):
        pass  # Prevent deletion

    @classmethod
    def get_config(cls):
        """Get the singleton config, with caching."""
        config = cache.get(cls.CACHE_KEY)
        if config is None:
            config, created = cls.objects.get_or_create(
                pk=1,
                defaults={'level_definitions': cls.DEFAULT_LEVELS}
            )
            cache.set(cls.CACHE_KEY, config, cls.CACHE_TTL)
        return config

    @classmethod
    def invalidate_cache(cls):
        cache.delete(cls.CACHE_KEY)

    def get_points_dict(self):
        """Return a dict matching the old POINTS format."""
        return {
            'ner_complete': self.points_ner_complete,
            'mcn_complete': self.points_mcn_complete,
            'first_of_day': self.points_first_of_day,
            'speed_bonus': self.points_speed_bonus,
            'quality_bonus': self.points_quality_bonus,
            'streak_bonus_per_day': self.points_streak_per_day,
            'streak_bonus_max': self.points_streak_max,
            'level_up': self.points_level_up,
        }

    def get_level_thresholds(self):
        """Return level definitions as list of tuples: [(level, title, threshold), ...]"""
        levels = self.level_definitions or self.DEFAULT_LEVELS
        return [(int(l[0]), str(l[1]), int(l[2])) for l in levels]


class AnnotatorProfile(models.Model):
    """
    Extended profile for gamification data.
    Tracks points, levels, streaks, and activity.
    """
    
    # Level definitions with point thresholds
    LEVEL_THRESHOLDS = [
        (1, 'Novice Annotator', 0),
        (2, 'Junior Annotator', 50),
        (3, 'Annotator', 150),
        (4, 'Senior Annotator', 400),
        (5, 'Expert Annotator', 800),
        (6, 'Master Annotator', 1500),
        (7, 'Grand Master', 3000),
        (8, 'Annotation Legend', 5000),
    ]
    
    user = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        related_name='gamification_profile'
    )
    
    # Points tracking
    total_points = models.PositiveIntegerField(default=0, help_text="All-time total points")
    weekly_points = models.PositiveIntegerField(default=0, help_text="Points this week (resets Monday)")
    weekly_reset_date = models.DateField(null=True, blank=True, help_text="Date of last weekly reset")
    
    # Level
    level = models.PositiveIntegerField(default=1, help_text="Current level (1-8)")
    
    # Streaks
    current_streak = models.PositiveIntegerField(default=0, help_text="Current consecutive days")
    longest_streak = models.PositiveIntegerField(default=0, help_text="Best streak ever")
    last_annotation_date = models.DateField(null=True, blank=True, help_text="Date of last annotation")
    
    # Statistics
    total_annotations = models.PositiveIntegerField(default=0, help_text="Total annotations completed")
    total_entities = models.PositiveIntegerField(default=0, help_text="Total entities annotated")
    ner_annotations = models.PositiveIntegerField(default=0, help_text="NER annotations completed")
    mcn_annotations = models.PositiveIntegerField(default=0, help_text="MCN annotations completed")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-total_points']
        indexes = [
            models.Index(fields=['-total_points']),
            models.Index(fields=['-weekly_points']),
            models.Index(fields=['level']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - Level {self.level} ({self.total_points} pts)"
    
    def _get_level_thresholds(self):
        """Get level thresholds from config, with fallback to class constant."""
        try:
            config = GamificationConfig.get_config()
            return config.get_level_thresholds()
        except Exception:
            return self.LEVEL_THRESHOLDS

    def get_level_title(self):
        """Get the title for the current level"""
        for lvl, title, _ in self._get_level_thresholds():
            if lvl == self.level:
                return title
        return "Annotator"

    def get_points_for_next_level(self):
        """Get points needed for next level"""
        for lvl, _, threshold in self._get_level_thresholds():
            if lvl == self.level + 1:
                return threshold
        return None  # Max level reached

    def get_level_progress_percent(self):
        """Get progress percentage to next level"""
        thresholds = self._get_level_thresholds()
        max_level = max(lvl for lvl, _, _ in thresholds) if thresholds else 8

        if self.level >= max_level:
            return 100

        current_threshold = 0
        next_threshold = 50  # Default

        for lvl, _, threshold in thresholds:
            if lvl == self.level:
                current_threshold = threshold
            elif lvl == self.level + 1:
                next_threshold = threshold
                break

        points_in_level = self.total_points - current_threshold
        points_needed = next_threshold - current_threshold

        if points_needed <= 0:
            return 100

        return min(100, int((points_in_level / points_needed) * 100))

    def check_weekly_reset(self):
        """Reset weekly points based on configured reset day."""
        today = timezone.now().date()

        try:
            config = GamificationConfig.get_config()
            reset_day = config.weekly_reset_day
        except Exception:
            reset_day = 0  # Monday

        days_since_reset = (today.weekday() - reset_day) % 7
        current_reset_date = today - timezone.timedelta(days=days_since_reset)

        if self.weekly_reset_date is None or self.weekly_reset_date < current_reset_date:
            self.weekly_points = 0
            self.weekly_reset_date = current_reset_date
            self.save(update_fields=['weekly_points', 'weekly_reset_date'])
    
    @classmethod
    def get_or_create_profile(cls, user):
        """Get or create a gamification profile for a user"""
        profile, created = cls.objects.get_or_create(user=user)
        if not created:
            profile.check_weekly_reset()
        return profile


class Achievement(models.Model):
    """
    Definition of achievements/badges that users can earn.
    """
    
    DIFFICULTY_CHOICES = [
        ('bronze', 'Bronze'),
        ('silver', 'Silver'),
        ('gold', 'Gold'),
        ('platinum', 'Platinum'),
    ]
    
    code = models.CharField(max_length=50, unique=True, help_text="Unique identifier for the achievement")
    name = models.CharField(max_length=100, help_text="Display name")
    description = models.TextField(help_text="How to earn this achievement")
    icon = models.CharField(max_length=10, default='🏅', help_text="Emoji icon")
    
    # Rewards
    points_reward = models.PositiveIntegerField(default=0, help_text="Bonus points when earned")
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='bronze')
    
    # Requirements (for automatic checking)
    required_annotations = models.PositiveIntegerField(null=True, blank=True)
    required_entities = models.PositiveIntegerField(null=True, blank=True)
    required_streak = models.PositiveIntegerField(null=True, blank=True)
    required_ner_count = models.PositiveIntegerField(null=True, blank=True)
    required_mcn_count = models.PositiveIntegerField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_secret = models.BooleanField(default=False, help_text="Hidden until earned")
    
    # Order
    display_order = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['display_order', 'difficulty', 'name']
        indexes = [
            models.Index(fields=['code']),
            models.Index(fields=['is_active']),
        ]
    
    def __str__(self):
        return f"{self.icon} {self.name}"
    
    def get_difficulty_color(self):
        """Get CSS color class for difficulty"""
        colors = {
            'bronze': '#cd7f32',
            'silver': '#c0c0c0',
            'gold': '#ffd700',
            'platinum': '#e5e4e2',
        }
        return colors.get(self.difficulty, '#6c757d')


class UserAchievement(models.Model):
    """
    Tracks which achievements users have earned.
    """
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='earned_achievements'
    )
    achievement = models.ForeignKey(
        Achievement, 
        on_delete=models.CASCADE, 
        related_name='earners'
    )
    
    # When earned
    earned_at = models.DateTimeField(auto_now_add=True)
    
    # Optional: context about how it was earned
    context = models.JSONField(default=dict, blank=True, help_text="Additional context about earning")
    
    # Has user seen the notification?
    is_notified = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-earned_at']
        unique_together = ['user', 'achievement']
        indexes = [
            models.Index(fields=['user', 'earned_at']),
            models.Index(fields=['is_notified']),
        ]
    
    def __str__(self):
        return f"{self.user.username} earned {self.achievement.name}"


class PointTransaction(models.Model):
    """
    Log of all point transactions for audit and history.
    """
    
    ACTION_TYPES = [
        ('ner_complete', 'NER Annotation Completed'),
        ('mcn_complete', 'MCN Annotation Completed'),
        ('streak_bonus', 'Streak Bonus'),
        ('speed_bonus', 'Speed Bonus'),
        ('quality_bonus', 'Quality Bonus'),
        ('first_of_day', 'First Annotation of Day'),
        ('achievement', 'Achievement Bonus'),
        ('level_up', 'Level Up Bonus'),
        ('manual_adjustment', 'Manual Adjustment'),
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='point_transactions'
    )
    
    action_type = models.CharField(max_length=30, choices=ACTION_TYPES)
    points = models.IntegerField(help_text="Points added (positive) or removed (negative)")
    description = models.CharField(max_length=255, blank=True)
    
    # Reference to related objects
    annotation_id = models.PositiveIntegerField(null=True, blank=True)
    achievement_id = models.PositiveIntegerField(null=True, blank=True)
    
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['action_type']),
        ]
    
    def __str__(self):
        sign = '+' if self.points > 0 else ''
        return f"{self.user.username}: {sign}{self.points} pts ({self.get_action_type_display()})"
