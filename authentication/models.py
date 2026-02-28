from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Custom User model with role-based authentication"""

    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('annotator', 'Annotator'),
        ('expert', 'Expert Annotator'),
    ]

    ANNOTATION_MODE_CHOICES = [
        ('manual_only', 'Manual Annotation Only'),
        ('llm_assisted_only', 'LLM-Assisted Annotation Only'),
        ('both', 'Both Manual and LLM-Assisted'),
    ]

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='annotator',
        help_text='User role determines access permissions'
    )

    annotation_mode = models.CharField(
        max_length=20,
        choices=ANNOTATION_MODE_CHOICES,
        default='manual_only',
        help_text='Determines which annotation methods this annotator can use'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_activity = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'auth_user'

    def is_admin(self):
        """Check if user has admin role"""
        return self.role == 'admin'

    def is_annotator(self):
        """Check if user has annotator role"""
        return self.role == 'annotator'

    def is_expert(self):
        """Check if user has expert annotator role"""
        return self.role == 'expert'

    def can_do_manual_annotation(self):
        """Check if user can perform manual annotation"""
        return self.annotation_mode in ['manual_only', 'both']

    def can_do_llm_assisted_annotation(self):
        """Check if user can perform LLM-assisted annotation"""
        return self.annotation_mode in ['llm_assisted_only', 'both']

    def get_required_annotation_types(self):
        """Get list of annotation types this user must complete"""
        if self.annotation_mode == 'manual_only':
            return ['manual']
        elif self.annotation_mode == 'llm_assisted_only':
            return ['llm_assisted']
        else:  # both
            return ['manual', 'llm_assisted']  # Either one completes the task

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"


class UserSession(models.Model):
    """Track user sessions for clear separation between annotators"""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    session_key = models.CharField(max_length=40, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-last_activity']

    def __str__(self):
        return f"{self.user.username} - {self.session_key[:8]}..."


# Import gamification models so Django can discover them
from .gamification_models import AnnotatorProfile, Achievement, UserAchievement, PointTransaction

