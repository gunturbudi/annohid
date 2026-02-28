"""
Models for centralized API key management
Store and manage LLM provider API keys securely
"""

from django.db import models
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.utils import timezone
import os

User = get_user_model()


class LLMProvider(models.Model):
    """
    Model for LLM provider definitions
    Defines available providers and their configuration
    """

    PROVIDER_CHOICES = [
        ('openrouter', 'OpenRouter (100+ models, recommended)'),
        ('gemini', 'Google Gemini'),
        ('openai', 'OpenAI'),
        ('anthropic', 'Anthropic Claude'),
    ]

    name = models.CharField(
        max_length=50,
        unique=True,
        choices=PROVIDER_CHOICES,
        help_text="LLM provider name"
    )
    display_name = models.CharField(
        max_length=100,
        help_text="Human-readable provider name"
    )
    description = models.TextField(
        blank=True,
        help_text="Provider description and features"
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this provider is active and available"
    )
    priority = models.IntegerField(
        default=0,
        help_text="Priority order (higher = preferred), recommend OpenRouter highest"
    )

    # Configuration
    env_var_name = models.CharField(
        max_length=100,
        help_text="Environment variable name (e.g., OPENROUTER_API_KEY)"
    )
    base_url = models.URLField(
        blank=True,
        help_text="API base URL (if applicable)"
    )
    documentation_url = models.URLField(
        blank=True,
        help_text="Link to provider documentation"
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-priority', 'name']
        verbose_name = "LLM Provider"
        verbose_name_plural = "LLM Providers"

    def __str__(self):
        return f"{self.display_name} ({'Active' if self.is_active else 'Inactive'})"

    def has_api_key(self):
        """Check if API key is configured (in env or database)"""
        # Check environment variable
        if os.getenv(self.env_var_name):
            return True

        # Check database
        return APIKey.objects.filter(provider=self, is_active=True).exists()

    def get_api_key(self):
        """Get API key from environment or database (env takes precedence)"""
        # First check environment variable
        env_key = os.getenv(self.env_var_name)
        if env_key:
            return env_key

        # Then check database
        try:
            api_key_obj = APIKey.objects.filter(
                provider=self,
                is_active=True
            ).order_by('-updated_at').first()

            if api_key_obj:
                return api_key_obj.get_decrypted_key()
        except APIKey.DoesNotExist:
            pass

        return None


class APIKey(models.Model):
    """
    Model for storing API keys
    Supports both environment variables and database storage
    """

    provider = models.ForeignKey(
        LLMProvider,
        on_delete=models.CASCADE,
        related_name='api_keys',
        help_text="LLM provider this key belongs to"
    )

    # Key storage (encrypted in production)
    key_name = models.CharField(
        max_length=100,
        help_text="Descriptive name for this key (e.g., 'Production Key', 'Test Key')"
    )
    api_key = models.CharField(
        max_length=500,
        help_text="API key value (stored securely)"
    )
    key_prefix = models.CharField(
        max_length=20,
        blank=True,
        help_text="First few characters of key for identification"
    )

    # Status
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this key is currently active"
    )
    is_from_env = models.BooleanField(
        default=False,
        help_text="Whether this key comes from environment variable"
    )

    # Testing
    last_tested = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last time this key was tested"
    )
    test_status = models.CharField(
        max_length=20,
        blank=True,
        choices=[
            ('untested', 'Not Tested'),
            ('valid', 'Valid'),
            ('invalid', 'Invalid'),
            ('expired', 'Expired'),
            ('rate_limited', 'Rate Limited'),
        ],
        default='untested',
        help_text="Result of last test"
    )
    test_message = models.TextField(
        blank=True,
        help_text="Details from last test"
    )

    # Usage tracking
    usage_count = models.IntegerField(
        default=0,
        help_text="Number of times this key has been used"
    )
    last_used = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last time this key was used"
    )

    # Metadata
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='created_api_keys',
        help_text="Admin who added this key"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Notes
    notes = models.TextField(
        blank=True,
        help_text="Internal notes about this key"
    )

    class Meta:
        ordering = ['-is_active', '-updated_at']
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"
        indexes = [
            models.Index(fields=['provider', 'is_active']),
        ]

    def __str__(self):
        status = "Active" if self.is_active else "Inactive"
        source = " (Env)" if self.is_from_env else ""
        return f"{self.provider.display_name} - {self.key_name} ({status}){source}"

    def save(self, *args, **kwargs):
        # Auto-generate key prefix for identification
        if self.api_key and not self.key_prefix:
            self.key_prefix = self.api_key[:10] + "..."

        super().save(*args, **kwargs)

    def get_decrypted_key(self):
        """Get the decrypted API key"""
        # In production, implement proper encryption/decryption
        # For now, return as-is (but marked as TODO for production)
        return self.api_key

    def mask_key(self):
        """Return masked version of key for display"""
        if not self.api_key:
            return "No key set"

        if len(self.api_key) <= 10:
            return "***"

        return f"{self.api_key[:10]}...{self.api_key[-4:]}"

    def test_key(self):
        """Test if this API key is valid"""
        from .llm_providers import get_ner_provider

        self.last_tested = timezone.now()

        try:
            # Try to initialize provider with this key
            # Use a simple test based on provider type
            test_model = self._get_test_model()

            provider = get_ner_provider(test_model, api_key=self.get_decrypted_key())

            if provider and provider.is_available():
                # Try a simple annotation
                test_text = "Patient has fever"
                result = provider.annotate_text(test_text)

                if result.success:
                    self.test_status = 'valid'
                    self.test_message = f"Successfully tested at {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    self.usage_count += 1
                    self.last_used = timezone.now()
                else:
                    self.test_status = 'invalid'
                    self.test_message = result.error_message or "Unknown error"
            else:
                self.test_status = 'invalid'
                self.test_message = "Failed to initialize provider"

        except Exception as e:
            self.test_status = 'invalid'
            self.test_message = str(e)

        self.save()
        return self.test_status == 'valid'

    def _get_test_model(self):
        """Get appropriate test model for this provider"""
        test_models = {
            'openrouter': 'openai/gpt-4o-mini',
            'together': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
            'gemini': 'gemini-2.5-flash',
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-3-haiku-20240307',
        }
        return test_models.get(self.provider.name, 'gemini-2.5-flash')

    def record_usage(self):
        """Record that this key was used"""
        self.usage_count += 1
        self.last_used = timezone.now()
        self.save(update_fields=['usage_count', 'last_used'])


class LLMConfiguration(models.Model):
    """
    Global configuration for LLM settings
    Singleton model for system-wide settings
    """

    # Default provider preference
    preferred_provider = models.ForeignKey(
        LLMProvider,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='preferred_config',
        help_text="Preferred provider to use by default (recommend OpenRouter)"
    )

    # Fallback behavior
    enable_fallback = models.BooleanField(
        default=True,
        help_text="Try other providers if preferred fails"
    )

    # Default models for multi-LLM
    default_models = models.JSONField(
        default=list,
        blank=True,
        help_text="Default models to use in multi-LLM comparison"
    )

    # Cost settings
    enable_cost_tracking = models.BooleanField(
        default=True,
        help_text="Track API usage costs"
    )
    monthly_budget_usd = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Monthly API budget in USD"
    )

    # Usage limits
    max_requests_per_hour = models.IntegerField(
        null=True,
        blank=True,
        help_text="Max requests per hour (rate limiting)"
    )

    # Metadata
    updated_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='llm_configs',
        help_text="Last admin who updated configuration"
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "LLM Configuration"
        verbose_name_plural = "LLM Configuration"

    def __str__(self):
        return f"LLM Configuration (Updated: {self.updated_at.strftime('%Y-%m-%d')})"

    @classmethod
    def get_config(cls):
        """Get or create singleton configuration"""
        config, created = cls.objects.get_or_create(id=1)
        return config

    def save(self, *args, **kwargs):
        # Ensure singleton
        self.id = 1
        super().save(*args, **kwargs)

    def get_active_providers(self):
        """Get all active providers with API keys"""
        return LLMProvider.objects.filter(
            is_active=True
        ).filter(
            models.Q(api_keys__is_active=True) |
            models.Q(env_var_name__isnull=False)
        ).distinct()
