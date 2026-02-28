"""
Django Admin configuration for API Key management
"""

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone
from .api_key_models import LLMProvider, APIKey, LLMConfiguration


@admin.register(LLMProvider)
class LLMProviderAdmin(admin.ModelAdmin):
    """Admin interface for LLM Providers"""

    list_display = [
        'display_name',
        'name',
        'is_active',
        'priority',
        'key_status',
        'env_var_name',
        'documentation_link',
    ]
    list_filter = ['is_active', 'name']
    search_fields = ['name', 'display_name', 'description']
    ordering = ['-priority', 'name']

    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'display_name', 'description', 'is_active', 'priority')
        }),
        ('Configuration', {
            'fields': ('env_var_name', 'base_url', 'documentation_url')
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ['created_at', 'updated_at']

    def key_status(self, obj):
        """Show if API key is configured"""
        if obj.has_api_key():
            return format_html(
                '<span style="color: green;">✓ Configured</span>'
            )
        return format_html(
            '<span style="color: red;">✗ Not Configured</span>'
        )
    key_status.short_description = 'API Key Status'

    def documentation_link(self, obj):
        """Render documentation URL as clickable link"""
        if obj.documentation_url:
            return format_html(
                '<a href="{}" target="_blank">View Docs</a>',
                obj.documentation_url
            )
        return '-'
    documentation_link.short_description = 'Documentation'


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    """Admin interface for API Keys"""

    list_display = [
        'key_name',
        'provider',
        'is_active',
        'is_from_env',
        'key_display',
        'test_status_display',
        'usage_info',
        'last_used',
        'test_action',
    ]
    list_filter = ['provider', 'is_active', 'test_status', 'is_from_env']
    search_fields = ['key_name', 'notes', 'provider__name']
    ordering = ['-is_active', '-updated_at']

    fieldsets = (
        ('Key Information', {
            'fields': ('provider', 'key_name', 'api_key', 'is_active')
        }),
        ('Testing', {
            'fields': ('test_status', 'test_message', 'last_tested'),
            'classes': ('collapse',)
        }),
        ('Usage Tracking', {
            'fields': ('usage_count', 'last_used'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at', 'notes'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = [
        'key_prefix',
        'test_status',
        'test_message',
        'last_tested',
        'usage_count',
        'last_used',
        'created_at',
        'updated_at',
    ]

    def save_model(self, request, obj, form, change):
        """Set created_by on save"""
        if not change:  # New object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)

    def key_display(self, obj):
        """Show masked key"""
        return obj.mask_key()
    key_display.short_description = 'API Key'

    def test_status_display(self, obj):
        """Show test status with color coding"""
        colors = {
            'valid': 'green',
            'invalid': 'red',
            'untested': 'gray',
            'expired': 'orange',
            'rate_limited': 'orange',
        }
        color = colors.get(obj.test_status, 'gray')

        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color,
            obj.get_test_status_display()
        )
    test_status_display.short_description = 'Test Status'

    def usage_info(self, obj):
        """Show usage count"""
        return f"{obj.usage_count} uses"
    usage_info.short_description = 'Usage'

    def test_action(self, obj):
        """Render test button"""
        return format_html(
            '<a class="button" href="#" onclick="testKey({}); return false;">Test Key</a>',
            obj.pk
        )
    test_action.short_description = 'Actions'

    class Media:
        js = ('admin/js/api_key_test.js',)


@admin.register(LLMConfiguration)
class LLMConfigurationAdmin(admin.ModelAdmin):
    """Admin interface for LLM Configuration"""

    fieldsets = (
        ('Provider Settings', {
            'fields': ('preferred_provider', 'enable_fallback')
        }),
        ('Default Models', {
            'fields': ('default_models',),
            'description': 'Default models for multi-LLM comparison (JSON array)'
        }),
        ('Cost Management', {
            'fields': ('enable_cost_tracking', 'monthly_budget_usd')
        }),
        ('Rate Limiting', {
            'fields': ('max_requests_per_hour',)
        }),
        ('Metadata', {
            'fields': ('updated_by', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ['updated_by', 'updated_at']

    def has_add_permission(self, request):
        """Only allow one configuration instance"""
        return not LLMConfiguration.objects.exists()

    def has_delete_permission(self, request, obj=None):
        """Don't allow deletion of singleton"""
        return False

    def save_model(self, request, obj, form, change):
        """Set updated_by on save"""
        obj.updated_by = request.user
        super().save_model(request, obj, form, change)
