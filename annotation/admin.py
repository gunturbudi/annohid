from django.contrib import admin
from .models import EntityDefinition, TextDocument, TextImportBatch, AnnotationTask, AnnotationResult, AnnotationStatistics


@admin.register(EntityDefinition)
class EntityDefinitionAdmin(admin.ModelAdmin):
    list_display = ['entity_type', 'get_definition_preview', 'is_active', 'created_by', 'updated_at']
    list_filter = ['entity_type', 'is_active', 'created_by', 'created_at']
    search_fields = ['entity_type', 'definition', 'examples', 'guidelines']
    readonly_fields = ['created_at', 'updated_at']

    fieldsets = (
        ('Entity Information', {
            'fields': ('entity_type', 'is_active')
        }),
        ('Definition & Guidelines', {
            'fields': ('definition', 'examples', 'guidelines'),
            'description': 'Provide clear definitions and examples that annotators can reference'
        }),
        ('Metadata', {
            'fields': ('created_by', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def save_model(self, request, obj, form, change):
        if not change:  # If creating new object
            obj.created_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(TextDocument)
class TextDocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'status', 'uploaded_by', 'word_count', 'created_at']
    list_filter = ['status', 'uploaded_by', 'created_at']
    search_fields = ['title', 'content']
    filter_horizontal = ['assigned_to']
    readonly_fields = ['word_count', 'character_count', 'created_at', 'updated_at']


@admin.register(TextImportBatch)
class TextImportBatchAdmin(admin.ModelAdmin):
    list_display = ['filename', 'imported_by', 'successful_imports', 'failed_imports', 'total_lines', 'imported_at']
    list_filter = ['imported_by', 'imported_at', 'auto_assign']
    search_fields = ['filename', 'notes']
    readonly_fields = ['imported_at']


@admin.register(AnnotationTask)
class AnnotationTaskAdmin(admin.ModelAdmin):
    list_display = ['document', 'annotator', 'task_type', 'is_started', 'is_completed', 'assigned_at', 'get_duration_display']
    list_filter = ['task_type', 'is_started', 'is_completed', 'annotator', 'assigned_at']
    search_fields = ['document__title', 'document__content', 'annotator__username']
    readonly_fields = ['assigned_at', 'started_at', 'completed_at', 'actual_duration']

    def get_duration_display(self, obj):
        return obj.get_duration_display()
    get_duration_display.short_description = 'Duration'


@admin.register(AnnotationResult)
class AnnotationResultAdmin(admin.ModelAdmin):
    list_display = ['document', 'annotator', 'annotation_type', 'task_type', 'total_entities', 'review_status', 'created_at']
    list_filter = ['annotation_type', 'task_type', 'review_status', 'annotator', 'created_at', 'model_used']
    search_fields = ['document__title', 'document__content', 'annotator__username', 'admin_notes']
    readonly_fields = ['created_at', 'total_entities', 'confidence_score', 'processing_time']


@admin.register(AnnotationStatistics)
class AnnotationStatisticsAdmin(admin.ModelAdmin):
    list_display = [
        'annotator', 'task_type', 'period_type', 'total_tasks_completed',
        'average_annotation_time_minutes', 'acceptance_rate', 'calculated_at'
    ]
    list_filter = [
        'task_type', 'period_type', 'annotator', 'calculated_at'
    ]
    search_fields = ['annotator__username']
    readonly_fields = [
        'calculated_at', 'created_at', 'acceptance_rate', 'modification_rate',
        'rejection_rate', 'average_edit_distance', 'average_token_changes'
    ]

    fieldsets = (
        ('Basic Information', {
            'fields': ('annotator', 'task_type', 'period_type', 'period_start', 'period_end')
        }),
        ('Timing Statistics', {
            'fields': (
                'total_tasks_completed', 'total_annotation_time_minutes',
                'average_annotation_time_minutes', 'min_annotation_time_minutes',
                'max_annotation_time_minutes'
            )
        }),
        ('LLM Prediction Statistics', {
            'fields': (
                'total_llm_predictions', 'accepted_predictions', 'modified_predictions',
                'rejected_predictions', 'acceptance_rate', 'modification_rate', 'rejection_rate'
            )
        }),
        ('Edit Distance Statistics', {
            'fields': (
                'total_edit_distance', 'average_edit_distance',
                'total_token_changes', 'average_token_changes'
            )
        }),
        ('Metadata', {
            'fields': ('calculated_at', 'created_at')
        }),
    )

    def has_add_permission(self, request):
        # Prevent manual addition - statistics should be calculated automatically
        return False
