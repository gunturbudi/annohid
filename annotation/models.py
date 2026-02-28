from django.db import models
from django.utils import timezone
from django.db.models import Q, Avg, Count, Sum
from authentication.models import User
import difflib


class EntityDefinition(models.Model):
    """Model for storing entity type definitions that can be shown as tooltips"""

    ENTITY_TYPES = [
        ('SYMPTOM', 'Symptom'),
        ('DISEASE', 'Disease'),
        ('BODY_PART', 'Body Part'),
        ('MEDICATION', 'Medication'),
        ('PROCEDURE', 'Procedure'),
    ]

    # Default colors for each entity type
    DEFAULT_COLORS = {
        'SYMPTOM': '#ff6b6b',
        'DISEASE': '#4ecdc4',
        'BODY_PART': '#45b7d1',
        'MEDICATION': '#96ceb4',
        'PROCEDURE': '#ffeaa7'
    }

    entity_type = models.CharField(max_length=20, choices=ENTITY_TYPES, unique=True,
                                 help_text="Type of medical entity")
    definition = models.TextField(help_text="Clear definition of what constitutes this entity type")
    examples = models.TextField(blank=True, help_text="Examples of this entity type (optional)")
    guidelines = models.TextField(blank=True, help_text="Additional annotation guidelines for this entity type")
    color = models.CharField(max_length=7, default='#6c757d',
                           help_text="Hex color code for highlighting this entity type (e.g., #ff6b6b)")

    # Metadata
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='created_entity_definitions', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Status
    is_active = models.BooleanField(default=True, help_text="Whether this definition is currently active")

    class Meta:
        ordering = ['entity_type']
        indexes = [
            models.Index(fields=['entity_type']),
            models.Index(fields=['is_active']),
        ]

    def __str__(self):
        return f"{self.get_entity_type_display()} Definition"

    def get_definition_preview(self, max_length=100):
        """Get a preview of the definition"""
        if len(self.definition) <= max_length:
            return self.definition
        return self.definition[:max_length] + "..."

    @classmethod
    def get_entity_colors(cls):
        """Get all entity colors as a dictionary"""
        # Get colors from database
        definitions = cls.objects.filter(is_active=True)
        colors = {}

        for definition in definitions:
            colors[definition.entity_type] = definition.color

        # Fill in defaults for any missing types
        for entity_type, default_color in cls.DEFAULT_COLORS.items():
            if entity_type not in colors:
                colors[entity_type] = default_color

        return colors


class Guideline(models.Model):
    """Model for storing annotation guidelines that admins can upload"""

    title = models.CharField(max_length=255, help_text="Title of the guideline")
    description = models.TextField(blank=True, help_text="Optional description of the guideline")
    content = models.TextField(help_text="Guideline content")

    # Metadata
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_guidelines')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Status
    is_active = models.BooleanField(default=True, help_text="Whether this guideline is currently active")

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_active']),
            models.Index(fields=['uploaded_by']),
        ]

    def __str__(self):
        return self.title

    def get_content_preview(self, max_length=200):
        """Get a preview of the content"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


class TextDocument(models.Model):
    """Model for storing medical texts to be annotated"""

    STATUS_CHOICES = [
        ('pending', 'Pending Assignment'),
        ('assigned', 'Assigned'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('reviewed', 'Reviewed'),
    ]

    title = models.CharField(max_length=255, blank=True, help_text="Optional title for the document")
    content = models.TextField(help_text="Medical text content to be annotated")
    source_file = models.CharField(max_length=255, blank=True, help_text="Original filename if imported from file")
    line_number = models.PositiveIntegerField(null=True, blank=True, help_text="Line number in source file")

    # Status and assignment
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    assigned_to = models.ManyToManyField(User, blank=True, related_name='assigned_documents',
                                        limit_choices_to={'role': 'annotator', 'is_active': True})

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_documents')

    # Content metadata
    word_count = models.PositiveIntegerField(default=0)
    character_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['uploaded_by']),
        ]

    def save(self, *args, **kwargs):
        # Auto-calculate word and character counts
        if self.content:
            self.word_count = len(self.content.split())
            self.character_count = len(self.content)
        super().save(*args, **kwargs)

    def __str__(self):
        title = self.title or f"Document {self.id}"
        return f"{title} ({self.get_status_display()})"

    def get_content_preview(self, max_length=100):
        """Get a preview of the content"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."

    def get_assigned_annotators(self):
        """Get list of assigned annotators"""
        return self.assigned_to.all()

    def get_assigned_annotators_names(self):
        """Get comma-separated list of assigned annotator names"""
        annotators = self.get_assigned_annotators()
        if annotators.exists():
            return ", ".join([user.username for user in annotators])
        return "Unassigned"

    def is_assigned_to_user(self, user):
        """Check if document is assigned to specific user"""
        return self.assigned_to.filter(id=user.id).exists()

    def get_assignment_count(self):
        """Get number of annotators assigned to this document"""
        return self.assigned_to.count()

    def has_ner_annotation(self):
        """Check if document has any NER annotation"""
        return self.annotation_results.filter(task_type='ner').exists()

    def has_mcn_annotation(self):
        """Check if document has any MCN annotation"""
        return self.annotation_results.filter(task_type='mcn').exists()

    def get_latest_ner_annotation(self):
        """Get the most recent NER annotation result"""
        try:
            return self.annotation_results.filter(task_type='ner').latest('created_at')
        except:
            return None

    def get_latest_mcn_annotation(self):
        """Get the most recent MCN annotation result"""
        try:
            return self.annotation_results.filter(task_type='mcn').latest('created_at')
        except:
            return None


class TextImportBatch(models.Model):
    """Model for tracking batch imports of text data"""

    filename = models.CharField(max_length=255)
    imported_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='import_batches')
    imported_at = models.DateTimeField(auto_now_add=True)

    # Statistics
    total_lines = models.PositiveIntegerField(default=0)
    successful_imports = models.PositiveIntegerField(default=0)
    failed_imports = models.PositiveIntegerField(default=0)

    # Import settings
    skip_empty_lines = models.BooleanField(default=True)
    auto_assign = models.BooleanField(default=False)

    notes = models.TextField(blank=True, help_text="Import notes or error messages")

    class Meta:
        ordering = ['-imported_at']

    def __str__(self):
        return f"Import: {self.filename} ({self.successful_imports}/{self.total_lines} successful)"


class AnnotationTask(models.Model):
    """Model for tracking annotation tasks and progress"""

    TASK_TYPES = [
        ('ner', 'Named Entity Recognition'),
        ('mcn', 'Medical Concept Normalization'),
    ]

    document = models.ForeignKey(TextDocument, on_delete=models.CASCADE, related_name='annotation_tasks')
    annotator = models.ForeignKey(User, on_delete=models.CASCADE, related_name='annotation_tasks')
    task_type = models.CharField(max_length=10, choices=TASK_TYPES, default='ner')

    # Task metadata
    assigned_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Progress tracking
    is_started = models.BooleanField(default=False)
    is_completed = models.BooleanField(default=False)

    # Time tracking
    estimated_duration = models.DurationField(null=True, blank=True, help_text="Estimated time to complete")
    actual_duration = models.DurationField(null=True, blank=True, help_text="Actual time taken")

    notes = models.TextField(blank=True, help_text="Annotator notes")

    class Meta:
        ordering = ['-assigned_at']
        unique_together = ['document', 'annotator', 'task_type']

    def __str__(self):
        return f"{self.get_task_type_display()}: {self.document} -> {self.annotator.username}"

    def start_task(self):
        """Mark task as started"""
        if not self.is_started:
            self.is_started = True
            self.started_at = timezone.now()
            self.document.status = 'in_progress'
            self.document.save()
            self.save()

    def complete_task(self):
        """Mark task as completed and award gamification points"""
        if not self.is_completed:
            self.is_completed = True
            self.completed_at = timezone.now()
            if self.started_at:
                self.actual_duration = self.completed_at - self.started_at

            # Update document status only if all tasks are completed based on annotator's role
            all_tasks = AnnotationTask.objects.filter(
                document=self.document,
                annotator=self.annotator
            )
            # Check if all tasks are completed based on role-specific logic
            all_completed = all(task.is_completed_based_on_role() for task in all_tasks)
            if all_completed:
                self.document.status = 'completed'
                self.document.save()

            self.save()

            # Award gamification points
            self._award_gamification_points()

    def _award_gamification_points(self):
        """Award gamification points after task completion"""
        try:
            from authentication.gamification_service import GamificationService

            # Calculate duration in seconds
            duration_seconds = None
            if self.actual_duration:
                duration_seconds = self.actual_duration.total_seconds()

            # Get the latest annotation result for confidence score and entity count
            confidence_score = None
            annotation_result = None
            try:
                from annotation.models import AnnotationResult
                annotation_result = AnnotationResult.objects.filter(
                    document=self.document,
                    annotator=self.annotator,
                    task_type=self.task_type
                ).order_by('-created_at').first()

                if annotation_result and annotation_result.confidence_score:
                    confidence_score = annotation_result.confidence_score
            except Exception:
                pass

            # Award points
            result = GamificationService.award_annotation_points(
                user=self.annotator,
                task_type=self.task_type,
                annotation_result=annotation_result,
                confidence_score=confidence_score,
                duration_seconds=duration_seconds,
            )

            # Store the gamification result on the instance for views to access
            self._gamification_result = result
        except Exception as e:
            # Don't let gamification errors break the main workflow
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Gamification error for task {self.id}: {e}")
            self._gamification_result = None

    def get_duration_minutes(self):
        """Get task duration in minutes"""
        if self.actual_duration:
            return round(self.actual_duration.total_seconds() / 60, 2)
        elif self.is_started and self.started_at:
            # Calculate current duration for in-progress tasks
            current_duration = timezone.now() - self.started_at
            return round(current_duration.total_seconds() / 60, 2)
        return 0.0

    def get_duration_display(self):
        """Get human-readable duration"""
        minutes = self.get_duration_minutes()
        if minutes < 1:
            return f"{int(minutes * 60)} seconds"
        elif minutes < 60:
            return f"{minutes:.1f} minutes"
        else:
            hours = minutes / 60
            return f"{hours:.1f} hours"

    @classmethod
    def get_time_statistics(cls, annotator=None, task_type=None, date_from=None, date_to=None):
        """Get time statistics for tasks"""
        queryset = cls.objects.filter(is_completed=True)

        if annotator:
            queryset = queryset.filter(annotator=annotator)
        if task_type:
            queryset = queryset.filter(task_type=task_type)
        if date_from:
            queryset = queryset.filter(completed_at__gte=date_from)
        if date_to:
            queryset = queryset.filter(completed_at__lte=date_to)

        stats = {
            'total_tasks': queryset.count(),
            'total_time_minutes': 0,
            'average_time_minutes': 0,
            'min_time_minutes': None,
            'max_time_minutes': None,
        }

        completed_tasks = list(queryset.filter(actual_duration__isnull=False))

        if completed_tasks:
            durations = [task.get_duration_minutes() for task in completed_tasks]
            stats['total_time_minutes'] = sum(durations)
            stats['average_time_minutes'] = stats['total_time_minutes'] / len(durations)
            stats['min_time_minutes'] = min(durations)
            stats['max_time_minutes'] = max(durations)

        return stats

    def can_access_mcn(self):
        """Check if MCN task can be accessed (NER must be completed first)"""
        if self.task_type == 'ner':
            return True  # NER is always accessible

        # For MCN, check if NER is completed
        try:
            ner_task = AnnotationTask.objects.get(
                document=self.document,
                annotator=self.annotator,
                task_type='ner'
            )
            return ner_task.is_completed_based_on_role()
        except AnnotationTask.DoesNotExist:
            return False  # No NER task exists

    def is_completed_based_on_role(self):
        """Check if task is completed based on annotator's annotation mode"""
        if self.is_completed:
            return True

        # Get existing annotations for this task
        annotations = AnnotationResult.objects.filter(
            document=self.document,
            annotator=self.annotator,
            task_type=self.task_type,
            review_status='completed'
        )

        # Check completion based on annotator's role
        if self.annotator.annotation_mode == 'manual_only':
            # For manual-only, check for manual annotations
            if self.task_type == 'ner':
                return annotations.filter(annotation_type='manual').exists()
            else:  # mcn
                return annotations.filter(annotation_type='mcn_manual').exists()

        elif self.annotator.annotation_mode == 'llm_assisted_only':
            # For LLM-assisted-only, check for LLM-assisted annotations
            if self.task_type == 'ner':
                return annotations.filter(annotation_type='llm_assisted').exists()
            else:  # mcn
                return annotations.filter(annotation_type='mcn_llm_assisted').exists()

        else:  # both
            # For both, check if any annotation exists (current behavior)
            if self.task_type == 'ner':
                return annotations.filter(annotation_type__in=['manual', 'llm_assisted']).exists()
            else:  # mcn
                return annotations.filter(annotation_type__in=['mcn_manual', 'mcn_llm_assisted']).exists()

        return False

    @classmethod
    def get_ner_task(cls, document, annotator):
        """Get NER task for a document and annotator"""
        try:
            return cls.objects.get(document=document, annotator=annotator, task_type='ner')
        except cls.DoesNotExist:
            return None

    @classmethod
    def get_mcn_task(cls, document, annotator):
        """Get MCN task for a document and annotator"""
        try:
            return cls.objects.get(document=document, annotator=annotator, task_type='mcn')
        except cls.DoesNotExist:
            return None


class AnnotationResult(models.Model):
    """Model for storing annotation results - supports both manual and LLM-assisted modes"""

    ANNOTATION_TYPES = [
        ('manual', 'Manual Annotation'),
        ('llm_assisted', 'LLM-Assisted Annotation'),
        ('llm_only', 'LLM Only (Legacy)'),
        ('mcn_manual', 'MCN Manual'),
        ('mcn_llm_assisted', 'MCN LLM-Assisted'),
    ]

    TASK_TYPES = [
        ('ner', 'Named Entity Recognition'),
        ('mcn', 'Medical Concept Normalization'),
    ]

    document = models.ForeignKey(TextDocument, on_delete=models.CASCADE, related_name='annotation_results')
    annotator = models.ForeignKey(User, on_delete=models.CASCADE, related_name='annotation_results')

    # Task metadata
    task_type = models.CharField(max_length=10, choices=TASK_TYPES, default='ner')
    annotation_type = models.CharField(max_length=20, choices=ANNOTATION_TYPES, default='llm_only')
    model_used = models.CharField(max_length=50, blank=True, help_text="LLM model used for annotation")
    created_at = models.DateTimeField(auto_now_add=True)

    # Results data
    entities_json = models.JSONField(default=list, help_text="JSON array of extracted entities with source/status tracking")
    total_entities = models.PositiveIntegerField(default=0)
    processing_time = models.FloatField(null=True, blank=True, help_text="Processing time in seconds")

    # Agreement tracking for LLM-assisted mode
    decisions_json = models.JSONField(default=dict, help_text="Track approve/reject decisions for analysis")

    # Quality metrics
    confidence_score = models.FloatField(null=True, blank=True, help_text="Average confidence score")
    review_status = models.CharField(max_length=20, default='pending',
                                   choices=[('pending', 'Pending Review'),
                                          ('in_progress', 'In Progress'),
                                          ('completed', 'Completed'),
                                          ('approved', 'Approved'),
                                          ('rejected', 'Rejected')])

    # Admin notes
    admin_notes = models.TextField(blank=True, help_text="Admin review notes")

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['document', 'annotator']),
            models.Index(fields=['annotation_type']),
            models.Index(fields=['model_used']),
        ]

    def __str__(self):
        return f"{self.get_annotation_type_display()} by {self.annotator.username} for {self.document.title[:50]}"

    def get_entities_by_type(self) -> dict:
        """Get entities grouped by type"""
        entities_by_type = {}
        for entity in self.entities_json:
            entity_type = entity.get('label', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        return entities_by_type

    @staticmethod
    def _filter_subset_entities(entities):
        """Remove entities whose span is completely contained within a longer entity."""
        if not entities:
            return entities

        filtered = []
        for i, e in enumerate(entities):
            s1 = e.get('start_offset', e.get('start', 0))
            e1 = e.get('end_offset', e.get('end', 0))
            is_subset = False
            for j, other in enumerate(entities):
                if i == j:
                    continue
                s2 = other.get('start_offset', other.get('start', 0))
                e2 = other.get('end_offset', other.get('end', 0))
                if s2 <= s1 and e1 <= e2 and (e2 - s2) > (e1 - s1):
                    is_subset = True
                    break
            if not is_subset:
                filtered.append(e)
        return filtered

    def get_filtered_entities_by_type(self) -> dict:
        """Get entities grouped by type, with subset spans removed for display."""
        filtered = self._filter_subset_entities(self.entities_json or [])
        entities_by_type = {}
        for entity in filtered:
            entity_type = entity.get('label', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        return entities_by_type

    def get_entity_statistics(self, exclude_rejected=False) -> dict:
        """Get entity count statistics by type.
        If exclude_rejected=True, rejected LLM entities are not counted.
        """
        stats = {}
        for entity in self.entities_json:
            if exclude_rejected and entity.get('status') == 'rejected':
                continue
            entity_type = entity.get('label', 'UNKNOWN')
            stats[entity_type] = stats.get(entity_type, 0) + 1
        return stats

    def get_decision_statistics(self) -> dict:
        """Get statistics about approve/reject decisions"""
        stats = {
            'total': len(self.entities_json),
            'manual': 0,
            'llm_approved': 0,
            'llm_rejected': 0,
            'llm_pending': 0,
        }

        for entity in self.entities_json:
            source = entity.get('source', 'llm')
            status = entity.get('status', 'pending')

            if source == 'manual':
                stats['manual'] += 1
            elif status == 'approved':
                stats['llm_approved'] += 1
            elif status == 'rejected':
                stats['llm_rejected'] += 1
            else:
                stats['llm_pending'] += 1

        return stats

    def get_agreement_rate(self) -> float:
        """Calculate the agreement rate for LLM suggestions"""
        decision_stats = self.get_decision_statistics()
        total_llm = decision_stats['llm_approved'] + decision_stats['llm_rejected']

        if total_llm == 0:
            return 0.0

        return (decision_stats['llm_approved'] / total_llm) * 100

    def track_llm_decision(self, entity_index, decision, original_text=None, modified_text=None):
        """Track a decision made on an LLM prediction"""
        if entity_index < len(self.entities_json):
            entity = self.entities_json[entity_index]

            # Store original text if this is the first modification
            if 'original_text' not in entity and original_text:
                entity['original_text'] = original_text

            # Update status and track changes
            entity['status'] = decision
            entity['decision_timestamp'] = timezone.now().isoformat()

            if decision == 'modified' and modified_text:
                entity['modified_text'] = modified_text
                entity['text'] = modified_text  # Update the actual text

                # Calculate edit distance for this specific modification
                original = entity.get('original_text', entity.get('text', ''))
                if original:
                    entity['edit_distance'] = self.calculate_entity_edit_distance(original, modified_text)
                    entity['token_changes'] = self.calculate_entity_token_changes(original, modified_text)

            # Update decisions_json for tracking
            if 'decisions' not in self.decisions_json:
                self.decisions_json['decisions'] = []

            self.decisions_json['decisions'].append({
                'entity_index': entity_index,
                'decision': decision,
                'timestamp': timezone.now().isoformat(),
                'edit_distance': entity.get('edit_distance', 0),
                'token_changes': entity.get('token_changes', 0)
            })

            self.save()

    def calculate_entity_edit_distance(self, text1, text2):
        """Calculate edit distance between two entity texts"""
        return AnnotationStatistics.calculate_edit_distance(text1, text2)

    def calculate_entity_token_changes(self, text1, text2):
        """Calculate token changes between two entity texts"""
        return AnnotationStatistics.calculate_token_changes(text1, text2)

    def get_llm_decision_summary(self) -> dict:
        """Get a comprehensive summary of LLM decisions for this annotation,
        including manual entities added by the annotator in LLM-assisted mode."""
        summary = {
            'total_llm_entities': 0,
            'accepted': 0,
            'rejected': 0,
            'pending': 0,
            'manual_added': 0,  # Manual entities added by annotator in LLM mode
            'acceptance_rate': 0.0,
            'rejection_rate': 0.0,
            'active_entities': 0,  # Non-rejected entity count
        }

        for entity in self.entities_json:
            source = entity.get('source', 'llm')
            status = entity.get('status', 'pending')

            if source == 'manual':
                summary['manual_added'] += 1
                summary['active_entities'] += 1
            elif source == 'llm':
                summary['total_llm_entities'] += 1

                if status == 'accepted' or status == 'approved':
                    summary['accepted'] += 1
                    summary['active_entities'] += 1
                elif status == 'rejected':
                    summary['rejected'] += 1
                    # Rejected entities are NOT active
                else:
                    summary['pending'] += 1
                    summary['active_entities'] += 1

        # Calculate rates
        if summary['total_llm_entities'] > 0:
            total_decided = summary['accepted'] + summary['rejected']
            if total_decided > 0:
                summary['acceptance_rate'] = (summary['accepted'] / total_decided) * 100
                summary['rejection_rate'] = (summary['rejected'] / total_decided) * 100

        return summary

    def calculate_confidence_score(self):
        """Calculate and update average confidence score"""
        if self.entities_json:
            confidences = [entity.get('confidence', 0.0) for entity in self.entities_json]
            self.confidence_score = sum(confidences) / len(confidences) if confidences else 0.0
        else:
            self.confidence_score = 0.0

    def save(self, *args, **kwargs):
        # Auto-calculate fields before saving
        # total_entities counts only active (non-rejected) entities
        self.total_entities = sum(
            1 for e in self.entities_json
            if e.get('status') != 'rejected'
        )
        self.calculate_confidence_score()
        super().save(*args, **kwargs)

        # Auto-create gold standard if annotator is an expert
        if self.annotator.is_expert():
            self._create_gold_standard_from_annotation()

    def _create_gold_standard_from_annotation(self):
        """
        Automatically create or update gold standard when expert completes annotation
        """
        # Import here to avoid circular dependency
        from annotation.models import GoldStandard

        # Check if gold standard already exists for this document and task type
        gold_standard, created = GoldStandard.objects.update_or_create(
            document=self.document,
            task_type=self.task_type,
            defaults={
                'entities_json': self.entities_json,
                'created_by': self.annotator,
                'validation_notes': f'Auto-created from expert annotation ID {self.id}',
                'is_active': True
            }
        )
        return gold_standard


class AnnotationStatistics(models.Model):
    """Model for storing aggregated annotation statistics for admin dashboard"""

    PERIOD_CHOICES = [
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('all_time', 'All Time'),
    ]

    TASK_TYPES = [
        ('ner', 'Named Entity Recognition'),
        ('mcn', 'Medical Concept Normalization'),
        ('all', 'All Tasks'),
    ]

    # Identifiers
    annotator = models.ForeignKey(User, on_delete=models.CASCADE, related_name='annotation_statistics', null=True, blank=True)
    task_type = models.CharField(max_length=10, choices=TASK_TYPES, default='all')
    period_type = models.CharField(max_length=10, choices=PERIOD_CHOICES, default='all_time')
    period_start = models.DateTimeField(null=True, blank=True)
    period_end = models.DateTimeField(null=True, blank=True)

    # Timing Statistics
    total_tasks_completed = models.PositiveIntegerField(default=0)
    total_annotation_time_minutes = models.FloatField(default=0.0)
    average_annotation_time_minutes = models.FloatField(default=0.0)
    min_annotation_time_minutes = models.FloatField(null=True, blank=True)
    max_annotation_time_minutes = models.FloatField(null=True, blank=True)

    # LLM Prediction Statistics
    total_llm_predictions = models.PositiveIntegerField(default=0)
    accepted_predictions = models.PositiveIntegerField(default=0)
    modified_predictions = models.PositiveIntegerField(default=0)
    rejected_predictions = models.PositiveIntegerField(default=0)

    # Calculated Rates (stored for efficiency)
    acceptance_rate = models.FloatField(default=0.0, help_text="Percentage of LLM predictions accepted without change")
    modification_rate = models.FloatField(default=0.0, help_text="Percentage of LLM predictions modified")
    rejection_rate = models.FloatField(default=0.0, help_text="Percentage of LLM predictions rejected")

    # Edit Distance Statistics for Modifications
    total_edit_distance = models.PositiveIntegerField(default=0)
    average_edit_distance = models.FloatField(default=0.0)
    total_token_changes = models.PositiveIntegerField(default=0)
    average_token_changes = models.FloatField(default=0.0)

    # Metadata
    calculated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-calculated_at']
        unique_together = ['annotator', 'task_type', 'period_type', 'period_start', 'period_end']
        indexes = [
            models.Index(fields=['annotator', 'task_type']),
            models.Index(fields=['period_type', 'period_start', 'period_end']),
        ]

    def __str__(self):
        annotator_name = self.annotator.username if self.annotator else "All Annotators"
        return f"{annotator_name} - {self.get_task_type_display()} - {self.get_period_type_display()}"

    def calculate_rates(self):
        """Calculate and update acceptance, modification, and rejection rates"""
        if self.total_llm_predictions > 0:
            self.acceptance_rate = (self.accepted_predictions / self.total_llm_predictions) * 100
            self.modification_rate = (self.modified_predictions / self.total_llm_predictions) * 100
            self.rejection_rate = (self.rejected_predictions / self.total_llm_predictions) * 100
        else:
            self.acceptance_rate = 0.0
            self.modification_rate = 0.0
            self.rejection_rate = 0.0

    def calculate_edit_distances(self):
        """Calculate average edit distances and token changes"""
        if self.modified_predictions > 0:
            self.average_edit_distance = self.total_edit_distance / self.modified_predictions
            self.average_token_changes = self.total_token_changes / self.modified_predictions
        else:
            self.average_edit_distance = 0.0
            self.average_token_changes = 0.0

    def save(self, *args, **kwargs):
        """Auto-calculate rates before saving"""
        self.calculate_rates()
        self.calculate_edit_distances()
        super().save(*args, **kwargs)

    @classmethod
    def calculate_and_store_statistics(cls, annotator=None, task_type='all', period_type='all_time', period_start=None, period_end=None):
        """Calculate and store statistics for given parameters"""

        # Build queryset for AnnotationTask
        task_queryset = AnnotationTask.objects.filter(is_completed=True)

        if annotator:
            task_queryset = task_queryset.filter(annotator=annotator)
        if task_type != 'all':
            task_queryset = task_queryset.filter(task_type=task_type)
        if period_start:
            task_queryset = task_queryset.filter(completed_at__gte=period_start)
        if period_end:
            task_queryset = task_queryset.filter(completed_at__lte=period_end)

        # DEDUPLICATION: Handle potential duplicates by keeping only the most recent task
        # per (document, annotator, task_type) combination
        # This ensures accurate statistics even if duplicate tasks exist
        unique_tasks = []
        seen_combinations = set()

        for task in task_queryset.order_by('-assigned_at'):
            combo = (task.document_id, task.annotator_id, task.task_type)
            if combo not in seen_combinations:
                unique_tasks.append(task)
                seen_combinations.add(combo)

        # Replace queryset with deduplicated tasks
        task_queryset = AnnotationTask.objects.filter(id__in=[t.id for t in unique_tasks])

        # Calculate timing statistics
        timing_stats = task_queryset.aggregate(
            total_tasks=Count('id'),
            total_time=Sum('actual_duration'),
            avg_time=Avg('actual_duration'),
        )

        # Convert timedelta to minutes
        total_minutes = 0.0
        avg_minutes = 0.0
        min_minutes = None
        max_minutes = None

        if timing_stats['total_time']:
            total_minutes = timing_stats['total_time'].total_seconds() / 60
        if timing_stats['avg_time']:
            avg_minutes = timing_stats['avg_time'].total_seconds() / 60

        # Get min/max times
        completed_tasks = task_queryset.exclude(actual_duration__isnull=True)
        if completed_tasks.exists():
            durations = [task.get_duration_minutes() for task in completed_tasks]
            if durations:
                min_minutes = min(durations)
                max_minutes = max(durations)

        # Build queryset for AnnotationResult (LLM statistics)
        result_queryset = AnnotationResult.objects.filter(
            annotation_type__in=['llm_assisted', 'mcn_llm_assisted']
        )

        if annotator:
            result_queryset = result_queryset.filter(annotator=annotator)
        if task_type != 'all':
            result_queryset = result_queryset.filter(task_type=task_type)
        if period_start:
            result_queryset = result_queryset.filter(created_at__gte=period_start)
        if period_end:
            result_queryset = result_queryset.filter(created_at__lte=period_end)

        # Calculate LLM prediction statistics
        llm_stats = {
            'total_predictions': 0,
            'accepted': 0,
            'modified': 0,
            'rejected': 0,
            'total_edit_distance': 0,
            'total_token_changes': 0,
        }

        for result in result_queryset:
            for entity in result.entities_json:
                if entity.get('source') == 'llm':
                    llm_stats['total_predictions'] += 1
                    status = entity.get('status', 'pending')

                    if status == 'accepted' or status == 'approved':
                        llm_stats['accepted'] += 1
                    elif status == 'modified':
                        llm_stats['modified'] += 1
                        # Calculate edit distance if original_text and modified_text exist
                        original = entity.get('original_text', '')
                        modified = entity.get('text', '')
                        if original and modified:
                            edit_distance = cls.calculate_edit_distance(original, modified)
                            token_changes = cls.calculate_token_changes(original, modified)
                            llm_stats['total_edit_distance'] += edit_distance
                            llm_stats['total_token_changes'] += token_changes
                    elif status == 'rejected':
                        llm_stats['rejected'] += 1

        # Create or update statistics record
        stats, created = cls.objects.update_or_create(
            annotator=annotator,
            task_type=task_type,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            defaults={
                'total_tasks_completed': timing_stats['total_tasks'] or 0,
                'total_annotation_time_minutes': total_minutes,
                'average_annotation_time_minutes': avg_minutes,
                'min_annotation_time_minutes': min_minutes,
                'max_annotation_time_minutes': max_minutes,
                'total_llm_predictions': llm_stats['total_predictions'],
                'accepted_predictions': llm_stats['accepted'],
                'modified_predictions': llm_stats['modified'],
                'rejected_predictions': llm_stats['rejected'],
                'total_edit_distance': llm_stats['total_edit_distance'],
                'total_token_changes': llm_stats['total_token_changes'],
            }
        )

        return stats

    @staticmethod
    def calculate_edit_distance(text1, text2):
        """Calculate edit distance between two texts"""
        return len(list(difflib.unified_diff(text1.split(), text2.split())))

    @staticmethod
    def calculate_token_changes(text1, text2):
        """Calculate number of token changes between two texts"""
        tokens1 = text1.split()
        tokens2 = text2.split()

        # Use difflib to find actual changes
        differ = difflib.SequenceMatcher(None, tokens1, tokens2)
        changes = 0
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag != 'equal':
                changes += max(i2 - i1, j2 - j1)

        return changes


class AutoMCNMapping(models.Model):
    """
    Model for storing automatic Medical Concept Normalization (MCN) mappings
    Maps Indonesian medical terms to SNOMED-CT concepts using LLM
    """

    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('modified', 'Modified'),
    ]

    # Document relationship
    document = models.ForeignKey(
        TextDocument,
        on_delete=models.CASCADE,
        related_name='auto_mcn_mappings',
        help_text="Document this mapping belongs to"
    )

    # Extracted term info
    indonesian_term = models.CharField(
        max_length=255,
        help_text="Original Indonesian medical term extracted from text"
    )
    term_context = models.TextField(
        blank=True,
        help_text="Surrounding text context where term was found"
    )

    # LLM-generated mappings (JSON array of candidate SNOMED terms)
    suggested_mappings = models.JSONField(
        default=list,
        help_text="Array of suggested SNOMED-CT mappings from LLM with preferred_term_en and notes"
    )

    # Selected/approved mapping
    selected_snomed_term = models.CharField(
        max_length=255,
        blank=True,
        help_text="The SNOMED-CT preferred term selected/approved by annotator"
    )
    snomed_concept_id = models.CharField(
        max_length=50,
        blank=True,
        help_text="SNOMED-CT concept ID (if verified)"
    )

    # Mapping metadata
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    confidence_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence score from LLM (0-1)"
    )
    mapping_notes = models.TextField(
        blank=True,
        help_text="Notes about ambiguity, cultural terms, etc from LLM"
    )

    # Review metadata
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_mcn_mappings',
        help_text="Annotator who reviewed this mapping"
    )
    reviewed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the mapping was reviewed"
    )
    reviewer_notes = models.TextField(
        blank=True,
        help_text="Notes from the reviewer"
    )

    # Creation metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='created_mcn_mappings',
        help_text="User who triggered the automatic MCN mapping"
    )
    model_used = models.CharField(
        max_length=100,
        blank=True,
        help_text="LLM model used for mapping"
    )

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['document', 'status']),
            models.Index(fields=['indonesian_term']),
            models.Index(fields=['status']),
            models.Index(fields=['created_by']),
        ]
        unique_together = ['document', 'indonesian_term']

    def __str__(self):
        return f"{self.indonesian_term} -> {self.selected_snomed_term or 'Pending'}"

    def get_primary_suggestion(self):
        """Get the first/primary SNOMED-CT suggestion"""
        if self.suggested_mappings and len(self.suggested_mappings) > 0:
            first_mapping = self.suggested_mappings[0]
            preferred_terms = first_mapping.get('preferred_term_en', [])
            if preferred_terms and len(preferred_terms) > 0:
                return preferred_terms[0]
        return None

    def get_all_suggestions(self):
        """Get all SNOMED-CT term suggestions as a flat list"""
        all_terms = []
        for mapping in self.suggested_mappings:
            if 'preferred_term_en' in mapping:
                terms = mapping['preferred_term_en']
                if isinstance(terms, list):
                    all_terms.extend(terms)
                elif isinstance(terms, str):
                    all_terms.append(terms)
        return all_terms

    def approve_mapping(self, snomed_term, user, snomed_id=None, notes=""):
        """Approve a mapping with a selected SNOMED term"""
        self.selected_snomed_term = snomed_term
        self.snomed_concept_id = snomed_id or ""
        self.status = 'approved'
        self.reviewed_by = user
        self.reviewed_at = timezone.now()
        self.reviewer_notes = notes
        self.save()

    def reject_mapping(self, user, notes=""):
        """Reject the automatic mapping"""
        self.status = 'rejected'
        self.reviewed_by = user
        self.reviewed_at = timezone.now()
        self.reviewer_notes = notes
        self.save()

    def modify_mapping(self, snomed_term, user, snomed_id=None, notes=""):
        """Modify the mapping with a custom SNOMED term"""
        self.selected_snomed_term = snomed_term
        self.snomed_concept_id = snomed_id or ""
        self.status = 'modified'
        self.reviewed_by = user
        self.reviewed_at = timezone.now()
        self.reviewer_notes = notes
        self.save()

    @classmethod
    def get_pending_count(cls, document=None, user=None):
        """Get count of pending mappings"""
        queryset = cls.objects.filter(status='pending')
        if document:
            queryset = queryset.filter(document=document)
        if user:
            queryset = queryset.filter(document__assigned_to=user)
        return queryset.count()

    @classmethod
    def get_approved_count(cls, document=None, user=None):
        """Get count of approved mappings"""
        queryset = cls.objects.filter(status='approved')
        if document:
            queryset = queryset.filter(document=document)
        if user:
            queryset = queryset.filter(reviewed_by=user)
        return queryset.count()

    @classmethod
    def get_document_mappings(cls, document):
        """Get all mappings for a document, grouped by status"""
        mappings = cls.objects.filter(document=document)
        return {
            'pending': mappings.filter(status='pending'),
            'approved': mappings.filter(status='approved'),
            'modified': mappings.filter(status='modified'),
            'rejected': mappings.filter(status='rejected'),
            'total': mappings.count()
        }


class GoldStandard(models.Model):
    """
    Model for storing expert-validated gold standard annotations
    Used for calculating precision, recall, and F1-score
    """

    TASK_TYPES = [
        ('ner', 'Named Entity Recognition'),
        ('mcn', 'Medical Concept Normalization'),
    ]

    document = models.ForeignKey(
        TextDocument,
        on_delete=models.CASCADE,
        related_name='gold_standards',
        help_text="Document with gold standard annotation"
    )

    task_type = models.CharField(
        max_length=10,
        choices=TASK_TYPES,
        default='ner',
        help_text="Type of annotation task"
    )

    # Gold standard data
    entities_json = models.JSONField(
        default=list,
        help_text="JSON array of expert-validated entities"
    )

    # Metadata
    created_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='created_gold_standards',
        help_text="Expert who created this gold standard"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Validation info
    validation_notes = models.TextField(
        blank=True,
        help_text="Notes about validation process and decisions"
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this gold standard is currently active"
    )

    class Meta:
        ordering = ['-created_at']
        unique_together = ['document', 'task_type']
        indexes = [
            models.Index(fields=['document', 'task_type']),
            models.Index(fields=['is_active']),
        ]

    def __str__(self):
        return f"Gold Standard - {self.document.title[:50]} ({self.get_task_type_display()})"

    def get_entity_count(self):
        """Get total number of entities in gold standard"""
        return len(self.entities_json)

    def get_entities_by_type(self):
        """Get entities grouped by type"""
        entities_by_type = {}
        for entity in self.entities_json:
            entity_type = entity.get('label', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        return entities_by_type

    @staticmethod
    def _filter_subset_entities(entities):
        """Remove entities whose span is completely contained within a longer entity."""
        if not entities:
            return entities

        filtered = []
        for i, e in enumerate(entities):
            s1 = e.get('start_offset', e.get('start', 0))
            e1 = e.get('end_offset', e.get('end', 0))
            is_subset = False
            for j, other in enumerate(entities):
                if i == j:
                    continue
                s2 = other.get('start_offset', other.get('start', 0))
                e2 = other.get('end_offset', other.get('end', 0))
                if s2 <= s1 and e1 <= e2 and (e2 - s2) > (e1 - s1):
                    is_subset = True
                    break
            if not is_subset:
                filtered.append(e)
        return filtered

    def get_filtered_entities_by_type(self) -> dict:
        """Get entities grouped by type, with subset spans removed for display."""
        filtered = self._filter_subset_entities(self.entities_json or [])
        entities_by_type = {}
        for entity in filtered:
            entity_type = entity.get('label', 'UNKNOWN')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        return entities_by_type


class AnnotatorAgreement(models.Model):
    """
    Model for tracking inter-annotator agreement metrics
    Stores Cohen's kappa and other agreement statistics
    """

    TASK_TYPES = [
        ('ner', 'Named Entity Recognition'),
        ('mcn', 'Medical Concept Normalization'),
    ]

    # Documents and annotators being compared
    document = models.ForeignKey(
        TextDocument,
        on_delete=models.CASCADE,
        related_name='annotator_agreements',
        help_text="Document that was annotated by multiple annotators"
    )

    annotator1 = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='agreements_as_annotator1',
        help_text="First annotator"
    )

    annotator2 = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='agreements_as_annotator2',
        help_text="Second annotator"
    )

    task_type = models.CharField(
        max_length=10,
        choices=TASK_TYPES,
        default='ner',
        help_text="Type of annotation task"
    )

    # Agreement metrics
    cohens_kappa = models.FloatField(
        null=True,
        blank=True,
        help_text="Cohen's kappa coefficient (-1 to 1)"
    )

    percent_agreement = models.FloatField(
        null=True,
        blank=True,
        help_text="Simple percentage agreement (0-100)"
    )

    # Entity-level agreement details
    total_entities_annotator1 = models.PositiveIntegerField(
        default=0,
        help_text="Total entities marked by annotator 1"
    )

    total_entities_annotator2 = models.PositiveIntegerField(
        default=0,
        help_text="Total entities marked by annotator 2"
    )

    agreed_entities = models.PositiveIntegerField(
        default=0,
        help_text="Number of entities both annotators agreed on"
    )

    disagreed_entities = models.PositiveIntegerField(
        default=0,
        help_text="Number of entities annotators disagreed on"
    )

    # Detailed agreement data
    agreement_details_json = models.JSONField(
        default=dict,
        help_text="Detailed breakdown of agreements/disagreements by entity type"
    )

    # Metadata
    calculated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    notes = models.TextField(
        blank=True,
        help_text="Additional notes about this agreement calculation"
    )

    class Meta:
        ordering = ['-calculated_at']
        unique_together = ['document', 'annotator1', 'annotator2', 'task_type']
        indexes = [
            models.Index(fields=['document', 'task_type']),
            models.Index(fields=['annotator1', 'annotator2']),
            models.Index(fields=['cohens_kappa']),
        ]

    def __str__(self):
        return f"Agreement: {self.annotator1.username} vs {self.annotator2.username} on {self.document.title[:30]}"

    @classmethod
    def calculate_agreement(cls, document, annotator1, annotator2, task_type='ner'):
        """
        Calculate inter-annotator agreement between two annotators
        Returns the AnnotatorAgreement instance with calculated metrics
        """
        from .utils.agreement import calculate_cohens_kappa, calculate_entity_agreement

        # Get annotations from both annotators
        annotation1 = AnnotationResult.objects.filter(
            document=document,
            annotator=annotator1,
            task_type=task_type,
            review_status='completed'
        ).first()

        annotation2 = AnnotationResult.objects.filter(
            document=document,
            annotator=annotator2,
            task_type=task_type,
            review_status='completed'
        ).first()

        if not annotation1 or not annotation2:
            return None

        # Calculate agreement metrics
        agreement_data = calculate_entity_agreement(
            annotation1.entities_json,
            annotation2.entities_json
        )

        kappa = calculate_cohens_kappa(
            annotation1.entities_json,
            annotation2.entities_json
        )

        # Create or update agreement record
        agreement, created = cls.objects.update_or_create(
            document=document,
            annotator1=annotator1,
            annotator2=annotator2,
            task_type=task_type,
            defaults={
                'cohens_kappa': kappa,
                'percent_agreement': agreement_data['percent_agreement'],
                'total_entities_annotator1': len(annotation1.entities_json),
                'total_entities_annotator2': len(annotation2.entities_json),
                'agreed_entities': agreement_data['agreed_entities'],
                'disagreed_entities': agreement_data['disagreed_entities'],
                'agreement_details_json': agreement_data['details'],
            }
        )

        return agreement

    def get_agreement_level(self):
        """Get qualitative description of agreement level based on kappa"""
        if self.cohens_kappa is None:
            return "Not calculated"

        kappa = self.cohens_kappa
        if kappa < 0:
            return "Poor (worse than chance)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"


class AnnotationMetrics(models.Model):
    """
    Model for storing comprehensive annotation evaluation metrics
    Includes precision, recall, F1-score against gold standard
    """

    TASK_TYPES = [
        ('ner', 'Named Entity Recognition'),
        ('mcn', 'Medical Concept Normalization'),
    ]

    # Reference data
    annotation_result = models.ForeignKey(
        AnnotationResult,
        on_delete=models.CASCADE,
        related_name='metrics',
        help_text="Annotation result being evaluated"
    )

    gold_standard = models.ForeignKey(
        GoldStandard,
        on_delete=models.CASCADE,
        related_name='evaluation_metrics',
        help_text="Gold standard used for evaluation",
        null=True,
        blank=True
    )

    # Core metrics
    precision = models.FloatField(
        null=True,
        blank=True,
        help_text="Precision score (0-1)"
    )

    recall = models.FloatField(
        null=True,
        blank=True,
        help_text="Recall score (0-1)"
    )

    f1_score = models.FloatField(
        null=True,
        blank=True,
        help_text="F1 score (0-1)"
    )

    # Confusion matrix elements
    true_positives = models.PositiveIntegerField(
        default=0,
        help_text="Correctly identified entities"
    )

    false_positives = models.PositiveIntegerField(
        default=0,
        help_text="Incorrectly identified entities"
    )

    false_negatives = models.PositiveIntegerField(
        default=0,
        help_text="Missed entities"
    )

    true_negatives = models.PositiveIntegerField(
        default=0,
        help_text="Correctly identified non-entities"
    )

    # Efficiency metrics
    annotation_time_seconds = models.FloatField(
        null=True,
        blank=True,
        help_text="Total time spent on annotation"
    )

    average_time_per_entity = models.FloatField(
        null=True,
        blank=True,
        help_text="Average time per entity in seconds"
    )

    # Per-entity-type metrics
    metrics_by_entity_type = models.JSONField(
        default=dict,
        help_text="Precision/recall/F1 breakdown by entity type"
    )

    # Metadata
    calculated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    notes = models.TextField(
        blank=True,
        help_text="Additional evaluation notes"
    )

    class Meta:
        ordering = ['-calculated_at']
        indexes = [
            models.Index(fields=['annotation_result']),
            models.Index(fields=['gold_standard']),
            models.Index(fields=['f1_score']),
        ]

    def __str__(self):
        return f"Metrics: {self.annotation_result.annotator.username} - F1: {self.f1_score:.3f if self.f1_score else 'N/A'}"

    @classmethod
    def calculate_metrics(cls, annotation_result, gold_standard=None):
        """
        Calculate all metrics for an annotation result
        If gold_standard is provided, calculates precision/recall/F1
        """
        from .utils.metrics import calculate_precision_recall_f1, calculate_entity_level_metrics

        # Get or create gold standard if not provided
        if not gold_standard:
            gold_standard = GoldStandard.objects.filter(
                document=annotation_result.document,
                task_type=annotation_result.task_type,
                is_active=True
            ).first()

        metrics_data = {}

        if gold_standard:
            # Calculate precision, recall, F1
            metrics_data = calculate_precision_recall_f1(
                annotation_result.entities_json,
                gold_standard.entities_json
            )

            # Calculate per-entity-type metrics
            entity_metrics = calculate_entity_level_metrics(
                annotation_result.entities_json,
                gold_standard.entities_json
            )
            metrics_data['metrics_by_entity_type'] = entity_metrics

        # Calculate efficiency metrics
        task = AnnotationTask.objects.filter(
            document=annotation_result.document,
            annotator=annotation_result.annotator,
            task_type=annotation_result.task_type
        ).first()

        annotation_time = None
        avg_time_per_entity = None

        if task and task.actual_duration:
            annotation_time = task.actual_duration.total_seconds()
            if annotation_result.total_entities > 0:
                avg_time_per_entity = annotation_time / annotation_result.total_entities

        # Create or update metrics
        metric, created = cls.objects.update_or_create(
            annotation_result=annotation_result,
            defaults={
                'gold_standard': gold_standard,
                'precision': metrics_data.get('precision'),
                'recall': metrics_data.get('recall'),
                'f1_score': metrics_data.get('f1_score'),
                'true_positives': metrics_data.get('true_positives', 0),
                'false_positives': metrics_data.get('false_positives', 0),
                'false_negatives': metrics_data.get('false_negatives', 0),
                'true_negatives': metrics_data.get('true_negatives', 0),
                'annotation_time_seconds': annotation_time,
                'average_time_per_entity': avg_time_per_entity,
                'metrics_by_entity_type': metrics_data.get('metrics_by_entity_type', {}),
            }
        )

        return metric

    def get_accuracy(self):
        """Calculate accuracy from confusion matrix"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    def get_annotation_time_display(self):
        """Get human-readable annotation time"""
        if not self.annotation_time_seconds:
            return "N/A"

        minutes = self.annotation_time_seconds / 60
        if minutes < 1:
            return f"{self.annotation_time_seconds:.0f} seconds"
        elif minutes < 60:
            return f"{minutes:.1f} minutes"
        else:
            hours = minutes / 60
            return f"{hours:.1f} hours"
