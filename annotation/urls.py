from django.urls import path
from . import views
from . import multi_llm_views
from . import api_key_views
from . import metrics_views

app_name = 'annotation'

urlpatterns = [
    # Admin data management
    path('admin/data/', views.admin_data_management, name='admin_data_management'),
    path('admin/import/', views.import_text_data, name='import_text_data'),
    path('admin/documents/', views.document_list, name='document_list'),
    path('admin/documents/<int:document_id>/', views.document_detail, name='document_detail'),
    path('admin/documents/<int:document_id>/assign/', views.assign_document, name='assign_document'),
    path('admin/documents/bulk-assign/', views.bulk_assign_documents, name='bulk_assign_documents'),
    path('admin/documents/auto-assign/', views.auto_assign_documents, name='auto_assign_documents'),
    path('admin/import-history/', views.import_history, name='import_history'),

    # LLM Annotation (Legacy)
    path('admin/documents/<int:document_id>/annotate/', views.annotate_document, name='annotate_document'),
    path('admin/documents/<int:document_id>/perform-annotation/', views.perform_annotation, name='perform_annotation'),
    path('admin/annotations/<int:annotation_id>/', views.view_annotation_result, name='view_annotation_result'),
    path('admin/annotations/', views.annotation_results_list, name='annotation_results_list'),
    path('admin/annotations/<int:annotation_id>/update-status/', views.update_annotation_status, name='update_annotation_status'),

    # Batch LLM Annotation
    path('admin/documents/batch-llm-annotate/', views.batch_llm_annotate, name='batch_llm_annotate'),

    # Annotator Interface
    path('my-documents/', views.annotator_documents, name='annotator_documents'),

    # NER Annotation Modes
    path('documents/<int:document_id>/manual/', views.manual_annotate, name='manual_annotate'),
    path('documents/<int:document_id>/llm-assisted/', views.llm_assisted_annotate, name='llm_assisted_annotate'),

    # MCN Annotation Modes
    path('documents/<int:document_id>/mcn/manual/', views.mcn_manual, name='mcn_manual'),
    path('documents/<int:document_id>/mcn/llm-assisted/', views.mcn_llm_assisted, name='mcn_llm_assisted'),

    # SNOMED-CT Integration
    path('api/search-snomed/', views.search_snomed, name='search_snomed'),
    path('api/map-entity-snomed/', views.map_entity_to_snomed, name='map_entity_to_snomed'),
    path('api/normalize-terms/', views.normalize_medical_terms, name='normalize_medical_terms'),
    path('api/auto-map-snomed/', views.auto_map_to_snomed, name='auto_map_to_snomed'),
    path('api/translate-indonesian/', views.translate_indonesian_entities, name='translate_indonesian_entities'),

    # Entity Definitions API (includes colors and definitions)
    path('api/entity-definitions/', views.get_entity_definitions, name='get_entity_definitions'),

    # Time Analytics
    path('admin/time-analytics/', views.time_analytics, name='time_analytics'),

    # Annotation Statistics
    path('admin/statistics/', views.admin_statistics_dashboard, name='admin_statistics_dashboard'),
    path('admin/statistics/refresh/', views.refresh_statistics, name='refresh_statistics'),
    path('admin/statistics/export/', views.export_statistics, name='export_statistics'),
    path('admin/export-sql/', views.export_sql, name='export_sql'),

    # Granular Statistics (Drill-down views)
    path('admin/statistics/annotator/<int:annotator_id>/', views.annotator_detail_stats, name='annotator_detail_stats'),
    path('admin/statistics/annotator/<int:annotator_id>/tasks/', views.annotator_task_breakdown, name='annotator_task_breakdown'),
    path('admin/statistics/document/<int:document_id>/', views.document_stats, name='document_stats'),
    path('admin/statistics/task/<int:task_id>/', views.task_detail_stats, name='task_detail_stats'),

    # Guidelines Management
    path('admin/guidelines/', views.admin_guidelines, name='admin_guidelines'),
    path('admin/guidelines/upload/', views.upload_guideline, name='upload_guideline'),
    path('admin/guidelines/<int:guideline_id>/edit/', views.edit_guideline, name='edit_guideline'),
    path('admin/guidelines/<int:guideline_id>/delete/', views.delete_guideline, name='delete_guideline'),
    path('admin/guidelines/<int:guideline_id>/toggle-status/', views.toggle_guideline_status, name='toggle_guideline_status'),

    # Guidelines Viewing (for annotators and admins)
    path('guidelines/', views.view_guidelines, name='view_guidelines'),
    path('guidelines/<int:guideline_id>/', views.guideline_detail, name='guideline_detail'),

    # Multi-LLM NER
    path('documents/<int:document_id>/multi-llm-ner/', multi_llm_views.multi_llm_ner_run, name='multi_llm_ner_run'),
    path('documents/<int:document_id>/multi-llm-ner/comparison/', multi_llm_views.multi_llm_ner_comparison, name='multi_llm_ner_comparison'),
    path('documents/<int:document_id>/multi-llm-ner/approve/', multi_llm_views.multi_llm_ner_approve, name='multi_llm_ner_approve'),
    path('admin/multi-llm-ner/stats/', multi_llm_views.multi_llm_ner_stats, name='multi_llm_ner_stats'),

    # Batch Multi-LLM NER
    path('batch-multi-llm-ner/', multi_llm_views.batch_multi_llm_ner, name='batch_multi_llm_ner'),

    # LangExtract
    path('api/run-langextract/', views.run_langextract, name='run_langextract'),

    # Single Document 3-Step MCN
    path('api/run-three-step-mcn/', views.run_three_step_mcn, name='run_three_step_mcn'),

    # Bulk MCN 3-Step Processing
    path('api/bulk-mcn-three-step/', views.bulk_mcn_three_step, name='bulk_mcn_three_step'),

    # Admin Popup Views - Get Annotation Data
    path('api/documents/<int:document_id>/ner-data/', views.get_ner_annotation_data, name='get_ner_annotation_data'),
    path('api/documents/<int:document_id>/mcn-data/', views.get_mcn_annotation_data, name='get_mcn_annotation_data'),

    # API Key Management
    path('admin/api-keys/', api_key_views.api_key_management, name='api_key_management'),
    path('admin/api-keys/add/<int:provider_id>/', api_key_views.add_api_key, name='add_api_key'),
    path('api-keys/<int:key_id>/test/', api_key_views.test_api_key, name='test_api_key'),
    path('api-keys/<int:key_id>/toggle/', api_key_views.toggle_api_key, name='toggle_api_key'),
    path('api-keys/<int:key_id>/delete/', api_key_views.delete_api_key, name='delete_api_key'),
    path('api-keys/test-all/', api_key_views.test_all_keys, name='test_all_keys'),
    path('admin/llm-config/update/', api_key_views.update_llm_config, name='update_llm_config'),

    # Evaluation Metrics
    path('admin/metrics/', metrics_views.evaluation_metrics_dashboard, name='evaluation_metrics_dashboard'),
    path('admin/metrics/calculate-agreement/', metrics_views.calculate_agreement, name='calculate_agreement'),
    path('admin/metrics/calculate-metrics/', metrics_views.calculate_metrics, name='calculate_metrics'),
    path('admin/metrics/create-gold-standard/', metrics_views.create_gold_standard, name='create_gold_standard'),
    path('admin/metrics/annotator-performance/', metrics_views.annotator_performance_report, name='annotator_performance_report'),
    path('admin/metrics/annotator-performance/<int:annotator_id>/', metrics_views.annotator_performance_report, name='annotator_performance_detail'),
    path('admin/metrics/export/', metrics_views.export_metrics_data, name='export_metrics_data'),
]