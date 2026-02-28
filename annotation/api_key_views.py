"""
Views for API Key management
Admin-only interface for managing LLM API keys
"""

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils import timezone

from .api_key_models import LLMProvider, APIKey, LLMConfiguration
from .llm_providers import get_all_available_providers, get_openrouter_models


def is_admin(user):
    """Check if user is admin/staff"""
    return user.is_staff


@login_required
@user_passes_test(is_admin)
def api_key_management(request):
    """
    Main API key management dashboard
    Shows all providers and their configured keys
    """
    # Get or create providers
    _ensure_default_providers()

    providers = LLMProvider.objects.all().order_by('-priority', 'name')
    config = LLMConfiguration.get_config()

    # Get available models
    available_providers = get_all_available_providers()
    openrouter_models = get_openrouter_models()

    context = {
        'providers': providers,
        'config': config,
        'available_providers': available_providers,
        'openrouter_models': openrouter_models,
        'total_providers': providers.count(),
        'configured_providers': sum(1 for p in providers if p.has_api_key()),
    }

    return render(request, 'annotation/api_key_management.html', context)


@login_required
@user_passes_test(is_admin)
def add_api_key(request, provider_id):
    """Add a new API key for a provider"""
    provider = get_object_or_404(LLMProvider, id=provider_id)

    if request.method == 'POST':
        key_name = request.POST.get('key_name')
        api_key = request.POST.get('api_key')
        notes = request.POST.get('notes', '')

        if not key_name or not api_key:
            messages.error(request, "Key name and API key are required.")
            return redirect('annotation:api_key_management')

        # Create API key
        api_key_obj = APIKey.objects.create(
            provider=provider,
            key_name=key_name,
            api_key=api_key,
            notes=notes,
            created_by=request.user
        )

        # Test the key
        if api_key_obj.test_key():
            messages.success(
                request,
                f"API key '{key_name}' added and tested successfully for {provider.display_name}!"
            )
        else:
            messages.warning(
                request,
                f"API key '{key_name}' added but test failed: {api_key_obj.test_message}"
            )

        return redirect('annotation:api_key_management')

    # GET request - show form
    context = {
        'provider': provider,
    }
    return render(request, 'annotation/add_api_key.html', context)


@login_required
@user_passes_test(is_admin)
@require_http_methods(["POST"])
def test_api_key(request, key_id):
    """Test an API key"""
    api_key = get_object_or_404(APIKey, id=key_id)

    success = api_key.test_key()

    return JsonResponse({
        'success': success,
        'status': api_key.test_status,
        'message': api_key.test_message,
        'last_tested': api_key.last_tested.isoformat() if api_key.last_tested else None
    })


@login_required
@user_passes_test(is_admin)
@require_http_methods(["POST"])
def toggle_api_key(request, key_id):
    """Toggle API key active status"""
    api_key = get_object_or_404(APIKey, id=key_id)

    api_key.is_active = not api_key.is_active
    api_key.save()

    status = "activated" if api_key.is_active else "deactivated"
    messages.success(request, f"API key '{api_key.key_name}' {status}.")

    return redirect('annotation:api_key_management')


@login_required
@user_passes_test(is_admin)
@require_http_methods(["POST"])
def delete_api_key(request, key_id):
    """Delete an API key"""
    api_key = get_object_or_404(APIKey, id=key_id)

    key_name = api_key.key_name
    api_key.delete()

    messages.success(request, f"API key '{key_name}' deleted.")
    return redirect('annotation:api_key_management')


@login_required
@user_passes_test(is_admin)
@require_http_methods(["POST"])
def update_llm_config(request):
    """Update LLM configuration"""
    config = LLMConfiguration.get_config()

    # Update preferred provider
    preferred_provider_id = request.POST.get('preferred_provider')
    if preferred_provider_id:
        try:
            config.preferred_provider = LLMProvider.objects.get(id=preferred_provider_id)
        except LLMProvider.DoesNotExist:
            pass

    # Update settings
    config.enable_fallback = request.POST.get('enable_fallback') == 'on'
    config.enable_cost_tracking = request.POST.get('enable_cost_tracking') == 'on'

    # Update budget
    monthly_budget = request.POST.get('monthly_budget_usd')
    if monthly_budget:
        try:
            config.monthly_budget_usd = float(monthly_budget)
        except ValueError:
            pass

    # Update rate limit
    max_requests = request.POST.get('max_requests_per_hour')
    if max_requests:
        try:
            config.max_requests_per_hour = int(max_requests)
        except ValueError:
            pass

    config.updated_by = request.user
    config.save()

    messages.success(request, "LLM configuration updated successfully.")
    return redirect('annotation:api_key_management')


@login_required
@user_passes_test(is_admin)
def test_all_keys(request):
    """Test all active API keys"""
    api_keys = APIKey.objects.filter(is_active=True)

    results = []
    for key in api_keys:
        success = key.test_key()
        results.append({
            'provider': key.provider.display_name,
            'key_name': key.key_name,
            'success': success,
            'status': key.test_status,
            'message': key.test_message
        })

    return JsonResponse({
        'results': results,
        'total': len(results),
        'successful': sum(1 for r in results if r['success'])
    })


def _ensure_default_providers():
    """Ensure default providers exist in database"""
    defaults = [
        {
            'name': 'openrouter',
            'display_name': 'OpenRouter',
            'description': 'Universal gateway to 100+ LLM models. One API key for OpenAI, Anthropic, Google, Meta, and more. Recommended!',
            'env_var_name': 'OPENROUTER_API_KEY',
            'base_url': 'https://openrouter.ai/api/v1',
            'documentation_url': 'https://openrouter.ai/docs',
            'priority': 100,  # Highest priority
        },
        {
            'name': 'together',
            'display_name': 'Together AI',
            'description': '100+ open-source models with credits. Llama, DeepSeek, Qwen, Mistral, and more. Great for bulk processing!',
            'env_var_name': 'TOGETHER_API_KEY',
            'base_url': 'https://api.together.xyz/v1',
            'documentation_url': 'https://docs.together.ai',
            'priority': 90,  # Second highest
        },
        {
            'name': 'anthropic',
            'display_name': 'Anthropic Claude',
            'description': 'Anthropic Claude models (4.5, 3.5 Sonnet, Opus, etc.). Excellent reasoning and nuanced understanding.',
            'env_var_name': 'ANTHROPIC_API_KEY',
            'base_url': 'https://api.anthropic.com',
            'documentation_url': 'https://docs.anthropic.com',
            'priority': 80,
        },
        {
            'name': 'openai',
            'display_name': 'OpenAI',
            'description': 'OpenAI GPT models (GPT-4o, o1, etc.). High quality and accurate.',
            'env_var_name': 'OPENAI_API_KEY',
            'base_url': 'https://api.openai.com/v1',
            'documentation_url': 'https://platform.openai.com/docs',
            'priority': 70,
        },
        {
            'name': 'gemini',
            'display_name': 'Google Gemini',
            'description': 'Google Gemini models (2.5 Flash, 1.5 Pro, etc.). Fast and good for overlapping entity detection.',
            'env_var_name': 'GEMINI_API_KEY',
            'base_url': '',
            'documentation_url': 'https://ai.google.dev/docs',
            'priority': 50,
        },
    ]

    for provider_data in defaults:
        LLMProvider.objects.get_or_create(
            name=provider_data['name'],
            defaults=provider_data
        )
