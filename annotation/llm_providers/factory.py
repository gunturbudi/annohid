"""
Factory for creating LLM providers
Manages provider instantiation and availability checking
"""

from typing import Dict, List, Optional, Type
from .base import BaseLLMProvider
from .gemini_provider import GeminiNERProvider
from .openai_provider import OpenAINERProvider
from .anthropic_provider import AnthropicNERProvider
from .openrouter_provider import OpenRouterNERProvider
from .together_provider import TogetherNERProvider


# Registry of available providers
PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    # Gemini models (direct API)
    'gemini-2.5-flash': GeminiNERProvider,
    'gemini-1.5-pro': GeminiNERProvider,
    'gemini-1.5-flash': GeminiNERProvider,
    'gemini-pro': GeminiNERProvider,

    # OpenAI models (direct API)
    'gpt-4o': OpenAINERProvider,
    'gpt-4-turbo': OpenAINERProvider,
    'gpt-4': OpenAINERProvider,
    'gpt-3.5-turbo': OpenAINERProvider,

    # Anthropic models (direct API)
    'claude-3-5-sonnet-20241022': AnthropicNERProvider,
    'claude-3-5-sonnet-20240620': AnthropicNERProvider,
    'claude-3-opus-20240229': AnthropicNERProvider,
    'claude-3-sonnet-20240229': AnthropicNERProvider,
    'claude-3-haiku-20240307': AnthropicNERProvider,
}

# OpenRouter models (100+ models via single API key!)
# Add all OpenRouter models to the registry
for model_id in OpenRouterNERProvider.SUPPORTED_MODELS.keys():
    PROVIDER_REGISTRY[model_id] = OpenRouterNERProvider

# Together AI models (100+ open-source models with credits!)
# Add all Together AI models to the registry
for model_id in TogetherNERProvider.SUPPORTED_MODELS.keys():
    PROVIDER_REGISTRY[model_id] = TogetherNERProvider


def get_ner_provider(model_id: str, api_key: Optional[str] = None) -> Optional[BaseLLMProvider]:
    """
    Get an NER provider for the specified model

    Args:
        model_id: The model identifier (e.g., 'gemini-2.5-flash', 'gpt-4o')
        api_key: Optional API key (uses database or environment variable if not provided)

    Returns:
        Initialized provider instance, or None if model not found
    """
    provider_class = PROVIDER_REGISTRY.get(model_id)

    if provider_class is None:
        print(f"Unknown model: {model_id}")
        return None

    # If no API key provided, try to get from database
    if api_key is None:
        provider_class_to_db_name = {
            GeminiNERProvider: 'gemini',
            OpenAINERProvider: 'openai',
            AnthropicNERProvider: 'anthropic',
            OpenRouterNERProvider: 'openrouter',
            TogetherNERProvider: 'together',
        }

        db_provider_name = provider_class_to_db_name.get(provider_class)

        if db_provider_name:
            try:
                # Import here to avoid circular dependency
                from ..api_key_models import LLMProvider as DBLLMProvider

                db_provider = DBLLMProvider.objects.filter(
                    name=db_provider_name,
                    is_active=True
                ).first()

                if db_provider:
                    api_key = db_provider.get_api_key()
            except Exception:
                # Database might not be set up yet, continue with env vars only
                pass

    try:
        provider = provider_class(model_id=model_id, api_key=api_key)

        if provider.initialize():
            return provider
        else:
            print(f"Failed to initialize provider for {model_id}")
            return None

    except Exception as e:
        print(f"Error creating provider for {model_id}: {e}")
        return None


def get_all_available_providers() -> List[Dict[str, str]]:
    """
    Get list of all available providers (those with API keys configured)
    Checks both environment variables and database-stored API keys

    Returns:
        List of dicts with model info: {'model_id': ..., 'display_name': ..., 'provider': ...}
    """
    available = []

    # Map provider classes to their database names
    provider_class_to_db_name = {
        GeminiNERProvider: 'gemini',
        OpenAINERProvider: 'openai',
        AnthropicNERProvider: 'anthropic',
        OpenRouterNERProvider: 'openrouter',
        TogetherNERProvider: 'together',
    }

    # Check each unique provider class
    checked_classes = set()

    for model_id, provider_class in PROVIDER_REGISTRY.items():
        if provider_class in checked_classes:
            continue
        checked_classes.add(provider_class)

        # Get API key from database if available
        api_key = None
        db_provider_name = provider_class_to_db_name.get(provider_class)

        if db_provider_name:
            try:
                # Import here to avoid circular dependency
                from ..api_key_models import LLMProvider as DBLLMProvider

                db_provider = DBLLMProvider.objects.filter(
                    name=db_provider_name,
                    is_active=True
                ).first()

                if db_provider:
                    api_key = db_provider.get_api_key()
            except Exception as e:
                # Database might not be set up yet, continue with env vars only
                pass

        # Try to initialize with database key or env var
        try:
            provider = provider_class(model_id=model_id, api_key=api_key)
            if provider.initialize():
                # Provider is available, add all its models
                for mid, pclass in PROVIDER_REGISTRY.items():
                    if pclass == provider_class:
                        temp_provider = pclass(model_id=mid, api_key=api_key)
                        available.append({
                            'model_id': mid,
                            'display_name': temp_provider.get_model_display_name(),
                            'provider': temp_provider.get_provider_name(),
                        })
        except Exception as e:
            # Failed to initialize this provider, skip it
            pass

    return available


def get_default_models() -> List[str]:
    """
    Get list of recommended default models for multi-LLM annotation

    Returns:
        List of model IDs that should be used by default if available
    """
    # Prefer OpenRouter if available (single API key for all models!)
    openrouter_models = [
        'openai/gpt-4o',                    # OpenRouter: GPT-4o
        'anthropic/claude-3.5-sonnet',      # OpenRouter: Claude 3.5
        'google/gemini-2.5-flash',           # OpenRouter: Gemini 2.5
    ]

    # Together AI free models (great for testing!)
    together_models = [
        'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',  # Together: Llama 3.3 70B (FREE)
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',  # Together: Llama 3.1 70B
    ]

    # Fallback to direct APIs
    direct_api_models = [
        'gemini-2.5-flash',                 # Direct: Gemini
        'gpt-4o',                           # Direct: OpenAI
        'claude-3-5-sonnet-20241022',       # Direct: Anthropic
    ]

    # Return OpenRouter models first (preferred), then Together AI, then direct APIs
    return openrouter_models + together_models + direct_api_models


def get_provider_display_info() -> Dict[str, Dict[str, str]]:
    """
    Get display information for all registered providers

    Returns:
        Dict mapping model_id to display information
    """
    info = {}

    for model_id, provider_class in PROVIDER_REGISTRY.items():
        provider = provider_class(model_id=model_id)
        info[model_id] = {
            'display_name': provider.get_model_display_name(),
            'provider_name': provider.get_provider_name(),
            'model_id': model_id,
        }

    return info


def get_openrouter_models() -> Dict[str, str]:
    """
    Get all OpenRouter models

    Returns:
        Dict mapping model_id to display name
    """
    return OpenRouterNERProvider.get_all_supported_models()


def get_free_models() -> List[str]:
    """
    Get list of free models (great for testing without API costs!)

    Returns:
        List of free model IDs
    """
    return OpenRouterNERProvider.get_free_models()


def is_openrouter_available() -> bool:
    """
    Check if OpenRouter is configured

    Returns:
        True if OpenRouter API key is set
    """
    try:
        provider = OpenRouterNERProvider(model_id='openai/gpt-4o-mini')
        return provider.initialize()
    except:
        return False
