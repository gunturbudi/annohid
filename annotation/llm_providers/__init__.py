"""
LLM Providers for Multi-Model NER
Supports multiple LLM providers with unified interface
"""

from .base import BaseLLMProvider, NERResult, Entity
from .gemini_provider import GeminiNERProvider
from .openrouter_provider import OpenRouterNERProvider
from .together_provider import TogetherNERProvider
from .factory import (
    get_ner_provider,
    get_all_available_providers,
    get_default_models,
    get_openrouter_models,
    get_free_models,
    is_openrouter_available,
)

__all__ = [
    'BaseLLMProvider',
    'NERResult',
    'Entity',
    'GeminiNERProvider',
    'OpenRouterNERProvider',
    'TogetherNERProvider',
    'get_ner_provider',
    'get_all_available_providers',
    'get_default_models',
    'get_openrouter_models',
    'get_free_models',
    'is_openrouter_available',
]
