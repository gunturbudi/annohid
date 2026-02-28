"""
Centralized LLM Settings Manager
Provides unified interface for API key management across env and database
"""

import os
from typing import Optional, Dict, List
from .api_key_models import LLMProvider, APIKey, LLMConfiguration


class LLMSettingsManager:
    """
    Centralized manager for LLM API keys and configuration
    Checks both environment variables and database
    Priority: Environment Variables > Database
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache = {}
        self._initialized = True

    def get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get API key for a provider
        Checks environment first, then database

        Args:
            provider_name: Provider name ('openrouter', 'gemini', 'openai', 'anthropic')

        Returns:
            API key string or None
        """
        # Check cache
        if provider_name in self._cache:
            return self._cache[provider_name]

        # Map provider names to environment variables
        env_vars = {
            'openrouter': 'OPENROUTER_API_KEY',
            'together': 'TOGETHER_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
        }

        # Check environment variable first (highest priority)
        env_var = env_vars.get(provider_name)
        if env_var:
            api_key = os.getenv(env_var)
            if api_key:
                self._cache[provider_name] = api_key
                return api_key

        # Check database
        try:
            provider = LLMProvider.objects.get(name=provider_name, is_active=True)
            api_key = provider.get_api_key()

            if api_key:
                self._cache[provider_name] = api_key
                return api_key

        except LLMProvider.DoesNotExist:
            pass

        return None

    def has_api_key(self, provider_name: str) -> bool:
        """Check if API key exists for provider"""
        return self.get_api_key(provider_name) is not None

    def get_all_configured_providers(self) -> List[str]:
        """Get list of all configured provider names"""
        configured = []

        provider_names = ['openrouter', 'gemini', 'openai', 'anthropic']

        for name in provider_names:
            if self.has_api_key(name):
                configured.append(name)

        return configured

    def get_preferred_provider(self) -> Optional[str]:
        """Get the preferred provider from configuration"""
        try:
            config = LLMConfiguration.get_config()
            if config.preferred_provider:
                return config.preferred_provider.name
        except:
            pass

        # Default: OpenRouter if available, else first configured
        if self.has_api_key('openrouter'):
            return 'openrouter'

        configured = self.get_all_configured_providers()
        return configured[0] if configured else None

    def get_configuration(self) -> Dict:
        """Get LLM configuration"""
        try:
            config = LLMConfiguration.get_config()
            return {
                'preferred_provider': config.preferred_provider.name if config.preferred_provider else None,
                'enable_fallback': config.enable_fallback,
                'enable_cost_tracking': config.enable_cost_tracking,
                'monthly_budget_usd': float(config.monthly_budget_usd) if config.monthly_budget_usd else None,
                'max_requests_per_hour': config.max_requests_per_hour,
            }
        except:
            return {
                'preferred_provider': None,
                'enable_fallback': True,
                'enable_cost_tracking': True,
                'monthly_budget_usd': None,
                'max_requests_per_hour': None,
            }

    def clear_cache(self):
        """Clear API key cache"""
        self._cache = {}

    def get_provider_info(self, provider_name: str) -> Optional[Dict]:
        """Get information about a provider"""
        try:
            provider = LLMProvider.objects.get(name=provider_name)
            return {
                'name': provider.name,
                'display_name': provider.display_name,
                'description': provider.description,
                'is_active': provider.is_active,
                'has_key': provider.has_api_key(),
                'priority': provider.priority,
                'env_var_name': provider.env_var_name,
                'documentation_url': provider.documentation_url,
            }
        except LLMProvider.DoesNotExist:
            return None

    def get_all_providers_info(self) -> List[Dict]:
        """Get information about all providers"""
        providers = []

        try:
            for provider in LLMProvider.objects.all().order_by('-priority', 'name'):
                providers.append({
                    'name': provider.name,
                    'display_name': provider.display_name,
                    'description': provider.description,
                    'is_active': provider.is_active,
                    'has_key': provider.has_api_key(),
                    'priority': provider.priority,
                })
        except:
            pass

        return providers


# Global instance
_settings_manager = LLMSettingsManager()


def get_llm_settings() -> LLMSettingsManager:
    """Get global LLM settings manager instance"""
    return _settings_manager


def get_api_key_for_provider(provider_name: str) -> Optional[str]:
    """
    Convenience function to get API key for a provider

    Args:
        provider_name: 'openrouter', 'gemini', 'openai', or 'anthropic'

    Returns:
        API key string or None
    """
    return get_llm_settings().get_api_key(provider_name)
