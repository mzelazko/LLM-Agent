"""
Simple configuration management for agent.
Reads from environment variables with defaults.
"""
import os
from typing import Optional


class Config:
    """Configuration settings for the agent."""
    WORKING_DIR: str = os.environ.get("AGENT_WORKING_DIR", "./working_dir")
    MAX_STEPS: int = int(os.environ.get("AGENT_MAX_STEPS", "20"))
    COMMAND_TIMEOUT: int = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "30"))
    PROMPT_VERSION: str = os.environ.get("AGENT_PROMPT_VERSION", "v2")
    # Model name in format "provider/model-name"
    # Examples of supported providers:
    #   - openai: openai/gpt-5-mini-2025-08-07
    #   - anthropic: anthropic/claude-sonnet-4-5-20250929, anthropic/claude-haiku-4-5-20251001
    #   - openrouter: openrouter/x-ai/grok-code-fast-1, openrouter/minimax/minimax-m2, openrouter/mistralai/devstral-2512 
    #   - pcss: pcss/DeepSeek-V3.1-vLLM, pcss/DeepSeek-V3.1-vLLM-2, pcss/bielik:11b, pcss/llama3.3:70b, pcss/gpt-oss_120b
    #   - bedrock: bedrock/converse/openai.gpt-oss-120b-1:0, arn:aws:bedrock:us-east-1:082447916006:inference-profile/us.mistral.pixtral-large-2502-v1:0, mistral doesnt work
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "openai/gpt-5-mini-2025-08-07") 
    # API settings
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
    PCSS_API_KEY: Optional[str] = os.environ.get("PCSS_API_KEY")
    PCSS_API_BASE: Optional[str] = os.environ.get("PCSS_API_BASE", "https://llm.hpc.psnc.pl")
    AWS_BEARER_TOKEN_BEDROCK: Optional[str] = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    AWS_BEDROCK_API_BASE: Optional[str] = os.environ.get("AWS_BEDROCK_API_BASE", "https://bedrock-runtime.us-east-1.amazonaws.com") # e.g. https://bedrock-runtime.us-east-1.amazonaws.com, https://bedrock-runtime.eu-central-1.amazonaws.com
    # Checks if API key is available for the MODEL_NAME
    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""

        provider = cls.MODEL_NAME.split("/", 1)[0].lower() if "/" in cls.MODEL_NAME else "openai"

        provider_required_key = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "together_ai": ["TOGETHER_API_KEY"],
            "together": ["TOGETHER_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "xai": ["XAI_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "fireworks_ai": ["FIREWORKS_API_KEY"],
            "fireworks": ["FIREWORKS_API_KEY"],
            "azure": ["AZURE_OPENAI_API_KEY"],
            "vertex_ai": ["VERTEXAI_API_KEY"],
            "google": ["GEMINI_API_KEY"],
            "gemini": ["GEMINI_API_KEY"],
            "pcss": ["PCSS_API_KEY"], # PCSS_API_BASE is set to default value https://llm.hpc.psnc.pl   
            "bedrock": ["AWS_BEARER_TOKEN_BEDROCK"] # AWS_BEDROCK_API_BASE is set to default value https://bedrock-runtime.eu-central-1.amazonaws.com
        }

        required_env_vars = provider_required_key.get(provider)
        if not required_env_vars:
            raise ValueError(f"Unknown provider '{provider}' in MODEL_NAME='{cls.MODEL_NAME}'.")

        missing_keys = [k for k in required_env_vars if not os.environ.get(k)]
        if missing_keys:
            raise ValueError(
                f"Missing required env var(s) for provider '{provider}': {missing_keys}. "
                f"MODEL_NAME='{cls.MODEL_NAME}'"
            )
