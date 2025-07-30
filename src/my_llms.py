import os
import json
import yaml

from langchain_openai import ChatOpenAI

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from langchain_openai import OpenAIEmbeddings

# Base paths and defaults
API_SECRET_DIR = "/run/secrets"
CONFIG_FILE_DEFAULT = os.getenv("LLM_CONFIG_FILE", "/app/mcp_config.yaml")

# Supported providers and their default models
SUPPORTED_PROVIDERS = ["openai", "azure", "lite", "custom", "anthropic"]
MODEL_DEFAULTS = {
    "openai":   "gpt-4o-mini",
    "azure":    "gpt-4o-mini",
    "lite":     "gpt-4o-mini",
    "custom":   "gpt-4o-mini",
    "anthropic": "claude-2",
}

SUPPORTED_LLMS = {
    "openai":   ["gpt-4o-mini"],
    "azure":    ["gpt-4o-mini"],
    "lite":     ["gpt-4o-mini"],
    "custom":   ["gpt-4o-mini"],
    "anthropic": ["claude-2"],
}


# Supported embedding providers and their default embedding models
EMBEDDING_MODEL_DEFAULTS = {
    "openai":   "text-embedding-3-small",
    "azure":    "text-embedding-3-small",
    "lite":     "text-embedding-3-small",
    "custom":   "text-embedding-3-small",
    # anthropic: no embedding support
}

# Supported embedding models (flattened list for annotation)
SUPPORTED_EMBEDDINGS = [
    EMBEDDING_MODEL_DEFAULTS[provider]
    for provider in EMBEDDING_MODEL_DEFAULTS
]


def detect_provider() -> str:
    """
    Determine the LLM provider by explicit override or by presence of provider-specific API keys.
    Defaults to 'openai'.
    """
    env_provider = os.getenv("LLM_PROVIDER")
    if env_provider and env_provider.lower() in SUPPORTED_PROVIDERS:
        return env_provider.lower()

    for p in SUPPORTED_PROVIDERS:
        secret_file = os.path.join(API_SECRET_DIR, f"{p}_api_key")
        if os.path.isfile(secret_file) or os.getenv(f"{p.upper()}_API_KEY"):
            return p

    return "openai"


def load_api_key(provider: str) -> str:
    """
    Load API key for the given provider, checking Docker secret then provider-specific env var.
    For OpenAI, falls back to generic API_KEY.
    """
    secret_file = os.path.join(API_SECRET_DIR, f"{provider}_api_key")
    if os.path.isfile(secret_file):
        return open(secret_file).read().strip()

    env_var = f"{provider.upper()}_API_KEY"
    key = os.getenv(env_var)
    if key:
        return key

    if provider == "openai":
        key = os.getenv("API_KEY")
        if key:
            return key

    raise EnvironmentError(
        f"No API key found for provider '{provider}'. Checked secret '{secret_file}' and env var '{env_var}'"
    )


def load_llm_settings(provider: str) -> dict:
    """
    Load core LLM settings using provider-specific default model, overlaying env vars and config file.
    """
    default_model = MODEL_DEFAULTS.get(provider, MODEL_DEFAULTS["openai"])
    settings = {
        "model":        os.getenv("LLM_MODEL", default_model),
        "temperature":  float(os.getenv("LLM_TEMPERATURE", "0.0")),
        "max_tokens":   int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "extra_params": json.loads(os.getenv("LLM_EXTRA_PARAMS", "{}")),
    }

    cfg_file = CONFIG_FILE_DEFAULT
    if cfg_file and os.path.isfile(cfg_file):
        loader = yaml.safe_load if cfg_file.endswith(('.yml', '.yaml')) else json.load
        with open(cfg_file) as f:
            data = loader(f)
        file_cfg = data.get("llm", {})
        for k, v in file_cfg.items():
            if k == "extra_params" and isinstance(v, dict):
                settings["extra_params"].update(v)
            else:
                settings[k] = v

    extra = settings.pop("extra_params", {})
    settings.update(extra)
    return settings


def detect_embedding_provider() -> str:
    """
    Determine the embedding provider by explicit override or default to 'openai'.
    """
    env_provider = os.getenv("EMBEDDING_PROVIDER")
    if env_provider and env_provider.lower() in EMBEDDING_MODEL_DEFAULTS:
        return env_provider.lower()
    # default to OpenAI for embeddings
    return "openai"


def get_llm(model: str = None):
    """
    Instantiate an LLM client based on chosen provider and loaded settings.
    """
    provider = detect_provider()
    api_key = load_api_key(provider)
    llm_args = load_llm_settings(provider)

    if provider in ["openai", "azure", "lite", "custom"]:
        params = {
            "model":            llm_args["model"],
            "temperature":      llm_args["temperature"],
            "max_tokens":       llm_args["max_tokens"],
            "openai_api_key":   api_key,
        }
        if os.getenv("OPENAI_API_BASE"):    params["openai_api_base"]    = os.getenv("OPENAI_API_BASE")
        if os.getenv("OPENAI_API_TYPE"):    params["openai_api_type"]    = os.getenv("OPENAI_API_TYPE")
        if os.getenv("OPENAI_API_VERSION"): params["openai_api_version"] = os.getenv("OPENAI_API_VERSION")

        if provider == "lite":
            params["openai_api_base"] = os.getenv("LITE_API_BASE", params.get("openai_api_base"))
        elif provider == "azure":
            params.setdefault("openai_api_type", "azure")
        elif provider == "custom":
            params["openai_api_base"] = os.getenv("CUSTOM_API_BASE", params.get("openai_api_base"))

        return ChatOpenAI(**params)


    if provider == "anthropic":
        if ChatAnthropic is None:
            raise ImportError(
                "Anthropic support requested but 'langchain_anthropic' is not installed."
            )
        anth_params = {
            "model":                llm_args["model"],
            "temperature":          llm_args["temperature"],
            "max_tokens":           llm_args["max_tokens"],
            "anthropic_api_key":    api_key,
        }
        if os.getenv("ANTHROPIC_API_BASE"):
            anth_params["anthropic_api_base"] = os.getenv("ANTHROPIC_API_BASE")
        return ChatAnthropic(**anth_params)

    raise EnvironmentError(f"Unhandled LLM_PROVIDER '{provider}'.")


def get_embedding_model(model:str = None) -> OpenAIEmbeddings:
    """
    Instantiate an embedding model client based on chosen provider and settings.
    """
    provider = detect_embedding_provider()
    # Anthropic does not support embeddings
    if provider == "anthropic":
        raise EnvironmentError("Anthropic provider does not support embeddings.")

    api_key = load_api_key(provider)
    model = os.getenv("EMBEDDING_-MODEL", EMBEDDING_MODEL_DEFAULTS.get(provider))
    if not model:
        raise EnvironmentError(
            f"No default embedding model for provider '{provider}'. Please set EMBEDDING_MODEL."
        )

    extra = json.loads(os.getenv("EMBEDDING_EXTRA_PARAMS", "{}"))
    params = {
        "model": model,
        "openai_api_key": api_key,
    }
    # propagate OpenAI API protocol overrides
    if os.getenv("OPENAI_API_BASE"):    params["openai_api_base"]    = os.getenv("OPENAI_API_BASE")
    if os.getenv("OPENAI_API_TYPE"):    params["openai_api_type"]    = os.getenv("OPENAI_API_TYPE")
    if os.getenv("OPENAI_API_VERSION"): params["openai_api_version"] = os.getenv("OPENAI_API_VERSION")

    params.update(extra)
    return OpenAIEmbeddings(**params)


print(get_llm())
print(get_embedding_model())