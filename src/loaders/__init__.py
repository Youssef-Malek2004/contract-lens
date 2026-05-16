from .interface import ModelHandle, ModelLoader
from .local import LocalLoader, get_device
from .openrouter import OpenRouterConfig, OpenRouterLoader
from .vllm import VllmConfig, VllmLoader


def get_loader(mode: str = "local", **kwargs) -> ModelLoader:
    """
    Factory for model loaders.

    mode="local"      → LocalLoader(device=...) — loads weights into process memory
    mode="vllm"       → VllmLoader(config=VllmConfig(...)) — connects to running servers
    mode="openrouter" → OpenRouterLoader(config=OpenRouterConfig(...)) — hits OpenRouter API

    Examples:
        get_loader("local")
        get_loader("local", device="cuda")
        get_loader("vllm")
        get_loader("vllm", config=VllmConfig(nli_url="http://localhost:9002/v1"))
        get_loader("openrouter")
        get_loader("openrouter", orchestrator_model_id="qwen/qwen3-32b")
    """
    if mode == "local":
        return LocalLoader(device=kwargs.get("device"))
    if mode == "vllm":
        _VLLM_KEYS = {"orchestrator_url", "orchestrator_model_id",
                      "nli_url", "nli_model_id", "base_url", "base_model_id"}
        config = kwargs.get("config") or VllmConfig(
            **{k: v for k, v in kwargs.items() if k in _VLLM_KEYS}
        )
        return VllmLoader(config)
    if mode == "openrouter":
        _OR_KEYS = {"api_key", "base_url", "orchestrator_model_id",
                    "base_model_id", "http_referer", "x_title"}
        config = kwargs.get("config") or OpenRouterConfig(
            **{k: v for k, v in kwargs.items() if k in _OR_KEYS}
        )
        return OpenRouterLoader(config)
    raise ValueError(
        f"Unknown loader mode '{mode}'. Choose 'local', 'vllm', or 'openrouter'."
    )


__all__ = [
    "ModelHandle",
    "ModelLoader",
    "LocalLoader",
    "VllmLoader",
    "VllmConfig",
    "OpenRouterLoader",
    "OpenRouterConfig",
    "get_loader",
    "get_device",
]
