from .interface import ModelHandle, ModelLoader
from .local import LocalLoader, get_device
from .vllm import VllmLoader, VllmConfig


def get_loader(mode: str = "local", **kwargs) -> ModelLoader:
    """
    Factory for model loaders.

    mode="local"  → LocalLoader(device=...) — loads weights into process memory
    mode="vllm"   → VllmLoader(config=VllmConfig(...)) — connects to running servers

    Examples:
        get_loader("local")
        get_loader("local", device="cuda")
        get_loader("vllm")
        get_loader("vllm", config=VllmConfig(nli_url="http://localhost:9002/v1"))
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
    raise ValueError(f"Unknown loader mode '{mode}'. Choose 'local' or 'vllm'.")


__all__ = [
    "ModelHandle",
    "ModelLoader",
    "LocalLoader",
    "VllmLoader",
    "VllmConfig",
    "get_loader",
    "get_device",
]
