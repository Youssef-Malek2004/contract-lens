# Backward-compatibility shim. New code should import from src.loaders directly.
from src.loaders import get_device, get_loader, ModelHandle, LocalLoader, VllmLoader, VllmConfig
from src.loaders.vllm import VllmModelHandle as RemoteOrchestrator  # legacy alias


def load_orchestrator(device: str = None, remote: bool = False):
    loader = get_loader("vllm" if remote else "local", device=device)
    handle = loader.load_orchestrator()
    return handle, handle.tokenizer


def load_nli_model(device: str = None):
    handle = LocalLoader(device=device).load_nli_model()
    return handle, handle.tokenizer
