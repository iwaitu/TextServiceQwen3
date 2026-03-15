import os
import platform

import onnxruntime as ort


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"

_PROVIDER_ALIASES = {
    "cpu": CPU_PROVIDER,
    "cpuexecutionprovider": CPU_PROVIDER,
    "cuda": CUDA_PROVIDER,
    "cudaexecutionprovider": CUDA_PROVIDER,
    "coreml": COREML_PROVIDER,
    "coremlexecutionprovider": COREML_PROVIDER,
    "auto": "auto",
}


def _normalize_provider_name(name):
    normalized = name.strip()
    alias = _PROVIDER_ALIASES.get(normalized.lower())
    return alias or normalized


def _parse_provider_chain(raw_value):
    providers = []
    for item in raw_value.split(","):
        normalized = _normalize_provider_name(item)
        if normalized and normalized != "auto":
            providers.append(normalized)
    return providers


def choose_execution_providers():
    available = ort.get_available_providers()

    configured_chain = os.environ.get("ONNX_EXECUTION_PROVIDERS", "").strip()
    if configured_chain:
        requested = [provider for provider in _parse_provider_chain(configured_chain) if provider in available]
        if requested:
            if CPU_PROVIDER not in requested:
                requested.append(CPU_PROVIDER)
            return requested, f"configured via ONNX_EXECUTION_PROVIDERS={configured_chain}", available

    preferred_provider = os.environ.get("ONNX_PREFERRED_PROVIDER", os.environ.get("ONNX_PROVIDER", "auto")).strip()
    normalized_preferred = _normalize_provider_name(preferred_provider or "auto")
    if normalized_preferred != "auto":
        if normalized_preferred in available:
            if normalized_preferred == CPU_PROVIDER:
                return [CPU_PROVIDER], f"preferred provider {normalized_preferred} from environment", available
            return [normalized_preferred, CPU_PROVIDER], f"preferred provider {normalized_preferred} from environment", available
        return [CPU_PROVIDER], f"preferred provider {normalized_preferred} unavailable, falling back to CPU", available

    if platform.system() == "Darwin" and COREML_PROVIDER in available:
        return [COREML_PROVIDER, CPU_PROVIDER], "auto-detected macOS CoreML support", available

    if CUDA_PROVIDER in available:
        return [CUDA_PROVIDER, CPU_PROVIDER], "auto-detected CUDA support", available

    return [CPU_PROVIDER], "defaulted to CPUExecutionProvider", available


def provider_chain_to_string(providers):
    return " -> ".join(providers)