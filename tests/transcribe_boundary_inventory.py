"""Exact module inventory for each transcription ownership boundary."""

EXPECTED_PEER_MODULES = {
    "adapters": {
        "adapters",
        "adapters.aligner",
        "adapters.diarizer",
        "adapters.firered",
        "adapters.firered_ledger",
        "adapters.qwen",
        "adapters.silero",
        "adapters.vibevoice",
    },
    "execution": {
        "execution",
        "execution.materialization",
        "execution.preflight",
        "execution.product_validation",
        "execution.publication",
        "execution.runtime",
        "execution.vad",
    },
    "orchestrator": {
        "orchestrator",
        "orchestrator.common",
        "orchestrator.firered",
        "orchestrator.qwen",
        "orchestrator.vibevoice",
    },
    "planner": {"planner", "planner.build", "planner.request"},
    "refusals": {"refusals", "refusals.request"},
    "result": {"result", "result.serialization", "result.types", "result.validation"},
    "stages": {
        "stages",
        "stages._firered_protocol",
        "stages.aligner",
        "stages.firered",
        "stages.qwen",
        "stages.vibevoice",
    },
    "transport": {
        "transport",
        "transport.process_runner",
        "transport.service",
        "transport.types",
    },
}

EXPECTED_STAGE_RUNTIME_IMPORTS = {
    "stages": set(),
    "stages._firered_protocol": set(),
    "stages.aligner": {"mlx", "mlx_audio", "numpy"},
    "stages.firered": {"_firered_protocol", "fireredasr2s", "soundfile"},
    "stages.qwen": {"mlx", "mlx_audio", "mlx_lm", "numpy"},
    "stages.vibevoice": {"numpy", "torch", "vibevoice"},
}
