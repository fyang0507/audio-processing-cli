"""Lazy MLX model loading and deterministic bucketed turn inference."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import plan as plan_module
from . import runtime


@dataclass(frozen=True)
class LoadedModel:
    """The loaded private API plus evidence binding it to installed source."""

    model: Any
    mlx: Any
    parameter_bytes: int
    api_probe: dict[str, Any]


@dataclass(frozen=True)
class InferenceResult:
    """Chronological transcript output and its generation ledger."""

    segments: list[dict[str, Any]]
    qwen: dict[str, Any]
    status: str


def load_model(
    model_path: Path,
    *,
    mlx_holder: dict[str, Any],
    timing: dict[str, float],
    explicit_mlx_peaks: dict[str, int],
) -> LoadedModel:
    """Load MLX lazily and validate the private batched method this runner uses."""

    import mlx.core as mx  # noqa: PLC0415
    from mlx.utils import tree_flatten  # noqa: PLC0415
    from mlx_audio.stt.utils import load_model as load_mlx_model  # noqa: PLC0415

    mlx_holder["module"] = mx
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal device is unavailable")

    mx.reset_peak_memory()
    started = time.perf_counter()
    model = load_mlx_model(model_path, lazy=False, strict=False)
    mx.eval(model.parameters())
    mx.synchronize()
    timing["model_load_s"] = time.perf_counter() - started
    explicit_mlx_peaks["model_load"] = int(mx.get_peak_memory())

    flattened = tree_flatten(model.parameters())
    parameter_bytes = sum(int(value.nbytes) for _, value in flattened)
    model_source = Path(inspect.getfile(type(model))).resolve()
    signature = inspect.signature(model._generate_chunks_batched)
    required = {
        "chunks",
        "max_tokens",
        "sampler",
        "language",
        "system_prompt",
        "batch_size",
        "verbose",
    }
    if not required.issubset(signature.parameters):
        raise RuntimeError("installed mlx-audio private batched API does not match runner")
    api_probe = {
        "model_generate_signature": str(inspect.signature(model.generate)),
        "private_batched_method": "_generate_chunks_batched",
        "private_batched_signature": str(signature),
        "source_path": str(model_source),
        "source_sha256": runtime.sha256(model_source),
        "reason": (
            "The public generate() accepts one waveform and batches only its "
            "internally split chunks. This runner calls the inspected private "
            "batched method to pass already bounded diarizer turns while keeping "
            "one loaded model. The source hash and signature make that coupling explicit."
        ),
    }
    return LoadedModel(model, mx, parameter_bytes, api_probe)


def run_batches(
    loaded: LoadedModel,
    accepted: list[plan_module.Turn],
    audio: Any,
    *,
    language: str,
    batch_size: int,
    max_tokens: int,
    process_start: float,
    rss_reader: runtime.RssReader,
    samples: list[dict[str, Any]],
    timing: dict[str, float],
    explicit_mlx_peaks: dict[str, int],
) -> InferenceResult:
    """Transcribe duration-bucketed turns and restore chronological output order."""

    from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

    model = loaded.model
    mx = loaded.mlx
    mx.reset_peak_memory()
    started = time.perf_counter()

    # Duration bucketing reduces zero-padding without changing published order.
    inference_order = sorted(
        range(len(accepted)),
        key=lambda index: (accepted[index].end - accepted[index].start, index),
    )
    chunks = [
        (audio[accepted[index].start : accepted[index].end], 0.0) for index in inference_order
    ]
    texts: list[str] = []
    generated: list[int] = []
    prompts: list[int] = []
    processed: list[bool] = []
    remaining_tokens = max_tokens
    cache_observations: list[dict[str, Any]] = []

    for batch_index, batch_start in enumerate(range(0, len(chunks), batch_size)):
        if remaining_tokens <= 0:
            break
        group = chunks[batch_start : batch_start + batch_size]
        group_texts, group_generated, group_prompts, group_processed = (
            model._generate_chunks_batched(
                group,
                max_tokens=remaining_tokens,
                sampler=make_sampler(temp=0.0),
                language=language,
                system_prompt=None,
                batch_size=batch_size,
                verbose=False,
            )
        )
        if not all(isinstance(value, bool) for value in group_processed):
            raise TypeError("private batched API processed result is not bool list")

        mx.synchronize()
        observation = {
            "batch_index": batch_index,
            "batch_size": len(group),
            "before_clear_active_bytes": int(mx.get_active_memory()),
            "before_clear_cache_bytes": int(mx.get_cache_memory()),
        }
        samples.append(
            {
                "elapsed_s": time.perf_counter() - process_start,
                "phase": "inference_before_batch_cache_clear",
                "sample_origin": "synchronous_runner_observation",
                "batch_index": batch_index,
                "rss_bytes": rss_reader.read(),
                "mlx_active_bytes": observation["before_clear_active_bytes"],
                "mlx_cache_bytes": observation["before_clear_cache_bytes"],
                "mlx_peak_active_bytes": int(mx.get_peak_memory()),
            }
        )
        texts.extend(group_texts)
        generated.extend(group_generated)
        prompts.extend(group_prompts)
        processed.extend(group_processed)
        remaining_tokens -= sum(group_generated)

        mx.clear_cache()
        mx.synchronize()
        observation.update(
            {
                "after_clear_active_bytes": int(mx.get_active_memory()),
                "after_clear_cache_bytes": int(mx.get_cache_memory()),
            }
        )
        cache_observations.append(observation)
        samples.append(
            {
                "elapsed_s": time.perf_counter() - process_start,
                "phase": "inference_after_batch_cache_clear",
                "sample_origin": "synchronous_runner_observation",
                "batch_index": batch_index,
                "rss_bytes": rss_reader.read(),
                "mlx_active_bytes": observation["after_clear_active_bytes"],
                "mlx_cache_bytes": observation["after_clear_cache_bytes"],
                "mlx_peak_active_bytes": int(mx.get_peak_memory()),
            }
        )

    if len(processed) < len(chunks):
        missing = len(chunks) - len(processed)
        texts.extend([""] * missing)
        generated.extend([0] * missing)
        prompts.extend([0] * missing)
        processed.extend([False] * missing)
    timing["inference_s"] = time.perf_counter() - started
    explicit_mlx_peaks["inference"] = int(mx.get_peak_memory())

    restored = {
        chronological_index: (text, generation, prompt, was_processed)
        for chronological_index, text, generation, prompt, was_processed in zip(
            inference_order, texts, generated, prompts, processed, strict=True
        )
    }
    segments: list[dict[str, Any]] = []
    unprocessed: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(accepted):
        text, generation, prompt, was_processed = restored[turn_index]
        if not was_processed:
            unprocessed.append(plan_module.turn_record(turn, turn_index))
            continue
        segments.append(
            {
                **plan_module.turn_record(turn, turn_index),
                "text": text,
                "language": language,
                "prompt_tokens": prompt,
                "generation_tokens": generation,
                "timestamp_source": "FluidAudio anonymous diarization turn",
            }
        )

    qwen = {
        "input_turns": len(accepted),
        "processed_turns": sum(bool(value) for value in processed),
        "unprocessed_turns": unprocessed,
        "prompt_tokens": sum(prompts),
        "generation_tokens": sum(generated),
        "global_generation_budget_tokens": max_tokens,
        "generation_budget_remaining_tokens": max_tokens - sum(generated),
        "inference_order": "ascending duration_samples then chronological index",
        "output_restored_to_chronological_order": True,
        "inference_batch_count": len(cache_observations),
        "cache_cleared_between_batches": True,
        "cache_clear_observations": cache_observations,
    }
    status = "ok" if not unprocessed else "partial_generation_budget"
    return InferenceResult(segments, qwen, status)
