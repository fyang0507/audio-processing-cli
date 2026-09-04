"""Plan construction over a resolved request and the package manifest."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

from audio_cli import environments as env

from . import stacks
from .catalog import InputMetadata
from .plan import Plan
from .planner_request import ResolvedRequest

_STAGE_ORDER = ("decode", "vad", "diarizer", "lid", "asr", "aligner", "punctuator")
# These templates are executable configuration, not a second capability table.  Their values
# come from the runner sources named here; stacks.json separately cites the artifacts behind
# the catalog and evidence claims.
# - Qwen: model_tests/benchmark/run_turn_attributed_mlx_asr.py
# - FluidAudio: model_tests/benchmark/run_fluidaudio_diarization.py
# - Silero: model_tests/benchmark/run_silero_vad.py
# - aligner: model_tests/benchmark/run_mlx_forced_aligner_probe.py
# - VibeVoice: model_tests/benchmark/run_vibevoice.py
# - FireRed: model_tests/benchmark/run_firered.py
_DECODE = {
    "backend": "ffmpeg",
    "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"},
}
_FLUIDAUDIO = {
    "backend": "fluidaudio",
    "version": "0.15.5",
    "revision": "19600a485baa4998812e4654b70d2bab8f2c9949",
    "environment": "swift",
    "config": {
        "step_ratio": 0.1,
        "min_segment_duration": 0.0,
        "threshold": 0.6,
        "batch_size": 32,
    },
    "config_note": (
        "the shipped default supplies no speaker-count prior and enables "
        "overlapping_segments only when overlapped_speech is requested; the cited quality "
        "figures used both --num-speakers 2 and --overlapping-segments"
    ),
}
_SILERO = {
    "backend": "silero-vad",
    "environment": "core",
    "version": "silero-vad-6.2.1",
    "config": {
        "threshold": 0.5,
        "exit_threshold": 0.35,
        "min_speech_ms": 100,
        "min_silence_ms": 300,
        "speech_pad_ms": 120,
    },
}
_ALIGNER = {
    "backend": "qwen3-forcedaligner",
    "environment": "mlx",
    "config": {
        "scope": "all_segments",
        "language_rule": (
            "Chinese when text matches [一-鿿], otherwise English; the ASR --language hint "
            "is never forwarded"
        ),
    },
}
_VIBEVOICE = {
    "backend": "vibevoice-asr-7b",
    "environment": "torch-vibevoice",
    "patch": "vibevoice-logits-to-keep",
    "config": {
        "device": "mps",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "seed": 1234,
        "max_new_tokens": 16384,
    },
    "deterministic": True,
    "determinism_tolerance_ms": 0.0,
    "determinism_basis": (
        "three seeded repeats shared one normalized-output hash; text decode is do_sample=False"
    ),
    "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
    "selected_by": "stack",
}
_FIRERED_ROLES = {
    "vad": {
        "backend": "firered-vad",
        "environment": "torch-firered",
        "selected_by": "stack",
    },
    "asr": {
        "backend": "firered-asr2-aed",
        "environment": "torch-firered",
        "config": {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 4,
            "return_timestamp": True,
            "beam_size": 3,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0,
        },
        "selected_by": "stack",
        "deterministic": True,
        "determinism_tolerance_ms": 2.0,
        "determinism_basis": (
            "exact-repeat 60-minute fixture repeated the text and speaker-null sequences; "
            "maximum rebased timestamp drift was 1.0000000000002037 ms, within the frozen "
            "2.0 ms tolerance, so normalized segments were not byte-equal"
        ),
    },
    "punctuator": {
        "backend": "firered-punc",
        "environment": "torch-firered",
        "config": {"batch_size": 4},
        "selected_by": "floor:punctuated_sentence_segmented_text",
        "recases_text": True,
    },
    "lid": {
        "backend": "firered-lid",
        "environment": "torch-firered",
        "config": {"batch_size": 4},
        "selected_by": "requirement:lid",
        "granularity": "vad_region",
        "cost_note": "162.09 s with LID versus 84.24 s without, on the 139.284 s probe",
    },
}


def _manifest_role_source(package_id: str, role: str) -> dict[str, str]:
    source = env.packages()[package_id].source
    if source.get("type") == "huggingface_multi":
        matches = [
            repository
            for repository in source.get("repos", ())
            if repository.get("role") == role
        ]
        if len(matches) != 1:
            raise stacks.StackTableError(
                f"package {package_id!r} has no single source for role {role!r}"
            )
        source = matches[0]
    repository = source.get("repo")
    revision = source.get("revision")
    if not isinstance(repository, str) or not isinstance(revision, str):
        raise stacks.StackTableError(
            f"package {package_id!r} role {role!r} has no pinned Hub source"
        )
    return {"repository": repository, "revision": revision}


def _manifest_checkout_commit(package_id: str) -> str:
    checkout = env.packages()[package_id].checkout
    commit = checkout.get("resolved_commit") if checkout is not None else None
    if not isinstance(commit, str) or len(commit) != 40:
        raise stacks.StackTableError(
            f"package {package_id!r} has no fully resolved checkout commit"
        )
    return commit


def _manifest_revision(backend_id: str) -> str:
    backend = env.backends()[backend_id]
    return _manifest_role_source(backend.package, backend.role)["revision"]


def _role_template(
    role: str,
    backend: str,
    request: ResolvedRequest,
    selected_by: str | None = None,
) -> dict[str, Any]:
    if backend.startswith("qwen3-asr-"):
        value = {
            "backend": backend,
            "environment": "mlx",
            "revision": _manifest_revision(backend),
            "config": {
                "batch_size": 1,
                "clear_mlx_cache_after_every_batch": True,
                "language": request.language,
                "api_path": "_generate_chunks_batched",
                "max_tokens": 16384,
            },
            "adapter_strips": ["language <label><asr_text> scaffold"],
            "selected_by": "stack",
            "deterministic": True,
            "determinism_tolerance_ms": 0.0,
            "determinism_basis": (
                "argmax decode; back-to-back calls in one process produced byte-identical "
                "text, cross-process repetition untested"
            ),
        }
    elif backend == "vibevoice-asr-7b":
        value = copy.deepcopy(_VIBEVOICE)
        value["revision"] = _manifest_revision(backend)
        value["source_commit"] = _manifest_checkout_commit("vibevoice-asr-7b")
        value["tokenizer"] = {
            "materialized_role": "tokenizer",
            **_manifest_role_source("vibevoice-asr-7b", "tokenizer"),
        }
    elif backend in {"firered-vad", "firered-asr2-aed", "firered-punc", "firered-lid"}:
        value = copy.deepcopy(_FIRERED_ROLES[role])
        value["revision"] = _manifest_revision(backend)
        value["source_commit"] = _manifest_checkout_commit("firered-asr2s")
    elif backend == "fluidaudio":
        value = copy.deepcopy(_FLUIDAUDIO)
        if "overlapped_speech" in request.wants:
            value["config"]["overlapping_segments"] = True
    elif backend == "silero-vad":
        value = copy.deepcopy(_SILERO)
    elif backend == "qwen3-forcedaligner":
        value = copy.deepcopy(_ALIGNER)
        value["revision"] = _manifest_revision(backend)
    else:
        raise stacks.StackTableError(f"no plan role template for backend {backend!r}")
    if selected_by is not None:
        value["selected_by"] = selected_by
    return value


def _backend_for_package(package_id: str) -> env.Backend:
    candidates = [binding for binding in env.backends().values() if binding.package == package_id]
    if len(candidates) != 1:
        raise stacks.StackTableError(
            f"add-on package {package_id!r} must supply exactly one planner backend"
        )
    return candidates[0]


def _resolved_roles(request: ResolvedRequest) -> dict[str, dict[str, Any]]:
    backend_by_role = dict(request.stack.base_roles)
    selected_by: dict[str, str] = {}
    for capability in request.wants:
        cell = request.stack.capabilities[capability]
        if cell["resolution"] == "native_stage" and capability not in backend_by_role:
            backend_by_role[capability] = cell["backend"]
        if cell["resolution"] == "add_on":
            binding = _backend_for_package(cell["package"])
            backend_by_role[binding.role] = binding.id
            selected_by.setdefault(binding.role, f"add_on_required_by:{capability}")

    # FireRed's native VAD may be explicitly substituted by the one shipped VAD pin.  The pin
    # changes the satisfying backend, not the capability namespace.
    if request.vad == "silero-vad":
        backend_by_role["vad"] = "silero-vad"
        selected_by["vad"] = "pin:--vad"

    roles = {"decode": copy.deepcopy(_DECODE)}
    for role in _STAGE_ORDER:
        if role == "decode" or role not in backend_by_role:
            continue
        roles[role] = _role_template(
            role, backend_by_role[role], request, selected_by.get(role)
        )
    return roles


def _capability_results(request: ResolvedRequest) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for capability in request.wants:
        cell = request.stack.capabilities[capability]
        resolution = cell["resolution"]
        if capability == "vad" and request.vad == "silero-vad":
            resolution = "add_on"
            backend = "silero-vad"
            quality = "measured"
            note = stacks.get_stack("qwen-1.7b").capabilities["vad"]["plan_note"]
        else:
            backend = cell.get("package")
            quality = cell["quality"]
            note = cell.get("plan_note")
        entry: dict[str, Any] = {
            "satisfaction": "derived" if resolution == "add_on" else "native"
        }
        if resolution == "native_stage":
            entry["stage"] = cell["stage"]
        if resolution == "add_on":
            entry["backend"] = backend
        entry["evidence"] = {"interface": "verified", "quality": quality}
        if note:
            entry["note"] = note
        found[capability] = entry
    return found


def _ordered_packages(
    request: ResolvedRequest,
    roles: dict[str, dict[str, Any]],
) -> list[env.Package]:
    role_backends = {
        role: value["backend"] for role, value in roles.items()
    }
    selected = env.packages_for(request.stack.id, role_backends)
    by_id = {package.id: package for package in selected}
    wanted: list[str] = []

    for backend_id in request.stack.base_roles.values():
        package_id = env.backends()[backend_id].package
        if package_id not in wanted:
            wanted.append(package_id)
    for capability in request.wants:
        cell = request.stack.capabilities[capability]
        package_id = cell.get("package")
        if capability == "vad" and request.vad == "silero-vad":
            package_id = "silero-vad"
        if package_id in by_id and package_id not in wanted:
            wanted.append(package_id)
        if package_id == "fluidaudio" and "speaker-diarization-coreml" in by_id \
                and "speaker-diarization-coreml" not in wanted:
            wanted.append("speaker-diarization-coreml")
    wanted.extend(sorted(set(by_id) - set(wanted)))
    return [by_id[identifier] for identifier in wanted]


def _package_record(
    package: env.Package,
    request: ResolvedRequest,
    provisioned: frozenset[str],
) -> dict[str, Any]:
    ready = package.auto_fetch or package.id in provisioned
    item: dict[str, Any] = {
        "package": package.id,
        "environment": package.environment,
        "kind": package.kind,
    }
    if package.requires_tool:
        item["requires_tool"] = list(package.requires_tool)
    item.update({"bytes": package.bytes, "provisioned": ready})
    if package.auto_fetch:
        item.update({
            "auto_fetch": True,
            "note": "hash-pinned single file; fetched on first use, so it never returns exit 3",
        })
    if package.id == "firered-asr2s" and "lid" in request.wants:
        item["includes_lid_weights"] = True
    return item


def build_plan(
    request: ResolvedRequest,
    metadata: InputMetadata,
    *,
    provisioned_packages: Iterable[str] = (),
) -> Plan:
    """Resolve roles, packages, and output shape without loading a backend."""
    ready = frozenset(provisioned_packages)
    roles = _resolved_roles(request)
    selected_packages = _ordered_packages(request, roles)
    package_records = tuple(
        _package_record(package, request, ready) for package in selected_packages
    )
    unready = [
        package for package, record in zip(selected_packages, package_records)
        if not record["provisioned"]
    ]
    total_known = sum(package.bytes or 0 for package in unready)
    unsized = tuple(package.id for package in unready if package.bytes is None)

    environments: list[str] = []
    for role in roles.values():
        environment = role.get("environment")
        if environment and environment != "core" and environment not in environments:
            environments.append(environment)
    execution = {
        "stage_order": list(roles),
        "residency": request.stack.execution["residency"],
        "environments_spanned": environments,
        "note": (
            request.stack.execution.get("lid_note", request.stack.execution["note"])
            if "lid" in request.wants
            else request.stack.execution["note"]
        ),
    }

    sample_abstention_reason = None
    diarization_cell = request.stack.capabilities["diarization"]
    if "diarization" in request.wants and diarization_cell["resolution"] == "add_on":
        sample_abstention_reason = "raw_fragment"
    elif "overlapped_speech" in request.wants:
        sample_abstention_reason = "overlap"

    return Plan(
        stack=request.stack.id,
        source=metadata.source,
        requested_capabilities=request.wants,
        roles=roles,
        execution=execution,
        capabilities=_capability_results(request),
        packages=package_records,
        total_known_download_bytes=total_known,
        unsized_packages=unsized,
        warnings=tuple(copy.deepcopy(request.stack.warnings)),
        sample_abstention_reason=sample_abstention_reason,
    )
