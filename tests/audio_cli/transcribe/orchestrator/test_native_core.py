from __future__ import annotations

from pathlib import Path

from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    _REAL_INSPECT_CHECKOUT,
    InputMetadata,
    StageOutcome,
    _ready_multi_package,
    _runtime,
    audio_paths,
    env,
    orchestrator,
    pkg,
    pytest,
    refusals,
    resolve_request,
    subprocess,
    wave,
)
from tests.audio_cli.transcribe.orchestrator.transcribe_native_test_support import (
    trusted_checkout_probe as trusted_checkout_probe,
)


def test_checkout_probe_reads_live_head_tracked_and_untracked_state(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=checkout,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        return completed.stdout.strip()

    git("init", "--quiet")
    (checkout / ".gitignore").write_text("ignored.tmp\n__pycache__/\n", encoding="utf-8")
    (checkout / "tracked.py").write_text("original\n", encoding="utf-8")
    git("add", ".gitignore", "tracked.py")
    git(
        "-c",
        "user.name=Audio Test",
        "-c",
        "user.email=audio@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    git("config", "core.abbrev", "12")
    git("config", "diff.noprefix", "true")
    git("config", "diff.interHunkContext", "100")
    git("config", "diff.suppressBlankEmpty", "true")

    clean = _REAL_INSPECT_CHECKOUT(checkout)
    assert clean.head == git("rev-parse", "HEAD")
    assert clean.modified == ()
    assert clean.untracked == ()

    (checkout / "tracked.py").write_text("changed\n", encoding="utf-8")
    (checkout / "rogue.py").write_text("rogue\n", encoding="utf-8")
    (checkout / "ignored.tmp").write_text("ignored\n", encoding="utf-8")
    bytecode = checkout / "__pycache__" / "tracked.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"forged bytecode")
    changed = _REAL_INSPECT_CHECKOUT(checkout)
    assert changed.modified == ("tracked.py",)
    assert changed.untracked == (
        "__pycache__/tracked.cpython-312.pyc",
        "ignored.tmp",
        "rogue.py",
    )


@pytest.mark.parametrize(
    "canonical_bytes",
    [b"", b"not a wave", bytes.fromhex("52494646a8eddd645741564545e52057cfbfd3cf")],
)
def test_firered_malformed_canonical_wav_is_a_typed_decode_failure(
    tmp_path: Path,
    monkeypatch,
    canonical_bytes: bytes,
) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "field.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants=(),
    )
    metadata = InputMetadata(str(source), 5.0, "wav", 48_000, 1)

    class MalformedDecodeTransport:
        def decode(self, _source, target):
            target.write_bytes(canonical_bytes)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
    }
    with pytest.raises(refusals.Refusal) as raised:
        orchestrator.run(
            request,
            metadata,
            registry=registry,
            transport=MalformedDecodeTransport(),
        )

    assert raised.value.exit_code == 1
    assert raised.value.payload["code"] == "backend_failed"
    assert raised.value.payload["role"] == "decode"
    assert raised.value.payload["backend"] == "ffmpeg"
    assert isinstance(raised.value.payload["detail"], str)


def _ready_single_package(tmp_path: Path, package_id: str) -> dict:
    package = env.packages()[package_id]
    target = (
        tmp_path
        / "hub"
        / f"models--{package.source['repo'].replace('/', '--')}"
        / "snapshots"
        / package.source["revision"]
    )
    target.mkdir(parents=True, exist_ok=True)
    for pattern in package.source.get("allow_patterns", ()):
        marker = target / (f"{pattern[:-3]}/model.mil" if pattern.endswith("/**") else pattern)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"")
    return {
        "state": "ready",
        "materialized": {
            "path": str(target),
            "revision": package.source["revision"],
            "bytes": 0,
        },
    }


def _ready_fluidaudio(tmp_path: Path) -> dict:
    package = env.packages()["fluidaudio"]
    target = audio_paths.checkout_dir(package.environment, package.id)
    target.mkdir(parents=True, exist_ok=True)
    product = target / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_text("#!/bin/sh\n", encoding="utf-8")
    product.chmod(0o755)
    patched = target / "Sources" / "FluidAudioCLI" / "Commands" / "ProcessCommand.swift"
    patched.parent.mkdir(parents=True)
    patched.write_bytes(b"patched\n")
    patch_names, _modified, patch_digests = pkg.checkout_patch_expectation(package)
    return {
        "state": "ready",
        "materialized": {
            "path": str(target),
            "revision": env.packages()["fluidaudio"].source["commit"],
            "built": True,
            "product_runs": True,
            "product_path": product.relative_to(target).as_posix(),
            "product_sha256": pkg.sha256_file(product),
            "patches_applied": list(patch_names),
            "patched_file_digests": patch_digests,
        },
    }


@pytest.mark.parametrize(
    ("stack_id", "runtime", "package_id", "vad"),
    [
        ("firered", "torch-firered", "firered-asr2s", "silero-vad"),
        ("vibevoice", "torch-vibevoice", "vibevoice-asr-7b", None),
    ],
)
def test_native_silero_bounds_are_typed_before_model_work(
    tmp_path: Path,
    monkeypatch,
    stack_id: str,
    runtime: str,
    package_id: str,
    vad: str | None,
) -> None:
    _runtime(tmp_path, monkeypatch, runtime)
    source = tmp_path / f"{stack_id}-invalid-vad.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id=stack_id,
        input_path=source,
        wants="vad",
        vad=vad,
    )
    metadata = InputMetadata(str(source), 2.0, "wav", 48_000, 1)

    class Detector:
        def detect(self, _samples, _rate, **_config):
            return [{"start": 0.25, "end": 3.0}]

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 32_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1)

        def firered(self, **_kwargs):
            raise AssertionError("FireRed must not receive invalid Silero bounds")

        def vibevoice(self, **_kwargs):
            raise AssertionError("VibeVoice must not run after invalid Silero bounds")

    with pytest.raises(refusals.Refusal) as caught:
        orchestrator.run(
            request,
            metadata,
            registry={
                "environments": {runtime: {"state": "ready"}},
                "packages": {
                    package_id: _ready_multi_package(tmp_path, package_id),
                },
            },
            transport=Transport(),
            vad_detector=Detector(),
        )

    assert caught.value.exit_code == 1
    assert caught.value.payload["code"] == "backend_failed"
    assert caught.value.payload["role"] == "vad"
    assert caught.value.payload["backend"] == "silero-vad"
    assert "exceeds canonical source duration" in caught.value.payload["detail"]


def test_firered_run_crosses_transport_and_adapter_boundaries(tmp_path: Path, monkeypatch) -> None:
    _runtime(tmp_path, monkeypatch, "torch-firered")
    source = tmp_path / "field.wav"
    source.write_bytes(b"source")
    request = resolve_request(
        stack_id="firered",
        input_path=source,
        wants="verbatim,word_timestamps,vad,segment_timestamps,lid",
    )
    metadata = InputMetadata(str(source), 5.0, "wav", 48_000, 1)

    class Transport:
        def decode(self, _source, target):
            with wave.open(str(target), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16_000)
                handle.writeframes(b"\0\0" * 80_000)
            return StageOutcome("decode", "ffmpeg", {}, 0.1, peak_rss_bytes=10)

        def firered(self, **kwargs):
            assert kwargs["lid_enabled"] is True
            assert kwargs["vad_regions"] is None
            assert kwargs["asr_config"]["beam_size"] == 3
            assert set(kwargs["models"]) == {"vad", "lid", "asr", "punctuator"}
            return StageOutcome(
                "firered_process",
                "firered-asr2s",
                {
                    "complete": True,
                    "result": {
                        "sentences": [
                            {
                                "start_ms": 100,
                                "end_ms": 900,
                                "text": "Hello!",
                                "lang": "en",
                                "lang_confidence": 0.9,
                            }
                        ],
                        "words": [
                            {
                                "start_ms": 120,
                                "end_ms": 800,
                                "text": "hello",
                            }
                        ],
                        "vad_segments_ms": [[100, 1000]],
                    },
                    "regions": [
                        {
                            "region_id": "vad_0",
                            "start": 0.1,
                            "end": 1.0,
                            "processed": True,
                        }
                    ],
                },
                30.0,
                peak_rss_bytes=100,
                wall_seconds_by_stage={
                    "vad": 0.2,
                    "lid": 0.3,
                    "asr": 2.0,
                    "punctuator": 0.5,
                },
            )

    registry = {
        "environments": {"torch-firered": {"state": "ready"}},
        "packages": {"firered-asr2s": _ready_multi_package(tmp_path, "firered-asr2s")},
    }
    payload = orchestrator.run(request, metadata, registry=registry, transport=Transport()).payload
    assert payload["segments"] == [
        {
            "segment_id": "seg_0",
            "text": "Hello!",
            "start": 0.1,
            "end": 0.9,
            "words": [
                {
                    "word_id": "w_0",
                    "text": "hello",
                    "start": 0.12,
                    "end": 0.8,
                }
            ],
        }
    ]
    assert payload["vad_regions"] == [{"start": 0.1, "end": 1.0}]
    assert payload["lid_regions"] == [
        {
            "start": 0.1,
            "end": 1.0,
            "language": "en",
            "confidence": 0.9,
        }
    ]
    observed = payload["provenance"]["observed"]
    assert observed["stage_wall_seconds"] == {
        "decode": 0.1,
        "vad": 0.2,
        "lid": 0.3,
        "asr": 2.0,
        "punctuator": 0.5,
        "firered_process_overhead": 27.0,
    }
    assert observed["total_wall_seconds"] == 30.1
    assert observed["peak_rss_bytes_by_stage"] == {
        "decode": 10,
        "firered_process": 100,
    }
    assert observed["punctuation_invariant_checked"] is True
