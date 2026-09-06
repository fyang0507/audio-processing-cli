"""FluidAudio launch-boundary tests for transcription stage transport."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from audio_cli import packages as pkg
from audio_cli.transcribe.transport import StageFailure, StageTransport


def test_swift_stage_runs_built_product_offline_without_speaker_prior(tmp_path) -> None:
    checkout = tmp_path / "fluidaudio"
    product = checkout / ".build" / "arm64-apple-macosx" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"binary")
    product.chmod(0o755)
    model = tmp_path / "snapshot"
    model.mkdir()

    class SwiftRunner:
        def __init__(self):
            self.command = None

        def run(self, command):
            self.command = command
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps({"segments": []}))
            return subprocess.CompletedProcess(command, 0, "", "")

    runner = SwiftRunner()
    StageTransport(runner).diarize(
        checkout=checkout,
        product="fluidaudiocli",
        product_path=product.relative_to(checkout).as_posix(),
        product_sha256=pkg.sha256_file(product),
        model=model,
        audio=tmp_path / "canonical.wav",
        config={
            "threshold": 0.6,
            "step_ratio": 0.1,
            "min_segment_duration": 0.0,
            "batch_size": 32,
        },
        overlap=True,
        directory=tmp_path,
    )
    assert runner.command[0] == str(product.resolve())
    assert runner.command[1:3] == ["process", str(tmp_path / "canonical.wav")]
    assert runner.command[runner.command.index("--mode") + 1] == "offline"
    model_root = Path(runner.command[runner.command.index("--model-dir") + 1])
    assert model_root.parent == tmp_path
    binding = model_root / "speaker-diarization"
    assert binding.is_symlink()
    assert binding.resolve() == model.resolve()
    assert "--num-speakers" not in runner.command
    assert runner.command[-1] == "--overlapping-segments"
    assert not (tmp_path / "diarizer.request.json").exists()
    assert runner.command[runner.command.index("--batch-size") + 1] == "32"


def test_swift_stage_recursion_is_a_typed_failure(tmp_path: Path) -> None:
    checkout = tmp_path / "fluidaudio"
    product = checkout / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"binary")
    product.chmod(0o755)
    model = tmp_path / "snapshot"
    model.mkdir()

    class DeepResultRunner:
        def run(self, command):
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                "[" * 10_000 + "0" + "]" * 10_000,
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(StageFailure, match="invalid result JSON"):
        StageTransport(DeepResultRunner()).diarize(
            checkout=checkout,
            product="fluidaudiocli",
            product_path=product.relative_to(checkout).as_posix(),
            product_sha256=pkg.sha256_file(product),
            model=model,
            audio=tmp_path / "canonical.wav",
            config={
                "threshold": 0.6,
                "step_ratio": 0.1,
                "min_segment_duration": 0.0,
                "batch_size": 32,
            },
            overlap=False,
            directory=tmp_path,
        )


def test_swift_stage_rejects_duplicate_raw_json_keys(tmp_path: Path) -> None:
    checkout = tmp_path / "fluidaudio"
    product = checkout / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"binary")
    product.chmod(0o755)
    model = tmp_path / "snapshot"
    model.mkdir()

    class DuplicateResultRunner:
        def run(self, command):
            output = Path(command[command.index("--output") + 1])
            output.write_text(
                '{"segments":[{"startTimeSeconds":0.0,'
                '"endTimeSeconds":1.0,"speakerId":"speaker_0"}],'
                '"segments":[]}',
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(StageFailure, match="duplicate JSON object key 'segments'"):
        StageTransport(DuplicateResultRunner()).diarize(
            checkout=checkout,
            product="fluidaudiocli",
            product_path=product.relative_to(checkout).as_posix(),
            product_sha256=pkg.sha256_file(product),
            model=model,
            audio=tmp_path / "canonical.wav",
            config={
                "threshold": 0.6,
                "step_ratio": 0.1,
                "min_segment_duration": 0.0,
                "batch_size": 32,
            },
            overlap=False,
            directory=tmp_path,
        )


def test_swift_stage_rechecks_product_containment_after_preflight(tmp_path) -> None:
    from audio_cli.packages import built_product_candidates

    checkout = tmp_path / "fluidaudio"
    product = checkout / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"trusted")
    product.chmod(0o755)
    assert built_product_candidates(checkout, "fluidaudiocli") == [product.resolve()]

    external = tmp_path / "external-product"
    external.write_bytes(b"untrusted")
    external.chmod(0o755)
    product.unlink()
    product.symlink_to(external)

    with pytest.raises(StageFailure, match="launch-boundary receipt check"):
        StageTransport._swift_product(
            checkout,
            "fluidaudiocli",
            product.relative_to(checkout).as_posix(),
            pkg.sha256_file(external),
        )


def test_swift_stage_rechecks_product_digest_after_preflight(tmp_path) -> None:
    checkout = tmp_path / "fluidaudio"
    product = checkout / ".build" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"trusted")
    product.chmod(0o755)
    relative = product.relative_to(checkout).as_posix()
    digest = pkg.sha256_file(product)
    product.write_bytes(b"replaced after preflight")
    product.chmod(0o755)

    with pytest.raises(StageFailure, match="sha256"):
        StageTransport._swift_product(checkout, "fluidaudiocli", relative, digest)
