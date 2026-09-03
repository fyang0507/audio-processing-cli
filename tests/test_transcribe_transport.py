from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

from audio_cli import paths
from audio_cli.transcribe.stages import qwen as qwen_stage
from audio_cli.transcribe.transport import StageTransport, SubprocessRunner


class RecordingRunner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"wav")
        else:
            request = json.loads(Path(command[-2]).read_text())
            Path(command[-1]).write_text(json.dumps({
                "units": [{"unit_id": item["unit_id"], "processed": True, "text": "ok"}
                          for item in request["units"]],
                "metrics": {"wall_seconds": 1.25, "peak_rss_bytes": 40},
            }))
        return subprocess.CompletedProcess(command, 0, "", "")


def test_decode_writes_one_canonical_pcm_wav_without_targeting_source(tmp_path) -> None:
    source = tmp_path / "source.m4a"
    source.write_bytes(b"source")
    target = tmp_path / "canonical.wav"
    runner = RecordingRunner()
    StageTransport(runner).decode(source, target)
    command = runner.commands[0]
    assert command[-1] == str(target)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert source.read_bytes() == b"source"


def test_mlx_stage_uses_provisioned_interpreter_and_absolute_shipped_script(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(tmp_path / "root"))
    runner = RecordingRunner()
    outcome = StageTransport(runner).qwen(
        backend="qwen3-asr-0.6b-8bit",
        model=tmp_path / "model",
        audio=tmp_path / "canonical.wav",
        units=[{"unit_id": "u0", "start": 0.0, "end": 1.0}],
        language=None,
        max_tokens=16_384,
        batch_size=1,
        clear_cache_after_every_batch=True,
        directory=tmp_path,
    )
    command = runner.commands[0]
    assert command[0] == str(paths.env_python("mlx"))
    assert Path(command[1]).is_absolute()
    assert Path(command[1]).name == "qwen.py"
    assert Path(command[2]).name == "asr.request.json"
    assert Path(command[3]).name == "asr.result.json"
    assert outcome.peak_rss_bytes == 40
    request = json.loads((tmp_path / "asr.request.json").read_text(encoding="utf-8"))
    assert request["batch_size"] == 1
    assert request["clear_cache_after_every_batch"] is True


def test_environment_stage_scripts_do_not_import_core_package() -> None:
    stages = Path(__file__).parents[1] / "src/audio_cli/transcribe/stages"
    for path in (stages / "qwen.py", stages / "aligner.py"):
        source = path.read_text()
        assert "import audio_cli" not in source
        assert "from audio_cli" not in source


def test_stage_sources_pin_the_measured_private_api_and_aligner_language_rule() -> None:
    stages = Path(__file__).parents[1] / "src/audio_cli/transcribe/stages"
    qwen = (stages / "qwen.py").read_text()
    aligner = (stages / "aligner.py").read_text()
    assert "model._generate_chunks_batched" in qwen
    assert 'max_tokens=remaining' in qwen
    assert "mx.clear_cache()" in qwen
    assert 'CJK = re.compile(r"[一-鿿]")' in aligner
    assert '"Chinese" if CJK.search(text) else "English"' in aligner
    assert 'request.get("language")' not in aligner
    assert '"words": None' in aligner
    runner = (
        Path(__file__).parents[1]
        / "model_tests/benchmark/run_turn_attributed_mlx_asr.py"
    ).read_text(encoding="utf-8")
    assert "group_texts, group_generated, group_prompts, group_processed" in runner
    assert "texts, generated, prompts, processed = method(" in qwen


def test_missing_stage_script_preserves_dispatchable_role_and_backend() -> None:
    from audio_cli.transcribe.transport import StageFailure

    try:
        StageTransport._stage_script("does-not-exist", "asr", "qwen3-asr-0.6b-8bit")
    except StageFailure as exc:
        assert exc.role == "asr"
        assert exc.backend == "qwen3-asr-0.6b-8bit"
    else:
        raise AssertionError("missing stage script did not fail")


def test_qwen_stage_salvages_completed_units_after_mid_generation_error(
    tmp_path, monkeypatch
) -> None:
    class Model:
        calls = 0

        def parameters(self):
            return []

        def _generate_chunks_batched(self, clips, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom after first unit")
            return ["first"], [3], [2], [True]

    mx = types.ModuleType("mlx.core")
    mx.eval = lambda value: None
    mx.synchronize = lambda: None
    mx.clear_cache = lambda: None
    mx.get_peak_memory = lambda: 123
    mlx = types.ModuleType("mlx")
    mlx.core = mx
    audio_io = types.ModuleType("mlx_audio.audio_io")
    audio_io.read = lambda *args, **kwargs: (np.zeros((32_000, 1), dtype=np.float32), 16_000)
    stt_utils = types.ModuleType("mlx_audio.stt.utils")
    stt_utils.load_model = lambda *args, **kwargs: Model()
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **kwargs: object()
    for name, module in {
        "mlx": mlx,
        "mlx.core": mx,
        "mlx_audio": types.ModuleType("mlx_audio"),
        "mlx_audio.audio_io": audio_io,
        "mlx_audio.stt": types.ModuleType("mlx_audio.stt"),
        "mlx_audio.stt.utils": stt_utils,
        "mlx_lm": types.ModuleType("mlx_lm"),
        "mlx_lm.sample_utils": sample_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(json.dumps({
        "model": str(tmp_path / "model"),
        "audio": str(tmp_path / "audio.wav"),
        "units": [
            {"unit_id": "u0", "start": 0.0, "end": 1.0},
            {"unit_id": "u1", "start": 1.0, "end": 2.0},
        ],
        "language": None,
        "max_tokens": 100,
        "batch_size": 1,
        "clear_cache_after_every_batch": True,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["qwen.py", str(request_path), str(result_path)])
    assert qwen_stage.main() == 4
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["units"][0]["processed"] is True
    assert result["units"][1]["processed"] is False
    assert result["error"]["message"] == "boom after first unit"


def test_real_subprocess_runner_streams_stderr_and_samples_child_rss() -> None:
    progress = io.StringIO()
    completed = SubprocessRunner(progress).run([
        sys.executable,
        "-c",
        "import sys,time; value=bytearray(1000000); print('working', file=sys.stderr, flush=True); time.sleep(0.1)",
    ])
    assert completed.returncode == 0
    assert completed.stderr == "working\n"
    assert progress.getvalue() == "working\n"
    assert completed.peak_rss_bytes is not None
    assert completed.peak_rss_bytes > 0


def test_swift_stage_runs_built_product_offline_without_speaker_prior(tmp_path) -> None:
    checkout = tmp_path / "fluidaudio"
    product = checkout / ".build" / "arm64-apple-macosx" / "release" / "fluidaudiocli"
    product.parent.mkdir(parents=True)
    product.write_bytes(b"binary")

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
        audio=tmp_path / "canonical.wav",
        config={
            "threshold": 0.6, "step_ratio": 0.1,
            "min_segment_duration": 0.0, "batch_size": 32,
        },
        overlap=True,
        directory=tmp_path,
    )
    assert runner.command[0] == str(product.resolve())
    assert runner.command[1:3] == ["process", str(tmp_path / "canonical.wav")]
    assert runner.command[runner.command.index("--mode") + 1] == "offline"
    assert "--num-speakers" not in runner.command
    assert runner.command[-1] == "--overlapping-segments"
    assert not (tmp_path / "diarizer.request.json").exists()
    assert runner.command[runner.command.index("--batch-size") + 1] == "32"
