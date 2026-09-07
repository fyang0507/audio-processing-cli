"""Both enhancement commands resolve shared VAD configuration before inference.

Small controlled model bytes exercise real hashing and cache publication; only ONNX loading
and audio processing are doubled. Fresh-agent acceptance covers the real pinned model/media.
"""

import hashlib
import io
import json
from pathlib import Path

import pytest

from audio_cli import cli, vad


@pytest.fixture(params=["inspect", "enhance"])
def command(request):
    return request.param


def arguments(command, source):
    args = [command, str(source)]
    if command == "enhance":
        args += ["--profile", "product-demo", "--dry-run"]
    return args


@pytest.fixture
def processing(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original media is never opened by these controlled processing doubles")
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUDIO_PROCESSING_MODEL_CACHE", str(cache))
    monkeypatch.delenv("AUDIO_PROCESSING_VAD_MODEL", raising=False)
    monkeypatch.setattr(cli, "probe_media", lambda *args: {})
    monkeypatch.setattr(cli, "media_summary", lambda *args: {"duration_seconds": 2})
    loaded = []

    def session(path, **kwargs):
        loaded.append(Path(path))
        return object()

    def inspect(source, *, profile, detector):
        return {"model_path": str(detector.model_path)}

    class Pipeline:
        def __init__(self, profile, *, detector, **kwargs):
            self.detector = detector

        def run(self, *args, **kwargs):
            return {"model_path": str(self.detector.model_path)}

    monkeypatch.setattr(vad.ort, "InferenceSession", session)
    monkeypatch.setattr(cli, "inspect_source", inspect)
    monkeypatch.setattr(cli, "EnhancementPipeline", Pipeline)
    return source, cache, loaded


@pytest.mark.parametrize("location", ["managed", "override", "bootstrap"])
def test_shared_configuration_reaches_both_commands(
    command, location, processing, monkeypatch, capsys
):
    source, cache, loaded = processing
    content = b"controlled model content matching the test pin"
    monkeypatch.setattr(vad, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    target = cache / "models" / vad.MODEL_FILENAME
    downloads = []

    def download(request, *, timeout):
        assert location == "bootstrap", "a populated model must not be downloaded"
        downloads.append(request.full_url)
        return io.BytesIO(content)

    monkeypatch.setattr(vad.urllib.request, "urlopen", download)
    if location == "override":
        target = source.parent / "prepopulated.onnx"
        monkeypatch.setenv("AUDIO_PROCESSING_VAD_MODEL", str(target))
    if location != "bootstrap":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    assert cli.main(arguments(command, source)) == 0
    assert json.loads(capsys.readouterr().out) == {"model_path": str(target)}
    assert loaded == [target]
    assert target.read_bytes() == content
    assert downloads == ([vad.MODEL_URL] if location == "bootstrap" else [])
    if location == "override":
        assert not cache.exists(), "the environment override must precede cache resolution"


@pytest.mark.parametrize("location", ["override", "bootstrap"])
def test_mismatched_model_is_refused_before_loading(
    command, location, processing, monkeypatch, capsys
):
    source, cache, loaded = processing
    content = b"not the pinned Silero model"
    if location == "override":
        override = source.parent / "wrong.onnx"
        override.write_bytes(content)
        monkeypatch.setenv("AUDIO_PROCESSING_VAD_MODEL", str(override))
        monkeypatch.setattr(
            vad.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no override fallback")
        )
    else:
        monkeypatch.setattr(vad.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))

    assert cli.main(arguments(command, source)) == 2
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    error = json.loads(stderr)["error"]
    assert error["type"] == "VadError"
    assert "checksum mismatch" in error["message"]
    assert loaded == []
    assert not (cache / "models" / vad.MODEL_FILENAME).exists()
    if location == "override":
        assert override.read_bytes() == content
        assert not cache.exists()


@pytest.mark.parametrize("form", ["separate", "equals"])
def test_removed_vad_option_is_an_argument_error(command, form, monkeypatch, capsys):
    monkeypatch.setattr(cli, "SileroOnnxVad", lambda: pytest.fail("argument error precedes VAD"))
    option = ["--vad-model", "model.onnx"] if form == "separate" else ["--vad-model=model.onnx"]
    with pytest.raises(SystemExit) as raised:
        cli.main([*arguments(command, "source.wav"), *option])
    assert raised.value.code == 2
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert "unrecognized arguments: --vad-model" in stderr


def test_help_does_not_advertise_removed_option(command, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main([command, "--help"])
    assert raised.value.code == 0
    assert "--vad-model" not in capsys.readouterr().out
