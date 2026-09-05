import subprocess
import sys

from audio_cli import dsp, pipeline, vad
from audio_cli.vad_contract import SpeechRegion, VadDetector


def test_vad_runtime_preserves_contract_import_identity() -> None:
    assert vad.SpeechRegion is SpeechRegion
    assert vad.VadDetector is VadDetector
    assert dsp.SpeechRegion is SpeechRegion
    assert pipeline.VadDetector is VadDetector


def test_importing_dsp_does_not_load_the_concrete_vad_runtime() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import audio_cli.dsp; "
                "assert 'audio_cli.vad' not in sys.modules; "
                "assert 'onnxruntime' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stderr == ""
