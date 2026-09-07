"""Public CLI regression fixtures: timestamp origins, codec padding and audible landmarks."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import signal

from audio_cli.cli import main
from audio_cli.media import decode_audio, hash_file, write_float_wav
from audio_cli.media.ffmpeg import _run
from audio_cli.profiles import STAGE_ORDER


class NoSpeech:
    model_version = "synthetic-timing-fixture"

    def detect(self, samples, sample_rate, **kwargs):
        return []


def cli(capsys, *args):
    assert main(list(map(str, args))) == 0
    return json.loads(capsys.readouterr().out)


def make_source(tmp_path, case):
    """Distinct chirps near both ends and the middle expose shift, truncation, and drift."""
    rate = 48000
    samples = np.zeros((round(2.137 * rate), 1), dtype=np.float32)
    for start, frequency in [(0.08, 430), (0.83, 870), (1.87, 1430)]:
        count = round(0.12 * rate)
        t = np.arange(count) / rate
        chirp = 0.06 * signal.chirp(t, frequency, 0.12, frequency * 2) * np.hanning(count)
        first = round(start * rate)
        samples[first : first + count, 0] = chirp
    wave = tmp_path / "landmarks.wav"
    write_float_wav(wave, samples, rate)
    if case == "mp3-delay":
        source = tmp_path / "source.mp3"
        _run(["ffmpeg", "-v", "error", "-i", str(wave), "-c:a", "libmp3lame", str(source)])
        return source, ".mp3"
    source = tmp_path / f"source{'.mkv' if case == 'different-origins' else '.mp4'}"
    offset = "0.24" if case == "different-origins" else "0.044"
    _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=32x32:rate=25:duration=2.4",
            "-itsoffset",
            offset,
            "-i",
            str(wave),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "mpeg4",
            "-bf",
            "0",
            "-c:a",
            "aac",
            str(source),
        ]
    )
    return source, source.suffix


def landmark_lag(original, delivered, rate, start):
    first, last = round((start - 0.035) * rate), round((start + 0.16) * rate)
    before, after = original[first:last, 0], delivered[first:last, 0]
    correlation = signal.correlate(after, before)
    index = int(np.argmax(correlation))
    similarity = correlation[index] / np.sqrt(np.sum(before**2) * np.sum(after**2))
    assert similarity > 0.95, "a landmark was lost or altered beyond this fixture's codec tolerance"
    return signal.correlation_lags(len(after), len(before))[index]


def video_frames(path):
    data = _run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-f",
            "framemd5",
            "-",
        ]
    ).stdout.decode()
    return [line.split(",")[-1].strip() for line in data.splitlines() if not line.startswith("#")]


@pytest.mark.parametrize("case", ["aac-nonzero-start", "mp3-delay", "different-origins"])
def test_cli_duration_evidence_does_not_overclaim_content_or_av_sync(
    tmp_path, monkeypatch, capsys, case
):
    monkeypatch.setattr("audio_cli.cli.SileroOnnxVad", lambda: NoSpeech())
    source, extension = make_source(tmp_path, case)
    original_hash = hash_file(source)
    before = cli(capsys, "inspect", source)
    plan = cli(capsys, "transcribe", "plan", "--input", source, "--stack", "firered")
    assert plan["sample_output"]["source"]["duration_basis"] == before["source"]["duration_basis"]
    output = tmp_path / f"delivered{extension}"
    args = ["enhance", source, "--profile", "product-demo"]
    for stage in STAGE_ORDER:
        if case == "aac-nonzero-start" and stage == "program-loudness":
            continue
        args.extend(["--skip", stage])
    dry = cli(capsys, *args, "--dry-run")
    assert "timeline_preserved" not in dry
    assert dry["timeline_verification"]["status"] == "not_run"
    report = cli(capsys, *args, "-o", output)
    delivered = cli(capsys, "inspect", output)
    assert hash_file(source) == original_hash
    assert report == json.loads((tmp_path / f"delivered{extension}.report.json").read_text())
    assert report["source"] == before["source"]
    assert report["output"] == delivered["source"]
    assert report["timeline_preserved"] is True
    assert report["timeline_verification"] == {
        "status": "pass",
        "scope": "decoded_audio_duration_only",
        "tolerance_ms": 50,
        "content_alignment": {"status": "abstained", "reason": "not_measured"},
        "av_sync": {"status": "abstained", "reason": "not_measured"},
    }
    original, rate = decode_audio(source)
    final, final_rate = decode_audio(output)
    assert rate == final_rate == 48000
    assert report["source"]["decoded_audio"]["sample_count"] == len(original)
    assert report["output"]["decoded_audio"]["sample_count"] == len(final)
    assert report["measurements"]["after"]["duration_basis"] == "decoded_pcm"
    assert report["measurements"]["after"]["duration_delta_ms"] == round(
        (len(final) - len(original)) / 48, 3
    )
    for start in (0.08, 0.83, 1.87):
        assert abs(landmark_lag(original, final, rate, start)) <= 1
    # A same-duration shifted control must fail the landmark test, even though the gate passes.
    shifted = np.roll(final, 480, axis=0)
    assert abs(landmark_lag(original, shifted, rate, 0.83)) >= 470
    assert before["source"]["audio_start_seconds"] > 0
    if case == "mp3-delay":
        packets = json.loads(
            _run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_packets",
                    "-of",
                    "json",
                    str(source),
                ]
            ).stdout
        )["packets"]
        sides = [side for packet in packets for side in packet.get("side_data_list", [])]
        assert any(side.get("skip_samples", 0) > 0 for side in sides)
        assert any(side.get("discard_padding", 0) > 0 for side in sides)
    else:
        assert abs(before["source"]["duration_seconds"] - len(original) / rate) > 0.001
    if case != "mp3-delay":
        assert video_frames(source) == video_frames(output)
        source_video = report["source"]["timing"]["video"][0]
        output_video = report["output"]["timing"]["video"][0]
        assert source_video["primary_audio_start_minus_video_start_seconds"] > 0
        assert (
            abs(
                source_video["primary_audio_start_minus_video_start_seconds"]
                - output_video["primary_audio_start_minus_video_start_seconds"]
            )
            > 0.001
        )


def test_cli_rejects_shortened_decoded_audio_before_publication(tmp_path, monkeypatch, capsys):
    from audio_cli.pipeline import publication

    monkeypatch.setattr("audio_cli.cli.SileroOnnxVad", lambda: NoSpeech())
    source, _ = make_source(tmp_path, "mp3-delay")
    source_hash = hash_file(source)
    real_encode = publication.encode_output

    def shortened(original, enhanced, output, **kwargs):
        samples, rate = decode_audio(enhanced)
        shortened_wav = tmp_path / "shortened.wav"
        write_float_wav(shortened_wav, samples[:-4800], rate)
        real_encode(original, shortened_wav, output, **kwargs)

    monkeypatch.setattr(publication, "encode_output", shortened)
    target = tmp_path / "rejected.wav"
    args = ["enhance", str(source), "--profile", "product-demo", "-o", str(target)]
    for stage in STAGE_ORDER:
        args.extend(["--skip", stage])
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "Decoded audio duration verification failed" in captured.err
    assert "-100.0 ms" in captured.err
    assert not target.exists()
    assert not (tmp_path / "rejected.wav.report.json").exists()
    assert hash_file(source) == source_hash
