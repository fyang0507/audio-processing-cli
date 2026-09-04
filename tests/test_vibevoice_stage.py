from __future__ import annotations

from vibevoice_test_support import (
    Path,
    _fixture,
    _install_fake_vibevoice,
    _stage_request,
    json,
    sys,
    threading,
    vibevoice_stage,
)


def test_vibevoice_stage_uses_one_seeded_offline_whole_media_call(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    request_path, result_path, request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert output["generated_tokens"] == 3
    assert output["eos_observed"] is True
    assert output["hit_max_new_tokens"] is False
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1
    assert state["torch_seeds"] == [1234]
    assert state["mps_seeds"] == [1234]
    assert state["numpy_seeds"] == [1234]
    assert state["processor_load"] == (
        request["model"],
        {
            "language_model_pretrained_name": request["tokenizer"],
            "local_files_only": True,
        },
    )
    generate = state["generate_kwargs"]
    assert generate["max_new_tokens"] == 16384
    assert generate["do_sample"] is False
    assert generate["input_ids"] is not None
    assert state["processor_call"]["audio"] == [request["audio"]]
    assert state["mps_sample_calls"] >= 3
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        assert __import__("os").environ[name] == "1"


def test_vibevoice_stage_sampler_start_failure_is_nonfatal(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    state["sampler_join_calls"] = 0

    class UnstartableThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("synthetic sampler start failure")

        def is_alive(self):
            return False

        def join(self):
            state["sampler_join_calls"] = int(state["sampler_join_calls"]) + 1

    monkeypatch.setattr(vibevoice_stage.threading, "Thread", UnstartableThread)
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1
    assert state["sampler_join_calls"] == 0
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_sampler_join_failure_is_nonfatal(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    state["sampler_join_calls"] = 0

    class UnjoinableThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

        def join(self):
            state["sampler_join_calls"] = int(state["sampler_join_calls"]) + 1
            raise RuntimeError("synthetic sampler join failure")

    monkeypatch.setattr(vibevoice_stage.threading, "Thread", UnjoinableThread)
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1
    assert state["sampler_join_calls"] == 1
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_unexpected_sampler_stop_failure_keeps_result_envelope(
    tmp_path, monkeypatch
) -> None:
    segments = [{"start_time": 0, "end_time": 1, "speaker_id": 0, "text": "Hi"}]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text='assistant\n[{"Start":0,"End":1,"Speaker":0,"Content":"Hi"}]',
        segments=segments,
        generated_tokens=3,
        eos_positions=[2],
    )
    monkeypatch.setattr(vibevoice_stage._MpsHighWater, "start", lambda self: None)

    def fail_stop(_self):
        raise RuntimeError("synthetic sampler stop failure")

    monkeypatch.setattr(vibevoice_stage._MpsHighWater, "stop", fail_stop)
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 0
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["segments"] == segments
    assert "peak_mps_live_bytes" not in output["metrics"]
    assert state["generate_calls"] == 1


def test_vibevoice_stage_reports_the_recorded_generation_cap_as_exit_four(
    tmp_path, monkeypatch
) -> None:
    raw = _fixture("vibevoice_30m_raw_prefix_excerpt.json")["raw_text"]
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text=raw,
        segments=[],
        generated_tokens=16384,
        eos_positions=[],
    )
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 4
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["hit_max_new_tokens"] is True
    assert output["eos_observed"] is False
    assert output["raw_text"] == raw
    assert output["segments"] == []
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 1


def test_vibevoice_model_load_failure_has_no_partial_result(tmp_path, monkeypatch) -> None:
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text="",
        segments=[],
        generated_tokens=0,
        eos_positions=[],
        fail_load=True,
    )
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 1
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["error"]["type"] == "RuntimeError"
    assert "OOM" in output["error"]["message"]
    assert "raw_text" not in output
    assert "hit_max_new_tokens" not in output
    assert output["metrics"]["peak_mps_live_bytes"] == 30
    assert state["generate_calls"] == 0
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_omits_mps_peak_when_mps_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    state = _install_fake_vibevoice(
        monkeypatch,
        raw_text="",
        segments=[],
        generated_tokens=0,
        eos_positions=[],
        mps_available=False,
    )
    request_path, result_path, _request = _stage_request(tmp_path)
    monkeypatch.setattr(sys, "argv", ["vibevoice.py", str(request_path), str(result_path)])

    assert vibevoice_stage.main() == 1
    output = json.loads(result_path.read_text(encoding="utf-8"))
    assert output["error"]["type"] == "RuntimeError"
    assert "peak_mps_live_bytes" not in output["metrics"]
    assert state["mps_sample_calls"] == 0
    assert not any(
        thread.name == "vibevoice-mps-high-water" for thread in threading.enumerate()
    )


def test_vibevoice_stage_is_environment_owned_and_imports_no_core_package() -> None:
    source = Path(vibevoice_stage.__file__).read_text(encoding="utf-8")
    assert "import audio_cli" not in source
    assert "from audio_cli" not in source
