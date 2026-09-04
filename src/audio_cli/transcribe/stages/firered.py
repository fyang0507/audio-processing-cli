"""Execute one co-resident FireRedASR2S request without importing core CLI code."""

from __future__ import annotations

import functools
import json
import os
import re
import resource
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

try:
    from ._firered_protocol import (
        _format_region,
        _mapping,
        _number,
        _positive_int,
        _probability,
        _public_region_bounds,
        _range_bounds,
        _region_records,
        _result_array,
        _validate_region_semantics,
    )
except ImportError:  # Executed directly by the isolated model environment.
    from _firered_protocol import (
        _format_region,
        _mapping,
        _number,
        _positive_int,
        _probability,
        _public_region_bounds,
        _range_bounds,
        _region_records,
        _result_array,
        _validate_region_semantics,
    )

_AED_PARAMETERS = (
    "beam_size",
    "nbest",
    "decode_max_len",
    "softmax_smoothing",
    "aed_length_penalty",
    "eos_penalty",
)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _json_default(value: object) -> object:
    """Convert NumPy-like scalar values without taking a dependency on NumPy here."""

    item = getattr(value, "item", None)
    if callable(item):
        return item()
    values = getattr(value, "tolist", None)
    if callable(values):
        return values()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")




def _timed(
    name: str, function: Callable[..., Any], stage_time: dict[str, float]
) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            stage_time[name] += time.perf_counter() - started

    return wrapper




def _run_pipeline(
    system: Any,
    wav: Any,
    sample_rate: int,
    regions: list[dict[str, Any]],
    *,
    asr_batch_size: int,
    punc_batch_size: int,
    lid_enabled: bool,
) -> tuple[dict[str, Any], int, Exception | None]:
    """Run the pinned phases with a recoverable chronological unit ledger.

    This mirrors ``fireredasr2system.py:59-124``: ASR/LID batch the original VAD
    regions first, blank ASR results are filtered, and only then does punctuation
    batch the single global post-filter stream.  Keeping those phase boundaries is
    important—a blank in one ASR batch must not reset punctuation batching.
    """

    entries: list[dict[str, Any]] = []
    completed_regions = 0
    failure: Exception | None = None
    for offset in range(0, len(regions), asr_batch_size):
        batch = regions[offset:offset + asr_batch_size]
        uttids = [
            f"audio_s{int(float(item['start']) * 1000)}_e"
            f"{int(float(item['end']) * 1000)}"
            for item in batch
        ]
        batch_wav = [
            (
                sample_rate,
                wav[
                    int(float(item["start"]) * sample_rate):
                    int(float(item["end"]) * sample_rate)
                ],
            )
            for item in batch
        ]
        try:
            asr_results = _result_array(
                system.asr.transcribe(uttids, batch_wav),
                "FireRed ASR batch",
                len(batch),
            )
            if lid_enabled:
                lid_results: list[Mapping[str, Any] | None] = list(_result_array(
                    system.lid.process(uttids, batch_wav),
                    "FireRed LID batch",
                    len(batch),
                ))
            else:
                lid_results = [None] * len(batch)
        except Exception as exc:  # noqa: BLE001 - retain the earlier unit prefix
            failure = exc
            break

        accepted = 0
        for local_index, (region, uttid, asr_result, lid_result) in enumerate(
            zip(batch, uttids, asr_results, lid_results, strict=True)
        ):
            try:
                if asr_result.get("uttid") != uttid:
                    raise ValueError("FireRed ASR returned an unexpected utterance id")
                if lid_result is not None and lid_result.get("uttid") != uttid:
                    raise ValueError("FireRed LID returned an unexpected utterance id")
                if lid_enabled:
                    if lid_result is None:
                        raise ValueError("FireRed LID omitted a requested region result")
                    language = lid_result.get("lang")
                    if not isinstance(language, str) or not language:
                        # FireRedLID reports feature-extraction failures as data
                        # with ``lang: ""`` (fireredlid/lid.py:62-66).
                        raise ValueError(
                            "FireRed LID returned an empty region language label"
                        )
                    try:
                        _probability(
                            lid_result.get("confidence"), "FireRed LID confidence"
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "FireRed LID confidence must be finite and between 0 and 1"
                        ) from exc
                text = asr_result.get("text")
                if not isinstance(text, str):
                    raise TypeError("FireRed ASR text must be a string")
                if not text.strip() or re.search(r"(<blank>)|(<sil>)", text):
                    if lid_enabled:
                        # Upstream filters the paired LID entry but retains this
                        # VAD region. That raw shape cannot satisfy the public
                        # one-label-per-VAD invariant, so stop at the last
                        # publishable per-region prefix instead.
                        raise ValueError(
                            "FireRed cannot publish region LID for a blank ASR region"
                        )
                else:
                    entries.append({
                        "region_index": offset + local_index,
                        "region": region,
                        "asr": asr_result,
                        "lid": lid_result,
                    })
                accepted += 1
            except Exception as exc:  # noqa: BLE001 - retain valid earlier batch items
                failure = exc
                break
        completed_regions += accepted
        if accepted != len(batch):
            break

    punctuated: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    first_unpunctuated_region: int | None = None
    for offset in range(0, len(entries), punc_batch_size):
        batch = entries[offset:offset + punc_batch_size]
        try:
            # The pinned runner uses ``get("timestamp", [])`` when timestamp
            # output is enabled (fireredasr2system.py:102-114), then punctuates
            # the single filtered ASR stream rather than each ASR batch.
            timestamps = [entry["asr"].get("timestamp", []) for entry in batch]
            uttids = [entry["asr"]["uttid"] for entry in batch]
            punc_results = _result_array(
                system.punc.process_with_timestamp(timestamps, uttids),
                "FireRed punctuation batch",
                len(batch),
            )
        except Exception as exc:  # noqa: BLE001 - retain earlier punctuation batches
            failure = failure or exc
            first_unpunctuated_region = int(batch[0]["region_index"])
            break
        accepted = 0
        for entry, punc_result in zip(batch, punc_results, strict=True):
            if punc_result.get("uttid") != entry["asr"].get("uttid"):
                failure = failure or ValueError(
                    "FireRed punctuation returned an unexpected utterance id"
                )
                first_unpunctuated_region = int(entry["region_index"])
                break
            punctuated.append((entry, punc_result))
            accepted += 1
        if accepted != len(batch):
            break

    prefix_count = completed_regions
    if first_unpunctuated_region is not None:
        prefix_count = min(prefix_count, first_unpunctuated_region)
    aggregate: dict[str, Any] = {
        "sentences": [],
        "words": [],
        "vad_segments_ms": [],
    }
    previous_sentence_end = -1.0
    previous_word_start = -1.0
    previous_word_end = -1.0
    for entry, punc_result in punctuated:
        region_index = int(entry["region_index"])
        if region_index >= prefix_count:
            break
        try:
            sentences, words = _format_region(entry, punc_result)
            (
                previous_sentence_end,
                previous_word_start,
                previous_word_end,
            ) = _validate_region_semantics(
                sentences,
                words,
                region=_mapping(entry["region"], "FireRed region"),
                previous_sentence_end=previous_sentence_end,
                previous_word_start=previous_word_start,
                previous_word_end=previous_word_end,
            )
        except Exception as exc:  # noqa: BLE001 - retain earlier formatted regions
            failure = failure or exc
            prefix_count = min(prefix_count, region_index)
            break
        aggregate["sentences"].extend(sentences)
        aggregate["words"].extend(words)
    aggregate["vad_segments_ms"] = [
        [int(float(item["start"]) * 1000), int(float(item["end"]) * 1000)]
        for item in regions[:prefix_count]
    ]
    if prefix_count < len(regions) and failure is None:
        failure = RuntimeError("FireRed pipeline stopped before every selected region")
    return aggregate, prefix_count, failure


def main() -> int:
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    aggregate: dict[str, Any] = {
        "sentences": [],
        "words": [],
        "vad_segments_ms": [],
    }
    output: dict[str, Any] = {
        "complete": False,
        "result": aggregate,
        "regions": [],
        "metrics": {},
    }
    stage_time: dict[str, float] = {"asr": 0.0, "punctuator": 0.0}
    captured_regions: list[dict[str, Any]] = []
    processed_count = 0
    try:
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"

        checkout = Path(request["checkout"])
        sys.path.insert(0, str(checkout))
        models = _mapping(request.get("models"), "models")
        asr_values = _mapping(request.get("asr_config"), "asr_config")
        punc_values = _mapping(request.get("punctuator_config"), "punctuator_config")
        lid_enabled = request.get("lid_enabled", False)
        if not isinstance(lid_enabled, bool):
            raise TypeError("lid_enabled must be a boolean")
        if asr_values.get("device") != "cpu" or asr_values.get("dtype") != "float32":
            raise ValueError("FireRed v1 supports only CPU float32 execution")
        if asr_values.get("return_timestamp") is not True:
            raise ValueError("FireRed word timing requires return_timestamp=true")
        missing = [name for name in _AED_PARAMETERS if name not in asr_values]
        if missing:
            raise ValueError(f"asr_config is missing AED parameters: {missing}")
        asr_batch_size = _positive_int(asr_values.get("batch_size"), "asr_config.batch_size")
        punc_batch_size = _positive_int(
            punc_values.get("batch_size"), "punctuator_config.batch_size"
        )
        required_models = {"asr", "punctuator"}
        if request.get("vad_regions") is None:
            required_models.add("vad")
        if lid_enabled:
            required_models.add("lid")
        absent_models = sorted(required_models - set(models))
        if absent_models:
            raise ValueError(f"models is missing required paths: {absent_models}")

        range_start, range_end = _range_bounds(request)
        # Transport uses ``None`` for native FireRedVAD and a list (including an
        # empty one) for a real external-VAD result.
        supplied_vad = request.get("vad_regions") is not None
        if supplied_vad:
            captured_regions.extend(_region_records(
                request["vad_regions"],
                range_start=range_start,
                range_end=range_end,
            ))

        print("firered stage: loading local co-resident models", file=sys.stderr, flush=True)
        from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
        from fireredasr2s.fireredasr2 import FireRedAsr2Config
        from fireredasr2s.fireredlid import FireRedLidConfig
        from fireredasr2s.fireredpunc import FireRedPuncConfig
        from fireredasr2s.fireredvad import FireRedVadConfig
        import soundfile as sf

        # These are the six AED decode values used by the recorded runner at
        # model_tests/benchmark/run_firered.py:360-370.  The stage takes them from
        # the executed plan rather than maintaining a second set of defaults.
        asr_config = FireRedAsr2Config(
            use_gpu=False,
            use_half=False,
            beam_size=asr_values["beam_size"],
            nbest=asr_values["nbest"],
            decode_max_len=asr_values["decode_max_len"],
            softmax_smoothing=asr_values["softmax_smoothing"],
            aed_length_penalty=asr_values["aed_length_penalty"],
            eos_penalty=asr_values["eos_penalty"],
            return_timestamp=True,
        )
        system_config = FireRedAsr2SystemConfig(
            # The pinned FireRedVAD Hub repository contains several products;
            # the measured pipeline loads its ``VAD`` subdirectory
            # (model_tests/benchmark/run_firered.py:253-258).
            vad_model_dir=(
                str(Path(str(models["vad"])) / "VAD") if not supplied_vad else ""
            ),
            lid_model_dir=str(models.get("lid", "")),
            asr_type="aed",
            asr_model_dir=str(models["asr"]),
            punc_model_dir=str(models["punctuator"]),
            vad_config=FireRedVadConfig(use_gpu=False),
            lid_config=FireRedLidConfig(use_gpu=False, use_half=False),
            asr_config=asr_config,
            punc_config=FireRedPuncConfig(use_gpu=False),
            asr_batch_size=asr_batch_size,
            punc_batch_size=punc_batch_size,
            # A supplied Silero region list is a real substitution: native
            # FireRedVAD is not loaded.
            enable_vad=not supplied_vad,
            enable_lid=lid_enabled,
            enable_punc=True,
        )
        system = FireRedAsr2System(system_config)
        if not lid_enabled and system.lid is not None:
            raise RuntimeError("FireRed loaded its LID model although LID was not requested")
        if lid_enabled and system.lid is None:
            raise RuntimeError("FireRed did not load its requested LID model")

        # Read the source exactly once before VAD, as the pinned runner does at
        # fireredasr2system.py:46-52.
        wav, sample_rate = sf.read(str(request["audio"]), dtype="int16")
        if sample_rate != 16000:
            raise ValueError("FireRed requires 16 kHz canonical audio")

        if supplied_vad:
            if system.vad is not None:
                raise RuntimeError("FireRed loaded native VAD despite supplied VAD regions")
        else:
            stage_time["vad"] = 0.0
            native_detect = system.vad.detect
            timed_detect = _timed("vad", native_detect, stage_time)
            vad_result, _probability = timed_detect(str(request["audio"]))
            vad_result_map = _mapping(vad_result, "FireRedVAD result")
            captured_regions.extend(_region_records(
                vad_result_map.get("timestamps"),
                range_start=range_start,
                range_end=range_end,
            ))

        system.asr.transcribe = _timed("asr", system.asr.transcribe, stage_time)
        if lid_enabled:
            stage_time["lid"] = 0.0
            system.lid.process = _timed("lid", system.lid.process, stage_time)
        system.punc.process_with_timestamp = _timed(
            "punctuator", system.punc.process_with_timestamp, stage_time
        )

        # The pinned phase boundaries are mirrored below so a later failure can
        # expose only an already completed chronological prefix without reloading
        # models or resetting global punctuation batches.
        print(
            "firered stage: processing canonical audio",
            file=sys.stderr,
            flush=True,
        )
        aggregate, processed_count, pipeline_failure = _run_pipeline(
            system,
            wav,
            sample_rate,
            captured_regions,
            asr_batch_size=asr_batch_size,
            punc_batch_size=punc_batch_size,
            lid_enabled=lid_enabled,
        )
        output["result"] = aggregate
        if pipeline_failure is not None:
            raise pipeline_failure
        output["complete"] = True
        output["regions"] = [
            {**item, "processed": True} for item in captured_regions
        ]
        code = 0
    except Exception as exc:  # noqa: BLE001 - the stage must always write an envelope
        output["regions"] = [
            {**item, "processed": index < processed_count}
            for index, item in enumerate(captured_regions)
        ]
        output["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        # Only formatted VAD-region units before the failure are published.  That
        # chronological prefix conforms to the same raw result schema as a full
        # run; without one there is no honest partial document.
        code = 4 if processed_count else 1

    output["metrics"] = {
        "wall_seconds": round(time.perf_counter() - started, 6),
        "stage_wall_seconds": {
            name: round(value, 6) for name, value in stage_time.items()
        },
        "peak_rss_bytes": _rss_bytes(),
    }
    result_path.write_text(
        json.dumps(output, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
