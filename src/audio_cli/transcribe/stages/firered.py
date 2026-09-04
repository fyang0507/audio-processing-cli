"""Execute one co-resident FireRedASR2S request without importing core CLI code."""

from __future__ import annotations

import functools
import json
import math
import os
import re
import resource
import sys
import time
import traceback
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


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


def _number(value: object, field: str) -> float:
    # FireRed can retain NumPy/Torch scalar timestamps in-process; the recorded
    # runner likewise serializes scalar ``item()`` values in run_firered.py:224-230.
    if not isinstance(value, (bool, int, float)):
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _probability(value: object, field: str) -> float:
    parsed = _number(value, field)
    if parsed > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return parsed


def _plain(text: str) -> str:
    return "".join(
        character.casefold()
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _range_bounds(request: Mapping[str, Any]) -> tuple[float, float | None]:
    start = _number(request.get("range_start", 0.0), "range_start")
    raw_end = request.get("range_end")
    end = None if raw_end is None else _number(raw_end, "range_end")
    if end is not None and end < start:
        raise ValueError("range_end must be greater than or equal to range_start")
    return start, end


def _public_region_bounds(start: float, end: float) -> tuple[float, float]:
    """Project raw VAD seconds onto FireRed's durable integer-ms timeline."""

    published_start = round(int(start * 1000) / 1000.0, 6)
    published_end = round(int(end * 1000) / 1000.0, 6)
    if published_end <= published_start:
        raise ValueError("VAD region is empty at FireRed's millisecond precision")
    return published_start, published_end


def _region_records(
    values: object, *, range_start: float, range_end: float | None
) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise TypeError("VAD regions must be an array")
    result: list[dict[str, Any]] = []
    previous_end = -1.0
    seen: set[str] = set()
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            start = _number(item.get("start"), f"vad_regions[{index}].start")
            end = _number(item.get("end"), f"vad_regions[{index}].end")
            identifier = item.get("region_id", item.get("unit_id", f"vad_{index}"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start = _number(item[0], f"vad_regions[{index}][0]")
            end = _number(item[1], f"vad_regions[{index}][1]")
            identifier = f"vad_{index}"
        else:
            raise TypeError(f"vad_regions[{index}] must be an object or [start, end]")
        if end <= start:
            raise ValueError(f"vad_regions[{index}] must satisfy start < end")
        if start < previous_end:
            raise ValueError("VAD regions must be chronological and non-overlapping")
        previous_end = end
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("VAD region ids must be unique non-empty strings")
        seen.add(identifier)
        # FireRed publishes integer-millisecond bounds. Select on that same
        # timeline so a continuation at the next published start cannot
        # reselect a predecessor whose raw end falls inside the same millisecond.
        published_start, published_end = _public_region_bounds(start, end)
        if published_end > range_start and (
            range_end is None or published_start < range_end
        ):
            result.append({"region_id": identifier, "start": start, "end": end})
    return result


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


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _result_array(value: object, field: str, expected: int) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != expected:
        raise TypeError(f"{field} must contain one object per requested region")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field}[{index}] must be an object")
        result.append(item)
    return result


def _format_region(
    entry: Mapping[str, Any], punc_result: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirror the pinned formatter at ``fireredasr2system.py:126-184``."""

    asr_result = _mapping(entry["asr"], "FireRed ASR result")
    lid_result = entry.get("lid")
    region = _mapping(entry["region"], "FireRed region")
    uttid = asr_result.get("uttid")
    if not isinstance(uttid, str) or punc_result.get("uttid") != uttid:
        raise ValueError("FireRed ASR and punctuation utterance ids must match")
    start_ms = int(_number(region.get("start"), "FireRed region start") * 1000)
    end_ms = int(_number(region.get("end"), "FireRed region end") * 1000)
    raw_sentences = punc_result.get("punc_sentences")
    if not isinstance(raw_sentences, list) or not raw_sentences:
        raise ValueError("FireRed punctuation must return at least one sentence")
    confidence = _probability(
        asr_result.get("confidence"), "FireRed ASR sentence confidence"
    )

    sentences: list[dict[str, Any]] = []
    for index, raw_sentence in enumerate(raw_sentences):
        sentence = _mapping(raw_sentence, f"FireRed punctuation sentence {index}")
        text = sentence.get("punc_text")
        if not isinstance(text, str):
            raise TypeError("FireRed punctuation sentence text must be a string")
        start_s = _number(sentence.get("start_s"), "punctuation sentence start_s")
        end_s = _number(sentence.get("end_s"), "punctuation sentence end_s")
        sentence_start = start_ms + int(start_s * 1000)
        sentence_end = start_ms + int(end_s * 1000)
        if index == 0:
            sentence_start = start_ms
        if index == len(raw_sentences) - 1:
            sentence_end = end_ms
        normalized = {
            "start_ms": sentence_start,
            "end_ms": sentence_end,
            "text": text,
            "asr_confidence": confidence,
            "lang": None,
            "lang_confidence": 0,
        }
        if lid_result is not None:
            lid = _mapping(lid_result, "FireRed LID result")
            normalized["lang"] = lid.get("lang")
            normalized["lang_confidence"] = _probability(
                lid.get("confidence"), "FireRed LID confidence"
            )
        sentences.append(normalized)

    timestamps = asr_result.get("timestamp")
    if not isinstance(timestamps, list):
        raise TypeError("FireRed ASR timestamps must be an array")
    words: list[dict[str, Any]] = []
    for index, item in enumerate(timestamps):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise TypeError(f"FireRed ASR timestamp {index} must be [text, start, end]")
        text, start, end = item
        if not isinstance(text, str):
            raise TypeError(f"FireRed ASR timestamp {index} text must be a string")
        start_s = _number(start, f"FireRed ASR timestamp {index} start")
        end_s = _number(end, f"FireRed ASR timestamp {index} end")
        words.append({
            "start_ms": int(start_s * 1000 + start_ms),
            "end_ms": int(end_s * 1000 + start_ms),
            "text": text,
        })
    return sentences, words


def _validate_region_semantics(
    sentences: list[dict[str, Any]],
    words: list[dict[str, Any]],
    *,
    region: Mapping[str, Any],
    previous_sentence_end: float,
    previous_word_start: float,
    previous_word_end: float,
) -> tuple[float, float, float]:
    """Validate one formatted unit before it can enter the salvageable prefix.

    The pinned formatter only zips stage arrays and copies punctuation text.  The
    core adapter later enforces the semantic contract, but waiting until then would
    make one malformed late region discard every earlier valid region.  Keep this
    stage-local mirror narrow: sentence/VAD membership and chronology, native-word
    chronology, and exact reproduction after punctuation removal.
    """

    # The pinned formatter publishes integer-millisecond truncation.  Validate
    # against those exact durable bounds rather than the higher-precision input
    # seconds, or a legitimate 0.2009 s region would reject its 200 ms sentence.
    region_start = int(_number(region.get("start"), "FireRed region start") * 1000)
    region_end = int(_number(region.get("end"), "FireRed region end") * 1000)
    word_cursor = 0
    for sentence_index, sentence in enumerate(sentences):
        field = f"FireRed punctuation sentence {sentence_index}"
        start = _number(sentence.get("start_ms"), f"{field} start_ms")
        end = _number(sentence.get("end_ms"), f"{field} end_ms")
        if end <= start:
            raise ValueError("FireRed sentence bounds must have positive duration")
        if start < previous_sentence_end:
            raise ValueError("FireRed sentences must be chronological and non-overlapping")
        if start < region_start or end > region_end:
            raise ValueError("FireRed sentence must belong to its VAD region")
        previous_sentence_end = end

        text = sentence.get("text")
        if not isinstance(text, str):
            raise TypeError(f"{field} text must be a string")
        target = _plain(text)
        if not target:
            raise ValueError(f"{field} text must contain a non-punctuation character")
        joined = ""
        while word_cursor < len(words) and len(joined) < len(target):
            word = words[word_cursor]
            word_field = f"FireRed ASR word {word_cursor}"
            word_text = word.get("text")
            if not isinstance(word_text, str):
                raise TypeError(f"{word_field} text must be a string")
            plain_word = _plain(word_text)
            if not plain_word:
                raise ValueError(
                    f"{word_field} text must contain a non-punctuation character"
                )
            word_start = _number(word.get("start_ms"), f"{word_field} start_ms")
            word_end = _number(word.get("end_ms"), f"{word_field} end_ms")
            if word_end <= word_start:
                raise ValueError("FireRed word bounds must have positive duration")
            if word_start < region_start or word_end > region_end:
                raise ValueError("FireRed word must stay within its VAD region")
            if word_start <= previous_word_start or word_start < previous_word_end:
                raise ValueError(
                    "FireRed words must be strictly chronological and non-overlapping"
                )
            candidate = joined + plain_word
            if not target.startswith(candidate):
                raise ValueError(
                    f"FireRed words do not reproduce {field} text after punctuation removal"
                )
            joined = candidate
            previous_word_start = word_start
            previous_word_end = word_end
            word_cursor += 1
        if joined != target:
            raise ValueError(
                f"FireRed words do not reproduce {field} text after punctuation removal"
            )

    if word_cursor != len(words):
        raise ValueError("FireRed words remain after the final punctuation sentence")
    return previous_sentence_end, previous_word_start, previous_word_end


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
