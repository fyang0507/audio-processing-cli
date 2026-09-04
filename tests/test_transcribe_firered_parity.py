from __future__ import annotations

from typing import Any

from ast_test_support import stable_ast_source
from transcribe_firered_test_support import (
    _PINNED_FIRERED_PROCESS,
    ROOT,
    ast,
    deepcopy,
    firered_stage,
    hashlib,
    json,
    np,
    pytest,
    re,
    types,
)


@pytest.mark.parametrize("lid_enabled", [True, False])
def test_firered_phase_mirror_matches_pinned_upstream_process(
    lid_enabled: bool,
) -> None:
    parsed = ast.parse(_PINNED_FIRERED_PROCESS)
    function = parsed.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = ast.Module(body=function.body, type_ignores=[])
    digest = hashlib.sha256(stable_ast_source(body).encode()).hexdigest()
    assert digest == "d5ad569f4ee772f29ef5c55298b31ad2881c8c28c34d52f816acf2c26c223d3c"
    upstream_path = ROOT / "model_tests/firered/FireRedASR2S/fireredasr2s/fireredasr2system.py"
    if upstream_path.is_file():
        upstream_source = upstream_path.read_text(encoding="utf-8")
        assert hashlib.sha256(upstream_source.encode()).hexdigest() == (
            "8bfada7cfc7ff424c73484a94150bc6fa5030a05b7146db983cb7ac374ad1782"
        )
        upstream_tree = ast.parse(upstream_source)
        upstream_class = next(
            node
            for node in upstream_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FireRedAsr2System"
        )
        upstream_function = next(
            node
            for node in upstream_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "process"
        )
        assert ast.dump(
            ast.Module(body=upstream_function.body, type_ignores=[]),
            annotate_fields=True,
            include_attributes=False,
        ) == ast.dump(body, annotate_fields=True, include_attributes=False)

    wav = np.zeros(14 * 16_000, dtype=np.int16)
    sample_rate = 16_000
    blank_start = None if lid_enabled else 2000
    native_regions = [(float(index * 2), float(index * 2 + 1)) for index in range(7)]
    record: dict[str, Any] = {}

    def bounds(uttid: str) -> tuple[int, int]:
        start_token, end_token = uttid.split("_")[-2:]
        return int(start_token[1:]), int(end_token[1:])

    class Vad:
        def detect(self, audio: str) -> tuple[dict[str, Any], None]:
            record.setdefault("vad_calls", []).append(audio)
            return {"timestamps": native_regions}, None

    class Asr:
        def transcribe(
            self, uttids: list[str], batch_wav: list[tuple[int, Any]]
        ) -> list[dict[str, Any]]:
            assert len(uttids) == len(batch_wav)
            record.setdefault("asr_batch_sizes", []).append(len(uttids))
            results = []
            for uttid in uttids:
                start_ms, end_ms = bounds(uttid)
                text = "" if start_ms == blank_start else f"region{start_ms}"
                record.setdefault("asr_text", {})[uttid] = text
                results.append(
                    {
                        "uttid": uttid,
                        "text": text,
                        "confidence": 0.9,
                        "timestamp": (
                            [] if not text else [(text, 0.001, (end_ms - start_ms) / 1000 - 0.001)]
                        ),
                    }
                )
            return results

    class Lid:
        def process(
            self, uttids: list[str], batch_wav: list[tuple[int, Any]]
        ) -> list[dict[str, Any]]:
            assert len(uttids) == len(batch_wav)
            record.setdefault("lid_batch_sizes", []).append(len(uttids))
            return [{"uttid": uttid, "lang": "en", "confidence": 0.9} for uttid in uttids]

    class Punc:
        def process_with_timestamp(
            self, timestamps: list[Any], uttids: list[str]
        ) -> list[dict[str, Any]]:
            assert len(timestamps) == len(uttids)
            call = record.get("punc_calls", 0) + 1
            record["punc_calls"] = call
            record.setdefault("punc_batch_sizes", []).append(len(uttids))
            results = []
            for uttid in uttids:
                start_ms, end_ms = bounds(uttid)
                text = record["asr_text"][uttid]
                results.append(
                    {
                        "uttid": uttid,
                        "punc_sentences": [
                            {
                                "start_s": 0.0,
                                "end_s": (end_ms - start_ms) / 1000,
                                # Recasing is source-backed punctuator behavior and
                                # survives in raw output while the punctuation floor's
                                # casefolded comparison still reproduces the ASR word.
                                "punc_text": (text.upper() if call == 1 else text.lower()) + ".",
                            }
                        ],
                    }
                )
            return results

    config = types.SimpleNamespace(
        enable_vad=True,
        enable_lid=lid_enabled,
        enable_punc=True,
        asr_batch_size=3,
        punc_batch_size=4,
        asr_config=types.SimpleNamespace(return_timestamp=True),
    )
    system = types.SimpleNamespace(
        config=config,
        vad=Vad(),
        asr=Asr(),
        lid=Lid(),
        punc=Punc(),
    )
    namespace = {
        "logger": types.SimpleNamespace(info=lambda *_args: None),
        "re": re,
        "sf": types.SimpleNamespace(read=lambda *_args, **_kwargs: (wav, sample_rate)),
    }
    exec(_PINNED_FIRERED_PROCESS, namespace)
    pinned_process = types.MethodType(namespace["process"], system)

    upstream = pinned_process("/canonical.wav", "audio")
    upstream_calls = deepcopy(record)
    record.clear()

    vad_result, _probability = system.vad.detect("/canonical.wav")
    regions = [
        {"region_id": f"vad_{index}", "start": start, "end": end}
        for index, (start, end) in enumerate(vad_result["timestamps"])
    ]
    mirrored, completed, failure = firered_stage._run_pipeline(
        system,
        wav,
        sample_rate,
        regions,
        asr_batch_size=3,
        punc_batch_size=4,
        lid_enabled=lid_enabled,
    )
    mirrored_calls = deepcopy(record)

    assert failure is None
    assert completed == len(regions)
    assert upstream_calls == mirrored_calls
    assert upstream_calls["asr_batch_sizes"] == [3, 3, 1]
    assert upstream_calls["punc_batch_sizes"] == ([4, 3] if lid_enabled else [4, 2])
    if lid_enabled:
        assert upstream_calls["lid_batch_sizes"] == [3, 3, 1]
    else:
        assert "lid_batch_sizes" not in upstream_calls

    def public_subset(result: dict[str, Any]) -> dict[str, Any]:
        # The stage boundary is JSON, which canonicalizes upstream's tuple VAD bounds.
        subset = {name: result[name] for name in ("sentences", "words", "vad_segments_ms")}
        return json.loads(json.dumps(subset))

    assert public_subset(mirrored) == public_subset(upstream)
    sentence_texts = [item["text"] for item in mirrored["sentences"]]
    if not lid_enabled:
        assert all("region2000" not in text for text in sentence_texts)

    # Mutation witness: resetting punctuation after each ASR batch would group the
    # six post-filter items as 2/3/1 instead of the pinned global 4/2 stream.  The
    # stateful recasing makes that semantic drift observable in raw output without
    # inventing lexical content that the ASR word stream cannot reproduce.
    starts_by_asr_batch = (
        ((0, 2000, 4000), (6000, 8000, 10000), (12000,))
        if lid_enabled
        else ((0, 4000), (6000, 8000, 10000), (12000,))
    )
    reset_batching_texts = [
        (f"region{start_ms}".upper() if call == 1 else f"region{start_ms}".lower()) + "."
        for call, starts in enumerate(starts_by_asr_batch, 1)
        for start_ms in starts
    ]
    assert reset_batching_texts != sentence_texts
    first_second_batch = 3 if lid_enabled else 2
    assert reset_batching_texts[first_second_batch].startswith("region")
    assert sentence_texts[first_second_batch].startswith("REGION")
