
## 2. Product-demo editing — `vibevoice`

Verbatim-oriented text with native anonymous speaker structure and word
intervals for an editing agent.

```bash
audio transcribe plan --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps
```

`diarization` and `segment_timestamps` are `native`, so this stack needs no
diarizer at all. Only `word_timestamps` adds the aligner. **Abridged to the
fields that differ from §1.1** — the envelope, `packages`, and
`sample_output` all take the same shape.

```json
{
  "roles": {
    "decode":  {"backend": "ffmpeg",
                "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "asr":     {"backend": "vibevoice-asr-7b", "environment": "torch-vibevoice",
                "revision": "d0c9efdb8d614685062c04425d91e01b6f37d944",
                "tokenizer": {"materialized_role": "tokenizer",
                              "repository": "Qwen/Qwen2.5-7B",
                              "revision": "d149729398750b98c0af14eb82c78cfe92750796"},
                "source_commit": "94da20d98b2fa7688e9cbfaf7692ddb4954f7600",
                "patch": "vibevoice-logits-to-keep",
                "config": {"device": "mps", "dtype": "bfloat16", "attention": "sdpa",
                           "seed": 1234, "max_new_tokens": 16384},
                "deterministic": true,
                "determinism_tolerance_ms": 0.0,
                "determinism_basis": "three seeded repeats shared one normalized-output hash; text decode is do_sample=False (run_vibevoice.py:267)",
                "determinism_note": "acoustic tokenizer samples a Gaussian latent; fixed seed required",
                "selected_by": "stack"},
    "aligner": {"backend": "qwen3-forcedaligner", "environment": "mlx",
                "revision": "0e1a68e91d815300c7c9754b2a7639378b23db15",
                "config": {"scope": "all_segments",
                           "language_rule": "Chinese when text matches [一-鿿], otherwise English; the ASR --language hint is never forwarded"},
                "selected_by": "add_on_required_by:word_timestamps"}
  },
  "capabilities": {
    "verbatim":           {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "refuted"},
                           "note": "Emits disfluencies rather than cleaning them — 28 filler hits on the probe, the highest of the four stacks — but a recorded run refuted dialect preservation twice: 看哈 became 看一下 and 耍啥子 became 刷啥子, both retained by firered."
                           },
    "diarization":        {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "measured"},
                           "note": "Each bounded speech segment forms its own native turn, so no gap or non-speech event is filled."
                           },
    "segment_timestamps": {"satisfaction": "native",
                           "evidence": {"interface": "verified", "quality": "unmeasured"}},
    "word_timestamps":    {"satisfaction": "derived", "backend": "qwen3-forcedaligner",
                           "evidence": {"interface": "verified", "quality": "unmeasured"},
                           "note": "Boundary error against labels is unmeasured; non-speech events remain wordless, while unavailable ordinary-speech alignment records alignment_unavailable."}
  },
  "unsized_packages": [],
  "warnings": [
    {"code": "measured_peak_exceeds_target", "blocking": false,
     "detail": "measured 20.28 GiB live MPS allocation on spice-30min-participant; a strict 16 GiB MPS cap OOMs at model load, measured on a 27.8 s probe, while an 18 GiB cap passed that probe"}
  ]
}
```

The aligner rule is executable provenance, not a language-quality claim: the recorded probe
selects `Chinese` when its text regex sees a CJK ideograph and `English` otherwise
(`model_tests/benchmark/run_mlx_forced_aligner_probe.py:55,94`). It does not forward Qwen's
`Cantonese` hint. The pinned aligner implementation branches only for Japanese and Korean;
Chinese, Cantonese, English, and every other value use `tokenize_space_lang`, whose own CJK
splitter handles ideographs (`qwen3_forced_aligner.py:129-145,236-247`). There is therefore no
Chinese-only path and no Cantonese-specific tokenization failure. A future adapter declares
this rule because it is the configuration the recorded probe actually ran, not because the
pinned source proves it is better than forwarding the ASR hint.

```bash
audio packages pull --stack vibevoice
audio packages verify
audio transcribe run --input demo.mp4 --stack vibevoice \
  --want verbatim,diarization,segment_timestamps,word_timestamps \
  --format json -o demo.transcript.json
```

`verbatim` is the reason this stack's catalog is worth reading before choosing it, and
it is worth being precise about what the capability claims. It claims the stack **can
produce** verbatim text — that it emits what it heard rather than a cleaned rendering —
and that is now measured rather than assumed: 28 filler hits here, 26 on both Qwen
sizes, 24 on FireRed, no stack cleaning and no stack complete. Accuracy is a separate
story, carried by `quality`. So `verbatim` resolves `native` here exactly as it does on
`firered`, and the entire difference is that a recorded run *refuted* the quality half
twice, on two clips and two lexemes. `quality: "refuted"` with an `observed_limit` is
not the same statement as `unmeasured`; filing the normalization as unmeasured would
have made the stack that failed the probe read like the stack that was never tested.

Nothing selects this. No backend exposes a verbatim switch — four verbatim-requesting
system prompts left Qwen's output byte-identical to its unprompted baseline — and
nothing in v1 cleans, so the request asserts an interface and the plan answers for
fidelity. It is the one requestable capability that never changes plan composition, by
design rather than by oversight.

Three adapter obligations this stack creates, all from its recorded output.
VibeVoice emits `Speaker: "N/A"` on non-speech segments; that is the absence of a
label, so the adapter emits no speaker rather than a speaker whose id is `"N/A"`.
And it emits bracketed non-speech event tags such as `[Environmental Sounds]` as
segment `text`. Those segments are real segments with real bounds and no words; they
are not sent to the aligner, survive into the transcript, and are not abstentions.
Finally, if an ordinary speech segment's requested alignment stream is absent or
nonconforming, the adapter preserves the text and native bounds, omits `words`, records
one `alignment_unavailable` abstention at those exact bounds, and marks the run-level
`word_timestamps` outcome `abstained`. Valid word streams on other segments remain.

The memory warning is advisory by explicit product decision, and it is emitted
from the plan rather than as a mid-run OOM. Its reference run took roughly
fourteen minutes of generation for thirty minutes of audio — an RTF near 0.47,
which is the figure to scale by; the plan cannot know `demo.mp4`'s duration cost
in advance. Cut and rerender from the original media; this command only reads it.

The live catalog's `failure_recovery.note` also scopes the cap risk: on
`spice-30min-participant`, 11,345 generated tokens over 1,800 seconds is 6.30
tokens/second, so at that observed rate the declared 16,384-token cap projects to
about 43 minutes of comparable audio. That is a rate extrapolation, not an observed
truncation.

Both `vibevoice-asr-7b` and `qwen3-forcedaligner` are sized from their pinned Hub snapshots, so
neither appears in `unsized_packages`. The VibeVoice package total includes its explicit offline
Qwen tokenizer subset as well as the ASR checkpoint; the receipt records both revisions.

## 3. Dialect and audit — `firered`

The only stack with native word timing, native speech bounds, and a region language
label.

```bash
audio transcribe plan --input field.wav --stack firered \
  --want verbatim,word_timestamps,vad,segment_timestamps
```

Every requirement is `native`, so there are no add-ons at all — and this is the only
plan in this document that resolves `punctuator`, and the only one whose `vad` comes
from inside the stack rather than as an add-on. Abridged to `roles` and `execution`:

```json
{
  "roles": {
    "decode":     {"backend": "ffmpeg",
                   "config": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}},
    "vad":        {"backend": "firered-vad", "environment": "torch-firered",
                   "revision": "7990aaccc6b7aec1e527743bd30201f2c4a03b8c",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "selected_by": "stack"},
    "asr":        {"backend": "firered-asr2-aed", "environment": "torch-firered",
                   "revision": "2304afed56eacfee6256dee5937ed22ffa0b64ec",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "config": {"device": "cpu", "dtype": "float32", "batch_size": 4,
                              "return_timestamp": true, "beam_size": 3, "nbest": 1,
                              "decode_max_len": 0, "softmax_smoothing": 1.25,
                              "aed_length_penalty": 0.6, "eos_penalty": 1.0},
                   "selected_by": "stack",
                   "deterministic": true,
                   "determinism_tolerance_ms": 2.0,
                   "determinism_basis": "exact-repeat 60-minute fixture repeated the text and speaker-null sequences; maximum rebased timestamp drift was 1.0000000000002037 ms, within the frozen 2.0 ms tolerance, so normalized segments were not byte-equal"},
    "punctuator": {"backend": "firered-punc", "environment": "torch-firered",
                   "revision": "e448fd967f44182a1c323cc30f5d89f2400c28da",
                   "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
                   "config": {"batch_size": 4},
                   "selected_by": "floor:punctuated_sentence_segmented_text",
                   "recases_text": true}
  },
  "execution": {
    "stage_order": ["decode", "vad", "asr", "punctuator"],
    "residency": "one_environment_process_at_a_time",
    "environments_spanned": ["torch-firered"],
    "note": "FireRed runs one process with VAD, ASR and punctuator co-resident; requesting lid loads that model into the same process. The measured LID-off peak is 9830449152 bytes (9.16 GiB), not a maximum of imaginary per-role processes."
  }
}
```

`punctuator` is the one role in any plan selected by a floor rather than by the
stack or a requirement. It is not optional and not requestable: floor one requires
punctuated, sentence-segmented text, and on this stack that means FireRedPunc always
runs.

`determinism_tolerance_ms` is why `deterministic: true` means something here.
`0.0` claims byte-identical normalized output on repeat, which is what VibeVoice's
three seeded repeats measured. FireRed declares the artifact's frozen `2.0` ms policy:
its recorded exact-repeat run repeated the text and speaker-null sequences, while the
maximum rebased timestamp drift was `1.0000000000002037` ms, so its normalized segments
are *not* byte-equal. A single boolean would
have had to either overclaim that or discard a real result; a downstream `word_id`
scheme has to know which.

```bash
audio packages pull --stack firered
audio packages verify
audio transcribe run --input field.wav --stack firered \
  --want verbatim,word_timestamps,vad,segment_timestamps \
  --format json -o field.transcript.json
```

`firered-asr2s` is one package pinning four repositories and `pull` materializes all of
them, LID weights included, whatever the plan asked for: narrowing a pull to the roles a
plan actually uses is what `--want` is reserved for, and `pull` refuses that flag today
rather than appearing to honour it. Neither the whole-package figure nor a narrowed one is
recorded in a tracked artifact — the only tracked source is a pre-harness "~9.2 GB" note that
its own document marks as history rather than decision evidence — so both appear
as `approximate, unrecorded` until per-artifact sizes are recorded the way the
MLX runs record `weight_bytes`.

### 3.1 What the punctuation floor actually requires here

This is the stack `punctuation_is_sentence_level` is aimed at, and it is worth being
exact about why, because the earlier draft of this floor named a risk FireRed does
not have and prescribed a rule that could never fire.

FireRedPunc does not emit marks with their own bounds. It returns punctuated
*sentence* strings with *sentence* bounds (`fireredpunc/punc.py:109-119`), while
`words` is built separately from the pre-punctuation AED timestamps
(`fireredasr2system.py:181-184`). There is no parallel per-mark stream, so there is
nothing to strip bounds from: measured on the recorded artifacts, 0 of 379 and 0 of
246 word tokens carry a sentence mark, and the only punctuation that appears inside
any word token across all 12,370 recorded words is the apostrophe in 20 English
contractions, which the ASR itself produced.

What the adapter must actually guarantee is the invariant cue splitting depends on:
stripping punctuation and whitespace from a sentence's `text` yields exactly the
concatenation of its word `text` values, compared case-insensitively. Case-insensitively,
because `RuleBaedTxtFix.fix` lowercases the ASR text and then re-capitalizes sentence
starts and standalone `i` (`fireredpunc/punc.py:349-382`) — 234 characters differ by
case across the recorded artifacts, which is why the role above declares
`recases_text: true`. Sentence text carries the marks and the casing; the word stream
carries the bounds; neither is derivable from the other, and the sentence text is
canonical for reading and for subtitles.

Adding the region language label pulls LID and roughly doubles inference: 162.09
versus 84.24 seconds on the 139.284-second probe, CPU float32 at batch size 4,
with identical ASR text and all 246 word texts and times in both runs. The label is
produced once per VAD region and copied onto every sentence in that region, so a
consumer reading per-sentence `lang` as per-sentence detection would be reading
variation that was never measured.

```bash
audio transcribe plan --input field.wav --stack firered \
  --want verbatim,word_timestamps,lid
audio packages pull --stack firered
audio transcribe run --input field.wav --stack firered \
  --want verbatim,word_timestamps,lid --format json -o field.lid.json
```

That plan adds the eighth and last role, and it is the one case where a requirement
turns on a stage the stack already contains rather than adding a package:

```json
{
  "roles": {
    "lid": {"backend": "firered-lid", "environment": "torch-firered",
            "revision": "1bb4d285c8456429385d9c0810300df4297bc11b",
            "source_commit": "4e7d9aaf4482a47cec1724807026b9b151926eb5",
            "config": {"batch_size": 4},
            "selected_by": "requirement:lid",
            "granularity": "vad_region",
            "cost_note": "162.09 s with LID versus 84.24 s without, on the 139.284 s probe"}
  },
  "capabilities": {
    "lid": {"satisfaction": "native",
                        "evidence": {"interface": "verified", "quality": "unmeasured"},
                        "note": "one label per VAD region, copied onto the sentences inside it; no per-sentence detection was measured"}
  }
}
```

Note `selected_by: "requirement:lid"` rather than
`add_on_required_by:lid`. Nothing was added — the `lid` role is declared
by the stack and named as conditional in the `capabilities` report's `roles` sentence. The five
`selected_by` forms are `stack`, `requirement:<capability>`,
`add_on_required_by:<capability>`, `floor:<floor>`, and `pin:<flag>`. `decode` is the
one role that carries no `selected_by` at all, because it is unconditional; the field
exists to explain why a role that could have been absent is present.

FireRed has no speaker output, so speaker attribution here is an add-on like it
is on Qwen. Note the `pull` — entering at this section without it exits 3, since
nothing earlier in §3 provisioned a diarizer:

```bash
audio packages pull --stack firered
audio transcribe run --input interview.wav --stack firered \
  --want verbatim,word_timestamps,diarization,overlapped_speech \
  --format json -o interview.firered.json
```
