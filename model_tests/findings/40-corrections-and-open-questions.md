
### Corrections (2026-08-17)

Found while specifying the `transcribe` command against the recorded artifacts
rather than against these summaries. Both corrections supersede the earlier
statements wherever they appear here, in `DECISION_REPORT.md`, and in
`EXPERIMENT_RESULTS.md`.

- **FireRed emits no per-word confidence.** Round 1's granularity row said
  "word-level timestamps + confidence". Every word object is exactly
  `{start_ms, end_ms, text}` (`fireredasr2s/fireredasr2system.py:181-184`),
  confirmed over 12,370 word tokens across all five recorded FireRed artifacts.
  Confidence exists at *sentence* level as `asr_confidence`, non-null on every
  sentence and ranging 0.158–0.997 on the recorded runs. The capability was part of
  the case for FireRed as an audit route; that specific argument does not hold, while
  its dialect-form, region-LID, native-word-time, and resource arguments are
  unaffected.
- **FireRedPunc emits punctuated sentences, not marks with bounds.** It returns
  `punc_sentences` — punctuated sentence strings with sentence bounds
  (`fireredpunc/punc.py:109-119`) — while `words` is built from the pre-punctuation
  AED timestamps. Two consequences that were not recorded before: the punctuation
  stage also **recases** text, because `RuleBaedTxtFix.fix` lowercases its input and
  then re-capitalizes sentence starts and standalone `i`
  (`fireredpunc/punc.py:349-382`), so sentence text and word text differ in case by
  construction — 234 characters across the recorded artifacts. And the sentence text
  is therefore not reconstructible from the word stream: the invariant that does hold
  is that stripping punctuation and whitespace from a sentence's text yields exactly
  the concatenation of its word texts, compared case-insensitively. Verified on all
  five FireRed artifacts and on all 17 aligned segments of
  `forced_aligner/result_hybrid_multispeaker.json`.

Two smaller attributions corrected in the same pass, both from
`results/2026-08-13-turn-attributed-fast-asr.json`: the MLX cache is cleared after
every *batch* rather than every turn (the two coincide only at `batch_size: 1`), and
every figure in that record's controlled A/B was produced with the language hint
`"Cantonese"` rather than on the no-hint path.

### What remains unproven

- No physical 16 GB machine test or trustworthy quantized VibeVoice 7B path.
  Process/allocator proxies do not prove whole-system safety; FluidAudio RSS
  omits Core ML service memory.
- The 45.67-second Qwen-plus-Fluid result is a measured sequential end-to-end
  systems path, but it still lacks physical-16-GB validation, two-sided
  transcript/activity labels, overlap transcription, and verified
  candidate/interviewer role mapping. Its Core ML memory scope is incomplete.
- No completed two-channel interview pipeline: the interviewer microphone,
  channel-dominance/bleed gates, cross-talk deduplication, and merged transcript
  have not been benchmarked.
- No monolithic 60-minute VibeVoice run or repeated independent long-form
  distribution. The 60-minute FireRed, Qwen, and FluidAudio fixtures duplicate
  the same 30 minutes and provide only duration/resource/repeat evidence.
- No population-level dialect conclusion. CantoMap is one speaker pair/slice;
  SpiCE is one participant/session; non-Cantonese dialect breadth is a staged plan.
- No shared Traditional/Simplified/particle equivalence set, semantic review,
  genuine English-span labels, or formal filler/repair recall.
- No ground-truth forced-aligner boundary MAE/P95. Monotonic timestamps are a
  structural gate, not proof of edit precision.
- CantoMap's collar sensitivity and SpiCE's participant-only transcript prevent
  a simple “diarization accuracy” headline. Report the denominator and label
  scope with every speaker metric.
- No reliable dense conversational-turn diarizer from the tested set: VibeVoice
  matched 39/75 CantoMap changes, the best sherpa row 9/75, and the selected
  FluidAudio quality row 3/75 at one second. These are annotation-order changes,
  not adjudicated conversational turns.
- CrisperWhisper, Whisper 8-bit/whisper.cpp, pyannote
  Community-1, and Sortformer remain unmeasured; Issue #2's dedicated VAD and
  audio-event tracks also remain incomplete.
