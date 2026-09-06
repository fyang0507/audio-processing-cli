
### Qwen3-ASR capability probe (2026-08-16)

A fresh-process MLX probe tested both cached 8-bit checkpoints on the same
139.284-second Mandarin-English, Sichuanese-bearing, two-speaker filler clip.
The runner passed `language=None` (`--language auto`) rather than relying on an
upstream capability declaration. The compact record is
[`benchmark/results/2026-08-16-qwen-capabilities.json`](../../../model_tests/benchmark/results/2026-08-16-qwen-capabilities.json).

Both checkpoints returned Han and Latin text in the same segment: 0.6B emitted
120 Han characters and 129 Latin word tokens; 1.7B emitted 121 and 125. That
verifies mixed-script retention through this adapter, not code-switch accuracy.
The one unlabelled dialect probe was mixed: 1.7B retained the observed
Sichuanese `耍啥子`, while 0.6B rendered `刷啥子`; it cannot validate the
claimed dialect inventory or a population-level error rate.

Automatic language output exists, but it is not a code-switch locator or a
safe routing signal from this probe. On the same audio, 0.6B returned one
`Chinese` label and 1.7B returned one `English` label for the entire processing
container. Both returned `speaker=null`, one 0.0–139.284-second container bound,
and no speech, word, or character times. Therefore Qwen remains a transcript
worker behind external VAD/channel/diarization bounds; add ForcedAligner only
when word or character timing is required. No native filler-specific or
verbatim mode was observed.

### VibeVoice resource and acceleration conclusions

1. **The tested 16 GiB MPS allocator limit hard-fails stock BF16 model load.**
   `torch.mps.set_per_process_memory_fraction` produced OOM at 16.04 GiB and
   did not automatically offload or merely slow down. The checkpoint has 8.674B
   BF16 parameters = 16.157 GiB before KV cache, activations, acoustic encoder
   state, or the OS. CPU BF16 reduces sampled process RSS to 15.31 GiB on 28
   seconds but takes 163.0 seconds (RTF 5.87). This is not a physical 16 GB Mac
   test and does not rule out an explicitly quantized/offloaded implementation.
2. **The old CPU-over-GPU conclusion is retired.** With explicit
   `torch.mps.synchronize()`, `PYTORCH_ENABLE_MPS_FALLBACK=0`, a fixed seed, and
   three fresh processes per configuration, synchronized `model.generate()`
   wall time was 2.66× lower for MPS FP32 than CPU FP32 on this host. That timer
   covers speech encoding, prompt prefill, and autoregressive decode—not an
   isolated GPU kernel. MPS BF16 was another observed 1.70× lower than MPS FP32,
   but emitted 96 rather than 97 tokens, so this is a configuration ratio rather
   than a pure dtype-kernel gain. Disabling unsupported-op fallback does not prove
   full GPU residency or saturation. Output hashes are stable within each
   device/dtype configuration but differ across configurations, so
   “byte-identical across CPU/GPU” is false.
3. **Use a persistent worker.** For the 28-second clip, median runner job-window
   latency is 25.18s on CPU FP32 and 27.21s on MPS FP32 despite the generation
   ratio; loading dominates. MPS BF16 cuts it to 12.42s. The job window starts
   after Python/module imports, so it is not complete CLI cold-start latency. A
   service should load once and queue jobs instead of spawning a model per clip.
4. **Plain answer on `logits_to_keep`: keep the patch. It removes avoidable
   work, but this benchmark did not measure a speed or peak-memory improvement.**
   Upstream projects the entire audio prompt through a 152,064-token
   language-model head even though generation consumes only the last position.
   The tracked patch preserves training behavior and lets Transformers request
   one logit position. Seeded 28-second and 150-second A/B pairs have equal
   normalized outputs and neutral timing (4.665s vs 4.734s; 75.31s vs 76.34s).
   At the 30-minute BF16 prompt, the logical full-prompt output tensor would be
   about 3.84 GiB if materialized, but lazy execution and the 0.2-second sampler
   did not establish that allocation or resolve a peak delta. Treat this as a
   correctness-gated scaling cleanup, not a claimed optimization result.
5. **Chunking does not solve the weight floor.** It bounds prompt/KV growth,
   output-token caps, retries, and memory pressure, but stock BF16 7B cannot fit
   under the tested strict 16 GiB MPS allocator cap.
   A true 16 GB route needs a smaller trustworthy model, quantization that
   preserves verbatim behavior, explicit CPU/disk offload, or a remote worker.

### Recommended processing pipelines

The backend contract should be capability-driven rather than one fixed chain:

```text
capture/probe
  -> preserve dedicated role-mic channels when available
  -> bounded audio jobs + conservative VAD
  -> ASR backend selected by use case and memory
  -> normalize raw text, times, anonymous speakers, confidence/provenance
  -> add only missing capabilities (role mapping, diarization, word alignment)
  -> descriptive features / evidence-linked analysis
  -> human review
```

| Use case | Recommended v1 route | Why / limit |
|---|---|---|
| **Product-demo editing** | Persistent **VibeVoice MPS BF16 + `logits_to_keep`**, then **Qwen3-ForcedAligner per selected segment** only when edit-grade word boundaries are requested | Best qualitative verbatim/structure result from the local demo tests. Always cut/rerender from the original media. This route is provisional until boundary MAE/P95 is labeled, and it is not a 16 GB pipeline. |
| **Interview analysis, dedicated participant/interviewer microphones** | Preserve channels → validate dominance/bleed → transcribe each channel with **Qwen 1.7B** for quality/speed, **Qwen 0.6B** for minimum footprint, or **FireRed LID-off/batch-4** for native word times → deduplicate cross-talk → merge by retained time → map channel→role from capture metadata | Prefer this route whenever capture permits it; channel metadata is cheaper and more trustworthy than inferring identity. Only the participant mic was ASR-scored, so interviewer-channel transcription, deduplication, and merged latency/quality remain integration gates. |
| **Interview analysis, mono/mixed, quality-speed default** | **FluidAudio quality diarization/VAD → sample-exact turn reconciliation with overlap abstention → persistent Qwen3-ASR 1.7B 8-bit, batch one with cache clearing → reattach turn bounds → role confirmation** | On the same 195 turns, ASR took **53.77s** and oracle-participant error was **33.56%**, with 3.02 GiB RSS / 4.86 GiB MLX proxy. The observed Fluid plus ASR stages total about 69s, but this exact 1.7B chain was not timed by the end-to-end orchestrator. |
| **Interview analysis, mono/mixed, minimum footprint/latency** | Same FluidAudio route → **Qwen3-ASR 0.6B 8-bit**, batch one with cache clearing | **45.67s measured sequential wall on 30m**, 195/195 turns, 91.42% single-speaker activity coverage, and 52.64% oracle-participant error; Qwen stage 1.66 GiB RSS / 3.20 GiB MLX proxy. Prefer only when its quality tradeoff is acceptable. |
| **Interview analysis, mono/mixed, timestamp/quality balance** | **FireRed LID-off/batch-4 + FluidAudio intervals**, normalized into one word/speaker schema | FireRed is the more conservative Cantonese ASR/timestamp route: native word intervals, 49.02% participant-mic error, and 9.12 GiB RSS. It misses the under-five-minute 30m target at 665.26s, and the combined pipeline has not been run or memory-profiled. |
| **Chinese dialect conversation** | Use **FireRed** as the current balanced multilingual/dialect default; use **VibeVoice 7B + aligner** when product-editing structure matters and memory is available; use **Qwen 1.7B + FluidAudio** only as the measured Cantonese interview route | Evidence is Cantonese-only and cannot rank Chinese dialects generally. FireRed and VibeVoice differ by only 11/381 CantoMap edits. VibeVoice, sherpa, and FluidAudio all missed many of 75 dense annotation-order changes, so none is a validated dense-turn diarizer. |

For hour-long interviews, use bounded jobs with stable absolute offsets,
idempotent retries, and explicit merge/deduplication. Qwen's measured 180-second
low-energy policy reached one hour, but it is not itself a timestamp source;
the production speed route should feed it attributed speech turns from channels
or diarization. VibeVoice should use bounded overlapping jobs rather than a
monolithic hour. Mixed recordings need global speaker reconciliation; never
assume `Speaker 0` remains the same person across independent generations.

Post-interview analysis needs speaker intervals and a reliable
candidate/interviewer role map, not word-level alignment by default. After role
and interval validation, the pipeline can report **non-evaluative observations**:
attributed speaking-time duration/share, distributions of response-gap and
speaker-segment durations, timestamp-linked question/response pairs, overlap or
unassigned intervals, and verbatim evidence snippets. These describe the
recording; they do not label an overlap as an interruption, score a response,
or infer a trait. The Qwen-plus-Fluid route now validates anonymous
interval-to-ASR integration, but both it and the channel-first route still need
non-oracle role mapping and two-sided human labels before even those descriptors
are a product default. ASR agreement and diarization do **not**
validate personality, competence, deception, or hiring recommendations. Any
evaluative construct needs separate consent, frozen human labels/rubric,
fairness analysis, and human oversight.
