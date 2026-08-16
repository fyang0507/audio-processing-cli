# ASR stack selection

Use this field guide to select the smallest processing stack that satisfies the requested output contract. See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for the measured runs and [FINDINGS.md](FINDINGS.md) for the full research record. A declared interface is not evidence of output quality.

## Verified model cards

| Tested backend | Language | Speaker and timing | Fit and limits |
| --- | --- | --- | --- |
| **VibeVoice-ASR 7B** | Advertises 50+ languages and native code-switching without language selection. Locally, it retained English switches but normalized Sichuanese `看哈→看一下`; Cantonese CantoMap error was 44.88%. | Anonymous speaker-labelled segments and segment bounds; no LID or native word timing. | Best tested editing structure. Add ForcedAligner only where needed. Not a validated 16 GiB route. |
| **FireRedASR2S full pipeline** | ASR supports Mandarin, English, code-switching, and 20+ Chinese dialects/accents; VAD/LID support 100+ languages. It retained `看哈`; CantoMap error was 47.77%. | **FireRedVAD → optional FireRedLID → FireRedASR2-AED → FireRedPunc** yields VAD regions, region LID/confidence, word time/confidence, and punctuation bounds; no speakers. | Prefer for dialect form, LID, native word timing, or tighter resources. LID is region-level and nearly doubled the measured inference time. |
| **Qwen3-ASR 1.7B / 0.6B 8-bit (MLX)** | 30 language labels, including Chinese, English, and Cantonese; no dialect selector. On the same Cantonese turns, 1.7B error was 33.56% vs 0.6B 52.64%; 1.7B retained Sichuanese `耍啥子`, while 0.6B rendered `刷啥子`. | Transcript plus one container-level language label; no speakers or speech/word timestamps. | Use 1.7B for transcript quality and 0.6B for footprint/latency. Auto-LID disagreed on the mixed clip, so do not use it for routing. |

Language figures are scoped local mixed-token results, not general language or dialect rankings. See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for the denominators, fixtures, and non-comparable runs.

| Auxiliary processor | Verified output | Limit |
| --- | --- | --- |
| **Qwen3-ForcedAligner 0.6B** | Accepted reference text → word/character candidates with intervals. | Not ASR, speaker detection, or proof of edit-grade boundary accuracy. |
| **FluidAudio 0.15.5** | Anonymous speaker intervals, including overlap-permitting output. | Keep overlap/short fragments in an abstention ledger; do not infer roles. |
| **Silero VAD 6.2.1 ONNX** | Fast 16 kHz mono speech-activity regions with probabilities. | Speech activity only; not speaker turns or event detection. |

## Recommended stacks

| Use case | Capability requirements | Recommended route | Guardrails |
| --- | --- | --- | --- |
| **Interview transcription—mixed mono or dedicated participant/interviewer channels** | Fast Mandarin/Chinese dialects/English transcript; stable absolute speech bounds; anonymous speakers or capture-derived roles; explicit overlap handling. Word timing is optional. | Preserve dedicated channels when available, validate dominance/bleed, and deduplicate cross-talk. Otherwise use **FluidAudio quality → sample-exact reconciliation with overlap abstention**. Both paths converge on persistent **Qwen3-ASR 1.7B 8-bit**, `batch=1`, clearing the MLX cache after every turn; choose **0.6B** only for minimum latency/footprint. Reattach the external bounds and confirm roles. | Qwen container bounds are not timestamps. The 1.7B/0.6B accuracy comparison is Cantonese-only; validate Mandarin, English, and each target variety separately. Channel dominance is not diarization. The measured role label is an oracle, not identity inference. |
| **Product-demo editing—speaker structure, code switches, and fillers matter** | Verbatim-oriented text; anonymous speaker segments; selective word/character intervals; memory-adequate local runtime. | Persistent **VibeVoice 7B MPS BF16** with the tracked `logits_to_keep` patch → normalize native segments → run **Qwen3-ForcedAligner only on selected segments**. | Cut and rerender from the original media. Boundary MAE/P95 and filler recall remain unmeasured. Do not deploy this route as a 16 GiB stack. |
| **Dialect/LID/native-word-time or constrained-runtime exception** | Chinese dialect emphasis, dominant region LID, native word time/confidence, or lower memory/latency than VibeVoice. | Run the **full FireRedASR2S pipeline**. Keep LID off unless its region label is required; use ASR and punctuation batch size 4. Add external speaker intervals only when needed. | FireRed has no speaker output, and its regional LID cannot locate code switches. Native times are not yet validated as edit-grade. |

### FireRed substitution rule

Use **VibeVoice 7B plus selective ForcedAligner as the higher-capability substitute for FireRedASR2S** whenever memory/latency are acceptable and the workflow is not dialect-heavy or dependent on FireRed LID. It adds native speaker structure and produced the strongest local editing-oriented output.
Keep FireRed for the explicit exception row: dialect/LID needs, native word time/confidence, or performance, memory, and latency constraints.

Reject **VibeVoice-ASR-BitNet** for verbatim or behavioral evidence: the tested build hallucinated and compressed content. Keep all speaker identities and semantic roles external to ASR unless capture metadata establishes them.
