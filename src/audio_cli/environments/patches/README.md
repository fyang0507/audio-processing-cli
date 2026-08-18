# Local VibeVoice benchmark patches

## `vibevoice-logits-to-keep.patch`

The Hugging Face generation loop only needs the last prompt-position logit,
but the VibeVoice ASR wrapper at upstream commit
`94da20d98b2fa7688e9cbfaf7692ddb4954f7600` does not expose Transformers'
`logits_to_keep` forward argument. It therefore projects every audio-prompt
hidden state through the 152,064-token language-model head before discarding
all but the final row.

This patch adds the standard argument and slices hidden states before the
language-model head. The default value of `0` still selects the full sequence.
Calls with `labels` explicitly force that full-sequence path, preserving the
existing shifted-label training loss even if a caller supplies another value.
Transformers detects the explicit parameter and supplies `logits_to_keep=1`
during ordinary generation.

Apply it from the repository root:

```bash
git -C model_tests/vibevoice/VibeVoice apply \
  ../../benchmark/patches/vibevoice-logits-to-keep.patch
```

The patch intentionally excludes other local changes in the ignored nested
checkout, including the existing XPU-availability guard in the demo script.

Validate that the patch still applies to a clean checkout:

```bash
git -C model_tests/vibevoice/VibeVoice apply --check \
  ../../benchmark/patches/vibevoice-logits-to-keep.patch
```

After applying it, run the instrumented BF16/MPS calibration:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 \
  model_tests/vibevoice/VibeVoice/.venv/bin/python \
  model_tests/benchmark/run_vibevoice.py \
  --model-path model_tests/vibevoice/VibeVoice/pretrained_models/VibeVoice-ASR \
  --audio autio-test-sample.m4a \
  --output model_tests/benchmark_runs/vibe_mps_bf16_seed1234_logitskeep_single.json \
  --device mps --dtype bfloat16 --attention sdpa --max-new-tokens 2048 \
  --seed 1234
```

VibeVoice's acoustic tokenizer samples a Gaussian latent during inference, so
an unseeded historical output is not a valid output-regression baseline. To
produce a controlled unoptimized/optimized pair without changing the checkout
between runs, use the runner's `--full-prompt-logits` baseline mode. It disables
Transformers' feature detection, causing the patched method's default value of
`0` to retain the original full-prompt projection:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 \
  model_tests/vibevoice/VibeVoice/.venv/bin/python \
  model_tests/benchmark/run_vibevoice.py \
  --model-path model_tests/vibevoice/VibeVoice/pretrained_models/VibeVoice-ASR \
  --audio autio-test-sample.m4a \
  --output model_tests/benchmark_runs/vibe_mps_bf16_seed1234_fullprompt_single.json \
  --device mps --dtype bfloat16 --attention sdpa --max-new-tokens 2048 \
  --seed 1234 --full-prompt-logits
```

Exact normalized-output hash equality across this seeded pair is the output
regression gate. Generation time and sampled MPS allocator peaks are
performance observations and should be repeated before making stable
throughput claims. The older unseeded `vibe_mps_bf16_single.json` remains
useful for rough resource comparison only.

### Short calibration result

On the 27.8-second `autio-test-sample.m4a`, the controlled seed-1234 pair
produced the same normalized segment hash
`8669987e03b7f0dd3c086e5b49d97932e0c7ecc6f1626f4a07446bae5d953668`.
The full-prompt and optimized generation times were 4.665 seconds and 4.734
seconds, respectively. End-to-end times were 8.345 seconds and 8.304 seconds.
This is performance-neutral within single-run noise, not evidence of a short-
audio speedup. The prompt was only 269 tokens and the generated result was 96
tokens including EOS, so the avoided projection is small and the 0.2-second
memory sampler cannot reliably capture its transient allocation.

The first unseeded optimized run was faster than the older unseeded artifact
(4.566 versus 7.331 seconds generation), but the acoustic tokenizer samples a
Gaussian latent and the older run also had a much longer model load. That
comparison is confounded and must not be attributed to this patch.

The optimized path subsequently completed the seeded 30-minute SpiCE interview
with 13,562 prompt tokens and 11,345 generated tokens. It took 851.1 seconds to
generate and reached 20.28 GiB of live MPS allocation. A full-prompt baseline
was deliberately not run at that duration. If fully materialized at BF16, a
`[13,562, 152,064]` prompt-logit tensor has a logical size of about 3.84 GiB.
The patch removes that prompt-length vocabulary projection in code, but lazy
MPSGraph execution, allocation lifetimes, and 0.2-second polling mean neither
that tensor nor an end-to-end peak reduction was directly measured. An
unbaselined host-wide swap reading was also observed during the optimized run;
it is anecdotal system-pressure evidence, not process-attributable memory.
This validates output preservation and removal of unnecessary work, not a
measured speed or memory gain. A chunk-duration ladder is still needed to
measure the patched and unpatched memory delta safely.

A second seeded A/B on the 149.9-second CantoMap slice also produced equal
normalized output (`0e9136cb…`) with 1,186 prompt and 1,590 generated tokens.
Full-prompt and optimized generation took 75.31 and 76.34 seconds,
respectively, and the sampled live-allocation peaks were indistinguishable.
Together, the two pairs support correctness of the patch but neither a speedup
nor a measured memory reduction; the theoretical allocation removal is not a
substitute for higher-frequency allocator profiling.
