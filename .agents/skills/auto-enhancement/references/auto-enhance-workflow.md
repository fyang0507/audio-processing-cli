# Automatic Enhancement Workflow

## Contents

- Contract
- Profiles and stages
- Detection and boundary methodology
- Coverage limits
- Report interpretation
- Render verification

## Contract

Treat automatic enhancement as a narrow, versioned conformance loop:

```text
canonical original + declared profile
  -> inspect facts
  -> evaluate rules
  -> resolve bounded operations
  -> render from the original timeline
  -> remeasure and verify
```

Do not treat the profile as a universal audio-quality judge. Preserve `no_op`, `skipped`, `abstained`, and `failed` as meaningful dispositions alongside `applied`.

## Profiles and stages

The `transcription` profile optimizes speech intelligibility and an ASR-oriented proxy. The `product-demo` profile targets a single speaker plus non-overlapping machine audio or music. Profile identity, version, bounds, and processing order are recorded in the report; the current implementation is version 3.

Apply stages in this fixed order:

1. `channel-balance`: correct only correlated channel-level mismatch.
2. `environment-denoise`: evaluate speech-scoped DC, rumble, hum, and broadband-noise evidence; abstain when a reliable broadband correction is unavailable.
3. `voice-enhance`: apply bounded presence correction, leveling, and compression to resolved speech-treatment regions.
4. `source-balance`: for `product-demo`, balance detected non-overlapping program regions relative to treated speech.
5. `program-loudness`: resolve EBU R128 gain and true-peak limiting without undoing regional balance.

## Detection and boundary methodology

Separate semantic classification from boundary support.

- Use Silero VAD probabilities as high-confidence speech seeds.
- Expand each seed to overlapping 20 ms adaptive-energy clusters.
- Reserve a silent guard and place the treatment fade outside the guarded activity.
- Constrain expansion to the profile search window and neighboring detected machine-audio regions.
- Report the VAD, acoustic-activity, and selected treatment scopes separately under `voice-enhance.resolved_transition`.

Detect non-speech program regions separately. The current detector excludes VAD speech frames and promotes only salient activity above the estimated noise floor, untreated speech reference, and absolute floor. It is an energy-qualified `non_speech_program` detector, not a semantic music-intent classifier.

## Coverage limits

- Quiet intended music or machine output may remain below the source-balance threshold. It then receives global program operations but no relative regional correction.
- A negative VAD result means only “not classified as speech”; it does not mean “environmental noise.”
- Auto-enhance should not lower its thresholds merely because external intent reveals one missed region. Route that evidence through the surgical-adjustment workflow.
- Ordinary gain or EQ cannot independently rebalance overlapping speech and music in a mixed track. Preserve the source or abstain unless separate stems or an explicitly authorized separation capability exist.
- Do not infer perceptual improvement solely from the same measurements that selected the operations. Use listening feedback as independent evidence.

## Report interpretation

Read these surfaces before accepting a result:

- `source`, `engine`, and `profile`: hashes, runtime versions, model identity, thresholds, bounds, and processing order.
- `observations`: measured channel, environment, speech, non-speech-program, and program-loudness facts.
- `regions`: stable speech and `non_speech_program` scopes.
- `rule_evaluations`: profile decisions made from observations.
- `stages`: terminal status, reason, exact operations, component evaluations, and resolved transitions.
- `measurements.before`, `measurements.predicted`, and `measurements.after`: program and regional verification using stable region IDs.
- `adjustments`: resolved user- or agent-authorized operations.
- `resolved_operations_sha256`, source hash, and output hash: reproducibility surfaces.

## Render verification

Require all of the following:

- `rendered` is true.
- `timeline_preserved` is true and duration differs by no more than 50 ms.
- `final_peak_validation.status` is `pass` after encoding.
- Integrated loudness is within 0.6 LU of the profile target.
- Every eligible stage has a terminal status and reason.
- Applied voice enhancement reports `silence_anchored_outside_voice_activity` and per-region treatment scopes.
- Before/after regional measurements retain the same region IDs.
- The output decodes completely without FFmpeg errors.
