# Audio enhancement

`audio enhance` implements five automatic stages plus optional gain adjustments for speech recordings and product-demo audio. Broadband denoising is implemented as one explicitly selected component of `environment-denoise`; the other stages also work independently of it. This page covers the complete enhancement flow, with links to its implementation and separate evidence from recorded runs. The [broadband algorithm note](dsp/denoise/ALGORITHM.md) documents that component's estimator and filter, rather than the whole enhancement feature.

## Profiles and processing order

The two bundled [profiles](../src/audio_cli/profiles.py) are version 5. Their numbers are processing targets and bounds, not promises about the delivered sound.

| Setting | `transcription` | `product-demo` |
| --- | --- | --- |
| Automatic stages | All except `source-balance` | All five |
| High-pass cutoff when needed | 70 Hz | 75 Hz |
| Speech presence boost at 3 kHz | +1.5 dB | +2 dB |
| Speech RMS target before later program gain | −28 dBFS | −30 dBFS |
| Speech leveling gain bounds | −4 to +20 dB | −4 to +24 dB |
| Speech compression above −24 dBFS | 2:1 | 2.5:1 |
| Program loudness target | −23 LUFS | −16 LUFS |
| Final true-peak ceiling | −3 dBTP | −1.5 dBTP |
| Loudness-range target, measured but not dynamically enforced | 7 LU | 11 LU |

The [pipeline](../src/audio_cli/pipeline/stages.py) runs channel balance → environment cleanup → frequency-scoped adjustments → voice enhancement → source balance → full-band adjustments → [program loudness and limiting](../src/audio_cli/pipeline/loudness.py) → [encoding and verification](../src/audio_cli/pipeline/publication.py). `--skip` disables named automatic stages; profile-disabled stages report `no_op`. Enabling a stage means evaluating its conditions, not forcing a change.

## Analysis and treatment regions

[Preparation](../src/audio_cli/pipeline/preparation.py) measures the original and uses Silero VAD to seed speech regions. Enhancement [decodes](../src/audio_cli/media/ffmpeg.py) the first audio stream to 48 kHz floating-point mono or stereo; more than two channels are refused. [Inspection](../src/audio_cli/dsp/analysis.py) exposes channel levels/correlation, speech RMS, DC/subbass/hum observations, estimated noise floor, salient non-speech regions, and program loudness/peaks. `inspect --profile` also evaluates selected level targets; an inspection is not a completed enhancement run.

Speech cleanup and voice processing use [acoustic boundary expansion and smooth blends](../src/audio_cli/dsp/regions.py): nearby activity can extend VAD seeds, and 40 ms transitions sit outside the resolved treatment intervals. Other regional gains fade inside their time scopes. These are acoustic heuristics, not word boundaries or proof that every phoneme was found. The detector and treatment do not identify speakers or separate mixed sources.

## Implemented automatic stages

### 1. Channel balance

[`apply_channel_balance`](../src/audio_cli/dsp/treatment.py) adjusts left/right gain when their RMS difference exceeds 1.5 dB and correlation is at least 0.92. Each channel's correction is bounded at 6 dB. Mono and small differences are no-ops; insufficient correlation causes abstention because stereo differences may be intentional. A larger mismatch can remain outside target at the correction bound.

### 2. Environment cleanup

[`apply_environment_cleanup`](../src/audio_cli/dsp/spectral.py) contains three operations, in this order, blended over speech treatment regions and transitions:

- **DC/rumble filtering:** a second-order high-pass runs when the 20–70 Hz power share of the measured speech spectrum reaches the profile threshold (2.5% for `transcription`, 2% for `product-demo`) or absolute DC offset exceeds `1e-4`.
- **Hum filtering:** narrow notches at 60, 120, and 180 Hz run when measured hum excess reaches 9 dB for `transcription` or 8 dB for `product-demo`. Automatic 50 Hz de-hum is not implemented.
- **Stationary broadband denoising (default):** [`apply_broadband_denoise`](../src/audio_cli/dsp/denoise/processor.py) estimates a stationary background and applies linked spectral-subtraction gains across channels. The current maximum reduction is 6 dB per spectral bin, not guaranteed delivered noise reduction or a bound on the full enhancement chain.

Broadband processing requires at least 250 ms of contiguous reference audio away from speech treatment and salient non-speech regions. Its [estimator](../src/audio_cli/dsp/denoise/estimation.py) and [reference checks](../src/audio_cli/dsp/denoise/reference.py) reject inadequate, changing, tonal, or insufficient-contrast evidence; silent or sufficiently weak backgrounds can be no-ops. Reference consistency outside speech does not establish what the background does underneath speech. The [algorithm note](dsp/denoise/ALGORITHM.md) gives the exact eligibility checks and [filter](../src/audio_cli/dsp/denoise/filtering.py) mechanics. No speech means this whole stage abstains.

The optional `--denoiser rnnoise` selects [model-based speech denoising](rnnoise-denoising.md) before processing begins. It requires the explicitly provisioned `rnnoise-voice` package and uses the existing FFmpeg runtime. It does not require a stationary noise-only reference and never runs as a fallback from stationary abstention. The default remains `stationary`. Both choices apply within the resolved speech treatment scopes; they have different processing limits, and neither removes filler words or separates wanted sounds mixed with speech.

The parent stage can report `applied` because high-pass or de-hum filtering ran while `broadband-denoise` reports `abstained`. Read the component decision and operation list before attributing an output change to broadband suppression.

`noise_reference_scopes` exposes the candidate source-time intervals after speech/program exclusion, each run's eligibility or rejection reason, and the measurements available for that run. A locally eligible interval can still fail the pooled comparison against other intervals. The [recorded diagnostic dry runs](../model_tests/benchmark/results/2026-09-06-noise-reference-diagnostics.json) found no locally eligible complete reference run in either requested recording: 21 candidate runs in the video and 8 in the mixed-language recording. Selecting a nearer complete run does not by itself resolve those cases.

### 3. Voice enhancement

[`apply_voice_enhancement`](../src/audio_cli/dsp/dynamics.py) combines a presence EQ centered at 3 kHz, one bounded gain calculated from aggregate detected-speech RMS, and a compressor using 20 ms RMS frames with interpolated gain shared across channels. The processed signal is blended over resolved speech treatment intervals. No speech causes abstention. This is not per-speaker leveling; correction bounds or compression can leave speech outside its RMS target. Later program gain changes the final speech level, and gain can also raise background sound within the mix.

### 4. Source balance

In `product-demo`, [`apply_source_balance`](../src/audio_cli/dsp/treatment.py) adjusts salient non-speech time regions relative to the processed speech RMS. It aims for a −3 dB difference within a −4 to −2 dB band, with up to +12 dB boost or −30 dB attenuation. These are RMS differences, not separately measured regional LUFS.

The [region detector](../src/audio_cli/dsp/regions.py) uses energy and speech exclusion; `machine_audio_*` is a report-local name, not semantic recognition of music or a machine. Quiet/ambiguous events may not qualify. Missing speech causes abstention, missing eligible regions or levels already in range cause no-ops, and regions overlapping detected speech are left out of this stage. It cannot balance simultaneous speech and music independently in a single mixed track.

When source balance and program normalization both apply, the [loudness workflow](../src/audio_cli/pipeline/loudness.py) can make up to three bounded regional correction passes to compensate for downstream level changes. Overlap abstentions stay excluded; bounds can leave residual mismatches.

### 5. Program loudness and true-peak handling

[`normalize_loudness`](../src/audio_cli/pipeline/loudness.py) evaluates integrated loudness and peaks, then calls [`render_loudness_normalized`](../src/audio_cli/media/ffmpeg.py) for measured linear gain and an oversampled limiter when needed. It reserves 1 dB of codec headroom. The limiter runs at 192 kHz with latency compensation, followed by resampling to the processing rate.

Dynamic loudness-range control is deliberately not applied: excessive LRA remains an unresolved component because such control would change relative region balance. Silent or too-short audio without measurable integrated loudness cannot be normalized. Skipping `program-loudness` also skips this limiter; peak validation still applies, using a −0.1 dBTP ceiling instead of the profile ceiling.

## Optional manual adjustments

[`GainAdjustment` and `load_adjustments`](../src/audio_cli/adjustments.py) support gain from −24 to +12 dB over the whole recording or an explicit time interval, either full-band or over a frequency range. They remain separate from stage skips. Frequency-scoped gain runs before voice enhancement; full-band regional and whole-program gain run after source balance and before program normalization, which may offset a requested gain.

[`apply_frequency_adjustments`](../src/audio_cli/dsp/spectral.py) implements both accepted `band` and `notch` shapes with the same peaking biquad, using the range to calculate center frequency and width. These are smooth EQ responses, not hard frequency cutoffs or adaptive noise removal. [`apply_fullband_adjustments`](../src/audio_cli/dsp/treatment.py) uses gain with boundary fades. Overlapping adjustments compose in file order within their placement; they affect the mixed signal in their scopes.

## Renders, reports, and limits

[The runner](../src/audio_cli/pipeline/runner.py) supports a dry run that performs processing and predicts measurements without publishing final media. A render starts from the original; [preparation](../src/audio_cli/pipeline/preparation.py) refuses the same resolved input/output path and, by default, inputs marked as prior enhanced renders. The source remains canonical. Enhancement has no transcript-based cutting, silence trimming, or filler-word removal stage.

[`encode_output`](../src/audio_cli/media/ffmpeg.py) supports WAV, FLAC, MP3, AAC/M4A, and Opus outputs; supported video containers copy the original video streams and replace the audio. [`publish_output`](../src/audio_cli/pipeline/publication.py) remeasures encoded audio, can reduce gain and re-encode for codec peak overshoot, and checks loudness, true peak, duration, and applicable source-region targets before promoting the file. Its decoded-audio duration tolerance is 50 ms (inclusive) and enabled program-loudness tolerance is 0.6 LU. The reported `timeline_preserved` duration check and copied video do not prove exact A/V sync or visual alignment; audio stream start times can differ.

[Reports](../src/audio_cli/pipeline/reporting.py) distinguish `applied`, `no_op`, `skipped`, and `abstained` decisions. The [`unresolved` list](../src/audio_cli/pipeline/outcomes.py) records abstentions and unmet component/region targets even when other operations applied. Use `measurements.after.program_actual` for a rendered file's measured loudness, LRA, and peak; dry-run `predicted` values describe pre-encode audio. Regional comparisons reuse the original source intervals. Fresh VAD or ASR on enhanced media may change, so those comparisons do not establish classification invariance or recognition quality.

## What the recorded evidence establishes

**Synthetic signals:** the [broadband artifact](../model_tests/benchmark/results/2026-09-05-broadband-denoise.json) records a four-second voiced/unvoiced fixture with seeded white noise (seed 33, standard deviation 0.012), `product-demo` v5, at 16, 44.1, and 48 kHz. The isolated broadband component, before outer speech blending and later leveling, measured 3.09–3.23 dB pause-noise reduction and 2.68–2.97 dB clean-reference error reduction. The [fixture source](../tests/audio_cli/dsp/denoise/test_dsp_denoise.py) defines the signals and comparison windows. These are synthetic measurements, not real-recording denoising, listening, or intelligibility results.

**Real recordings:** the [2026-09-05 acceptance artifact](../model_tests/benchmark/results/2026-09-05-e2e-followup-acceptance.json) records two `product-demo` v5 renders. Both broadband components abstained with `noise_reference_is_nonstationary`.

| Recording | Applied stages | Encoded measurements and unresolved results |
| --- | --- | --- |
| `demo-video-audio-to-improve.mp4` | Voice enhancement, source balance, program loudness | −16.43 LUFS; −1.77 dBTP; LRA 17.5 LU exceeds the 11 LU target; voice gain and one source region reached correction bounds. |
| `autio-test-sample.m4a` | Environment cleanup, voice enhancement, program loudness | −16.19 LUFS; −2.32 dBTP; broadband still abstained despite the applied parent stage. Fresh classification left the intended balance near 9.16–9.34 s unresolved. |

Both renders passed the recorded peak checks; their requested-goal acceptance was `PARTIAL_FAIL`. Perceptual quality was unscored, and the video case has no exact sync proof. Successful rendering and synthetic attenuation do not establish that these recordings were broadband-denoised or that arbitrary recordings will improve.

See [duration and alignment evidence](timing-evidence.md) for the public duration bases, stream-origin fields, explicit alignment abstentions, and content-bearing regression scope.

The program-loudness stage's `loudness-range` component labels its input measurement `measured_at: before_initial_program_loudness`. That value precedes initial normalization and any later regional correction passes. The final `unresolved` LRA row and `measurements.after.program_actual` describe the encoded output; their numbers can differ. Older reports omit the component's stage label, so retain the distinction when interpreting those saved reports.
