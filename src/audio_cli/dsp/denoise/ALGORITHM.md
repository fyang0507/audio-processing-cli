# Bounded broadband denoising

`linked-spectral-subtraction-v1` is deterministic NumPy/SciPy processing with no
model runtime. Profile version 5 adds it between high-pass/de-hum and voice
processing. Existing demo leveling, source balance, compression, and peak targets
are unchanged. There are no new command-line controls.

## Evidence and abstention

Noise estimation uses complete 32 ms Hann frames, at quarter-window hops, outside
resolved speech treatment and salient non-speech program regions. A 120 ms guard
protects boundary material. An eligible run must cover at least 250 ms continuously;
Every eligible run is checked in 250–500 ms blocks before pooling. The final
stationary estimate still uses up to 256 evenly distributed frames; each eligibility
block is processed separately, bounding spectral memory without skipping reference
runs. These requirements are evidence checks, not proof that VAD found every phoneme.

The estimator rejects a 90th-to-10th percentile reference level spread above 6 dB.
It also checks each channel: within-block frame levels and between-block mean
levels must remain within 6 dB. Absolute octave-band powers must remain within
6 dB across all reference blocks on each channel. Bands cover DC through Nyquist
with edges at 200, 400, 800 Hz and successive doublings; the final band includes
Nyquist. Thus equal-RMS spectral changes and channel-level swaps cannot be hidden
by averaging across time or channels.

Spectral flatness must be at least 0.15 in every block's non-silent channels over
200 Hz to 8 kHz (clipped at Nyquist), as well as in the final pooled estimate.
Tonal program is not treated as broadband noise. A silent
reference is a no-op. Inadequate reference coverage, changing noise, insufficient
speech samples, or speech/noise contrast below 3 dB cause abstention. Contrast
above 20 dB is a no-op. Contrast compares mixture and reference power; it is not
a measured clean-speech SNR. Very quiet recordings use these same relative checks.

Successful operations include the measured block count, maximum band-power spread,
maximum channel-level spread, and minimum block flatness. Refusals from these checks
include the same available diagnostics; the older pooled diagnostics retain their
meaning. Incompatible blocks abstain rather than selecting a more convenient subset.

The stationary noise-power estimate is the per-bin/per-channel median periodogram
divided by log(2), with a three-bin average. The median resists occasional transient
contamination; the log(2) correction is the exponential-periodogram approximation
for Gaussian stationary noise. Colored and non-Gaussian backgrounds can violate
that approximation, which is one reason attenuation is bounded.

## Filter and reconstruction

For each time/frequency bin, use the largest observed-power/noise-power ratio among
channels. Compute amplitude gain `sqrt(max(0, 1 - 1/ratio))`, clipped to
`[10**(-6/20), 1]` by the profile's 6 dB maximum. Power is averaged over three
frames. Three-bin and three-frame local maxima protect neighboring harmonics and
onsets, then five-frame/three-bin gain averages reduce rapid fluctuations. The
same real gain multiplies every channel's complex spectrum. No channel is shifted,
no phase is synthesized, and proportional channels retain their ratio/polarity.
Arbitrary stereo mixtures retain a linked spectral treatment, not necessarily
identical sample-domain correlation after filtering.

Analysis and synthesis use periodic Hann windows. Reflection padding and normalized
weighted overlap-add reconstruct the original sample count, including endpoints;
no latency, trimming, or resampling is introduced. This follows the nonzero
window-overlap reconstruction described in the
[SciPy STFT documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html).
Processing batches contain 256 frames plus four-frame halos for identical adaptive
smoothing across batch boundaries. The filter reports its maximum spectral reduction
before the outer speech-region blend; it is not a final output noise measurement.

The enhancement caller blends only over the existing resolved speech treatment
regions and their smooth transitions. Other program remains unchanged by this
stage. The 6 dB bound applies to filter-bin gain, not each time-domain sample or
total later enhancement gain. Final program loudness/peak checks remain downstream.

## Evidence limits

`tests/test_dsp_denoise.py` constructs known clean voiced harmonics, syllable
transitions, an unvoiced burst, and seeded broadband noise at 16/44.1/48 kHz.
It checks noise-power reduction, clean-reference error, voiced projection,
unvoiced energy, timing, quiet-input scale behavior, linked stereo treatment,
refusal cases, and overlap-add endpoints/batch seams. These are synthetic signal
checks, not speech-recognition or listening-quality claims. Passing tests does not
establish intelligibility improvements on arbitrary voices/backgrounds.

`tests/test_dsp_noise_reference.py` reproduces the independent reviewer's equal-RMS
low/high-pass regime change and channel swap. It also covers changing broadband
references that individually pass flatness, a change within one contiguous run,
quiet-input versions, and stationary colored mono/stereo speech-preservation checks.
Consistency is observed on eligible reference blocks at this time/frequency resolution;
it does not establish an unchanged noise spectrum inside excluded speech intervals.

Reports distinguish component decisions from the applied parent stage. A high-pass
or de-hum operation alone does not mean broadband denoising ran. Fresh detection
and transcription of enhanced audio may differ; fixed-source regional verification
is not classification invariance. User-media listening and CLI acceptance remain
separate, after original-source transcription and parent integration.
