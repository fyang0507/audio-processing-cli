# Transcribe original media and export saved results

## Preserve the canonical transcript first

When transcription is requested or needed, transcribe the original source and save the normalized JSON unchanged. Reuse a saved original-source result when it already meets the request. If the task also includes enhancement, transcribe first and enhance last. A later ASR check on an enhanced copy is a separate comparison and does not replace this canonical record.

Canonical identifies the unedited result and its source. It does not guarantee that recognition retained every filler, repetition, repair, or dialect form. `verbatim` expresses a text-fidelity capability; it is not a filler-recall guarantee or a switch that makes omitted words reappear. Do not clean the canonical text yourself or describe absent fillers as deliberate editorial cuts. Filler deletion is future work, separate from enhancement; a video edit would need synchronized cuts in both audio and video, with timing evidence and a mapping back to the original.

## Choose, plan, and run

Choose the stack and the output needed, not every capability. Use `capabilities` when the choice is unclear and `plan` when it is known. Read the live catalog's quality limits in their fixture and configuration context; neither package size nor a declared capability establishes accuracy on new media. Native versus derived capabilities affect the extra packages and work the plan requires.

The catalog's `next` is a concrete floors-only plan command, not a recommendation that its minimal output satisfies the user. Add the capabilities the task needs before planning. Request real `word_timestamps` for subtitles; optional speaker labels remain anonymous unless separate evidence establishes a person's identity or role.

Planning never provisions. Read `packages`, costs, and warnings. When packages are missing, the plan's conditional `next` names a pull for exactly the missing ids; it is absent when no pull is needed. Apply the selected executable and concrete-command checks in [failures.md](failures.md). For provisioning or verification, read [model-packages.md](model-packages.md); a `provisioned` plan entry is not a fresh integrity check. A selected package failure blocks execution before inference.

If stack or input is missing, supply the actual missing value and repeat the intended command. Do not choose a different stack, drop the requested capabilities, or substitute an example file merely to turn a refusal into a zero exit.

## Read what was produced

Published bounds refer to the original source timeline. Processing-container extents are not speech or word timings. Keep absent keys absent, and read the abstention ledger even on success. A requested capability may have outcome `abstained` while useful text or other timed spans survive; report its reason and affected scope rather than calling the entire result complete in that sense. An empty overlap ledger does not prove no overlap when the selected plan did not detect it.

A bracketed non-speech event can deliberately have no aligned words. Ordinary speech with an `alignment_unavailable` abstention retains its available text; do not fabricate word bounds from its segment extent. A speaker label withheld over ambiguous overlap stays absent.

Exit 4 preserves a conforming partial document. Read `complete` and `coverage`, keep the saved result, and evaluate its concrete range-resume fix against the original input. Do not pre-clip or manually offset timestamps. If no progress was made and the fix is a sentence, replaying the same request is not a remedy. Exit 1 is a backend failure and must not be reported as a successful abstention; see [failures.md](failures.md).

## Export without running models again

Export saved JSON through the public CLI and inspect its live help for format and output controls. Compatible disjoint results can be merged on the original timeline. Independent VibeVoice runs with native diarization cannot be merged: identical anonymous labels across generations are not proof of shared speaker identity. Export those separately or generate the needed scope together.

SRT/VTT require real word timing. Bounded non-speech events may be omitted; bounded ordinary speech may be omitted beside real cues only with a same-bounds `alignment_unavailable` abstention. Unbounded ordinary text refuses subtitle export, as does a nonempty transcript with no real word stream unless it is the supported bounded-event-only case. Do not use container or guessed bounds as a workaround.

Plain text and Markdown remain untimed by default. Opt-in `--timestamps` uses supplied segment bounds, falling back to the real word-stream extent. Every segment must have one of those timing sources or export refuses; it does not silently skip untimed text. Inspect the selected invocation's help before requesting it. Do not synthesize readable timestamps from turns, coverage, source duration, or processing containers.

File output is atomic and protects the input transcript and canonical media even when replacement is requested. Use a distinct destination, and retain the original JSON for future exports. Explain recognition, timing, and coverage limits in the reply without rewriting machine output.
