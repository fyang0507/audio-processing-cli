# Transcribe local media

Start by deciding what the result must contain, not by asking for every capability. Omitting
`--want` is the small floors-only transcript; diarization adds anonymous speaker labels and turns,
word timestamps add the aligner, overlap detection adds overlap intervals, and VAD adds activity
regions. The abstention ledger is a floor and survives whenever a selected backend excludes a
span, independent of optional result arrays. Each addition changes packages or work, so use `capabilities` when the choice is unclear
and `plan` when it is already known.

The stack is a quality and capability decision. All four catalogued stacks — `qwen-1.7b`,
`qwen-0.6b`, `firered`, and `vibevoice` — can run. They do not obtain every capability the same way,
so inspect `capabilities` and the concrete plan instead of choosing from package size alone. Do not
claim the catalog's fixture-specific figures apply to new media.

Planning never provisions. Read the plan's package list and warnings, then run its explicit pull
command if needed. A run will fail at exit 3 before decode or model load when a required package is
absent or unusable. Follow the JSON `fix`; never download weights or create a runtime by hand.

The original input remains canonical. The command decodes a temporary working WAV, keeps published
bounds on the source timeline, and never promotes processing-container bounds into speech timing.
An absent key is evidence that the capability was not produced, not a blank to fill. Speaker labels
are anonymous and must not be mapped to people or roles without separate evidence.

Read wordless VibeVoice segments by type. A bracketed event is deliberately excluded from
alignment and is not an abstention. An ordinary speech segment without conforming requested words
keeps its text and bounds and records `alignment_unavailable`; valid streams elsewhere remain, but
the run-level word-timing outcome is `abstained`. Native turns do not bridge gaps or events. A
requested detected overlap intersecting a segment and document scope omits its sole `speaker`
label even when the overlap began before the range; overlap and abstention rows remain start-owned.

Exit 4 is usable but incomplete: a conforming partial JSON document was written and the error's
`coverage` object says exactly which source intervals remain. Run its `--range` fix against the
original input and keep both documents. `audio export` can merge compatible, disjoint partial and
resumed results on the original timeline and re-id their segments. Independently generated
VibeVoice results with native diarization are the exception: anonymous speaker labels are local to
one generation, so export them separately or rerun the desired ranges together. Subtitle formats require a real
word stream and fail closed instead of inventing cue bounds. In a mixed result they emit timed
segments and omit every wordless segment, including `alignment_unavailable` speech; an all-bounded
event result with already-produced timing is the narrow empty-subtitle exception. File output is
atomic UTF-8, refuses an existing destination without `--force`, and never overwrites an input
transcript or canonical source media even with it. Inspect `audio export --help` for formats and
output controls.

Interpret the result in your reply. Preserve the JSON output unchanged when another agent or tool
will dispatch on it, and distinguish what the backend produced from any inference you make from the
text.
