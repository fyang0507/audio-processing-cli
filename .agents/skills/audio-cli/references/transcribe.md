# Transcribe local media

Start by deciding what the result must contain, not by asking for every capability. Omitting
`--want` is the small floors-only transcript; diarization adds anonymous speaker labels and turns,
word timestamps add the aligner, overlap detection adds overlap intervals, and VAD adds activity
regions. The abstention ledger is a floor and survives whenever a selected backend excludes a
span, independent of optional result arrays. Each addition changes packages or work, so use `capabilities` when the choice is unclear
and `plan` when it is already known.

The stack is a quality decision. The two executable stacks today are `qwen-1.7b` and
`qwen-0.6b`: they resolve the same capabilities, while recorded Cantonese fidelity separates them.
Do not choose the smaller stack merely because it is available, and do not claim the catalog's
fixture-specific figures apply to new media. FireRed and VibeVoice can be inspected and planned but
their run adapters have not shipped.

Planning never provisions. Read the plan's package list and warnings, then run its explicit pull
command if needed. A run will fail at exit 3 before decode or model load when a required package is
absent or unusable. Follow the JSON `fix`; never download weights or create a runtime by hand.

The original input remains canonical. The command decodes a temporary working WAV, keeps published
bounds on the source timeline, and never promotes processing-container bounds into speech timing.
An absent key is evidence that the capability was not produced, not a blank to fill. Speaker labels
are anonymous and must not be mapped to people or roles without separate evidence.

Exit 4 is usable but incomplete: a conforming partial JSON document was written and the error's
`coverage` object says exactly which source intervals remain. Run its `--range` fix against the
original input. Keep both documents; deterministic merging and subtitle export are not shipped yet.

Interpret the result in your reply. Preserve the JSON output unchanged when another agent or tool
will dispatch on it, and distinguish what the backend produced from any inference you make from the
text.
