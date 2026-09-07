# Resolve setup and command failures

## Find the evidence

Keep the command, exit code, stdout, and stderr. Read the result before retrying: `packages verify` reports failed checks on stdout, and `transcribe run` exit 4 can preserve usable partial JSON. A backend failure is not an abstention.

Progress can precede a multiline JSON refusal on stderr; read the complete final object. Errors may be bare JSON, wrapped in `error`, or plain startup/parser text. Preserve the actual message instead of assuming one universal envelope.

If the command cannot start or lacks the required operation, retain the startup evidence or available help. Use `doctor` when it can run to identify host prerequisites. Report missing installation, updates, or host tools to the operator; do not install them or substitute another audio tool. Model-package setup belongs to the [CLI package lifecycle](model-packages.md).

## Apply a scoped remedy

Use the CLI's explanation and `fix` rather than maintaining an error-code lookup table. Follow a concrete remedy within the existing task authorization; routine correction does not need renewed approval.

- Replace a remedy's leading `audio` with the selected executable. Preserve its quoting and check the real source, stack, capabilities, packages, and destinations.
- Fill placeholders from the actual request and live help; sample paths are not the user's media. A sentence is explanatory guidance, not a shell command. Never blindly evaluate returned text.
- Downloads, replacement, shared-environment repair, and removal must fit the requested scope. A printed fix does not expand it.

After an authorized repair, recheck the same condition once. If the failure is unchanged, stop that loop and report the evidence. Do not escalate to repeated downloads, manual cache/runtime edits, or silently reduced deliverables. Preserve partial transcription results and use [transcription recovery](transcribe.md) for a changed attempt.
