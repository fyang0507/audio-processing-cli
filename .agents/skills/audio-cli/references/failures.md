# Read a failed command

Keep its exit code, stdout, and stderr separate. Empty stderr is not success, and a nonzero exit
does not always mean there is no usable result.

| Command or failure | Where to read it |
| --- | --- |
| `packages verify` completed its checks | stdout JSON: `failed[]`, `verified[]`, and `environments`; exit 3 when `failed` is nonempty, often with empty stderr |
| Package lifecycle failure, or `verify` could not start its checks | stderr JSON under `error` |
| `inspect` / `enhance` domain failure | stderr JSON under `error`, often just `type` and `message`; adjustment errors may add structured fields |
| Transcription / export refusal | a bare JSON object on stderr, usually with `code` and `fix`; a shared media error may instead use the `error` envelope |
| Argument parsing or command startup failure | stderr may be plain text with no JSON; use the selected executable's help or [readiness diagnostics](readiness.md) |

Progress can precede a stderr JSON object. Parse the complete final object, which may span many
lines; do not assume the final line is JSON. If no valid object exists, retain the text and exit
code instead of inventing fields. A `transcribe run` exit 4 preserves a partial result: follow
[transcribe.md](transcribe.md) before retrying. A `transcribe run` exit 1 is a backend failure,
not an abstention.

## Decide whether a fix is executable

A printed `fix` is guidance, not permission or proof that it fits this task.

- A concrete command has the actual input, package, and destination needed for this request.
  Preserve its quoted arguments, but replace its leading `audio` with the selected executable from
  [SKILL.md](../SKILL.md). Check that it preserves the intended stack, capabilities, source, and
  outputs unless a change is justified by the user's request.
- Examples such as `<stack>`, `<input>`, or a sample `meeting.m4a` are not runnable remedies.
  Reconstruct the invocation from the actual request and live help when the missing values are
  known; otherwise report what is missing. Do not execute a placeholder or assume an example path
  names the user's file.
- A sentence explains a condition; do not execute it as shell text. Do not evaluate a returned
  string blindly. Downloads, replacement flags, and removal still need to fit the authorized scope.

Apply a prescribed repair once when authorized, then recheck the same condition. If the same
failure remains, stop that repair loop and report the command, exit, affected package or
environment, and unchanged evidence. Do not escalate to repeated downloads, manual cache edits,
runtime changes, or reinstalling the user's PATH tool.

`verify` covers the provisioned root, not just the current plan. Match each failed package and
environment to the selected plan before deciding what is blocked; an unrelated failure does not
make every stack unusable. See [model-packages.md](model-packages.md) for readiness and repair scope.
