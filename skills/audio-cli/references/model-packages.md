# Provision and reclaim model packages

## Install for the request

For transcription, start from the concrete plan and pull its missing package IDs. Its suggested pull is scoped to that request. Selecting a whole stack can include optional packages and cost substantially more; use it when preparing the whole stack is intended. Preserve unknown download costs when describing the estimate.

For RNNoise enhancement, explicitly pull `rnnoise-voice`. All model/runtime provisioning uses `audio packages pull`; never hand-download weights, build a runtime, or edit pins to bypass a refusal. Silero's first-use bootstrap is the sole automatic exception and can exist without a pull receipt.

Keep long-running pulls alive and follow their progress. Silence alone does not establish a hang; do not start a duplicate pull because a tool call timed out. Read blocking warnings even after a successful exit.

## Check the relevant readiness

Use `doctor` for host readiness, `packages list`/`path` for installed state and locations, and `packages verify` for fresh checks. Registry readiness and a successful doctor call do not establish package integrity or runtime usability.

Verification covers the managed root. Match failures and environment limitations to the selected request, even at exit zero: unrelated failures do not block every stack or justify repairing unrelated packages. A blocked build toolchain also does not necessarily make an already-built executable unusable.

Follow the actual repair remedy in [failures](failures.md). Environment repair can affect the shared root, so check its scope. Describe the check actually performed; a pinned revision is not a verified content digest.

## Reclaim only what was requested

Use named removal for selected packages; preview a whole-root purge before performing it. Follow the CLI's ownership decisions for shared caches and retained environments. Download estimates, network traffic, and reclaimed disk space are different quantities.

Do not finish cleanup by deleting cache directories manually. When uninstalling is requested, reclaim intended managed packages first because their root survives CLI removal.
