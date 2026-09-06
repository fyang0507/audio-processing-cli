# Selecting or installing the command

Read this when the selected command is missing or cannot start. When using or testing a checkout,
select `uv run audio` from that root even if another `audio` is on PATH. Verify its source identity
as described in [SKILL.md](../SKILL.md). Missing commands in an older PATH tool do not require
changing that installation to work in a checkout.

## Host dependencies

FFmpeg and FFprobe provide decoding, encoding, and measurement; installing the CLI does not provide
these machine tools. Check them with `command -v ffmpeg` and `command -v ffprobe`.
Installation needs `uv`; if it is absent, follow its official installation instructions.

## Install when requested or needed outside a checkout

Choose the source appropriate to the task:

```bash
uv tool install .                                                          # from a checkout
uv tool install "git+https://github.com/fyang0507/audio-processing-cli.git"  # without one
```

This creates an isolated tool environment. The executable normally lands in `~/.local/bin`;
if it is not on PATH, `uv tool update-shell` and a new shell make it discoverable.

Verify from outside the repository with `command -v audio` and the resolved executable's `--help`.
Then use its `doctor` to inspect host dependencies and provisioning state. This distinguishes a
command that starts from a machine ready for the selected task.

Keep checkout work on its selected launcher. Updating or removing an installed PATH tool is a
separate task; a command mismatch or model failure is not authorization for it. If removal is
requested, reclaim only the intended provisioned models first — see
[model-packages.md](model-packages.md) — because their root survives uninstalling the command.
