# Installing the command

Read this only when `command -v audio` finds nothing, or the command will not start. Inside a
checkout of this repository, `uv run audio ...` from the root works without installing anything, so
try that first before installing on someone's machine.

## What the machine has to supply

FFmpeg and FFprobe do the decoding, encoding, and measurement. They are machine tools, not Python
packages, so no installation of this CLI provides them:

```bash
command -v ffmpeg
command -v ffprobe
```

Installing needs `uv`. If it is absent, follow the official `uv` installation instructions rather
than inventing an installer command.

## Install

```bash
uv tool install .                                                          # from a checkout
uv tool install "git+https://github.com/fyang0507/audio-processing-cli.git"  # without one
```

`uv tool install` puts the command in an isolated environment of its own, which is why it is
preferred over installing into whatever environment happens to be active. The executable normally
lands in `~/.local/bin`; if that is not on `PATH`, run `uv tool update-shell` once and open a new
shell.

## Verify

Check from outside the repository, and confirm the command resolves through the user tool directory
rather than a project `.venv`:

```bash
command -v audio
audio doctor
```

`doctor` is the real verification: it reports everything the installation does *not* own — FFmpeg,
FFprobe, the toolchains, memory, disk, and the provisioning root — so it tells you whether the
machine can actually do the work, not merely whether the command starts.

## Update or remove

```bash
uv tool install --force --reinstall .        # after changing a checkout
uv tool uninstall audio-processing-cli
```

Use `--reinstall` when a checkout changed without its version changing, or the cached build gets
reused and the changes never land. Reclaim provisioned models *before* uninstalling — see
[model-packages.md](model-packages.md) — because the provisioning root does not go away with the
command.
