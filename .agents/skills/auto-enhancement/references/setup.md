# Set Up the Audio CLI

Read this reference only when `command -v audio` fails or the installed command cannot start. Treat `uv` as the installer and tool manager, not as part of normal audio commands.

## Prerequisites

Require FFmpeg and FFprobe:

```bash
command -v ffmpeg
command -v ffprobe
```

Require `uv` for installation. Follow the official `uv` installation method when it is absent; do not invent a shell installer URL.

## Install

From a local checkout, install a non-editable snapshot into an isolated user-level environment:

```bash
uv tool install .
```

After the repository is published, install without a checkout:

```bash
uv tool install "git+https://github.com/fyang0507/audio-processing-cli.git"
```

The installed executable normally lives in `~/.local/bin`. If that directory is not on `PATH`, run this once and open a new shell:

```bash
uv tool update-shell
```

## Verify

Verify from outside the repository:

```bash
command -v audio
audio --help
```

Confirm the command resolves through the user tool directory rather than `.venv` or the current checkout. The isolated installation owns Python and package dependencies; FFmpeg and FFprobe remain machine runtime dependencies.

## Update or remove

Reinstall an updated local checkout:

```bash
uv tool install --force --reinstall .
```

Use `--reinstall` for unversioned checkout changes so `uv` rebuilds the local wheel instead of reusing the cached release version.

Remove the tool:

```bash
uv tool uninstall audio-processing-cli
```

Use `uv run audio ...` only to exercise an uninstalled development checkout. Return to ordinary `audio ...` commands after setup succeeds.
