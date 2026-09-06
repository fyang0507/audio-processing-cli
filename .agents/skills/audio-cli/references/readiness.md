# Check command readiness

Use the `audio` executable selected in [SKILL.md](../SKILL.md) from any directory.
Its help establishes which operations are available; `audio doctor` reports the executable path,
version, host dependencies, and managed package root. Record these CLI results when command
identity or readiness matters. A reported version alone does not identify an unversioned build.

If the command cannot start, retain the attempted command, working directory, exit code, and
stderr. Report that CLI setup is blocking the task. If a required operation is missing, retain
the available help and doctor result and report the mismatch to the operator.

For missing host dependencies, use the doctor's evidence to name the blocker. CLI installation,
updates, and host-tool setup belong to the operator; do not invoke installers, import the
implementation, edit environments, or substitute a different audio tool to continue processing.

When the command runs but a model package is missing or needs repair, use the CLI's own package
lifecycle within the requested scope; see [model-packages.md](model-packages.md). A successful
doctor call alone does not prove that the packages required by a transcription plan are ready.
