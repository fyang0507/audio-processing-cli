"""Run a host command in an already-open directory without revisiting its pathname."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def run_in_directory(
    args: list[str], directory_descriptor: int, *, timeout: int = 3600
) -> subprocess.CompletedProcess[str]:
    """Inherit one directory fd, fchdir in a fresh child, then replace it with the command.

    The caller keeps the descriptor open for the duration of this operation. No parent
    chdir or thread-unsafe preexec_fn is used. A descriptor pathname such as /dev/fd/N
    cannot serve as cwd on every POSIX host, including macOS, so the stdlib-only child
    entry below performs fchdir before exec. The target keeps the same child PID.
    """
    if not args:
        raise ValueError("a command is required")
    if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
        raise NotADirectoryError("command directory descriptor is not a directory")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(Path(__file__).resolve()),
            str(directory_descriptor),
            *args,
        ],
        pass_fds=(directory_descriptor,),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


if __name__ == "__main__":
    descriptor = int(sys.argv[1])
    os.fchdir(descriptor)
    os.close(descriptor)
    os.execvp(sys.argv[2], sys.argv[2:])
