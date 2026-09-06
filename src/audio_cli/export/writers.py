"""Pure export renderers plus a race-safe atomic text writer."""

from __future__ import annotations

import html
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..media import (
    ProtectedFileIdentity,
    ProtectedOutputError,
    assert_resolved_directory_binding,
    bound_directory,
    capture_file_identity,
    cleanup_temporary_file,
    file_identity_from_descriptor,
    publish_temporary_file,
    resolve_path_identity,
)
from .cues import Cue
from .errors import OutputExistsError, OutputWriteError, UnsafeOutputError
from .timing import readable_milliseconds


def _clock(total_ms: int, separator: str) -> str:
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def render_srt(cues: Sequence[Cue]) -> str:
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.extend(
            (
                str(index),
                f"{_clock(cue.start_ms, ',')} --> {_clock(cue.end_ms, ',')}",
                cue.text,
                "",
            )
        )
    return "\n".join(lines)


def normalize_voice_annotation(speaker: str) -> str | None:
    """Return a single-line WebVTT voice annotation, or none for no usable label."""
    printable = "".join(
        character for character in speaker if character.isprintable() or character.isspace()
    )
    voice = " ".join(printable.split())
    # A WebVTT voice annotation is cue text, not an HTML attribute.  WebVTT's
    # named character references cover ampersand and angle brackets; quotes are
    # ordinary label characters and must round-trip unchanged.
    return html.escape(voice, quote=False) if voice else None


def render_vtt(cues: Sequence[Cue]) -> str:
    if not cues:
        # The WebVTT body starts only after the blank line terminating the
        # header; even a cue-less document therefore needs two line endings.
        return "WEBVTT\n\n"
    lines: list[str] = ["WEBVTT", ""]
    for index, cue in enumerate(cues, start=1):
        text = html.escape(cue.text, quote=False)
        if cue.speaker is not None:
            # Voice annotations are single-line WebVTT tokens.  Model-provided
            # speaker labels are display data, so collapse whitespace, remove
            # control characters, and omit an empty label rather than permitting a
            # cue-text injection or emitting an invalid ``<v    >`` tag.
            annotation = normalize_voice_annotation(cue.speaker)
            if annotation is not None:
                text = f"<v {annotation}>{text}"
        lines.extend(
            (
                str(index),
                f"{_clock(cue.start_ms, '.')} --> {_clock(cue.end_ms, '.')}",
                text,
                "",
            )
        )
    return "\n".join(lines)


def _render_segments(segments: Sequence[Mapping[str, Any]], *, timestamps: bool) -> list[str]:
    lines = []
    for index, segment in enumerate(segments):
        prefix = ""
        if timestamps:
            bounds = readable_milliseconds(segment, segment_index=index)
            if bounds is None:
                raise ValueError("readable timestamps require segment or word bounds")
            start, end = (_clock(value, ".") for value in bounds)
            prefix = f"[{start} --> {end}] "
        speaker = f"[{segment['speaker']}] " if "speaker" in segment else ""
        lines.append(prefix + speaker + str(segment["text"]))
    return lines


def render_text(segments: Sequence[Mapping[str, Any]], *, timestamps: bool = False) -> str:
    lines = _render_segments(segments, timestamps=timestamps)
    return "\n".join(lines) + ("\n" if lines else "")


def render_markdown(segments: Sequence[Mapping[str, Any]], *, timestamps: bool = False) -> str:
    paragraphs = _render_segments(segments, timestamps=timestamps)
    return "# Transcript\n\n" + "\n\n".join(paragraphs) + ("\n" if paragraphs else "")


def render_jsonl(segments: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        json.dumps(
            dict(segment),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for segment in segments
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _resolved(path: Path) -> Path:
    try:
        # The CLI opens input and output paths literally.  Preserve that identity
        # here as well: a leading ``~name`` is a valid literal directory name and
        # must not become a late account lookup during the protected-path check.
        return resolve_path_identity(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OutputWriteError(path, str(exc)) from exc


def _same_file(left: Path, right: Path) -> bool:
    if _resolved(left) == _resolved(right):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def write_text_atomic(
    path: Path,
    text: str,
    *,
    force: bool = False,
    protected_paths: Sequence[Path] = (),
    protected_identities: Sequence[ProtectedFileIdentity] = (),
) -> None:
    """Publish UTF-8 text atomically without following a destination symlink.

    Without ``force``, a sibling hard-link is the atomic create-if-absent operation; the
    preflight ``exists`` check is only for a clearer error and is not relied on for safety.
    """
    output = Path(path)
    _resolved(output)
    identities = list(protected_identities)
    for protected in protected_paths:
        protected_path = Path(protected)
        try:
            captured = capture_file_identity(protected_path)
        except OSError as exc:
            raise OutputWriteError(
                output,
                f"protected path {protected_path} cannot be opened safely: {exc}",
            ) from exc
        if captured is not None and all(
            (captured.device, captured.inode) != (known.device, known.inode) for known in identities
        ):
            identities.append(captured)
        try:
            same_file = _same_file(output, protected_path)
        except OutputWriteError as exc:
            # A protected path can change after the input was validated.  Publication
            # must still fail closed, but the write error's output identity remains the
            # caller's destination rather than being replaced by the protected path.
            raise OutputWriteError(
                output,
                f"protected path {protected_path} cannot be resolved safely: {exc.reason}",
            ) from exc
        if same_file:
            raise UnsafeOutputError(output, Path(protected))
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = output.parent.resolve(strict=True)
        with bound_directory(
            resolved_parent,
            root=Path(resolved_parent.anchor),
            create=False,
        ) as parent_descriptor:
            try:
                output_mode = os.stat(
                    output.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                ).st_mode
            except FileNotFoundError:
                output_mode = None
            if output_mode is not None:
                if not stat.S_ISREG(output_mode):
                    # `--force` authorizes replacing an existing regular export, never a
                    # directory, symlink, FIFO, socket, or device node.
                    raise OutputExistsError(output, replaceable=False)
                if not force:
                    raise OutputExistsError(output)

            # A legal destination may already consume the filesystem's entire NAME_MAX.
            # Keep the sibling temporary name independent of it so atomic publication does
            # not reject an output the filesystem itself accepts.
            temporary_name = f".audio-export-{os.getpid()}-{uuid.uuid4().hex}.tmp"
            assert_resolved_directory_binding(parent_descriptor, output.parent)
            for protected in protected_paths:
                if _same_file(output, Path(protected)):
                    raise UnsafeOutputError(output, Path(protected))
            created = False
            temporary_identity = None
            try:
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o666,
                    dir_fd=parent_descriptor,
                )
                created = True
                try:
                    temporary_identity = file_identity_from_descriptor(
                        descriptor, output.parent / temporary_name
                    )
                finally:
                    if temporary_identity is None:
                        os.close(descriptor)
                if temporary_identity is None:
                    raise OSError(
                        f"writer temporary is not a regular file: {output.parent / temporary_name}"
                    )
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                assert_resolved_directory_binding(parent_descriptor, output.parent)
                try:
                    publish_temporary_file(
                        parent_descriptor,
                        temporary_name,
                        output.name,
                        output_path=output,
                        force=force,
                        temporary_identity=temporary_identity,
                        protected_identities=identities,
                    )
                except ProtectedOutputError as exc:
                    raise UnsafeOutputError(output, exc.protected) from exc
                except FileExistsError as exc:
                    raise OutputExistsError(output) from exc
                created = False
            finally:
                if created and temporary_identity is not None:
                    cleanup_temporary_file(parent_descriptor, temporary_name, temporary_identity)
    except (OutputExistsError, UnsafeOutputError):
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise OutputWriteError(output, str(exc)) from exc
