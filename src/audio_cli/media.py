from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ENHANCED_MARKER = "audio-processing-cli enhanced"


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProtectedFileIdentity:
    """A regular file held by device/inode identity across a long operation."""

    path: Path
    device: int
    inode: int


class ProtectedOutputError(OSError):
    """Publication would replace a file captured as canonical input."""

    def __init__(
        self,
        output: Path,
        protected: Path,
        *,
        preserved_at: Path | None = None,
    ) -> None:
        self.output = Path(output)
        self.protected = Path(protected)
        self.preserved_at = Path(preserved_at) if preserved_at is not None else None
        preservation = (
            f"; protected bytes remain at {self.preserved_at}"
            if self.preserved_at is not None
            else ""
        )
        super().__init__(
            f"output {self.output} resolves to protected input {self.protected}"
            f"{preservation}"
        )


def file_identity_from_descriptor(
    descriptor: int, path: Path
) -> ProtectedFileIdentity | None:
    state = os.fstat(descriptor)
    if not stat.S_ISREG(state.st_mode):
        return None
    return ProtectedFileIdentity(Path(path), state.st_dev, state.st_ino)


def capture_file_identity(path: Path) -> ProtectedFileIdentity | None:
    """Capture an existing regular file without trusting its pathname later."""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except FileNotFoundError:
        return None
    try:
        return file_identity_from_descriptor(descriptor, Path(path))
    finally:
        os.close(descriptor)


def matching_protected_identity(
    directory_descriptor: int,
    name: str,
    identities: Sequence[ProtectedFileIdentity],
) -> ProtectedFileIdentity | None:
    """Return the captured input currently occupying a descriptor-bound name."""
    try:
        state = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(state.st_mode):
        return None
    return next(
        (
            identity
            for identity in identities
            if (state.st_dev, state.st_ino) == (identity.device, identity.inode)
        ),
        None,
    )


def entry_matches_file_identity(
    directory_descriptor: int,
    name: str,
    identity: ProtectedFileIdentity,
) -> bool:
    """Return whether a descriptor-bound name still denotes one captured file."""
    try:
        state = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(state.st_mode)
        and (state.st_dev, state.st_ino) == (identity.device, identity.inode)
    )


def _rename_exchange(
    directory_descriptor: int, left_name: str, right_name: str
) -> None:
    """Atomically exchange two descriptor-relative directory entries."""
    library = ctypes.CDLL(None, use_errno=True)
    left = os.fsencode(left_name)
    right = os.fsencode(right_name)
    if sys.platform == "darwin":
        try:
            function = library.renameatx_np
        except AttributeError as exc:
            raise OSError(errno.ENOTSUP, "atomic rename exchange is unavailable") from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
    elif sys.platform.startswith("linux"):
        try:
            function = library.renameat2
        except AttributeError as exc:
            raise OSError(errno.ENOTSUP, "atomic rename exchange is unavailable") from exc
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
    else:
        raise OSError(errno.ENOTSUP, "atomic rename exchange is unavailable")
    if function(
        directory_descriptor,
        left,
        directory_descriptor,
        right,
        0x00000002,
    ) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), right_name)


def publish_temporary_file(
    directory_descriptor: int,
    temporary_name: str,
    output_name: str,
    *,
    output_path: Path,
    force: bool,
    temporary_identity: ProtectedFileIdentity,
    protected_identities: Sequence[ProtectedFileIdentity] = (),
    replace_non_directory: bool = False,
) -> None:
    """Claim a destination without a check/replace race.

    A force write atomically exchanges the temporary and current destination.  The
    displaced inode then occupies the private temporary name and can be inspected before
    deletion; a protected or refused inode is atomically exchanged back.  Managed-cache
    repair may explicitly replace a non-directory leaf (for example, a symlink) without
    following it.  Directories are always refused.  When no destination exists, a hard
    link remains the atomic create-if-absent primitive.
    """
    if not entry_matches_file_identity(
        directory_descriptor, temporary_name, temporary_identity
    ):
        raise OSError(
            f"writer temporary changed identity before publication: {output_path}"
        )

    if not force:
        protected = matching_protected_identity(
            directory_descriptor, output_name, protected_identities
        )
        if protected is not None:
            raise ProtectedOutputError(output_path, protected.path)
        os.link(
            temporary_name,
            output_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not entry_matches_file_identity(
            directory_descriptor, output_name, temporary_identity
        ):
            raise OSError(
                f"published destination does not contain the writer temporary: "
                f"{output_path}"
            )
        cleanup_temporary_file(
            directory_descriptor, temporary_name, temporary_identity
        )
        return

    for _attempt in range(8):
        try:
            os.stat(
                output_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            try:
                os.link(
                    temporary_name,
                    output_name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                continue
            if not entry_matches_file_identity(
                directory_descriptor, output_name, temporary_identity
            ):
                raise OSError(
                    f"published destination does not contain the writer temporary: "
                    f"{output_path}"
                )
            cleanup_temporary_file(
                directory_descriptor, temporary_name, temporary_identity
            )
            return
        try:
            _rename_exchange(
                directory_descriptor, temporary_name, output_name
            )
        except FileNotFoundError:
            continue

        if not entry_matches_file_identity(
            directory_descriptor, output_name, temporary_identity
        ):
            try:
                _rename_exchange(
                    directory_descriptor, temporary_name, output_name
                )
            except OSError as exc:
                raise OSError(
                    f"could not atomically restore destination after writer "
                    f"temporary identity changed for {output_path}: {exc}"
                ) from exc
            raise OSError(
                f"writer temporary changed identity during publication: {output_path}"
            )

        state = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        protected = matching_protected_identity(
            directory_descriptor, temporary_name, protected_identities
        )
        refused_kind = not stat.S_ISREG(state.st_mode) and (
            not replace_non_directory or stat.S_ISDIR(state.st_mode)
        )
        if refused_kind or protected is not None:
            try:
                _rename_exchange(
                    directory_descriptor, temporary_name, output_name
                )
            except OSError as exc:
                if protected is not None:
                    raise ProtectedOutputError(
                        output_path,
                        protected.path,
                        preserved_at=output_path.parent / temporary_name,
                    ) from exc
                raise OSError(
                    f"could not atomically restore refused destination {output_path}: {exc}"
                ) from exc
            if protected is not None:
                raise ProtectedOutputError(output_path, protected.path)
            raise OSError(
                f"destination changed to a non-regular file during publication: "
                f"{output_path}"
            )
        os.unlink(temporary_name, dir_fd=directory_descriptor)
        return
    raise FileExistsError(
        f"destination changed repeatedly during publication: {output_path}"
    )


def cleanup_temporary_file(
    directory_descriptor: int,
    temporary_name: str,
    temporary_identity: ProtectedFileIdentity,
) -> None:
    """Best-effort removal when the private name still matches at inspection.

    The random sibling is outside the public-path race boundary: a same-credential
    process that discovers and mutates it after this check can already unlink the
    caller's canonical files directly, and POSIX has no portable unlink-if-inode
    primitive.
    """
    if not entry_matches_file_identity(
        directory_descriptor, temporary_name, temporary_identity
    ):
        return
    os.unlink(temporary_name, dir_fd=directory_descriptor)


def _run(
    args: Sequence[str], *, capture_stdout: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            list(args),
            check=True,
            stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MediaError(f"Required executable not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise MediaError(f"Command failed ({args[0]}): {detail}") from exc


def require_runtime() -> None:
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise MediaError(
                f"{executable} is required. Install FFmpeg and ensure both ffmpeg and ffprobe are on PATH."
            )


def ffmpeg_version() -> str:
    result = _run(["ffmpeg", "-version"])
    first = result.stdout.decode("utf-8", errors="replace").splitlines()[0]
    return first.strip()


def probe_media(path: Path) -> dict[str, object]:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path}") from exc
    audio_streams = [
        s for s in payload.get("streams", []) if s.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise MediaError(f"No audio stream found in {path}")
    payload["primary_audio_stream"] = audio_streams[0]
    payload["has_video"] = any(
        s.get("codec_type") == "video" for s in payload.get("streams", [])
    )
    return payload


def media_summary(path: Path, probe: dict[str, object]) -> dict[str, object]:
    stream = probe["primary_audio_stream"]
    assert isinstance(stream, dict)
    fmt = probe.get("format", {})
    assert isinstance(fmt, dict)
    duration_raw = stream.get("duration", fmt.get("duration", 0.0))
    tags = fmt.get("tags", {}) or {}
    return {
        "path": str(path.resolve()),
        "sha256": hash_file(path),
        "duration_seconds": round(float(duration_raw), 6),
        "audio_start_seconds": round(float(stream.get("start_time", 0.0) or 0.0), 6),
        "sample_rate_hz": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "channel_layout": stream.get("channel_layout"),
        "audio_codec": stream.get("codec_name"),
        "has_video": bool(probe.get("has_video")),
        "format_name": fmt.get("format_name"),
        "tags": tags,
    }


def is_enhanced_media(probe: dict[str, object]) -> bool:
    fmt = probe.get("format", {})
    if not isinstance(fmt, dict):
        return False
    tags = fmt.get("tags", {}) or {}
    if not isinstance(tags, dict):
        return False
    text = " ".join(str(value) for value in tags.values()).lower()
    return ENHANCED_MARKER in text


def decode_audio(path: Path, *, sample_rate: int = 48_000) -> tuple[np.ndarray, int]:
    probe = probe_media(path)
    stream = probe["primary_audio_stream"]
    assert isinstance(stream, dict)
    source_channels = int(stream.get("channels", 1) or 1)
    if source_channels > 2:
        raise MediaError(
            f"V1 supports mono or stereo sources; {path} has {source_channels} audio channels"
        )
    channels = 1 if source_channels == 1 else 2
    result = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ]
    )
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if samples.size == 0 or samples.size % channels:
        raise MediaError(f"Could not decode a complete audio stream from {path}")
    return samples.reshape((-1, channels)), sample_rate


def _wav_duration_ms(path: Path) -> float:
    """Duration of a wav this tool wrote, read through the header rather than the samples."""
    rate, data = wavfile.read(path, mmap=True)
    return len(data) / float(rate) * 1000.0


def write_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.asarray(samples, dtype=np.float32))


def _extract_loudnorm_json(stderr: bytes) -> dict[str, float]:
    text = stderr.decode("utf-8", errors="replace")
    matches = re.findall(r"\{\s*\"input_i\"\s*:.*?\}", text, flags=re.DOTALL)
    if not matches:
        raise MediaError("FFmpeg loudnorm did not emit measurement JSON")
    raw = json.loads(matches[-1])
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            result[key] = math.nan
    return result


def measure_loudness(
    path: Path,
    *,
    target_lufs: float = -16.0,
    target_lra: float = 11.0,
    target_true_peak: float = -1.5,
) -> dict[str, float]:
    filter_spec = (
        f"loudnorm=I={target_lufs}:LRA={target_lra}:TP={target_true_peak}:"
        "print_format=json"
    )
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-af",
                filter_spec,
                "-f",
                "null",
                "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"").decode("utf-8", errors="replace")[-4000:]
        raise MediaError(f"Could not measure loudness for {path}: {detail}") from exc
    return _extract_loudnorm_json(result.stderr)


def render_loudness_normalized(
    source_wav: Path,
    output_wav: Path,
    *,
    target_lufs: float,
    target_lra: float,
    target_true_peak: float,
    measurement: dict[str, float],
    sample_rate: int,
) -> dict[str, object]:
    required = ("input_i", "input_lra", "input_tp", "input_thresh")
    if any(not math.isfinite(measurement.get(key, math.nan)) for key in required):
        # Two different conditions, and reporting them as one sent anyone with a short clip
        # looking for silence that was not there. A finite true peak is the evidence that the
        # signal exists: integrated loudness is gated in 400 ms blocks (EBU R128), so anything
        # shorter has no measurable value however loud it is, while true silence has no finite
        # peak either.
        peak = measurement.get("input_tp", math.nan)
        if math.isfinite(peak):
            raise MediaError(
                f"Cannot normalize {_wav_duration_ms(source_wav):.0f} ms of audio: integrated "
                f"loudness is measured over 400 ms gating blocks (EBU R128), so a shorter input "
                f"has none however loud it is. This input peaks at {peak:.2f} dBTP, so it is not "
                f"silent -- it is too short to measure. Enhance a longer excerpt, or skip "
                f"program-loudness with --skip program-loudness."
            )
        raise MediaError("Cannot normalize silent audio: it has no measurable level or peak")
    gain_db = target_lufs - measurement["input_i"]
    limiter_amplitude = 10.0 ** (target_true_peak / 20.0)
    iterations: list[dict[str, float]] = []
    measured_output: dict[str, float] = measurement
    for iteration in range(1, 6):
        filter_spec = (
            f"volume={gain_db:.9f}dB,"
            "aresample=192000,"
            f"alimiter=limit={limiter_amplitude:.9f}:attack=5:release=50:"
            "level=false:latency=true,"
            f"aresample={sample_rate}"
        )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(source_wav),
                "-af",
                filter_spec,
                "-c:a",
                "pcm_f32le",
                "-ar",
                str(sample_rate),
                "-y",
                str(output_wav),
            ],
            capture_stdout=False,
        )
        measured_output = measure_loudness(
            output_wav,
            target_lufs=target_lufs,
            target_lra=target_lra,
            target_true_peak=target_true_peak,
        )
        error_lu = target_lufs - measured_output["input_i"]
        iterations.append(
            {
                "iteration": float(iteration),
                "input_gain_db": round(gain_db, 6),
                "measured_lufs": round(measured_output["input_i"], 6),
                "measured_true_peak_dbtp": round(measured_output["input_tp"], 6),
                "loudness_error_lu": round(error_lu, 6),
            }
        )
        if abs(error_lu) <= 0.05:
            break
        gain_db += error_lu
    return {
        "method": "ebu-r128-measured-linear-gain-with-oversampled-true-peak-limiter",
        "resolved_input_gain_db": round(gain_db, 6),
        "limiter_ceiling_dbtp": target_true_peak,
        "limiter_oversample_hz": 192_000,
        "iterations": iterations,
        "measured_output": {
            key: round(value, 6) if math.isfinite(value) else None
            for key, value in sorted(measured_output.items())
        },
    }


def encode_output(
    original: Path,
    enhanced_wav: Path,
    output: Path,
    *,
    original_sha256: str,
    has_video: bool,
) -> None:
    suffix = output.suffix.lower()
    metadata = f"{ENHANCED_MARKER}; original_sha256={original_sha256}"
    if has_video and suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        if suffix == ".webm":
            audio_codec = "libopus"
            audio_args = ["-c:a", audio_codec, "-b:a", "192k"]
        elif suffix == ".mkv":
            audio_args = ["-c:a", "flac"]
        else:
            audio_args = ["-c:a", "aac", "-b:a", "192k"]
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(original),
            "-i",
            str(enhanced_wav),
            "-map",
            "0:v?",
            "-map",
            "1:a:0",
            "-map_metadata",
            "0",
            "-c:v",
            "copy",
            *audio_args,
            "-metadata",
            f"comment={metadata}",
        ]
        if suffix in {".mp4", ".mov"}:
            args.extend(["-movflags", "+faststart"])
        args.extend(["-y", str(output)])
        _run(args, capture_stdout=False)
        return

    codec_args: list[str]
    if suffix == ".wav":
        codec_args = ["-c:a", "pcm_s24le"]
    elif suffix == ".flac":
        codec_args = ["-c:a", "flac"]
    elif suffix == ".mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif suffix in {".m4a", ".aac", ".mp4"}:
        codec_args = ["-c:a", "aac", "-b:a", "192k"]
    elif suffix in {".ogg", ".opus"}:
        codec_args = ["-c:a", "libopus", "-b:a", "160k"]
    else:
        raise MediaError(
            f"Unsupported output extension {suffix!r}; use wav, flac, mp3, m4a, ogg, opus, mp4, mov, mkv, or webm"
        )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(enhanced_wav),
            *codec_args,
            "-metadata",
            f"comment={metadata}",
            "-y",
            str(output),
        ],
        capture_stdout=False,
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_open_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise OSError("this platform cannot open managed directories without following links")
    return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


def assert_directory_binding(descriptor: int, directory: Path) -> None:
    """Require a pathname to still name the directory held open by ``descriptor``."""

    current = os.stat(directory, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(
        os.fstat(descriptor), current
    ):
        raise OSError(f"managed directory identity changed during operation: {directory}")


def assert_resolved_directory_binding(descriptor: int, directory: Path) -> None:
    """Require a possibly symlinked caller path to still resolve to an open directory."""

    current = os.stat(directory, follow_symlinks=True)
    if not stat.S_ISDIR(current.st_mode) or not os.path.samestat(
        os.fstat(descriptor), current
    ):
        raise OSError(f"output directory identity changed during operation: {directory}")


@contextmanager
def bound_directory(
    directory: Path,
    *,
    root: Path,
    create: bool,
    verify_on_exit: bool = True,
) -> Iterator[int]:
    """Open a contained directory chain once, refusing symlinks at every component.

    Callers perform creation, replacement, or deletion relative to the yielded descriptor.
    Renaming a checked parent and replacing its pathname with a symlink therefore cannot retarget
    the operation outside the managed root.
    """

    root_path = Path(os.path.abspath(root))
    directory_path = Path(os.path.abspath(directory))
    if not directory_path.is_relative_to(root_path):
        raise OSError(
            f"managed directory is outside provisioning root: {directory_path}"
        )
    if create:
        root_path.mkdir(parents=True, exist_ok=True)
    flags = _directory_open_flags()
    descriptors: list[int] = []
    try:
        current = os.open(root_path, flags)
        descriptors.append(current)
        for part in directory_path.relative_to(root_path).parts:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o777, dir_fd=current)
                child = os.open(part, flags, dir_fd=current)
            descriptors.append(child)
            current = child
        try:
            yield current
        except BaseException:
            raise
        else:
            if verify_on_exit:
                assert_directory_binding(current, directory_path)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def sha256_regular_file_at(directory_descriptor: int, name: str) -> str | None:
    """Hash one descriptor-relative regular file without following a leaf symlink."""

    try:
        before = os.stat(
            name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        return None
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=directory_descriptor,
    )
    try:
        if not os.path.samestat(before, os.fstat(descriptor)):
            raise OSError(f"managed file identity changed while opening: {name}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


@contextmanager
def temporary_directory(prefix: str = "audio-processing-") -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix) as raw:
        yield Path(raw)


@contextmanager
def temporary_output_path(output: Path) -> Iterator[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temp = Path(raw)
    temp.unlink(missing_ok=True)
    try:
        yield temp
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    force: bool = True,
    protected_paths: Sequence[Path] = (),
    protected_identities: Sequence[ProtectedFileIdentity] = (),
) -> None:
    """Publish text through an exclusive sibling, optionally without clobbering."""
    identities = list(protected_identities)
    for protected_path in protected_paths:
        identity = capture_file_identity(Path(protected_path))
        if identity is not None and all(
            (identity.device, identity.inode) != (known.device, known.inode)
            for known in identities
        ):
            identities.append(identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = path.parent.resolve(strict=True)
    anchor = Path(resolved_parent.anchor)
    with bound_directory(
        resolved_parent,
        root=anchor,
        create=False,
    ) as parent_descriptor:
        temporary_name = f".audio-write-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        created = False
        temporary_identity = None
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o666,
                dir_fd=parent_descriptor,
            )
            created = True
            try:
                temporary_identity = file_identity_from_descriptor(
                    descriptor, path.parent / temporary_name
                )
            finally:
                if temporary_identity is None:
                    os.close(descriptor)
            if temporary_identity is None:
                raise OSError(
                    f"writer temporary is not a regular file: "
                    f"{path.parent / temporary_name}"
                )
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            assert_resolved_directory_binding(parent_descriptor, path.parent)
            publish_temporary_file(
                parent_descriptor,
                temporary_name,
                path.name,
                output_path=path,
                force=force,
                temporary_identity=temporary_identity,
                protected_identities=identities,
            )
            created = False
        finally:
            if created and temporary_identity is not None:
                cleanup_temporary_file(
                    parent_descriptor, temporary_name, temporary_identity
                )


def atomic_write_json(
    path: Path,
    payload: dict[str, object],
    *,
    force: bool = True,
    protected_paths: Sequence[Path] = (),
    protected_identities: Sequence[ProtectedFileIdentity] = (),
) -> None:
    """Written through a sibling and renamed, so a report is never observed half-written.

    Deliberately a plain `open` rather than `mkstemp`. `mkstemp` creates 0600 and `os.replace`
    carries that mode onto the destination, so a report arrived stricter than the render it
    describes -- 0600 beside an 0644 wav, under the same umask, for no reason a reader could act
    on. Everything else this tool publishes honours the umask, including `save_registry`, which
    reached the same conclusion by writing the same way.
    """
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    atomic_write_text(
        path,
        text,
        force=force,
        protected_paths=protected_paths,
        protected_identities=protected_identities,
    )
