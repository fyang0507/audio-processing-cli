"""Public media probing, filesystem, and publication API."""

from .errors import MediaError
from .ffmpeg import (
    ENHANCED_MARKER,
    canonical_decode_command,
    decode_audio,
    encode_output,
    ffmpeg_version,
    is_enhanced_media,
    measure_loudness,
    media_summary,
    probe_media,
    render_loudness_normalized,
    require_runtime,
    write_float_wav,
)
from .files import (
    assert_directory_binding,
    assert_resolved_directory_binding,
    bound_directory,
    hash_file,
    resolve_path_identity,
    sha256_regular_file_at,
    temporary_directory,
    temporary_output_path,
)
from .identity import (
    ProtectedFileIdentity,
    capture_file_identity,
    entry_matches_file_identity,
    file_identity_from_descriptor,
    matching_protected_identity,
)
from .pcm import (
    EmptyPcmRangeError,
    canonical_pcm_duration,
    clip_canonical_pcm,
    read_canonical_pcm16,
)
from .process import run_in_directory
from .publication import (
    ProtectedOutputError,
    atomic_write_json,
    atomic_write_text,
    cleanup_temporary_file,
    publish_temporary_file,
)

__all__ = [
    "ENHANCED_MARKER",
    "EmptyPcmRangeError",
    "MediaError",
    "ProtectedFileIdentity",
    "ProtectedOutputError",
    "assert_directory_binding",
    "assert_resolved_directory_binding",
    "atomic_write_json",
    "atomic_write_text",
    "bound_directory",
    "canonical_decode_command",
    "canonical_pcm_duration",
    "capture_file_identity",
    "cleanup_temporary_file",
    "clip_canonical_pcm",
    "decode_audio",
    "encode_output",
    "entry_matches_file_identity",
    "ffmpeg_version",
    "file_identity_from_descriptor",
    "hash_file",
    "is_enhanced_media",
    "matching_protected_identity",
    "measure_loudness",
    "media_summary",
    "probe_media",
    "publish_temporary_file",
    "read_canonical_pcm16",
    "render_loudness_normalized",
    "require_runtime",
    "resolve_path_identity",
    "run_in_directory",
    "sha256_regular_file_at",
    "temporary_directory",
    "temporary_output_path",
    "write_float_wav",
]
