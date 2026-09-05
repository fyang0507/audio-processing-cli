"""Injected download boundary for package provisioning."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..media import (
    assert_directory_binding,
    bound_directory,
    cleanup_temporary_file,
    file_identity_from_descriptor,
    publish_temporary_file,
    sha256_regular_file_at,
)
from .models import ProvisioningError


@dataclass
class Fetcher:
    """Downloads. Hub weights stay in the Hugging Face cache; the registry records revisions."""

    def cached_revisions(self) -> set[str]:
        """Revisions the Hub cache already holds, whoever put them there."""
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            return set()
        try:
            cache = scan_cache_dir()
        except Exception:  # noqa: BLE001 - an unreadable cache holds nothing we can claim
            return set()
        return {revision.commit_hash for repo in cache.repos for revision in repo.revisions}

    def hf_snapshot(
        self,
        repo: str,
        revision: str,
        *,
        force: bool = False,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        """Materialize a revision into the shared Hub cache, resuming a partial download."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # noqa: BLE001 - reported, not swallowed
            raise ProvisioningError(
                "toolchain_missing",
                "huggingface_hub is required to download weights",
                missing_tool="huggingface_hub",
                fix="uv pip install huggingface_hub",
            ) from exc
        return Path(
            snapshot_download(
                repo,
                revision=revision,
                force_download=force,
                allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
            )
        )

    def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
        """Delete exactly these revisions from the Hub cache and report reclaimed bytes."""
        if not revisions:
            return [], 0
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            return [], 0
        try:
            cache = scan_cache_dir()
        except Exception:  # noqa: BLE001 - a missing or unreadable cache frees nothing
            return [], 0
        present = {revision.commit_hash for repo in cache.repos for revision in repo.revisions}
        deletable = sorted(set(revisions) & present)
        if not deletable:
            return [], 0
        strategy = cache.delete_revisions(*deletable)
        freed = int(strategy.expected_freed_size)
        strategy.execute()
        return deletable, freed

    def url_file(self, url: str, sha256: str, target: Path) -> Path:
        """Hash and publish through one no-follow descriptor bound to the managed parent."""
        try:
            with bound_directory(
                target.parent,
                root=paths.root(),
                create=True,
            ) as parent_descriptor:
                try:
                    existing = sha256_regular_file_at(parent_descriptor, target.name)
                except OSError:
                    # An unreadable cache entry cannot earn the fast path. Replacement is
                    # descriptor-relative, so retrying the download does not follow it.
                    existing = None
                if existing == sha256:
                    return target
                partial_name = f".audio-download-{os.getpid()}-{uuid.uuid4().hex}.part"
                created = False
                temporary_identity = None
                try:
                    descriptor = os.open(
                        partial_name,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        0o666,
                        dir_fd=parent_descriptor,
                    )
                    created = True
                    try:
                        temporary_identity = file_identity_from_descriptor(
                            descriptor, target.parent / partial_name
                        )
                    finally:
                        if temporary_identity is None:
                            os.close(descriptor)
                    if temporary_identity is None:
                        raise OSError(
                            f"download temporary is not a regular file: "
                            f"{target.parent / partial_name}"
                        )
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "audio-processing-cli/0.1"}
                    )
                    with os.fdopen(descriptor, "wb") as output:
                        response = urllib.request.urlopen(request, timeout=60)
                        with response:
                            digest = hashlib.sha256()
                            while chunk := response.read(1024 * 1024):
                                output.write(chunk)
                                digest.update(chunk)
                            output.flush()
                            os.fsync(output.fileno())
                    actual = digest.hexdigest()
                    if actual != sha256:
                        raise ProvisioningError(
                            "package_integrity_failed",
                            f"{target.name} checksum mismatch: expected {sha256}, got {actual}",
                            expected=sha256,
                            actual=actual,
                        )
                    assert_directory_binding(parent_descriptor, target.parent)
                    publish_temporary_file(
                        parent_descriptor,
                        partial_name,
                        target.name,
                        output_path=target,
                        force=True,
                        temporary_identity=temporary_identity,
                        replace_non_directory=True,
                    )
                    created = False
                finally:
                    if created and temporary_identity is not None:
                        cleanup_temporary_file(parent_descriptor, partial_name, temporary_identity)
        except (OSError, urllib.error.URLError) as exc:
            raise ProvisioningError("download_failed", f"could not download {url}: {exc}") from exc
        return target


__all__ = ["Fetcher"]
