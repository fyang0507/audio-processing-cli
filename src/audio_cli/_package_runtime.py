"""Injected command and download boundaries for package provisioning."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import paths
from ._package_compat import facade_dependency
from ._package_core import CheckoutState, ProvisioningError, sha256_file
from ._package_requirements import _distribution_name
from .environments import Environment
from .media import (
    assert_directory_binding,
    bound_directory,
    cleanup_temporary_file,
    file_identity_from_descriptor,
    publish_temporary_file,
    sha256_regular_file_at,
)

# --------------------------------------------------------------------------------------
# Injected external surfaces
# --------------------------------------------------------------------------------------


@dataclass
class Toolchain:
    """The external commands provisioning shells out to."""

    def which(self, tool: str) -> str | None:
        return shutil.which(tool)

    def run(self, args: list[str], *, cwd: Path | None = None,
            timeout: int = 3600) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)

    def file_digest(self, path: Path) -> str:
        """Hash a checkout file; injectable so tests can model pinned upstream bytes."""
        return sha256_file(path)

    def create_environment(self, environment: Environment, target: Path) -> None:
        """`uv venv` then `uv pip sync` — no resolution at provisioning time, ever."""
        if self.which("uv") is None:
            raise ProvisioningError(
                "toolchain_missing", "uv is not on PATH and is required to create environments",
                missing_tool="uv", fix="Install uv: https://docs.astral.sh/uv/",
            )
        created = self.run(["uv", "venv", "--python", environment.python, str(target)])
        if created.returncode != 0:
            raise ProvisioningError(
                "environment_creation_failed",
                f"could not create {environment.name}: {created.stderr.strip()}",
                environment=environment.name,
            )
        synced = self.run(["uv", "pip", "sync", "--python", str(target / "bin" / "python"),
                           str(environment.lock)])
        if synced.returncode != 0:
            raise ProvisioningError(
                "environment_sync_failed",
                f"could not sync {environment.name} from its lock: {synced.stderr.strip()}",
                environment=environment.name, lock=environment.lock.name,
            )

    def frozen_packages(self, environment_python: Path) -> dict[str, str]:
        result = self.run(["uv", "pip", "freeze", "--python", str(environment_python)])
        if result.returncode != 0:
            return {}
        frozen: dict[str, str] = {}
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-e "):
                source = line[3:].strip()
                try:
                    fragment = urllib.parse.urlsplit(source).fragment
                except ValueError:
                    fragment = ""
                egg = urllib.parse.parse_qs(fragment).get("egg", [])
                name = _distribution_name(egg[0]) if egg else f"<editable:{source}>"
                frozen[name] = f"-e {source}"
            elif " @ " in line:
                name, _, source = line.partition(" @ ")
                frozen[_distribution_name(name)] = f"@ {source.strip()}"
            elif "==" in line:
                name, _, version = line.partition("==")
                frozen[_distribution_name(name)] = version.strip()
            else:
                # Unknown freeze syntax is installed state too. A stable synthetic name makes
                # equality fail closed instead of silently discarding the distribution.
                frozen[f"<unparsed:{line}>"] = line
        return frozen

    def clone(self, repo: str, commit: str, target: Path) -> None:
        if self.which("git") is None:
            raise ProvisioningError(
                "toolchain_missing", "git is not on PATH and is required for source checkouts",
                missing_tool="git",
            )
        if not target.exists():
            cloned = self.run(["git", "clone", "--quiet", repo, str(target)])
            if cloned.returncode != 0:
                raise ProvisioningError("checkout_failed",
                                        f"could not clone {repo}: {cloned.stderr.strip()}")
        checked = self.run(["git", "checkout", "--quiet", commit], cwd=target)
        if checked.returncode != 0:
            raise ProvisioningError("checkout_failed",
                                    f"could not check out {commit}: {checked.stderr.strip()}")

    def apply_patch(self, checkout: Path, patch: Path) -> None:
        applied = self.run(["git", "apply", str(patch)], cwd=checkout)
        if applied.returncode != 0:
            # Already applied is not a failure: pull must be repeatable.
            check = self.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=checkout)
            if check.returncode != 0:
                raise ProvisioningError(
                    "patch_failed",
                    f"could not apply {patch.name}: {applied.stderr.strip()}",
                    patch=patch.name,
                )

    def install_checkout(self, environment_python: Path, checkout: Path) -> None:
        result = self.run(["uv", "pip", "install", "--python", str(environment_python),
                           "--no-deps", str(checkout)])
        if result.returncode != 0:
            raise ProvisioningError(
                "checkout_install_failed",
                f"could not install {checkout.name} into the environment: "
                f"{result.stderr.strip()}",
            )

    def clean_ignored_checkout(self, checkout: Path) -> None:
        """Remove install-time ignored artifacts before the checkout becomes executable."""
        result = self.run(["git", "clean", "-fdX"], cwd=checkout)
        if result.returncode != 0:
            raise ProvisioningError(
                "checkout_cleanup_failed",
                f"could not remove ignored build artifacts from {checkout.name}: "
                f"{result.stderr.strip()}",
            )

    def inspect_checkout(self, checkout: Path) -> CheckoutState:
        """Read HEAD plus every provenance-relevant tracked and untracked path.

        Ignored bytecode and extension modules can still be imported from a checkout, so the
        second ``ls-files`` invocation is intentional.  A default ``git status`` would hide
        exactly the files this check exists to catch.
        """

        def git(*arguments: str) -> str:
            try:
                result = self.run(["git", *arguments], cwd=checkout)
            except OSError as exc:
                raise ValueError(f"could not inspect source checkout: {exc}") from exc
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise ValueError(
                    f"git {' '.join(arguments)} failed for {checkout}: {detail}"
                )
            return result.stdout

        head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
        modified = tuple(sorted(filter(None, git(
            "diff", "--name-only", "--no-ext-diff", "--no-textconv", "--no-renames",
            "HEAD", "--",
        ).splitlines())))
        ordinary_untracked = filter(None, git(
            "ls-files", "--others", "--exclude-standard",
        ).splitlines())
        ignored_untracked = filter(None, git(
            "ls-files", "--others", "--ignored", "--exclude-standard",
        ).splitlines())
        return CheckoutState(
            head=head,
            modified=modified,
            untracked=tuple(sorted({*ordinary_untracked, *ignored_untracked})),
        )

    def swift_build(self, checkout: Path) -> None:
        if self.which("swift") is None:
            raise ProvisioningError(
                "toolchain_missing",
                "swift is not on PATH; it is required only for the swift environment's packages",
                missing_tool="swift", requires_tool=["swift"],
            )
        built = self.run(["swift", "build", "-c", "release"], cwd=checkout)
        if built.returncode != 0:
            raise ProvisioningError("swift_build_failed",
                                    f"swift build failed: {built.stderr.strip()[-600:]}")

    def swift_product_runs(self, checkout: Path, product: str) -> bool:
        """Whether the built executable actually launches.

        The product name is pinned in the manifest rather than written here. It was `fluidaudio`
        in this call, and `Package.swift` declares `.executable(name: "fluidaudiocli")` -- so
        `swift run` answered `no executable product named 'fluidaudio'` on a perfectly good build
        and this check could never return True. A name that lives beside the commit it belongs to
        is reviewable when the commit moves; a literal buried in a runner is not.
        """
        result = self.run(["swift", "run", "-c", "release", product, "--help"],
                          cwd=checkout, timeout=600)
        return result.returncode == 0

    def built_product_runs(self, executable: Path) -> bool:
        """Probe an already-built product without requiring its provisioning toolchain."""
        result = self.run([str(executable), "--help"], cwd=executable.parent, timeout=600)
        return result.returncode == 0


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
        """Materialize a revision into the shared Hub cache, resuming a partial download.

        `force` re-downloads what the cache already holds, and it is what makes `pull --repair`
        mean anything for a Hub package: without it `snapshot_download` sees the revision present
        and returns the snapshot unchanged, so a repair of a corrupt snapshot would report success
        having moved no bytes.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # noqa: BLE001 - reported, not swallowed
            raise ProvisioningError(
                "toolchain_missing",
                "huggingface_hub is required to download weights",
                missing_tool="huggingface_hub",
                fix="uv pip install huggingface_hub",
            ) from exc
        return Path(snapshot_download(
            repo,
            revision=revision,
            force_download=force,
            allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        ))

    def delete_hub_revisions(self, revisions: list[str]) -> tuple[list[str], int]:
        """Delete exactly these revisions from the Hub cache. Returns what went, and its size.

        Revision-scoped rather than repo-scoped: the cache may be shared with other tools, and
        another revision of the same repository is not ours to remove. Revisions the cache no
        longer holds are reported as not deleted rather than as deleted, because a teardown
        report that overstates what it freed is worse than one that reclaims less.
        """
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
        present = {
            revision.commit_hash
            for repo in cache.repos
            for revision in repo.revisions
        }
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
                    existing = facade_dependency(
                        "sha256_regular_file_at", sha256_regular_file_at
                    )(
                        parent_descriptor, target.name
                    )
                except OSError:
                    # An unreadable cache entry cannot earn the fast path. Replacement is
                    # descriptor-relative, so retrying the download does not follow it.
                    existing = None
                if existing == sha256:
                    return target
                partial_name = (
                    f".audio-download-{os.getpid()}-{uuid.uuid4().hex}.part"
                )
                created = False
                temporary_identity = None
                try:
                    descriptor = os.open(
                        partial_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
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
                        url, headers={"User-Agent": "audio-processing-cli/0.1"})
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
                            expected=sha256, actual=actual,
                        )
                    facade_dependency(
                        "assert_directory_binding", assert_directory_binding
                    )(parent_descriptor, target.parent)
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
                        cleanup_temporary_file(
                            parent_descriptor, partial_name, temporary_identity
                        )
        except (OSError, urllib.error.URLError) as exc:
            raise ProvisioningError("download_failed",
                                    f"could not download {url}: {exc}") from exc
        return target
