"""Injected command boundary for package provisioning."""

from __future__ import annotations

import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..environments import Environment
from ..media import bound_directory, run_in_directory
from .integrity import sha256_file
from .models import CheckoutState, ProvisioningError
from .requirements import _distribution_name

# --------------------------------------------------------------------------------------
# Injected command surface
# --------------------------------------------------------------------------------------


@dataclass
class Toolchain:
    """The external commands provisioning shells out to."""

    def which(self, tool: str) -> str | None:
        return shutil.which(tool)

    def run(
        self, args: list[str], *, cwd: Path | int | None = None, timeout: int = 3600
    ) -> subprocess.CompletedProcess:
        if isinstance(cwd, int):
            return run_in_directory(args, cwd, timeout=timeout)
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)

    def file_digest(self, path: Path) -> str:
        """Hash a checkout file; injectable so tests can model pinned upstream bytes."""
        return sha256_file(path)

    def create_environment(self, environment: Environment, target: Path) -> None:
        """`uv venv` then `uv pip sync` — no resolution at provisioning time, ever."""
        if self.which("uv") is None:
            raise ProvisioningError(
                "toolchain_missing",
                "uv is not on PATH and is required to create environments",
                missing_tool="uv",
                fix="Install uv: https://docs.astral.sh/uv/",
            )
        created = self.run(["uv", "venv", "--python", environment.python, str(target)])
        if created.returncode != 0:
            raise ProvisioningError(
                "environment_creation_failed",
                f"could not create {environment.name}: {created.stderr.strip()}",
                environment=environment.name,
            )
        synced = self.run(
            ["uv", "pip", "sync", "--python", str(target / "bin" / "python"), str(environment.lock)]
        )
        if synced.returncode != 0:
            raise ProvisioningError(
                "environment_sync_failed",
                f"could not sync {environment.name} from its lock: {synced.stderr.strip()}",
                environment=environment.name,
                lock=environment.lock.name,
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
                "toolchain_missing",
                "git is not on PATH and is required for source checkouts",
                missing_tool="git",
            )
        if not target.exists():
            cloned = self.run(["git", "clone", "--quiet", repo, str(target)])
            if cloned.returncode != 0:
                raise ProvisioningError(
                    "checkout_failed", f"could not clone {repo}: {cloned.stderr.strip()}"
                )
        checked = self.run(["git", "checkout", "--quiet", commit], cwd=target)
        if checked.returncode != 0:
            raise ProvisioningError(
                "checkout_failed", f"could not check out {commit}: {checked.stderr.strip()}"
            )

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
        result = self.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment_python),
                "--no-deps",
                str(checkout),
            ]
        )
        if result.returncode != 0:
            raise ProvisioningError(
                "checkout_install_failed",
                f"could not install {checkout.name} into the environment: {result.stderr.strip()}",
            )

    def clean_checkout_install_artifacts(self, checkout: Path) -> None:
        """Remove install residue after the caller proved an empty untracked baseline.

        Setuptools creates ordinary build/ and egg-info paths as well as ignored bytecode.
        This is build-transaction cleanup, never a verification exemption or a repair of
        an unverified checkout. A single -f deliberately leaves nested Git repositories
        alone; the caller's post-install integrity check will refuse such residue.
        """
        try:
            with bound_directory(checkout, root=paths.root(), create=False) as descriptor:
                self.require_checkout_binding(checkout)
                # The child starts in this held directory even if the checkout pathname
                # is replaced before launch. Git's worktree and metadata are relative to
                # that cwd; no cleanup argument can lead back through the replaced path.
                result = self.run(
                    ["git", "--git-dir=.git", "--work-tree=.", "clean", "-fdx"],
                    cwd=descriptor,
                )
        except (OSError, ValueError) as exc:
            raise ProvisioningError("checkout_cleanup_failed", str(exc)) from exc
        if result.returncode != 0:
            raise ProvisioningError(
                "checkout_cleanup_failed",
                f"could not remove install artifacts from {checkout.name}: {result.stderr.strip()}",
            )

    def require_checkout_binding(self, checkout: Path) -> None:
        """Require Git's worktree and metadata to be this ordinary managed clone."""
        try:
            root = checkout.resolve(strict=True)
            metadata = root / ".git"
            if metadata.is_symlink() or not metadata.is_dir():
                raise ValueError(f"checkout Git metadata is not a local directory: {metadata}")
            for name in ("config", "index"):
                if (metadata / name).is_symlink():
                    raise ValueError(f"checkout Git metadata is redirected: {metadata / name}")
            result = self.run(
                [
                    "git",
                    "rev-parse",
                    "--path-format=absolute",
                    "--show-toplevel",
                    "--absolute-git-dir",
                    "--git-common-dir",
                    "--git-path",
                    "index",
                ],
                cwd=root,
            )
        except OSError as exc:
            raise ValueError(f"could not bind checkout Git worktree: {exc}") from exc
        expected = [str(root), str(metadata), str(metadata), str(metadata / "index")]
        if result.returncode != 0 or result.stdout.splitlines() != expected:
            raise ValueError(
                f"checkout Git worktree or metadata is redirected: expected {expected!r}, "
                f"observed {result.stdout.splitlines()!r}; {result.stderr.strip()}"
            )

    def inspect_checkout(self, checkout: Path) -> CheckoutState:
        """Read HEAD plus every provenance-relevant tracked and untracked path.

        Ignored bytecode and extension modules can still be imported from a checkout, so the
        second ``ls-files`` invocation is intentional.  A default ``git status`` would hide
        exactly the files this check exists to catch.
        """

        self.require_checkout_binding(checkout)

        def git(*arguments: str) -> str:
            try:
                result = self.run(["git", *arguments], cwd=checkout)
            except OSError as exc:
                raise ValueError(f"could not inspect source checkout: {exc}") from exc
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise ValueError(f"git {' '.join(arguments)} failed for {checkout}: {detail}")
            return result.stdout

        head = git("rev-parse", "--verify", "HEAD^{commit}").strip()
        modified = tuple(
            sorted(
                filter(
                    None,
                    git(
                        "diff",
                        "--name-only",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--no-renames",
                        "HEAD",
                        "--",
                    ).splitlines(),
                )
            )
        )
        ordinary_untracked = filter(
            None,
            git(
                "ls-files",
                "--others",
                "--exclude-standard",
            ).splitlines(),
        )
        ignored_untracked = filter(
            None,
            git(
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
            ).splitlines(),
        )
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
                missing_tool="swift",
                requires_tool=["swift"],
            )
        built = self.run(["swift", "build", "-c", "release"], cwd=checkout)
        if built.returncode != 0:
            raise ProvisioningError(
                "swift_build_failed", f"swift build failed: {built.stderr.strip()[-600:]}"
            )

    def swift_product_runs(self, checkout: Path, product: str) -> bool:
        """Whether the built executable actually launches.

        The product name is pinned in the manifest rather than written here. It was `fluidaudio`
        in this call, and `Package.swift` declares `.executable(name: "fluidaudiocli")` -- so
        `swift run` answered `no executable product named 'fluidaudio'` on a perfectly good build
        and this check could never return True. A name that lives beside the commit it belongs to
        is reviewable when the commit moves; a literal buried in a runner is not.
        """
        result = self.run(
            ["swift", "run", "-c", "release", product, "--help"], cwd=checkout, timeout=600
        )
        return result.returncode == 0

    def built_product_runs(self, executable: Path) -> bool:
        """Probe an already-built product without requiring its provisioning toolchain."""
        result = self.run([str(executable), "--help"], cwd=executable.parent, timeout=600)
        return result.returncode == 0
