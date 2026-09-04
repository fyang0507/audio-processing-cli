"""Environment creation and package materialization state machine."""

from __future__ import annotations

from . import paths
from ._package_compat import facade_dependency
from ._package_core import ProvisioningError, _now, _tree_bytes, sha256_file
from ._package_paths import (
    built_product_candidates,
    checkout_patch_expectation,
    managed_environment_creation_target_issue,
    managed_provisioning_root_issue,
    managed_url_artifact_path,
    materialize_checkout_patch,
)
from ._package_registry import load_registry, save_registry
from ._package_reports import is_ready
from ._package_teardown import (
    _delete_managed,
    _selection_bytes,
    _source_revisions,
    _toolchain_missing,
)
from .environments import ManifestError, Package, environments


def _load_registry() -> dict:
    return facade_dependency("load_registry", load_registry)()


def _save_registry(document: dict) -> None:
    facade_dependency("save_registry", save_registry)(document)


def _measure_tree(path):
    return facade_dependency("_tree_bytes", _tree_bytes)(path)


class PullMixin:
    def ensure_environment(self, name: str, document: dict) -> bool:
        """Create an environment if it is not already `ready`. Returns whether it was created."""
        environment = environments()[name]
        if not environment.provisioned:
            return False
        target = paths.env_dir(name)
        target_issue = managed_environment_creation_target_issue(name)
        if target_issue is not None:
            raise ProvisioningError(
                "environment_drifted",
                target_issue,
                environment=name,
                fix=(
                    f"Replace the redirected environment path {target} and run "
                    "audio packages verify --repair"
                ),
            )

        if not environment.has_interpreter:
            # `swift` has no interpreter and so nothing to sync, but it still holds a build
            # product and a Core ML package. It gets a registry entry anyway: reference
            # counting and purge treat every provisioned environment the same way, and an
            # untracked directory is one nothing can reclaim.
            if document["environments"].get(name, {}).get("state") == "ready":
                return False
            target.mkdir(parents=True, exist_ok=True)
            document["environments"][name] = {
                "state": "ready", "path": str(target), "python": None, "lock_sha256": None,
                "created_utc": _now(),
            }
            _save_registry(document)
            return True

        entry = document["environments"].get(name, {})
        lock_digest = sha256_file(environment.lock)
        if entry.get("state") == "ready" and entry.get("lock_sha256") == lock_digest:
            return False

        # Intent first: a crash between here and the flip leaves a `creating` entry, which
        # reads as absent and as reclaimable rather than as a working environment.
        document["environments"][name] = {
            "state": "creating", "path": str(target), "python": environment.python,
            "lock_sha256": lock_digest, "created_utc": _now(),
        }
        _save_registry(document)

        self.toolchain.create_environment(environment, target)
        document["environments"][name]["state"] = "ready"
        _save_registry(document)
        return True

    # -- packages ----------------------------------------------------------------------

    def pull(self, selection: list[Package], *, repair: bool = False,
             stack: str | None = None) -> dict:
        """Materialize what is not already provisioned. Two asymmetries, both deliberate.

        **A `ready` package is skipped, not re-materialized.** Re-hashing a multi-gigabyte
        artifact, re-cloning and re-installing a checkout, or rebuilding the Swift product costs
        minutes and produces what is already there. Worse, the `pulling` entry that has to be
        written first would leave a working install downgraded if the pointless re-pull were
        interrupted — the crash-safety rule turned against a package nothing was wrong with.
        `--repair` is how a caller asks for the work anyway, and it *forces* re-materialization
        rather than trusting what is on disk, because a corrupt-but-present artifact is exactly
        the case it exists for.

        **A stack tolerates a toolchain-blocked package; a named one does not.** `--stack` is a
        superset guess, so an absent `swift` blocks `fluidaudio` and the rest of the stack still
        provisions, with the blocked package reported in `warnings` — which is what
        TRANSCRIBE_HAPPY_PATH.md §0 promises and what raising on the first blocked package broke,
        since `fluidaudio` sorts first and took the whole stack down with it. Naming a package is
        an instruction, so there the missing tool is still exit 3: silently skipping what a caller
        asked for by name is worse than refusing.
        """
        root_issue = managed_provisioning_root_issue()
        if root_issue is not None:
            raise ProvisioningError(
                "environment_drifted",
                root_issue,
                fix=(
                    f"Replace the redirected provisioning root {paths.root()} and run "
                    "audio packages verify --repair"
                ),
            )
        # Refuse a redirected root before even reading its registry.  Reading first
        # crosses the same ownership boundary as writing and can also surface attacker-
        # controlled JSON as a misleading registry error instead of the root failure.
        document = _load_registry()
        # Read the cache once, before anything downloads. Everything after this point works
        # from that snapshot, so a revision fetched by this pull is never mistaken for one that
        # was already there.
        cached = self.fetcher.cached_revisions()
        pulled: list[dict] = []
        skipped: list[str] = []
        created: list[str] = []
        warnings: list[dict] = []
        blocked: list[tuple[Package, str]] = []

        for package in selection:
            if not repair and is_ready(document, package.id):
                # Nothing needs doing, so nothing is touched — in particular the entry is not
                # transitioned to `pulling`, which an interrupted no-op would leave behind.
                skipped.append(package.id)
                continue

            missing_tool = next((tool for tool in package.requires_tool
                                 if self.toolchain.which(tool) is None), None)
            if missing_tool is not None:
                if stack is None:
                    raise _toolchain_missing(package, missing_tool)
                blocked.append((package, missing_tool))
                continue

            if self.ensure_environment(package.environment, document):
                created.append(package.environment)

            previous = document["packages"].get(package.id, {})
            pre_existing = self._pre_existing_revisions(package, previous, cached)
            entry = {
                "state": "pulling", "environment": package.environment, "kind": package.kind,
                "source": package.source, "license_declared": package.license_declared,
                "license_reviewed": package.license_reviewed, "pulled_utc": _now(),
                # Decided before any bytes move, and carried across a retry: see
                # _pre_existing_revisions for why re-deciding would be wrong.
                "hub_revisions_pre_existing": sorted(pre_existing),
            }
            document["packages"][package.id] = entry
            _save_registry(document)

            materialized = self._materialize(package, document, pre_existing, repair=repair)
            entry["materialized"] = materialized
            entry["state"] = "ready"
            document["packages"][package.id] = entry
            _save_registry(document)

            receipt = {"package": package.id, "environment": package.environment,
                       "bytes": materialized.get("bytes")}
            for key in ("revision", "revisions", "digest_verified", "built", "product_runs",
                        "patches_applied"):
                if key in materialized:
                    receipt[key] = materialized[key]
            if materialized.get("hub_revisions_pre_existing"):
                receipt["hub_revisions_pre_existing"] = \
                    materialized["hub_revisions_pre_existing"]
                receipt["pre_existing_note"] = (
                    "already in the Hugging Face cache; not downloaded, and teardown here will "
                    "not delete it"
                )
            pulled.append(receipt)

        if blocked and not pulled and not skipped:
            # Nothing in the stack was provisionable, so there is no partial success to report
            # and the caller needs the exit code rather than a receipt of an empty pull.
            raise _toolchain_missing(*blocked[0])

        if blocked:
            tools = sorted({tool for _, tool in blocked})
            names = [package.id for package, _ in blocked]
            warnings.append({
                "code": "toolchain_missing", "blocking": True,
                "packages": names, "requires_tool": tools,
                "detail": (
                    f"{', '.join(names)} {'needs' if len(names) == 1 else 'need'} "
                    f"{', '.join(tools)}, which {'is' if len(tools) == 1 else 'are'} not on "
                    f"PATH, so {'it' if len(names) == 1 else 'they'} "
                    f"{'was' if len(names) == 1 else 'were'} not provisioned; the rest of "
                    f"{f'stack {stack}' if stack else 'the selection'} was. Install the "
                    f"toolchain and pull {'it' if len(names) == 1 else 'them'} by name."
                ),
            })

        # A blocked package provisioned nothing, so it carries no license claim and no bytes.
        # `pulled_known_bytes` says what this pull added, and a skipped package added none of it.
        blocked_ids = {package.id for package, _ in blocked}
        provisioned = [p for p in selection if p.id not in blocked_ids]
        unreviewed = sorted(p.id for p in provisioned if not p.license_reviewed)
        if unreviewed:
            warnings.append({
                "code": "license_unreviewed", "blocking": False,
                "packages": unreviewed,
                "detail": (
                    f"{', '.join(unreviewed)} "
                    f"{'reports' if len(unreviewed) == 1 else 'report'} a license their model "
                    "card declares but nobody has reviewed. A declared license is evidence "
                    "that one exists, not a redistribution clearance."
                ),
            })

        known, unsized = _selection_bytes(
            [p for p in provisioned if p.id not in set(skipped)], document)
        report = {
            "pulled": pulled,
            "skipped": skipped,
            "environments_created": sorted(set(created)),
            "root": str(paths.root()),
            "registry": str(paths.registry_path()),
            # This pull's packages only. `audio packages list` reports the cumulative total;
            # naming both `reclaimable` invited reading one as the other, and this figure
            # legitimately goes down between pulls.
            "pulled_known_bytes": known,
            "unsized_packages": unsized,
            "warnings": warnings,
        }
        if skipped:
            report["skipped_reason"] = (
                "already ready in the registry, so nothing was re-materialized and nothing "
                "counts toward pulled_known_bytes; audio packages pull --repair <package> "
                "re-materializes anyway, and audio packages verify is what checks them"
            )
        return report

    @staticmethod
    def _pre_existing_revisions(package: Package, previous: dict,
                                cached: set[str]) -> set[str]:
        """Which of this package's revisions the Hub cache held before this root wanted them.

        Two rules, and the second is the subtle one:

        - A revision already in the shared cache is not ours to delete later, whoever put it
          there — another tool, another provisioning root, or an earlier experiment.
        - **A retry must not re-decide.** `snapshot_download` publishes a snapshot directory as
          files land, so an interrupted 16 GiB pull leaves a revision that a later cache scan
          reports as present. Asking again would classify this root's own partly-finished
          download as somebody else's, and teardown would then refuse to reclaim 16 GiB it did
          in fact fetch. The first attempt's answer is the truthful one, so it is recorded in
          the `pulling` entry and reused.
        """
        if "hub_revisions_pre_existing" in previous:
            return set(previous["hub_revisions_pre_existing"])
        return {revision for revision in _source_revisions(package) if revision in cached}

    def _materialize(self, package: Package, document: dict, pre_existing: set[str], *,
                     repair: bool = False) -> dict:
        """Put the package on disk. `digest_verified` appears only where a digest was taken.

        One source kind pins a content hash — `url` — and it is the only one whose materialization
        can claim to have been verified against the manifest. The Hub kinds pin a *revision*; no
        `sha256` exists in the manifest to hash a snapshot against, so they record the revision
        and nothing more. They used to record `digest_verified: True` regardless, which made
        `verify` print `digest: "ok"` for a check no code performs.
        """
        kind = package.source["type"]
        if kind == "url":
            # The filename is manifest data, not derived: the shipped Silero backend resolves
            # this exact name, and a pull that invented one would leave two copies on disk and
            # re-download on first use. tests/test_environments.py ties the two together.
            target = paths.models_dir() / package.source["filename"]
            # `repair` needs no force here, and the digest is the reason: url_file re-hashes what
            # is on disk against the manifest pin and downloads again unless it matches, so a
            # match already is the strongest re-materialization available. Forcing the transfer
            # would spend the bytes to arrive at the same file.
            resolved = self.fetcher.url_file(package.source["url"], package.source["sha256"],
                                             target)
            _managed, location_issue = managed_url_artifact_path(package, resolved)
            if location_issue is not None:
                raise ProvisioningError(
                    "package_integrity_failed",
                    location_issue,
                    fix=f"audio packages pull --repair {package.id}",
                )
            return {"path": str(resolved), "bytes": _measure_tree(resolved),
                    "digest_verified": True}

        if kind == "huggingface":
            revision = package.source["revision"]
            patterns = package.source.get("allow_patterns")
            if patterns is None:
                snapshot = self.fetcher.hf_snapshot(
                    package.source["repo"], revision, force=repair
                )
            else:
                snapshot = self.fetcher.hf_snapshot(
                    package.source["repo"],
                    revision,
                    force=repair,
                    allow_patterns=tuple(patterns),
                )
            snapshot_bytes = self._require_hub_snapshot(
                package,
                package.source["repo"],
                revision,
                snapshot,
                tuple(patterns or ()),
            )
            result = {
                "path": str(snapshot), "bytes": snapshot_bytes,
                "revision": revision,
                # Only what this pull fetched is ours to delete later, decided before the
                # download rather than after it — see _pre_existing_revisions.
                "hub_revisions": [] if revision in pre_existing else [revision],
                "hub_revisions_pre_existing": sorted(pre_existing),
            }
            result.update(self._checkout_and_install(package, repair=repair))
            return result

        if kind == "huggingface_multi":
            snapshots = {}
            total = 0
            ours: list[str] = []
            for repo in package.source["repos"]:
                patterns = repo.get("allow_patterns")
                snapshot = self.fetcher.hf_snapshot(
                    repo["repo"],
                    repo["revision"],
                    force=repair,
                    allow_patterns=tuple(patterns) if patterns is not None else None,
                )
                snapshot_bytes = self._require_hub_snapshot(
                    package,
                    repo["repo"],
                    repo["revision"],
                    snapshot,
                    tuple(patterns or ()),
                )
                snapshots[repo["repo"]] = str(snapshot)
                total += snapshot_bytes
                if repo["revision"] not in pre_existing:
                    ours.append(repo["revision"])
            result = {
                "paths": snapshots, "bytes": total,
                # Plural, because this package spans four repositories. A single `revision`
                # key would have to pick one of them, and the receipt promises the revisions
                # a pull materialized.
                "revisions": [repo["revision"] for repo in package.source["repos"]],
                "hub_revisions": ours, "hub_revisions_pre_existing": sorted(pre_existing),
            }
            result.update(self._checkout_and_install(package, repair=repair))
            return result

        if kind == "git+build":
            checkout = paths.checkout_dir(package.environment, package.id)
            # This method only runs for a non-ready package or an explicit repair. Reusing a
            # prior tree would let an interrupted pull's untracked build hook run before verify.
            _delete_managed(checkout)
            checkout.parent.mkdir(parents=True, exist_ok=True)
            self.toolchain.clone(package.source["repo"], package.source["commit"], checkout)
            applied, digests = materialize_checkout_patch(
                package, checkout, self.toolchain
            )
            try:
                state = self.toolchain.inspect_checkout(checkout)
            except ValueError as exc:
                raise ProvisioningError(
                    "checkout_integrity_failed",
                    f"could not inspect fresh checkout: {exc}",
                    package=package.id,
                    fix=f"audio packages pull --repair {package.id}",
                ) from exc
            _expected_patches, expected_names, _expected_digests = (
                checkout_patch_expectation(package)
            )
            if (
                state.head != package.source["commit"]
                or set(state.modified) != set(expected_names)
                or state.untracked
            ):
                raise ProvisioningError(
                    "checkout_integrity_failed",
                    f"{package.id} checkout was not exact before build",
                    package=package.id,
                    expected={
                        "head": package.source["commit"],
                        "modified": sorted(expected_names),
                        "untracked": [],
                    },
                    actual={
                        "head": state.head,
                        "modified": list(state.modified),
                        "untracked": list(state.untracked),
                    },
                    fix=f"audio packages pull --repair {package.id}",
                )
            self.toolchain.swift_build(checkout)
            product = package.source["product"]
            runs = self.toolchain.swift_product_runs(checkout, product)
            if not runs:
                # A build that produces an executable nobody can launch is not a provisioned
                # package, and this used to be recorded and then ignored: `pull` returned exit 0
                # with an empty `warnings`, `verify` reported `failed: []`, and the environment
                # read `ok`, while the one thing the package exists to do was impossible. The
                # entry stays `pulling`, so `list` and `run` both report it absent.
                raise ProvisioningError(
                    "package_build_unusable",
                    f"{package.id} built, but its product {product!r} does not run, so nothing "
                    f"in the {package.environment} environment can use it",
                    package=package.id, product=product, built=True,
                    fix=f"audio packages pull --repair {package.id}",
                )
            candidates = built_product_candidates(checkout, product)
            if len(candidates) != 1:
                raise ProvisioningError(
                    "package_build_unusable",
                    f"{package.id} built, but expected one contained executable product "
                    f"{product!r} and found {len(candidates)}",
                    package=package.id, product=product, built=True,
                    fix=f"audio packages pull --repair {package.id}",
                )
            executable = candidates[0]
            product_path = executable.relative_to(checkout.resolve(strict=True)).as_posix()
            return {"path": str(checkout), "bytes": _measure_tree(checkout / ".build"),
                    "revision": package.source["commit"], "built": True,
                    "product_runs": runs, "product_path": product_path,
                    "product_sha256": sha256_file(executable),
                    "patches_applied": applied,
                    "patched_file_digests": digests}

        raise ManifestError(f"{package.id}: unsupported source type {kind!r}")
