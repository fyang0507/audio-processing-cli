"""Package removal and purge operations."""

from __future__ import annotations

from .. import paths
from ..environments import environments
from . import catalog, registry
from .fetcher import Fetcher
from .models import HUB_CACHE_NOTE, UNTOUCHED, ProvisioningError
from .teardown import (
    _delete_managed,
    _managed_package_locations,
    _selection_bytes,
    _teardown_revisions,
    _users_by_environment,
)


def remove(fetcher: Fetcher, package_ids: list[str]) -> dict:
    """Remove named packages. Every name is resolved before anything is deleted.

    The ordering is the point. Deleting inside the same loop that raised on an unknown name,
    with one `save_registry` after it, meant `remove vibevoice-asr-7b firered-asr2` discarded
    17 GiB and then rolled the registry *back* — leaving an entry that still read `ready` for
    a package whose bytes were gone. Nothing downstream notices, because `missing_packages`
    keys on `state`, so the caller finds out at model load. That is the mirror image of what
    the `pulling` state exists to prevent: a pull that dies is honest about being incomplete,
    and a teardown that died was not. A caller naming several packages already assumes
    all-or-nothing, so validate the whole list first.

    Two consequences, both intended. A repeated name removes once and is reported once. And
    each entry is dropped and saved as soon as its own files are gone rather than in one write
    at the end, so no later failure in the teardown can restore a claim to bytes that no
    longer exist.
    """
    document = registry.load_registry()
    targets: list[str] = []
    for identifier in package_ids:
        if identifier not in document["packages"]:
            raise ProvisioningError(
                "package_not_provisioned",
                f"{identifier} is not in the registry",
                exit_code=2,
                package=identifier,
                fix="audio packages list",
            )
        if identifier not in targets:
            targets.append(identifier)

    removed: list[str] = []
    hub_revisions: list[str] = []
    retained: list[str] = []
    local_freed = 0
    package_catalog = catalog.packages()

    for identifier in targets:
        materialized = document["packages"][identifier].get("materialized", {})
        materialized = materialized if isinstance(materialized, dict) else {}
        owned, kept_revisions = _teardown_revisions(package_catalog.get(identifier), materialized)
        hub_revisions.extend(owned)
        retained.extend(kept_revisions)
        for location in _managed_package_locations(package_catalog.get(identifier)):
            local_freed += _delete_managed(location)
        document["packages"].pop(identifier)
        removed.append(identifier)
        registry.save_registry(document)

    # After the entries are gone, so a cache that fails here costs reclaimable space in a
    # shared cache rather than leaving a package whose local bytes are already deleted
    # reading as ready. There is no ordering that keeps that entry honest.
    deleted, hub_freed = fetcher.delete_hub_revisions(hub_revisions)
    kept, dropped, environment_freed = _collect_environments(document)
    local_freed += environment_freed
    registry.save_registry(document)
    report = {
        "removed": removed,
        "environments_removed": dropped,
        "environments_kept": kept,
        "hub_revisions_deleted": deleted,
        "hub_revisions_not_found": sorted(set(hub_revisions) - set(deleted)),
        "hub_revisions_retained": sorted(set(retained)),
        "hub_cache_note": HUB_CACHE_NOTE,
        "reclaimed_bytes": hub_freed + local_freed,
    }
    if retained:
        report["hub_revisions_retained_reason"] = (
            "not owned by this root under the current package manifest, so they are not "
            "this root's to delete"
        )
    if dropped:
        report["environments_removed_reason"] = (
            f"no other provisioned package targets {', '.join(dropped)}"
        )
    if kept:
        report["environments_kept_reason"] = "; ".join(
            f"{', '.join(sorted(users))} still {'needs' if len(users) == 1 else 'need'} {name}"
            for name, users in sorted(_users_by_environment(document).items())
            if name in kept
        )
    return report


def _collect_environments(document: dict) -> tuple[list[str], list[str], int]:
    """Reference counting, derived from the package table each time it is asked."""
    users = _users_by_environment(document)
    kept, dropped, freed = [], [], 0
    known = environments()
    for name in sorted(document["environments"]):
        if users.get(name):
            kept.append(name)
            continue
        if name in known:
            freed += _delete_managed(paths.env_dir(name))
        document["environments"].pop(name)
        dropped.append(name)
    return kept, dropped, freed


def purge(fetcher: Fetcher, *, dry_run: bool) -> dict:
    document = registry.load_registry()
    package_catalog = catalog.packages()
    package_ids = sorted(document["packages"])
    environment_names = sorted(document["environments"])
    known, unsized = _selection_bytes(
        [package_catalog[i] for i in package_ids if i in package_catalog], document
    )

    if dry_run:
        deletable, keeping = [], []
        for identifier in package_ids:
            materialized = document["packages"][identifier].get("materialized", {})
            materialized = materialized if isinstance(materialized, dict) else {}
            owned, retained_revisions = _teardown_revisions(
                package_catalog.get(identifier), materialized
            )
            deletable.extend(owned)
            keeping.extend(retained_revisions)
        return {
            "would_remove": {
                "packages": package_ids,
                "environments": environment_names,
                "root": str(paths.root()),
                "hub_revisions": sorted(set(deletable)),
            },
            "would_keep": {"hub_revisions": sorted(set(keeping))},
            "hub_cache_note": HUB_CACHE_NOTE,
            "reclaimable_known_bytes": known,
            "reclaimable_note": (
                "projected from manifest-bounded registry receipts and package sizes; "
                "it counts deletion-eligible Hub weights and recorded local package "
                "artifacts, but excludes retained revisions and environment bytes"
            ),
            "unsized_packages": unsized,
            "untouched": UNTOUCHED,
        }

    hub_revisions: list[str] = []
    retained: list[str] = []
    local_freed = 0
    # `purge` cannot take a name that is not in the registry — it reads the list *from* the
    # registry — so it never had `remove`'s validation defect. It shared the narrower half:
    # every local file was deleted and the registry was cleared in one write afterwards, so
    # anything that raised in between left every package reading as `ready` with nothing
    # behind it. Same rule as `remove`, then: an entry goes as soon as its own bytes do.
    for identifier in package_ids:
        materialized = document["packages"][identifier].get("materialized", {})
        materialized = materialized if isinstance(materialized, dict) else {}
        owned, kept_revisions = _teardown_revisions(package_catalog.get(identifier), materialized)
        hub_revisions.extend(owned)
        retained.extend(kept_revisions)
        for location in _managed_package_locations(package_catalog.get(identifier)):
            local_freed += _delete_managed(location)
        document["packages"].pop(identifier)
        registry.save_registry(document)
    known_environments = environments()
    for name in environment_names:
        if name in known_environments:
            local_freed += _delete_managed(paths.env_dir(name))
        document["environments"].pop(name, None)
        registry.save_registry(document)
    deleted, hub_freed = fetcher.delete_hub_revisions(hub_revisions)
    return {
        "removed": {"packages": package_ids, "environments": environment_names},
        "hub_revisions_deleted": deleted,
        "hub_revisions_not_found": sorted(set(hub_revisions) - set(deleted)),
        "hub_revisions_retained": sorted(set(retained)),
        "hub_cache_note": HUB_CACHE_NOTE,
        "reclaimed_bytes": hub_freed + local_freed,
        "unsized_packages": unsized,
        "untouched": UNTOUCHED,
    }
