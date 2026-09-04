"""Compatibility facade for ``audio packages`` provisioning and lifecycle APIs.

Implementation lives in task-shaped private modules so callers can continue importing the
established ``audio_cli.packages`` surface without depending on its internal decomposition.
"""

from __future__ import annotations

# Keep the historically module-visible dependencies available. Several downstream test suites
# patch these shared module objects to exercise descriptor and publication race boundaries.
import fnmatch  # noqa: F401
import hashlib  # noqa: F401
import json  # noqa: F401
import os  # noqa: F401
import platform  # noqa: F401
import re  # noqa: F401
import shutil  # noqa: F401
import stat  # noqa: F401
import subprocess  # noqa: F401
import sys  # noqa: F401
import urllib.error  # noqa: F401
import urllib.parse  # noqa: F401
import urllib.request  # noqa: F401
import uuid  # noqa: F401
from collections.abc import Mapping  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
from datetime import datetime, timezone  # noqa: F401
from pathlib import Path  # noqa: F401
from typing import Any  # noqa: F401

from . import paths  # noqa: F401
from ._package_core import (
    HUB_CACHE_NOTE,
    REGISTRY_SCHEMA_VERSION,
    UNTOUCHED,
    CheckoutState,
    ProvisioningError,
    _hub_snapshot_index,
    _inspect_hub_snapshot,
    _now,
    _tree_bytes,
    hub_materialization_issues,
    sha256_file,
)
from ._package_doctor import _memory, doctor
from ._package_paths import (
    _checkout_integrity_issues,
    _environment_built_runtime_runs,
    _environment_has_built_runtime,
    _patch_touched_names,
    _patched_files,
    built_product_candidates,
    checkout_file_matches,
    checkout_patch_expectation,
    managed_checkout_path,
    managed_environment_creation_target_issue,
    managed_environment_path,
    managed_provisioning_root_issue,
    managed_url_artifact_path,
    materialize_checkout_patch,
    validated_built_product,
)
from ._package_provisioner import Provisioner
from ._package_registry import (
    _reject_duplicate_registry_keys,
    blank_registry,
    load_registry,
    save_registry,
)
from ._package_reports import (
    _path_location_fields,
    is_ready,
    list_report,
    missing_packages,
    path_report,
    select,
)
from ._package_requirements import (
    _checkout_install_drift,
    _class_of,
    _direct_file_install_path,
    _distribution_name,
    _environment_drift,
    _locked_versions,
    _managed_checkout_requirements,
    _method_of,
    _module_available,
    _module_of,
)
from ._package_runtime import Fetcher, Toolchain
from ._package_teardown import (
    _delete_at,
    _delete_managed,
    _managed_package_locations,
    _owned_tree_bytes,
    _owned_tree_bytes_at,
    _selection_bytes,
    _source_revision_report,
    _source_revisions,
    _teardown_revisions,
    _toolchain_missing,
    _users_by_environment,
)
from .environments import (  # noqa: F401
    HERE as ENVIRONMENTS_DIR,
    Environment,
    ManifestError,
    Package,
    environments,
    packages,
)
from .media import (  # noqa: F401
    assert_directory_binding,
    bound_directory,
    cleanup_temporary_file,
    file_identity_from_descriptor,
    publish_temporary_file,
    sha256_regular_file_at,
)
