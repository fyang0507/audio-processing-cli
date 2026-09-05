"""Manifest-backed package catalog boundary.

All provisioning workflows resolve the catalog through this module. Tests that need a
different manifest view patch this owner rather than the public package facade.
"""

from ..environments import packages

__all__ = ["packages"]
