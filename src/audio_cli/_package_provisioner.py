"""Composed package provisioner with injectable external boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

from ._package_checkout import CheckoutMixin
from ._package_lifecycle import LifecycleMixin
from ._package_pull import PullMixin
from ._package_runtime import Fetcher, Toolchain
from ._package_verify import VerifyMixin


@dataclass
class Provisioner(PullMixin, CheckoutMixin, VerifyMixin, LifecycleMixin):
    toolchain: Toolchain = field(default_factory=Toolchain)
    fetcher: Fetcher = field(default_factory=Fetcher)
