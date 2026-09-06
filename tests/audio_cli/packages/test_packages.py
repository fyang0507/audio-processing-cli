"""Compatibility exports for package fakes used by cross-module specification tests."""

from tests.audio_cli.packages.package_test_support import FakeFetcher, FakeToolchain

__all__ = ["FakeFetcher", "FakeToolchain"]
