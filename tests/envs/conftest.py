"""Mark tests that exercise real simulation environments as integration tests."""

from pathlib import Path

import pytest


ENV_TEST_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.is_relative_to(ENV_TEST_ROOT):
            item.add_marker(pytest.mark.integration)
