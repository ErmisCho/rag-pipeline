"""Pytest configuration and fixtures."""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "-F",
        "--fast",
        action="store_true",
        default=False,
        help="Run fast tests only (skip integration tests requiring infrastructure)",
    )


def pytest_configure(config):
    """Load .env before tests run and register markers."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests that require live infrastructure",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if --fast flag is used."""
    if config.getoption("--fast"):
        skip_integration = pytest.mark.skip(
            reason="skipped with -f (fast mode)")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
