"""Unit tests for registry rollback logic."""

import pytest


def determine_rollback_target(
    aliases: dict[str, str], to_version: str | None = None
) -> str:
    """Helper verifying rollback version determination."""
    target = to_version or aliases.get("previous_champion")
    if not target:
        raise ValueError("No target version available for rollback.")
    return target


def test_rollback_to_previous_champion():
    aliases = {"champion": "2", "previous_champion": "1"}
    target = determine_rollback_target(aliases)
    assert target == "1"


def test_rollback_to_explicit_version():
    aliases = {"champion": "3", "previous_champion": "2"}
    target = determine_rollback_target(aliases, to_version="1")
    assert target == "1"


def test_rollback_no_previous_raises():
    aliases = {"champion": "1"}
    with pytest.raises(ValueError, match="No target version available"):
        determine_rollback_target(aliases)
