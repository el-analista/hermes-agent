"""Tests for Hermes-managed Camofox state helpers."""

from unittest.mock import patch

import pytest


def _load_module():
    from tools import browser_camofox_state as state
    return state


class TestCamofoxStatePaths:
    def test_paths_are_profile_scoped(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            assert state.get_camofox_state_dir() == tmp_path / "browser_auth" / "camofox"


class TestCamofoxIdentity:
    def test_identity_is_deterministic(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            first = state.get_camofox_identity("task-1")
            second = state.get_camofox_identity("task-1")
            assert first == second

    def test_identity_differs_by_task(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            a = state.get_camofox_identity("task-a")
            b = state.get_camofox_identity("task-b")
            # Same user (same profile), different session keys
            assert a["user_id"] == b["user_id"]
            assert a["session_key"] != b["session_key"]

    def test_identity_differs_by_profile(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path / "profile-a"):
            a = state.get_camofox_identity("task-1")
        with patch.object(state, "get_hermes_home", return_value=tmp_path / "profile-b"):
            b = state.get_camofox_identity("task-1")
        assert a["user_id"] != b["user_id"]

    def test_default_task_id(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            identity = state.get_camofox_identity()
            assert "user_id" in identity
            assert "session_key" in identity
            assert identity["user_id"].startswith("hermes_")
            assert identity["session_key"].startswith("task_")


class TestCamofoxConfigDefaults:
    def test_default_config_includes_managed_persistence_toggle(self):
        from hermes_cli.config import DEFAULT_CONFIG

        browser_cfg = DEFAULT_CONFIG["browser"]
        assert browser_cfg["camofox"]["managed_persistence"] is False

    def test_optional_env_vars_include_managed_persistence(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "CAMOFOX_MANAGED_PERSISTENCE" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["CAMOFOX_MANAGED_PERSISTENCE"]["category"] == "setting"
        assert OPTIONAL_ENV_VARS["CAMOFOX_MANAGED_PERSISTENCE"]["password"] is False

    def test_config_version_tracks_camofox_settings(self):
        from hermes_cli.config import DEFAULT_CONFIG, ENV_VARS_BY_VERSION

        current_version = DEFAULT_CONFIG["_config_version"]
        assert current_version == 11
        assert "CAMOFOX_MANAGED_PERSISTENCE" in ENV_VARS_BY_VERSION[11]
