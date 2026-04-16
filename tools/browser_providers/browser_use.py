"""Browser Use cloud browser provider.

This provider creates ephemeral browser sessions via browser-use cloud API.
Sessions are automatically cleaned up when Hermes exits to prevent orphaned
sessions from consuming the free tier's 3-session limit.
"""

import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import requests

from tools.browser_providers.base import CloudBrowserProvider
from tools.managed_tool_gateway import resolve_managed_tool_gateway
from tools.tool_backend_helpers import managed_nous_tools_enabled

logger = logging.getLogger(__name__)
_pending_create_keys: Dict[str, str] = {}
_pending_create_keys_lock = threading.Lock()
_active_browser_sessions: Dict[str, str] = {}  # task_id -> session_id
_active_sessions_lock = threading.Lock()

_BASE_URL = "https://api.browser-use.com/api/v3"
_DEFAULT_MANAGED_TIMEOUT_MINUTES = 5
_DEFAULT_MANAGED_PROXY_COUNTRY_CODE = "us"


def _get_or_create_pending_create_key(task_id: str) -> str:
    with _pending_create_keys_lock:
        existing = _pending_create_keys.get(task_id)
        if existing:
            return existing

        created = f"browser-use-session-create:{uuid.uuid4().hex}"
        _pending_create_keys[task_id] = created
        return created


def _clear_pending_create_key(task_id: str) -> None:
    with _pending_create_keys_lock:
        _pending_create_keys.pop(task_id, None)


def _register_session(task_id: str, session_id: str) -> None:
    """Register an active browser session for tracking and cleanup."""
    with _active_sessions_lock:
        _active_browser_sessions[task_id] = session_id
        logger.debug("Registered browser session %s for task %s", session_id, task_id)


def _unregister_session(task_id: str) -> None:
    """Unregister a browser session after successful cleanup."""
    with _active_sessions_lock:
        _active_browser_sessions.pop(task_id, None)


def _get_active_sessions() -> List[str]:
    """Get list of all active browser session IDs."""
    with _active_sessions_lock:
        return list(_active_browser_sessions.values())


def _should_preserve_pending_create_key(response: requests.Response) -> bool:
    if response.status_code >= 500:
        return True

    if response.status_code != 409:
        return False

    try:
        payload = response.json()
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False

    error = payload.get("error")
    if not isinstance(error, dict):
        return False

    message = str(error.get("message") or "").lower()
    return "already in progress" in message


class BrowserUseProvider(CloudBrowserProvider):
    """Browser Use (https://browser-use.com) cloud browser backend."""

    def provider_name(self) -> str:
        return "Browser Use"

    def is_configured(self) -> bool:
        return self._get_config_or_none() is not None

    # ------------------------------------------------------------------
    # Config resolution (direct API key OR managed Nous gateway)
    # ------------------------------------------------------------------

    def _get_config_or_none(self) -> Optional[Dict[str, Any]]:
        api_key = os.environ.get("BROWSER_USE_API_KEY")
        if api_key:
            return {
                "api_key": api_key,
                "base_url": _BASE_URL,
                "managed_mode": False,
            }

        managed = resolve_managed_tool_gateway("browser-use")
        if managed is None:
            return None

        return {
            "api_key": managed.nous_user_token,
            "base_url": managed.gateway_origin.rstrip("/"),
            "managed_mode": True,
        }

    def _get_config(self) -> Dict[str, Any]:
        config = self._get_config_or_none()
        if config is None:
            message = (
                "Browser Use requires a direct BROWSER_USE_API_KEY credential."
            )
            if managed_nous_tools_enabled():
                message = (
                    "Browser Use requires either a direct BROWSER_USE_API_KEY "
                    "credential or a managed Browser Use gateway configuration."
                )
            raise ValueError(message)
        return config

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Browser-Use-API-Key": config["api_key"],
        }
        return headers

    def create_session(self, task_id: str) -> Dict[str, object]:
        config = self._get_config()
        managed_mode = bool(config.get("managed_mode"))

        headers = self._headers(config)
        if managed_mode:
            headers["X-Idempotency-Key"] = _get_or_create_pending_create_key(task_id)

        # Keep gateway-backed sessions short so billing authorization does not
        # default to a long Browser-Use timeout when Hermes only needs a task-
        # scoped ephemeral browser.
        payload = (
            {
                "timeout": _DEFAULT_MANAGED_TIMEOUT_MINUTES,
                "proxyCountryCode": _DEFAULT_MANAGED_PROXY_COUNTRY_CODE,
            }
            if managed_mode
            else {}
        )

        response = requests.post(
            f"{config['base_url']}/browsers",
            headers=headers,
            json=payload,
            timeout=30,
        )

        if not response.ok:
            if managed_mode and not _should_preserve_pending_create_key(response):
                _clear_pending_create_key(task_id)
            raise RuntimeError(
                f"Failed to create Browser Use session: "
                f"{response.status_code} {response.text}"
            )

        session_data = response.json()
        if managed_mode:
            _clear_pending_create_key(task_id)
        session_name = f"hermes_{task_id}_{uuid.uuid4().hex[:8]}"
        external_call_id = response.headers.get("x-external-call-id") if managed_mode else None

        logger.info("Created Browser Use session %s", session_name)

        cdp_url = session_data.get("cdpUrl") or session_data.get("connectUrl") or ""
        
        # Register the session for tracking and cleanup
        session_id = session_data["id"]
        _register_session(task_id, session_id)

        return {
            "session_name": session_name,
            "bb_session_id": session_id,
            "cdp_url": cdp_url,
            "features": {"browser_use": True},
            "external_call_id": external_call_id,
        }

    def close_session(self, session_id: str, task_id: Optional[str] = None) -> bool:
        """Close a browser session and destroy the sandbox entirely.

        Args:
            session_id: The browser session ID to close
            task_id: Optional task ID for cleanup tracking
        """
        try:
            config = self._get_config()
        except ValueError:
            logger.warning("Cannot close Browser Use session %s — missing credentials", session_id)
            return False

        try:
            # Use the stop endpoint with strategy=session to destroy the sandbox
            # This is different from just stopping the task (which keeps the session alive)
            response = requests.post(
                f"{config['base_url']}/sessions/{session_id}/stop",
                headers=self._headers(config),
                json={"strategy": "session"},  # Destroy sandbox entirely
                timeout=15,
            )

            if response.status_code in (200, 201, 204):
                logger.debug("Successfully closed Browser Use session %s", session_id)
                # Unregister the session if we have the task_id
                if task_id:
                    _unregister_session(task_id)
                return True
            else:
                logger.warning(
                    "Failed to close Browser Use session %s: HTTP %s - %s",
                    session_id,
                    response.status_code,
                    response.text[:200],
                )
                return False
        except Exception as e:
            logger.error("Exception closing Browser Use session %s: %s", session_id, e)
            return False

    def emergency_cleanup(self, session_id: str) -> None:
        """Emergency cleanup for a specific session.
        
        Tries to stop the session with minimal error handling for use in
        atexit handlers and exception contexts.
        """
        config = self._get_config_or_none()
        if config is None:
            logger.warning("Cannot emergency-cleanup Browser Use session %s — missing credentials", session_id)
            return
        try:
            requests.post(
                f"{config['base_url']}/sessions/{session_id}/stop",
                headers=self._headers(config),
                json={"strategy": "session"},
                timeout=5,
            )
        except Exception as e:
            logger.debug("Emergency cleanup failed for Browser Use session %s: %s", session_id, e)

    def cleanup_all_sessions(self) -> int:
        """Clean up all tracked browser sessions.
        
        This method should be called when Hermes shuts down to prevent
        orphaned sessions from consuming the free tier's 3-session limit.
        
        Returns:
            Number of sessions successfully cleaned up
        """
        sessions = _get_active_sessions()
        if not sessions:
            logger.debug("No active browser sessions to clean up")
            return 0

        logger.info("Cleaning up %d active browser session(s)", len(sessions))
        
        cleaned = 0
        for session_id in sessions:
            try:
                config = self._get_config()
                response = requests.post(
                    f"{config['base_url']}/sessions/{session_id}/stop",
                    headers=self._headers(config),
                    json={"strategy": "session"},
                    timeout=5,
                )
                if response.status_code in (200, 201, 204):
                    logger.debug("Cleaned up session %s", session_id)
                    cleaned += 1
                else:
                    logger.warning(
                        "Failed to clean up session %s: HTTP %s",
                        session_id, response.status_code
                    )
            except Exception as e:
                logger.warning("Exception cleaning up session %s: %s", session_id, e)

        # Clear the tracking dict after cleanup attempt
        with _active_sessions_lock:
            _active_browser_sessions.clear()

        logger.info("Cleaned up %d/%d browser session(s)", cleaned, len(sessions))
        return cleaned
