"""Share Desktop login + active match auth with ffkillblock (no local_config token)."""

from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple

KEYRING_SERVICE = "16ScoreAI"
SESSION_MAX_AGE_S = 86400  # 24h


def session_file_path(base_dir: Optional[str] = None) -> str:
    root = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, ".desktop_session.json")


def save_desktop_user(user_email: str, base_dir: Optional[str] = None) -> None:
    """Called on Desktop login — stores email for keyring lookup."""
    email = (user_email or "").strip()
    if not email:
        return
    path = session_file_path(base_dir)
    data = _read_raw(path)
    data["user_email"] = email
    data.setdefault("updated_at", time.time())
    _write_raw(path, data)


def save_desktop_token(
    access_token: str,
    user_email: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> None:
    """Store login token after Desktop sign-in (match_id added on Start Match)."""
    tok = (access_token or "").strip()
    if not tok:
        return
    path = session_file_path(base_dir)
    data = _read_raw(path)
    data["access_token"] = tok
    email = (user_email or "").strip()
    if email:
        data["user_email"] = email
    data["updated_at"] = time.time()
    _write_raw(path, data)


def save_desktop_match(
    match_id: str,
    access_token: Optional[str],
    user_email: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> None:
    """Called on Desktop Start Match — match_id + login token for TMS."""
    mid = str(match_id or "").strip()
    if not mid:
        return
    path = session_file_path(base_dir)
    data = _read_raw(path)
    data.update(
        {
            "match_id": mid,
            "access_token": (access_token or "").strip() or data.get("access_token", ""),
            "user_email": (user_email or "").strip() or data.get("user_email", ""),
            "started_at": time.time(),
            "updated_at": time.time(),
        }
    )
    _write_raw(path, data)


def load_desktop_auth(
    match_id: Optional[str] = None,
    access_token: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Fill missing match_id / access_token from Desktop session or system keyring.

    Priority: explicit args → session file → keyring (using stored user_email).
    """
    mid = str(match_id or "").strip() or "1"
    token = (access_token or "").strip() or None

    path = session_file_path(base_dir)
    data = _read_raw(path)
    age = time.time() - float(data.get("started_at") or data.get("updated_at") or 0)
    session_fresh = age <= SESSION_MAX_AGE_S

    if session_fresh:
        if not token:
            tok = (data.get("access_token") or "").strip()
            if tok:
                token = tok
        file_mid = (data.get("match_id") or "").strip()
        if file_mid and (not mid or mid == "1"):
            mid = file_mid

    if not token:
        email = (data.get("user_email") or "").strip()
        if email:
            token = _keyring_token(email)

    return mid, token


def _keyring_token(user_email: str) -> Optional[str]:
    try:
        import keyring  # type: ignore

        return keyring.get_password(KEYRING_SERVICE, user_email)
    except Exception:
        return None


def _read_raw(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_raw(path: str, data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception:
        pass
