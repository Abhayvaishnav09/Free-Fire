"""
Bridge 16score-desktop to the Free Fire killfeed pipeline (ffkillblock.py).

When Free Fire is selected in Settings, runs ffkillblock.py with bestffmax.pt
(local YOLO + PaddleOCR) and posts killfeeds to the TMS API.
"""
import os
import sys


def _resolve_freefire_dir() -> str:
    """Locate ffkillblock.py inside 16score-desktop (or legacy paths)."""
    desktop_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    parent_root = os.path.abspath(os.path.join(desktop_root, ".."))
    candidates = [
        os.path.join(parent_root, "Free-Fire"),
        os.path.join(parent_root, "Free Fire"),
        parent_root,
        desktop_root,
    ]
    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.isfile(os.path.join(normalized, "ffkillblock.py")):
            return normalized
    raise FileNotFoundError(
        "ffkillblock.py not found. Expected it in the 16score-desktop repo root "
        f"(checked {desktop_root})"
    )


def _setup_import_paths(freefire_dir: str) -> None:
    """Ensure ffkillblock.py and PaddleOCR helper (text.py) are importable."""
    if freefire_dir not in sys.path:
        sys.path.insert(0, freefire_dir)

    for ocr_dir in (
        os.path.join(freefire_dir, "ocr"),
        os.path.normpath(os.path.join(freefire_dir, "..", "Free Fire", "ocr")),
    ):
        if os.path.isfile(os.path.join(ocr_dir, "text.py")) and ocr_dir not in sys.path:
            sys.path.insert(0, ocr_dir)
            break


def _resolve_model_path(freefire_dir: str, model_path: str) -> str:
    if os.path.isabs(model_path) and os.path.isfile(model_path):
        return model_path
    candidates = [
        os.path.join(freefire_dir, model_path),
        os.path.join(os.path.dirname(freefire_dir), model_path),
        os.path.join(freefire_dir, "16score_ai", "models", os.path.basename(model_path)),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(freefire_dir, model_path))


def sync_desktop_auth(match_id=None, access_token=None, user_email=None):
    """Write Desktop login/match into Free-Fire/.desktop_session.json for standalone runs."""
    try:
        freefire_dir = _resolve_freefire_dir()
        _setup_import_paths(freefire_dir)
        from killfeed.desktop_session import save_desktop_match, save_desktop_token

        if access_token:
            save_desktop_token(access_token, user_email, base_dir=freefire_dir)
        if match_id:
            save_desktop_match(str(match_id), access_token, user_email, base_dir=freefire_dir)
    except Exception as exc:
        print(f"[Free Fire] session sync skipped: {exc}", flush=True)


def obs_frame_capture(match_id=1, access_token=None, camera_index=1, stop_flag=None):
    """
    Start Free Fire killfeed detection from OBS virtual camera.

    Matches the signature used by camera_setup_pyqt.py and killblocks.py.
    """
    from score_ai.core.config_manager import config

    freefire_dir = _resolve_freefire_dir()
    _setup_import_paths(freefire_dir)

    backend_url = config.get("api.backend_url", "").rstrip("/")
    if backend_url:
        os.environ["SCORE_API_URL"] = backend_url

    use_local = config.get("freefire.use_local_model", True)
    grpc_port = int(config.get("freefire.grpc_port", 50051))
    model_path = config.get("freefire.model_path", "bestffmax.pt")
    model_path = _resolve_model_path(freefire_dir, model_path)

    if use_local and not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Free Fire YOLO model not found: {model_path}. "
            "Place bestffmax.pt next to ffkillblock.py or set freefire.model_path in config.json."
        )

    print(
        f"[Free Fire] ffkillblock.py | match={match_id} | camera={camera_index} | "
        f"model={model_path} | local_yolo={use_local} | API={backend_url}",
        flush=True,
    )
    print(
        "[Free Fire] Loading YOLO model (OCR loads in background — first start may take ~1 min)",
        flush=True,
    )

    sync_desktop_auth(match_id=match_id, access_token=access_token)

    original_cwd = os.getcwd()
    os.chdir(freefire_dir)
    try:
        from ffkillblock import run_obs_capture

        run_obs_capture(
            match_id=match_id,
            access_token=access_token,
            camera_index=camera_index,
            stop_flag=stop_flag,
            use_local_model=use_local,
            grpc_port=grpc_port,
            model_path=model_path,
        )
    finally:
        os.chdir(original_cwd)


def get_obs_frame_capture():
    """Return the capture function for Free Fire (used by camera_setup routing)."""
    return obs_frame_capture
