"""
Busy Auto Mastering - public frontend only, signed-upload private-worker client.

This repository intentionally contains only the UI and private worker client.
All analysis, processing, decision logic, and rendering run inside a private worker.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests
import streamlit as st

APP_BUILD_ID = "v8.5.4-frontend-only-signed-upload-cloudrun-20260614"
DEFAULT_MODE = "Auto Commercial Master"
MAX_UPLOAD_MB = 250


class WorkerError(RuntimeError):
    pass


def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or "").strip()


def _worker_config() -> Dict[str, str]:
    return {
        "url": _secret("PRIVATE_WORKER_URL"),
        "token": _secret("PRIVATE_WORKER_TOKEN"),
        "timeout": _secret("PRIVATE_WORKER_TIMEOUT_SEC", "60"),
        "start_timeout": _secret("PRIVATE_WORKER_START_TIMEOUT_SEC", "3600"),
    }


def _headers(token: str) -> Dict[str, str]:
    headers = {"X-Client-Build": APP_BUILD_ID}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _request_json(method: str, url: str, token: str, timeout: float, **kwargs: Any) -> Dict[str, Any]:
    try:
        resp = requests.request(method, url, headers=_headers(token), timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise WorkerError(f"Private worker connection failed: {exc}") from exc
    if resp.status_code >= 400:
        raise WorkerError(f"Private worker returned HTTP {resp.status_code}: {resp.text[:1200]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise WorkerError("Private worker returned a non-JSON response.") from exc
    if not isinstance(payload, dict):
        raise WorkerError("Private worker returned an invalid response shape.")
    return payload


def _upload_to_signed_url(url: str, data: bytes, content_type: str) -> None:
    headers = {"Content-Type": content_type or "audio/wav"}
    # Supabase signed upload URLs normally accept PUT. Some deployments/proxies
    # may accept POST, so POST is tried as a fallback.
    resp = requests.put(url, data=data, headers=headers, timeout=600)
    if resp.status_code >= 400:
        resp2 = requests.post(url, data=data, headers=headers, timeout=600)
        if resp2.status_code >= 400:
            raise WorkerError(f"Signed upload failed: PUT {resp.status_code} / POST {resp2.status_code}: {resp2.text[:800]}")


def start_job(uploaded_file: Any, mode: str, user_note: str) -> Dict[str, Any]:
    cfg = _worker_config()
    if not cfg["url"]:
        raise WorkerError("PRIVATE_WORKER_URL is not configured in Streamlit Secrets.")
    token = cfg["token"]
    timeout = float(cfg["timeout"] or 60)
    start_timeout = float(cfg["start_timeout"] or 3600)
    data = uploaded_file.getvalue()
    content_type = uploaded_file.type or "audio/wav"

    init_payload = _request_json(
        "POST",
        _api_url(cfg["url"], "/v1/jobs/init"),
        token,
        timeout,
        json={
            "filename": uploaded_file.name,
            "size_bytes": len(data),
            "content_type": content_type,
            "mode": mode,
            "user_note": user_note or "",
            "client_build_id": APP_BUILD_ID,
        },
    )
    signed_url = str(init_payload.get("signed_upload_url") or "")
    job_id = str(init_payload.get("job_id") or "")
    if not signed_url or not job_id:
        raise WorkerError("Private worker did not return signed_upload_url/job_id.")

    _upload_to_signed_url(signed_url, data, content_type)

    # Start is intentionally a long-running request. This avoids relying on
    # background tasks in request-based Cloud Run instances.
    return _request_json(
        "POST",
        _api_url(cfg["url"], f"/v1/jobs/{job_id}/start"),
        token,
        start_timeout,
        json={"mode": mode, "user_note": user_note or ""},
    )


def get_job(job_id: str) -> Dict[str, Any]:
    cfg = _worker_config()
    if not cfg["url"]:
        raise WorkerError("PRIVATE_WORKER_URL is not configured in Streamlit Secrets.")
    timeout = float(cfg["timeout"] or 60)
    return _request_json("GET", _api_url(cfg["url"], f"/v1/jobs/{job_id}"), cfg["token"], timeout)


def download_result(job_id: str, download_url: Optional[str] = None) -> bytes:
    cfg = _worker_config()
    timeout = float(cfg["timeout"] or 60)
    if download_url and download_url.startswith(("http://", "https://")):
        url = download_url
        headers = {}
    else:
        if not cfg["url"]:
            raise WorkerError("PRIVATE_WORKER_URL is not configured in Streamlit Secrets.")
        url = _api_url(cfg["url"], f"/v1/jobs/{job_id}/download")
        headers = _headers(cfg["token"])
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise WorkerError(f"Result download failed: {exc}") from exc
    if resp.status_code >= 400:
        raise WorkerError(f"Result download returned HTTP {resp.status_code}: {resp.text[:800]}")
    return resp.content


def render_status(payload: Dict[str, Any]) -> None:
    status = str(payload.get("status", "unknown")).lower()
    stage = str(payload.get("stage", ""))
    message = str(payload.get("message", ""))
    progress = payload.get("progress_pct")

    if status in {"done", "completed", "success"}:
        st.success("Mastering completed.")
    elif status in {"failed", "error"}:
        st.error("Mastering failed.")
    else:
        st.info("Processing in private worker...")

    if isinstance(progress, (int, float)):
        st.progress(max(0, min(100, int(progress))))
    if stage:
        st.caption(f"Stage: {stage}")
    if message:
        st.caption(message)


def main() -> None:
    st.set_page_config(page_title="Busy Auto Mastering", page_icon="🎧", layout="centered")
    st.title("Busy Auto Mastering")
    st.caption(f"Public frontend build: {APP_BUILD_ID}")

    cfg = _worker_config()
    if not cfg["url"]:
        st.warning("Private worker is not configured. Add PRIVATE_WORKER_URL and PRIVATE_WORKER_TOKEN in Streamlit Secrets.")

    uploaded = st.file_uploader("Upload WAV file", type=["wav"], accept_multiple_files=False)
    mode = st.selectbox("Mode", [DEFAULT_MODE], index=0)
    user_note = st.text_area("Optional style note", value="", height=80, placeholder="Optional. Leave blank for automatic mastering.")

    if "busy_job_id" not in st.session_state:
        st.session_state.busy_job_id = ""
    if "busy_job_payload" not in st.session_state:
        st.session_state.busy_job_payload = {}

    start = st.button("Start mastering", type="primary", disabled=uploaded is None or not cfg["url"])
    refresh = st.button("Refresh last status", disabled=not bool(st.session_state.busy_job_id))

    if start and uploaded is not None:
        file_data = uploaded.getvalue()
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"File is too large: {size_mb:.1f} MB. Limit is {MAX_UPLOAD_MB} MB.")
        else:
            with st.spinner("Uploading to private storage and running private worker..."):
                try:
                    payload = start_job(uploaded, mode, user_note)
                except WorkerError as exc:
                    st.error(str(exc))
                else:
                    job_id = str(payload.get("job_id", ""))
                    st.session_state.busy_job_id = job_id
                    st.session_state.busy_job_payload = payload
                    render_status(payload)

    if refresh and st.session_state.busy_job_id:
        try:
            payload = get_job(st.session_state.busy_job_id)
        except WorkerError as exc:
            st.error(str(exc))
            payload = st.session_state.busy_job_payload or {}
        else:
            st.session_state.busy_job_payload = payload
        if payload:
            render_status(payload)

    payload = st.session_state.busy_job_payload or {}
    status = str(payload.get("status", "")).lower()
    if status in {"done", "completed", "success"}:
        job_id = str(payload.get("job_id") or st.session_state.busy_job_id)
        filename = str(payload.get("filename") or f"busy_mastered_{job_id}.wav")
        download_url = payload.get("download_url")
        if st.button("Prepare download"):
            with st.spinner("Fetching mastered WAV..."):
                try:
                    mastered = download_result(job_id, str(download_url or ""))
                except WorkerError as exc:
                    st.error(str(exc))
                else:
                    st.download_button("Download mastered WAV", data=mastered, file_name=filename, mime="audio/wav")

    with st.expander("Security note", expanded=False):
        st.write(
            "This public repository contains only the upload UI and private-worker client. "
            "Audio analysis, decision logic, DSP mapping, rendering, private rules, and calibration data are not included."
        )


if __name__ == "__main__":
    main()
