from __future__ import annotations

import math
import re
import threading
import time
import wave
from io import BytesIO
from typing import Any, Dict, Optional

import requests
import streamlit as st

APP_BUILD_ID = "v8.5.4.5-clean-progress-singlebar-20260614"
DEFAULT_MODE = "Auto Commercial Master"
MAX_UPLOAD_MB = 250


class ServiceError(RuntimeError):
    pass


def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return str(value or "").strip()


def _service_config() -> Dict[str, str]:
    return {
        "url": _secret("SERVICE_ENDPOINT_URL") or _secret("PRIVATE_WORKER_URL"),
        "token": _secret("SERVICE_ACCESS_TOKEN") or _secret("PRIVATE_WORKER_TOKEN"),
        "timeout": _secret("SERVICE_TIMEOUT_SEC", _secret("PRIVATE_WORKER_TIMEOUT_SEC", "60")),
        "start_timeout": _secret("SERVICE_START_TIMEOUT_SEC", _secret("PRIVATE_WORKER_START_TIMEOUT_SEC", "3600")),
    }


def _headers(token: str) -> Dict[str, str]:
    headers = {"X-Client-Build": APP_BUILD_ID}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _friendly_http_error(status_code: int, body: str = "") -> str:
    if status_code == 429:
        return "현재 다른 작업을 처리 중입니다. 잠시 후 다시 시도해 주세요."
    if status_code in (401, 403):
        return "서비스 인증 설정을 확인해 주세요."
    if status_code == 413:
        return "파일 용량이 너무 큽니다. 더 짧은 WAV로 먼저 테스트해 주세요."
    if status_code >= 500:
        return "처리 서비스에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    return f"서비스 오류가 발생했습니다. HTTP {status_code}: {body[:500]}"


def _request_json(method: str, url: str, token: str, timeout: float, *, retries: int = 1, **kwargs: Any) -> Dict[str, Any]:
    last_error = ""
    for attempt in range(max(1, retries + 1)):
        try:
            resp = requests.request(method, url, headers=_headers(token), timeout=timeout, **kwargs)
        except requests.RequestException as exc:
            last_error = f"서비스 연결에 실패했습니다: {exc}"
            if attempt < retries:
                time.sleep(1.0)
                continue
            raise ServiceError(last_error) from exc
        if resp.status_code < 400:
            try:
                payload = resp.json()
            except ValueError as exc:
                raise ServiceError("서비스 응답 형식이 올바르지 않습니다.") from exc
            if not isinstance(payload, dict):
                raise ServiceError("서비스 응답 형식이 올바르지 않습니다.")
            return payload
        if resp.status_code == 429 and attempt < retries:
            retry_after = resp.headers.get("Retry-After", "")
            try:
                wait_sec = min(10.0, max(1.5, float(retry_after))) if retry_after else min(10.0, 2.0 + attempt)
            except Exception:
                wait_sec = min(10.0, 2.0 + attempt)
            time.sleep(wait_sec)
            continue
        raise ServiceError(_friendly_http_error(resp.status_code, resp.text or ""))
    raise ServiceError(last_error or "서비스 요청에 실패했습니다.")


def _upload_to_signed_url(url: str, data: bytes, content_type: str) -> None:
    headers = {"Content-Type": content_type or "audio/wav"}
    try:
        resp = requests.put(url, data=data, headers=headers, timeout=900)
    except requests.RequestException as exc:
        raise ServiceError(f"업로드에 실패했습니다: {exc}") from exc
    if resp.status_code >= 400:
        try:
            resp2 = requests.post(url, data=data, headers=headers, timeout=900)
        except requests.RequestException as exc:
            raise ServiceError(f"업로드에 실패했습니다: {exc}") from exc
        if resp2.status_code >= 400:
            raise ServiceError(_friendly_http_error(resp2.status_code, resp2.text or ""))


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "0초"
    seconds = int(float(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}시간 {minutes}분 {sec}초"
    if minutes:
        return f"{minutes}분 {sec}초"
    return f"{sec}초"


def _read_wav_duration_sec(data: bytes) -> Optional[float]:
    try:
        with wave.open(BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate > 0:
                return frames / float(rate)
    except Exception:
        return None
    return None


def _estimate_processing_sec(data: bytes, duration_sec: Optional[float]) -> float:
    size_mb = len(data) / (1024 * 1024)
    if duration_sec and duration_sec > 0:
        return max(90.0, min(3600.0, duration_sec * 3.9 + 60.0))
    return max(90.0, min(3600.0, size_mb * 18.0 + 75.0))


def _download_filename_from_original(name: str) -> str:
    base = str(name or "BAM_mastered.wav").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in base:
        stem = ".".join(base.split(".")[:-1]) or base
    else:
        stem = base
    stem = re.sub(r'[\\/:*?"<>|]+', "_", stem).strip(" ._") or "mastered"
    return f"{stem}_BAM_mastered_48k_24bit.wav"


def _job_is_done(payload: Dict[str, Any]) -> bool:
    return str(payload.get("status", "")).lower() in {"done", "completed", "success"}


def _job_is_failed(payload: Dict[str, Any]) -> bool:
    return str(payload.get("status", "")).lower() in {"failed", "error"}


def _job_is_active(payload: Dict[str, Any]) -> bool:
    if not payload:
        return False
    return not (_job_is_done(payload) or _job_is_failed(payload))


def _poll_job(job_id: str) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    cfg = _service_config()
    if not cfg["url"]:
        return None
    try:
        return _request_json("GET", _api_url(cfg["url"], f"/v1/jobs/{job_id}"), cfg["token"], float(cfg["timeout"] or 60), retries=1)
    except ServiceError:
        return None


def _start_worker_request(job_id: str, user_note: str) -> None:
    cfg = _service_config()
    try:
        _request_json(
            "POST",
            _api_url(cfg["url"], f"/v1/jobs/{job_id}/start"),
            cfg["token"],
            float(cfg["start_timeout"] or 3600),
            retries=3,
            json={"mode": DEFAULT_MODE, "user_note": user_note or ""},
        )
    except Exception:
        # The UI polls persisted job state, so the thread should not crash Streamlit.
        pass


def _launch_start_thread(job_id: str, user_note: str) -> None:
    current = st.session_state.get("busy_start_thread")
    if current is not None and getattr(current, "is_alive", lambda: False)():
        return
    thread = threading.Thread(target=_start_worker_request, args=(job_id, user_note or ""), daemon=True)
    st.session_state.busy_start_thread = thread
    thread.start()


def _init_upload_and_start(uploaded_file: Any, user_note: str) -> None:
    cfg = _service_config()
    if not cfg["url"]:
        raise ServiceError("서비스 주소가 설정되지 않았습니다.")
    data = uploaded_file.getvalue()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ServiceError(f"파일이 너무 큽니다: {size_mb:.1f} MB / 제한 {MAX_UPLOAD_MB} MB")
    duration_sec = _read_wav_duration_sec(data)
    estimate_sec = _estimate_processing_sec(data, duration_sec)
    content_type = uploaded_file.type or "audio/wav"
    started_at = time.time()

    init_payload = _request_json(
        "POST",
        _api_url(cfg["url"], "/v1/jobs/init"),
        cfg["token"],
        float(cfg["timeout"] or 60),
        retries=1,
        json={
            "filename": uploaded_file.name,
            "size_bytes": len(data),
            "content_type": content_type,
            "mode": DEFAULT_MODE,
            "user_note": user_note or "",
            "client_build_id": APP_BUILD_ID,
        },
    )
    signed_url = str(init_payload.get("signed_upload_url") or "")
    job_id = str(init_payload.get("job_id") or "")
    if not signed_url or not job_id:
        raise ServiceError("업로드 준비 응답이 올바르지 않습니다.")

    st.session_state.busy_job_id = job_id
    st.session_state.busy_started_at = started_at
    st.session_state.busy_estimate_sec = estimate_sec
    st.session_state.busy_original_filename = uploaded_file.name
    st.session_state.busy_download_filename = _download_filename_from_original(uploaded_file.name)
    st.session_state.busy_download_bytes = b""
    st.session_state.busy_job_payload = {
        **init_payload,
        "job_id": job_id,
        "status": "uploading",
        "stage": "uploading",
        "progress_pct": 10,
    }

    _upload_to_signed_url(signed_url, data, content_type)
    st.session_state.busy_job_payload.update({"status": "processing", "stage": "mastering", "progress_pct": 20})
    _launch_start_thread(job_id, user_note)


def _display_job(payload: Dict[str, Any]) -> None:
    started_at = float(st.session_state.get("busy_started_at") or time.time())
    elapsed = time.time() - started_at
    estimate_sec = float(st.session_state.get("busy_estimate_sec") or 900.0)

    if _job_is_done(payload):
        progress = 100
        st.progress(progress)
        st.success(f"마스터링이 완료됐습니다. · 경과 시간: {_format_duration(elapsed)}")
        return

    if _job_is_failed(payload):
        st.progress(100)
        st.error("마스터링에 실패했습니다.")
        msg = str(payload.get("message") or payload.get("error_tail") or "")
        if msg:
            st.caption(msg[:1000])
        return

    worker_progress = payload.get("progress_pct")
    estimated_progress = 20
    if estimate_sec > 0:
        estimated_progress = int(min(95, max(20, (elapsed / estimate_sec) * 95)))
    if isinstance(worker_progress, (int, float)):
        progress = max(int(worker_progress), estimated_progress)
    else:
        progress = estimated_progress
    progress = max(1, min(95, progress))
    st.progress(progress)
    st.info(f"마스터링 중입니다. · 경과 시간: {_format_duration(elapsed)}")


def _download_result(job_id: str, download_url: Optional[str] = None) -> bytes:
    cfg = _service_config()
    timeout = float(cfg["timeout"] or 60)
    if download_url and str(download_url).startswith(("http://", "https://")):
        url = str(download_url)
        headers: Dict[str, str] = {}
    else:
        if not cfg["url"]:
            raise ServiceError("서비스 주소가 설정되지 않았습니다.")
        url = _api_url(cfg["url"], f"/v1/jobs/{job_id}/download")
        headers = _headers(cfg["token"])
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise ServiceError(f"다운로드에 실패했습니다: {exc}") from exc
    if resp.status_code >= 400:
        raise ServiceError(_friendly_http_error(resp.status_code, resp.text or ""))
    return resp.content


def _reset_if_new_upload(uploaded_name: str) -> None:
    prev = st.session_state.get("busy_last_uploaded_name")
    if prev and prev != uploaded_name and _job_is_done(st.session_state.get("busy_job_payload") or {}):
        for key in [
            "busy_job_id", "busy_job_payload", "busy_started_at", "busy_estimate_sec",
            "busy_original_filename", "busy_download_filename", "busy_download_bytes", "busy_start_thread",
        ]:
            st.session_state.pop(key, None)
    st.session_state.busy_last_uploaded_name = uploaded_name


def main() -> None:
    st.set_page_config(page_title="Busy Auto Mastering", page_icon="🎧", layout="centered")
    st.title("Busy Auto Mastering")

    cfg = _service_config()
    if not cfg["url"]:
        st.warning("서비스 설정이 필요합니다.")

    uploaded = st.file_uploader("WAV 파일 업로드", type=["wav"], accept_multiple_files=False)
    user_note = st.text_area("선택 메모", value="", height=70, placeholder="비워두면 자동으로 처리됩니다.")

    if uploaded is not None:
        _reset_if_new_upload(uploaded.name)
        file_data = uploaded.getvalue()
        size_mb = len(file_data) / (1024 * 1024)
        duration_sec = _read_wav_duration_sec(file_data)
        cols = st.columns(2)
        cols[0].metric("파일 크기", f"{size_mb:.1f} MB")
        cols[1].metric("곡 길이", _format_duration(duration_sec))

    payload = st.session_state.get("busy_job_payload") or {}
    active = _job_is_active(payload)
    start_disabled = uploaded is None or not cfg["url"] or active
    start_clicked = st.button("마스터링 시작", type="primary", disabled=start_disabled)

    if start_clicked and uploaded is not None:
        try:
            _init_upload_and_start(uploaded, user_note)
        except ServiceError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    job_id = str(st.session_state.get("busy_job_id") or "")
    if job_id:
        latest = _poll_job(job_id)
        if latest:
            st.session_state.busy_job_payload = latest
            payload = latest
        _display_job(payload)

        if _job_is_done(payload):
            download_url = payload.get("download_url")
            if not st.session_state.get("busy_download_bytes"):
                try:
                    st.session_state.busy_download_bytes = _download_result(job_id, str(download_url or ""))
                except ServiceError as exc:
                    st.error(str(exc))
                    st.session_state.busy_download_bytes = b""
            if st.session_state.get("busy_download_bytes"):
                st.download_button(
                    "마스터링 WAV 다운로드",
                    data=st.session_state.busy_download_bytes,
                    file_name=st.session_state.get("busy_download_filename") or "BAM_mastered_48k_24bit.wav",
                    mime="audio/wav",
                    type="primary",
                )
        elif _job_is_active(payload):
            time.sleep(1)
            st.rerun()


if __name__ == "__main__":
    main()
