from __future__ import annotations

import math
import time
import wave
from io import BytesIO
from typing import Any, Dict, Optional

import requests
import streamlit as st

APP_BUILD_ID = "v8.5.4.4-clean-download-ui-20260614"
SAFE_DOWNLOAD_FILENAME = "BAM_mastered_48k_24bit.wav"
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
    # Public-facing names are preferred. Older PRIVATE_WORKER_* names are accepted
    # only as compatibility fallback so existing deployments do not break.
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
                time.sleep(1.2)
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
                wait_sec = min(8.0, max(1.5, float(retry_after))) if retry_after else 2.0 + attempt
            except Exception:
                wait_sec = 2.0 + attempt
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
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) <= 0:
        return "-"
    seconds = int(round(float(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}시간 {minutes}분"
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


def _update_progress(progress_box: Any, status_box: Any, pct: int, text: str, started_at: Optional[float] = None, estimate_sec: Optional[float] = None) -> None:
    pct = max(0, min(100, int(pct)))
    progress_box.progress(pct)
    extra = []
    if started_at:
        extra.append(f"경과 시간: {_format_duration(time.time() - started_at)}")
    if estimate_sec:
        extra.append(f"예상 소요시간: 약 {_format_duration(estimate_sec)}")
    suffix = " · ".join(extra)
    status_box.info(text + (f"\n\n{suffix}" if suffix else ""))


def start_job(uploaded_file: Any, user_note: str, progress_box: Any, status_box: Any) -> Dict[str, Any]:
    cfg = _service_config()
    if not cfg["url"]:
        raise ServiceError("서비스 주소가 설정되지 않았습니다.")
    token = cfg["token"]
    timeout = float(cfg["timeout"] or 60)
    start_timeout = float(cfg["start_timeout"] or 3600)
    data = uploaded_file.getvalue()
    content_type = uploaded_file.type or "audio/wav"
    started_at = time.time()
    duration_sec = _read_wav_duration_sec(data)
    estimate_sec = _estimate_processing_sec(data, duration_sec)

    _update_progress(progress_box, status_box, 5, "파일을 확인하는 중입니다.", started_at, estimate_sec)

    init_payload = _request_json(
        "POST",
        _api_url(cfg["url"], "/v1/jobs/init"),
        token,
        timeout,
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

    # Store job id immediately after init. If the long start request finishes
    # on the service but the browser/Streamlit request is interrupted, the user
    # can still press refresh and recover the finished job.
    st.session_state.busy_job_id = job_id
    st.session_state.busy_job_payload = {
        **init_payload,
        "job_id": job_id,
        "status": "uploading",
        "stage": "uploading",
        "progress_pct": 18,
        "client_estimated_sec": round(estimate_sec, 3),
    }

    _update_progress(progress_box, status_box, 18, "업로드를 준비하는 중입니다.", started_at, estimate_sec)
    _upload_to_signed_url(signed_url, data, content_type)
    st.session_state.busy_job_payload.update({"status": "uploaded", "stage": "uploaded", "progress_pct": 38})
    _update_progress(progress_box, status_box, 38, "파일 업로드가 완료됐습니다.", started_at, estimate_sec)

    _update_progress(progress_box, status_box, 45, "마스터링을 시작하는 중입니다.", started_at, estimate_sec)
    try:
        payload = _request_json(
            "POST",
            _api_url(cfg["url"], f"/v1/jobs/{job_id}/start"),
            token,
            start_timeout,
            retries=0,
            json={"mode": DEFAULT_MODE, "user_note": user_note or ""},
        )
    except ServiceError:
        # If the service completed but the long HTTP response was lost, recover
        # the latest persisted state before showing the error.
        try:
            recovered = get_job(job_id)
        except Exception:
            raise
        recovered.setdefault("client_elapsed_sec", round(time.time() - started_at, 3))
        recovered.setdefault("client_estimated_sec", round(estimate_sec, 3))
        return recovered
    payload.setdefault("client_elapsed_sec", round(time.time() - started_at, 3))
    payload.setdefault("client_estimated_sec", round(estimate_sec, 3))
    return payload


def get_job(job_id: str) -> Dict[str, Any]:
    cfg = _service_config()
    if not cfg["url"]:
        raise ServiceError("서비스 주소가 설정되지 않았습니다.")
    timeout = float(cfg["timeout"] or 60)
    return _request_json("GET", _api_url(cfg["url"], f"/v1/jobs/{job_id}"), cfg["token"], timeout, retries=1)


def download_result(job_id: str, download_url: Optional[str] = None) -> bytes:
    cfg = _service_config()
    timeout = float(cfg["timeout"] or 60)
    if download_url and download_url.startswith(("http://", "https://")):
        url = download_url
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


def render_status(payload: Dict[str, Any]) -> None:
    status = str(payload.get("status", "unknown")).lower()
    stage = str(payload.get("stage", ""))
    message = str(payload.get("message", ""))
    progress = payload.get("progress_pct")
    elapsed = payload.get("elapsed_sec", payload.get("client_elapsed_sec"))
    estimate = payload.get("estimated_processing_sec", payload.get("client_estimated_sec"))

    if status in {"done", "completed", "success"}:
        st.success("마스터링이 완료됐습니다.")
    elif status in {"failed", "error"}:
        st.error("마스터링에 실패했습니다.")
    elif status in {"waiting_upload"}:
        st.info("업로드를 기다리는 중입니다.")
    else:
        st.info("마스터링 중입니다.")

    if isinstance(progress, (int, float)):
        st.progress(max(0, min(100, int(progress))))

    cols = st.columns(3)
    if stage:
        cols[0].metric("현재 단계", _stage_label(stage))
    if isinstance(elapsed, (int, float)):
        cols[1].metric("경과 시간", _format_duration(float(elapsed)))
    if isinstance(estimate, (int, float)):
        cols[2].metric("예상 소요시간", f"약 {_format_duration(float(estimate))}")

    if message:
        st.caption(_clean_message(message))


def _stage_label(stage: str) -> str:
    table = {
        "starting": "준비 중",
        "private_assets_ready": "준비 완료",
        "mastering": "분석/마스터링 중",
        "completed": "완료",
        "failed": "실패",
        "timeout": "시간 초과",
        "exception": "오류",
        "waiting_upload": "업로드 대기",
        "uploading": "업로드 중",
        "uploaded": "업로드 완료",
        "processing": "처리 중",
    }
    return table.get(str(stage).lower(), str(stage))


def _clean_message(message: str) -> str:
    replacements = {
        "Private worker started.": "처리를 시작했습니다.",
        "Private assets loaded.": "처리 준비가 완료됐습니다.",
        "Analysis and mastering are running in the private worker.": "분석과 마스터링을 진행 중입니다.",
        "Mastering completed.": "마스터링이 완료됐습니다.",
        "Upload WAV to signed_upload_url, then call /start.": "업로드를 준비했습니다.",
    }
    return replacements.get(message, message)


def main() -> None:
    st.set_page_config(page_title="Busy Auto Mastering", page_icon="🎧", layout="centered")
    st.title("Busy Auto Mastering")

    cfg = _service_config()
    if not cfg["url"]:
        st.warning("서비스 설정이 필요합니다.")

    uploaded = st.file_uploader("WAV 파일 업로드", type=["wav"], accept_multiple_files=False)
    user_note = st.text_area("장르/스타일태그(선택)", value="", height=70, placeholder="비워두면 자동으로 처리됩니다.")

    if "busy_job_id" not in st.session_state:
        st.session_state.busy_job_id = ""
    if "busy_job_payload" not in st.session_state:
        st.session_state.busy_job_payload = {}

    if uploaded is not None:
        file_data = uploaded.getvalue()
        size_mb = len(file_data) / (1024 * 1024)
        duration_sec = _read_wav_duration_sec(file_data)
        estimate_sec = _estimate_processing_sec(file_data, duration_sec)
        cols = st.columns(3)
        cols[0].metric("파일 크기", f"{size_mb:.1f} MB")
        cols[1].metric("곡 길이", _format_duration(duration_sec))
        cols[2].metric("예상 소요시간", f"약 {_format_duration(estimate_sec)}")

    start = st.button("마스터링 시작", type="primary", disabled=uploaded is None or not cfg["url"])

    progress_box = st.empty()
    status_box = st.empty()

    if start and uploaded is not None:
        file_data = uploaded.getvalue()
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"파일이 너무 큽니다: {size_mb:.1f} MB / 제한 {MAX_UPLOAD_MB} MB")
        else:
            try:
                payload = start_job(uploaded, user_note, progress_box, status_box)
            except ServiceError as exc:
                st.error(str(exc))
                if st.session_state.busy_job_id and st.session_state.busy_job_payload:
                    render_status(st.session_state.busy_job_payload)
            else:
                job_id = str(payload.get("job_id", ""))
                st.session_state.busy_job_id = job_id
                st.session_state.busy_job_payload = payload
                render_status(payload)

    payload = st.session_state.busy_job_payload or {}
    if payload and not start:
        # Quietly recover the latest state on rerun without exposing a manual refresh button.
        job_id_for_status = str(payload.get("job_id") or st.session_state.busy_job_id or "")
        status_now = str(payload.get("status", "")).lower()
        if job_id_for_status and status_now not in {"done", "completed", "success", "failed", "error"}:
            try:
                latest = get_job(job_id_for_status)
            except ServiceError:
                latest = payload
            else:
                payload = latest
                st.session_state.busy_job_payload = latest
        render_status(payload)
    status = str(payload.get("status", "")).lower()
    if status in {"done", "completed", "success"}:
        job_id = str(payload.get("job_id") or st.session_state.busy_job_id)
        download_url = payload.get("download_url")
        cache_key = f"busy_download_bytes_{job_id}"
        if cache_key not in st.session_state:
            try:
                st.session_state[cache_key] = download_result(job_id, str(download_url or ""))
            except ServiceError as exc:
                st.error(str(exc))
                st.session_state[cache_key] = b""
        mastered = st.session_state.get(cache_key, b"")
        if mastered:
            st.download_button(
                "마스터링 WAV 다운로드",
                data=mastered,
                file_name=SAFE_DOWNLOAD_FILENAME,
                mime="audio/wav",
                type="primary",
            )


if __name__ == "__main__":
    main()
