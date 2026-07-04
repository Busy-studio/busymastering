from __future__ import annotations

import math
import re
import threading
import unicodedata
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Any, Dict, Optional

import requests
import streamlit as st

APP_BUILD_ID = "v8.5.4.20.1-public-busy-auto-mixing-start-error-hotfix-20260625"
DEFAULT_MODE = "Auto Commercial Master"
MAX_FILE_UPLOAD_MB = 500
MAX_UPLOAD_MB = MAX_FILE_UPLOAD_MB
MAX_STEM_ZIP_MB = MAX_FILE_UPLOAD_MB


class ServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


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
        "poll_timeout": _secret("SERVICE_POLL_TIMEOUT_SEC", "2.0"),
        "poll_interval": _secret("SERVICE_STATUS_POLL_INTERVAL_SEC", "5.0"),
    }


def _safe_float(value: Any, default: float) -> float:
    try:
        value_f = float(value)
        if math.isfinite(value_f):
            return value_f
    except Exception:
        pass
    return float(default)


def _poll_timeout_sec() -> float:
    cfg = _service_config()
    return max(0.8, min(8.0, _safe_float(cfg.get("poll_timeout"), 2.0)))


def _poll_interval_sec() -> float:
    cfg = _service_config()
    return max(2.0, min(20.0, _safe_float(cfg.get("poll_interval"), 5.0)))


def _status_backoff_max_sec() -> float:
    return max(5.0, min(120.0, _safe_float(_secret("SERVICE_STATUS_POLL_BACKOFF_MAX_SEC", "45.0"), 45.0)))


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


def _retry_after_sec(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        parsed = float(str(value).strip())
        if math.isfinite(parsed) and parsed > 0:
            return max(1.0, min(120.0, parsed))
    except Exception:
        return None
    return None


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
        retry_after = _retry_after_sec(resp.headers.get("Retry-After", ""))
        if resp.status_code == 429 and attempt < retries:
            wait_sec = retry_after if retry_after is not None else min(10.0, 2.0 + attempt)
            time.sleep(wait_sec)
            continue
        raise ServiceError(_friendly_http_error(resp.status_code, resp.text or ""), status_code=resp.status_code, retry_after=retry_after)
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


def _payload_estimate_sec(payload: Dict[str, Any]) -> Optional[float]:
    value = payload.get("estimated_processing_sec")
    if value is None and isinstance(payload.get("processing_estimate"), dict):
        value = payload["processing_estimate"].get("estimated_processing_sec")
    try:
        if value is None:
            return None
        value_f = float(value)
        if value_f > 0 and math.isfinite(value_f):
            return value_f
    except Exception:
        return None
    return None


def _download_filename_from_original(name: str) -> str:
    base = str(name or "BAM.wav").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    base = unicodedata.normalize("NFC", base)
    if "." in base:
        stem = ".".join(base.split(".")[:-1]) or base
    else:
        stem = base
    # Keep Unicode letters from Korean/English/other languages.  Only replace
    # filesystem-invalid characters and invisible control chars.
    stem = re.sub(r'[\\/:*?"<>|]+', "_", stem)
    stem = re.sub(r"[\x00-\x1f\x7f]+", "_", stem)
    stem = stem.strip(" ._") or "mastered"
    return f"{stem}_BAM_48k_24bit.wav"



def _is_zip_upload(uploaded: Any) -> bool:
    name = str(getattr(uploaded, "name", "") or "").lower()
    return name.endswith(".zip")


def _is_wav_upload(uploaded: Any) -> bool:
    name = str(getattr(uploaded, "name", "") or "").lower()
    return name.endswith(".wav")


def _split_upload_selection(uploaded_files: Any) -> tuple[Optional[Any], Optional[Any], str]:
    files = list(uploaded_files or [])
    wavs = [f for f in files if _is_wav_upload(f)]
    zips = [f for f in files if _is_zip_upload(f)]
    others = [f for f in files if not _is_wav_upload(f) and not _is_zip_upload(f)]
    if others:
        return None, None, "WAV 파일과 ZIP 파일만 업로드할 수 있습니다."
    if not wavs:
        return None, zips[0] if zips else None, "완성본 WAV 파일을 1개 업로드해 주세요."
    if len(wavs) > 1:
        return None, zips[0] if zips else None, "완성본 WAV는 1개만 업로드해 주세요. 분리 트랙은 ZIP으로 올려주세요."
    if len(zips) > 1:
        return wavs[0], None, "분리트랙 ZIP은 1개만 업로드해 주세요."
    return wavs[0], zips[0] if zips else None, ""


def _upload_selection_key(uploaded_files: Any) -> str:
    files = list(uploaded_files or [])
    return "|".join(f"{getattr(f, 'name', '')}:{getattr(f, 'size', '')}" for f in files)

def _job_is_done(payload: Dict[str, Any]) -> bool:
    return str(payload.get("status", "")).lower() in {"done", "completed", "success"}


def _job_is_failed(payload: Dict[str, Any]) -> bool:
    return str(payload.get("status", "")).lower() in {"failed", "error"}


def _job_is_active(payload: Dict[str, Any]) -> bool:
    if not payload:
        return False
    return not (_job_is_done(payload) or _job_is_failed(payload))


def _poll_job(job_id: str, *, timeout_sec: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    cfg = _service_config()
    if not cfg["url"]:
        return None
    timeout = _safe_float(timeout_sec, _poll_timeout_sec()) if timeout_sec is not None else _poll_timeout_sec()
    try:
        payload = _request_json("GET", _api_url(cfg["url"], f"/v1/jobs/{job_id}"), cfg["token"], timeout, retries=0)
        st.session_state.busy_status_poll_error_count = 0
        st.session_state.busy_status_poll_backoff_until = 0.0
        st.session_state.busy_status_poll_last_error = ""
        return payload
    except ServiceError as exc:
        count = int(st.session_state.get("busy_status_poll_error_count") or 0) + 1
        st.session_state.busy_status_poll_error_count = count
        if exc.status_code == 429:
            wait_sec = exc.retry_after if exc.retry_after is not None else min(_status_backoff_max_sec(), max(_poll_interval_sec(), 3.0) * min(4.0, 1.0 + count * 0.5))
            st.session_state.busy_status_poll_backoff_until = max(
                float(st.session_state.get("busy_status_poll_backoff_until") or 0.0),
                time.time() + wait_sec,
            )
            st.session_state.busy_status_poll_last_error = "rate_limited"
            st.session_state.busy_status_poll_retry_after_sec = wait_sec
        elif count >= 2:
            wait_sec = min(_status_backoff_max_sec(), _poll_interval_sec() + count)
            st.session_state.busy_status_poll_backoff_until = max(
                float(st.session_state.get("busy_status_poll_backoff_until") or 0.0),
                time.time() + wait_sec,
            )
            st.session_state.busy_status_poll_last_error = "temporary_poll_error"
        return None


def _status_poll_due() -> bool:
    now = time.time()
    backoff_until = _safe_float(st.session_state.get("busy_status_poll_backoff_until"), 0.0)
    if now < backoff_until:
        return False
    job_id = str(st.session_state.get("busy_job_id") or "")
    last_job_id = str(st.session_state.get("busy_last_status_poll_job_id") or "")
    if job_id and last_job_id and job_id != last_job_id:
        st.session_state.busy_status_poll_error_count = 0
        st.session_state.busy_status_poll_backoff_until = 0.0
    last = _safe_float(st.session_state.get("busy_last_status_poll_at"), 0.0)
    if now - last < _poll_interval_sec():
        return False
    st.session_state.busy_last_status_poll_at = now
    st.session_state.busy_last_status_poll_job_id = job_id
    return True


def _start_worker_request_with_cfg(cfg: Dict[str, str], job_id: str, user_note: str, busy_auto_mixing: bool = False, *, suppress_errors: bool = True) -> None:
    try:
        _request_json(
            "POST",
            _api_url(cfg["url"], f"/v1/jobs/{job_id}/start"),
            cfg["token"],
            float(cfg["start_timeout"] or 3600),
            retries=3,
            json={"mode": DEFAULT_MODE, "user_note": user_note or "", "busy_auto_mixing": bool(busy_auto_mixing)},
        )
    except Exception:
        # Background polling can tolerate a swallowed exception, but the initial
        # bootstrap path must surface /start errors such as missing stem ZIP handoff.
        if suppress_errors:
            pass
        else:
            raise


def _start_worker_request(job_id: str, user_note: str, busy_auto_mixing: bool = False) -> None:
    cfg = _service_config()
    _start_worker_request_with_cfg(cfg, job_id, user_note, busy_auto_mixing)


def _launch_start_thread(job_id: str, user_note: str, busy_auto_mixing: bool = False) -> None:
    current = st.session_state.get("busy_start_thread")
    if current is not None and getattr(current, "is_alive", lambda: False)():
        return
    thread = threading.Thread(target=_start_worker_request, args=(job_id, user_note or "", bool(busy_auto_mixing)), daemon=True)
    st.session_state.busy_start_thread = thread
    thread.start()


def _async_payload(status: str, stage: str, progress_pct: float, message: str = "") -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": status,
        "stage": stage,
        "progress_pct": progress_pct,
    }
    if message:
        payload["message"] = message
    return payload


def _start_job_bootstrap_worker(
    *,
    cfg: Dict[str, str],
    data: bytes,
    filename: str,
    content_type: str,
    duration_sec: Optional[float],
    user_note: str,
    shared: Dict[str, Any],
    stem_zip_data: bytes = b"",
    stem_zip_filename: str = "",
    stem_zip_content_type: str = "application/zip",
    busy_auto_mixing: bool = False,
) -> None:
    """Runs the slow init/upload/start path off the Streamlit render thread.

    The main script renders an instant local progress shell first, then syncs this
    small shared dict on each fragment refresh. The thread deliberately avoids
    Streamlit APIs so missing ScriptRunContext warnings and UI blocking are avoided.
    """
    try:
        shared.update({
            "phase": "initializing",
            "payload": _async_payload("initializing", "preparing", 3, "업로드 준비 중"),
        })
        init_payload = _request_json(
            "POST",
            _api_url(cfg["url"], "/v1/jobs/init"),
            cfg["token"],
            float(cfg["timeout"] or 60),
            retries=1,
            json={
                "filename": filename,
                "size_bytes": len(data),
                "audio_duration_sec": duration_sec,
                "content_type": content_type,
                "stem_zip_filename": stem_zip_filename or "",
                "stem_zip_size_bytes": len(stem_zip_data or b""),
                "stem_zip_content_type": stem_zip_content_type or "application/zip",
                "busy_auto_mixing": bool(busy_auto_mixing),
                "mode": DEFAULT_MODE,
                "user_note": user_note or "",
                "client_build_id": APP_BUILD_ID,
            },
        )
        signed_url = str(init_payload.get("signed_upload_url") or "")
        stem_signed_url = str(init_payload.get("signed_stem_zip_upload_url") or "")
        job_id = str(init_payload.get("job_id") or "")
        if not signed_url or not job_id:
            raise ServiceError("업로드 준비 응답이 올바르지 않습니다.")
        if stem_zip_data and not stem_signed_url:
            raise ServiceError("분리트랙 ZIP 업로드 준비 응답이 올바르지 않습니다.")

        payload = {
            **init_payload,
            "job_id": job_id,
            "status": "uploading",
            "stage": "uploading",
            "progress_pct": 10,
            "message": "파일 업로드 중",
        }
        shared.update({
            "phase": "uploading",
            "job_id": job_id,
            "estimate_sec": _payload_estimate_sec(init_payload),
            "payload": payload,
        })

        if stem_zip_data and stem_signed_url:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_main = ex.submit(_upload_to_signed_url, signed_url, data, content_type)
                fut_zip = ex.submit(_upload_to_signed_url, stem_signed_url, stem_zip_data, stem_zip_content_type or "application/zip")
                fut_main.result()
                fut_zip.result()
        else:
            _upload_to_signed_url(signed_url, data, content_type)
        payload = {
            **init_payload,
            "job_id": job_id,
            "status": "processing",
            "stage": "busy_auto_mixing" if busy_auto_mixing else "mastering",
            "progress_pct": 20,
            "message": "오토 믹싱 및 마스터링 중" if busy_auto_mixing else "마스터링 중",
        }
        shared.update({"phase": "processing", "payload": payload})
        _start_worker_request_with_cfg(cfg, job_id, user_note, busy_auto_mixing, suppress_errors=False)
        shared.update({"phase": "started", "payload": payload, "done": True})
    except Exception as exc:
        shared.update({
            "phase": "error",
            "error": str(exc),
            "done": True,
            "payload": _async_payload("failed", "start_failed", 100, str(exc)),
        })


def _sync_bootstrap_state() -> None:
    shared = st.session_state.get("busy_async_start_state")
    if not isinstance(shared, dict):
        return
    payload = shared.get("payload")
    job_id = str(shared.get("job_id") or "")
    phase = str(shared.get("phase") or "")

    if isinstance(payload, dict):
        # Do not overwrite a terminal worker response or a newer polled worker status with
        # the older local bootstrap payload after the real job has started.
        current = st.session_state.get("busy_job_payload") or {}
        current_status = str(current.get("status") or "").lower()
        current_job_id = str(current.get("job_id") or "")
        shared_status = str(payload.get("status") or "").lower()
        should_update = not (_job_is_done(current) or _job_is_failed(current))
        if (
            should_update
            and job_id
            and current_job_id == job_id
            and phase in {"processing", "started"}
            and current_status not in {"", "initializing", "queued", "uploading"}
            and shared_status in {"processing", "uploading", "initializing"}
        ):
            should_update = False
        if should_update:
            st.session_state.busy_job_payload = dict(payload)
    if job_id:
        st.session_state.busy_job_id = job_id
    if shared.get("estimate_sec") is not None:
        st.session_state.busy_estimate_sec = shared.get("estimate_sec")
    if phase == "error":
        st.session_state.busy_job_payload = dict(shared.get("payload") or _async_payload("failed", "start_failed", 100, str(shared.get("error") or "")))


def _begin_async_start_feedback(uploaded_file: Any, stem_zip_file: Optional[Any], user_note: str, busy_auto_mixing: bool = False) -> None:
    cfg = _service_config()
    if not cfg["url"]:
        raise ServiceError("서비스 주소가 설정되지 않았습니다.")
    if busy_auto_mixing and stem_zip_file is None:
        raise ServiceError("Busy Auto Mixing을 사용하려면 분리트랙 ZIP 파일이 필요합니다.")
    data = uploaded_file.getvalue()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ServiceError(f"완성본 WAV 파일이 너무 큽니다: {size_mb:.1f} MB / 제한 {MAX_UPLOAD_MB} MB")
    stem_zip_data = b""
    stem_zip_filename = ""
    stem_zip_content_type = "application/zip"
    if stem_zip_file is not None:
        stem_zip_data = stem_zip_file.getvalue()
        zip_mb = len(stem_zip_data) / (1024 * 1024)
        if zip_mb > MAX_STEM_ZIP_MB:
            raise ServiceError(f"분리트랙 ZIP 파일이 너무 큽니다: {zip_mb:.1f} MB / 제한 {MAX_STEM_ZIP_MB} MB")
        stem_zip_filename = str(stem_zip_file.name or "separated_tracks.zip")
        stem_zip_content_type = stem_zip_file.type or "application/zip"
    duration_sec = _read_wav_duration_sec(data)
    content_type = uploaded_file.type or "audio/wav"
    filename = str(uploaded_file.name or "upload.wav")
    started_at = time.time()

    shared: Dict[str, Any] = {
        "phase": "queued",
        "job_id": "",
        "estimate_sec": None,
        "error": "",
        "done": False,
        "payload": _async_payload("initializing", "preparing", 3, "업로드 준비 중"),
    }
    st.session_state.busy_async_start_state = shared
    st.session_state.busy_job_id = ""
    st.session_state.busy_started_at = started_at
    st.session_state.busy_estimate_sec = None
    st.session_state.busy_original_filename = filename
    st.session_state.busy_stem_zip_filename = stem_zip_filename
    st.session_state.busy_stem_zip_provided = bool(stem_zip_data)
    st.session_state.busy_auto_mixing_enabled = bool(busy_auto_mixing and stem_zip_data)
    st.session_state.busy_download_filename = _download_filename_from_original(filename)
    st.session_state.busy_download_bytes = b""
    st.session_state.busy_last_status_poll_at = 0.0
    st.session_state.busy_last_status_poll_job_id = ""
    st.session_state.busy_status_poll_backoff_until = 0.0
    st.session_state.busy_status_poll_error_count = 0
    st.session_state.busy_status_poll_last_error = ""
    st.session_state.busy_job_payload = dict(shared["payload"])

    thread = threading.Thread(
        target=_start_job_bootstrap_worker,
        kwargs={
            "cfg": dict(cfg),
            "data": data,
            "filename": filename,
            "content_type": content_type,
            "duration_sec": duration_sec,
            "user_note": user_note or "",
            "shared": shared,
            "stem_zip_data": stem_zip_data,
            "stem_zip_filename": stem_zip_filename,
            "stem_zip_content_type": stem_zip_content_type,
            "busy_auto_mixing": bool(busy_auto_mixing and stem_zip_data),
        },
        daemon=True,
    )
    st.session_state.busy_start_bootstrap_thread = thread
    thread.start()


def _render_live_progress_component(*, started_at: float, estimate_sec: Optional[float], worker_progress: Optional[float], label: str = "마스터링 중", min_progress: float = 3.0) -> None:
    """Native Streamlit progress UI.

    This intentionally avoids st.components.v1.html because newer Streamlit versions
    emit a deprecation warning on every fragment refresh. The surrounding
    st.fragment(run_every="1s") keeps elapsed time and progress responsive without
    browser-side JavaScript.
    """
    elapsed = max(0.0, time.time() - float(started_at or time.time()))

    def _fmt(sec: float) -> str:
        sec_i = max(0, int(sec or 0))
        h = sec_i // 3600
        m = (sec_i % 3600) // 60
        s = sec_i % 60
        if h > 0:
            return f"{h}시간 {m}분 {s}초"
        if m > 0:
            return f"{m}분 {s}초"
        return f"{s}초"

    progress = max(1.0, min(30.0, float(min_progress)))
    estimate_text = ""
    if estimate_sec is not None:
        try:
            estimate_val = float(estimate_sec)
        except Exception:
            estimate_val = 0.0
        if math.isfinite(estimate_val) and estimate_val > 0:
            estimate_text = f" / 약 {_fmt(estimate_val)}"
            progress = min(95.0, max(progress, (elapsed / estimate_val) * 95.0))

    if worker_progress is not None and isinstance(worker_progress, (int, float)):
        try:
            worker_val = float(worker_progress)
        except Exception:
            worker_val = 0.0
        if math.isfinite(worker_val):
            progress = max(progress, worker_val)

    progress = min(95.0, max(progress, 1.0))
    pct = int(round(progress))
    st.info(f"{label} [{_fmt(elapsed)}{estimate_text}]")
    st.progress(pct)
    st.caption(f"진행률 {pct}%")


def _init_upload_and_start(uploaded_file: Any, user_note: str, busy_auto_mixing: bool = False) -> None:
    cfg = _service_config()
    if not cfg["url"]:
        raise ServiceError("서비스 주소가 설정되지 않았습니다.")
    data = uploaded_file.getvalue()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ServiceError(f"파일이 너무 큽니다: {size_mb:.1f} MB / 제한 {MAX_UPLOAD_MB} MB")
    duration_sec = _read_wav_duration_sec(data)
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
            "audio_duration_sec": duration_sec,
            "content_type": content_type,
            "mode": DEFAULT_MODE,
            "user_note": user_note or "",
            "client_build_id": APP_BUILD_ID,
            "busy_auto_mixing": bool(busy_auto_mixing),
        },
    )
    signed_url = str(init_payload.get("signed_upload_url") or "")
    job_id = str(init_payload.get("job_id") or "")
    if not signed_url or not job_id:
        raise ServiceError("업로드 준비 응답이 올바르지 않습니다.")

    st.session_state.busy_job_id = job_id
    st.session_state.busy_started_at = started_at
    st.session_state.busy_estimate_sec = _payload_estimate_sec(init_payload)
    st.session_state.busy_original_filename = uploaded_file.name
    st.session_state.busy_download_filename = _download_filename_from_original(uploaded_file.name)
    st.session_state.busy_download_bytes = b""
    st.session_state.busy_last_status_poll_at = 0.0
    st.session_state.busy_last_status_poll_job_id = ""
    st.session_state.busy_status_poll_backoff_until = 0.0
    st.session_state.busy_status_poll_error_count = 0
    st.session_state.busy_status_poll_last_error = ""
    st.session_state.busy_job_payload = {
        **init_payload,
        "job_id": job_id,
        "status": "uploading",
        "stage": "uploading",
        "progress_pct": 10,
    }

    _upload_to_signed_url(signed_url, data, content_type)
    st.session_state.busy_job_payload.update({"status": "processing", "stage": "mastering", "progress_pct": 20})
    _launch_start_thread(job_id, user_note, busy_auto_mixing)


def _display_job(payload: Dict[str, Any]) -> None:
    started_at = float(st.session_state.get("busy_started_at") or time.time())
    elapsed = max(0.0, time.time() - started_at)
    estimate_raw = _payload_estimate_sec(payload)
    if estimate_raw is None:
        estimate_raw = st.session_state.get("busy_estimate_sec")
    try:
        estimate_sec = float(estimate_raw) if estimate_raw is not None else None
    except Exception:
        estimate_sec = None

    if _job_is_done(payload):
        st.progress(100)
        st.caption("진행률 100%")
        st.success("마스터링이 완료되었습니다.")
        return

    if _job_is_failed(payload):
        st.progress(100)
        st.caption("진행률 100%")
        st.error("마스터링에 실패했습니다.")
        msg = str(payload.get("message") or payload.get("error_tail") or "")
        if msg:
            st.caption(msg[:1000])
        return

    worker_progress = payload.get("progress_pct")
    worker_progress_f: Optional[float]
    if isinstance(worker_progress, (int, float)) and math.isfinite(float(worker_progress)):
        worker_progress_f = float(worker_progress)
    else:
        worker_progress_f = None

    status = str(payload.get("status") or "").lower()
    stage = str(payload.get("stage") or "").lower()
    if status in {"initializing", "queued"} or stage in {"preparing", "start_pending"}:
        label = "마스터링 준비 중"
        min_progress = 3.0
    elif status == "uploading" or stage == "uploading":
        label = "파일 업로드 중"
        min_progress = 10.0
    elif stage in {"busy_auto_mixing", "auto_mixing", "bamix"}:
        label = "오토 믹싱 및 마스터링 중"
        min_progress = 20.0
    else:
        bamix = payload.get("busy_auto_mixing") if isinstance(payload.get("busy_auto_mixing"), dict) else {}
        label = "오토 믹싱 및 마스터링 중" if bamix.get("requested") or bamix.get("requested_at_init") else "마스터링 중"
        min_progress = 20.0
    _render_live_progress_component(
        started_at=started_at,
        estimate_sec=estimate_sec,
        worker_progress=worker_progress_f,
        label=label,
        min_progress=min_progress,
    )


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


def _render_done_download(job_id: str, payload: Dict[str, Any]) -> None:
    download_url = payload.get("download_url")
    if not st.session_state.get("busy_download_bytes"):
        try:
            st.session_state.busy_download_bytes = _download_result(job_id, str(download_url or ""))
        except ServiceError as exc:
            st.error(str(exc))
            st.session_state.busy_download_bytes = b""
    if st.session_state.get("busy_download_bytes"):
        col_download, col_reset = st.columns([3, 1])
        with col_download:
            st.download_button(
                "마스터링 WAV 다운로드",
                data=st.session_state.busy_download_bytes,
                file_name=st.session_state.get("busy_download_filename") or "BAM_48k_24bit.wav",
                mime="audio/wav",
                type="primary",
                use_container_width=True,
            )
        with col_reset:
            if st.button("초기화", type="secondary", use_container_width=True):
                _reset_app_to_initial()


def _render_job_status_panel() -> None:
    _sync_bootstrap_state()
    job_id = str(st.session_state.get("busy_job_id") or "")
    payload = st.session_state.get("busy_job_payload") or {}
    if not job_id and not payload:
        return

    is_active = _job_is_active(payload)

    # Render from cached/bootstrap state first. This keeps the elapsed timer responsive even if
    # the start request, upload, or status endpoint is temporarily slow. Polling is intentionally
    # short-timeout and throttled, so a blocked status request cannot freeze the visible timer.
    _display_job(payload)

    if _job_is_done(payload):
        if job_id:
            _render_done_download(job_id, payload)
        return

    if not job_id or not is_active:
        return

    if _status_poll_due():
        latest = _poll_job(job_id, timeout_sec=_poll_timeout_sec())
        if latest:
            previous_status = str(payload.get("status", "")).lower()
            latest_status = str(latest.get("status", "")).lower()
            st.session_state.busy_job_payload = latest
            if latest_status != previous_status or _job_is_done(latest) or _job_is_failed(latest):
                st.rerun()


@st.fragment(run_every="1s")
def _live_job_status_panel() -> None:
    # Fragment refresh keeps the page interactive while the status area updates every second.
    # Avoid a top-level sleep/rerun loop because Streamlit renders the whole page as busy/disabled while the script is running.
    _render_job_status_panel()




def _reset_app_to_initial() -> None:
    """Clear all UI/job state and return to the initial upload screen."""
    for key in [
        "busy_job_id", "busy_job_payload", "busy_started_at", "busy_estimate_sec",
        "busy_original_filename", "busy_download_filename", "busy_download_bytes",
        "busy_start_thread", "busy_start_bootstrap_thread", "busy_async_start_state",
        "busy_last_status_poll_at", "busy_last_status_poll_job_id", "busy_status_poll_backoff_until",
        "busy_status_poll_error_count", "busy_status_poll_last_error", "busy_status_poll_retry_after_sec",
        "busy_last_uploaded_name", "busy_stem_zip_filename", "busy_stem_zip_provided", "busy_auto_mixing_enabled", "busy_auto_mixing_checkbox",
    ]:
        st.session_state.pop(key, None)
    st.session_state["busy_uploader_nonce"] = int(st.session_state.get("busy_uploader_nonce", 0) or 0) + 1
    st.session_state["busy_note_nonce"] = int(st.session_state.get("busy_note_nonce", 0) or 0) + 1
    st.rerun()


def _reset_if_new_upload(uploaded_name: str) -> None:
    prev = st.session_state.get("busy_last_uploaded_name")
    if prev and prev != uploaded_name and _job_is_done(st.session_state.get("busy_job_payload") or {}):
        for key in [
            "busy_job_id", "busy_job_payload", "busy_started_at", "busy_estimate_sec",
            "busy_original_filename", "busy_download_filename", "busy_download_bytes", "busy_start_thread",
            "busy_start_bootstrap_thread", "busy_async_start_state", "busy_last_status_poll_at",
            "busy_last_status_poll_job_id", "busy_status_poll_backoff_until", "busy_status_poll_error_count",
            "busy_status_poll_last_error", "busy_status_poll_retry_after_sec",
            "busy_stem_zip_filename", "busy_stem_zip_provided", "busy_auto_mixing_enabled", "busy_auto_mixing_checkbox",
        ]:
            st.session_state.pop(key, None)
    st.session_state.busy_last_uploaded_name = uploaded_name


def main() -> None:
    st.set_page_config(page_title="Busy Auto Mastering", page_icon="🎧", layout="centered")
    st.title("Busy Auto Mastering")

    cfg = _service_config()
    if not cfg["url"]:
        st.warning("서비스 설정이 필요합니다.")

    uploader_nonce = int(st.session_state.get("busy_uploader_nonce", 0) or 0)
    note_nonce = int(st.session_state.get("busy_note_nonce", 0) or 0)
    uploaded_files = st.file_uploader("파일 업로드", type=["wav", "zip"], accept_multiple_files=True, key=f"busy_file_uploader_{uploader_nonce}")
    uploaded, stem_zip, upload_error = _split_upload_selection(uploaded_files)
    user_note = st.text_area("장르/스타일 태그(선택)", height=70, placeholder="비워두면 자동으로 처리됩니다.", key=f"busy_user_note_{note_nonce}")

    start_clicked = False
    payload = st.session_state.get("busy_job_payload") or {}
    job_id = str(st.session_state.get("busy_job_id") or "")

    # The start button is intentionally not rendered on the initial screen.
    # It appears only after a valid main WAV is loaded and the file metadata is displayed.
    if upload_error and uploaded_files:
        st.warning(upload_error)
    if uploaded is not None:
        _reset_if_new_upload(_upload_selection_key(uploaded_files))
        file_data = uploaded.getvalue()
        size_mb = len(file_data) / (1024 * 1024)
        duration_sec = _read_wav_duration_sec(file_data)
        cols = st.columns(3)
        cols[0].metric("완성본 파일", uploaded.name)
        cols[1].metric("파일 크기", f"{size_mb:.1f} MB")
        cols[2].metric("곡 길이", _format_duration(duration_sec))
        busy_auto_mixing_enabled = False
        if stem_zip is not None:
            zip_size_mb = len(stem_zip.getvalue()) / (1024 * 1024)
            st.caption(f"분리트랙 ZIP 감지: {stem_zip.name} ({zip_size_mb:.1f} MB)")
            busy_auto_mixing_enabled = st.checkbox(
                "Busy Auto Mixing 사용",
                value=False,
                key="busy_auto_mixing_checkbox",
                help=(
                    "선택하면 업로드한 분리트랙 ZIP으로 자동 믹싱 premaster를 먼저 만든 뒤 "
                    "그 premaster를 마스터링합니다. 선택하지 않으면 기존 완성본 WAV 기준으로 마스터링합니다."
                ),
            )
            if busy_auto_mixing_enabled:
                st.caption("Busy Auto Mixing: 분리트랙으로 자동 믹싱 premaster를 생성한 뒤 마스터링합니다.")
            else:
                st.caption("Busy Auto Mixing 미사용: 분리트랙 ZIP은 보조 데이터로만 전달되고 기존 완성본 WAV 기준으로 진행됩니다.")
        else:
            st.caption("분리트랙 ZIP 없이 기존 방식으로 진행됩니다.")

        _sync_bootstrap_state()
        job_id = str(st.session_state.get("busy_job_id") or "")
        payload = st.session_state.get("busy_job_payload") or {}
        active = _job_is_active(payload)
        terminal = _job_is_done(payload) or _job_is_failed(payload)
        if not job_id and not active and not terminal:
            if cfg["url"]:
                start_label = "오토 믹싱 + 마스터링 시작" if busy_auto_mixing_enabled else "마스터링 시작"
                start_clicked = st.button(start_label, type="primary", use_container_width=True)
            else:
                st.warning("서비스 설정이 필요합니다.")

    if start_clicked and uploaded is not None:
        try:
            _begin_async_start_feedback(uploaded, stem_zip, user_note, busy_auto_mixing_enabled)
        except ServiceError as exc:
            st.error(str(exc))
        else:
            st.rerun()

    _sync_bootstrap_state()
    job_id = str(st.session_state.get("busy_job_id") or "")
    payload = st.session_state.get("busy_job_payload") or {}
    if job_id or _job_is_active(payload) or _job_is_failed(payload):
        if _job_is_active(payload):
            _live_job_status_panel()
        else:
            _render_job_status_panel()


if __name__ == "__main__":
    main()
