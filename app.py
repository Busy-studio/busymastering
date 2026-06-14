from __future__ import annotations

from typing import Any, Dict, Optional

import requests
import streamlit as st

APP_BUILD_ID = "v8.5.4.1-clean-ui-20260614"
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
    url = _secret("SERVICE_ENDPOINT_URL") or _secret("PRIVATE_WORKER_URL")
    token = _secret("SERVICE_ACCESS_TOKEN") or _secret("PRIVATE_WORKER_TOKEN")
    return {
        "url": url,
        "token": token,
        "timeout": _secret("SERVICE_TIMEOUT_SEC") or _secret("PRIVATE_WORKER_TIMEOUT_SEC", "60"),
        "start_timeout": _secret("SERVICE_START_TIMEOUT_SEC") or _secret("PRIVATE_WORKER_START_TIMEOUT_SEC", "3600"),
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
        raise ServiceError(f"서비스 연결에 실패했습니다: {exc}") from exc
    if resp.status_code >= 400:
        if resp.status_code == 429:
            raise ServiceError("현재 처리 요청이 많습니다. 잠시 후 다시 시도해 주세요. (429)")
        if resp.status_code in (401, 403):
            raise ServiceError("서비스 인증 설정을 확인해 주세요.")
        raise ServiceError(f"서비스 오류가 발생했습니다. HTTP {resp.status_code}: {resp.text[:800]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise ServiceError("서비스 응답 형식이 올바르지 않습니다.") from exc
    if not isinstance(payload, dict):
        raise ServiceError("서비스 응답 형식이 올바르지 않습니다.")
    return payload


def _upload_to_signed_url(url: str, data: bytes, content_type: str) -> None:
    headers = {"Content-Type": content_type or "audio/wav"}
    resp = requests.put(url, data=data, headers=headers, timeout=600)
    if resp.status_code >= 400:
        resp2 = requests.post(url, data=data, headers=headers, timeout=600)
        if resp2.status_code >= 400:
            raise ServiceError(f"업로드에 실패했습니다. HTTP {resp.status_code}/{resp2.status_code}")


def start_job(uploaded_file: Any, user_note: str) -> Dict[str, Any]:
    cfg = _service_config()
    if not cfg["url"]:
        raise ServiceError("서비스 주소가 설정되지 않았습니다.")
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
            "mode": DEFAULT_MODE,
            "user_note": user_note or "",
            "client_build_id": APP_BUILD_ID,
        },
    )
    signed_url = str(init_payload.get("signed_upload_url") or "")
    job_id = str(init_payload.get("job_id") or "")
    if not signed_url or not job_id:
        raise ServiceError("업로드 준비 응답이 올바르지 않습니다.")

    _upload_to_signed_url(signed_url, data, content_type)

    return _request_json(
        "POST",
        _api_url(cfg["url"], f"/v1/jobs/{job_id}/start"),
        token,
        start_timeout,
        json={"mode": DEFAULT_MODE, "user_note": user_note or ""},
    )


def get_job(job_id: str) -> Dict[str, Any]:
    cfg = _service_config()
    if not cfg["url"]:
        raise ServiceError("서비스 주소가 설정되지 않았습니다.")
    timeout = float(cfg["timeout"] or 60)
    return _request_json("GET", _api_url(cfg["url"], f"/v1/jobs/{job_id}"), cfg["token"], timeout)


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
        raise ServiceError(f"다운로드 오류가 발생했습니다. HTTP {resp.status_code}")
    return resp.content


def render_status(payload: Dict[str, Any]) -> None:
    status = str(payload.get("status", "unknown")).lower()
    stage = str(payload.get("stage", ""))
    message = str(payload.get("message", ""))
    progress = payload.get("progress_pct")

    if status in {"done", "completed", "success"}:
        st.success("마스터링이 완료됐습니다.")
    elif status in {"failed", "error"}:
        st.error("마스터링에 실패했습니다.")
    else:
        st.info("마스터링 중입니다.")

    if isinstance(progress, (int, float)):
        st.progress(max(0, min(100, int(progress))))
    if stage:
        st.caption(f"단계: {stage}")
    if message:
        st.caption(message)


def main() -> None:
    st.set_page_config(page_title="Busy Auto Mastering", page_icon="🎧", layout="centered")
    st.title("Busy Auto Mastering")

    cfg = _service_config()
    if not cfg["url"]:
        st.warning("서비스 설정이 필요합니다.")

    uploaded = st.file_uploader("WAV 파일 업로드", type=["wav"], accept_multiple_files=False)
    user_note = st.text_area("선택 메모", value="", height=70, placeholder="비워두면 자동으로 처리됩니다.")

    if "busy_job_id" not in st.session_state:
        st.session_state.busy_job_id = ""
    if "busy_job_payload" not in st.session_state:
        st.session_state.busy_job_payload = {}

    start = st.button("마스터링 시작", type="primary", disabled=uploaded is None or not cfg["url"])
    refresh = st.button("상태 새로고침", disabled=not bool(st.session_state.busy_job_id))

    if start and uploaded is not None:
        file_data = uploaded.getvalue()
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"파일이 너무 큽니다: {size_mb:.1f} MB / 제한 {MAX_UPLOAD_MB} MB")
        else:
            with st.spinner("업로드 및 마스터링을 진행 중입니다..."):
                try:
                    payload = start_job(uploaded, user_note)
                except ServiceError as exc:
                    st.error(str(exc))
                else:
                    job_id = str(payload.get("job_id", ""))
                    st.session_state.busy_job_id = job_id
                    st.session_state.busy_job_payload = payload
                    render_status(payload)

    if refresh and st.session_state.busy_job_id:
        try:
            payload = get_job(st.session_state.busy_job_id)
        except ServiceError as exc:
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
        if st.button("다운로드 준비"):
            with st.spinner("마스터링 파일을 가져오는 중입니다..."):
                try:
                    mastered = download_result(job_id, str(download_url or ""))
                except ServiceError as exc:
                    st.error(str(exc))
                else:
                    st.download_button("마스터링 WAV 다운로드", data=mastered, file_name=filename, mime="audio/wav")


if __name__ == "__main__":
    main()
