from __future__ import annotations

"""v64.0.4 Finishing Intelligence + Commercial Density Replay overlay.

The v64.0.3 selection-semantics fix is preserved: delivery accepted/rejected
state remains separate from quality_gate/quality_rejected.  v64.0.4 additionally
records the actual Commercial/Maximum density candidate decision path so a WAV
maximum master can honor the -0.1 dBTP target while codec-safe distribution
remains a separate advisory candidate.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .analyzer import analyze_audio_fast_qc

ENGINE_VERSION = "busy_auto_mastering_v8_5_3_64_0_4_finishing_intelligence_commercial_maximum_density_push_foundation"
PIPELINE_VERSION = "busy_auto_mastering_pipeline_v8_5_3_64_0_4_finishing_intelligence_commercial_maximum_density_push_foundation"
REPORT_SCHEMA = "busy_master_report_v8_5_3_64_0_4_finishing_intelligence_commercial_maximum_density_push_foundation"
SCHEMA_VERSION = "busy_v6404_finishing_intelligence_density_push_overlay"


def _env_on(name: str, default: str = "1") -> bool:
    raw = str(os.environ.get(name, default) if os.environ.get(name, default) is not None else default).strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, str(default)) or default)
        if not math.isfinite(value):
            value = float(default)
    except Exception:
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


def _clip01(value: Any) -> float:
    try:
        x = float(value)
        if not math.isfinite(x):
            return 0.0
        return float(np.clip(x, 0.0, 1.0))
    except Exception:
        return 0.0


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        x = float(value)
        return float(x) if math.isfinite(x) else default
    except Exception:
        return default


def _round(value: Any, ndigits: int = 4) -> float | None:
    x = _num(value)
    return round(float(x), ndigits) if x is not None else None


def _nested(obj: Any, *path: str, default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _metric(metrics: dict[str, Any], key: str, default: float | None = None) -> float | None:
    if not isinstance(metrics, dict):
        return default
    if key in metrics:
        return _num(metrics.get(key), default)
    if key == "integrated_lufs":
        return _num(_nested(metrics, "loudness", "integrated_lufs"), default)
    if key in {"approx_true_peak_dbfs", "true_peak_dbtp", "true_peak_dbfs"}:
        return _num(metrics.get("approx_true_peak_dbfs", metrics.get("true_peak_dbtp", metrics.get("true_peak_dbfs", _nested(metrics, "loudness", "approx_true_peak_dbfs")))), default)
    if key == "crest_factor_db":
        return _num(metrics.get("crest_factor_db", _nested(metrics, "loudness", "crest_factor_db")), default)
    if key == "lra_lu":
        return _num(metrics.get("lra_lu", _nested(metrics, "loudness", "short_term_lufs", "lra_lu")), default)
    if key == "correlation":
        return _num(metrics.get("correlation", _nested(metrics, "ms", "correlation_avg")), default)
    return default


def _band(metrics: dict[str, Any], name: str, default: float = -120.0) -> float:
    val = _nested(metrics, "band_energy_db", name)
    return float(_num(val, default) or default)


def _side_ratio(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    if name == "high":
        val = _nested(metrics, "ms", "side_ratio_8k_14k")
        if val is None:
            val = max(_nested(metrics, "ms", "side_to_mid_ratio", "sibilance_5000_8000", default=0.0), _nested(metrics, "ms", "side_to_mid_ratio", "air_8000_14000", default=0.0))
    elif name == "low":
        val = _nested(metrics, "ms", "side_ratio_lowband")
        if val is None:
            val = max(_nested(metrics, "ms", "side_to_mid_ratio", "sub_20_60", default=0.0), _nested(metrics, "ms", "side_to_mid_ratio", "bass_60_120", default=0.0))
    else:
        val = _nested(metrics, "ms", "side_to_mid_ratio", name)
    return float(_num(val, default) or default)


def _side_ratio_bands(name: str) -> tuple[str, ...]:
    if name == "high":
        return ("sibilance_5000_8000", "air_8000_14000")
    if name == "low":
        return ("sub_20_60", "bass_60_120")
    return (name,)


def _side_ratio_for_risk(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
    """Return a bounded side ratio for v64 risk math.

    The legacy analyzer's side/mid ratio is mathematically valid, but it can
    explode when both mid and side energy in a narrow band are near the noise
    floor.  v64.0 must not convert that numerical instability into a fake
    codec/translation failure.  We therefore use the raw ratio only when the
    relevant M/S bands contain enough energy, and cap the value for risk
    scoring while leaving the analyzer itself untouched.
    """
    raw = _side_ratio(metrics, name, default)
    try:
        raw = float(raw)
        if not math.isfinite(raw) or raw < 0.0:
            return float(default)
    except Exception:
        return float(default)
    floor_db = _env_float("BUSY_V6400_SIDE_RATIO_STABILITY_FLOOR_DB", -72.0, -120.0, -24.0)
    bands = _side_ratio_bands(name)
    strongest = -120.0
    for band in bands:
        mid_db = _num(_nested(metrics, "ms", "mid_band_db", band), None)
        side_db = _num(_nested(metrics, "ms", "side_band_db", band), None)
        if mid_db is not None:
            strongest = max(strongest, float(mid_db))
        if side_db is not None:
            strongest = max(strongest, float(side_db))
    if strongest < floor_db:
        return 0.0
    cap = _env_float("BUSY_V6400_SIDE_RATIO_RISK_CAP", 2.0, 0.6, 12.0)
    return float(min(raw, cap))


def _target_reached_flag(report: dict[str, Any]) -> bool:
    """Return the authoritative target-reached flag.

    Do not use legacy ``target_met`` aliases here.  In older BAMix reports,
    ``commercial_loudness_finish.target_met`` can mean the commercial floor was
    accepted, not that the preferred target LUFS was actually reached.  v64
    decision replay must preserve the v63.6.2.10 distinction:
    commercial_floor_met != target_reached.
    """
    if not isinstance(report, dict):
        return False
    sem = report.get("target_semantics_v6362_10") if isinstance(report.get("target_semantics_v6362_10"), dict) else {}
    clf = report.get("commercial_loudness_finish") if isinstance(report.get("commercial_loudness_finish"), dict) else {}
    for src in (report, sem, clf):
        if isinstance(src, dict) and "target_reached" in src:
            return bool(src.get("target_reached"))
    return False


def _target_lufs_gap(report: dict[str, Any]) -> Any:
    clf = report.get("commercial_loudness_finish") if isinstance(report.get("commercial_loudness_finish"), dict) else {}
    sem = report.get("target_semantics_v6362_10") if isinstance(report.get("target_semantics_v6362_10"), dict) else {}
    return report.get("target_lufs_gap_lu") if report.get("target_lufs_gap_lu") is not None else sem.get("target_lufs_gap_lu") if sem.get("target_lufs_gap_lu") is not None else clf.get("target_lufs_gap_lu")


def _risk_over(value: float | None, start: float, full: float) -> float:
    if value is None:
        return 0.0
    return _clip01((float(value) - float(start)) / max(1e-9, float(full) - float(start)))


def _risk_under(value: float | None, start: float, full: float) -> float:
    if value is None:
        return 0.0
    return _clip01((float(start) - float(value)) / max(1e-9, float(start) - float(full)))


def _compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    m = metrics if isinstance(metrics, dict) else {}
    return {
        "integrated_lufs": _round(_metric(m, "integrated_lufs"), 3),
        "true_peak_dbtp": _round(_metric(m, "approx_true_peak_dbfs"), 3),
        "crest_factor_db": _round(_metric(m, "crest_factor_db"), 3),
        "lra_lu": _round(_metric(m, "lra_lu"), 3),
        "correlation": _round(_metric(m, "correlation"), 4),
        "stereo_minus_mono_lufs_db": _round(_nested(m, "measurement_authority", "stereo_minus_mono_lufs_db", default=_nested(m, "loudness", "measurement_authority", "stereo_minus_mono_lufs_db")), 3),
        "side_ratio_lowband": _round(_side_ratio_for_risk(m, "low"), 5),
        "side_ratio_8k_14k": _round(_side_ratio_for_risk(m, "high"), 5),
        "bass_index_db": _round(_nested(m, "quality_indices", "bass_index_db"), 3),
        "mud_index_db": _round(_nested(m, "quality_indices", "mud_index_db"), 3),
        "harshness_index_db": _round(_nested(m, "quality_indices", "harshness_index_db"), 3),
        "air_index_db": _round(_nested(m, "quality_indices", "air_index_db"), 3),
    }


def _load_final_metrics(output_path: str | Path | None, report: dict[str, Any]) -> tuple[np.ndarray | None, int | None, dict[str, Any], dict[str, Any]]:
    final_report_metrics = report.get("output_analysis") if isinstance(report.get("output_analysis"), dict) else {}
    meta = {
        "output_path": str(output_path) if output_path else None,
        "loaded_audio": False,
        "fallback_metrics_source": "report.output_analysis",
    }
    if output_path:
        try:
            p = Path(output_path)
            if p.exists() and p.stat().st_size > 0:
                y, sr = sf.read(str(p), always_2d=True, dtype="float32")
                if y.shape[1] == 1:
                    y = np.repeat(y, 2, axis=1)
                elif y.shape[1] > 2:
                    y = y[:, :2]
                metrics = analyze_audio_fast_qc(y, int(sr), true_peak_oversample=int(_env_float("BUSY_V6400_FAST_QC_TRUE_PEAK_OS", 2.0, 1.0, 8.0)))
                meta.update({"loaded_audio": True, "sample_rate_hz": int(sr), "samples": int(y.shape[0]), "channels": int(y.shape[1]), "metrics_source": "final_output_wav_fast_qc"})
                return y, int(sr), metrics, meta
        except Exception as exc:
            meta["load_error"] = f"{type(exc).__name__}: {str(exc)[:220]}"
    return None, None, final_report_metrics, meta


def _collect_warnings(report: dict[str, Any], state: dict[str, Any] | None = None) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()

    def add(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            text = v.strip()
            if text and len(text) <= 180 and text not in seen:
                seen.add(text)
                found.append(text)
        elif isinstance(v, (list, tuple)):
            for item in v[:80]:
                add(item)
        elif isinstance(v, dict):
            for key in ("warnings", "warning", "display_warnings", "reason_codes", "reject_reasons", "soft_limit_reasons", "hard_stop_reasons"):
                if key in v:
                    add(v.get(key))

    for src in (report, state or {}):
        if isinstance(src, dict):
            add(src.get("warnings"))
            add(src.get("display_warnings"))
            add(src.get("quality_warnings"))
            add(_nested(src, "commercial_loudness_finish", "warnings"))
            add(_nested(src, "adaptive_input_relative_qc_governance", "warnings"))
            add(_nested(src, "adaptive_input_relative_qc_governance", "display_warnings"))
            add(_nested(src, "v6362_dml_ownership_contract", "reject_reasons"))
    return found[:60]


def _playback_translation_auditor(final_metrics: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    bass_idx = _num(_nested(final_metrics, "quality_indices", "bass_index_db"), 0.0) or 0.0
    mud_idx = _num(_nested(final_metrics, "quality_indices", "mud_index_db"), 0.0) or 0.0
    harsh_idx = _num(_nested(final_metrics, "quality_indices", "harshness_index_db"), 0.0) or 0.0
    air_idx = _num(_nested(final_metrics, "quality_indices", "air_index_db"), 0.0) or 0.0
    side_high = _side_ratio_for_risk(final_metrics, "high")
    side_low = _side_ratio_for_risk(final_metrics, "low")
    corr = _metric(final_metrics, "correlation", 1.0)
    tp = _metric(final_metrics, "approx_true_peak_dbfs", -99.0)
    crest = _metric(final_metrics, "crest_factor_db", 10.0)
    stereo_mono = _nested(final_metrics, "measurement_authority", "stereo_minus_mono_lufs_db", default=_nested(final_metrics, "loudness", "measurement_authority", "stereo_minus_mono_lufs_db"))
    stereo_mono_abs = abs(float(_num(stereo_mono, 0.0) or 0.0))

    # The risk functions are intentionally conservative and deterministic.  They
    # are not taste targets; they decide whether final checks need caution.
    bass_translation_risk = max(_risk_over(bass_idx, 6.0, 13.0), _risk_under(_nested(report, "v6380_bass_harmonic_translation", "bass_translation_amount"), 0.12, 0.0) * 0.25)
    side_high_codec_risk = max(_risk_over(side_high, 0.42, 0.85), _risk_over(air_idx, 5.0, 12.0), _risk_over(harsh_idx, 4.5, 10.0))
    vocal_recession_on_small_speaker = max(_risk_over(mud_idx, 4.5, 11.0), _risk_under(harsh_idx, -3.0, -10.0), _risk_over(bass_idx, 8.0, 15.0) * 0.75)
    mono_low_mid_blur = max(_risk_under(corr, 0.45, 0.05), _risk_over(side_low, 0.14, 0.38), _risk_over(stereo_mono_abs, 0.8, 2.2), _risk_over(mud_idx, 5.0, 12.0) * 0.65)
    true_peak_codec_margin_risk = _risk_over(tp, -1.0, -0.15)
    perceived_brightness_after_codec = max(_risk_over(air_idx, 4.5, 12.0), _risk_over(harsh_idx, 4.8, 11.0))

    phone_speaker_risk = max(vocal_recession_on_small_speaker, bass_translation_risk * 0.75, mono_low_mid_blur * 0.55)
    earbud_risk = max(side_high_codec_risk, perceived_brightness_after_codec * 0.85, true_peak_codec_margin_risk * 0.5)
    small_speaker_risk = max(vocal_recession_on_small_speaker, mono_low_mid_blur * 0.65, _risk_under(crest, 6.5, 3.8) * 0.5)
    car_low_end_risk = max(bass_translation_risk, mono_low_mid_blur * 0.45, _risk_over(bass_idx, 9.0, 16.0))
    mono_translation_risk = max(mono_low_mid_blur, _risk_under(corr, 0.35, 0.0))
    codec_sensitive_high_side_sibilance_bass_risk = max(side_high_codec_risk, true_peak_codec_margin_risk, bass_translation_risk * 0.72)
    aggregate = max(phone_speaker_risk, earbud_risk, small_speaker_risk, car_low_end_risk, mono_translation_risk, codec_sensitive_high_side_sibilance_bass_risk)

    return {
        "schema_version": "busy_v6400_playback_translation_auditor",
        "active": True,
        "applied": True,
        "method": "deterministic_proxy_playback_translation_from_final_audio_metrics",
        "phone_speaker_risk": round(phone_speaker_risk, 4),
        "earbud_risk": round(earbud_risk, 4),
        "small_speaker_risk": round(small_speaker_risk, 4),
        "car_low_end_risk": round(car_low_end_risk, 4),
        "mono_translation_risk": round(mono_translation_risk, 4),
        "bass_translation_risk": round(bass_translation_risk, 4),
        "side_high_codec_risk": round(side_high_codec_risk, 4),
        "vocal_recession_on_small_speaker": round(vocal_recession_on_small_speaker, 4),
        "mono_low_mid_blur": round(mono_low_mid_blur, 4),
        "true_peak_codec_margin_risk": round(true_peak_codec_margin_risk, 4),
        "perceived_brightness_after_codec": round(perceived_brightness_after_codec, 4),
        "codec_sensitive_high_side_sibilance_bass_risk": round(codec_sensitive_high_side_sibilance_bass_risk, 4),
        "aggregate_translation_risk": round(aggregate, 4),
        "input_metrics": {
            "bass_index_db": round(float(bass_idx), 3),
            "mud_index_db": round(float(mud_idx), 3),
            "harshness_index_db": round(float(harsh_idx), 3),
            "air_index_db": round(float(air_idx), 3),
            "side_ratio_lowband": round(float(side_low), 5),
            "side_ratio_8k_14k": round(float(side_high), 5),
            "correlation": _round(corr, 4),
            "true_peak_dbtp": _round(tp, 3),
            "crest_factor_db": _round(crest, 3),
            "stereo_minus_mono_lufs_db": _round(stereo_mono, 3),
        },
        "interaction_notes": {
            "v6380_bass_harmonic_translation": "consumed_as_context_only_no_strength_change",
            "v6390_side_texture_stereo_cleanliness": "consumed_as_context_only_no_side_cleanup_strength_change",
            "dml_authority": "translation auditor cannot authorize extra limiter push",
        },
        "policy": "Proxy playback translation audit only; no threshold relaxation, no limiter push, no warning hiding.",
    }


def _codec_profiles_from_env() -> list[str]:
    raw = str(os.environ.get("BUSY_V6400_CODEC_PROFILES", "aac,mp3") or "aac,mp3")
    profiles = []
    for item in raw.replace(";", ",").split(","):
        name = item.strip().lower()
        if name in {"aac", "mp3", "opus"} and name not in profiles:
            profiles.append(name)
    return profiles[:3] or ["aac"]


def _codec_command(profile: str, encoded_path: Path) -> list[str]:
    if profile == "aac":
        return ["-c:a", "aac", "-b:a", str(os.environ.get("BUSY_V6400_AAC_BITRATE", "192k") or "192k"), str(encoded_path)]
    if profile == "mp3":
        return ["-c:a", "libmp3lame", "-b:a", str(os.environ.get("BUSY_V6400_MP3_BITRATE", "192k") or "192k"), str(encoded_path)]
    if profile == "opus":
        return ["-c:a", "libopus", "-b:a", str(os.environ.get("BUSY_V6400_OPUS_BITRATE", "160k") or "160k"), str(encoded_path)]
    return [str(encoded_path)]


def _codec_ext(profile: str) -> str:
    return {"aac": ".m4a", "mp3": ".mp3", "opus": ".opus"}.get(profile, f".{profile}")


def _codec_delta_metrics(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    side_pre = _side_ratio_for_risk(pre, "high")
    side_post = _side_ratio_for_risk(post, "high")
    low_pre = _side_ratio_for_risk(pre, "low")
    low_post = _side_ratio_for_risk(post, "low")
    return {
        "integrated_lufs_delta": _round((_metric(post, "integrated_lufs", 0.0) or 0.0) - (_metric(pre, "integrated_lufs", 0.0) or 0.0), 3),
        "true_peak_delta_db": _round((_metric(post, "approx_true_peak_dbfs", -99.0) or -99.0) - (_metric(pre, "approx_true_peak_dbfs", -99.0) or -99.0), 3),
        "crest_delta_db": _round((_metric(post, "crest_factor_db", 0.0) or 0.0) - (_metric(pre, "crest_factor_db", 0.0) or 0.0), 3),
        "side_high_ratio_delta": _round(float(side_post) - float(side_pre), 5),
        "side_low_ratio_delta": _round(float(low_post) - float(low_pre), 5),
        "air_index_delta_db": _round((_num(_nested(post, "quality_indices", "air_index_db"), 0.0) or 0.0) - (_num(_nested(pre, "quality_indices", "air_index_db"), 0.0) or 0.0), 3),
        "harshness_index_delta_db": _round((_num(_nested(post, "quality_indices", "harshness_index_db"), 0.0) or 0.0) - (_num(_nested(pre, "quality_indices", "harshness_index_db"), 0.0) or 0.0), 3),
        "mono_correlation_delta": _round((_metric(post, "correlation", 0.0) or 0.0) - (_metric(pre, "correlation", 0.0) or 0.0), 4),
    }


def _codec_risk(pre: dict[str, Any], post: dict[str, Any], delta: dict[str, Any]) -> float:
    post_tp = _metric(post, "approx_true_peak_dbfs", -99.0)
    post_side = _side_ratio_for_risk(post, "high")
    post_low = _side_ratio_for_risk(post, "low")
    post_corr = _metric(post, "correlation", 1.0)
    return round(max(
        _risk_over(post_tp, -0.7, -0.05),
        _risk_over(abs(_num(delta.get("true_peak_delta_db"), 0.0) or 0.0), 0.25, 1.2),
        _risk_over(post_side, 0.45, 0.9),
        _risk_over(abs(_num(delta.get("side_high_ratio_delta"), 0.0) or 0.0), 0.04, 0.22),
        _risk_over(abs(_num(delta.get("harshness_index_delta_db"), 0.0) or 0.0), 0.75, 4.0),
        _risk_over(post_low, 0.16, 0.42),
        _risk_under(post_corr, 0.35, 0.0),
        _risk_under(_num(delta.get("crest_delta_db"), 0.0), -0.25, -1.5),
    ), 4)


def _actual_codec_stress_test(y: np.ndarray | None, sr: int | None, final_metrics: dict[str, Any], playback: dict[str, Any]) -> dict[str, Any]:
    base = {
        "schema_version": "busy_v6400_actual_codec_stress_test",
        "active": True,
        "actual_available": False,
        "codec_profiles": [],
        "pre_metrics": _compact_metrics(final_metrics),
        "post_metrics": {},
        "delta": {},
        "risk": round(float(playback.get("codec_sensitive_high_side_sibilance_bass_risk") or playback.get("aggregate_translation_risk") or 0.0), 4),
        "fallback_reason": None,
        "policy": "Actual codec round-trip is used only when ffmpeg succeeds. Fallback proxy is never labeled actual and never authorizes extra limiter push.",
    }
    if y is None or sr is None or not isinstance(y, np.ndarray) or y.size == 0:
        base["fallback_reason"] = "final_audio_not_loaded_proxy_only"
        return base
    ffmpeg = shutil.which(str(os.environ.get("BUSY_FFMPEG_BINARY", "ffmpeg") or "ffmpeg"))
    if not ffmpeg:
        base["fallback_reason"] = "ffmpeg_not_available_proxy_only"
        return base
    if not _env_on("BUSY_V6400_ACTUAL_CODEC_STRESS", "1"):
        base["fallback_reason"] = "disabled_by_env_proxy_only"
        return base

    max_sec = _env_float("BUSY_V6400_CODEC_STRESS_MAX_SEC", 75.0, 10.0, 600.0)
    sample_count = min(int(y.shape[0]), int(max_sec * int(sr)))
    excerpt = np.asarray(y[:sample_count, :], dtype=np.float32)
    timeout_sec = _env_float("BUSY_V6400_CODEC_STRESS_TIMEOUT_SEC", 45.0, 8.0, 240.0)
    profiles = _codec_profiles_from_env()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="busy_v6400_codec_") as td:
        tdir = Path(td)
        src_wav = tdir / "stress_in.wav"
        sf.write(str(src_wav), excerpt, int(sr), subtype="PCM_24")
        pre_metrics = analyze_audio_fast_qc(excerpt, int(sr), true_peak_oversample=int(_env_float("BUSY_V6400_CODEC_QC_TRUE_PEAK_OS", 2.0, 1.0, 8.0)))
        for profile in profiles:
            encoded = tdir / f"stress{_codec_ext(profile)}"
            decoded = tdir / f"stress_{profile}_decoded.wav"
            try:
                cmd1 = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src_wav), "-vn", *_codec_command(profile, encoded)]
                subprocess.run(cmd1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=float(timeout_sec))
                cmd2 = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(encoded), "-ar", str(int(sr)), "-ac", "2", str(decoded)]
                subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=float(timeout_sec))
                dec, dec_sr = sf.read(str(decoded), always_2d=True, dtype="float32")
                if dec.shape[1] == 1:
                    dec = np.repeat(dec, 2, axis=1)
                elif dec.shape[1] > 2:
                    dec = dec[:, :2]
                post_metrics = analyze_audio_fast_qc(dec, int(dec_sr), true_peak_oversample=int(_env_float("BUSY_V6400_CODEC_QC_TRUE_PEAK_OS", 2.0, 1.0, 8.0)))
                delta = _codec_delta_metrics(pre_metrics, post_metrics)
                risk = _codec_risk(pre_metrics, post_metrics, delta)
                successes.append({
                    "codec": profile,
                    "actual_roundtrip": True,
                    "scope": "full_audio" if sample_count >= int(y.shape[0]) else f"excerpt_first_{round(sample_count / max(1, int(sr)), 3)}s",
                    "encoded_bytes": int(encoded.stat().st_size) if encoded.exists() else None,
                    "pre_metrics": _compact_metrics(pre_metrics),
                    "post_metrics": _compact_metrics(post_metrics),
                    "delta": delta,
                    "risk": risk,
                })
            except Exception as exc:
                failures.append({"codec": profile, "actual_roundtrip": False, "error_type": type(exc).__name__, "error": str(exc)[:260]})
    if successes:
        worst = max(successes, key=lambda x: float(x.get("risk") or 0.0))
        base.update({
            "actual_available": True,
            "codec_profiles": successes,
            "failed_profiles": failures,
            "post_metrics": worst.get("post_metrics", {}),
            "delta": worst.get("delta", {}),
            "risk": round(float(worst.get("risk") or 0.0), 4),
            "worst_codec": worst.get("codec"),
            "fallback_reason": None,
        })
    else:
        base["failed_profiles"] = failures
        base["fallback_reason"] = "ffmpeg_roundtrip_failed_proxy_only"
    return base


def _dml_contract(report: dict[str, Any]) -> dict[str, Any]:
    direct = report.get("v6362_dml_ownership_contract") if isinstance(report.get("v6362_dml_ownership_contract"), dict) else {}
    if direct:
        return direct
    v60 = report.get("v60_deterministic_multistage_limiter") if isinstance(report.get("v60_deterministic_multistage_limiter"), dict) else {}
    for path in (("v6362_dml_ownership_contract",), ("acceptance_gate", "v6362_dml_ownership_contract")):
        cand = _nested(v60, *path, default={})
        if isinstance(cand, dict) and cand:
            return cand
    return {}


def _shadow_master_candidate_selection(final_metrics: dict[str, Any], playback: dict[str, Any], codec: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    dml = _dml_contract(report)
    warnings = _collect_warnings(report)
    base_risk = max(float(playback.get("aggregate_translation_risk") or 0.0), float(codec.get("risk") or 0.0))
    lufs = _metric(final_metrics, "integrated_lufs")
    tp = _metric(final_metrics, "approx_true_peak_dbfs")
    crest = _metric(final_metrics, "crest_factor_db")
    lra = _metric(final_metrics, "lra_lu")
    target_miss_allowed = bool(dml.get("target_miss_allowed") or report.get("target_miss_allowed_by_dml"))
    commercial_floor_met = bool(report.get("commercial_floor_met") or _nested(report, "commercial_loudness_finish", "commercial_floor_met"))
    target_reached = _target_reached_flag(report)
    codec_risk = float(codec.get("risk") or 0.0)
    translation_risk = float(playback.get("aggregate_translation_risk") or 0.0)
    damage_caution = any(str(w).strip() in {"adaptive_qc_accept_with_caution", "lra_loss_relative", "mastering_induced_lra_collapse"} for w in warnings)

    candidates: list[dict[str, Any]] = []
    main_quality_gate = "fail" if base_risk >= 0.92 else "caution" if base_risk >= 0.72 or damage_caution else "pass"
    main_quality_reject_reason = "severe_codec_or_translation_risk" if main_quality_gate == "fail" else None
    candidates.append({
        "candidate_id": "main_final_candidate",
        "candidate_source": "delivered_final_audio",
        "deliverable_audio_available": True,
        "selected_for_delivery": True,
        "selection_status": "delivered_with_v64_quality_caution" if main_quality_gate != "pass" else "delivered",
        "candidate_metrics": {"lufs": _round(lufs, 3), "true_peak_dbtp": _round(tp, 3), "crest": _round(crest, 3), "lra": _round(lra, 3)},
        "candidate_risks": {"aggregate": round(base_risk, 4), "codec": round(codec_risk, 4), "translation": round(translation_risk, 4), "damage_caution": bool(damage_caution)},
        # In v64.0.3 accepted/rejected means delivery selection status, not quality pass/fail.
        # Quality failure is preserved separately so decision replay does not show a selected candidate as rejected.
        "accepted": True,
        "rejected": False,
        "reject_reason": None,
        "quality_gate": main_quality_gate,
        "quality_accepted": bool(main_quality_gate == "pass"),
        "quality_rejected": bool(main_quality_gate == "fail"),
        "quality_reject_reason": main_quality_reject_reason,
        "quality_caution_reason": "codec_or_translation_risk_requires_shadow_candidate" if main_quality_gate != "pass" else None,
    })

    shadow_defs = [
        ("quieter_clean_candidate", -0.45, 0.14, "reduce_limiter_workload_and_codec_overshoot_risk"),
        ("limiter_relax_candidate", -0.30, 0.10, "relax_final_limiter_pressure_when_lra_or_crest_caution_exists"),
        ("codec_safe_candidate", -0.35, 0.18 if codec_risk >= 0.50 else 0.08, "prefer_codec_margin_when_actual_or_proxy_codec_risk_is_high"),
        ("pqs_safe_candidate", -0.25, 0.10 if damage_caution else 0.04, "prefer_preservation_when_adaptive_qc_accepts_with_caution"),
        ("translation_safe_candidate", -0.25, 0.16 if translation_risk >= 0.50 else 0.06, "prefer_playback_translation_when_mono_or_small_speaker_risk_is_high"),
    ]
    for cid, lufs_shift, risk_reduction, reason in shadow_defs:
        pred_risk = max(0.0, base_risk - float(risk_reduction))
        candidates.append({
            "candidate_id": cid,
            "candidate_source": "lightweight_proxy_shadow_metrics_no_audio_substitution",
            "deliverable_audio_available": False,
            "candidate_metrics": {"predicted_lufs": _round((lufs or 0.0) + lufs_shift, 3) if lufs is not None else None, "predicted_true_peak_margin_db": _round((tp or -99.0) - lufs_shift, 3) if tp is not None else None},
            "candidate_risks": {"aggregate": round(float(pred_risk), 4), "risk_reduction_proxy": round(float(risk_reduction), 4)},
            "accepted": False,
            "rejected": True,
            "reject_reason": "proxy_only_no_deliverable_audio_rendered_in_v64_0_foundation",
            "selected_if_rendered_reason": reason,
        })

    # v64.0 does not silently replace the delivered audio with a shadow that was
    # not rendered.  It can recommend a future shadow render while selecting the
    # only deliverable final candidate for the actual report.
    recommended = min(candidates, key=lambda x: float((x.get("candidate_risks") or {}).get("aggregate", 1.0)))
    selected = candidates[0]
    selected_reason = "main_final_candidate_selected_for_delivery_because_v64_0_foundation_does_not_audio_substitute_shadow_candidates"
    if selected.get("quality_gate") == "fail":
        selected_reason += "; delivered_with_quality_risk_flag_not_quality_pass"
    elif selected.get("quality_gate") == "caution":
        selected_reason += "; delivered_with_quality_caution"
    if recommended.get("candidate_id") != "main_final_candidate" and float(((recommended.get("candidate_risks") or {}).get("aggregate") or 1.0)) + 0.08 < base_risk:
        selected_reason += f"; recommended_shadow_candidate={recommended.get('candidate_id')} for future proxy_or_actual_render"

    return {
        "schema_version": "busy_v6400_shadow_master_candidate_selection",
        "active": True,
        "applied": True,
        "method": "lightweight_candidate_metrics_existing_delivered_audio_plus_proxy_shadows",
        "candidates": candidates,
        "selected_candidate_id": selected.get("candidate_id"),
        "selected_candidate": selected,
        "selected_reason": selected_reason,
        "selection_semantics": "accepted/rejected indicate delivery selection; quality_gate/quality_rejected indicate codec/translation quality status",
        "recommended_shadow_candidate_id": recommended.get("candidate_id"),
        "recommended_shadow_reason": recommended.get("selected_if_rendered_reason") if recommended.get("candidate_id") != "main_final_candidate" else "main_candidate_lowest_risk_or_no_shadow_advantage",
        "selection_applied_to_audio": False,
        "rejected_candidates": [c for c in candidates[1:]],
        "authority": {
            "dml_contract_active": bool(dml.get("active")),
            "target_miss_allowed_by_dml": bool(target_miss_allowed),
            "commercial_floor_met": bool(commercial_floor_met),
            "target_reached": bool(target_reached),
            "policy": "commercial_floor_met is not target_reached; target miss allowance remains owned by DML/ownership telemetry.",
        },
    }


def _input_relative_damage_ownership(input_metrics: dict[str, Any], final_metrics: dict[str, Any], codec: dict[str, Any], playback: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    inp_lra = _metric(input_metrics, "lra_lu")
    out_lra = _metric(final_metrics, "lra_lu")
    inp_crest = _metric(input_metrics, "crest_factor_db")
    out_crest = _metric(final_metrics, "crest_factor_db")
    inp_tp = _metric(input_metrics, "approx_true_peak_dbfs")
    out_tp = _metric(final_metrics, "approx_true_peak_dbfs")
    lra_delta = (out_lra - inp_lra) if inp_lra is not None and out_lra is not None else None
    crest_delta = (out_crest - inp_crest) if inp_crest is not None and out_crest is not None else None
    tp_delta = (out_tp - inp_tp) if inp_tp is not None and out_tp is not None else None
    warnings = _collect_warnings(report)

    input_damage_flags = []
    if inp_lra is not None and inp_lra < _env_float("BUSY_V6400_INPUT_LOW_LRA_AT", 2.8, 0.5, 8.0):
        input_damage_flags.append("input_low_lra")
    if inp_crest is not None and inp_crest < _env_float("BUSY_V6400_INPUT_LOW_CREST_AT", 7.0, 2.0, 14.0):
        input_damage_flags.append("input_low_crest")
    if inp_tp is not None and inp_tp > _env_float("BUSY_V6400_INPUT_NEAR_TP_AT", -0.45, -3.0, 0.0):
        input_damage_flags.append("input_already_near_true_peak_ceiling")
    if _side_ratio(input_metrics, "low") > 0.18:
        input_damage_flags.append("input_low_side_width_risk")
    if _side_ratio(input_metrics, "high") > 0.60:
        input_damage_flags.append("input_side_high_hash_or_fizz_risk")

    mastering_flags = []
    if lra_delta is not None and lra_delta <= -_env_float("BUSY_V6400_MASTERING_LRA_DAMAGE_DELTA_LU", 0.35, 0.05, 2.5):
        mastering_flags.append("mastering_induced_lra_loss")
    if crest_delta is not None and crest_delta <= -_env_float("BUSY_V6400_MASTERING_CREST_DAMAGE_DELTA_DB", 0.5, 0.05, 3.0):
        mastering_flags.append("mastering_induced_crest_loss")
    if "mastering_induced_lra_collapse" in warnings:
        mastering_flags.append("warning_mastering_induced_lra_collapse")
    if "lra_loss_relative" in warnings:
        mastering_flags.append("warning_lra_loss_relative")

    codec_risk = float(codec.get("risk") or 0.0)
    codec_flags = []
    if codec.get("actual_available") and codec_risk >= 0.5:
        codec_flags.append("actual_codec_roundtrip_risk")
    elif not codec.get("actual_available") and codec_risk >= 0.5:
        codec_flags.append("proxy_codec_risk_actual_unavailable")

    translation_risk = float(playback.get("aggregate_translation_risk") or 0.0)
    translation_flags = []
    for k in ("mono_translation_risk", "phone_speaker_risk", "earbud_risk", "small_speaker_risk", "car_low_end_risk"):
        if float(playback.get(k) or 0.0) >= 0.55:
            translation_flags.append(k)

    return {
        "schema_version": "busy_v6400_input_relative_damage_ownership",
        "active": True,
        "input_damage": {"owned": bool(input_damage_flags), "flags": input_damage_flags, "metrics": _compact_metrics(input_metrics)},
        "mastering_induced_damage": {"owned": bool(mastering_flags), "flags": mastering_flags, "lra_delta_lu": _round(lra_delta, 3), "crest_delta_db": _round(crest_delta, 3), "tp_delta_db": _round(tp_delta, 3), "warnings_preserved": [w for w in warnings if w in {"adaptive_qc_accept_with_caution", "lra_loss_relative", "mastering_induced_lra_collapse"}]},
        "candidate_induced_damage": {"owned": bool(mastering_flags), "flags": mastering_flags[:], "source": "delivered_main_candidate_metrics"},
        "codec_induced_damage": {"owned": bool(codec_flags), "flags": codec_flags, "risk": round(codec_risk, 4), "actual_available": bool(codec.get("actual_available"))},
        "translation_induced_damage": {"owned": bool(translation_flags), "flags": translation_flags, "risk": round(translation_risk, 4)},
        "policy": "Classifies source/mastering/candidate/codec/translation damage separately. Does not blame input for mastering damage or hide caution warnings.",
    }


def _proxy_actual_calibration(playback: dict[str, Any], codec: dict[str, Any]) -> dict[str, Any]:
    proxy_codec = float(playback.get("codec_sensitive_high_side_sibilance_bass_risk") or playback.get("aggregate_translation_risk") or 0.0)
    actual_codec = float(codec.get("risk") or proxy_codec)
    available = bool(codec.get("actual_available"))
    delta = actual_codec - proxy_codec if available else None
    return {
        "schema_version": "busy_v6400_proxy_actual_calibration_foundation",
        "active": True,
        "calibration_available": bool(available),
        "proxy_metrics": {
            "proxy_codec_risk": round(proxy_codec, 4),
            "proxy_translation_risk": round(float(playback.get("aggregate_translation_risk") or 0.0), 4),
            "proxy_side_high_codec_risk": round(float(playback.get("side_high_codec_risk") or 0.0), 4),
        },
        "actual_metrics": {
            "actual_codec_risk": round(actual_codec, 4) if available else None,
            "actual_available": bool(available),
            "worst_codec": codec.get("worst_codec"),
        },
        "delta": {"codec_risk_delta_actual_minus_proxy": round(float(delta), 4) if delta is not None else None},
        "proxy_underestimated": bool(delta is not None and delta > 0.10),
        "proxy_overestimated": bool(delta is not None and delta < -0.10),
        "policy": "Foundation storage for later v64.7 calibration; no online learning or parameter mutation in v64.0.",
    }


def _decision_replay(selection: dict[str, Any], playback: dict[str, Any], codec: dict[str, Any], damage: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    dml = _dml_contract(report)
    v6404 = report.get("v6404_commercial_maximum_density") if isinstance(report.get("v6404_commercial_maximum_density"), dict) else report.get("commercial_maximum_density") if isinstance(report.get("commercial_maximum_density"), dict) else {}
    v6404_candidates = v6404.get("candidates") if isinstance(v6404.get("candidates"), list) else []
    v6404_selected_candidate = None
    v6404_rejected_candidates: list[dict[str, Any]] = []
    for cand in v6404_candidates:
        if not isinstance(cand, dict):
            continue
        is_selected_delivery = bool(cand.get("selected_delivery_candidate") or cand.get("candidate_id") == v6404.get("selected_candidate_id"))
        has_blocking = bool(cand.get("blocking_reasons") or cand.get("warning_blocking"))
        quality_rejected = bool(cand.get("quality_rejected") or has_blocking)
        # v64.0.4 must preserve v64.0.3 semantics: delivery selection is not
        # the same as codec/translation quality caution.  Older or synthetic
        # candidate payloads may only carry selected_delivery_candidate, so infer
        # accepted/rejected from delivery selection when explicit flags are absent.
        accepted = bool(cand.get("accepted")) if "accepted" in cand else is_selected_delivery
        rejected = bool(cand.get("rejected")) if "rejected" in cand else (not is_selected_delivery and bool(cand.get("candidate_id")))
        if is_selected_delivery:
            accepted = True
            rejected = False
        qgate = cand.get("quality_gate")
        if not qgate:
            qgate = "fail" if quality_rejected else "caution" if (cand.get("damage_advisories") or v6404.get("warning_advisory")) and is_selected_delivery else "pass"
        compact = {
            "candidate_id": cand.get("candidate_id"),
            "accepted": accepted,
            "rejected": rejected,
            "quality_gate": qgate,
            "quality_rejected": quality_rejected,
            "metrics": cand.get("metrics") if isinstance(cand.get("metrics"), dict) else None,
            "blocking_reasons": cand.get("blocking_reasons") or [],
            "warning_advisory": bool(cand.get("warning_advisory") or (is_selected_delivery and v6404.get("warning_advisory"))),
            "warning_blocking": bool(cand.get("warning_blocking") or has_blocking),
        }
        if is_selected_delivery:
            v6404_selected_candidate = compact
        elif rejected or quality_rejected or cand.get("blocking_reasons"):
            v6404_rejected_candidates.append(compact)
    if v6404_selected_candidate is None and isinstance(v6404, dict) and v6404.get("selected_candidate_id"):
        v6404_selected_candidate = {
            "candidate_id": v6404.get("selected_candidate_id"),
            "accepted": bool(v6404.get("applied")),
            "rejected": False,
            "quality_gate": "caution" if v6404.get("warning_advisory") else "pass",
            "quality_rejected": False,
            "metrics": v6404.get("selected_candidate_metrics") if isinstance(v6404.get("selected_candidate_metrics"), dict) else None,
            "blocking_reasons": v6404.get("blocking_reasons") or [],
            "warning_advisory": bool(v6404.get("warning_advisory")),
            "warning_blocking": bool(v6404.get("warning_blocking")),
        }
    commercial_density_priority = bool(v6404.get("active") and str(v6404.get("mode_policy") or "").lower() in {"commercial", "maximum"})
    warnings_as_advisory = bool(v6404.get("warning_advisory") and not v6404.get("warning_blocking"))
    path = [
        {"step": "collect_final_metrics", "authority": "delivered_final_audio_fast_qc_or_report_output_analysis"},
        {"step": "playback_translation_auditor", "aggregate_translation_risk": playback.get("aggregate_translation_risk"), "mono_translation_risk": playback.get("mono_translation_risk")},
        {"step": "codec_stress_test", "actual_available": codec.get("actual_available"), "risk": codec.get("risk"), "fallback_reason": codec.get("fallback_reason")},
        {"step": "shadow_candidate_selection", "selected_candidate_id": selection.get("selected_candidate_id"), "selected_reason": selection.get("selected_reason"), "selection_applied_to_audio": selection.get("selection_applied_to_audio")},
        {"step": "v6404_commercial_maximum_density", "active": bool(v6404.get("active")), "applied": bool(v6404.get("applied")), "mode_policy": v6404.get("mode_policy"), "selected_candidate_id": v6404.get("selected_candidate_id"), "selected_reason": v6404.get("selected_reason"), "true_peak_target_honored": bool(v6404.get("true_peak_target_honored")), "commercial_density_priority": commercial_density_priority, "warnings_as_advisory": warnings_as_advisory, "warning_blocking": bool(v6404.get("warning_blocking")), "blocking_reasons": v6404.get("blocking_reasons") or [], "codec_policy": v6404.get("codec_safe_vs_maximum_wav_decision") if isinstance(v6404.get("codec_safe_vs_maximum_wav_decision"), dict) else None},
        {"step": "input_relative_damage_ownership", "input_damage": _nested(damage, "input_damage", "owned"), "mastering_induced_damage": _nested(damage, "mastering_induced_damage", "owned"), "codec_induced_damage": _nested(damage, "codec_induced_damage", "owned"), "translation_induced_damage": _nested(damage, "translation_induced_damage", "owned")},
        {"step": "dml_authority_check", "target_miss_allowed": bool(dml.get("target_miss_allowed") or report.get("target_miss_allowed_by_dml")), "commercial_floor_met": bool(report.get("commercial_floor_met") or _nested(report, "commercial_loudness_finish", "commercial_floor_met")), "target_reached": _target_reached_flag(report)},
    ]
    return {
        "schema_version": "busy_v6404_decision_replay",
        "active": True,
        "selected_candidate": v6404_selected_candidate or selection.get("selected_candidate"),
        "v6400_shadow_selected_candidate": selection.get("selected_candidate"),
        "mode_policy": v6404.get("mode_policy"),
        "true_peak_target_honored": bool(v6404.get("true_peak_target_honored")),
        "commercial_density_priority": commercial_density_priority,
        "warnings_as_advisory": warnings_as_advisory,
        "rejected_candidates": v6404_rejected_candidates,
        "selected_reason": v6404.get("selected_reason") or selection.get("selected_reason"),
        "decision_path": path,
        "authority": {
            "dml_contract": {
                "active": bool(dml.get("active")),
                "contract_passed": bool(dml.get("contract_passed")),
                "target_miss_allowed": bool(dml.get("target_miss_allowed") or report.get("target_miss_allowed_by_dml")),
                "push_authority": dml.get("push_authority"),
            },
            "target_semantics": {
                "commercial_floor_met": bool(report.get("commercial_floor_met") or _nested(report, "commercial_loudness_finish", "commercial_floor_met")),
                "target_reached": _target_reached_flag(report),
                "target_lufs_gap_lu": _target_lufs_gap(report),
            },
            "commercial_density": {
                "active": bool(v6404.get("active")),
                "applied": bool(v6404.get("applied")),
                "target_true_peak_dbtp": v6404.get("target_true_peak_dbtp"),
                "final_true_peak_dbtp": v6404.get("final_true_peak_dbtp"),
                "under_pushed": bool(v6404.get("under_pushed")),
                "codec_safe_vs_maximum_wav_decision": v6404.get("codec_safe_vs_maximum_wav_decision") if isinstance(v6404.get("codec_safe_vs_maximum_wav_decision"), dict) else None,
            },
        },
        "policy": "Replay preserves v64.0.3 delivery-vs-quality semantics and records v64.0.4 Commercial/Maximum density ownership. Codec risk remains advisory/separate from the WAV maximum candidate unless severe blocking damage is measured; DML target semantics are not rewritten.",
    }


def _corpus_regression_snapshot(selection: dict[str, Any], playback: dict[str, Any], codec: dict[str, Any], damage: dict[str, Any], final_metrics: dict[str, Any], report: dict[str, Any], state: dict[str, Any] | None, output_meta: dict[str, Any]) -> dict[str, Any]:
    warnings = _collect_warnings(report, state)
    seed = json.dumps({
        "job_id": (state or {}).get("job_id") or report.get("job_id"),
        "output_path": output_meta.get("output_path"),
        "engine": ENGINE_VERSION,
        "metrics": _compact_metrics(final_metrics),
    }, ensure_ascii=False, sort_keys=True, default=str)
    case_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    fail_reasons = []
    if float(playback.get("aggregate_translation_risk") or 0.0) >= 0.92:
        fail_reasons.append("severe_translation_risk")
    if float(codec.get("risk") or 0.0) >= 0.92:
        fail_reasons.append("severe_codec_risk")
    if _nested(damage, "mastering_induced_damage", "owned") and "mastering_induced_lra_collapse" in warnings:
        fail_reasons.append("mastering_induced_lra_collapse_warning_preserved")
    status = "fail" if any(r.startswith("severe") for r in fail_reasons) else "pass_with_caution" if warnings or fail_reasons else "pass"
    return {
        "schema_version": "busy_v6400_corpus_regression_foundation",
        "active": True,
        "case_id": case_id,
        "patch_version": "v64.0.4",
        "engine_version": ENGINE_VERSION,
        "snapshot": {
            "key_metrics": _compact_metrics(final_metrics),
            "selected_candidate_type": selection.get("selected_candidate_id"),
            "recommended_shadow_candidate_id": selection.get("recommended_shadow_candidate_id"),
            "warnings": warnings[:20],
            "risk_deltas": {
                "translation_risk": playback.get("aggregate_translation_risk"),
                "codec_risk": codec.get("risk"),
                "codec_actual_available": codec.get("actual_available"),
                "mastering_lra_delta_lu": _nested(damage, "mastering_induced_damage", "lra_delta_lu"),
                "mastering_crest_delta_db": _nested(damage, "mastering_induced_damage", "crest_delta_db"),
            },
        },
        "status": status,
        "pass_fail": status,
        "fail_reasons": fail_reasons,
        "policy": "Foundation snapshot only; later corpus regression can diff this structure across patch versions, including v64.0.4 commercial density candidate ownership.",
    }


def build_finishing_intelligence(report: dict[str, Any] | None, state: dict[str, Any] | None = None, decision: dict[str, Any] | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    rep = report if isinstance(report, dict) else {}
    st = state if isinstance(state, dict) else {}
    dec = decision if isinstance(decision, dict) else {}
    enabled = _env_on("BUSY_V6400_FINISHING_INTELLIGENCE", "1")
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "active": bool(enabled),
        "applied": False,
        "method": "report_only_finishing_intelligence_foundation",
    }
    if not enabled:
        base["reason"] = "disabled_by_env"
        return base
    y, sr, final_metrics, output_meta = _load_final_metrics(output_path, rep)
    input_metrics = rep.get("input_analysis") if isinstance(rep.get("input_analysis"), dict) else st.get("input_analysis") if isinstance(st.get("input_analysis"), dict) else {}
    playback = _playback_translation_auditor(final_metrics, rep)
    codec = _actual_codec_stress_test(y, sr, final_metrics, playback)
    selection = _shadow_master_candidate_selection(final_metrics, playback, codec, rep)
    damage = _input_relative_damage_ownership(input_metrics, final_metrics, codec, playback, rep)
    calibration = _proxy_actual_calibration(playback, codec)
    v6404_density = rep.get("v6404_commercial_maximum_density") if isinstance(rep.get("v6404_commercial_maximum_density"), dict) else rep.get("commercial_maximum_density") if isinstance(rep.get("commercial_maximum_density"), dict) else {}
    replay = _decision_replay(selection, playback, codec, damage, rep)
    regression = _corpus_regression_snapshot(selection, playback, codec, damage, final_metrics, rep, st, output_meta)
    base.update({
        "applied": True,
        "method": "deterministic_report_only_playback_codec_shadow_damage_replay_regression_foundation",
        "output_metrics_source": output_meta,
        "playback_translation_auditor": playback,
        "codec_stress_test": codec,
        "shadow_master_candidate_selection": selection,
        "input_relative_damage_ownership": damage,
        "proxy_actual_calibration": calibration,
        "v6404_commercial_maximum_density": v6404_density,
        "commercial_maximum_density": v6404_density,
        "decision_replay": replay,
        "corpus_regression": regression,
        "module_interaction_notes": {
            "v6361_drum_transient_punch": "telemetry_consumed_only_no_dtp_intensity_change",
            "v6370_body_center_vocal_support": "telemetry_consumed_only_no_body_center_vocal_strength_change",
            "v6380_bass_harmonic_translation": "telemetry_consumed_only_no_harmonic_strength_change",
            "v6390_side_texture_stereo_cleanliness": "telemetry_consumed_only_no_side_cleanup_strength_change",
            "dml_authority": "preserved; target miss/floor semantics are read not rewritten",
            "v6404_commercial_maximum_density": "actual loud candidate selection is replayed when present; codec-safe distribution remains a separate advisory path",
        },
        "forbidden_actions_observed": [],
        "policy": "v64.0.4 preserves v64.0.3 selection semantics and adds Commercial/Maximum density decision replay without hiding warnings, relaxing thresholds, or rewriting DML target authority.",
    })
    return base


def attach_finishing_intelligence_to_report(report: dict[str, Any] | None, *, state: dict[str, Any] | None = None, decision: dict[str, Any] | None = None, output_path: str | Path | None = None) -> dict[str, Any]:
    rep = report if isinstance(report, dict) else {}
    legacy_playback = rep.get("playback_translation_auditor") if isinstance(rep.get("playback_translation_auditor"), dict) else None
    if legacy_playback and str(legacy_playback.get("schema_version", "")).strip() != "busy_v6400_playback_translation_auditor":
        rep.setdefault("legacy_playback_translation_auditor", legacy_playback)
    fi = build_finishing_intelligence(rep, state=state, decision=decision, output_path=output_path)
    if legacy_playback and "legacy_playback_translation_auditor" not in fi:
        fi["legacy_playback_translation_auditor"] = {
            "preserved": True,
            "schema_version": legacy_playback.get("schema_version"),
            "reason": "v64_top_level_playback_translation_auditor_uses_new_schema_but_legacy_auditor_was_preserved",
        }
    rep["v6400_finishing_intelligence"] = fi
    rep["finishing_intelligence"] = fi
    for key in (
        "playback_translation_auditor",
        "codec_stress_test",
        "shadow_master_candidate_selection",
        "input_relative_damage_ownership",
        "proxy_actual_calibration",
        "v6404_commercial_maximum_density",
        "commercial_maximum_density",
        "decision_replay",
        "corpus_regression",
    ):
        if isinstance(fi.get(key), dict):
            rep[key] = fi.get(key)
    rep["engine_version"] = ENGINE_VERSION
    rep["pipeline_version"] = PIPELINE_VERSION
    rep["mastering_report_schema_version"] = REPORT_SCHEMA
    rep["v6400_report_authority_overlay"] = {
        "schema_version": "busy_v6400_report_authority_overlay",
        "active": True,
        "applied": True,
        "reason": "finishing_intelligence_foundation_attached_after_dml_authority_without_rewriting_dml_contract",
        "dml_contract_present": isinstance(rep.get("v6362_dml_ownership_contract"), dict) or isinstance(_nested(rep, "v60_deterministic_multistage_limiter", "v6362_dml_ownership_contract"), dict),
        "policy": "Latest engine/schema identify v64.0.4 finishing/density telemetry overlay; DML final_source/target semantics remain in v6362 authority fields.",
    }
    if isinstance(state, dict):
        state_legacy_playback = state.get("playback_translation_auditor") if isinstance(state.get("playback_translation_auditor"), dict) else None
        if state_legacy_playback and str(state_legacy_playback.get("schema_version", "")).strip() != "busy_v6400_playback_translation_auditor":
            state.setdefault("legacy_playback_translation_auditor", state_legacy_playback)
        state["v6400_finishing_intelligence"] = fi
        state["finishing_intelligence"] = fi
        for key in (
            "playback_translation_auditor",
            "codec_stress_test",
            "shadow_master_candidate_selection",
            "input_relative_damage_ownership",
            "proxy_actual_calibration",
            "v6404_commercial_maximum_density",
            "commercial_maximum_density",
            "decision_replay",
            "corpus_regression",
        ):
            if isinstance(fi.get(key), dict):
                state[key] = fi.get(key)
        state["v6400_lineage"] = {
            "schema_version": "busy_v6400_stage_state_lineage",
            "active": bool(fi.get("active")),
            "applied": bool(fi.get("applied")),
            "dml_authority_link": "v6362_dml_ownership_contract / target_semantics_v6362_10",
            "module_interactions": fi.get("module_interaction_notes", {}),
            "decision_replay_path_available": isinstance(fi.get("decision_replay"), dict),
            "v6404_commercial_maximum_density_available": isinstance(fi.get("v6404_commercial_maximum_density"), dict),
        }
    return rep
